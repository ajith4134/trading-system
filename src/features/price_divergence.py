"""Mark against the reference each venue funds on, judged rather than reported.

DM-021 in the ledger, and its capture half has been complete since 2026-08-09:
binance publishes mark and index, bybit mark and index, hyperliquid mark and
oracle, all per minute and all on disk. What was missing is the half the row is
named for - **reconciliation**: comparing them and saying when the gap is wrong.

The distinction from `features.spot_perp_basis` is the whole point of this
module. That one *measures* the gap and reports it in basis points, which is
the carry a trade earns. This one *judges* the same gap as a data-quality
question: is this distance normal for this instrument, or has a price feed
broken? The same 40 bps is an ordinary funding basis on one symbol and a
frozen index on another, and no single number separates them.

So the threshold is not a number in this file. It is each symbol's own recent
distribution, exactly as `features.peg_monitor` judges a peg against its own
history rather than against 1.0 - a fixed band would fire constantly on
illiquid alts and never on BTC. What IS fixed here are the two robust-z cutoffs,
and they are a judgement rather than a measurement; they are named constants so
the judgement is visible instead of buried in an expression.

Three states that are not verdicts about the market:

* `STALE_MARK` - the mark did not move all window while the reference did. A
  frozen feed produces a smoothly drifting basis and looks like a trend.
* `UNJUDGED_TOO_FEW` - fewer observations than the scale needs. Stated, never
  defaulted to normal.
* `UNJUDGED_NO_DISPERSION` - the gap never varied, so there is no scale to
  judge against. A dispersion of zero cannot be divided by, and substituting a
  fallback band here would reintroduce the fixed threshold this module exists
  to avoid.

Precision, deliberately mixed and worth stating: the CURRENT gap is computed in
`Decimal` from the stored strings, because it is a price fact. The window's
history is converted to float first, because it feeds a median and a dispersion
- a robustness scale measured to the cent is not more robust, and 670,000
`Decimal` conversions per call is a cost with nothing bought by it.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from features.spot_perp_basis import REFERENCE_COLUMN_BY_FUNDS_ON
from store.clock_gated_reader import ClockGatedReader

_DATASET = "funding"
_BPS = Decimal(10_000)

# How far back the scale is measured, and it is set by the SLOWEST feed rather
# than by a round number. Measured on this box 2026-08-10: hyperliquid and
# bybit land a funding poll per minute, but binance's `premiumIndex` fan-out
# over ~861 symbols lands roughly one poll per 7-11 minutes. A six-hour window
# is 360 observations on two venues and about 36 on the third - above the
# minimum on paper and below it after any restart, so binance would sit
# permanently unjudged while the tile looked fine for everyone else.
#
# Twenty-four hours is ~144 observations on the slowest feed and still recent
# enough to be this regime rather than last week's.
DEFAULT_WINDOW_NS = 24 * 60 * 60 * 1_000_000_000

# Below this many prior observations the gap is not judged at all. A median
# absolute deviation over a handful of points is a number, not a scale.
MIN_OBSERVATIONS = 30

# Consistency constant making a median absolute deviation comparable to a
# standard deviation on normally distributed data. Not a tuning knob - it is
# the definition of the robust scale being used.
_MAD_TO_SIGMA = 1.4826

# The two cutoffs, in robust standard deviations away from the symbol's own
# median gap. These ARE a judgement: 4 is "wider than this instrument usually
# goes", 8 is "this is not the same distribution". Named here rather than
# written into the comparison so that the arbitrary part of this module is the
# part you can see.
WIDE_Z = 4.0
EXTREME_Z = 8.0

NORMAL = "NORMAL"
WIDE = "WIDE"
EXTREME = "EXTREME"
STALE_MARK = "STALE_MARK"
UNJUDGED_TOO_FEW = "UNJUDGED_TOO_FEW"
UNJUDGED_NO_DISPERSION = "UNJUDGED_NO_DISPERSION"

_COLUMNS = ("venue", "symbol", "reference", "event_time_ns", "observations",
            "gap_bps", "median_gap_bps", "robust_scale_bps", "robust_z",
            "verdict")


@dataclass(frozen=True)
class DivergenceTable:
    """Every instrument's current gap, its verdict, and what was refused.

    `rows` carries one row per (venue, symbol) that could be judged or that was
    explicitly left unjudged - the unjudged ones are rows, not omissions,
    because a display that drops them reads as a market where everything is
    fine. `refused` counts what could not be priced at all.
    """
    rows: pd.DataFrame
    refused: dict[str, int]
    window_ns: int

    def by_verdict(self) -> dict[str, int]:
        """How many instruments landed in each state. The tile's summary."""
        if self.rows.empty:
            return {}
        counts = self.rows["verdict"].value_counts()
        return {str(k): int(v) for k, v in counts.items()}


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in _COLUMNS})


def _exact_gap_bps(mark_raw, reference_raw) -> Decimal | None:
    """The current gap, in basis points, from the stored strings.

    None when either price is missing or the reference is non-positive - a zero
    reference is a placeholder wearing a price's clothes, and dividing by it
    gives an exception or an absurd number depending on sign.
    """
    if mark_raw is None or reference_raw is None:
        return None
    if isinstance(mark_raw, float) and pd.isna(mark_raw):
        return None
    if isinstance(reference_raw, float) and pd.isna(reference_raw):
        return None
    try:
        mark = Decimal(str(mark_raw))
        reference = Decimal(str(reference_raw))
    except (InvalidOperation, ValueError):
        return None
    if reference <= 0:
        return None
    return (mark - reference) / reference * _BPS


def reconcile_prices(store_root: Path, as_of_ns: int,
                     window_ns: int = DEFAULT_WINDOW_NS,
                     custodian=None) -> DivergenceTable:
    """Judge each instrument's mark-against-reference gap at `as_of_ns`.

    Read through `ClockGatedReader`, so both the current gap and the history it
    is judged against are only what had arrived by that clock. A scale fitted on
    observations from after the decision point would make every past divergence
    look ordinary - the flattering direction, and invisible.
    """
    as_of_ns = int(as_of_ns)
    window_ns = int(window_ns)
    refused = {"unrecognised_funds_on": 0, "no_reference_price": 0,
               "unparseable_price": 0}

    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)
    if frame.empty:
        return DivergenceTable(rows=_empty_rows(), refused=refused,
                               window_ns=window_ns)

    frame = frame[frame["availability_time_ns"] > as_of_ns - window_ns]
    if frame.empty:
        return DivergenceTable(rows=_empty_rows(), refused=refused,
                               window_ns=window_ns)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False):
        ordered = group.sort_values("availability_time_ns", kind="mergesort")
        newest = ordered.iloc[-1]

        reference_column = REFERENCE_COLUMN_BY_FUNDS_ON.get(newest["funds_on"])
        if reference_column is None:
            refused["unrecognised_funds_on"] += 1
            continue

        gap = _exact_gap_bps(newest["mark_price"], newest[reference_column])
        if gap is None:
            # Which of the two it was, so the count says something actionable:
            # a venue publishing no reference at all is a different problem
            # from one publishing an unusable price.
            if newest[reference_column] is None or pd.isna(newest[reference_column]):
                refused["no_reference_price"] += 1
            else:
                refused["unparseable_price"] += 1
            continue

        marks = pd.to_numeric(ordered["mark_price"], errors="coerce")
        references = pd.to_numeric(ordered[reference_column], errors="coerce")
        usable = marks.notna() & references.notna() & (references > 0)
        history_bps = ((marks[usable] - references[usable])
                       / references[usable] * 10_000.0)

        # The current observation is judged against its own past, never against
        # a window that includes it - a single extreme point drags the median
        # and the scale toward itself and partly hides its own excursion.
        prior = history_bps.iloc[:-1]
        median = float(np.median(prior)) if len(prior) else float("nan")
        mad = (float(np.median(np.abs(prior - median))) if len(prior)
               else float("nan"))
        scale = mad * _MAD_TO_SIGMA

        stale_mark = (usable.sum() >= MIN_OBSERVATIONS
                      and marks[usable].nunique() == 1
                      and references[usable].nunique() > 1)

        if stale_mark:
            verdict, robust_z = STALE_MARK, None
        elif len(prior) < MIN_OBSERVATIONS:
            verdict, robust_z = UNJUDGED_TOO_FEW, None
        elif not np.isfinite(scale) or scale <= 0:
            verdict, robust_z = UNJUDGED_NO_DISPERSION, None
        else:
            robust_z = abs(float(gap) - median) / scale
            if robust_z >= EXTREME_Z:
                verdict = EXTREME
            elif robust_z >= WIDE_Z:
                verdict = WIDE
            else:
                verdict = NORMAL

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["reference"].append(reference_column)
        out["event_time_ns"].append(int(newest["event_time_ns"]))
        out["observations"].append(int(len(prior)))
        out["gap_bps"].append(gap)
        out["median_gap_bps"].append(median)
        out["robust_scale_bps"].append(scale)
        out["robust_z"].append(robust_z)
        out["verdict"].append(verdict)

    rows = pd.DataFrame(out)
    if not rows.empty:
        rows = rows.sort_values(["venue", "symbol"], ignore_index=True)
    return DivergenceTable(rows=rows, refused=refused, window_ns=window_ns)
