"""How much of a venue's reported volume left any evidence it happened.

`FEATURES.md` §1 marks this `[MISSED]`: **never size off raw aggregate
volume.** Reported volume is the most manipulated number in crypto and it is
the one every naive sizing rule reaches for first.

## What is measured, and what is deliberately not claimed

This does NOT detect wash trading, and nothing here says it does. It measures
one thing that is checkable from the bars already in Layer 1: **the share of
reported volume that traded in minutes where the price never moved at all** -
bars whose high equals their low. Turnover of that size with no price
formation leaves no evidence it met a real counterparty.

That is an *upper bound* on suspicious volume, not a verdict. A genuinely
quiet minute in an illiquid pair produces the same shape - and volume in that
minute is volume nobody should size off either, which is why the bound is
useful without the accusation. The column is named `no_impact_fraction` rather
than `wash_fraction` for exactly that reason: a name that claims more than the
measurement supports is how a heuristic becomes a fact two modules downstream.

## No threshold, on purpose

Nothing here decides what counts as too much. It reports the fraction and a
`discounted_volume` with that fraction removed, and the caller sizes off
whichever it can defend. Every threshold this module could have carried would
have been a number nobody measured - and the one honest use of the number
(size off the discounted figure instead of the reported one) needs no
threshold at all.

The second measure is `size_uniformity`: the coefficient of variation of
per-bar average trade size. Real flow is ragged; a series of identical
clip sizes is a machine trading with itself. Reported, again without a cutoff,
because what is normal differs by orders of magnitude between a major pair and
a new listing.

## What changes when it is wrong

A sizer reading `discounted_volume` takes smaller positions in symbols whose
volume cannot be corroborated, and identical positions where it can. No sizer
exists yet - the axis verdict says so rather than implying a consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# One trading day of minute bars. The window is the only fixed number here and
# it is disclosed: shorter and a single quiet hour dominates the estimate,
# longer and a venue that started faking volume today stays clean for a week.
_WINDOW_BARS = 1440
# Below this a fraction is an artefact of the sample rather than a measurement.
_MIN_BARS = 60


@dataclass(frozen=True)
class VolumeQuality:
    """Per (venue, symbol) volume corroboration, and what could not be judged."""
    rows: pd.DataFrame
    skipped: dict[str, int]


def measure_volume_quality(store_root: Path, as_of_ns: int,
                           custodian=None) -> VolumeQuality:
    """Corroborate each symbol's reported volume against its own price action."""
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    skipped = {"too_few_bars": 0, "no_volume": 0}
    if frame.empty:
        return VolumeQuality(rows=_empty_rows(), skipped=skipped)

    out = {"venue": [], "symbol": [], "reported_volume": [],
           "discounted_volume": [], "no_impact_fraction": [],
           "size_uniformity": [], "bars": []}

    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        recent = group.sort_values("event_time_ns").tail(_WINDOW_BARS)
        if len(recent) < _MIN_BARS:
            skipped["too_few_bars"] += 1
            continue

        volume = pd.to_numeric(recent["volume"], errors="coerce").fillna(0.0)
        high = pd.to_numeric(recent["high"], errors="coerce")
        low = pd.to_numeric(recent["low"], errors="coerce")
        trades = pd.to_numeric(recent["trades"], errors="coerce").fillna(0)

        reported = float(volume.sum())
        if reported <= 0:
            skipped["no_volume"] += 1
            continue

        # Volume that traded in a minute whose price never moved. Comparing
        # high to low rather than open to close on purpose: a bar that opened
        # and closed at the same price after moving between them DID form
        # price, and counting it here would inflate the bound.
        no_impact = float(volume[(high == low)].sum())
        fraction = no_impact / reported

        # Ragged flow versus a machine trading with itself. Guarded on bars
        # that actually traded, because 0 trades gives no average size and a
        # zero would read as perfect uniformity - the opposite of unknown.
        traded = trades > 0
        average_size = (volume[traded] / trades[traded]).dropna()
        if len(average_size) >= 2 and float(average_size.mean()) > 0:
            uniformity = float(average_size.std() / average_size.mean())
        else:
            uniformity = float("nan")

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["reported_volume"].append(reported)
        out["discounted_volume"].append(reported * (1.0 - fraction))
        out["no_impact_fraction"].append(fraction)
        out["size_uniformity"].append(uniformity)
        out["bars"].append(len(recent))

    return VolumeQuality(rows=pd.DataFrame(out), skipped=skipped)


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({"venue": [], "symbol": [], "reported_volume": [],
                         "discounted_volume": [], "no_impact_fraction": [],
                         "size_uniformity": [], "bars": []})
