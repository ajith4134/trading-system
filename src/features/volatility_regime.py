"""Which volatility decile this instrument is in — and nothing that can veto.

`FEATURES.md` §2 (P1): *"Volatility-regime decile | Feature only — never a
gate"*. Ledger FE-011, whose note points at the decision it descends from:
FE-032, a standalone regime detector **with veto power**, DECLINED. This row is
what survived that decline, and the constraint is the whole content of the row.

## "Feature only, never a gate" is enforced by the output shape, not by a comment

A comment saying *do not use this as a gate* stops nobody, and the corpus records
what happens when a constraint lives only in prose: `~/research/` has five
circuit breakers and health checks that existed as docstrings and nothing else.

So this module **emits no boolean and no threshold**. There is no
`is_high_volatility`, no `regime`, no `should_trade`, no configured cutoff
anywhere in it. A gate needs a boolean; the only way to get one from this table
is for a caller to write the comparison itself — at which point the threshold is
in the caller's code, visible, reviewable, and attributable to whoever chose it,
instead of hiding inside a feature module where it would read as a property of
the market. `test_no_boolean_or_threshold_is_emitted` pins the absence, because
the most likely way this constraint dies is a later convenience helper.

The decile is reported as an integer 1-10 and as the underlying percentile.
Both are positions, neither is a verdict.

## The regime is measured against the instrument's OWN history, never a level

Goal-doc §5a.5 (ledger FE-012) permits per-symbol variation only through
normalisation against a symbol's own history, and calls fitted per-symbol
parameters *"the single most dangerous thing that could be implemented here"*.
An absolute volatility threshold is exactly such a parameter wearing a market
fact's clothes: 60% annualised is a crisis on one instrument and a Tuesday on
another. A decile within the key's own visible history has no such parameter to
get wrong, and it is comparable across instruments, which an absolute number is
not.

## The history is built from the same non-overlapping periods HAR-RV uses

`features.har_rv.base_period_variances` already cuts a bar series into
consecutive non-overlapping hourly realized variances, cut **on the clock** so a
gap in the tape leaves a missing hour rather than a silently longer one. That is
exactly the series a regime decile needs, and reimplementing it here would put
the gap-handling rule in two places — where the copy that drifts is the one
nobody re-derived.

Non-overlapping matters as much here as it does there. A rolling window slid one
bar at a time would report sixty times as many observations for the same
information, and the decile boundaries would be set by a sample whose size is an
artefact of the step.

## Deciles need enough history to be deciles

`MIN_HISTORY_PERIODS` is 100 — ten observations per bin at the coarsest. Below
that the boundaries are set by single observations, and the tenth decile is
whichever hour happened to be worst rather than a region of the distribution.
A key below the floor is REFUSED (`too_few_periods`), not served with a decile
computed from twelve points, because a decile computed from twelve points looks
exactly like one computed from twelve thousand.

## The decile is of variance, and the ranking is the same either way

Realized variance is used rather than volatility: the square root is monotone, so
the rank — which is all a decile is — is identical, and taking it would be
arithmetic performed for appearance. The annualised volatility a human wants to
read lives in `features.realized_volatility`.

## Staleness (FE-001)

Stamped per (venue, symbol) against the whole visible bar history. A decile from
a feed that died an hour ago is a decile of a market that has moved since, and
the stamp is the only thing that says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from features.har_rv import base_period_variances
from features.realized_volatility import decimal_close
from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# Ten per bin at the coarsest. See the module docstring for why a decile from a
# dozen observations is not a decile.
MIN_HISTORY_PERIODS = 100

DECILES = 10

_COLUMNS = ("venue", "symbol", "period_start_ns", "period_end_ns",
            "realized_variance", "decile", "percentile", "observations")

_REFUSAL_REASONS = ("too_few_periods", "no_current_period",
                    "non_positive_close", "unparseable_close",
                    "insufficient_period_coverage")


@dataclass(frozen=True)
class VolatilityRegimeTable:
    """The current volatility decile per (venue, symbol). No verdicts.

    Every column is a position or a count. Nothing here says whether the regime
    is acceptable, and nothing here can be read as saying so - see the module
    docstring on why that is the row's entire content.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def decile_of(history: list[float], value: float) -> tuple[int, float]:
    """Which tenth of `history` contains `value`, and the percentile behind it.

    The percentile is a midrank over ties, matching `features.funding_basis`: a
    tape whose variance has been identical for a hundred hours sits in the
    MIDDLE of its own distribution, and a strictly-below definition would put it
    in decile 1 and have a consumer read 'unusually calm'.

    Deciles are numbered 1-10 with 10 the most volatile. The bottom edge is
    clamped to 1 rather than allowed to reach 0: a value below everything seen
    is in the first tenth, not in a zeroth one.
    """
    total = len(history)
    below = sum(1 for item in history if item < value)
    at_or_below = below + sum(1 for item in history if item == value)
    percentile = (below + at_or_below) / (2 * total)
    return max(1, min(DECILES, int(percentile * DECILES) + 1)), percentile


def compute_volatility_regime(store_root: Path, as_of_ns: int,
                              custodian=None) -> VolatilityRegimeTable:
    """The volatility decile per (venue, symbol) at this clock.

    Reads the one-minute bar dataset once through `ClockGatedReader`, so the
    history the decile ranks against is the history that was knowable at the
    clock asked for - a decile computed against a future the model had not lived
    through is the same leak `features.funding_basis` guards its percentile
    against, and it is silent in exactly the same way.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return VolatilityRegimeTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        ordered = group.sort_values("event_time_ns")

        closes: list[Decimal] = []
        refusal_reason = None
        for raw_close in ordered["close"]:
            close = decimal_close(raw_close)
            if close is None:
                refusal_reason = "unparseable_close"
                break
            if close <= 0:
                refusal_reason = "non_positive_close"
                break
            closes.append(close)
        if refusal_reason is not None:
            refused[refusal_reason] += 1
            continue

        periods, period_refusals = base_period_variances(
            ordered["event_time_ns"].tolist(), closes)
        refused["insufficient_period_coverage"] += period_refusals[
            "insufficient_period_coverage"]

        clean = [p for p in periods if p is not None]
        if len(clean) < MIN_HISTORY_PERIODS:
            refused["too_few_periods"] += 1
            continue

        current = clean[-1]
        # The current period is ranked against the history INCLUDING itself.
        # Excluding it would make the newest observation unrankable whenever it
        # was the most extreme - which is the case a regime feature exists for.
        history = [p.variance for p in clean]
        decile, percentile = decile_of(history, current.variance)

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["period_start_ns"].append(current.start_ns)
        out["period_end_ns"].append(current.end_ns)
        out["realized_variance"].append(current.variance)
        out["decile"].append(decile)
        out["percentile"].append(percentile)
        out["observations"].append(len(history))

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return VolatilityRegimeTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                                 refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
