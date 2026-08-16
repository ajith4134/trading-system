"""Time-of-day, day-of-week and funding-hour position — and whether it matters.

`FEATURES.md` §2 (P1), marked `[MISSED]`: *"funding settles at fixed times,
creating predictable flow"*. Ledger FE-010.

## Two halves, and the second is the one worth building

The first half is arithmetic: what hour is it, what day is it, how far is the
next funding settlement. Any implementation gets that right, and on its own it
is worth almost nothing — a model handed `hour_of_day = 14` learns whatever
spurious hour-shaped pattern the training window happened to contain.

The second half is the claim the catalogue actually makes: that flow around
settlement is **predictably different**. That is an empirical statement about a
particular instrument on a particular venue, it is false on some of them, and
nothing about emitting a clock reading tests it. So this module measures it,
per (venue, symbol), and reports the effect size with a p-value beside it. A
consumer that wants the calendar position gets it; a consumer that wants to know
whether the position means anything gets an answer rather than an assumption.

## Cyclical position is encoded as sine and cosine, never as an integer

Hour 23 and hour 0 are one hour apart. As integers they are 23 apart, and every
distance-based or linear model handed the raw integer learns a discontinuity
that does not exist in the world — the single most common way this feature is
built wrong. `sin(2*pi*h/24)`, `cos(2*pi*h/24)` puts the hours on a circle where
23 and 0 are adjacent, and the same for the day of week. The raw integers are
reported too, because they are what a human reads on the board, but the pair is
what a model should take.

## The funding-hour contrast, and the venue where it does not exist

`cost.funding_carry.SETTLEMENT_HOURS` says Binance settles at 00, 08 and 16 UTC
and Hyperliquid settles **every hour**. On Hyperliquid there is therefore no
contrast to measure: every hour is a funding hour, the ratio is 1.0 by
construction, and a p-value computed over it is arithmetic performed on a
tautology. That case is REFUSED (`no_contrast_in_schedule`) rather than
reported. A naive implementation returns 1.00 with a tidy confidence interval,
and the number is not wrong so much as empty — which is worse, because nothing
about it looks empty.

## The test permutes the CLOCK, not the returns

The obvious test — shuffle the returns and see how often the settlement hours
look this special — is wrong here, and wrong in the flattering direction.
Volatility is clustered: a shuffle destroys that clustering, the shuffled
distribution is far tighter than the real one, and every hour looks significant
against it.

What is permuted instead is the **alignment**: the settlement hours are shifted
around the clock, so the window keeps its size and its shape and keeps sampling
the same clustered series, and only its alignment to the true settlement times
is destroyed. If 00/08/16 really are special, no other alignment matches them.
The shifts are ENUMERATED, not sampled — there is no seed here because there is
no randomness, and a sampled version of a test with eight possible outcomes
would add variance for nothing.

That test has a floor on its resolution, the floor depends on the venue, and it
is **reported on every row** (`min_achievable_p_value`) rather than assumed.

The subtlety that sets it: a schedule of 00/08/16 is periodic with period 8, so
shifting it by 8 hours maps it onto itself. There are 24 offsets but only **8
distinct alignments**, and counting all 24 would compare the observation against
a distribution containing three copies of itself. `distinct_alignments` collapses
them, which means Binance's floor is `1/8 = 0.125` — not the `1/24` a naive
count suggests. A p-value below its own floor is not achievable, and a module
reporting `p=0.001` from this test would be inventing precision.

That floor is a real limitation of a fixed-schedule contrast and it is the
honest ceiling on what this feature can claim. It is not raised by collecting
more data: more bars make the ratio more precise, and leave the number of
alignments a 3-per-day schedule admits at eight.

## Effect size is a ratio of mean squared return, and the direction is stated

`funding_hour_variance_ratio` is the mean squared log return inside settlement
hours divided by the mean outside them. Above 1.0 means settlement hours are
more volatile — the predictable flow the catalogue names. Below 1.0 is a real
finding too and is reported as one; the p-value is two-sided in the sense that
the permutation distribution is compared on absolute deviation from 1.0, so a
suspiciously CALM settlement hour is as detectable as a violent one.

## Staleness (FE-001)

Stamped per (venue, symbol) against the whole visible bar history, matching
every sibling: the calendar position is a fact about the clock, but the effect
behind it is a fact about the tape, and a tape that stopped an hour ago should
say so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from cost.funding_carry import SETTLEMENT_HOURS
from features.realized_volatility import BAR_INTERVAL_NS, decimal_close
from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"
_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR

# Unix epoch 0 was a Thursday. Stated as a constant rather than derived through
# `datetime`, because the whole store is int64 nanoseconds and one conversion
# through a timezone-aware object is one place for an off-by-one hour to enter.
_EPOCH_WEEKDAY = 3                       # Monday = 0

# Below this, a bucket mean is a handful of bars and the ratio between two of
# them is noise. Applied to BOTH buckets: a settlement window with 800
# observations against 12 outside it is as unusable as the reverse.
MIN_BUCKET_OBSERVATIONS = 50

_COLUMNS = ("venue", "symbol", "as_of_ns", "hour_of_day_utc", "day_of_week",
            "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
            "is_funding_hour", "hours_to_next_settlement",
            "settlement_hours_per_day", "funding_hour_variance_ratio",
            "p_value", "min_achievable_p_value", "observations_in_window",
            "observations_outside_window")

_REFUSAL_REASONS = ("unknown_settlement_schedule", "no_contrast_in_schedule",
                    "too_few_observations", "non_positive_close",
                    "unparseable_close")


class NoSettlementSchedule(LookupError):
    """The venue's funding schedule is not on record.

    Raised rather than defaulted for the reason `cost.funding_carry` raises the
    same shape: assuming 8-hourly for an unrecognised venue is exactly the
    mistake the schedule exists to prevent, and it would be silent.
    """


@dataclass(frozen=True)
class CalendarEffectsTable:
    """Calendar position per (venue, symbol), with the measured funding effect.

    One row per key that could support the contrast. `refused` counts the rest
    by reason, including the venues where the contrast does not exist at all -
    which is a property of the schedule, not a shortage of data, and would
    otherwise be indistinguishable from one.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def hour_of_day_utc(timestamp_ns: int) -> int:
    """UTC hour from an int64 nanosecond timestamp, by integer arithmetic."""
    return int(timestamp_ns % _NS_PER_DAY) // _NS_PER_HOUR


def day_of_week(timestamp_ns: int) -> int:
    """Monday = 0. Epoch 0 was a Thursday, which is `_EPOCH_WEEKDAY`."""
    return int((timestamp_ns // _NS_PER_DAY + _EPOCH_WEEKDAY) % 7)


def cyclical(value: int, period: int) -> tuple[float, float]:
    """A position on a cycle as (sine, cosine).

    The encoding exists so that the last unit of the cycle is adjacent to the
    first. Returned as a pair rather than one angle because a single angle has
    the same wrap-around discontinuity the integer had - moving the problem
    rather than solving it.
    """
    angle = 2 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def hours_to_next_settlement(venue: str, timestamp_ns: int) -> float:
    """Hours until this venue's next funding settlement, from the clock alone.

    Never from the dataset's `next_funding_time_ns`: that field is flagged
    `next_funding_time_unknown` on a documented fraction of rows and would make
    the feature's availability depend on a venue's chattiness rather than on its
    schedule. The schedule is known; the poll is not always.
    """
    hours = SETTLEMENT_HOURS.get(venue)
    if hours is None:
        raise NoSettlementSchedule(
            f"no funding schedule known for venue {venue!r}. Refusing rather "
            f"than assuming a default")
    if not hours:
        raise NoSettlementSchedule(
            f"venue {venue!r} has no funding settlements at all (spot), so "
            f"'time to the next one' has no answer")

    into_day_hours = (timestamp_ns % _NS_PER_DAY) / _NS_PER_HOUR
    candidates = [h - into_day_hours for h in hours if h > into_day_hours]
    if candidates:
        return min(candidates)
    return 24.0 - into_day_hours + min(hours)


def _squared_returns_by_hour(times: list[int], closes: list[Decimal],
                             ) -> dict[int, list[float]]:
    """Squared log returns, bucketed by the UTC hour the return CLOSED in.

    Attributed to the closing bar rather than the opening one so that a return
    spanning an hour boundary belongs to exactly one hour, and to the hour whose
    flow produced its second half.
    """
    buckets: dict[int, list[float]] = {}
    for i in range(1, len(closes)):
        # Bars that are not adjacent do not make a return. A gap in the tape
        # would otherwise produce one enormous "return" attributed to whatever
        # hour the tape resumed in - and the resumption hour is arbitrary.
        if times[i] - times[i - 1] != BAR_INTERVAL_NS:
            continue
        log_return = float((closes[i] / closes[i - 1]).ln())
        buckets.setdefault(hour_of_day_utc(times[i]), []).append(
            log_return * log_return)
    return buckets


def variance_ratio(buckets: dict[int, list[float]], window_hours: set[int],
                   ) -> tuple[float, int, int] | None:
    """Mean squared return inside `window_hours` over the mean outside it.

    None when either side is below `MIN_BUCKET_OBSERVATIONS` or the outside mean
    is zero. A zero denominator is a tape that did not move outside the window
    at all, where the ratio is infinite rather than large.
    """
    inside = [x for hour, values in buckets.items() if hour in window_hours
              for x in values]
    outside = [x for hour, values in buckets.items() if hour not in window_hours
               for x in values]
    if (len(inside) < MIN_BUCKET_OBSERVATIONS
            or len(outside) < MIN_BUCKET_OBSERVATIONS):
        return None
    mean_outside = sum(outside) / len(outside)
    if mean_outside <= 0.0:
        return None
    return (sum(inside) / len(inside)) / mean_outside, len(inside), len(outside)


def distinct_alignments(settlement_hours: tuple[int, ...],
                        ) -> list[frozenset[int]]:
    """Every whole-hour shift of the window that is not the window itself.

    Deduplicated, and that is the whole function. A schedule of 00/08/16 has
    period 8, so shifting it by 8 or by 16 reproduces it exactly: enumerating
    all 23 non-zero offsets would compare the observation against a distribution
    holding three copies of the observation, which drags every p-value toward
    the floor. Verified directly - this returns 7 alignments for Binance, not 23.
    """
    original = frozenset(settlement_hours)
    seen: dict[frozenset[int], None] = {}
    for offset in range(1, 24):
        shifted = frozenset((hour + offset) % 24 for hour in settlement_hours)
        if shifted != original:
            seen.setdefault(shifted)
    return list(seen)


def min_achievable_p_value(settlement_hours: tuple[int, ...]) -> float:
    """The smallest p-value this schedule's contrast can ever produce.

    Reported on every row rather than assumed by the reader. It is a property of
    the SCHEDULE, not of the sample: more bars sharpen the ratio and leave the
    number of alignments a 3-per-day schedule admits at eight.
    """
    return 1.0 / (len(distinct_alignments(settlement_hours)) + 1)


def alignment_permutation_p_value(buckets: dict[int, list[float]],
                                  settlement_hours: tuple[int, ...],
                                  observed_ratio: float) -> float:
    """How often a shifted alignment looks at least this unusual.

    The window is shifted around the clock rather than the returns being
    shuffled - see the module docstring for why shuffling is the flattering
    error. Compared on absolute deviation from 1.0 so that a settlement hour
    which is suspiciously CALM is as detectable as a violent one.

    Deterministic: the distinct alignments are enumerated rather than sampled.
    There is no seed because there is no randomness - a sampled version of a
    test with eight possible outcomes would add variance for nothing.
    """
    alignments = distinct_alignments(settlement_hours)
    observed_deviation = abs(observed_ratio - 1.0)
    at_least_as_extreme = 1                       # the observation itself
    for shifted in alignments:
        result = variance_ratio(buckets, shifted)
        if result is None:
            # An alignment this tape cannot support counts as NOT more extreme,
            # which is the conservative direction: it can only raise the
            # p-value, never lower it.
            continue
        if abs(result[0] - 1.0) >= observed_deviation:
            at_least_as_extreme += 1
    return at_least_as_extreme / (len(alignments) + 1)


def compute_calendar_effects(store_root: Path, as_of_ns: int,
                             custodian=None) -> CalendarEffectsTable:
    """Calendar position and the measured funding-hour effect, per key.

    Reads the one-minute bar dataset once through `ClockGatedReader`. The
    calendar half is a function of `as_of_ns` alone and would need no data at
    all; it is emitted on the same row as the effect so a consumer cannot take
    the position without seeing whether it means anything on this instrument.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return CalendarEffectsTable(rows=_empty_rows(), refused=refused)

    hour = hour_of_day_utc(as_of_ns)
    weekday = day_of_week(as_of_ns)
    sin_hour, cos_hour = cyclical(hour, 24)
    sin_day, cos_day = cyclical(weekday, 7)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        settlement_hours = SETTLEMENT_HOURS.get(venue)
        if settlement_hours is None:
            refused["unknown_settlement_schedule"] += 1
            continue
        if len(settlement_hours) in (0, 24):
            # No settlements at all (spot), or one every hour (Hyperliquid).
            # Either way there is nothing to contrast against - see docstring.
            refused["no_contrast_in_schedule"] += 1
            continue

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

        times = [int(t) for t in ordered["event_time_ns"]]
        buckets = _squared_returns_by_hour(times, closes)
        window = set(settlement_hours)
        measured = variance_ratio(buckets, window)
        if measured is None:
            refused["too_few_observations"] += 1
            continue
        ratio, inside, outside = measured
        p_value = alignment_permutation_p_value(buckets, settlement_hours, ratio)

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["as_of_ns"].append(as_of_ns)
        out["hour_of_day_utc"].append(hour)
        out["day_of_week"].append(weekday)
        out["sin_hour"].append(sin_hour)
        out["cos_hour"].append(cos_hour)
        out["sin_day_of_week"].append(sin_day)
        out["cos_day_of_week"].append(cos_day)
        out["is_funding_hour"].append(hour in window)
        out["hours_to_next_settlement"].append(
            hours_to_next_settlement(venue, as_of_ns))
        out["settlement_hours_per_day"].append(len(settlement_hours))
        out["funding_hour_variance_ratio"].append(ratio)
        out["p_value"].append(p_value)
        out["min_achievable_p_value"].append(
            min_achievable_p_value(settlement_hours))
        out["observations_in_window"].append(inside)
        out["observations_outside_window"].append(outside)

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return CalendarEffectsTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                                refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
