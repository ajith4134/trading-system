"""How old the inputs behind a feature value were, on every feature value.

`FEATURES.md` §2 P0, ledger FE-001. The goal spec names the failure this exists
against in one line: *"the failure mode this standard is designed against is not
stupidity. It is **confident staleness** - a system that learned something true,
never noticed it stopped being true, and keeps betting on it."*

A feature value is a claim about the market at an instant. Without the age of
what it was computed from, a value derived from a book that stopped updating
eleven hours ago is indistinguishable from one computed a second ago, and the
first is far more dangerous than a missing value: it is confident and wrong,
and it arrives in the same column as the good ones.

## The bound is the feed's own cadence, never a constant

Binance polls its premium index every 7-11 minutes and hyperliquid every
minute; coinbase's book is five minutes by design and its trade tape is
whenever someone trades. One shared limit either waves through a frozen fast
feed or condemns a healthy slow one. So staleness is measured against each
series' own median gap - the rule `capture.venue_recorder` already applies to a
stream that has gone quiet, and `features.consolidated_price` had a private
copy of. This module is that copy promoted, and the copy deleted.

## Three states, because two would lie

`FRESH` and `STALE` are the answers. `UNKNOWN_CADENCE` is the third, and it is
not a synonym for fresh: a series with one observation has demonstrated no
cadence, so calling it fresh asserts something nobody measured. Consumers
choose - `consolidated_price` admits it, because a venue that has published one
book is not evidence of a frozen venue - but they choose knowing, rather than
by inheriting a `False` that meant "no evidence" and reads as "fine".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Three times the series' own routine gap. The same multiple
# `capture.venue_recorder` uses to decide a subscribed stream has stopped
# speaking, because it is the same question asked of a different series - and a
# second number here would drift from that one without either being wrong.
STALE_MULTIPLE = 3.0
# How many recent observations measure the cadence. Enough that one slow tick
# does not redefine "routine", short enough that a cadence change from an hour
# ago is what gets measured rather than yesterday's.
CADENCE_SAMPLE = 20

FRESH = "FRESH"
STALE = "STALE"
UNKNOWN_CADENCE = "UNKNOWN_CADENCE"

# The columns every feature value carries. Named here so the contract is one
# list rather than a convention each module spells its own way.
COLUMNS = ("as_of_ns", "input_event_ns", "age_ns", "routine_gap_ns",
           "freshness")


@dataclass(frozen=True)
class Staleness:
    """The age of the newest input behind a value, and the verdict on it.

    `routine_gap_ns` is None exactly when the verdict is UNKNOWN_CADENCE - a
    series that has not shown two observations has no gap to measure, and a
    zero there would read as an infinitely fast feed.
    """

    as_of_ns: int
    input_event_ns: int
    age_ns: int
    routine_gap_ns: int | None
    verdict: str

    @property
    def is_stale(self) -> bool:
        """True only when measured to be stale. UNKNOWN_CADENCE is not stale,
        and it is not fresh either - a caller wanting that distinction reads
        `verdict`, which is why it exists."""
        return self.verdict == STALE

    def as_columns(self) -> dict:
        """The stamp, as the columns a feature frame carries."""
        return {
            "as_of_ns": self.as_of_ns,
            "input_event_ns": self.input_event_ns,
            "age_ns": self.age_ns,
            "routine_gap_ns": self.routine_gap_ns,
            "freshness": self.verdict,
        }


def measure_staleness(event_times, as_of_ns: int,
                      sample: int = CADENCE_SAMPLE,
                      multiple: float = STALE_MULTIPLE) -> Staleness:
    """Age the newest observation in `event_times` against the series' own cadence.

    `event_times` is the history of the series behind one feature value, in any
    order - the newest is taken by value rather than by position, so a caller
    that hands over an unsorted frame is not silently told the wrong age.

    An empty series raises rather than returning a stamp. There is no value to
    stamp without an input, and a zero age on nothing is the most flattering
    possible answer.
    """
    times = pd.Series(list(event_times), dtype="int64")
    if times.empty:
        raise ValueError("no observations to measure staleness against - a "
                         "feature value with no input is not stale, it is absent")

    newest = int(times.max())
    age = int(as_of_ns) - newest

    recent = times.sort_values().tail(int(sample))
    gaps = recent.diff().dropna()
    if gaps.empty:
        return Staleness(int(as_of_ns), newest, age, None, UNKNOWN_CADENCE)

    routine = float(gaps.median())
    if routine <= 0:
        # Every observation shares a timestamp. A venue batching its updates
        # under one clock has shown no cadence either, and dividing by this
        # would call everything fresh forever.
        return Staleness(int(as_of_ns), newest, age, None, UNKNOWN_CADENCE)

    verdict = STALE if age > multiple * routine else FRESH
    return Staleness(int(as_of_ns), newest, age, int(routine), verdict)


def stamp(frame: pd.DataFrame, staleness_by_key: dict, key_columns) -> pd.DataFrame:
    """Attach the staleness columns to a feature frame, one stamp per key.

    Returns a new frame rather than mutating: a caller holding the unstamped
    version is usually a test comparing the two, and an in-place stamp would
    make that comparison pass for the wrong reason.

    A row whose key has no stamp is left with nulls rather than dropped, and
    that is deliberate - a value silently disappearing because nobody measured
    its age is the same class of loss this module exists to prevent.
    """
    stamped = frame.copy()
    if stamped.empty:
        for column in COLUMNS:
            stamped[column] = pd.Series(dtype="object")
        return stamped

    keys = list(key_columns)
    lookup = [staleness_by_key.get(tuple(row)) for row in
              stamped[keys].itertuples(index=False, name=None)]
    for column in COLUMNS:
        stamped[column] = [None if item is None else item.as_columns()[column]
                           for item in lookup]
    return stamped
