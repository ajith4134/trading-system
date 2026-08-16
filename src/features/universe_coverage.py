"""Every symbol we can see, in all three segments, including the quiet ones.

Added 2026-08-16 at the user's instruction: *"I need the bot to keep an eye on
all the universe symbols in all three segments."*

## What this is, and what it is deliberately not

`features.cross_sectional` builds a **tradeable set** and drops a symbol the
moment it stops printing — correctly, because ranking an instrument on
three-day-old prices is worse than not ranking it. This module is the opposite
posture on purpose: it is a **watch list**, and a symbol that went quiet is the
single most interesting row in it.

The two must not be merged. A watch list that silently drops what it stopped
seeing cannot answer *"is that instrument gone, or did our feed stop?"*, and that
question is the reason to have one.

## The three segments, established by evidence

* **perpetual** — the key appears in the `funding` dataset
* **dated future** — the key appears in `dated_futures`
* **spot** — the key has bars and no funding

Classified by **which dataset carries the key**, never by reading the venue or
symbol name. `binance-spot` obviously means spot to a human, and `BTCUSDT-14AUG26`
obviously means a dated contract — but `features.term_structure` already refuses
to establish an underlying by splitting a symbol, and the reasoning is the same
here: a naming convention is a convention, it changes without notice, and a
classifier that reads names is wrong silently on the first venue that spells
things differently.

The consequence is worth stating: a perpetual whose funding poll has died stops
looking like a perpetual to this module and starts looking like spot. That is not
a defect being hidden — it is the funding feed's outage showing up as a
reclassification, and `funding_observations` on the row is what makes it legible.

## Freshness is each series' own cadence, never a shared constant

`features.staleness.measure_staleness` — median gap of the last 20 observations,
times three, which is the multiple `capture.venue_recorder` already uses to decide
a subscribed stream has stopped speaking. One shared threshold across a segment
would call a five-second funding poll and a one-minute bar the same thing, and it
would be wrong about exactly one of them.

`UNKNOWN_CADENCE` is a first-class answer, not a gap: a symbol with one
observation, or with every observation sharing a timestamp, has shown no cadence,
and folding that into FRESH asserts something nobody measured.

## "The venue stopped speaking" and "our build is behind" are different findings

This module's first live run reported **1,887 of 1,887 perpetuals stale, 48 of 48
dated contracts stale, and 5 of 1,804 spot symbols fresh** — and the feeds were
fine. Every dataset's newest row was ~1h old, uniformly, because the Layer-1
store build lags the capture processes. Bars are a one-minute series, so three
minutes makes them stale, and an hour-old build makes the entire universe stale
at once.

A watch list that reports 3,739 rows of STALE informs nobody. So each segment
also carries **`build_lag_ns`** — the age of the newest row in that whole dataset
— and each row carries **`age_beyond_build_ns`**, how much staler that symbol is
than the freshest thing the pipeline has produced. Near zero means *"as current
as this pipeline allows"*; large means *"this instrument specifically went
quiet"*. `behind_build` counts the second kind, and it is the number worth
watching, because it is the one that does not move when the build catches up.

The build lag itself is a known defect with a home: ledger FE-001 records that
*"the store supervisor serialises its polled builds behind a multi-hour
universe-wide bars build"*, and that it is not yet fixed.

**And `behind_build` is an upper bound, not a clean count of dead instruments.**
That same serialisation is why: the universe-wide build walks symbols in turn, so
the dataset's newest row belongs to whichever symbol was written last, and a
symbol built earlier in the same sweep is "behind the build" without anything
being wrong with it. On the first live run this read 1,770 of 1,804 spot symbols,
which is far too many to be genuine silence and is mostly the sweep's own shape.

It is still worth carrying, for a reason that is about direction rather than
precision: the number can only fall as the build tightens, so a symbol that stays
in it across sweeps is a real finding. What it cannot do today is name that
symbol on one reading, and pretending otherwise is how a board gets trusted for
something it does not measure. Attributing the lag per symbol needs the build to
record when each partition was written, which it does not.

## Counts always carry their denominator

`SegmentCoverage` reports totals, never bare percentages. *"92% fresh"* over 1,800
symbols and over 12 are different statements that print identically, and this
module exists precisely so the second one cannot be mistaken for the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from features.staleness import (
    FRESH, STALE, STALE_MULTIPLE, UNKNOWN_CADENCE, measure_staleness,
)
from store.clock_gated_reader import ClockGatedReader

BARS_DATASET = "bars_60000000000ns"
FUNDING_DATASET = "funding"
DATED_FUTURES_DATASET = "dated_futures"

SPOT = "spot"
PERPETUAL = "perpetual"
DATED_FUTURE = "dated-future"

# Ordered as a reader scans them: the two segments with a funding or expiry
# schedule first, then everything else.
SEGMENTS = (PERPETUAL, DATED_FUTURE, SPOT)

_FRESHNESS_STATES = (FRESH, STALE, UNKNOWN_CADENCE)

_COLUMNS = ("segment", "venue", "symbol", "observations", "newest_event_ns",
            "age_ns", "age_beyond_build_ns", "routine_gap_ns", "freshness",
            "funding_observations")


@dataclass(frozen=True)
class SegmentCoverage:
    """One segment's roll-call. Counts, never bare percentages.

    `symbols` is the denominator every other number here needs: "92% fresh" over
    1,800 symbols and over 12 print identically and are different statements.
    """
    segment: str
    symbols: int
    venues: int
    fresh: int
    stale: int
    unknown_cadence: int
    # The age of the newest row in the whole dataset. A universe that is
    # uniformly stale by exactly this much has a lagging build, not a dead feed.
    build_lag_ns: int
    # Symbols materially staler than the build itself - the ones that went quiet
    # on their own. This is the count that does not move when the build catches
    # up, which is what makes it the one worth watching.
    behind_build: int

    def describe(self) -> str:
        lag_minutes = self.build_lag_ns / 60_000_000_000
        return (f"{self.segment}: {self.symbols} symbol(s) across "
                f"{self.venues} venue(s) - {self.fresh} fresh, {self.stale} "
                f"stale, {self.unknown_cadence} with no measurable cadence; "
                f"build lag {lag_minutes:.0f}min, {self.behind_build} symbol(s) "
                f"quiet beyond it")


@dataclass(frozen=True)
class UniverseCoverage:
    """The whole watch list, and a roll-call per segment.

    `rows` holds one row per (segment, venue, symbol) that has EVER been seen
    through the clock gate, including the ones that have gone quiet - which is
    the difference between this and `features.cross_sectional`.
    """
    rows: pd.DataFrame
    segments: list[SegmentCoverage]
    refused: dict[str, int]

    @property
    def total_symbols(self) -> int:
        return sum(segment.symbols for segment in self.segments)

    def describe(self) -> str:
        if not self.segments:
            return "no symbols visible in any segment at this clock"
        return (f"{self.total_symbols} symbol(s) watched; "
                + "; ".join(segment.describe() for segment in self.segments))


def _behind_build(rows: pd.DataFrame) -> pd.Series:
    """Symbols quiet even after the build lag is allowed for, by their OWN cadence.

    A fixed threshold was the first version of this and it was wrong in the way
    this module's own docstring forbids: a tail symbol that trades once an hour
    has an hour-old bar and is behaving normally, while a core symbol three
    minutes behind the build has stopped. One number cannot be right about both.

    So the comparison is `age_beyond_build > STALE_MULTIPLE x routine_gap` - the
    same multiple `features.staleness` and `capture.venue_recorder` already use,
    applied to each series' own measured gap. A symbol with no measurable cadence
    is NOT counted: it has shown nothing to be late against, and counting it
    would put "we have never seen this trade twice" in the same bucket as "this
    stopped".
    """
    if rows.empty:
        return pd.Series(dtype=bool)
    gaps = pd.to_numeric(rows["routine_gap_ns"], errors="coerce")
    beyond = pd.to_numeric(rows["age_beyond_build_ns"], errors="coerce")
    return gaps.notna() & (beyond > STALE_MULTIPLE * gaps)


def _keys_with_times(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Event times per (venue, symbol), for a frame that has both columns."""
    if frame.empty:
        return {}
    return {
        (str(venue), str(symbol)): group["event_time_ns"].astype("int64")
        for (venue, symbol), group in frame.groupby(["venue", "symbol"],
                                                    sort=False)
    }


def _read(store_root: Path, dataset: str, as_of_ns: int,
          custodian) -> pd.DataFrame:
    """One clock-gated read, or an empty frame if the dataset is not there yet.

    A missing dataset is not an error: `dated_futures` exists only because bybit
    publishes expiring contracts, and a store built before that capture landed
    has no such directory. Reported through `refused` rather than raised, so one
    absent segment does not cost the roll-call of the other two.
    """
    try:
        return ClockGatedReader(store_root, dataset, custodian=custodian
                                ).read_as_of(as_of_ns)
    except FileNotFoundError:
        return pd.DataFrame()


def compute_universe_coverage(store_root: Path, as_of_ns: int,
                              custodian=None) -> UniverseCoverage:
    """Roll-call every (venue, symbol) in every segment at this clock.

    Reads the three datasets that define the segments. A key's segment is
    whichever dataset carries it - funding makes it a perpetual, `dated_futures`
    makes it a dated contract, bars alone make it spot - so nothing here parses a
    venue or symbol name.
    """
    store_root = Path(store_root)
    as_of_ns = int(as_of_ns)
    refused = {"no_observations": 0, "unmeasurable_cadence": 0}

    bars = _read(store_root, BARS_DATASET, as_of_ns, custodian)
    funding = _read(store_root, FUNDING_DATASET, as_of_ns, custodian)
    dated = _read(store_root, DATED_FUTURES_DATASET, as_of_ns, custodian)

    bar_times = _keys_with_times(bars)
    funding_times = _keys_with_times(funding)
    dated_times = _keys_with_times(dated)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    # Keyed on (venue, symbol) and NOT on the segment as well. Keying it on all
    # three let a key carried by two datasets be counted twice - once as a
    # perpetual and again as spot - which inflates the totals a reader trusts
    # while every individual row looks correct. The iteration order below is the
    # classification precedence: dated future, then perpetual, then spot.
    seen: set[tuple[str, str]] = set()

    def newest_in(frame: pd.DataFrame) -> int | None:
        if frame.empty:
            return None
        return int(frame["event_time_ns"].max())

    build_newest = {
        PERPETUAL: newest_in(bars) or newest_in(funding),
        DATED_FUTURE: newest_in(dated),
        SPOT: newest_in(bars),
    }

    def add(segment: str, key: tuple[str, str], times: pd.Series,
            funding_observations: int) -> None:
        if key in seen:
            return
        if times.empty:
            refused["no_observations"] += 1
            return
        staleness = measure_staleness(times, as_of_ns)
        if staleness.verdict == UNKNOWN_CADENCE:
            refused["unmeasurable_cadence"] += 1
        seen.add(key)
        venue, symbol = key
        out["segment"].append(segment)
        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["observations"].append(int(len(times)))
        out["newest_event_ns"].append(staleness.input_event_ns)
        out["age_ns"].append(staleness.age_ns)
        newest = build_newest.get(segment)
        out["age_beyond_build_ns"].append(
            int(newest - staleness.input_event_ns) if newest is not None else 0)
        out["routine_gap_ns"].append(staleness.routine_gap_ns)
        out["freshness"].append(staleness.verdict)
        out["funding_observations"].append(funding_observations)

    # Dated futures first: a key in that dataset is a dated contract whatever
    # else carries it, because an expiry is not a property anything else has.
    for key, times in dated_times.items():
        add(DATED_FUTURE, key, times, 0)
    # Then perpetuals - a funding print is what makes an instrument perpetual.
    # Bar times are preferred for the cadence when they exist, because bars are
    # what a strategy reads; the funding count rides the row so a perpetual whose
    # funding poll died is legible rather than merely reclassified.
    for key, times in funding_times.items():
        add(PERPETUAL, key, bar_times.get(key, times), int(len(times)))
    # Everything else with bars is spot.
    for key, times in bar_times.items():
        add(SPOT, key, times, 0)

    rows = pd.DataFrame(out)
    segments = []
    for segment in SEGMENTS:
        subset = rows[rows["segment"] == segment] if not rows.empty else rows
        if len(subset) == 0:
            continue
        counts = subset["freshness"].value_counts()
        newest = build_newest.get(segment)
        segments.append(SegmentCoverage(
            segment=segment, symbols=len(subset),
            venues=int(subset["venue"].nunique()),
            fresh=int(counts.get(FRESH, 0)),
            stale=int(counts.get(STALE, 0)),
            unknown_cadence=int(counts.get(UNKNOWN_CADENCE, 0)),
            build_lag_ns=int(as_of_ns - newest) if newest is not None else 0,
            behind_build=int(_behind_build(subset).sum())))

    return UniverseCoverage(rows=rows, segments=segments, refused=refused)
