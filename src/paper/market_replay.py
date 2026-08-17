"""The tape, fed through one door, for replay and for forward alike.

`ClockGatedReader` exists so a backtest and a live loop cannot window the data
differently — the failure where each reads its own way and the discrepancy stays
invisible until capital is behind it. This module is that argument one level up:
**replay and forward differ only in what supplies the clock**, so they are the
same method called with a different number rather than two loops that will drift.

    replay   ->  poll(simulated_clock_ns), advanced by the caller
    forward  ->  poll(time.time_ns()), on the supervisor's cadence

**A reader with no custodian is refused.** `ClockGatedReader` takes `custodian`
as an optional argument and warns in its own docstring that a reader built
without one is unguarded. The paper engine is the first component for which that
is not optional, so "always pass one" is enforced here instead of remembered.

**An event is emitted once.** A correction to a bar already traded on cannot
un-trade it, and re-emitting it as a fresh print would double the volume the fill
model reasons about. Corrections landing after their key was emitted are counted
in `corrections_after_emission` — neither applied nor silently dropped, because a
store that is correcting bars we have traded on is a fact about the run.

**The touch is derived, and says so.** There is no order book at these
timestamps: the depth archive covers 52 minutes of one day and the bars cover
months. `best_bid`/`best_ask` are the bar's low and high, which makes the
pessimistic crossing price the bar's extreme — worse than reality, which is the
safe direction. `touch_is_derived` rides on every event so no consumer can mistake
it for a real book.

Every refusal is counted rather than logged and forgotten: a run that fed nothing
because every bar was refused must be distinguishable from a run that fed nothing
because the market was quiet.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from paper.fill_model import MarketEvent
from store.clock_gated_reader import ClockGatedReader
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, SYMBOL, VENUE


class UnguardedReader(ValueError):
    """A reader with no holdout custodian was handed to the paper engine.

    Refused at construction. The custodian's whole value is that the holdout has
    never been seen, and a peeked score is indistinguishable from an honest one
    after the fact — so this cannot be a check that runs once the read is under
    way.
    """


@dataclass(frozen=True)
class TapeEvent:
    """One print, with the identity the broker needs to route it."""

    symbol: str
    venue: str
    event_time_ns: int
    market_event: MarketEvent
    touch_is_derived: bool


def _decimal(value) -> Decimal:
    """Through `str`, never `float` — `Decimal(0.1)` carries the binary error in."""
    return Decimal(str(value))


class MarketReplay:
    """Feeds `TapeEvent`s from the store, each one exactly once."""

    def __init__(self, reader: ClockGatedReader,
                 symbols: Sequence[str] | None = None) -> None:
        if not reader.is_guarded:
            raise UnguardedReader(
                "the paper engine requires a ClockGatedReader built WITH a "
                "holdout custodian. An unguarded reader will serve the sealed "
                "range without complaint, and a holdout that has been seen "
                "cannot be un-seen — its only value is never having been read")
        self._reader = reader
        self._symbols = list(symbols) if symbols is not None else None
        # key -> the availability time the key was emitted at. The VALUE is what
        # separates "this poll re-read the same row" from "a corrected version of
        # a bar we already traded on". Holding only the keys made every re-read
        # look like a correction: measured on the live store 2026-08-15, poll 2
        # reported 3,494 corrections where nothing had been corrected at all.
        self._emitted: dict[tuple[str, str, int], int] = {}

        # The highest AVAILABILITY time this feed has read. Every read is bounded
        # below by it, which is what stops the engine re-materialising the whole
        # archive on a 60-second cadence.
        #
        # Availability, never event time. A correction to an old bar carries a
        # LATER availability time by definition, so it still arrives through the
        # bound; a bound on event time would hide corrections, which is the one
        # thing this store exists to preserve.
        #
        # Added 2026-08-17 after the engine was OOM-killed holding 5.5 GB (exit
        # 137, fourth restart that day) on a 30 GB box with no swap. It is also
        # what keeps `_emitted` bounded: rows below the watermark are never read
        # again, so the set only holds what has arrived since.
        self._availability_watermark = 0

        self.events_emitted = 0
        self.corrections_after_emission = 0
        self.refused_invalid_price = 0
        self.refused_no_volume = 0

    @property
    def availability_watermark(self) -> int:
        """The highest availability time read. Monotonic, and safe to persist."""
        return self._availability_watermark

    def resume_at(self, availability_ns: int) -> None:
        """Start bounded at a watermark recovered from a previous process.

        The caller must still `mark_seen(clock, not_before_ns=availability_ns)`
        before polling: the bound is inclusive, so rows sitting exactly at the
        watermark would otherwise be read with an empty emitted set and TRADED.
        That is the 2026-08-15 defect - 3,494 archived bars became 1,074 fills at
        prices days old - and it must not be reachable through the resume path.
        """
        self._availability_watermark = max(self._availability_watermark,
                                           int(availability_ns))

    def mark_seen(self, sim_clock_ns: int,
                  not_before_ns: int | None = None) -> int:
        """Mark rows as fed WITHOUT building or returning any event.

        `prime()` used to call `len(poll(...))`, which constructed a TapeEvent
        and a MarketEvent for all 2.3M archived rows purely to count them. The
        count is the only thing anybody wanted.
        """
        frame = self._reader.read_as_of(
            int(sim_clock_ns), symbols=self._symbols,
            not_before_ns=self._bound(not_before_ns))
        if frame.empty:
            return 0

        marked = 0
        for row in frame.itertuples(index=False):
            key = (getattr(row, SYMBOL), getattr(row, VENUE),
                   int(getattr(row, EVENT_TIME)))
            available_at = int(getattr(row, AVAILABILITY_TIME))
            self._emitted[key] = available_at
            self._advance(available_at)
            marked += 1
        return marked

    def _bound(self, not_before_ns: int | None) -> int | None:
        """The lower bound to read from: the caller's, else the watermark."""
        if not_before_ns is not None:
            return int(not_before_ns)
        return self._availability_watermark or None

    def _advance(self, available_at: int) -> None:
        if available_at > self._availability_watermark:
            self._availability_watermark = available_at

    def poll(self, sim_clock_ns: int) -> tuple[TapeEvent, ...]:
        """Everything knowable at this clock that has not been fed already.

        Bounded below by the watermark, so a poll costs what ARRIVED rather than
        what exists. The emitted set still decides what is fed, so the bound is
        an optimisation and never the correctness argument.

        Raises `HoldoutSealed` from the reader when the clock is inside the
        sealed range — before any data is touched, so a refused poll reads
        nothing and `events_emitted` does not move.
        """
        frame = self._reader.read_as_of(int(sim_clock_ns), symbols=self._symbols,
                                        not_before_ns=self._bound(None))
        if frame.empty:
            return ()

        produced: list[TapeEvent] = []
        ordered = frame.sort_values([EVENT_TIME, SYMBOL, VENUE], kind="mergesort")
        for row in ordered.itertuples(index=False):
            key = (getattr(row, SYMBOL), getattr(row, VENUE),
                   int(getattr(row, EVENT_TIME)))
            available_at = int(getattr(row, AVAILABILITY_TIME))
            self._advance(available_at)
            if key in self._emitted:
                # Seen before. The reader resolves corrections to the newest
                # visible version, so a LATER availability time means a genuine
                # correction arrived after we traded on the original; an equal one
                # is simply this poll re-reading a row it already fed, which is
                # every row of every poll and is not news.
                if available_at > self._emitted[key]:
                    self.corrections_after_emission += 1
                continue

            close, high, low = (_decimal(row.close), _decimal(row.high),
                                _decimal(row.low))
            volume = _decimal(row.volume)

            # A price of zero is not a price. Binance emits placeholder frames on
            # its trade stream and 746 early bars ate one into `low` via min(),
            # every one otherwise looking normal.
            if close <= 0 or high <= 0 or low <= 0:
                self._emitted[key] = available_at
                self.refused_invalid_price += 1
                continue
            if volume <= 0:
                self._emitted[key] = available_at
                self.refused_no_volume += 1
                continue

            self._emitted[key] = available_at
            self.events_emitted += 1
            produced.append(TapeEvent(
                symbol=key[0], venue=key[1], event_time_ns=key[2],
                market_event=MarketEvent(trade_price=close,
                                         trade_quantity=volume,
                                         best_bid=low, best_ask=high),
                touch_is_derived=True))

        return tuple(produced)

    @property
    def watermark_note(self) -> str:
        """What this feed has done, for a journal header or a wall tile.

        Counts rather than a verdict: 'fed nothing because every bar was refused'
        and 'fed nothing because the market was quiet' are different runs, and a
        single OK would erase the difference.
        """
        return (f"{self.events_emitted} event(s) fed; "
                f"{self.refused_invalid_price} refused on a non-positive price, "
                f"{self.refused_no_volume} on zero volume, "
                f"{self.corrections_after_emission} correction(s) arrived after "
                f"their bar had already been traded on")
