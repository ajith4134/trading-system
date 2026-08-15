"""One door onto the tape, for replay and for forward alike.

`ARCHITECTURE.md` keeps exactly one reader so that a backtest and a live loop
cannot window the data differently — the failure where the backtest reads one way,
live reads another, and the discrepancy stays invisible until capital is behind
it. This module is the same argument one level up: replay and forward differ only
in what supplies the clock, so they are the *same method* called with a different
number, not two loops that will drift apart.

Three things are defended here:

**A reader with no custodian is refused.** `ClockGatedReader` takes `custodian` as
an optional argument and its own docstring warns that a reader built without one
is unguarded. The paper engine is the first component that must always pass one,
so this is where "always" is enforced rather than remembered.

**An event is emitted once.** A correction arriving for a bar already traded on
cannot un-trade it, so re-emitting it as a fresh print would double the volume the
fill model reasons about. Corrections that land after their key was emitted are
counted and reported, never silently applied and never silently dropped.

**The touch is derived from the bar, and says so.** There is no book at these
timestamps. `best_bid`/`best_ask` come from the bar's low and high, which makes
the pessimistic crossing price the bar's extreme — worse than reality, which is
the safe direction, and flagged rather than assumed away.
"""
from decimal import Decimal

import pandas as pd
import pytest

from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)
from validation.holdout_custodian import HoldoutCustodian, HoldoutSealed
from paper.market_replay import MarketReplay, UnguardedReader

DATASET = "bars_60000000000ns"


def _bar(event: int, available: int, *, symbol="BTCUSDT", venue="binance",
         close=100.0, high=101.0, low=99.0, volume=10.0) -> dict:
    return {SYMBOL: symbol, VENUE: venue, EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available,
            "open": close, "high": high, "low": low, "close": close,
            "volume": volume, "trades": 5}


def _write(root, rows, snapshot="snap1"):
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64",
         AVAILABILITY_TIME: "int64"})
    append_partition(root, DATASET, frame, snapshot)


def a_custodian(tmp_path, holdout_start_ns=10**18):
    """Sealed far in the future by default, so ordinary reads pass through."""
    return HoldoutCustodian(tmp_path / "holdout",
                            holdout_start_ns=holdout_start_ns,
                            holdout_end_ns=holdout_start_ns + 10**18)


def a_replay(tmp_path, **kwargs):
    reader = ClockGatedReader(tmp_path, DATASET, custodian=a_custodian(tmp_path))
    return MarketReplay(reader=reader, **kwargs)


# --- the guard that must not be optional here -------------------------------

def test_a_reader_without_a_custodian_is_refused(tmp_path):
    unguarded = ClockGatedReader(tmp_path, DATASET)
    with pytest.raises(UnguardedReader, match="custodian"):
        MarketReplay(reader=unguarded)


def test_a_clock_inside_the_sealed_holdout_raises_and_reads_nothing(tmp_path):
    _write(tmp_path, [_bar(100, 200)])
    reader = ClockGatedReader(tmp_path, DATASET,
                              custodian=a_custodian(tmp_path, holdout_start_ns=150))
    replay = MarketReplay(reader=reader)
    with pytest.raises(HoldoutSealed):
        replay.poll(300)
    assert replay.events_emitted == 0


# --- one method, two clocks -------------------------------------------------

def test_nothing_is_emitted_before_the_bar_becomes_available(tmp_path):
    _write(tmp_path, [_bar(100, 200)])
    assert a_replay(tmp_path).poll(199) == ()


def test_a_bar_is_emitted_once_its_availability_time_has_arrived(tmp_path):
    _write(tmp_path, [_bar(100, 200)])
    (event,) = a_replay(tmp_path).poll(200)
    assert event.symbol == "BTCUSDT" and event.venue == "binance"
    assert event.event_time_ns == 100


def test_the_same_bar_is_never_emitted_twice(tmp_path):
    _write(tmp_path, [_bar(100, 200)])
    replay = a_replay(tmp_path)
    assert len(replay.poll(200)) == 1
    assert replay.poll(300) == ()


def test_advancing_the_clock_emits_only_what_newly_became_available(tmp_path):
    _write(tmp_path, [_bar(100, 200), _bar(300, 400)])
    replay = a_replay(tmp_path)
    replay.poll(200)
    (second,) = replay.poll(400)
    assert second.event_time_ns == 300


def test_events_arrive_in_event_time_order_within_a_poll(tmp_path):
    _write(tmp_path, [_bar(300, 400), _bar(100, 400), _bar(200, 400)])
    events = a_replay(tmp_path).poll(400)
    assert [e.event_time_ns for e in events] == [100, 200, 300]


# --- corrections cannot un-trade a bar --------------------------------------

def test_a_correction_to_an_already_emitted_bar_is_counted_not_re_emitted(
        tmp_path):
    _write(tmp_path, [_bar(100, 200, close=100.0)], "snap1")
    replay = a_replay(tmp_path)
    replay.poll(200)
    _write(tmp_path, [_bar(100, 900, close=105.0)], "snap2")
    assert replay.poll(1_000) == ()
    assert replay.corrections_after_emission == 1


def test_re_reading_the_same_row_is_not_a_correction(tmp_path):
    """Found by running it, 2026-08-15: poll 2 against the live store reported
    3,494 corrections where nothing had been corrected. Every poll re-reads every
    row it has already fed, so holding only the keys made a re-read and a genuine
    correction indistinguishable — and the count that was meant to expose a real
    problem became noise proportional to uptime."""
    _write(tmp_path, [_bar(100, 200), _bar(160, 200)])
    replay = a_replay(tmp_path)
    replay.poll(200)
    for _ in range(5):
        assert replay.poll(300) == ()
    assert replay.corrections_after_emission == 0


def test_a_correction_that_lands_before_the_bar_was_ever_emitted_is_the_one_used(
        tmp_path):
    _write(tmp_path, [_bar(100, 200, close=100.0)], "snap1")
    _write(tmp_path, [_bar(100, 900, close=105.0)], "snap2")
    replay = a_replay(tmp_path)
    (event,) = replay.poll(1_000)
    assert event.market_event.trade_price == Decimal("105.0")
    assert replay.corrections_after_emission == 0


# --- the shape the fill model takes -----------------------------------------

def test_the_trade_price_is_the_bar_close_and_the_size_its_volume(tmp_path):
    _write(tmp_path, [_bar(100, 200, close=100.0, volume=7.0)])
    (event,) = a_replay(tmp_path).poll(200)
    assert event.market_event.trade_price == Decimal("100.0")
    assert event.market_event.trade_quantity == Decimal("7.0")


def test_the_touch_comes_from_the_bar_extremes_and_is_flagged_as_derived(
        tmp_path):
    _write(tmp_path, [_bar(100, 200, high=101.0, low=99.0)])
    (event,) = a_replay(tmp_path).poll(200)
    assert event.market_event.best_bid == Decimal("99.0")
    assert event.market_event.best_ask == Decimal("101.0")
    assert event.touch_is_derived is True


def test_prices_are_decimal_because_a_float_price_is_a_float_pnl(tmp_path):
    _write(tmp_path, [_bar(100, 200)])
    (event,) = a_replay(tmp_path).poll(200)
    assert isinstance(event.market_event.trade_price, Decimal)


# --- refusals over quiet defaults -------------------------------------------

def test_a_bar_with_a_non_positive_price_is_refused_not_traded(tmp_path):
    """Binance emits placeholder frames; 746 bars ate one into `low` once."""
    _write(tmp_path, [_bar(100, 200, close=0.0)])
    replay = a_replay(tmp_path)
    assert replay.poll(200) == ()
    assert replay.refused_invalid_price == 1


def test_a_bar_with_no_volume_prints_nothing_to_fill_against(tmp_path):
    _write(tmp_path, [_bar(100, 200, volume=0.0)])
    replay = a_replay(tmp_path)
    assert replay.poll(200) == ()
    assert replay.refused_no_volume == 1


# --- selection --------------------------------------------------------------

def test_only_the_requested_symbols_are_fed(tmp_path):
    _write(tmp_path, [_bar(100, 200, symbol="BTCUSDT"),
                      _bar(100, 200, symbol="ETHUSDT")])
    replay = a_replay(tmp_path, symbols=("BTCUSDT",))
    assert {e.symbol for e in replay.poll(200)} == {"BTCUSDT"}


def test_the_same_symbol_on_two_venues_stays_two_streams(tmp_path):
    _write(tmp_path, [_bar(100, 200, venue="binance"),
                      _bar(100, 200, venue="hyperliquid")])
    events = a_replay(tmp_path).poll(200)
    assert {e.venue for e in events} == {"binance", "hyperliquid"}


def test_an_empty_store_is_quiet_rather_than_an_error(tmp_path):
    assert a_replay(tmp_path).poll(10**12) == ()
