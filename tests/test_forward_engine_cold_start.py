"""A restart must not re-read the archive to learn what it already knew.

Measured 2026-08-17: the forward engine was OOM-killed at 09:49:54Z (exit 137,
7,183s, the fourth restart that day) holding 5.5 GB. Three causes, and a
watermark answers all three:

  `prime()` built a list of 2.3M TapeEvents only to len() it.
  `poll()` re-read the ENTIRE bars dataset every 60 seconds.
  `_emitted` accumulated one entry per event ever seen, forever.

With a persisted availability watermark a restart skips the full prime, each
poll reads only what is new, and the emitted set only ever holds rows since the
watermark.

The property that must NOT break is the 2026-08-15 defect: a fresh journal
replaying archived bars as though they were live produced 1,074 fills at prices
days old. So a resume marks seen and never trades, a marker from another
strategy is ignored, and a corrupt marker costs a slow start rather than a wrong
one.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from paper import forward_engine
from paper.market_replay import MarketReplay
from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)
from validation.holdout_custodian import HoldoutCustodian


def _row(event: int, available: int, close: float) -> dict:
    return {SYMBOL: "BTCUSDT", VENUE: "binance", EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available,
            "close": close, "high": close, "low": close, "volume": 1.0}


def _store(tmp_path, rows, snapshot="snap1"):
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, "bars_1m", frame, snapshot)
    return tmp_path


def _replay(store):
    custodian = HoldoutCustodian(store, holdout_start_ns=10**18,
                                 holdout_end_ns=10**18 + 1)
    return MarketReplay(ClockGatedReader(store, "bars_1m", custodian=custodian))


# --- the marker -----------------------------------------------------------

def test_a_prime_marker_is_written_and_names_the_watermark(tmp_path):
    marker = forward_engine.write_prime_marker(tmp_path, last_availability_ns=123)
    assert marker.is_file()
    assert forward_engine.read_prime_marker(tmp_path) == 123


def test_no_marker_means_a_full_prime_rather_than_trading_the_archive(tmp_path):
    assert forward_engine.read_prime_marker(tmp_path) is None


def test_a_marker_from_a_different_strategy_is_ignored(tmp_path):
    forward_engine.write_prime_marker(tmp_path, last_availability_ns=123,
                                      strategy="plumbing-momentum")
    assert forward_engine.read_prime_marker(tmp_path, strategy="other") is None
    assert forward_engine.read_prime_marker(
        tmp_path, strategy="plumbing-momentum") == 123


def test_a_corrupt_marker_falls_back_to_a_full_prime(tmp_path):
    (tmp_path / "prime-marker.json").write_text("{not json")
    assert forward_engine.read_prime_marker(tmp_path) is None


# --- marking seen without materialising -----------------------------------

def test_mark_seen_counts_without_producing_events(tmp_path):
    replay = _replay(_store(tmp_path, [_row(100, 100, 1.0), _row(200, 200, 2.0)]))
    assert replay.mark_seen(1_000) == 2
    assert replay.poll(1_000) == (), "everything marked seen must not be fed"


def test_mark_seen_advances_the_watermark_so_polls_read_only_what_is_new(tmp_path):
    store = _store(tmp_path, [_row(100, 100, 1.0), _row(200, 200, 2.0)])
    replay = _replay(store)
    replay.mark_seen(1_000)
    assert replay.availability_watermark == 200

    _store(store, [_row(300, 300, 3.0)], snapshot="snap2")
    produced = replay.poll(2_000)
    assert [e.event_time_ns for e in produced] == [300]
    assert replay.availability_watermark == 300


def test_a_bounded_resume_never_trades_a_row_it_marked(tmp_path):
    """The 2026-08-15 defect: archived bars must never enter as forward fills."""
    store = _store(tmp_path, [_row(100, 100, 1.0), _row(200, 200, 2.0)])
    resumed = _replay(store)
    resumed.mark_seen(1_000, not_before_ns=150)
    # Rows at or after the bound are marked, not traded.
    assert resumed.poll(1_000) == ()


def test_a_correction_after_the_watermark_is_still_recognised_as_one(tmp_path):
    store = _store(tmp_path, [_row(100, 100, 1.0)])
    replay = _replay(store)
    replay.poll(1_000)
    assert replay.corrections_after_emission == 0

    _store(store, [_row(100, 500, 1.5)], snapshot="snap2")
    replay.poll(1_000)
    assert replay.corrections_after_emission == 1, (
        "a corrected bar we already traded on is a fact about the run")


def test_the_watermark_never_moves_backwards(tmp_path):
    store = _store(tmp_path, [_row(100, 100, 1.0), _row(200, 200, 2.0)])
    replay = _replay(store)
    replay.poll(1_000)
    high = replay.availability_watermark
    replay.poll(1_000)
    assert replay.availability_watermark == high
