"""The reader is the only door into the store, so these tests try to walk past it."""
from __future__ import annotations

import pandas as pd
import pytest

from store.clock_gated_reader import ClockGatedReader, join_as_of
from store.parquet_partition import append_partition
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE


def _row(symbol: str, event: int, ingested: int, available: int, close: float) -> dict:
    return {SYMBOL: symbol, VENUE: "binance", EVENT_TIME: event,
            INGESTION_TIME: ingested, AVAILABILITY_TIME: available, "close": close}


def _write(tmp_path, rows: list[dict], snapshot: str) -> None:
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, "bars_1m", frame, snapshot)


def test_rows_not_yet_available_are_invisible(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 63000.0)], "snap1")
    reader = ClockGatedReader(tmp_path, "bars_1m")
    assert reader.read_as_of(199).empty
    assert len(reader.read_as_of(200)) == 1


def test_availability_is_inclusive_at_the_exact_instant(tmp_path):
    """A row available at T is usable at T. Off by one here is a silent half-bar."""
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 63000.0)], "snap1")
    assert len(ClockGatedReader(tmp_path, "bars_1m").read_as_of(200)) == 1


def test_a_correction_is_invisible_until_its_own_availability_time(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 63000.0)], "snap1")
    _write(tmp_path, [_row("BTCUSDT", 100, 900, 950, 63500.0)], "snap2")
    reader = ClockGatedReader(tmp_path, "bars_1m")
    early = reader.read_as_of(500)
    assert len(early) == 1 and early.iloc[0]["close"] == 63000.0


def test_the_latest_available_version_wins(tmp_path):
    """Two rows for one event: the reader must pick the freshest one it may see."""
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 63000.0)], "snap1")
    _write(tmp_path, [_row("BTCUSDT", 100, 900, 950, 63500.0)], "snap2")
    late = ClockGatedReader(tmp_path, "bars_1m").read_as_of(1_000)
    assert len(late) == 1, "a correction must replace, not duplicate"
    assert late.iloc[0]["close"] == 63500.0


def test_reading_later_never_removes_what_an_earlier_read_showed(tmp_path):
    """Monotonicity. A backtest that re-reads must never see history shrink."""
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 1.0),
                      _row("BTCUSDT", 300, 350, 400, 2.0)], "snap1")
    reader = ClockGatedReader(tmp_path, "bars_1m")
    early_events = set(reader.read_as_of(250)[EVENT_TIME])
    later_events = set(reader.read_as_of(500)[EVENT_TIME])
    assert early_events <= later_events


def test_symbol_filter_narrows_without_changing_gating(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 1.0),
                      _row("ETHUSDT", 100, 150, 200, 2.0)], "snap1")
    reader = ClockGatedReader(tmp_path, "bars_1m")
    assert set(reader.read_as_of(200, symbols=["BTCUSDT"])[SYMBOL]) == {"BTCUSDT"}


def test_empty_store_reads_empty(tmp_path):
    assert ClockGatedReader(tmp_path, "absent").read_as_of(10**18).empty


def test_distinct_venues_sharing_symbol_and_event_time_both_survive(tmp_path):
    """Correction resolution must key on (symbol, venue, event_time), not just
    (symbol, event_time).

    build_bars keys bars by (symbol, venue, bar_open_ns), so two venues can
    legitimately produce distinct bars carrying the same symbol and the same
    event time. Grouping without venue would silently collapse them into one
    and drop a venue's bar - a data-loss bug, not a correction. A correction
    replaces the same venue's bar, never another venue's.
    """
    frame = pd.DataFrame([
        {SYMBOL: "BTC", VENUE: "binance", EVENT_TIME: 100, INGESTION_TIME: 150,
         AVAILABILITY_TIME: 200, "close": 63000.0},
        {SYMBOL: "BTC", VENUE: "hyperliquid", EVENT_TIME: 100, INGESTION_TIME: 150,
         AVAILABILITY_TIME: 200, "close": 63010.0},
    ]).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, "bars_1m", frame, "snap1")
    visible = ClockGatedReader(tmp_path, "bars_1m").read_as_of(200)
    assert len(visible) == 2, "one row coming back means a venue's bar was dropped"
    assert set(visible[VENUE]) == {"binance", "hyperliquid"}


def test_correction_resolution_does_not_splice_a_stale_field_from_an_older_row(tmp_path):
    """`.last()` on a groupby picks the last non-null value per column
    independently, not the last physical row. If a correction carries a null in
    a non-key column, that composes a row that never existed in the store -
    stitching the older row's value onto the newer row's identity. The reader
    must return the correction's own null, not resurrect the stale value.
    """
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 63000.0)], "snap1")
    correction = pd.DataFrame([{
        SYMBOL: "BTCUSDT", VENUE: "binance", EVENT_TIME: 100,
        INGESTION_TIME: 250, AVAILABILITY_TIME: 300, "close": None,
    }]).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, "bars_1m", correction, "snap2")

    visible = ClockGatedReader(tmp_path, "bars_1m").read_as_of(1_000)
    assert len(visible) == 1
    assert pd.isna(visible.iloc[0]["close"]), "the newer row's null was overwritten by a stale value"


def test_symbol_filter_matching_nothing_returns_empty_without_raising(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", 100, 150, 200, 1.0)], "snap1")
    result = ClockGatedReader(tmp_path, "bars_1m").read_as_of(200, symbols=["DOGEUSDT"])
    assert result.empty


def test_join_keys_on_availability_time_not_event_time():
    """Joining on event time is the classic leak, so the join refuses to do it.

    The right-hand row describes event 100 but only became available at 900. A
    join on event time attaches it to the left row at event 100; the correct
    answer is that nothing was available yet.
    """
    left = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [100],
                         AVAILABILITY_TIME: [200], "signal": [1.0]})
    right = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [100],
                          AVAILABILITY_TIME: [900], "funding": [0.01]})
    joined = join_as_of(left, right, suffix="_funding")
    assert pd.isna(joined.loc[0, "funding"])


def test_join_attaches_the_most_recent_already_available_row():
    left = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [500],
                         AVAILABILITY_TIME: [500], "signal": [1.0]})
    right = pd.DataFrame({SYMBOL: ["BTCUSDT", "BTCUSDT"], EVENT_TIME: [100, 400],
                          AVAILABILITY_TIME: [100, 400], "funding": [0.01, 0.02]})
    joined = join_as_of(left, right, suffix="_funding")
    assert joined.loc[0, "funding"] == pytest.approx(0.02)


def test_join_tolerance_refuses_a_stale_match():
    """An hour-old funding rate is not context; it is a different market."""
    left = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [10_000],
                         AVAILABILITY_TIME: [10_000], "signal": [1.0]})
    right = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [1], AVAILABILITY_TIME: [1],
                          "funding": [0.01]})
    joined = join_as_of(left, right, suffix="_funding", tolerance_ns=100)
    assert pd.isna(joined.loc[0, "funding"])
