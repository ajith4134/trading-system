"""Reading only what is new, without changing what a full read means.

`read_as_of` loads the whole dataset and filters in pandas. For a board that is
fine; for the forward paper engine polling every 60 seconds against an archive
growing ~3.4 GB/day it is not. Measured 2026-08-17: the engine held 5.5 GB,
`statuswall.cli` held 4.9 GB, and the kernel OOM-killed the engine (exit 137)
after 7,183s - the fourth restart that day.

So `not_before_ns` pushes a lower bound on AVAILABILITY time into the parquet
scan. The bound is on availability rather than event time on purpose: a
correction to an old bar carries a LATER availability time by definition, so it
still arrives, while rows already fed cannot come back.

The tests that matter are the ones asserting nothing else changed. A default of
None must read exactly what it read before, and the correction-resolution rule -
newest visible version wins - must survive the bound.
"""
from __future__ import annotations

import pandas as pd

from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition, read_dataset
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)


def _row(symbol: str, event: int, available: int, close: float) -> dict:
    return {SYMBOL: symbol, VENUE: "binance", EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available, "close": close}


def _write(tmp_path, rows: list[dict], snapshot: str) -> None:
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, "bars_1m", frame, snapshot)


def _stocked(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", 100, 100, 1.0),
                      _row("BTCUSDT", 200, 200, 2.0),
                      _row("BTCUSDT", 300, 300, 3.0),
                      _row("ETHUSDT", 200, 200, 9.0)], "snap1")
    return tmp_path


def test_no_bound_reads_exactly_what_it_read_before(tmp_path):
    """The default must be indistinguishable from the previous behaviour."""
    store = _stocked(tmp_path)
    assert len(read_dataset(store, "bars_1m")) == 4
    assert len(read_dataset(store, "bars_1m", not_before_ns=None)) == 4


def test_a_bound_drops_only_rows_already_available_before_it(tmp_path):
    store = _stocked(tmp_path)
    frame = read_dataset(store, "bars_1m", not_before_ns=200)
    assert sorted(frame[AVAILABILITY_TIME]) == [200, 200, 300]


def test_the_bound_is_inclusive_so_a_row_at_the_watermark_is_not_lost(tmp_path):
    """Off by one here silently drops a bar on every poll."""
    store = _stocked(tmp_path)
    frame = read_dataset(store, "bars_1m", not_before_ns=300)
    assert list(frame[AVAILABILITY_TIME]) == [300]


def test_a_bound_past_everything_reads_empty_rather_than_raising(tmp_path):
    store = _stocked(tmp_path)
    assert read_dataset(store, "bars_1m", not_before_ns=10_000).empty


def test_the_reader_passes_the_bound_through_and_still_gates_on_the_clock(tmp_path):
    store = _stocked(tmp_path)
    reader = ClockGatedReader(store, "bars_1m")
    assert len(reader.read_as_of(1_000)) == 4
    assert len(reader.read_as_of(1_000, not_before_ns=200)) == 3
    # The clock still wins: a bound cannot reveal a row that is not yet available.
    assert len(reader.read_as_of(150, not_before_ns=0)) == 1


def test_a_correction_still_arrives_through_the_bound(tmp_path):
    """A correction carries a LATER availability time, so a bound cannot hide it."""
    store = _stocked(tmp_path)
    _write(store, [_row("BTCUSDT", 100, 500, 1.5)], "snap2")
    reader = ClockGatedReader(store, "bars_1m")
    bounded = reader.read_as_of(1_000, not_before_ns=400)
    corrected = bounded[bounded[EVENT_TIME] == 100]
    assert len(corrected) == 1
    assert float(corrected.iloc[0]["close"]) == 1.5, (
        "the newest visible version must win, bound or no bound")
