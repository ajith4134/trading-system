"""Append-only is a property, not an intention, so these tests try to violate it."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from store.parquet_partition import (
    PartitionExistsError, append_partition, compute_snapshot_id, read_dataset,
)
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE


def _frame(symbol: str = "BTCUSDT", available: int = 1_100) -> pd.DataFrame:
    return pd.DataFrame({
        SYMBOL: [symbol],
        VENUE: ["binance"],
        EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050],
        AVAILABILITY_TIME: [available],
        "close": [63113.20],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})


def test_snapshot_id_is_derived_from_content_not_names(tmp_path):
    """Two files with identical bytes and different names must share an id.

    The snapshot id answers "which data produced this result". Deriving it from
    filenames would let a rename look like different data, and a rewrite of the
    same name look like the same data.
    """
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"same"); second.write_bytes(b"same")
    assert compute_snapshot_id([first]) == compute_snapshot_id([second])


def test_snapshot_id_changes_when_content_changes(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"one")
    before = compute_snapshot_id([path])
    path.write_bytes(b"two")
    assert compute_snapshot_id([path]) != before


def test_snapshot_id_is_order_independent(tmp_path):
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"one"); second.write_bytes(b"two")
    assert compute_snapshot_id([first, second]) == compute_snapshot_id([second, first])


def test_rows_are_partitioned_by_symbol(tmp_path):
    append_partition(tmp_path, "bars_1m", _frame("BTCUSDT"), "snap1")
    append_partition(tmp_path, "bars_1m", _frame("ETHUSDT"), "snap1")
    symbols = {p.name for p in (tmp_path / "bars_1m").iterdir()}
    assert symbols == {"symbol=BTCUSDT", "symbol=ETHUSDT"}


def test_writing_the_same_snapshot_twice_is_refused(tmp_path):
    """Rebuilding over an existing part would erase history the store promises to keep."""
    append_partition(tmp_path, "bars_1m", _frame(), "snap1")
    with pytest.raises(PartitionExistsError, match="snap1"):
        append_partition(tmp_path, "bars_1m", _frame(), "snap1")


def test_a_partial_collision_leaves_no_part_behind(tmp_path):
    """A refused write must leave no trace, even when only one symbol in a multi-symbol
    frame collides.

    append_partition used to check-then-write symbol by symbol, so a frame with two
    symbols where only the second collided would already have the first symbol's part
    on disk by the time the collision was discovered - a half-written snapshot, which
    is exactly the partial state the store promises never to hold.
    """
    append_partition(tmp_path, "bars_1m", _frame("ETHUSDT"), "snap1")
    mixed = pd.concat([_frame("BTCUSDT"), _frame("ETHUSDT")], ignore_index=True)
    with pytest.raises(PartitionExistsError, match="snap1"):
        append_partition(tmp_path, "bars_1m", mixed, "snap1")
    assert not (tmp_path / "bars_1m" / "symbol=BTCUSDT").exists(), (
        "the non-colliding symbol's part must not be written when another symbol in "
        "the same frame collides")


def test_a_correction_is_a_new_part_not_an_edit(tmp_path):
    append_partition(tmp_path, "bars_1m", _frame(available=1_100), "snap1")
    append_partition(tmp_path, "bars_1m", _frame(available=9_999), "snap2")
    stored = read_dataset(tmp_path, "bars_1m")
    assert len(stored) == 2, "the original row must survive the correction"
    assert sorted(stored[AVAILABILITY_TIME]) == [1_100, 9_999]


def test_invalid_frames_are_refused_before_they_reach_disk(tmp_path):
    from store.temporal_schema import TemporalInvariantError
    bad = _frame()
    bad[AVAILABILITY_TIME] = bad[INGESTION_TIME] - 1
    with pytest.raises(TemporalInvariantError):
        append_partition(tmp_path, "bars_1m", bad, "snap1")
    assert not (tmp_path / "bars_1m").exists(), "a refused write must leave no trace"


def test_reading_an_absent_dataset_returns_empty_not_error(tmp_path):
    stored = read_dataset(tmp_path, "never_written")
    assert stored.empty
