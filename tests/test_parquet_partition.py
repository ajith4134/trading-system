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
    symbols = {p.parent.name for p in (tmp_path / "bars_1m").glob("*/symbol=*/*.parquet")}
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
    assert not any((tmp_path / "bars_1m").glob("*/symbol=BTCUSDT")), (
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


def test_a_crash_mid_write_leaves_the_whole_dataset_readable(tmp_path, monkeypatch):
    """A torn part must never become the dataset's problem.

    `pq.write_table` writing straight to its target means a crash or a power loss
    mid-write leaves a truncated Parquet file under the target name, and
    `read_dataset` then raises ArrowInvalid for the ENTIRE dataset - every symbol
    becomes unreadable, not just the one being written. Rebuilding cannot help:
    the snapshot id is content-derived and unchanged, so it collides with the
    truncated part, and there is no delete path by design. Recovery needs manual
    filesystem surgery.

    Writing to a temporary file and renaming it into place makes a part either
    absent or complete, which leaves the rebuild as the whole recovery path.
    """
    import pyarrow.parquet as pq

    from store import parquet_partition

    append_partition(tmp_path, "bars_1m", _frame("BTCUSDT"), "snap1")

    def die_halfway(table, where, **kwargs):
        if hasattr(where, "write"):
            where.write(b"PAR1this is half a parquet file")
        else:
            Path(where).write_bytes(b"PAR1this is half a parquet file")
        raise OSError("simulated power loss mid-write")

    monkeypatch.setattr(parquet_partition.pq, "write_table", die_halfway)
    with pytest.raises(OSError):
        append_partition(tmp_path, "bars_1m", _frame("ETHUSDT"), "snap2")

    survivors = read_dataset(tmp_path, "bars_1m")
    assert list(survivors[SYMBOL]) == ["BTCUSDT"], "the crash cost more than its own part"

    leftovers = sorted(p.name for p in (tmp_path / "bars_1m").glob("*/symbol=ETHUSDT/*"))
    assert leftovers == [], f"a partial part survived the crash: {leftovers}"


def test_a_column_added_by_a_later_partition_is_not_silently_dropped(tmp_path):
    """pyarrow infers a dataset's schema from the FIRST fragment it discovers,
    so a column added later is absent from that schema and never returned. The
    data is on disk; the read simply does not produce it, and nothing says so.

    Measured 2026-08-09: `funding_interval_hours` was written into bybit's
    funding partition - the only venue publishing a per-symbol interval, and
    annualising a 4-hourly rate as 8-hourly is wrong by a factor of two - and
    `read_dataset` returned a frame without the column at all.

    An append-only store whose reader quietly forgets a field is worse than one
    that refuses, because the refusal is visible.
    """
    import pandas as pd
    from store.parquet_partition import append_partition, read_dataset

    def row(symbol, value, **extra):
        return pd.DataFrame({
            "symbol": [symbol], "venue": ["binance"], "value": [value],
            "event_time_ns": [1], "ingestion_time_ns": [1],
            "availability_time_ns": [1], **{k: [v] for k, v in extra.items()}})

    early = row("A", 1)
    later = row("B", 2, added_later="yes")
    append_partition(tmp_path, "d", early, "first")
    append_partition(tmp_path, "d", later, "second")

    frame = read_dataset(tmp_path, "d")

    assert "added_later" in frame.columns, "a written column vanished on read"
    assert set(frame["value"]) == {1, 2}
    # Absent in the older partition reads as null - the honest answer, because it
    # was not recorded then.
    assert frame.loc[frame["symbol"] == "A", "added_later"].isna().all()
    assert frame.loc[frame["symbol"] == "B", "added_later"].iloc[0] == "yes"


def test_the_column_survives_whichever_partition_is_discovered_first(tmp_path):
    """The bug depended on discovery order, so the fix must not."""
    import pandas as pd
    from store.parquet_partition import append_partition, read_dataset

    def row(symbol, value, **extra):
        return pd.DataFrame({
            "symbol": [symbol], "venue": ["binance"], "value": [value],
            "event_time_ns": [1], "ingestion_time_ns": [1],
            "availability_time_ns": [1], **{k: [v] for k, v in extra.items()}})

    append_partition(tmp_path, "d", row("Z", 1, added_later="yes"), "zfirst")
    append_partition(tmp_path, "d", row("A", 2), "asecond")

    frame = read_dataset(tmp_path, "d")
    assert "added_later" in frame.columns
    assert len(frame) == 2


# --- SL-17 / RL-032: whole-universe datasets partition by hour alone --------
#
# One sealed funding hour held 1,892 fragments for 26,015 rows and 16.2 MiB -
# 13.8 rows and 8.8 KiB per file - at 29 ms each to open, so the whole 850 MB
# dataset cost 30.6 minutes to read and grew by ~1,021 files an hour. That is
# what froze the status wall for two days. Nothing reads funding one symbol at a
# time: every consumer wants the whole universe for a window, so the `symbol=`
# level buys nothing and costs a file per symbol per hour.

def _funding_frame(symbols=("BTCUSDT", "ETHUSDT"), hour_ns=None, venue="binance"):
    import pandas as pd
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)
    base = hour_ns if hour_ns is not None else 1_787_000_000_000_000_000
    n = len(symbols)
    return pd.DataFrame({
        SYMBOL: list(symbols),
        VENUE: [venue] * n,
        EVENT_TIME: pd.array([base + i for i in range(n)], dtype="int64"),
        INGESTION_TIME: pd.array([base + i for i in range(n)], dtype="int64"),
        AVAILABILITY_TIME: pd.array([base + i for i in range(n)], dtype="int64"),
        "funding_rate": [0.0001 * (i + 1) for i in range(n)],
    })


def test_a_whole_universe_dataset_writes_one_part_per_hour_not_per_symbol(tmp_path):
    from store.parquet_partition import append_partition
    written = append_partition(tmp_path, "funding",
                               _funding_frame(("BTCUSDT", "ETHUSDT", "SOLUSDT")),
                               "funding-binance-2026-08-19-upto1")

    assert len(written) == 1, "three symbols, one part"
    assert "symbol=" not in str(written[0]), written[0]
    assert "availability_hour=" in str(written[0])


def test_the_symbol_travels_in_the_body_when_the_path_no_longer_carries_it(tmp_path):
    """It is the column this store has already lost once, silently, on
    2026-08-09 - `funding_interval_hours`, which turned a 4-hourly funding rate
    into an 8-hourly one."""
    from store.parquet_partition import append_partition, read_dataset
    from store.temporal_schema import SYMBOL
    append_partition(tmp_path, "funding", _funding_frame(("BTCUSDT", "ETHUSDT")),
                     "funding-binance-2026-08-19-upto1")

    frame = read_dataset(tmp_path, "funding")
    assert set(frame[SYMBOL]) == {"BTCUSDT", "ETHUSDT"}
    assert set(frame["funding_rate"]) == {0.0001, 0.0002}


def test_a_dataset_nobody_prunes_by_symbol_gets_no_symbol_partition(tmp_path):
    """**This test used to assert the opposite, on a premise the code does not
    support.** Bars was kept per-symbol because it is "read one symbol at a
    time" - but `ClockGatedReader` applies its `symbols` argument to the
    already-materialised frame, and the only pushdown filters the store builds
    are on availability time and the hour. Nothing prunes by the `symbol=`
    partition, so it cost `bars_60000000000ns` 208,880 fragments and pruned for
    nobody (RL-034, measured 2026-08-19).

    A symbol level earns its keep again the day a reader pushes a symbol filter
    INTO the scan; adding the dataset back to WHOLE_UNIVERSE_DATASETS is how
    that gets reversed.
    """
    from store.parquet_partition import append_partition
    written = append_partition(tmp_path, "bars_60000000000ns",
                               _funding_frame(("BTCUSDT", "ETHUSDT")),
                               "bars-binance-2026-08-19-upto1")

    assert len(written) == 1
    assert "symbol=" not in str(written[0])


def test_an_unknown_dataset_still_keeps_its_symbol_partition(tmp_path):
    """The frozenset is the decision, not the default. A dataset nobody has
    reasoned about keeps the conservative layout rather than inheriting one."""
    from store.parquet_partition import append_partition
    written = append_partition(tmp_path, "some_new_dataset",
                               _funding_frame(("BTCUSDT", "ETHUSDT")),
                               "x-binance-2026-08-19-upto1")

    assert len(written) == 2
    assert all("symbol=" in str(p) for p in written)


def test_two_hours_in_one_frame_still_land_in_two_hour_directories(tmp_path):
    """A frame is not one hour, and the hour in the path may never be a lie."""
    import pandas as pd
    from store.parquet_partition import append_partition
    early = _funding_frame(("BTCUSDT",), hour_ns=1_787_000_000_000_000_000)
    later = _funding_frame(("ETHUSDT",), hour_ns=1_787_003_600_000_000_000 + 10**10)
    written = append_partition(tmp_path, "funding",
                               pd.concat([early, later], ignore_index=True),
                               "funding-binance-2026-08-19-upto1")

    assert len(written) == 2
    assert len({p.parent for p in written}) == 2


def test_a_repeated_snapshot_id_is_still_refused_in_the_new_layout(tmp_path):
    """Corrections are new snapshots, never rewrites - the promise the reader
    is built on, and it must not weaken with the layout."""
    import pytest
    from store.parquet_partition import PartitionExistsError, append_partition
    append_partition(tmp_path, "funding", _funding_frame(),
                     "funding-binance-2026-08-19-upto1")
    with pytest.raises(PartitionExistsError):
        append_partition(tmp_path, "funding", _funding_frame(),
                         "funding-binance-2026-08-19-upto1")


def test_a_dataset_already_holding_symbol_parts_keeps_being_written_that_way(tmp_path):
    """**A mixed dataset is unreadable, not merely slow.** Measured 2026-08-19:
    one `symbol=` part beside one hour-only part in the same dataset gives
    `ArrowTypeError: Unable to merge: Field symbol has incompatible types:
    large_string vs string` - the path-derived column is `string`, the body one
    `large_string`, and pyarrow refuses.

    So the writer asks the DATA which layout it is in rather than trusting a
    constant, which could otherwise be true before the existing data was
    converted. The conversion flips the layout once and the writer follows.
    """
    from store.parquet_partition import append_partition, read_dataset
    from store.temporal_schema import SYMBOL

    # A dataset that already holds the old layout, as the live store does.
    old = tmp_path / "funding" / "availability_hour=2026-08-17T20" / "symbol=BTCUSDT"
    old.mkdir(parents=True)
    (old / "part-funding-binance-old-upto1.parquet").write_bytes(b"")

    written = append_partition(tmp_path, "funding",
                               _funding_frame(("ETHUSDT", "SOLUSDT")),
                               "funding-binance-2026-08-19-upto2")

    assert all("symbol=" in str(p) for p in written), (
        "a converted layout must never be written into an unconverted dataset")


def test_an_empty_dataset_takes_the_new_layout(tmp_path):
    """Nothing to mix with, so a fresh whole-universe dataset starts converted."""
    from store.parquet_partition import append_partition
    written = append_partition(tmp_path, "funding", _funding_frame(),
                               "funding-binance-2026-08-19-upto1")
    assert len(written) == 1 and "symbol=" not in str(written[0])
