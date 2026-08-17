"""Walking every fragment once per generation instead of once per read.

Measured 2026-08-17 on the live bars dataset: 49,100 parquet fragments across
2,229 symbol partitions, and `read_dataset` opened all of them to read
`physical_schema` BEFORE the scan opened them again. A read filtered to return
zero rows did not complete in 10 minutes, which isolates the cost to the file
walk rather than to any row work. By 11:30 the same walk had stalled the boards
generator: `status-wall.html` went stale for 1h50m while the process sat at
5.3 GB RSS and 482,473 read syscalls.

The walk cannot simply be deleted. Without it pyarrow infers the dataset schema
from the first fragment it discovers and SILENTLY DROPS columns that only later
partitions carry - on 2026-08-09 `funding_interval_hours` vanished from every
read that way, and annualising a 4-hourly rate as 8-hourly is wrong by a factor
of two.

So the walk is cached per dataset generation, and a later read pays only for the
fragments that appeared since. The tests that matter are the ones asserting the
cache cannot change an answer: the column-drop regression has to stay fixed
THROUGH the cache, and a fragment disappearing has to force a full rebuild
rather than leave a column asserted from a file that is gone.
"""
from __future__ import annotations

import pandas as pd
import pytest

from store.parquet_partition import (
    append_partition, clear_schema_cache, count_fragment_schema_reads,
    read_dataset,
)
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)


@pytest.fixture(autouse=True)
def _isolated_cache():
    """No test may inherit another's cache, or its fragment-read count."""
    clear_schema_cache()
    yield
    clear_schema_cache()


def _row(symbol: str, event: int, available: int, **extra) -> dict:
    return {SYMBOL: symbol, VENUE: "binance", EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available,
            "close": 1.0, **extra}


def _write(store, rows: list[dict], snapshot: str, dataset: str = "bars_1m") -> None:
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(store, dataset, frame, snapshot)


# --- the cache must not change any answer -----------------------------------

def test_a_cached_read_returns_what_an_uncached_read_returns(tmp_path):
    """Parity is the whole contract: same rows, same columns, same order."""
    _write(tmp_path, [_row("BTCUSDT", 100, 100), _row("ETHUSDT", 200, 200)], "snap1")
    first = read_dataset(tmp_path, "bars_1m")
    cached = read_dataset(tmp_path, "bars_1m")
    clear_schema_cache()
    uncached = read_dataset(tmp_path, "bars_1m")
    assert list(cached.columns) == list(uncached.columns) == list(first.columns)
    pd.testing.assert_frame_equal(cached, uncached)


def test_a_column_added_by_a_later_partition_survives_the_cache(tmp_path):
    """The `funding_interval_hours` regression, run through a warm cache.

    The first read populates the cache from partitions that do not carry the
    column. If the second read trusted that cache wholesale, the column would be
    dropped exactly as pyarrow dropped it on 2026-08-09 - the cache would have
    reintroduced the bug the walk exists to prevent.
    """
    _write(tmp_path, [_row("BTCUSDT", 100, 100)], "snap1")
    assert "funding_interval_hours" not in read_dataset(tmp_path, "bars_1m").columns

    _write(tmp_path, [_row("BYBITBTC", 200, 200, funding_interval_hours=4.0)], "snap2")
    frame = read_dataset(tmp_path, "bars_1m")
    assert "funding_interval_hours" in frame.columns
    assert frame.loc[frame[SYMBOL] == "BYBITBTC", "funding_interval_hours"].iloc[0] == 4.0


def test_a_disappearing_fragment_forces_a_full_rebuild(tmp_path):
    """Compaction or a restore must not leave a column asserted from a dead file.

    Union is monotonic - it can only add. So a cache that only ever grows would
    keep reporting a column whose only source has been removed, and a fresh
    process would disagree with a long-lived one about what the dataset holds.
    """
    _write(tmp_path, [_row("BTCUSDT", 100, 100)], "snap1")
    _write(tmp_path, [_row("BYBITBTC", 200, 200, funding_interval_hours=4.0)], "snap2")
    assert "funding_interval_hours" in read_dataset(tmp_path, "bars_1m").columns

    for part in (tmp_path / "bars_1m").rglob("*.parquet"):
        if part.parent.name.endswith("BYBITBTC"):
            part.unlink()
    assert "funding_interval_hours" not in read_dataset(tmp_path, "bars_1m").columns


def test_two_datasets_do_not_share_a_cache_entry(tmp_path):
    """Keyed by dataset root, or one dataset's schema leaks into another's."""
    _write(tmp_path, [_row("BTCUSDT", 100, 100, only_in_bars=1.0)], "snap1", "bars_1m")
    _write(tmp_path, [_row("BTCUSDT", 100, 100)], "snap1", "funding")
    assert "only_in_bars" in read_dataset(tmp_path, "bars_1m").columns
    assert "only_in_bars" not in read_dataset(tmp_path, "funding").columns


def test_the_bound_still_filters_through_a_warm_cache(tmp_path):
    """`not_before_ns` is orthogonal to the cache and has to stay that way."""
    _write(tmp_path, [_row("BTCUSDT", 100, 100), _row("BTCUSDT", 300, 300)], "snap1")
    read_dataset(tmp_path, "bars_1m")
    assert sorted(read_dataset(tmp_path, "bars_1m", not_before_ns=300)[AVAILABILITY_TIME]) \
        == [300]


def test_an_absent_dataset_reads_empty_and_caches_nothing(tmp_path):
    assert read_dataset(tmp_path, "never_written").empty
    assert read_dataset(tmp_path, "never_written").empty


# --- and it must actually save the work -------------------------------------

def test_a_second_read_opens_no_fragment_it_has_already_opened(tmp_path):
    """The measurement the row is accepted on, at test scale."""
    _write(tmp_path, [_row("BTCUSDT", 100, 100)], "snap1")
    _write(tmp_path, [_row("ETHUSDT", 100, 100)], "snap2")

    before = count_fragment_schema_reads()
    read_dataset(tmp_path, "bars_1m")
    after_first = count_fragment_schema_reads()
    read_dataset(tmp_path, "bars_1m")
    after_second = count_fragment_schema_reads()

    assert after_first - before == 2, "the first read pays for every fragment"
    assert after_second == after_first, "the second read pays for none of them"


def test_a_read_after_an_append_opens_only_the_new_fragment(tmp_path):
    """Cost proportional to what arrived, which is what SL-14 is accepted on."""
    _write(tmp_path, [_row("BTCUSDT", 100, 100)], "snap1")
    _write(tmp_path, [_row("ETHUSDT", 100, 100)], "snap2")
    read_dataset(tmp_path, "bars_1m")

    before = count_fragment_schema_reads()
    _write(tmp_path, [_row("SOLUSDT", 200, 200)], "snap3")
    read_dataset(tmp_path, "bars_1m")
    assert count_fragment_schema_reads() - before == 1
