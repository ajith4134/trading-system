"""Putting availability in the PATH, so a poll can skip files without opening them.

Measured 2026-08-17: a scan filtered to match no rows at all cost 185.6s cold on
52,487 fragments, because the partition key was `symbol` and availability was
nowhere in the path - so pyarrow had to open every file to discover that none of
its rows qualified. A filter that cannot be answered from the path is answered
from the file.

Hour rather than date because 538 MB across 52,487 files is ~6,000 new files a
day: a date directory holds a whole day of them by evening, an hour directory
holds ~250 however long capture runs.

The tests below are in two halves, and the first half matters more. Pruning is
worth nothing if it drops a row, and the store's existing promises - append-only,
all-or-nothing, no silently dropped column, no change to what a caller reads -
have to survive a change to where the bytes live.
"""
from __future__ import annotations

import pandas as pd
import pytest

from store.parquet_partition import (
    PartitionExistsError, append_partition, clear_schema_cache, floor_to_hour,
    read_dataset, select_fragments,
)
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

HOUR_NS = 3_600_000_000_000
# 2026-08-17T00:00:00Z, so an offset in hours reads directly as the hour.
MIDNIGHT_NS = 1_786_924_800_000_000_000


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


def _row(symbol: str, available_ns: int, close: float = 1.0, **extra) -> dict:
    return {SYMBOL: symbol, VENUE: "binance", EVENT_TIME: available_ns,
            INGESTION_TIME: available_ns, AVAILABILITY_TIME: available_ns,
            "close": close, **extra}


def _write(store, rows: list[dict], snapshot: str, dataset: str = "bars_1m") -> list:
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    return append_partition(store, dataset, frame, snapshot)


# --- the path carries the hour ----------------------------------------------

def test_a_part_is_written_under_its_availability_hour(tmp_path):
    written = _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + 11 * HOUR_NS)], "snap1")
    relative = written[0].relative_to(tmp_path / "bars_1m")
    assert relative.parts[0] == "availability_hour=2026-08-17T11"
    assert relative.parts[1] == "symbol=BTCUSDT"


def test_one_frame_spanning_two_hours_writes_into_both(tmp_path):
    """Grouping is by (hour, symbol), not by symbol - a frame is not one hour."""
    written = _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + 3 * HOUR_NS),
                                _row("BTCUSDT", MIDNIGHT_NS + 4 * HOUR_NS)], "snap1")
    hours = sorted(p.relative_to(tmp_path / "bars_1m").parts[0] for p in written)
    assert hours == ["availability_hour=2026-08-17T03", "availability_hour=2026-08-17T04"]


def test_the_hour_is_utc_and_floored(tmp_path):
    """Floored, never rounded: rounding up skips the rows of the current hour."""
    assert floor_to_hour(MIDNIGHT_NS + 5 * HOUR_NS + 3_599_999_999_999) \
        == "2026-08-17T05"
    assert floor_to_hour(MIDNIGHT_NS) == "2026-08-17T00"


# --- and none of the store's promises move ----------------------------------

def test_a_caller_sees_no_new_column(tmp_path):
    """The partition key is an index, not data. A reader must not learn about it."""
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS)], "snap1")
    frame = read_dataset(tmp_path, "bars_1m")
    assert "availability_hour" not in frame.columns
    assert SYMBOL in frame.columns and AVAILABILITY_TIME in frame.columns


def test_an_unbounded_read_still_returns_every_row(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + h * HOUR_NS) for h in range(5)],
           "snap1")
    assert len(read_dataset(tmp_path, "bars_1m")) == 5


def test_a_repeated_snapshot_is_still_refused(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS)], "snap1")
    with pytest.raises(PartitionExistsError):
        _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS, close=2.0)], "snap1")


def test_a_collision_in_one_hour_writes_nothing_at_all(tmp_path):
    """All-or-nothing across hours as well as symbols, or a refused write leaves
    a partial snapshot behind - the one state this store promises never to hold."""
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + 2 * HOUR_NS)], "snap1")
    before = sorted(p.name for p in (tmp_path / "bars_1m").rglob("*.parquet"))
    with pytest.raises(PartitionExistsError):
        _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + 9 * HOUR_NS),
                          _row("BTCUSDT", MIDNIGHT_NS + 2 * HOUR_NS)], "snap1")
    assert sorted(p.name for p in (tmp_path / "bars_1m").rglob("*.parquet")) == before


def test_a_column_added_by_a_later_hour_is_not_dropped(tmp_path):
    """The `funding_interval_hours` regression, now across hour directories."""
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS)], "snap1")
    _write(tmp_path, [_row("BYBITBTC", MIDNIGHT_NS + HOUR_NS,
                           funding_interval_hours=4.0)], "snap2")
    assert "funding_interval_hours" in read_dataset(tmp_path, "bars_1m").columns


# --- the bound is exact, and it prunes --------------------------------------

def test_the_bound_returns_exactly_the_rows_at_or_after_it(tmp_path):
    _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + h * HOUR_NS) for h in range(4)],
           "snap1")
    bound = MIDNIGHT_NS + 2 * HOUR_NS
    frame = read_dataset(tmp_path, "bars_1m", not_before_ns=bound)
    assert sorted(frame[AVAILABILITY_TIME]) == [bound, MIDNIGHT_NS + 3 * HOUR_NS]


def test_a_bound_mid_hour_keeps_the_rest_of_that_hour(tmp_path):
    """The floor is what makes this right. An hour-rounded bound would drop the
    row at +30m permanently: no later watermark ever reaches back for it."""
    early = MIDNIGHT_NS + 6 * HOUR_NS
    late = early + 1_800_000_000_000
    _write(tmp_path, [_row("BTCUSDT", early), _row("BTCUSDT", late)], "snap1")
    frame = read_dataset(tmp_path, "bars_1m", not_before_ns=early + 1)
    assert sorted(frame[AVAILABILITY_TIME]) == [late]


def test_a_correction_arriving_later_still_reaches_a_poll(tmp_path):
    """Why the key is availability and never event time: a correction to an old
    bar carries a LATER availability, so it lands in a LATER hour directory."""
    old_event = MIDNIGHT_NS
    _write(tmp_path, [_row("BTCUSDT", old_event, close=1.0)], "snap1")
    correction = _row("BTCUSDT", MIDNIGHT_NS + 8 * HOUR_NS, close=2.0)
    correction[EVENT_TIME] = old_event
    _write(tmp_path, [correction], "snap2")

    frame = read_dataset(tmp_path, "bars_1m", not_before_ns=MIDNIGHT_NS + 7 * HOUR_NS)
    assert list(frame["close"]) == [2.0]


def test_a_bound_opens_only_the_hours_at_or_after_it(tmp_path):
    """The measurement SL-15 exists for: files skipped without being opened."""
    for hour in range(6):
        _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + hour * HOUR_NS)],
               f"snap{hour}")
    assert len(select_fragments(tmp_path, "bars_1m", None)) == 6
    selected = select_fragments(tmp_path, "bars_1m", MIDNIGHT_NS + 4 * HOUR_NS)
    assert len(selected) == 2
    assert all("2026-08-17T0" in path for path in selected)
    assert all("T00" not in path and "T03" not in path for path in selected)


def test_pruning_does_not_depend_on_how_many_hours_are_behind_it(tmp_path):
    """The bound this row is accepted on: flat, not proportional to the archive."""
    for hour in range(24):
        _write(tmp_path, [_row("BTCUSDT", MIDNIGHT_NS + hour * HOUR_NS)],
               f"snap{hour}")
    newest = MIDNIGHT_NS + 23 * HOUR_NS
    assert len(select_fragments(tmp_path, "bars_1m", newest)) == 1


def test_an_absent_dataset_selects_nothing(tmp_path):
    assert select_fragments(tmp_path, "never_written", None) == []


# --- SL-17 / RL-032: the migration converts the layout, not just the path ---

def test_migrating_a_whole_universe_dataset_produces_the_hour_only_layout(tmp_path):
    """The building copy starts empty, so `partitions_by_hour_alone` sees a
    dataset with nothing to mix with and writes the converted layout. The swap
    then puts it in place and the live writer follows it - which is why the
    conversion and the writer change cannot be separated: a dataset holding both
    layouts raises `ArrowTypeError` on every read.
    """
    import pandas as pd
    from store.hourly_migration import migrate_dataset_to_hourly
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    base = 1_787_000_000_000_000_000
    for i, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        frame = pd.DataFrame({
            SYMBOL: [symbol], VENUE: ["binance"],
            EVENT_TIME: pd.array([base + i], dtype="int64"),
            INGESTION_TIME: pd.array([base + i], dtype="int64"),
            AVAILABILITY_TIME: pd.array([base + i], dtype="int64"),
            "funding_rate": [0.0001 * (i + 1)],
        })
        # Written the way the live store holds funding today: symbol at the top.
        legacy = tmp_path / "funding" / f"symbol={symbol}"
        legacy.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(frame.drop(columns=[SYMBOL]),
                                            preserve_index=False),
                       legacy / f"part-funding-binance-{i}.parquet")

    report = migrate_dataset_to_hourly(tmp_path, "funding")

    built = tmp_path / "funding.hourly-building"
    parts = sorted(built.rglob("*.parquet"))
    assert parts, "the migration wrote nothing"
    assert not any("symbol=" in str(p) for p in parts), (
        "a whole-universe dataset must migrate into the hour-only layout")
    assert report.legacy_rows == 3


def _legacy_funding_part(root, symbol, *, hour_ns, venue="binance", name="0",
                         under_hour=None):
    """One legacy part, in whichever sub-layout the test needs."""
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)
    frame = pd.DataFrame({
        SYMBOL: [symbol], VENUE: [venue],
        EVENT_TIME: pd.array([hour_ns], dtype="int64"),
        INGESTION_TIME: pd.array([hour_ns], dtype="int64"),
        AVAILABILITY_TIME: pd.array([hour_ns], dtype="int64"),
        "funding_rate": [0.0001],
    })
    folder = (root / "funding" / f"symbol={symbol}" if under_hour is None
              else root / "funding" / f"availability_hour={under_hour}" / f"symbol={symbol}")
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame.drop(columns=[SYMBOL]), preserve_index=False),
        folder / f"part-funding-{venue}-{name}.parquet")


def test_one_hour_of_a_whole_universe_dataset_migrates_to_one_part(tmp_path):
    """**The fan-out is the `symbol=` level, so removing it must remove the fan-out.**
    Migrating symbol-folder-by-symbol-folder would write one part per symbol per
    hour and leave the file count exactly where it started - 1,271 parts an hour
    for funding, which is the whole defect."""
    from store.hourly_migration import migrate_dataset_to_hourly

    base = 1_787_000_000_000_000_000
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        _legacy_funding_part(tmp_path, symbol, hour_ns=base)

    report = migrate_dataset_to_hourly(tmp_path, "funding")

    parts = sorted((tmp_path / "funding.hourly-building").rglob("*.parquet"))
    assert len(parts) == 1, [str(p) for p in parts]
    assert report.verified, report.mismatched_groups
    assert report.legacy_rows == 4 and report.migrated_rows == 4


def test_two_hours_migrate_to_one_part_each(tmp_path):
    from store.hourly_migration import migrate_dataset_to_hourly
    base = 1_787_000_000_000_000_000
    for symbol in ("BTCUSDT", "ETHUSDT"):
        _legacy_funding_part(tmp_path, symbol, hour_ns=base, name="a")
        _legacy_funding_part(tmp_path, symbol, hour_ns=base + 3_600 * 10**9,
                             name="b")

    report = migrate_dataset_to_hourly(tmp_path, "funding")

    parts = sorted((tmp_path / "funding.hourly-building").rglob("*.parquet"))
    assert len(parts) == 2, [str(p) for p in parts]
    assert len({p.parent for p in parts}) == 2
    assert report.verified and report.migrated_rows == 4


def test_parts_already_in_the_hour_layout_are_rewritten_not_linked(tmp_path):
    """They sit at `availability_hour=<H>/symbol=<S>/`, which is still the
    per-symbol sub-layout. Hard-linking them into the building copy unchanged
    would put a `symbol=` directory in the converted dataset - the mixed state
    that raises ArrowTypeError on every read."""
    from store.hourly_migration import migrate_dataset_to_hourly
    base = 1_787_000_000_000_000_000
    _legacy_funding_part(tmp_path, "BTCUSDT", hour_ns=base)
    _legacy_funding_part(tmp_path, "ETHUSDT", hour_ns=base,
                         under_hour="2026-08-17T20", name="h")

    report = migrate_dataset_to_hourly(tmp_path, "funding")

    built = tmp_path / "funding.hourly-building"
    assert not any("symbol=" in str(p) for p in built.rglob("*.parquet"))
    assert report.verified, report.mismatched_groups
    assert report.migrated_rows == 2


def test_the_converted_dataset_reads_back_every_symbol(tmp_path):
    """Verification is on content, and the column that matters here is the one
    the path stopped carrying."""
    from store.hourly_migration import migrate_dataset_to_hourly
    from store.parquet_partition import read_dataset
    from store.temporal_schema import SYMBOL
    base = 1_787_000_000_000_000_000
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        _legacy_funding_part(tmp_path, symbol, hour_ns=base)

    migrate_dataset_to_hourly(tmp_path, "funding")
    frame = read_dataset(tmp_path, "funding.hourly-building")

    assert sorted(frame[SYMBOL]) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_the_buffer_is_bounded_so_a_large_dataset_cannot_exhaust_memory(tmp_path,
                                                                       monkeypatch):
    """**Buffering every hour before the first write makes peak memory the whole
    dataset**, on a box with 29 GB, no swap, and a documented history of OOM
    kills. So the buffer has a row cap and flushes the largest hour when it is
    reached. More than one part in an hour is a fine outcome - it is still one
    part per flush rather than one per symbol, which is the fan-out being
    removed - and a dataset small enough to fit under the cap gets exactly one.
    """
    from store import hourly_migration
    from store.hourly_migration import migrate_dataset_to_hourly

    monkeypatch.setattr(hourly_migration, "MAX_BUFFERED_ROWS", 2)

    base = 1_787_000_000_000_000_000
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"):
        _legacy_funding_part(tmp_path, symbol, hour_ns=base)

    report = migrate_dataset_to_hourly(tmp_path, "funding")

    parts = sorted((tmp_path / "funding.hourly-building").rglob("*.parquet"))
    assert len(parts) > 1, "the cap must have forced a flush"
    assert not any("symbol=" in str(p) for p in parts)
    assert report.verified, report.mismatched_groups
    assert report.legacy_rows == 5 and report.migrated_rows == 5
