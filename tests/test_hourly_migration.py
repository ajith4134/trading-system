"""Moving 52,487 files into a layout that can be pruned, and proving nothing fell out.

The migration is the dangerous half of SL-15. Pruning by hour is worth nothing if
the rewrite that produced the hours dropped a row, a column or a symbol - and the
store has already lost a column silently once, on 2026-08-09, when
`funding_interval_hours` vanished from every read because pyarrow inferred a
schema from one fragment.

So the tests here are about what must still be true afterwards, and the migration
is not allowed to swap itself in until it can show it: same rows, same columns,
same symbols, and the old directory still on disk. The one test that would matter
most in an incident is `test_the_swap_is_refused_when_verification_failed` - the
failing path, exercised, because a migration that can only be observed succeeding
has never been observed at all.
"""
from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from store.hourly_migration import (
    LEGACY_SUFFIX, MigrationRefused, _read_consumed, migrate_dataset_to_hourly,
    swap_in_migrated_dataset,
)
from store.parquet_partition import (
    append_partition, clear_schema_cache, read_dataset, select_fragments,
)
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

DATASET = "bars_1m"
HOUR_NS = 3_600_000_000_000
MIDNIGHT_NS = 1_786_924_800_000_000_000


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


def _legacy_part(store, symbol: str, rows: list[dict], snapshot: str) -> None:
    """Write a part in the pre-SL-15 layout: symbol at the top, no hour anywhere."""
    folder = store / DATASET / f"symbol={symbol}"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).drop(columns=[SYMBOL]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False),
                   folder / f"part-{snapshot}.parquet", compression="zstd")


def _row(symbol: str, available_ns: int, close: float = 1.0, **extra) -> dict:
    return {SYMBOL: symbol, VENUE: "binance", EVENT_TIME: available_ns,
            INGESTION_TIME: available_ns, AVAILABILITY_TIME: available_ns,
            "close": close, **extra}


def _stocked(store):
    """Two symbols, three hours, several snapshots - the shape the live store has."""
    for hour in range(3):
        _legacy_part(store, "BTCUSDT",
                     [_row("BTCUSDT", MIDNIGHT_NS + hour * HOUR_NS, close=float(hour))],
                     f"snap{hour}")
    _legacy_part(store, "BTCUSDT",
                 [_row("BTCUSDT", MIDNIGHT_NS + 30_000_000_000, close=9.0)], "snapx")
    _legacy_part(store, "ETHUSDT",
                 [_row("ETHUSDT", MIDNIGHT_NS + HOUR_NS, close=5.0)], "snap1")
    return store


# --- the old layout has to keep working until its migration runs -------------

def test_a_bounded_read_of_an_unmigrated_dataset_still_works(tmp_path):
    """Every dataset is in the old layout until its migration has run, and the
    paper engine polls with a bound every 60 seconds. A filter naming
    `availability_hour` against a dataset that has no such field raises
    ArrowInvalid and takes the whole read down - so this change would have
    stopped the engine the moment it landed, before any migration."""
    _stocked(tmp_path)
    frame = read_dataset(tmp_path, DATASET, not_before_ns=MIDNIGHT_NS + HOUR_NS)
    assert sorted(frame[AVAILABILITY_TIME].unique()) == [MIDNIGHT_NS + HOUR_NS,
                                                         MIDNIGHT_NS + 2 * HOUR_NS]


def test_a_half_migrated_dataset_does_not_lose_its_legacy_rows(tmp_path):
    """The live failure, 2026-08-17. The store supervisors reload their child on
    restart, so `append_partition` began writing hour-partitioned parts into the
    LIVE bars dataset within minutes of the code landing: 4,595 new parts beside
    2,231 legacy symbol directories.

    Hive partitioning gives those legacy parts a NULL hour. `NULL >= '2026-08-17T12'`
    is null rather than true, so an hour-bounded read drops every legacy row -
    no error, no warning, a smaller answer. Whoever read it next would have been
    told the archive was mostly empty.
    """
    _stocked(tmp_path)
    late = _row("BTCUSDT", MIDNIGHT_NS + 4 * HOUR_NS, close=11.0)
    frame = pd.DataFrame([late]).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                                         AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, DATASET, frame, "snap-hourly")
    assert (tmp_path / DATASET / "symbol=BTCUSDT").is_dir(), "legacy layout present"
    assert any((tmp_path / DATASET).glob("availability_hour=*")), "hour layout present"

    everything = read_dataset(tmp_path, DATASET)
    bounded = read_dataset(tmp_path, DATASET, not_before_ns=MIDNIGHT_NS)
    assert len(bounded) == len(everything) == 6
    assert 9.0 in set(bounded["close"]), "a legacy row vanished from a bounded read"


def test_fragments_of_an_unmigrated_dataset_can_still_be_selected(tmp_path):
    """No pruning is possible without the hour in the path - the honest answer
    is every fragment, not a crash and not a wrong subset."""
    _stocked(tmp_path)
    assert len(select_fragments(tmp_path, DATASET, MIDNIGHT_NS + 2 * HOUR_NS)) == 5


# --- nothing may be lost ----------------------------------------------------

def test_every_legacy_row_survives_the_migration(tmp_path):
    _stocked(tmp_path)
    legacy = read_dataset(tmp_path, DATASET).sort_values(
        [SYMBOL, AVAILABILITY_TIME, "close"]).reset_index(drop=True)
    clear_schema_cache()

    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    migrated = read_dataset(tmp_path, report.building_dataset).sort_values(
        [SYMBOL, AVAILABILITY_TIME, "close"]).reset_index(drop=True)

    assert report.verified
    assert report.legacy_rows == report.migrated_rows == len(legacy)
    pd.testing.assert_frame_equal(legacy[sorted(legacy.columns)],
                                  migrated[sorted(migrated.columns)])


def test_a_column_only_one_symbol_carries_is_not_dropped(tmp_path):
    """The 2026-08-09 failure, asked of the migration instead of the reader."""
    _legacy_part(tmp_path, "BTCUSDT", [_row("BTCUSDT", MIDNIGHT_NS)], "snap1")
    _legacy_part(tmp_path, "BYBITBTC",
                 [_row("BYBITBTC", MIDNIGHT_NS, funding_interval_hours=4.0)], "snap1")

    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    migrated = read_dataset(tmp_path, report.building_dataset)
    assert report.columns_lost == ()
    assert "funding_interval_hours" in migrated.columns
    assert migrated.loc[migrated[SYMBOL] == "BYBITBTC",
                        "funding_interval_hours"].iloc[0] == 4.0


def test_the_legacy_directory_is_left_exactly_as_it_was(tmp_path):
    _stocked(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / DATASET).rglob("*"))
    migrate_dataset_to_hourly(tmp_path, DATASET)
    after = sorted(str(p.relative_to(tmp_path)) for p in (tmp_path / DATASET).rglob("*"))
    assert before == after


# --- and the point of it all ------------------------------------------------

def test_the_migrated_dataset_prunes_by_hour(tmp_path):
    _stocked(tmp_path)
    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    every = select_fragments(tmp_path, report.building_dataset, None)
    recent = select_fragments(tmp_path, report.building_dataset,
                              MIDNIGHT_NS + 2 * HOUR_NS)
    assert len(recent) == 1 < len(every)


def test_a_sealed_hour_is_compacted_to_one_part_per_symbol(tmp_path):
    """Four legacy parts for BTCUSDT become three - one per hour it wrote in."""
    _stocked(tmp_path)
    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    built = tmp_path / report.building_dataset
    assert len(list(built.glob("*/symbol=BTCUSDT/*.parquet"))) == 3
    assert report.legacy_parts == 5 and report.migrated_parts == 4


# --- and it can be interrupted ----------------------------------------------

def test_a_second_pass_migrates_only_what_arrived_since_the_first(tmp_path):
    """Capture keeps writing while the migration runs, and this is the bug that
    makes that dangerous: without a manifest of consumed parts, the second pass
    re-reads every part for the symbol, writes them all under a new
    content-derived snapshot id, and the store holds those rows TWICE. Duplicated
    bars read as real volume - nothing raises, and every number downstream is
    wrong."""
    _stocked(tmp_path)
    first = migrate_dataset_to_hourly(tmp_path, DATASET)
    assert first.verified

    _legacy_part(tmp_path, "BTCUSDT",
                 [_row("BTCUSDT", MIDNIGHT_NS + 5 * HOUR_NS, close=7.0)], "snaplate")
    clear_schema_cache()

    second = migrate_dataset_to_hourly(tmp_path, DATASET)
    assert second.verified, second.mismatched_groups
    assert second.legacy_rows == first.legacy_rows + 1
    assert second.migrated_rows == second.legacy_rows

    migrated = read_dataset(tmp_path, second.building_dataset)
    assert len(migrated[migrated["close"] == 0.0]) == 1, "an old row was migrated twice"
    assert len(migrated[migrated["close"] == 7.0]) == 1, "the new row is missing"


def test_a_manifest_written_by_the_first_version_is_still_read(tmp_path):
    """The manifest was a JSON array before it became one path per line, and one
    such file exists from the live migration that was mid-flight when the format
    changed. Failing to read it would mean re-migrating 62,000 parts that are
    already on disk and correct."""
    _stocked(tmp_path)
    first = migrate_dataset_to_hourly(tmp_path, DATASET)
    manifest = tmp_path / first.building_dataset / ".migrated-parts.json"
    entries = sorted(manifest.read_text().split())
    manifest.write_text(json.dumps(entries, indent=0) + "\n")

    second = migrate_dataset_to_hourly(tmp_path, DATASET)
    assert second.verified and second.migrated_parts == 0, \
        "a legacy-format manifest was ignored and the parts were migrated again"


def test_a_torn_final_line_is_dropped_rather_than_trusted(tmp_path):
    """A crash mid-append leaves half a path. Re-migrating that part is
    wasteful; trusting the fragment would skip a part nothing migrated."""
    _stocked(tmp_path)
    first = migrate_dataset_to_hourly(tmp_path, DATASET)
    manifest = tmp_path / first.building_dataset / ".migrated-parts.json"
    entries = manifest.read_text().split()
    manifest.write_text("\n".join(entries[:-1]) + "\n" + entries[-1][:12])

    assert entries[-1] not in _read_consumed(tmp_path, first.building_dataset)
    assert set(entries[:-1]) <= _read_consumed(tmp_path, first.building_dataset)


def test_running_it_twice_changes_nothing(tmp_path):
    """A migration over 52,487 files will be interrupted. Resuming is the design."""
    _stocked(tmp_path)
    first = migrate_dataset_to_hourly(tmp_path, DATASET)
    built = tmp_path / first.building_dataset
    fingerprint = sorted(p.name for p in built.rglob("*.parquet"))

    second = migrate_dataset_to_hourly(tmp_path, DATASET)
    assert sorted(p.name for p in built.rglob("*.parquet")) == fingerprint
    assert second.migrated_rows == first.migrated_rows
    assert second.verified


# --- the swap, and its refusal ----------------------------------------------

def test_the_swap_moves_the_old_layout_aside_and_keeps_it(tmp_path):
    _stocked(tmp_path)
    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    rows_before = len(read_dataset(tmp_path, report.building_dataset))
    clear_schema_cache()

    retired, live = swap_in_migrated_dataset(tmp_path, DATASET, report)

    assert retired.name == f"{DATASET}{LEGACY_SUFFIX}" and retired.is_dir()
    assert live == tmp_path / DATASET
    assert not (tmp_path / report.building_dataset).exists()
    assert len(read_dataset(tmp_path, DATASET)) == rows_before
    assert any((tmp_path / DATASET).glob("availability_hour=*"))


def test_the_swap_is_refused_when_verification_failed(tmp_path):
    """The failing path, exercised. A migration that can only be watched
    succeeding has never been watched at all."""
    _stocked(tmp_path)
    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    unverified = report.__class__(**{**report.__dict__, "verified": False,
                                     "columns_lost": ("close",)})
    with pytest.raises(MigrationRefused, match="close"):
        swap_in_migrated_dataset(tmp_path, DATASET, unverified)
    assert (tmp_path / DATASET / "symbol=BTCUSDT").is_dir(), "the store was left alone"


def test_a_mixed_dataset_migrates_both_layouts_into_one(tmp_path):
    """The state the live store was actually in. Parts the writers had already
    put in the new layout must end up in the swapped-in store too - refusing the
    dataset would refuse every dataset that matters, and ignoring those parts
    would leave their rows out of the store that replaces it."""
    _stocked(tmp_path)
    late = _row("BTCUSDT", MIDNIGHT_NS + 4 * HOUR_NS, close=11.0)
    frame = pd.DataFrame([late]).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                                         AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path, DATASET, frame, "snap-hourly")

    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    assert report.verified, report.mismatched_groups
    assert report.legacy_rows == 6 and report.migrated_rows == 6

    migrated = read_dataset(tmp_path, report.building_dataset)
    assert set(migrated["close"]) == {0.0, 1.0, 2.0, 9.0, 5.0, 11.0}
    assert not any((tmp_path / report.building_dataset).glob("symbol=*")), \
        "the built copy must be purely hour-partitioned, or pruning stays off"


def test_an_already_hourly_dataset_is_refused_rather_than_migrated_twice(tmp_path):
    _stocked(tmp_path)
    report = migrate_dataset_to_hourly(tmp_path, DATASET)
    swap_in_migrated_dataset(tmp_path, DATASET, report)
    with pytest.raises(MigrationRefused, match="already"):
        migrate_dataset_to_hourly(tmp_path, DATASET)


def test_an_absent_dataset_is_refused(tmp_path):
    with pytest.raises(MigrationRefused, match="no dataset"):
        migrate_dataset_to_hourly(tmp_path, "never_written")
