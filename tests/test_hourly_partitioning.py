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
