"""Correcting bars that are already written, in a store with no delete path.

The defect being remediated was real: until 2026-08-08 Binance's zero-price
placeholder frames became trades at price 0.0, and 746 of the store's 1,671 bars
carried a non-positive price as a result.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from store.clock_gated_reader import ClockGatedReader
from store.correct_bars import (
    correction_snapshot_id, nonpositive_bars, republish_day,
)
from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

DATASET = "bars_60000000000ns"
BAR_NS = 1785780360_000_000_000


def _write_bar(store_root: Path, snapshot: str, *, low: float, available_at: int):
    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [BAR_NS],
        INGESTION_TIME: [BAR_NS], AVAILABILITY_TIME: [available_at],
        "open": [63884.2], "high": [63897.2], "low": [low], "close": [63871.3],
        "volume": [1.5], "trades": [1781],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(store_root, DATASET, frame, snapshot)


def test_nonpositive_bars_reports_what_a_reader_would_actually_serve(tmp_path):
    """A corrupt row already superseded is not a problem to fix twice."""
    store = tmp_path / "store"
    _write_bar(store, "corrupt", low=0.0, available_at=BAR_NS)
    assert len(nonpositive_bars(store, DATASET)) == 1

    _write_bar(store, "fixed", low=63870.0, available_at=BAR_NS + 1_000)
    assert len(nonpositive_bars(store, DATASET)) == 0


def test_a_correction_needs_a_later_availability_time_to_win(tmp_path):
    """The reader breaks ties by parquet discovery order, so a tie is not a fix.

    `read_as_of` sorts on availability time with a stable mergesort and takes the
    last duplicate, so two rows sharing an availability time are separated only by
    which part file was discovered last - filesystem order. `build_bars_for_day`
    derives availability time from the data, so a plain rebuild recomputes the same
    value and is exactly this non-correction. Asserted so the reason the correction
    restamps that column cannot be optimised away by someone who thinks it is
    redundant.
    """
    store = tmp_path / "store"
    _write_bar(store, "corrupt", low=0.0, available_at=BAR_NS)
    _write_bar(store, "sametime", low=63870.0, available_at=BAR_NS)

    served = ClockGatedReader(store, DATASET).read_as_of(2**62)
    assert len(served) == 1, "one row per (symbol, venue, event_time), corrected or not"
    # Which one wins is not asserted - that is the point. It is not determined by
    # anything the store controls, which is why a tie may not be used as a fix.


def test_a_correction_supersedes_the_bad_bar_without_deleting_it(tmp_path):
    """Both rows survive on disk; the reader serves the later one.

    The old row is not merely tolerated, it is required: a `read_as_of` before the
    correction must still return what the store was believed to hold then, or the
    archive lies about its own past.
    """
    store = tmp_path / "store"
    _write_bar(store, "corrupt", low=0.0, available_at=BAR_NS)
    correction_ns = BAR_NS + 86_400_000_000_000
    _write_bar(store, correction_snapshot_id("binance", "2026-08-03", "zero-price", correction_ns),
               low=63870.0, available_at=correction_ns)

    reader = ClockGatedReader(store, DATASET)

    after = reader.read_as_of(correction_ns)
    assert float(after.iloc[0]["low"]) == 63870.0

    before = reader.read_as_of(correction_ns - 1)
    assert float(before.iloc[0]["low"]) == 0.0, (
        "history must not be rewritten; the corrupt row is what was believed then")

    parts = sorted(p.name for p in (store / DATASET).rglob("*.parquet"))
    assert len(parts) == 2, "append-only: the correction adds a part, never replaces one"


def test_the_correction_id_is_distinct_from_the_build_it_corrects(tmp_path):
    """A rebuild from unchanged raw digests to the same snapshot id.

    `append_partition` refuses an existing part, so a correction reusing that id
    would be rejected as an accidental re-run of the build that produced the defect.
    """
    first = correction_snapshot_id("binance", "2026-08-03", "zero-price", 1_000)
    again = correction_snapshot_id("binance", "2026-08-03", "zero-price", 1_000)
    later = correction_snapshot_id("binance", "2026-08-03", "zero-price", 2_000)

    assert first == again, "the same correction, replayed, must name the same part"
    assert first != later
    assert first.startswith("fix"), "a correction should be identifiable on sight"


def test_a_dry_run_writes_nothing(tmp_path):
    """Appending to an append-only store cannot be undone, so the default is a report."""
    store = tmp_path / "store"
    _write_bar(store, "corrupt", low=0.0, available_at=BAR_NS)
    before = sorted(p.name for p in (store / DATASET).rglob("*.parquet"))

    result = republish_day(tmp_path, store, "binance", "2026-08-03", ["BTCUSDT"],
                           "zero-price", apply=False)

    assert result["written"] == 0
    assert sorted(p.name for p in (store / DATASET).rglob("*.parquet")) == before
