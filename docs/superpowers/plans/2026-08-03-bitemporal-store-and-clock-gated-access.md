# Bitemporal Store and Clock-Gated Access — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn captured raw frames into a bitemporal store that can only be read through a clock-gated API, so look-ahead leakage is prevented structurally rather than by discipline.

**Architecture:** Every stored row carries three timestamps — `event_time_ns` (when the venue says it happened), `ingestion_time_ns` (when this machine received it), and `availability_time_ns` (the earliest moment a decision could have used it). The store is append-only: a correction is a new row with a later `availability_time_ns`, never an overwrite. One reader serves both backtest and live by filtering `availability_time_ns <= sim_clock_ns`, and all joins key on `availability_time_ns`. Storage is Parquet + ZSTD partitioned by symbol.

**Tech Stack:** Python 3.12, pandas (`merge_asof`), pyarrow (Parquet + dataset filtering), zstandard (already present via Layer 0), pytest.

## Global Constraints

- **Python `>=3.12,<3.13`** — matches `pyproject.toml`; do not widen it.
- **New package `src/store`** must be added to `[tool.hatch.build.targets.wheel] packages`, which currently reads `["src/capture", "src/statuswall"]`.
- **Do not adopt Feast or any feature-store library.** `ARCHITECTURE.md` rejects it explicitly: its leakage protection is off by default (`filter_by_created_timestamp: bool = False`), it has no native streaming, and it solves a multi-consumer coordination problem a solo operator does not have.
- **Append-only.** No code path may overwrite or delete a Parquet part. Corrections are new rows.
- **`availability_time_ns` is never null** and is the only column any read path may filter on.
- **`event_time_ns` may exceed `ingestion_time_ns`.** Venue clocks drift; that is recorded, never corrected. Only `availability_time_ns >= ingestion_time_ns` is enforced.
- **Timestamps are int64 nanoseconds UTC** everywhere. Never float, never naive datetime.
- Docstrings explain *why*, matching `src/capture` — the existing code documents the failure each decision prevents, and this package must read the same way.
- Tests are named for the behaviour they defend, not the function they call.

## Measured Facts This Plan Depends On

Taken from the real archive on 2026-08-02, not assumed:

| Venue | Frame shape | Lag p50 | Lag max | Notes |
|---|---|---|---|---|
| `binance` / `trade` | one trade per frame, `{"stream":…,"data":{…}}` | 70 ms | 205 ms | 0 of 111,980 frames over 1 s |
| `hyperliquid` / `trades` | **array** of trades, `{"channel":…,"data":[{…},{…}]}` | 277 ms | **33,823 ms** | 4 of 6,548 frames over 10 s; intra-frame event span up to 32,412 ms |

Hyperliquid's reconnect backfill is why corrections are an exercised path in Task 4 rather than a hypothetical one.

## File Structure

| File | Responsibility |
|---|---|
| `src/store/temporal_schema.py` | The three-timestamp contract and its invariants. No I/O. |
| `src/store/parquet_partition.py` | Append-only Parquet parts, symbol-partitioned, plus snapshot identity. |
| `src/store/trade_bars.py` | Raw frames → trades → bars carrying a correct `availability_time_ns`. |
| `src/store/clock_gated_reader.py` | The only read path. Filters by clock, resolves corrections, joins as-of. |
| `src/store/cli.py` | `build` command: raw archive → store. |
| `tests/test_temporal_schema.py` | Invariants hold and violations are refused. |
| `tests/test_parquet_partition.py` | Append-only is enforced; snapshot ids are content-derived. |
| `tests/test_trade_bars.py` | Both venue formats; late data sets availability correctly. |
| `tests/test_clock_gated_reader.py` | Clock filtering and correction resolution. |
| `tests/test_store_leakage.py` | Adversarial: synthetic leakage cases that must fail to leak. |

---

### Task 1: The three-timestamp contract

**Files:**
- Create: `src/store/__init__.py` (empty)
- Create: `src/store/temporal_schema.py`
- Create: `tests/test_temporal_schema.py`
- Modify: `pyproject.toml` — add `"src/store"` to wheel packages, add `pandas` and `pyarrow` dependencies

**Interfaces:**
- Produces: `EVENT_TIME`, `INGESTION_TIME`, `AVAILABILITY_TIME`, `SYMBOL`, `VENUE` (str constants); `REQUIRED_COLUMNS: tuple[str, ...]`; `TemporalInvariantError(ValueError)`; `validate_temporal_frame(frame: pd.DataFrame) -> None`

- [ ] **Step 1: Add dependencies**

```bash
cd ~/trading-system
uv add pandas pyarrow
```

- [ ] **Step 2: Register the package**

In `pyproject.toml`, change:

```toml
packages = ["src/capture", "src/statuswall"]
```

to:

```toml
packages = ["src/capture", "src/statuswall", "src/store"]
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_temporal_schema.py
"""The schema exists to make one class of bug impossible, so these tests try to commit it."""
from __future__ import annotations

import pandas as pd
import pytest

from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, REQUIRED_COLUMNS, SYMBOL, VENUE,
    TemporalInvariantError, validate_temporal_frame,
)


def _frame(**overrides) -> pd.DataFrame:
    base = {
        SYMBOL: ["BTCUSDT"],
        VENUE: ["binance"],
        EVENT_TIME: [1_000],
        INGESTION_TIME: [1_100],
        AVAILABILITY_TIME: [1_100],
    }
    base.update(overrides)
    return pd.DataFrame(base).astype({c: "int64" for c in base if c.endswith("_ns")})


def test_a_well_formed_frame_passes():
    validate_temporal_frame(_frame())


def test_missing_availability_time_is_refused():
    frame = _frame().drop(columns=[AVAILABILITY_TIME])
    with pytest.raises(TemporalInvariantError, match=AVAILABILITY_TIME):
        validate_temporal_frame(frame)


def test_null_availability_time_is_refused():
    """A row with no availability time cannot be clock-gated, so it cannot be stored."""
    frame = _frame()
    frame[AVAILABILITY_TIME] = pd.Series([pd.NA], dtype="Int64")
    with pytest.raises(TemporalInvariantError, match="null"):
        validate_temporal_frame(frame)


def test_availability_earlier_than_ingestion_is_refused():
    """Claiming data was usable before it arrived is the leak this whole layer prevents."""
    with pytest.raises(TemporalInvariantError, match="before"):
        validate_temporal_frame(_frame(**{AVAILABILITY_TIME: [900], INGESTION_TIME: [1_100]}))


def test_event_time_after_ingestion_is_allowed():
    """Venue clocks drift ahead. That is recorded, not corrected.

    Enforcing event <= ingestion would silently rewrite the venue's own account of
    when a trade happened, which is data loss disguised as validation.
    """
    validate_temporal_frame(_frame(**{EVENT_TIME: [2_000], INGESTION_TIME: [1_100]}))


def test_null_event_time_is_allowed():
    """Some frames carry no venue timestamp; absence is recorded, not invented."""
    frame = _frame()
    frame[EVENT_TIME] = pd.Series([pd.NA], dtype="Int64")
    validate_temporal_frame(frame)


def test_float_timestamps_are_refused():
    """Float nanoseconds lose precision above 2^53 and silently reorder events."""
    frame = _frame()
    frame[AVAILABILITY_TIME] = frame[AVAILABILITY_TIME].astype("float64")
    with pytest.raises(TemporalInvariantError, match="int64"):
        validate_temporal_frame(frame)


def test_required_columns_are_the_documented_five():
    assert set(REQUIRED_COLUMNS) == {SYMBOL, VENUE, EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_temporal_schema.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'store'`

- [ ] **Step 5: Write the implementation**

```python
# src/store/temporal_schema.py
"""The three timestamps every stored row carries, and what must be true of them.

`event_time_ns`  - when the venue says it happened.
`ingestion_time_ns` - when this machine received it.
`availability_time_ns` - the earliest moment a decision could have used it.

The third is the only one any read path may filter on, and it is the whole point
of the layer. Event time is what a naive backtest joins on, and joining on it
means a bar built from a frame that arrived 32 seconds late appears to have been
available 32 seconds before it existed. That produces an excellent backtest and a
broken system, and nothing in the result looks wrong.

Only `availability >= ingestion` is enforced. Event time is deliberately
unconstrained: venue clocks drift ahead of ours, and rejecting or clamping that
would rewrite the venue's own account of when a trade happened.
"""
from __future__ import annotations

import pandas as pd

SYMBOL = "symbol"
VENUE = "venue"
EVENT_TIME = "event_time_ns"
INGESTION_TIME = "ingestion_time_ns"
AVAILABILITY_TIME = "availability_time_ns"

REQUIRED_COLUMNS = (SYMBOL, VENUE, EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME)

# event_time may be absent when a venue sends no timestamp, so it is the one
# nullable member of the trio.
_NON_NULLABLE = (INGESTION_TIME, AVAILABILITY_TIME)
_INTEGER_DTYPES = {"int64", "Int64"}


class TemporalInvariantError(ValueError):
    """A frame violates the contract that makes clock-gating trustworthy."""


def validate_temporal_frame(frame: pd.DataFrame) -> None:
    """Raise unless every row can be safely clock-gated."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TemporalInvariantError(f"missing required column(s): {missing}")

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME):
        if str(frame[column].dtype) not in _INTEGER_DTYPES:
            raise TemporalInvariantError(
                f"{column} must be int64 nanoseconds, got {frame[column].dtype}. "
                f"Float nanoseconds lose precision above 2^53 and silently reorder events")

    for column in _NON_NULLABLE:
        if frame[column].isna().any():
            raise TemporalInvariantError(f"{column} contains null values")

    if frame.empty:
        return

    too_early = frame[AVAILABILITY_TIME] < frame[INGESTION_TIME]
    if bool(too_early.any()):
        first = frame.loc[too_early].iloc[0]
        raise TemporalInvariantError(
            f"{AVAILABILITY_TIME} is before {INGESTION_TIME} on {int(too_early.sum())} row(s) "
            f"(first: available {first[AVAILABILITY_TIME]}, ingested {first[INGESTION_TIME]}) - "
            f"a row cannot have been usable before it arrived")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_temporal_schema.py -q`
Expected: PASS, 8 tests

- [ ] **Step 7: Commit**

```bash
cd ~/trading-system
git add pyproject.toml uv.lock src/store/ tests/test_temporal_schema.py
git commit -m "feat: make a row that cannot be clock-gated impossible to store

Three timestamps per row, and only availability time may be filtered on. Event
time is left unconstrained on purpose: venue clocks drift ahead of ours, and
clamping that would rewrite the venue's own account of when a trade happened."
```

---

### Task 2: Append-only Parquet partitions with content-derived snapshot identity

**Files:**
- Create: `src/store/parquet_partition.py`
- Create: `tests/test_parquet_partition.py`

**Interfaces:**
- Consumes: `store.temporal_schema.validate_temporal_frame`, `REQUIRED_COLUMNS`, `SYMBOL`
- Produces: `compute_snapshot_id(paths: Sequence[Path]) -> str`; `append_partition(store_root: Path, dataset: str, frame: pd.DataFrame, snapshot_id: str) -> list[Path]`; `read_dataset(store_root: Path, dataset: str) -> pd.DataFrame`; `PartitionExistsError(FileExistsError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parquet_partition.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_parquet_partition.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'store.parquet_partition'`

- [ ] **Step 3: Write the implementation**

```python
# src/store/parquet_partition.py
"""Append-only Parquet parts, partitioned by symbol, identified by content.

Corrections are new parts. Nothing here opens an existing file for writing, and
`append_partition` refuses a snapshot id that already exists rather than
overwriting it - a rebuild that silently replaced a part would erase exactly the
history the store promises to keep, and the result would look like a clean run.

ZSTD because the archive it derives from is already ZSTD and the ratio on
columnar float data is worth more than the CPU at this volume.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from store.temporal_schema import SYMBOL, validate_temporal_frame

_CHUNK = 1 << 20


class PartitionExistsError(FileExistsError):
    """A part with this snapshot id is already on disk."""


def compute_snapshot_id(paths: Sequence[Path]) -> str:
    """A short digest over the *contents* of the inputs, order-independent.

    Content rather than filenames: the id answers "which data produced this
    result", and a rename must not read as different data. Sorted per-file
    digests rather than a running hash so the caller's iteration order cannot
    change the answer.
    """
    digests = sorted(_digest_file(Path(path)) for path in paths)
    combined = hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()
    return combined[:16]


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def append_partition(store_root: Path, dataset: str, frame: pd.DataFrame,
                     snapshot_id: str) -> list[Path]:
    """Write one part per symbol. Never touches an existing file."""
    # Validated before any directory is created so a refused write leaves no trace
    # and cannot be mistaken for a partial success.
    validate_temporal_frame(frame)
    if frame.empty:
        return []

    written: list[Path] = []
    for symbol, group in frame.groupby(SYMBOL, sort=True):
        folder = Path(store_root) / dataset / f"symbol={symbol}"
        target = folder / f"part-{snapshot_id}.parquet"
        if target.exists():
            raise PartitionExistsError(
                f"{target} already exists; snapshot '{snapshot_id}' has been written for "
                f"{symbol}. Corrections are new snapshots, never rewrites")
        folder.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(group.reset_index(drop=True), preserve_index=False)
        pq.write_table(table, target, compression="zstd")
        written.append(target)
    return written


def read_dataset(store_root: Path, dataset: str) -> pd.DataFrame:
    """Every part, concatenated. Absent datasets read empty, not as an error.

    An empty store is the correct state before the first build, and raising there
    would make "nothing captured yet" indistinguishable from a bug.
    """
    root = Path(store_root) / dataset
    if not root.is_dir():
        return pd.DataFrame()
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive")
    return dataset_handle.to_table().to_pandas()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_parquet_partition.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
cd ~/trading-system
git add src/store/parquet_partition.py tests/test_parquet_partition.py
git commit -m "feat: make a correction add history instead of destroying it

append_partition refuses a snapshot id already on disk rather than overwriting.
A rebuild that silently replaced a part would erase the history the store exists
to keep, and the run would look clean."
```

---

### Task 3: Trade bars carrying a correct availability time

**Files:**
- Create: `src/store/trade_bars.py`
- Create: `tests/test_trade_bars.py`

**Interfaces:**
- Consumes: `capture.frame_codec.IndexEntry`; `store.temporal_schema` constants
- Produces: `Trade` (frozen dataclass: `symbol: str`, `venue: str`, `price: float`, `size: float`, `event_time_ns: int`, `ingestion_time_ns: int`); `UnknownVenueFormat(ValueError)`; `extract_trades(payload: str, entry: IndexEntry, venue: str, symbol: str) -> list[Trade]`; `build_bars(trades: Iterable[Trade], interval_ns: int) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trade_bars.py
"""Bars are where look-ahead enters, so these tests attack the availability time."""
from __future__ import annotations

import pandas as pd
import pytest

from capture.frame_codec import IndexEntry
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL
from store.trade_bars import Trade, UnknownVenueFormat, build_bars, extract_trades

MINUTE_NS = 60_000_000_000

# Real frames from the archive, 2026-08-02.
BINANCE_FRAME = (
    '{"stream":"btcusdt@trade","data":{"e":"trade","E":1785685177439,'
    '"T":1785685177439,"s":"BTCUSDT","t":7947325360,"p":"63113.20","q":"0.001",'
    '"X":"MARKET","m":false,"st":1}}'
)
HYPERLIQUID_FRAME = (
    '{"channel":"trades","data":['
    '{"coin":"BTC","side":"A","px":"63123.0","sz":"0.0002","time":1785685210021,"tid":1},'
    '{"coin":"BTC","side":"B","px":"63124.0","sz":"0.00343","time":1785685212972,"tid":2}]}'
)


def _entry(t_recv_ns: int, t_exch_ms: int | None = None) -> IndexEntry:
    return IndexEntry(n=0, t_recv_ns=t_recv_ns, t_exch_ms=t_exch_ms,
                      seq=None, kind="data", esc=False)


def test_binance_frame_yields_one_trade():
    entry = _entry(1785685177508349176, 1785685177439)
    trades = extract_trades(BINANCE_FRAME, entry, "binance", "BTCUSDT")
    assert len(trades) == 1
    assert trades[0].price == pytest.approx(63113.20)
    assert trades[0].size == pytest.approx(0.001)
    assert trades[0].event_time_ns == 1785685177439 * 1_000_000
    assert trades[0].ingestion_time_ns == 1785685177508349176


def test_hyperliquid_frame_yields_every_trade_in_the_array():
    """One frame carries many trades; dropping to the first loses real volume."""
    entry = _entry(1785685241650215450, 1785685210021)
    trades = extract_trades(HYPERLIQUID_FRAME, entry, "hyperliquid", "BTC")
    assert len(trades) == 2
    assert [t.price for t in trades] == pytest.approx([63123.0, 63124.0])


def test_every_trade_in_a_batch_shares_the_frames_ingestion_time():
    """They arrived together, whatever their venue timestamps say.

    Hyperliquid's reconnect backfill puts trades up to 32 seconds apart in one
    frame. All of them became knowable at the instant the frame landed, and
    assigning each its own arrival would fabricate an arrival that never happened.
    """
    entry = _entry(1785685241650215450, 1785685210021)
    trades = extract_trades(HYPERLIQUID_FRAME, entry, "hyperliquid", "BTC")
    assert {t.ingestion_time_ns for t in trades} == {1785685241650215450}
    assert len({t.event_time_ns for t in trades}) == 2


def test_an_unknown_venue_is_refused_rather_than_guessed():
    with pytest.raises(UnknownVenueFormat, match="kraken"):
        extract_trades("{}", _entry(1), "kraken", "BTCUSD")


def test_bar_availability_is_its_close_when_data_arrived_promptly():
    """The ordinary case: a bar becomes usable when it closes, not before."""
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0, 1.0,
                    event_time_ns=open_ns + 1_000, ingestion_time_ns=open_ns + 71_000_000)]
    bars = build_bars(trades, MINUTE_NS)
    assert bars.loc[0, AVAILABILITY_TIME] == open_ns + MINUTE_NS


def test_a_late_trade_pushes_availability_past_the_bar_close():
    """The leak this layer exists to prevent, in its exact form.

    A trade whose venue time falls inside a bar but which arrived 32 seconds after
    that bar closed did not exist at close. Marking the bar available at close
    lets a backtest read a price that had not arrived, and nothing in the output
    looks wrong.
    """
    open_ns = 100 * MINUTE_NS
    arrived_ns = open_ns + MINUTE_NS + 32 * 1_000_000_000
    trades = [Trade("BTC", "hyperliquid", 100.0, 1.0,
                    event_time_ns=open_ns + 1_000, ingestion_time_ns=arrived_ns)]
    bars = build_bars(trades, MINUTE_NS)
    assert bars.loc[0, AVAILABILITY_TIME] == arrived_ns
    assert bars.loc[0, AVAILABILITY_TIME] > open_ns + MINUTE_NS


def test_bar_event_time_is_the_bar_open():
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0, 1.0, open_ns + 5, open_ns + 10)]
    assert build_bars(trades, MINUTE_NS).loc[0, EVENT_TIME] == open_ns


def test_ohlcv_is_computed_in_event_time_order():
    """Frames can arrive out of order; open and close must follow the venue's clock."""
    open_ns = 100 * MINUTE_NS
    trades = [
        Trade("BTCUSDT", "binance", 102.0, 1.0, open_ns + 30, open_ns + 40),
        Trade("BTCUSDT", "binance", 100.0, 2.0, open_ns + 10, open_ns + 50),
        Trade("BTCUSDT", "binance", 105.0, 3.0, open_ns + 20, open_ns + 60),
    ]
    bar = build_bars(trades, MINUTE_NS).iloc[0]
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (100.0, 105.0, 100.0, 102.0)
    assert bar["volume"] == pytest.approx(6.0)
    assert bar["trades"] == 3


def test_bars_from_no_trades_are_empty_not_zero_filled():
    """A bar with no trades did not happen; inventing a flat one invents liquidity."""
    assert build_bars([], MINUTE_NS).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_trade_bars.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'store.trade_bars'`

- [ ] **Step 3: Write the implementation**

```python
# src/store/trade_bars.py
"""Raw venue frames to OHLCV bars, each carrying when it became knowable.

The load-bearing line in this module is one expression:

    availability = max(bar_close, latest ingestion of the trades in the bar)

A bar is not usable at its close if the trades composing it had not arrived by
then. Measured on the real archive (2026-08-02): binance trade frames land 70 ms
after the venue timestamp at the median and never later than 205 ms, but 4 of
6,548 hyperliquid frames arrived over 10 seconds late, one by 33.8 seconds, and a
single frame carried trades spanning 32.4 seconds of venue time. Those are
reconnect backfill, they are rare, and they are exactly the rows that make a
backtest look better than the system can be.

Venue formats differ structurally and are not unified by guessing: binance sends
one trade per frame, hyperliquid sends an array. An unrecognised venue raises
rather than returning nothing, because a silent empty list reads downstream as a
quiet market.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from capture.frame_codec import IndexEntry
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

_MS_TO_NS = 1_000_000


@dataclass(frozen=True)
class Trade:
    symbol: str
    venue: str
    price: float
    size: float
    event_time_ns: int
    ingestion_time_ns: int


class UnknownVenueFormat(ValueError):
    """No extractor for this venue's frame shape."""


def _extract_binance(body: dict, entry: IndexEntry, symbol: str, venue: str) -> list[Trade]:
    data = body.get("data", {})
    if data.get("e") != "trade":
        return []
    # "T" is trade time; "E" is event-emission time. They are equal on this feed
    # today, but T is the one that describes the trade.
    event_ms = data.get("T", data.get("E"))
    return [Trade(
        symbol=data.get("s", symbol),
        venue=venue,
        price=float(data["p"]),
        size=float(data["q"]),
        event_time_ns=int(event_ms) * _MS_TO_NS,
        ingestion_time_ns=entry.t_recv_ns,
    )]


def _extract_hyperliquid(body: dict, entry: IndexEntry, symbol: str, venue: str) -> list[Trade]:
    if body.get("channel") != "trades":
        return []
    trades = []
    for item in body.get("data", []) or []:
        if "px" not in item or "time" not in item:
            continue
        trades.append(Trade(
            symbol=item.get("coin", symbol),
            venue=venue,
            price=float(item["px"]),
            size=float(item["sz"]),
            event_time_ns=int(item["time"]) * _MS_TO_NS,
            # Every trade in the batch shares the frame's arrival: they became
            # knowable together, whatever their venue timestamps say.
            ingestion_time_ns=entry.t_recv_ns,
        ))
    return trades


_EXTRACTORS = {
    "binance": _extract_binance,
    "hyperliquid": _extract_hyperliquid,
}


def extract_trades(payload: str, entry: IndexEntry, venue: str, symbol: str) -> list[Trade]:
    """Trades in one captured frame. Raises on a venue with no extractor."""
    extractor = _EXTRACTORS.get(venue)
    if extractor is None:
        raise UnknownVenueFormat(
            f"no trade extractor for venue '{venue}'; add one rather than letting "
            f"its frames read downstream as a quiet market")
    if entry.kind != "data":
        return []
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return extractor(body, entry, symbol, venue)


def build_bars(trades: Iterable[Trade], interval_ns: int) -> pd.DataFrame:
    """OHLCV per (symbol, interval), stamped with when each bar became knowable."""
    rows = list(trades)
    if not rows:
        # No trades is not a flat bar. Zero-filling would invent liquidity that
        # a backtest would then assume it could trade against.
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: t.symbol, VENUE: t.venue, "price": t.price, "size": t.size,
        EVENT_TIME: t.event_time_ns, INGESTION_TIME: t.ingestion_time_ns,
    } for t in rows])
    frame["bar_open_ns"] = (frame[EVENT_TIME] // interval_ns) * interval_ns

    # Sorted by event time so open and close follow the venue's clock rather than
    # the order frames happened to arrive in.
    frame = frame.sort_values(EVENT_TIME, kind="mergesort")

    grouped = frame.groupby([SYMBOL, VENUE, "bar_open_ns"], sort=True)
    bars = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trades=("price", "size"),
        latest_ingestion=(INGESTION_TIME, "max"),
    ).reset_index()

    bar_close = bars["bar_open_ns"] + interval_ns
    bars[EVENT_TIME] = bars["bar_open_ns"].astype("int64")
    bars[INGESTION_TIME] = bars["latest_ingestion"].astype("int64")
    # The whole point of this module.
    bars[AVAILABILITY_TIME] = bars[["latest_ingestion"]].assign(close=bar_close).max(axis=1).astype("int64")

    return bars.drop(columns=["bar_open_ns", "latest_ingestion"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_trade_bars.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
cd ~/trading-system
git add src/store/trade_bars.py tests/test_trade_bars.py
git commit -m "feat: stamp each bar with when it became knowable, not when it closed

Measured on the archive: 4 of 6,548 hyperliquid frames arrived over 10s late,
one by 33.8s, and a single frame carried trades spanning 32.4s of venue time.
Marking those bars available at close lets a backtest read prices that had not
arrived, and nothing in the output looks wrong."
```

---

### Task 4: The clock-gated reader, and correction resolution

**Files:**
- Create: `src/store/clock_gated_reader.py`
- Create: `tests/test_clock_gated_reader.py`

**Interfaces:**
- Consumes: `store.parquet_partition.read_dataset`; `store.temporal_schema` constants
- Produces: `ClockGatedReader(store_root: Path, dataset: str)` with `read_as_of(sim_clock_ns: int, symbols: Sequence[str] | None = None) -> pd.DataFrame`; `join_as_of(left: pd.DataFrame, right: pd.DataFrame, suffix: str, tolerance_ns: int | None = None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clock_gated_reader.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_clock_gated_reader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'store.clock_gated_reader'`

- [ ] **Step 3: Write the implementation**

```python
# src/store/clock_gated_reader.py
"""The only path to stored data, shared by backtest and live.

One reader for both, because two access paths is how an off-by-one in windowing
produces a great backtest and a broken system: the backtest reads one way, live
reads another, and the discrepancy is invisible until capital is behind it. Live
passes the wall clock as `sim_clock_ns`; a backtest passes its simulated clock.
Neither has any other door.

Filtering is on `availability_time_ns` and never on `event_time_ns`. Joining on
event time is the classic leak - it attaches a row to a moment before that row
existed - so `join_as_of` keys on availability time and there is no parameter to
change that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from store.parquet_partition import read_dataset
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, SYMBOL


class ClockGatedReader:
    """Serves rows whose availability time has arrived, newest version per event."""

    def __init__(self, store_root: Path, dataset: str) -> None:
        self._store_root = Path(store_root)
        self._dataset = dataset

    def read_as_of(self, sim_clock_ns: int,
                   symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Everything knowable at `sim_clock_ns`, and nothing else.

        Inclusive at the boundary: a row available exactly at T is usable at T.
        Off by one in this comparison silently drops the newest bar on every read.
        """
        frame = read_dataset(self._store_root, self._dataset)
        if frame.empty:
            return frame

        visible = frame[frame[AVAILABILITY_TIME] <= int(sim_clock_ns)]
        if symbols is not None:
            visible = visible[visible[SYMBOL].isin(list(symbols))]
        if visible.empty:
            return visible.reset_index(drop=True)

        # A correction is a second row for the same event with a later
        # availability time. Keeping both would double-count the bar; keeping the
        # first would ignore the correction. Sort ascending and take the last
        # visible version per event.
        ordered = visible.sort_values(AVAILABILITY_TIME, kind="mergesort")
        latest = ordered.groupby([SYMBOL, EVENT_TIME], as_index=False, sort=True).last()
        return latest.sort_values([SYMBOL, EVENT_TIME]).reset_index(drop=True)


def join_as_of(left: pd.DataFrame, right: pd.DataFrame, suffix: str,
               tolerance_ns: int | None = None) -> pd.DataFrame:
    """Attach the most recent right-hand row that was already available.

    `direction="backward"` is what makes this safe: it can only reach into the
    past. A forward or nearest join reaches into the future by construction.

    `tolerance_ns` bounds how stale a match may be. Without it, a funding rate
    from an hour ago attaches to a signal now and reads as current context when
    it describes a different market.
    """
    if left.empty:
        return left.copy()

    left_sorted = left.sort_values(AVAILABILITY_TIME, kind="mergesort")
    if right.empty:
        return left_sorted.reset_index(drop=True)
    right_sorted = right.sort_values(AVAILABILITY_TIME, kind="mergesort")

    return pd.merge_asof(
        left_sorted,
        right_sorted,
        on=AVAILABILITY_TIME,
        by=SYMBOL,
        direction="backward",
        tolerance=tolerance_ns,
        suffixes=("", suffix),
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_clock_gated_reader.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
cd ~/trading-system
git add src/store/clock_gated_reader.py tests/test_clock_gated_reader.py
git commit -m "feat: give backtest and live one door into the data

Two access paths is how an off-by-one in windowing yields a great backtest and a
broken system. Filtering is on availability time only, and join_as_of has no
parameter that would let it key on event time."
```

---

### Task 5: The synthetic leakage suite

**Files:**
- Create: `tests/test_store_leakage.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.

`ARCHITECTURE.md` names this deliverable explicitly: *"a few hundred lines, unit-tested against synthetic leakage cases."* This task is that suite. It is deliberately separate from the per-module tests: those check that each part behaves, this checks that the assembled store cannot be made to leak.

- [ ] **Step 1: Write the suite**

```python
# tests/test_store_leakage.py
"""Adversarial tests: try to read the future out of the store, and fail.

Each test here encodes a real way look-ahead enters a backtest. They are written
as attacks rather than as behaviour checks because a leak is not a wrong answer -
it is a plausible one that is better than reality, and it will not look like a bug.
"""
from __future__ import annotations

import pandas as pd
import pytest

from store.clock_gated_reader import ClockGatedReader, join_as_of
from store.parquet_partition import append_partition
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE
from store.trade_bars import Trade, build_bars

MINUTE_NS = 60_000_000_000


def _store(tmp_path, frame: pd.DataFrame, snapshot: str):
    append_partition(tmp_path, "bars_1m", frame, snapshot)
    return ClockGatedReader(tmp_path, "bars_1m")


def test_a_bar_built_from_late_data_is_invisible_at_its_close(tmp_path):
    """The headline case, end to end.

    A hyperliquid reconnect delivered a trade 32 seconds after its bar closed.
    Anyone reading at the close must not see that bar - it did not exist yet.
    """
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    arrived_ns = close_ns + 32 * 1_000_000_000
    bars = build_bars(
        [Trade("BTC", "hyperliquid", 63123.0, 0.0002, open_ns + 21, arrived_ns)],
        MINUTE_NS)
    reader = _store(tmp_path, bars, "snap1")

    assert reader.read_as_of(close_ns).empty, "the bar was readable before it arrived"
    assert len(reader.read_as_of(arrived_ns)) == 1


def test_a_promptly_built_bar_is_visible_at_its_close(tmp_path):
    """The guard against over-correcting: gating must not hide ordinary data."""
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    bars = build_bars(
        [Trade("BTCUSDT", "binance", 63113.2, 0.001, open_ns + 5, open_ns + 70_000_000)],
        MINUTE_NS)
    assert len(_store(tmp_path, bars, "snap1").read_as_of(close_ns)) == 1


def test_no_read_can_return_a_row_from_its_own_future(tmp_path):
    """Swept over many clocks rather than one: an off-by-one hides at a single point."""
    open_ns = 100 * MINUTE_NS
    trades = [
        Trade("BTCUSDT", "binance", 100.0, 1.0, open_ns + 5, open_ns + 70_000_000),
        Trade("BTCUSDT", "binance", 101.0, 1.0, open_ns + MINUTE_NS + 5,
              open_ns + MINUTE_NS + 80_000_000),
        Trade("BTCUSDT", "binance", 102.0, 1.0, open_ns + 2 * MINUTE_NS + 5,
              open_ns + 3 * MINUTE_NS),
    ]
    reader = _store(tmp_path, build_bars(trades, MINUTE_NS), "snap1")
    for step in range(0, 5 * 60, 7):
        clock = open_ns + step * 1_000_000_000
        visible = reader.read_as_of(clock)
        if visible.empty:
            continue
        assert visible[AVAILABILITY_TIME].max() <= clock, f"leaked at clock {clock}"


def test_a_correction_cannot_be_seen_before_it_was_made(tmp_path):
    """Late data revises a closed bar. The revision is not knowable in advance."""
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    prompt = build_bars(
        [Trade("BTC", "hyperliquid", 100.0, 1.0, open_ns + 5, open_ns + 300_000_000)],
        MINUTE_NS)
    revised = build_bars(
        [Trade("BTC", "hyperliquid", 100.0, 1.0, open_ns + 5, open_ns + 300_000_000),
         Trade("BTC", "hyperliquid", 999.0, 5.0, open_ns + 10, close_ns + 30_000_000_000)],
        MINUTE_NS)

    append_partition(tmp_path, "bars_1m", prompt, "snap1")
    append_partition(tmp_path, "bars_1m", revised, "snap2")
    reader = ClockGatedReader(tmp_path, "bars_1m")

    at_close = reader.read_as_of(close_ns)
    assert len(at_close) == 1
    assert at_close.iloc[0]["high"] == 100.0, "the revision leaked backwards"

    after = reader.read_as_of(close_ns + 60 * 1_000_000_000)
    assert len(after) == 1, "the correction duplicated the bar instead of replacing it"
    assert after.iloc[0]["high"] == 999.0


def test_joining_on_event_time_would_leak_and_the_api_will_not_do_it(tmp_path):
    """A funding row describing an old event that only arrived later.

    Keyed on event time it attaches to a signal an hour before it was published.
    Keyed on availability time it correctly attaches to nothing.
    """
    signal = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [1_000],
                           AVAILABILITY_TIME: [1_000], "signal": [1.0]})
    funding = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [900],
                            AVAILABILITY_TIME: [50_000], "funding": [0.01]})
    joined = join_as_of(signal, funding, suffix="_f")
    assert pd.isna(joined.loc[0, "funding"])


def test_the_same_reader_serves_backtest_and_live_identically(tmp_path):
    """Two paths would diverge; this asserts there is only one.

    'Live' is the reader with the wall clock, 'backtest' the same reader with a
    simulated one. For the same clock value they must be byte-identical.
    """
    open_ns = 100 * MINUTE_NS
    bars = build_bars(
        [Trade("BTCUSDT", "binance", 100.0, 1.0, open_ns + 5, open_ns + 70_000_000)],
        MINUTE_NS)
    append_partition(tmp_path, "bars_1m", bars, "snap1")

    clock = open_ns + 2 * MINUTE_NS
    backtest = ClockGatedReader(tmp_path, "bars_1m").read_as_of(clock)
    live = ClockGatedReader(tmp_path, "bars_1m").read_as_of(clock)
    pd.testing.assert_frame_equal(backtest, live)


def test_history_never_shrinks_as_the_clock_advances(tmp_path):
    """Monotonicity across a sweep. A row that was visible must stay visible."""
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0 + i, 1.0,
                    open_ns + i * MINUTE_NS + 5,
                    open_ns + i * MINUTE_NS + 70_000_000) for i in range(5)]
    reader = _store(tmp_path, build_bars(trades, MINUTE_NS), "snap1")

    seen: set[int] = set()
    for step in range(0, 8 * 60, 11):
        clock = open_ns + step * 1_000_000_000
        events = set(reader.read_as_of(clock)[EVENT_TIME]) if not reader.read_as_of(clock).empty else set()
        assert seen <= events, f"history shrank at clock {clock}"
        seen = events
```

- [ ] **Step 2: Run the suite**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_store_leakage.py -q`
Expected: PASS, 7 tests. **If any fail, the defect is in Tasks 1–4, not in this file — fix the implementation, never the assertion.**

- [ ] **Step 3: Prove the suite can actually catch a leak**

Temporarily break the gate to confirm the tests are load-bearing rather than decorative. In `clock_gated_reader.py`, change `<=` to `<` in `read_as_of`:

```python
visible = frame[frame[AVAILABILITY_TIME] < int(sim_clock_ns)]
```

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_store_leakage.py -q`
Expected: FAIL — `test_a_promptly_built_bar_is_visible_at_its_close`

Then revert the change and re-run to confirm PASS. A suite that passes against a broken gate is worse than none.

- [ ] **Step 4: Commit**

```bash
cd ~/trading-system
git add tests/test_store_leakage.py
git commit -m "test: try to read the future out of the store, and fail

ARCHITECTURE.md names this deliverable: unit-tested against synthetic leakage
cases. Written as attacks, because a leak is not a wrong answer - it is a
plausible one that is better than reality, and it will not look like a bug.
Verified the suite catches an inverted comparison in the gate."
```

---

### Task 6: Build the store from the real archive

**Files:**
- Create: `src/store/cli.py`
- Create: `tests/test_store_cli.py`

**Interfaces:**
- Consumes: `capture.raw_writer.read_pair`, `capture.raw_writer.paths_for`; Tasks 1–4
- Produces: `build_bars_for_day(capture_root: Path, store_root: Path, venue: str, date: str, symbols: Sequence[str], interval_ns: int) -> dict`; `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_cli.py
"""The build must refuse damaged input rather than quietly producing fewer bars."""
from __future__ import annotations

from pathlib import Path

import pytest

from store.cli import build_bars_for_day


def test_building_from_an_absent_day_reports_zero_rather_than_crashing(tmp_path):
    summary = build_bars_for_day(
        capture_root=tmp_path, store_root=tmp_path / "store",
        venue="binance", date="2026-01-01", symbols=["BTCUSDT"],
        interval_ns=60_000_000_000)
    assert summary["frames"] == 0
    assert summary["bars"] == 0


def test_building_twice_from_identical_input_is_refused(tmp_path, monkeypatch):
    """The snapshot id is content-derived, so a rebuild collides by design.

    That collision is the append-only guarantee working: rebuilding the same
    inputs cannot silently replace the parts a previous run wrote.
    """
    from capture.frame_codec import IndexEntry
    from store.parquet_partition import PartitionExistsError
    from store import cli as store_cli

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    raw_path = source / "trade_BTCUSDT_2026-08-02T00.ndjson.zst"
    raw_path.write_bytes(b"placeholder")
    (source / "trade_BTCUSDT_2026-08-02T00.idx.zst").write_bytes(b"placeholder")

    frame = (
        '{"stream":"btcusdt@trade","data":{"e":"trade","T":1785685177439,'
        '"s":"BTCUSDT","p":"63113.20","q":"0.001"}}'
    )
    entry = IndexEntry(n=0, t_recv_ns=1785685177508349176, t_exch_ms=1785685177439,
                       seq=None, kind="data", esc=False)
    monkeypatch.setattr(store_cli, "read_pair", lambda raw, idx: [(frame, entry)])

    build = lambda: store_cli.build_bars_for_day(
        capture_root=tmp_path, store_root=tmp_path / "store", venue="binance",
        date="2026-08-02", symbols=["BTCUSDT"], interval_ns=60_000_000_000)

    assert build()["bars"] == 1
    with pytest.raises(PartitionExistsError):
        build()
```

- [ ] **Step 2: Write the implementation**

```python
# src/store/cli.py
"""Builds the bitemporal store from the raw archive.

Reads through `capture.raw_writer.read_pair`, which refuses a torn file rather
than returning its readable prefix. That refusal is load-bearing here: silently
building from a truncated hour produces a store that is quietly missing trades,
and every statistic computed from it is wrong in a way nothing reports.

    python -m store.cli --venue binance --date 2026-08-02 --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from capture.raw_writer import RAW_SUFFIX, read_pair
from store.parquet_partition import append_partition, compute_snapshot_id
from store.trade_bars import build_bars, extract_trades

_TRADE_STREAMS = {"binance": "trade", "hyperliquid": "trades"}
DEFAULT_INTERVAL_NS = 60_000_000_000


def _hour_files(capture_root: Path, venue: str, date: str,
                stream: str, symbol: str) -> list[Path]:
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{stream}_{symbol}_*{RAW_SUFFIX}"))


def build_bars_for_day(capture_root: Path, store_root: Path, venue: str, date: str,
                       symbols: Sequence[str], interval_ns: int) -> dict:
    """Read one venue-day of trades and append the resulting bars."""
    stream = _TRADE_STREAMS.get(venue)
    if stream is None:
        raise SystemExit(f"no trade stream known for venue '{venue}'")

    trades = []
    sources: list[Path] = []
    frames = 0
    for symbol in symbols:
        for raw_path in _hour_files(capture_root, venue, date, stream, symbol):
            idx_path = raw_path.with_name(raw_path.name.replace(".ndjson.zst", ".idx.zst"))
            for payload, entry in read_pair(raw_path, idx_path):
                frames += 1
                trades.extend(extract_trades(payload, entry, venue, symbol))
            sources.extend([raw_path, idx_path])

    if not trades:
        return {"frames": frames, "trades": 0, "bars": 0, "snapshot_id": None}

    bars = build_bars(trades, interval_ns)
    snapshot_id = compute_snapshot_id(sources)
    append_partition(store_root, f"bars_{interval_ns}ns", bars, snapshot_id)
    return {"frames": frames, "trades": len(trades), "bars": len(bars),
            "snapshot_id": snapshot_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="store", description="Build the bitemporal store from captured frames.")
    parser.add_argument("--venue", required=True, choices=sorted(_TRADE_STREAMS))
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    summary = build_bars_for_day(
        Path(args.capture_root), Path(args.store_root),
        args.venue, args.date, symbols, args.interval_ns)
    print(f"{summary['frames']} frames -> {summary['trades']} trades -> "
          f"{summary['bars']} bars (snapshot {summary['snapshot_id']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the unit tests**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_store_cli.py -q`
Expected: PASS, 2 tests

- [ ] **Step 4: Build from the real archive**

```bash
cd ~/trading-system
.venv/bin/python -m store.cli --venue binance --date 2026-08-02 --symbols BTCUSDT,ETHUSDT,SOLUSDT
.venv/bin/python -m store.cli --venue hyperliquid --date 2026-08-02 --symbols BTC,ETH,SOL
```

Expected: non-zero frame, trade and bar counts for both, and a 16-character snapshot id.

- [ ] **Step 5: Verify the built store against the measured facts**

```bash
cd ~/trading-system && .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from store.clock_gated_reader import ClockGatedReader
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME

reader = ClockGatedReader(Path.home()/'capture'/'store', 'bars_60000000000ns')
bars = reader.read_as_of(10**19)
print('bars:', len(bars))
lag_s = (bars[AVAILABILITY_TIME] - (bars[EVENT_TIME] + 60_000_000_000)) / 1e9
late = lag_s[lag_s > 0]
print(f'bars available after their close: {len(late)} of {len(bars)}')
print(f'worst late bar: {late.max():.1f}s' if len(late) else 'none late')
"
```

Expected: some bars available strictly after their close, with the worst approaching the measured 32-second hyperliquid backfill. **If zero bars are late, the availability calculation is wrong** — the archive is known to contain late frames, so a clean result means the max is not being taken.

- [ ] **Step 6: Commit**

```bash
cd ~/trading-system
git add src/store/cli.py tests/test_store_cli.py
git commit -m "feat: build the store from the archive, refusing torn input

read_pair refuses a truncated hour rather than returning its readable prefix.
Building from one anyway produces a store quietly missing trades, and every
statistic computed from it is wrong in a way nothing reports."
```

---

### Task 7: Put the store on the status wall

**Files:**
- Modify: `src/statuswall/evidence.py` — add two probes and two `PROBES` entries
- Modify: `tests/test_statuswall.py` — add tests for the new probes

**Interfaces:**
- Consumes: `store.clock_gated_reader.ClockGatedReader`
- Produces: `probe_bitemporal_store(facts) -> ProbeResult`; `probe_clock_gated_access(facts) -> ProbeResult`

Rule 8: a feature that now exists must stop reading NOT BUILT, and the change must come from a probe rather than from editing a label. `FEATURES.md` §1 has no row for these — they live in the Phase 0 gate text — so this task also adds two rows to §1 so the wall can carry them.

- [ ] **Step 1: Add the catalogue rows**

In `~/research/FEATURES.md`, section `## 1. Market data & ingestion`, append two rows:

```markdown
| **Bitemporal store** | P0 | Every row carries event, ingestion and availability time. Append-only — corrections are new rows, never overwrites |
| **Clock-gated access API** | P0 | The only path to data, shared by backtest and live. Serves `availability_time <= sim_clock`; joins key on availability, never event time |
```

- [ ] **Step 2: Write the failing tests**

```python
# append to tests/test_statuswall.py

def test_store_probe_reports_not_built_when_no_store_exists(tmp_path):
    """Before the first build there is no store, and the wall must say so."""
    from statuswall.evidence import NOT_BUILT, probe_bitemporal_store
    facts = _facts(capture_root=tmp_path)
    assert probe_bitemporal_store(facts).state == NOT_BUILT


def test_store_probe_reports_ok_once_bars_are_readable(tmp_path):
    """The state must come from reading the store, not from the module existing."""
    import pandas as pd
    from statuswall.evidence import OK, probe_bitemporal_store
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050], AVAILABILITY_TIME: [1_100], "close": [63113.2],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, "snap1")

    result = probe_bitemporal_store(_facts(capture_root=tmp_path))
    assert result.state == OK
    assert "1" in result.detail
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_statuswall.py -q -k store`
Expected: FAIL — `ImportError: cannot import name 'probe_bitemporal_store'`

- [ ] **Step 4: Write the probes**

Add to `src/statuswall/evidence.py`, above the `PROBES` dict:

```python
def _store_root(facts: SystemFacts) -> Path:
    return facts.capture_root / "store"


def _bar_datasets(facts: SystemFacts) -> list[Path]:
    root = _store_root(facts)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("bars_"))


def probe_bitemporal_store(facts: SystemFacts) -> ProbeResult:
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store built from the archive yet",
                           "capture/store")
    parts = sum(1 for dataset in datasets for _ in dataset.rglob("*.parquet"))
    from store.clock_gated_reader import ClockGatedReader
    rows = len(ClockGatedReader(_store_root(facts), datasets[0].name).read_as_of(2**62))
    return ProbeResult(
        OK,
        f"{rows} rows across {parts} append-only part(s) in {len(datasets)} dataset(s)",
        f"capture/store/{datasets[0].name}",
    )


def probe_clock_gated_access(facts: SystemFacts) -> ProbeResult:
    """Reports on the gate by exercising it, not by checking the file exists."""
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store to gate", "src/store/clock_gated_reader.py")
    from store.clock_gated_reader import ClockGatedReader
    from store.temporal_schema import AVAILABILITY_TIME

    reader = ClockGatedReader(_store_root(facts), datasets[0].name)
    everything = reader.read_as_of(2**62)
    if everything.empty:
        return ProbeResult(DEGRADED, "store exists but reads empty", "ClockGatedReader.read_as_of")

    earliest = int(everything[AVAILABILITY_TIME].min())
    hidden = reader.read_as_of(earliest - 1)
    if not hidden.empty:
        # The gate is the whole layer. If it lets anything through early, that is
        # a failure of the system's core guarantee, not a degraded metric.
        return ProbeResult(FAILING, f"{len(hidden)} row(s) visible before their availability time",
                           "ClockGatedReader.read_as_of")
    return ProbeResult(OK, f"gate holds: nothing visible before {earliest}",
                       "ClockGatedReader.read_as_of, exercised live")
```

Then add to `PROBES`:

```python
    "bitemporal store": probe_bitemporal_store,
    "clock gated access api": probe_clock_gated_access,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ~/trading-system && .venv/bin/python -m pytest tests/test_statuswall.py -q`
Expected: PASS

- [ ] **Step 6: Regenerate the wall and confirm the change is visible**

```bash
cd ~/trading-system
.venv/bin/python -m statuswall.cli --out ~/research/dashboard/status-wall.html
```

Expected: measured count rises from 11 to 13, and both new features report `OK` rather than `NOT BUILT`.

- [ ] **Step 7: Run the whole suite**

Run: `cd ~/trading-system && .venv/bin/python -m pytest -q`
Expected: PASS — no regression in the 298 pre-existing tests.

- [ ] **Step 8: Commit**

```bash
cd ~/trading-system
git add src/statuswall/evidence.py tests/test_statuswall.py
git commit -m "feat: let the wall report the store, by exercising it

The clock-gate probe reads at one nanosecond before the earliest availability
time and asserts nothing comes back. Checking that the module exists would pass
just as happily against a gate that leaks."
cd ~/research && git add FEATURES.md && git commit -m "docs: catalogue the bitemporal store and clock-gated access

Both were only in the Phase 0 gate text, so the status wall had no row to
attach their probes to."
```

---

## Definition of Done

- [ ] `.venv/bin/python -m pytest -q` passes with no regression against the 298 pre-existing tests
- [ ] `tests/test_store_leakage.py` passes, and has been shown to fail against a deliberately inverted gate
- [ ] The store is built from the real 2026-08-02 archive for both venues
- [ ] Some bars are measurably available *after* their close, with the worst near 32 s — proving the availability calculation is doing work
- [ ] The status wall reports both new features as `OK`, from probes, at 13/178 measured

## Explicitly Not In This Plan

Deferred, and each needs its own plan:

- **L2 book snapshots and depth bars.** Depth is 80% of the archive by volume and needs order-book reconstruction from diffs — a different problem from trade aggregation.
- **Funding, basis and open interest datasets.** Same store, different builders.
- **Provenance stamper and frozen universe snapshots.** The other two Layer 0 components in `ARCHITECTURE.md`; they belong with the validation harness, which owns run identity.
- **Gap-flagged backfill.** The store records what was captured; deciding what an interpolated candle means is a separate decision, and `FEATURES.md` requires such rows be labelled rather than blended.
- **The validation harness** — experiment ledger, Trial Registry, Holdout Custodian, purge/embargo, DSR, MinBTL. This is the next sub-project, and it must land before the first model is trained: the trial count cannot be reconstructed afterwards.
