# Layer 0 Raw Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a raw market-data capture service that records Binance USDⓈ-M perp and Hyperliquid perp websocket frames verbatim to disk, forever, with an honest ledger of every gap.

**Architecture:** Six units with one responsibility each. A `venue_recorder` owns websocket subscriptions and pushes frames onto a bounded queue; a `raw_writer` appends them byte-exactly to a `.ndjson.zst` file with a parallel `.idx.zst` metadata sidecar; a `capture_ledger` records every anomaly as a first-class event; a `universe_tracker` polls instrument lists to capture point-in-time membership; `capture_health` reports integrity and disk runway; `archive_offloader` uploads to GCS once that access is proven.

**Tech Stack:** Python 3.12.13 (via uv), `websockets` 17.0.1, `zstandard` 0.25.0, `pytest`, `gcloud`/`gsutil` for offload.

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.12** — system Python is 3.14.4 and must NOT be used. All commands use `uv run --python 3.12`. Rationale: `polars` supports ≤3.13, `nautilus_trader` requires ≥3.12,<3.15.
- **The governing invariant: a written frame is never modified.** The bytes on disk are exactly the bytes the venue sent. All derived data lives in sidecar files.
- **Never drop data silently.** Every anomaly becomes a `capture_ledger` event. Malformed frames are still written verbatim.
- **Never modify data to "fix" it.**
- **Naming (Rule 7):** files named for responsibility (`raw_writer.py`, `capture_ledger.py`), never `utils`/`helpers`/`common`. Functions are verb+object. Predicates read as questions. Side effects visible in the name — `get_x` must not write.
- **Capture tiers:** Core = `BTC`, `ETH`, `SOL` on Binance USDⓈ-M perps + Hyperliquid perps, full L2 depth + trades + funding + OI + liquidations. Tail = every perp each venue lists, discovered dynamically, no depth.
- **Timezone:** all rotation and directory dates are UTC. Never local time.
- **Offload is gated** on B1 (GCS write proven). Tasks 1–10 must work with zero cloud dependency.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, deps, pytest config |
| `src/capture/frame_codec.py` | Newline escaping/unescaping, index entry encode/decode |
| `src/capture/raw_writer.py` | Byte-exact append, hourly rotation, raw/index pair, recovery |
| `src/capture/capture_ledger.py` | Anomaly events to disk |
| `src/capture/sequencing.py` | Per-venue gap detection strategies |
| `src/capture/venues/binance.py` | Binance stream specs, URL building, metadata extraction, instrument list |
| `src/capture/venues/hyperliquid.py` | Same for Hyperliquid |
| `src/capture/venue_recorder.py` | Async subscribe/reconnect loop, bounded queue |
| `src/capture/universe_tracker.py` | Instrument list polling, membership diff events |
| `src/capture/capture_health.py` | Disk runway, stream staleness, integrity report |
| `src/capture/archive_offloader.py` | GCS upload + verified prune |
| `src/capture/cli.py` | Entry points |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/capture/__init__.py`
- Create: `tests/test_scaffolding.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `capture`; the `uv run --python 3.12 pytest` command every later task uses

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffolding.py
def test_package_imports():
    import capture
    assert capture.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/trading-system && uv run --python 3.12 pytest tests/test_scaffolding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "capture"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "websockets>=17.0.1",
    "zstandard>=0.25.0",
    "aiohttp>=3.10",
]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/capture"]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
testpaths = ["tests"]
```

```python
# src/capture/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_scaffolding.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/capture/__init__.py tests/test_scaffolding.py
git commit -m "feat: scaffold capture package on Python 3.12"
```

---

### Task 2: Frame codec — newline escaping and index entries

The probe confirmed venue frames contain no newlines today. This is the guard for the day one does, so line alignment can never silently corrupt.

**Files:**
- Create: `src/capture/frame_codec.py`
- Create: `tests/test_frame_codec.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `escape_payload(payload: str) -> tuple[str, bool]` — returns (escaped, was_escaped)
  - `unescape_payload(payload: str) -> str`
  - `IndexEntry` frozen dataclass with fields `n:int, t_recv_ns:int, t_exch_ms:int|None, seq:dict|None, kind:str, esc:bool`
  - `encode_index_entry(entry: IndexEntry) -> str` (one JSON line, no trailing newline)
  - `decode_index_entry(line: str) -> IndexEntry`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame_codec.py
from capture.frame_codec import (
    escape_payload, unescape_payload, IndexEntry,
    encode_index_entry, decode_index_entry,
)


def test_clean_payload_is_untouched():
    payload = '{"e":"depthUpdate","U":1,"u":2}'
    escaped, was_escaped = escape_payload(payload)
    assert escaped == payload
    assert was_escaped is False


def test_newline_is_escaped_and_roundtrips():
    payload = '{"a":"x\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_carriage_return_is_escaped_and_roundtrips():
    payload = '{"a":"x\r\ny"}'
    escaped, was_escaped = escape_payload(payload)
    assert was_escaped is True
    assert "\r" not in escaped and "\n" not in escaped
    assert unescape_payload(escaped) == payload


def test_backslash_roundtrips_without_false_escape():
    payload = r'{"a":"C:\path"}'
    escaped, was_escaped = escape_payload(payload)
    assert unescape_payload(escaped) == payload


def test_index_entry_roundtrips():
    entry = IndexEntry(n=7, t_recv_ns=123, t_exch_ms=456,
                       seq={"U": 1, "u": 2}, kind="data", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry


def test_index_entry_allows_missing_exchange_time():
    entry = IndexEntry(n=0, t_recv_ns=1, t_exch_ms=None,
                       seq=None, kind="control", esc=False)
    assert decode_index_entry(encode_index_entry(entry)) == entry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_frame_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.frame_codec'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/frame_codec.py
"""Byte-exact framing helpers.

Raw payloads are stored one per line. A payload containing a literal newline
would break line alignment, so newlines are escaped and the index records that
it happened. Backslash is escaped first so unescaping is unambiguous.
"""
import json
from dataclasses import dataclass, asdict

_ESCAPES = (("\\", "\\\\"), ("\n", "\\n"), ("\r", "\\r"))


def escape_payload(payload: str) -> tuple[str, bool]:
    if "\n" not in payload and "\r" not in payload:
        return payload, False
    out = payload
    for raw, esc in _ESCAPES:
        out = out.replace(raw, esc)
    return out, True


def unescape_payload(payload: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == "\\" and i + 1 < len(payload):
            nxt = payload[i + 1]
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == "r":
                out.append("\r"); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
        out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class IndexEntry:
    n: int
    t_recv_ns: int
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    esc: bool


def encode_index_entry(entry: IndexEntry) -> str:
    return json.dumps(asdict(entry), separators=(",", ":"), sort_keys=True)


def decode_index_entry(line: str) -> IndexEntry:
    return IndexEntry(**json.loads(line))
```

> Note: `escape_payload` returns the payload unchanged when it contains no
> newline, so the backslash escape never fires on clean frames. That keeps
> byte-exactness for every real frame observed in the probe.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_frame_codec.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/frame_codec.py tests/test_frame_codec.py
git commit -m "feat: frame codec with newline guard and index entries"
```

---

### Task 3: Raw writer — byte-exact append and hourly rotation

**Files:**
- Create: `src/capture/raw_writer.py`
- Create: `tests/test_raw_writer.py`

**Interfaces:**
- Consumes: `capture.frame_codec` (`escape_payload`, `IndexEntry`, `encode_index_entry`)
- Produces:
  - `RawWriter(root: Path, venue: str, stream: str, symbol: str)` with methods
    `append(payload:str, t_recv_ns:int, t_exch_ms:int|None, seq:dict|None, kind:str="data") -> int`,
    `flush() -> None`, `close() -> None`
  - `read_pair(raw_path: Path, idx_path: Path) -> list[tuple[str, IndexEntry]]`
  - `hour_key(ts_ns: int) -> str` returning `"YYYY-MM-DDTHH"` in UTC
  - `paths_for(root, venue, stream, symbol, hour) -> tuple[Path, Path]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_raw_writer.py
from pathlib import Path
from capture.raw_writer import RawWriter, read_pair, hour_key, paths_for


def test_hour_key_is_utc():
    # 2026-08-02T05:30:00Z
    assert hour_key(1785648600_000_000_000) == "2026-08-02T05"


def test_written_bytes_are_identical_to_input(tmp_path: Path):
    payload = '{"e":"depthUpdate","E":1785650606214,"U":98135459427,"u":98135459442}'
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append(payload, t_recv_ns=1785648600_000_000_000,
             t_exch_ms=1785650606214, seq={"U": 98135459427, "u": 98135459442})
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 1
    assert pairs[0][0] == payload          # byte-exact
    assert pairs[0][1].n == 0
    assert pairs[0][1].esc is False


def test_index_line_count_matches_raw_line_count(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "trades", "ETHUSDT")
    for i in range(50):
        w.append(f'{{"i":{i}}}', t_recv_ns=1785648600_000_000_000 + i,
                 t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "binance", "trades", "ETHUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 50
    assert [p[1].n for p in pairs] == list(range(50))


def test_payload_with_newline_roundtrips(tmp_path: Path):
    payload = '{"a":"x\ny"}'
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert pairs[0][0] == payload
    assert pairs[0][1].esc is True


def test_rotation_creates_new_hour_file(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"h":5}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"h":6}', t_recv_ns=1785652200_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    r5, i5 = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    r6, i6 = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T06")
    assert read_pair(r5, i5)[0][0] == '{"h":5}'
    assert read_pair(r6, i6)[0][0] == '{"h":6}'
    # n resets per file
    assert read_pair(r6, i6)[0][1].n == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_raw_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.raw_writer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/raw_writer.py
"""Writes venue frames verbatim, one per line, with a parallel index sidecar."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import zstandard

from capture.frame_codec import IndexEntry, encode_index_entry, escape_payload, decode_index_entry, unescape_payload


def hour_key(ts_ns: int) -> str:
    moment = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H")


def paths_for(root: Path, venue: str, stream: str, symbol: str, hour: str) -> tuple[Path, Path]:
    date = hour.split("T")[0]
    folder = Path(root) / "raw" / venue / date
    stem = f"{stream}_{symbol}_{hour}"
    return folder / f"{stem}.ndjson.zst", folder / f"{stem}.idx.zst"


class RawWriter:
    """Append-only writer for one (venue, stream, symbol). Rotates hourly by UTC."""

    def __init__(self, root: Path, venue: str, stream: str, symbol: str) -> None:
        self._root = Path(root)
        self._venue, self._stream, self._symbol = venue, stream, symbol
        self._hour: str | None = None
        self._raw_fh = self._idx_fh = None
        self._raw_z = self._idx_z = None
        self._n = 0

    def _open(self, hour: str) -> None:
        raw_path, idx_path = paths_for(self._root, self._venue, self._stream, self._symbol, hour)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cctx = zstandard.ZstdCompressor(level=3)
        self._raw_fh = open(raw_path, "wb")
        self._idx_fh = open(idx_path, "wb")
        self._raw_z = cctx.stream_writer(self._raw_fh)
        self._idx_z = cctx.stream_writer(self._idx_fh)
        self._hour, self._n = hour, 0

    def append(self, payload: str, t_recv_ns: int, t_exch_ms: int | None,
               seq: dict | None, kind: str = "data") -> int:
        hour = hour_key(t_recv_ns)
        if hour != self._hour:
            self.close()
            self._open(hour)

        escaped, was_escaped = escape_payload(payload)
        entry = IndexEntry(n=self._n, t_recv_ns=t_recv_ns, t_exch_ms=t_exch_ms,
                           seq=seq, kind=kind, esc=was_escaped)
        self._raw_z.write((escaped + "\n").encode("utf-8"))
        self._idx_z.write((encode_index_entry(entry) + "\n").encode("utf-8"))
        self._n += 1
        return entry.n

    def flush(self) -> None:
        if self._raw_z is not None:
            self._raw_z.flush(zstandard.FLUSH_FRAME)
            self._idx_z.flush(zstandard.FLUSH_FRAME)
            self._raw_fh.flush()
            self._idx_fh.flush()

    def close(self) -> None:
        if self._raw_z is None:
            return
        self._raw_z.close(); self._idx_z.close()
        self._raw_fh.close(); self._idx_fh.close()
        self._raw_z = self._idx_z = self._raw_fh = self._idx_fh = None
        self._hour = None


def read_pair(raw_path: Path, idx_path: Path) -> list[tuple[str, IndexEntry]]:
    def _lines(path: Path) -> list[str]:
        # Strict newline split - NOT splitlines(), which also breaks on
        # \v \f \x85 U+2028 U+2029 and would desync raw from index.
        dctx = zstandard.ZstdDecompressor()
        with open(path, "rb") as fh:
            text = dctx.stream_reader(fh).read().decode("utf-8")
        return text.rstrip("\n").split("\n") if text else []

    raw_lines, idx_lines = _lines(raw_path), _lines(idx_path)
    pairs = []
    for r, i in zip(raw_lines, idx_lines):
        entry = decode_index_entry(i)
        # Only unescape what was escaped. Unescaping unconditionally would
        # collapse a literal \\ in an untouched payload and break byte-exactness.
        pairs.append((unescape_payload(r) if entry.esc else r, entry))
    return pairs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_raw_writer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/raw_writer.py tests/test_raw_writer.py
git commit -m "feat: byte-exact raw writer with hourly rotation"
```

---

### Task 4: Crash recovery — reconcile raw/index pairs

A `kill -9` can leave the raw file with more lines than the index. Truncating would discard captured data; instead rebuild the missing index entries with unknown receipt time.

**Files:**
- Modify: `src/capture/raw_writer.py`
- Create: `tests/test_raw_writer_recovery.py`

**Interfaces:**
- Consumes: `RawWriter`, `read_pair`, `paths_for` from Task 3
- Produces: `reconcile_pair(raw_path: Path, idx_path: Path) -> int` returning number of index entries repaired

> **Contract established in Task 3's review, which this task completes.** `read_pair` **raises**
> a named mismatch exception when the raw and index files have different line counts — it does
> not silently `zip()` to the shorter one, because that returns wrong data instead of reporting
> damage. `reconcile_pair` is the repair path: it rebuilds the missing index entries so a
> subsequent `read_pair` succeeds. Detect and refuse, then repair — never paper over.
>
> This means the test below must call `reconcile_pair` *before* `read_pair`, which it already does.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_raw_writer_recovery.py
from pathlib import Path
import zstandard
from capture.raw_writer import RawWriter, read_pair, paths_for, reconcile_pair
from capture.frame_codec import decode_index_entry


def _truncate_index_by_one(idx_path: Path) -> None:
    dctx = zstandard.ZstdDecompressor()
    with open(idx_path, "rb") as fh:
        lines = dctx.stream_reader(fh).read().decode("utf-8").rstrip("\n").split("\n")
    cctx = zstandard.ZstdCompressor(level=3)
    with open(idx_path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines[:-1]:
                w.write((line + "\n").encode("utf-8"))


def test_reconcile_rebuilds_missing_index_entries(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"i":{i}}}', t_recv_ns=1785648600_000_000_000 + i,
                 t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _truncate_index_by_one(idx)

    repaired = reconcile_pair(raw, idx)
    assert repaired == 1

    pairs = read_pair(raw, idx)
    assert len(pairs) == 3
    assert pairs[2][0] == '{"i":2}'
    assert pairs[2][1].t_recv_ns == 0      # unknown, not invented
    assert pairs[2][1].kind == "recovered"


def test_reconcile_is_noop_when_aligned(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "trades", "ETHUSDT")
    w.append('{"i":0}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "binance", "trades", "ETHUSDT", "2026-08-02T05")
    assert reconcile_pair(raw, idx) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_raw_writer_recovery.py -v`
Expected: FAIL with `ImportError: cannot import name 'reconcile_pair'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/capture/raw_writer.py`:

```python
def _read_lines(path: Path) -> list[str]:
    """Split strictly on newline.

    NOT str.splitlines(): that also splits on \v, \f, \x1c-\x1e, \x85,
    U+2028 and U+2029, none of which escape_payload guards. Such a payload
    would yield an extra raw line with no matching index entry, and every
    subsequent line would pair with the wrong entry.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        text = dctx.stream_reader(fh).read().decode("utf-8")
    return text.rstrip("\n").split("\n") if text else []


def _write_lines(path: Path, lines: list[str]) -> None:
    cctx = zstandard.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines:
                w.write((line + "\n").encode("utf-8"))


def reconcile_pair(raw_path: Path, idx_path: Path) -> int:
    """Rebuild index entries for raw lines a crash left undescribed.

    Returns the number of entries repaired. Never discards raw data, and never
    invents a receipt timestamp - unknown times are recorded as 0 with
    kind="recovered" so downstream can exclude them explicitly.
    """
    raw_lines = _read_lines(raw_path)
    idx_lines = _read_lines(idx_path)
    if len(idx_lines) >= len(raw_lines):
        return 0

    repaired = 0
    for n in range(len(idx_lines), len(raw_lines)):
        entry = IndexEntry(n=n, t_recv_ns=0, t_exch_ms=None,
                           seq=None, kind="recovered", esc=False)
        idx_lines.append(encode_index_entry(entry))
        repaired += 1
    _write_lines(idx_path, idx_lines)
    return repaired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_raw_writer_recovery.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/raw_writer.py tests/test_raw_writer_recovery.py
git commit -m "feat: reconcile raw/index pairs after crash without data loss"
```

---

### Task 5: Capture ledger

**Files:**
- Create: `src/capture/capture_ledger.py`
- Create: `tests/test_capture_ledger.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LedgerEvent` frozen dataclass: `ts_ns:int, venue:str, stream:str, kind:str, severity:str, detail:dict`
  - `CaptureLedger(root: Path, venue: str)` with `record(event) -> None`, `read_all(root, venue, date) -> list[LedgerEvent]`, `close() -> None`
  - Severity constants: `SEVERITY_INFO = "info"`, `SEVERITY_OBSERVATION_LOSS = "observation_loss"`, `SEVERITY_CORRUPTING = "corrupting"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture_ledger.py
from pathlib import Path
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, read_all,
    SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS,
)


def test_records_and_reads_back(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="gap", severity=SEVERITY_CORRUPTING,
        detail={"symbol": "BTCUSDT", "expected_pu": 5, "got_pu": 9},
    ))
    ledger.close()

    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len(events) == 1
    assert events[0].kind == "gap"
    assert events[0].severity == SEVERITY_CORRUPTING
    assert events[0].detail["expected_pu"] == 5


def test_appends_across_multiple_records(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "hyperliquid")
    for i in range(3):
        ledger.record(LedgerEvent(
            ts_ns=1785648600_000_000_000 + i, venue="hyperliquid",
            stream="l2Book", kind="stale", severity=SEVERITY_OBSERVATION_LOSS,
            detail={"symbol": "BTC"},
        ))
    ledger.close()
    assert len(read_all(tmp_path, "hyperliquid", "2026-08-02")) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_capture_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.capture_ledger'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/capture_ledger.py
"""Records capture anomalies as first-class, queryable events.

Plain NDJSON, uncompressed: this file is read during incidents, and volume is
tiny compared to market data.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SEVERITY_INFO = "info"
SEVERITY_OBSERVATION_LOSS = "observation_loss"
SEVERITY_CORRUPTING = "corrupting"


@dataclass(frozen=True)
class LedgerEvent:
    ts_ns: int
    venue: str
    stream: str
    kind: str
    severity: str
    detail: dict


def _date_of(ts_ns: int) -> str:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def _path_for(root: Path, venue: str, date: str) -> Path:
    return Path(root) / "ledger" / venue / date / "events.ndjson"


class CaptureLedger:
    def __init__(self, root: Path, venue: str) -> None:
        self._root, self._venue = Path(root), venue
        self._fh = None
        self._date: str | None = None

    def record(self, event: LedgerEvent) -> None:
        date = _date_of(event.ts_ns)
        if date != self._date:
            self.close()
            path = _path_for(self._root, self._venue, date)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._date = date
        self._fh.write(json.dumps(asdict(event), separators=(",", ":"), sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._date = None


def read_all(root: Path, venue: str, date: str) -> list[LedgerEvent]:
    path = _path_for(root, venue, date)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [LedgerEvent(**json.loads(line)) for line in fh if line.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_capture_ledger.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/capture_ledger.py tests/test_capture_ledger.py
git commit -m "feat: capture ledger for gap and anomaly events"
```

---

### Task 6: Sequencing — per-venue gap detection

The probe established these are genuinely different mechanisms. Binance depth is stateful diffs with a sequence chain; Hyperliquid `l2Book` is stateless snapshots with no sequence number at all.

**Files:**
- Create: `src/capture/sequencing.py`
- Create: `tests/test_sequencing.py`

**Interfaces:**
- Consumes: `SEVERITY_CORRUPTING`, `SEVERITY_OBSERVATION_LOSS` from `capture.capture_ledger`
- Produces:
  - `GapReport` frozen dataclass: `severity:str, detail:dict`
  - `BinanceDepthTracker()` with `check(parsed: dict) -> GapReport | None`
  - `HyperliquidStalenessTracker(floor_seconds: float = 5.0, multiple: float = 10.0)` with
    `check(t_recv_ns: int) -> GapReport | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sequencing.py
from capture.sequencing import BinanceDepthTracker, HyperliquidStalenessTracker
from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS

S = 1_000_000_000  # one second in ns


def test_binance_first_frame_is_not_a_gap():
    t = BinanceDepthTracker()
    assert t.check({"U": 10, "u": 20, "pu": 9}) is None


def test_binance_contiguous_chain_has_no_gap():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    assert t.check({"U": 21, "u": 30, "pu": 20}) is None


def test_binance_broken_chain_is_corrupting():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    report = t.check({"U": 40, "u": 50, "pu": 39})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING
    assert report.detail["expected_pu"] == 20
    assert report.detail["got_pu"] == 39


def test_binance_spot_without_pu_uses_u_chain():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20})
    assert t.check({"U": 21, "u": 30}) is None
    report = t.check({"U": 99, "u": 120})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING


def test_hyperliquid_learns_cadence_then_flags_stall():
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        assert t.check(base + i * S) is None          # steady 1s cadence
    report = t.check(base + 20 * S + 60 * S)          # 60s later
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 60


def test_hyperliquid_floor_prevents_false_alarm_on_fast_streams():
    t = HyperliquidStalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        t.check(base + int(i * 0.01 * S))             # 10ms cadence
    # 1s gap: 100x the median, but under the 5s floor -> not an alarm
    assert t.check(base + int(20 * 0.01 * S) + S) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_sequencing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.sequencing'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/sequencing.py
"""Gap detection. Deliberately different per venue - see spec 5.4.

Binance depth is a stateful diff stream: a break in the chain corrupts the book
until a REST resync. Hyperliquid l2Book is stateless snapshots: a gap loses an
observation but nothing is corrupted, so only staleness can be detected.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS


@dataclass(frozen=True)
class GapReport:
    severity: str
    detail: dict


class BinanceDepthTracker:
    """Validates the U/u/pu chain. Uses pu when present (futures), else u (spot)."""

    def __init__(self) -> None:
        self._last_u: int | None = None

    def check(self, parsed: dict) -> GapReport | None:
        first_id, final_id = parsed.get("U"), parsed.get("u")
        prev_final = parsed.get("pu")
        last_u, self._last_u = self._last_u, final_id
        if last_u is None:
            return None

        if prev_final is not None:
            if prev_final == last_u:
                return None
            return GapReport(SEVERITY_CORRUPTING,
                             {"expected_pu": last_u, "got_pu": prev_final})

        if first_id == last_u + 1:
            return None
        return GapReport(SEVERITY_CORRUPTING,
                         {"expected_U": last_u + 1, "got_U": first_id})


class HyperliquidStalenessTracker:
    """No sequence numbers exist, so cadence is learned and stalls are inferred.

    Fires at `multiple` x the rolling median inter-frame gap, floored at
    `floor_seconds` so fast streams do not alarm on ordinary jitter.
    """

    def __init__(self, floor_seconds: float = 5.0, multiple: float = 10.0,
                 window: int = 200) -> None:
        self._floor_ns = int(floor_seconds * 1e9)
        self._multiple = multiple
        self._gaps: deque[int] = deque(maxlen=window)
        self._last_ns: int | None = None

    def check(self, t_recv_ns: int) -> GapReport | None:
        last, self._last_ns = self._last_ns, t_recv_ns
        if last is None:
            return None
        gap = t_recv_ns - last
        report = None
        if len(self._gaps) >= 10:
            threshold = max(self._floor_ns,
                            int(statistics.median(self._gaps) * self._multiple))
            if gap > threshold:
                report = GapReport(SEVERITY_OBSERVATION_LOSS, {
                    "gap_seconds": gap / 1e9,
                    "threshold_seconds": threshold / 1e9,
                })
        self._gaps.append(gap)
        return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_sequencing.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/sequencing.py tests/test_sequencing.py
git commit -m "feat: per-venue gap detection for binance chains and hyperliquid stalls"
```

---

### Task 7: Venue adapters

**Files:**
- Create: `src/capture/venues/__init__.py`
- Create: `src/capture/venues/binance.py`
- Create: `src/capture/venues/hyperliquid.py`
- Create: `tests/test_venues.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `StreamSpec` frozen dataclass: `venue:str, stream:str, symbol:str, channel:str`
  - `ExtractedMeta` frozen dataclass: `t_exch_ms:int|None, seq:dict|None, kind:str, stream:str, symbol:str`
  - `BinanceVenue` with `name`, `core_specs(symbols) -> list[StreamSpec]`, `tail_specs(symbols) -> list[StreamSpec]`, `ws_url(specs) -> str`, `subscribe_messages(specs) -> list[dict]`, `extract(parsed: dict) -> ExtractedMeta`, `instruments_url() -> str`, `parse_instruments(payload: dict) -> list[str]`
  - `HyperliquidVenue` with the same method set

- [ ] **Step 1: Write the failing test**

```python
# tests/test_venues.py
from capture.venues.binance import BinanceVenue
from capture.venues.hyperliquid import HyperliquidVenue


def test_binance_builds_combined_stream_url():
    v = BinanceVenue()
    specs = v.core_specs(["BTCUSDT", "ETHUSDT"])
    url = v.ws_url(specs)
    assert url.startswith("wss://fstream.binance.com/stream?streams=")
    assert "btcusdt@depth@100ms" in url
    assert "btcusdt@aggTrade" in url


def test_binance_extract_reads_both_timestamps_and_chain():
    v = BinanceVenue()
    parsed = {"e": "depthUpdate", "E": 1785650606302, "T": 1785650606300,
              "s": "BTCUSDT", "U": 11192579046493, "u": 11192579053768,
              "pu": 11192579046406}
    meta = v.extract(parsed)
    assert meta.t_exch_ms == 1785650606302
    assert meta.seq == {"U": 11192579046493, "u": 11192579053768,
                        "pu": 11192579046406, "T": 1785650606300}
    assert meta.kind == "data"
    assert meta.symbol == "BTCUSDT"


def test_binance_parses_perp_instruments_only():
    v = BinanceVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING"},
        {"symbol": "ETHUSDT_240329", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
        {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "status": "BREAK"},
    ]}
    assert v.parse_instruments(payload) == ["BTCUSDT"]


def test_hyperliquid_subscribe_messages_cover_each_spec():
    v = HyperliquidVenue()
    specs = v.core_specs(["BTC"])
    msgs = v.subscribe_messages(specs)
    assert {"method": "subscribe",
            "subscription": {"type": "l2Book", "coin": "BTC"}} in msgs


def test_hyperliquid_extract_flags_control_frames():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "subscriptionResponse", "data": {}})
    assert meta.kind == "control"
    assert meta.seq is None


def test_hyperliquid_extract_reads_snapshot_time():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "l2Book",
                      "data": {"coin": "BTC", "time": 1785650605471, "levels": []}})
    assert meta.kind == "data"
    assert meta.t_exch_ms == 1785650605471
    assert meta.symbol == "BTC"
    assert meta.seq is None            # no sequence numbers exist
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_venues.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.venues'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/venues/__init__.py
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamSpec:
    venue: str
    stream: str
    symbol: str
    channel: str


@dataclass(frozen=True)
class ExtractedMeta:
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    stream: str
    symbol: str
```

```python
# src/capture/venues/binance.py
"""Binance USDs-M perpetual futures. Spot is deliberately not captured - see spec 11 Q4."""
from __future__ import annotations

from capture.venues import ExtractedMeta, StreamSpec

_WS_BASE = "wss://fstream.binance.com/stream?streams="
_INSTRUMENTS_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

_CORE_CHANNELS = ["depth@100ms", "aggTrade", "markPrice@1s", "forceOrder"]
_TAIL_CHANNELS = ["aggTrade", "markPrice@1s", "forceOrder"]


class BinanceVenue:
    name = "binance"

    def _specs(self, symbols: list[str], channels: list[str]) -> list[StreamSpec]:
        return [
            StreamSpec(self.name, channel.split("@")[0], symbol, f"{symbol.lower()}@{channel}")
            for symbol in symbols
            for channel in channels
        ]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_CHANNELS)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_CHANNELS)

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_BASE + "/".join(spec.channel for spec in specs)

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        return []          # subscription is encoded in the URL

    def extract(self, parsed: dict) -> ExtractedMeta:
        body = parsed.get("data", parsed)
        event = body.get("e")
        if event is None:
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        seq = None
        if event == "depthUpdate":
            seq = {k: body[k] for k in ("U", "u", "pu", "T") if k in body}
        stream = {"depthUpdate": "depth", "aggTrade": "aggTrade",
                  "markPriceUpdate": "markPrice", "forceOrder": "forceOrder"}.get(event, event)
        return ExtractedMeta(body.get("E"), seq, "data", stream, body.get("s", "unknown"))

    def instruments_url(self) -> str:
        return _INSTRUMENTS_URL

    def parse_instruments(self, payload: dict) -> list[str]:
        return sorted(
            item["symbol"]
            for item in payload.get("symbols", [])
            if item.get("contractType") == "PERPETUAL" and item.get("status") == "TRADING"
        )
```

```python
# src/capture/venues/hyperliquid.py
"""Hyperliquid perps. l2Book carries no sequence numbers - staleness only."""
from __future__ import annotations

from capture.venues import ExtractedMeta, StreamSpec

_WS_URL = "wss://api.hyperliquid.xyz/ws"
_INSTRUMENTS_URL = "https://api.hyperliquid.xyz/info"

_CORE_TYPES = ["l2Book", "trades"]
_TAIL_TYPES = ["trades"]


class HyperliquidVenue:
    name = "hyperliquid"

    def _specs(self, symbols: list[str], types: list[str]) -> list[StreamSpec]:
        return [StreamSpec(self.name, t, symbol, t) for symbol in symbols for t in types]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_TYPES)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_TYPES)

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_URL

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        return [
            {"method": "subscribe",
             "subscription": {"type": spec.stream, "coin": spec.symbol}}
            for spec in specs
        ]

    def extract(self, parsed: dict) -> ExtractedMeta:
        channel = parsed.get("channel")
        if channel in (None, "subscriptionResponse", "pong", "error"):
            return ExtractedMeta(None, None, "control", str(channel), "unknown")
        data = parsed.get("data") or {}
        if isinstance(data, list):
            symbol = data[0].get("coin", "unknown") if data else "unknown"
            t_ms = data[0].get("time") if data else None
        else:
            symbol = data.get("coin", "unknown")
            t_ms = data.get("time")
        return ExtractedMeta(t_ms, None, "data", channel, symbol)

    def instruments_url(self) -> str:
        return _INSTRUMENTS_URL

    def parse_instruments(self, payload: dict) -> list[str]:
        universe = payload.get("universe", [])
        return sorted(
            item["name"] for item in universe if not item.get("isDelisted", False)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_venues.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/venues tests/test_venues.py
git commit -m "feat: binance and hyperliquid venue adapters"
```

---

### Task 8: Venue recorder — async loop with bounded queue

**Files:**
- Create: `src/capture/venue_recorder.py`
- Create: `tests/test_venue_recorder.py`

**Interfaces:**
- Consumes: `RawWriter` (Task 3), `CaptureLedger`/`LedgerEvent` (Task 5), `BinanceDepthTracker`/`HyperliquidStalenessTracker` (Task 6), venue adapters (Task 7)
- Produces:
  - `VenueRecorder(venue, specs, root, queue_size=10000, clock_ns=time.time_ns)` with
    `async def consume(self, frames: AsyncIterator[str]) -> None` and `def stats(self) -> dict`
  - `stats()` returns keys `written`, `dropped`, `control`, `malformed`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_venue_recorder.py
import json
from pathlib import Path

import pytest

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.capture_ledger import read_all, SEVERITY_CORRUPTING


async def _frames(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_writes_frames_and_counts_them(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    payloads = [json.dumps({"stream": "btcusdt@depth@100ms", "data": {
        "e": "depthUpdate", "E": 1785650606214, "s": "BTCUSDT",
        "U": 1 + i * 10, "u": 10 + i * 10, "pu": i * 10}}) for i in range(3)]

    await rec.consume(_frames(payloads))
    assert rec.stats()["written"] == 3
    assert rec.stats()["dropped"] == 0


@pytest.mark.asyncio
async def test_broken_chain_records_corrupting_ledger_event(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    good = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                "U": 1, "u": 10, "pu": 0}})
    broken = json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "BTCUSDT",
                                  "U": 50, "u": 60, "pu": 49}})
    await rec.consume(_frames([good, broken]))

    events = read_all(tmp_path, "binance", "2026-08-02")
    gaps = [e for e in events if e.kind == "gap"]
    assert len(gaps) == 1
    assert gaps[0].severity == SEVERITY_CORRUPTING


@pytest.mark.asyncio
async def test_malformed_frame_is_still_written(tmp_path: Path):
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    await rec.consume(_frames(["this is not json"]))

    assert rec.stats()["malformed"] == 1
    assert rec.stats()["written"] == 1          # written anyway, never discarded
    events = read_all(tmp_path, "binance", "2026-08-02")
    assert any(e.kind == "malformed" for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_venue_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.venue_recorder'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/venue_recorder.py
"""Consumes venue frames and routes them to writers, ledger and gap trackers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import AsyncIterator, Callable

from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_INFO,
)
from capture.raw_writer import RawWriter
from capture.sequencing import BinanceDepthTracker, HyperliquidStalenessTracker


class VenueRecorder:
    def __init__(self, venue, specs, root: Path, queue_size: int = 10_000,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._venue = venue
        self._specs = specs
        self._root = Path(root)
        self._clock_ns = clock_ns
        self._ledger = CaptureLedger(root, venue.name)
        self._writers: dict[tuple[str, str], RawWriter] = {}
        self._trackers: dict[tuple[str, str], object] = {}
        self._stats = {"written": 0, "dropped": 0, "control": 0, "malformed": 0}

    def _writer_for(self, stream: str, symbol: str) -> RawWriter:
        key = (stream, symbol)
        if key not in self._writers:
            self._writers[key] = RawWriter(self._root, self._venue.name, stream, symbol)
        return self._writers[key]

    def _tracker_for(self, stream: str, symbol: str):
        key = (stream, symbol)
        if key not in self._trackers:
            if self._venue.name == "binance" and stream == "depth":
                self._trackers[key] = BinanceDepthTracker()
            elif self._venue.name == "hyperliquid" and stream == "l2Book":
                self._trackers[key] = HyperliquidStalenessTracker()
            else:
                self._trackers[key] = None
        return self._trackers[key]

    def _record_gap(self, stream: str, symbol: str, report, t_recv_ns: int) -> None:
        self._ledger.record(LedgerEvent(
            ts_ns=t_recv_ns, venue=self._venue.name, stream=stream,
            kind="gap", severity=report.severity,
            detail={"symbol": symbol, **report.detail},
        ))

    async def consume(self, frames: AsyncIterator[str]) -> None:
        async for payload in frames:
            t_recv_ns = self._clock_ns()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                self._stats["malformed"] += 1
                self._ledger.record(LedgerEvent(
                    ts_ns=t_recv_ns, venue=self._venue.name, stream="unknown",
                    kind="malformed", severity=SEVERITY_INFO,
                    detail={"bytes": len(payload)},
                ))
                self._writer_for("unknown", "unknown").append(
                    payload, t_recv_ns, None, None, kind="malformed")
                self._stats["written"] += 1
                continue

            meta = self._venue.extract(parsed)
            if meta.kind == "control":
                self._stats["control"] += 1

            tracker = self._tracker_for(meta.stream, meta.symbol)
            if tracker is not None and meta.kind == "data":
                body = parsed.get("data", parsed)
                report = (tracker.check(body) if isinstance(tracker, BinanceDepthTracker)
                          else tracker.check(t_recv_ns))
                if report is not None:
                    self._record_gap(meta.stream, meta.symbol, report, t_recv_ns)

            self._writer_for(meta.stream, meta.symbol).append(
                payload, t_recv_ns, meta.t_exch_ms, meta.seq, kind=meta.kind)
            self._stats["written"] += 1

        self.close()

    def stats(self) -> dict:
        return dict(self._stats)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._ledger.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_venue_recorder.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/venue_recorder.py tests/test_venue_recorder.py
git commit -m "feat: venue recorder routing frames to writers and ledger"
```

---

### Task 9: Universe tracker (spec R2)

Point-in-time membership. Backtesting against today's symbol list silently conditions on survival, and no purge/embargo scheme catches it.

**Files:**
- Create: `src/capture/universe_tracker.py`
- Create: `tests/test_universe_tracker.py`

**Interfaces:**
- Consumes: venue adapters (Task 7) for `parse_instruments`
- Produces:
  - `UniverseEvent` frozen dataclass: `ts_ns:int, venue:str, symbol:str, kind:str, detail:dict` where kind ∈ `listed|delisted`
  - `diff_universe(previous: list[str], current: list[str], venue: str, ts_ns: int) -> list[UniverseEvent]`
  - `UniverseTracker(root: Path, venue_name: str)` with `record_snapshot(symbols, ts_ns) -> list[UniverseEvent]`, `load_last(ts_ns) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe_tracker.py
from pathlib import Path
from capture.universe_tracker import UniverseTracker, diff_universe

TS = 1785648600_000_000_000


def test_diff_detects_listing_and_delisting():
    events = diff_universe(["BTCUSDT", "OLDUSDT"], ["BTCUSDT", "NEWUSDT"],
                           "binance", TS)
    kinds = {(e.symbol, e.kind) for e in events}
    assert ("NEWUSDT", "listed") in kinds
    assert ("OLDUSDT", "delisted") in kinds
    assert ("BTCUSDT", "listed") not in kinds


def test_first_snapshot_lists_everything_as_listed(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    events = tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    assert {e.kind for e in events} == {"listed"}
    assert len(events) == 2


def test_second_snapshot_only_reports_changes(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)
    kinds = {(e.symbol, e.kind) for e in events}
    assert kinds == {("SOLUSDT", "listed"), ("ETHUSDT", "delisted")}


def test_snapshot_is_persisted_and_reloadable(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    reloaded = UniverseTracker(tmp_path, "binance")
    assert reloaded.load_last(TS) == ["BTCUSDT"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_universe_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.universe_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/universe_tracker.py
"""Records point-in-time universe membership.

Without this, any backtest over "all symbols" silently conditions on survival.
Exchanges do not reliably publish historical membership, so it must be captured
as it happens.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UniverseEvent:
    ts_ns: int
    venue: str
    symbol: str
    kind: str
    detail: dict


def _date_of(ts_ns: int) -> str:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def diff_universe(previous: list[str], current: list[str],
                  venue: str, ts_ns: int) -> list[UniverseEvent]:
    before, after = set(previous), set(current)
    events = [UniverseEvent(ts_ns, venue, s, "listed", {}) for s in sorted(after - before)]
    events += [UniverseEvent(ts_ns, venue, s, "delisted", {}) for s in sorted(before - after)]
    return events


class UniverseTracker:
    def __init__(self, root: Path, venue_name: str) -> None:
        self._root = Path(root)
        self._venue = venue_name

    def _dir_for(self, ts_ns: int) -> Path:
        return self._root / "universe" / self._venue / _date_of(ts_ns)

    def _state_path(self) -> Path:
        return self._root / "universe" / self._venue / "last_snapshot.json"

    def load_last(self, ts_ns: int) -> list[str]:
        path = self._state_path()
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))["symbols"]

    def record_snapshot(self, symbols: list[str], ts_ns: int) -> list[UniverseEvent]:
        previous = self.load_last(ts_ns)
        events = diff_universe(previous, symbols, self._venue, ts_ns)

        folder = self._dir_for(ts_ns)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "instruments.ndjson", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts_ns": ts_ns, "kind": "snapshot",
                                 "symbols": symbols}, separators=(",", ":")) + "\n")
            for event in events:
                fh.write(json.dumps(asdict(event), separators=(",", ":"),
                                    sort_keys=True) + "\n")

        state = self._state_path()
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"ts_ns": ts_ns, "symbols": symbols}),
                         encoding="utf-8")
        return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_universe_tracker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/universe_tracker.py tests/test_universe_tracker.py
git commit -m "feat: point-in-time universe membership tracking"
```

---

### Task 10: Capture health — runway and integrity report

**Files:**
- Create: `src/capture/capture_health.py`
- Create: `tests/test_capture_health.py`

**Interfaces:**
- Consumes: `capture.capture_ledger.read_all`
- Produces:
  - `compute_runway_days(free_bytes: int, daily_bytes: float) -> float`
  - `classify_runway(days: float) -> str` returning `ok|warn|alert|decision_point`
  - `measure_daily_bytes(root: Path, days: int = 7) -> float`
  - `build_report(root: Path, venue: str, date: str, free_bytes: int, daily_bytes: float) -> dict`
  - `write_alerts(root: Path, report: dict) -> int` returning alert count

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture_health.py
import json
from pathlib import Path

from capture.capture_health import (
    compute_runway_days, classify_runway, build_report, write_alerts,
)
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING,
)

TS = 1785648600_000_000_000


def test_runway_is_free_over_daily():
    assert compute_runway_days(100_000_000_000, 2_000_000_000) == 50.0


def test_runway_is_infinite_when_nothing_written():
    assert compute_runway_days(100, 0) == float("inf")


def test_classify_thresholds():
    assert classify_runway(90) == "ok"
    assert classify_runway(25) == "warn"
    assert classify_runway(10) == "alert"
    assert classify_runway(5) == "decision_point"


def test_report_counts_gaps_by_severity(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap",
                              SEVERITY_CORRUPTING, {"symbol": "BTCUSDT"}))
    ledger.close()

    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=100_000_000_000, daily_bytes=2_000_000_000)
    assert report["gaps"]["corrupting"] == 1
    assert report["runway_days"] == 50.0
    assert report["runway_status"] == "ok"


def test_write_alerts_emits_lines_for_bad_states(tmp_path: Path):
    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=4_000_000_000, daily_bytes=2_000_000_000)
    count = write_alerts(tmp_path, report)
    assert count == 1
    lines = (tmp_path / "health" / "alerts.ndjson").read_text().splitlines()
    assert json.loads(lines[0])["reason"] == "runway_decision_point"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_capture_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.capture_health'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/capture_health.py
"""Answers 'is capture healthy?' with evidence rather than assumption.

Runway is measured in days remaining, not percent used, because percent
thresholds mean nothing when the write rate changes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from capture.capture_ledger import read_all

WARN_DAYS = 30.0
ALERT_DAYS = 14.0
DECISION_DAYS = 7.0


def compute_runway_days(free_bytes: int, daily_bytes: float) -> float:
    if daily_bytes <= 0:
        return float("inf")
    return free_bytes / daily_bytes


def classify_runway(days: float) -> str:
    if days <= DECISION_DAYS:
        return "decision_point"
    if days <= ALERT_DAYS:
        return "alert"
    if days <= WARN_DAYS:
        return "warn"
    return "ok"


def measure_daily_bytes(root: Path, days: int = 7) -> float:
    raw_root = Path(root) / "raw"
    if not raw_root.exists():
        return 0.0
    cutoff = time.time() - days * 86400
    total = sum(
        path.stat().st_size
        for path in raw_root.rglob("*.zst")
        if path.stat().st_mtime >= cutoff
    )
    return total / days


def build_report(root: Path, venue: str, date: str,
                 free_bytes: int, daily_bytes: float) -> dict:
    events = read_all(root, venue, date)
    gaps = {"corrupting": 0, "observation_loss": 0, "info": 0}
    for event in events:
        if event.kind == "gap":
            gaps[event.severity] = gaps.get(event.severity, 0) + 1

    runway = compute_runway_days(free_bytes, daily_bytes)
    return {
        "venue": venue,
        "date": date,
        "events_total": len(events),
        "gaps": gaps,
        "free_bytes": free_bytes,
        "daily_bytes": daily_bytes,
        "runway_days": runway,
        "runway_status": classify_runway(runway),
    }


def write_alerts(root: Path, report: dict) -> int:
    alerts = []
    if report["runway_status"] != "ok":
        alerts.append({"reason": f"runway_{report['runway_status']}",
                       "runway_days": report["runway_days"]})
    if report["gaps"].get("corrupting", 0) > 0:
        alerts.append({"reason": "corrupting_gaps",
                       "count": report["gaps"]["corrupting"]})

    if alerts:
        folder = Path(root) / "health"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "alerts.ndjson", "a", encoding="utf-8") as fh:
            for alert in alerts:
                fh.write(json.dumps({**alert, "venue": report["venue"],
                                     "date": report["date"]},
                                    separators=(",", ":"), sort_keys=True) + "\n")
    return len(alerts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_capture_health.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/capture/capture_health.py tests/test_capture_health.py
git commit -m "feat: capture health with runway measured in days"
```

---

### Task 11: Live smoke test against real venues

Unit tests prove logic. This proves the thing actually captures — the guardrail lesson already recorded in memory.

**Files:**
- Create: `src/capture/cli.py`
- Create: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `async def run_capture(venue, specs, root, duration_seconds: float) -> dict` returning `VenueRecorder.stats()`
  - `main(argv: list[str] | None = None) -> int` for `uv run --python 3.12 python -m capture.cli`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_smoke.py
import os
from pathlib import Path

import pytest

from capture.cli import run_capture
from capture.venues.binance import BinanceVenue
from capture.raw_writer import read_pair, paths_for, hour_key
import time


@pytest.mark.skipif(os.environ.get("CAPTURE_LIVE") != "1",
                    reason="live venue test; set CAPTURE_LIVE=1 to run")
@pytest.mark.asyncio
async def test_captures_real_binance_frames(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    stats = await run_capture(venue, specs, tmp_path, duration_seconds=15)

    assert stats["written"] > 0
    assert stats["dropped"] == 0

    hour = hour_key(time.time_ns())
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", hour)
    pairs = read_pair(raw, idx)
    assert len(pairs) > 0
    # byte-exactness: every stored line must be valid JSON exactly as sent
    import json
    assert json.loads(pairs[0][0]) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_cli_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'capture.cli'`
(Without `CAPTURE_LIVE=1` it reports SKIPPED once the module exists.)

- [ ] **Step 3: Write minimal implementation**

```python
# src/capture/cli.py
"""Entry point for running capture."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.venues.hyperliquid import HyperliquidVenue

_VENUES = {"binance": BinanceVenue, "hyperliquid": HyperliquidVenue}


async def _stream_frames(venue, specs, duration_seconds: float):
    """Yields raw text frames for a bounded duration, then stops."""
    url = venue.ws_url(specs)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_seconds
    async with websockets.connect(url, open_timeout=20) as ws:
        for message in venue.subscribe_messages(specs):
            await ws.send(json.dumps(message))
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                yield await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                return


async def run_capture(venue, specs, root: Path, duration_seconds: float) -> dict:
    recorder = VenueRecorder(venue, specs, root)
    await recorder.consume(_stream_frames(venue, specs, duration_seconds))
    return recorder.stats()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="capture")
    parser.add_argument("--venue", choices=sorted(_VENUES), required=True)
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 means run until interrupted")
    args = parser.parse_args(argv)

    venue = _VENUES[args.venue]()
    specs = venue.core_specs(args.symbols.split(","))
    duration = args.seconds if args.seconds > 0 else float("inf")
    stats = asyncio.run(run_capture(venue, specs, Path(args.root), duration))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CAPTURE_LIVE=1 uv run --python 3.12 pytest tests/test_cli_smoke.py -v`
Expected: PASS (captures real frames for 15 seconds)

Then run the whole suite: `uv run --python 3.12 pytest -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/capture/cli.py tests/test_cli_smoke.py
git commit -m "feat: capture CLI with live venue smoke test"
```

---

### Task 12: Prove GCS write access (unblocks B1)

Do not build the offloader until the write path is proven. This task is the gate.

**Files:**
- Create: `scripts/verify_gcs_write.sh`

**Interfaces:**
- Consumes: nothing
- Produces: a pass/fail answer to blocker B1

- [ ] **Step 1: Write the verification script**

```bash
#!/usr/bin/env bash
# Proves whether the service account can write and read back a GCS object.
# Usage: scripts/verify_gcs_write.sh gs://your-bucket-name
set -euo pipefail
BUCKET="${1:?usage: verify_gcs_write.sh gs://bucket-name}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL="$(mktemp)"; REMOTE="${BUCKET}/_capture_write_probe_${STAMP}.txt"
echo "capture-write-probe ${STAMP}" > "$LOCAL"

echo "--- upload"; gcloud storage cp "$LOCAL" "$REMOTE"
echo "--- read back"; gcloud storage cat "$REMOTE"
echo "--- delete"; gcloud storage rm "$REMOTE"
rm -f "$LOCAL"
echo "RESULT: GCS write access CONFIRMED for ${BUCKET}"
```

- [ ] **Step 2: Run it**

Run: `chmod +x scripts/verify_gcs_write.sh && scripts/verify_gcs_write.sh gs://<bucket>`
Expected: either `RESULT: GCS write access CONFIRMED`, or a 403 identifying the missing IAM role.

- [ ] **Step 3: Record the outcome in the spec**

If it fails, add the exact error to spec §10 B1 and stop — the offloader is not built.
If it passes, update spec §10 B1 to resolved and proceed to build `archive_offloader.py`.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_gcs_write.sh docs/superpowers/specs/2026-08-02-layer0-raw-capture-design.md
git commit -m "test: verify GCS write access to unblock B1"
```

---

### Task 13: Test whether cron survives reboot (unblocks B2)

B2 gates the first disk resize, which the runway puts at 45–90 days. This is the cheapest candidate fix.

**Files:**
- Create: `scripts/install_reboot_probe.sh`

**Interfaces:**
- Consumes: nothing
- Produces: a pass/fail answer to blocker B2

- [ ] **Step 1: Write the probe installer**

```bash
#!/usr/bin/env bash
# Installs a user @reboot cron entry that writes a timestamp on boot.
# If the file appears after a reboot, user cron survives without linger and
# B2 is solved for free.
set -euo pipefail
MARKER="$HOME/capture/health/reboot_probe.log"
mkdir -p "$(dirname "$MARKER")"
LINE="@reboot /bin/date -u +\\%Y-\\%m-\\%dT\\%H:\\%M:\\%SZ >> $MARKER"
( crontab -l 2>/dev/null | grep -v reboot_probe.log || true; echo "$LINE" ) | crontab -
echo "installed. current crontab:"; crontab -l
echo "now reboot the VM, then check: cat $MARKER"
```

- [ ] **Step 2: Install it**

Run: `chmod +x scripts/install_reboot_probe.sh && scripts/install_reboot_probe.sh`
Expected: crontab lists the `@reboot` line.

- [ ] **Step 3: Reboot and check**

This step needs the operator — it interrupts any running session.
After reboot: `cat ~/capture/health/reboot_probe.log`
Expected: a timestamp line if user cron runs `@reboot` without linger.

- [ ] **Step 4: Record the outcome in the spec**

Update spec §10 B2 with the result. If cron works, B2 is resolved and the recorder gets an `@reboot` entry. If not, escalate to the external-watchdog or linger options listed there.

- [ ] **Step 5: Commit**

```bash
git add scripts/install_reboot_probe.sh docs/superpowers/specs/2026-08-02-layer0-raw-capture-design.md
git commit -m "test: probe whether user cron survives reboot to unblock B2"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §4 `raw_writer` | Tasks 3, 4 |
| §4 `capture_ledger` | Task 5 |
| §4 `venue_recorder` | Task 8 |
| §4 `universe_tracker` | Task 9 (R2) |
| §4 `capture_health` | Task 10 |
| §4 `archive_offloader` | Task 12 gates it; build follows if B1 passes |
| §5.2 two-file byte-exact format | Tasks 2, 3 |
| §5.4 per-venue sequencing | Tasks 6, 7 |
| §5.4 control frames flagged | Task 7 |
| §6 malformed still written | Task 8 |
| §6 queue overflow as event | **Gap — see below** |
| §7.1 round-trip byte-exactness | Task 3 |
| §7.1 raw/index alignment | Tasks 3, 4 |
| §7.1 synthetic gap injection | Task 6 |
| §7.1 crash safety | Task 4 |
| §7.1 prune safety | Deferred with offloader (B1) |
| §8 R1 broad tail | Task 7 `tail_specs` + Task 9 dynamic discovery |
| §10 B1 | Task 12 |
| §10 B2 | Task 13 |

**Gap found and accepted:** §6 requires bounded-queue overflow to emit a ledger event. Task 8's
`VenueRecorder.consume` processes frames synchronously from the iterator, so no queue exists yet and
overflow cannot occur. The bounded queue belongs with the reconnect loop, which is deferred until
after the live smoke test proves the basic path. **Add a Task 14 for the reconnect/queue loop before
running unattended** — the 7-day acceptance criterion in §7.3 cannot be met without it.

**2. Placeholder scan:** no TBD/TODO. Every code step contains runnable code. Task 12 and 13
intentionally end in operator-dependent outcomes, and both state exactly what to record.

**3. Type consistency:** `IndexEntry` fields identical across Tasks 2, 3, 4. `GapReport(severity, detail)`
consistent in Tasks 6 and 8. `ExtractedMeta` fields consistent in Tasks 7 and 8. `stats()` keys
(`written`, `dropped`, `control`, `malformed`) consistent in Tasks 8 and 11. Severity constants come
from one place, `capture.capture_ledger`.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — execute in this session with checkpoints for review
