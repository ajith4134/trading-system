"""Repair must also clear markers left by processes that no longer exist.

Found 2026-08-08. One `.writing` marker had survived since 2026-08-02T15:42 -
left by the first session that crashed - and nothing had ever removed it. The
consequence was invisible and permanent: the GCS offload correctly refuses to
copy an hour a writer still holds, so that hour was excluded from every backup
for six days, and no check reported it.

`is_hour_being_written` already had the hard part right: it reads the pid out of
the marker and calls a marker whose process is gone stale. `repair_archive`
called it, used the answer to skip live hours, and then never removed the dead
markers it had just identified.
"""
import os
from pathlib import Path

from capture.raw_writer import RawWriter, writing_marker_path
from capture.repair_archive import repair_archive


def written_hour(root: Path, symbol: str = "BTCUSDT") -> Path:
    """One complete, healthy, closed hour on disk."""
    writer = RawWriter(root, "binance", "trade", symbol)
    writer.append('{"e":"trade","s":"%s"}' % symbol,
                  t_recv_ns=1785648600_000_000_000, t_exch_ms=1785650606214,
                  seq=None)
    writer.close()
    return next((root / "raw").rglob(f"trade_{symbol}_*.ndjson.zst"))


def test_a_marker_from_a_dead_process_is_cleared(tmp_path: Path):
    """The exact case that survived six days. The pair is healthy, so the repair
    pass reaches it and moves on - which is why the marker was never even
    looked at."""
    raw = written_hour(tmp_path)
    marker = writing_marker_path(raw)
    # A pid that cannot be running. Kernel pids do not go this high by default,
    # and `is_hour_being_written` treats an absent process as stale.
    marker.write_text("4194304", encoding="utf-8")

    result = repair_archive(tmp_path)

    assert not marker.exists(), "a marker whose process is gone must not survive"
    assert result["healthy"] >= 1
    assert any("trade_BTCUSDT" in str(p) for p in result["cleared_markers"])


def test_a_marker_held_by_a_live_process_is_left_alone(tmp_path: Path):
    """Clearing a live marker would let the offload copy a zstd frame mid-write,
    and the copy is a truncated archive that reads as complete. Refusing to
    clear is recoverable; clearing wrongly is not."""
    raw = written_hour(tmp_path, symbol="ETHUSDT")
    marker = writing_marker_path(raw)
    marker.write_text(str(os.getpid()), encoding="utf-8")

    repair_archive(tmp_path)

    assert marker.exists(), "cleared a marker a live process still holds"


def test_an_hour_with_no_marker_is_untouched(tmp_path: Path):
    """The overwhelmingly common case must stay free of side effects."""
    raw = written_hour(tmp_path, symbol="SOLUSDT")
    result = repair_archive(tmp_path)

    assert raw.exists()
    assert result["cleared_markers"] == []
