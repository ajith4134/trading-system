"""Repair must clear markers left by dead processes - and must not cost dark time.

Found 2026-08-08. One `.writing` marker had survived since 2026-08-02T15:42 -
left by the first session that crashed - and nothing had ever removed it. The
consequence was invisible and permanent: the GCS offload correctly refuses to
copy an hour a writer still holds, so that hour was excluded from every backup
for six days, and no check reported it.

`is_hour_being_written` already had the hard part right: it reads the pid out of
the marker and calls a marker whose process is gone stale. `repair_archive`
called it, used the answer to skip live hours, and then never removed the dead
markers it had just identified.

Found 2026-08-09, the reason for the scope tests below. The pass read every pair
in the archive end to end and it ran in front of the recorder, so its cost was
dark time. After a reboot, three venue supervisors each started their own
full-root pass over 22,668 pairs and 1.9 GB; eighteen minutes later all three
were still running and no venue had captured a frame.
"""
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from capture.raw_writer import RawWriter, writing_marker_path
from capture.repair_archive import Scope, hour_of, pairs_to_scan, repair_archive


def written_hour(root: Path, symbol: str = "BTCUSDT", venue: str = "binance",
                 t_recv_ns: int = 1785648600_000_000_000) -> Path:
    """One complete, healthy, closed hour on disk, stamped long ago."""
    writer = RawWriter(root, venue, "trade", symbol)
    writer.append('{"e":"trade","s":"%s"}' % symbol,
                  t_recv_ns=t_recv_ns, t_exch_ms=1785650606214,
                  seq=None)
    writer.close()
    return next((root / "raw" / venue).rglob(f"trade_{symbol}_*.ndjson.zst"))


def hour_in_progress(root: Path, symbol: str = "BTCUSDT", venue: str = "binance") -> Path:
    """An hour stamped now - the one a recorder starting now would open."""
    return written_hour(root, symbol, venue, t_recv_ns=time.time_ns())


def scanned_paths(root: Path, **kwargs) -> list[Path]:
    return [pair[0] for pair in pairs_to_scan(root, **kwargs)]


# --------------------------------------------------------------------------
# markers
# --------------------------------------------------------------------------

def test_a_marker_from_a_dead_process_is_cleared(tmp_path: Path):
    """The exact case that survived six days. The pair is healthy, so the repair
    pass reaches it and moves on - which is why the marker was never even
    looked at."""
    raw = hour_in_progress(tmp_path)
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
    raw = hour_in_progress(tmp_path, symbol="ETHUSDT")
    marker = writing_marker_path(raw)
    marker.write_text(str(os.getpid()), encoding="utf-8")

    repair_archive(tmp_path)

    assert marker.exists(), "cleared a marker a live process still holds"


def test_an_hour_with_no_marker_is_untouched(tmp_path: Path):
    """The overwhelmingly common case must stay free of side effects."""
    raw = hour_in_progress(tmp_path, symbol="SOLUSDT")
    result = repair_archive(tmp_path)

    assert raw.exists()
    assert result["cleared_markers"] == []


def test_a_stale_marker_is_swept_however_old_the_hour_is(tmp_path: Path):
    """The six-day-old marker is the case that must survive the scope bound.

    Its danger is precisely its age - it kept the offload from ever copying that
    hour, silently - so age cannot be what excludes it from the sweep. This is
    the regression a naive "only scan the current hour" would introduce, and it
    would be as invisible as the original was.
    """
    raw = written_hour(tmp_path, symbol="ANCIENT")
    # Established before the marker exists, because the marker is what puts the
    # pair back in scope - which is the whole claim. Without this, the assertion
    # below would pass just as well against a scope that included the fixture
    # anyway, and would prove nothing.
    assert raw not in scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    marker = writing_marker_path(raw)
    marker.write_text("4194304", encoding="utf-8")

    result = repair_archive(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    assert not marker.exists(), "the scope bound swallowed a stale marker"
    assert any("trade_ANCIENT" in str(p) for p in result["cleared_markers"])


def test_a_marked_hour_in_another_venue_is_still_swept(tmp_path: Path):
    """The sweep is unscoped in both dimensions for one reason: a marker nobody
    sweeps is an hour nobody backs up."""
    raw = written_hour(tmp_path, symbol="ETH", venue="hyperliquid")
    marker = writing_marker_path(raw)
    marker.write_text("4194304", encoding="utf-8")

    repair_archive(tmp_path, venue="binance")

    assert not marker.exists(), "a venue bound hid a stale marker from every pass"


# --------------------------------------------------------------------------
# scope: what a blocking pass reads end to end
#
# The pass decompresses every pair it scans and it runs in front of the
# recorder, so its scope IS the dark window after a restart.
# --------------------------------------------------------------------------

def test_the_blocking_pass_reads_only_the_hour_a_recorder_can_be_refused_by(tmp_path: Path):
    """A rotated hour is never appended to again, so a tear in it cannot refuse
    the recorder about to start. Reading it is pure dark time."""
    rotated = written_hour(tmp_path, symbol="OLDUSDT")
    opening = hour_in_progress(tmp_path, symbol="NEWUSDT")

    scanned = scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    assert opening in scanned
    assert rotated not in scanned, "blocked capture on an hour no recorder can resume into"


def test_the_blocking_pass_reads_only_the_venue_it_was_asked_for(tmp_path: Path):
    """Each supervisor repairs the subtree it is about to write into. Unscoped,
    three supervisors did the same whole-archive work three times over."""
    mine = hour_in_progress(tmp_path, symbol="BTCUSDT", venue="binance")
    theirs = hour_in_progress(tmp_path, symbol="BTC", venue="hyperliquid")

    scanned = scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    assert mine in scanned
    assert theirs not in scanned, "repaired a venue no recorder here is waiting on"


def test_the_archive_pass_takes_exactly_what_the_blocking_pass_leaves(tmp_path: Path):
    """Disjoint, and between them complete. Overlap would be the dangerous half:
    `reconcile_pair` swaps the index inode, so an archive pass that could reach
    the hour a recorder is opening would silently drop every index entry written
    after the swap. Backgrounding it is only safe because it cannot."""
    rotated = written_hour(tmp_path, symbol="OLDUSDT")
    opening = hour_in_progress(tmp_path, symbol="NEWUSDT")

    blocking = set(scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE))
    background = set(scanned_paths(tmp_path, venue="binance", scope=Scope.ARCHIVE))

    assert blocking & background == set(), "the background pass can reach a live hour"
    assert blocking | background == {rotated, opening}
    assert rotated in background and opening in blocking


def test_the_whole_archive_pass_is_still_available_when_asked_for(tmp_path: Path):
    """`all` keeps the deep pass as a deliberate act rather than deleting the
    capability - it is the right thing after a disk scare, and the wrong thing
    on every restart."""
    rotated = written_hour(tmp_path, symbol="OLDUSDT")
    opening = hour_in_progress(tmp_path, symbol="NEWUSDT")

    scanned = scanned_paths(tmp_path, venue="binance", scope=Scope.ALL)

    assert {rotated, opening} <= set(scanned)


def test_a_pair_whose_hour_cannot_be_read_is_left_to_the_archive_pass(tmp_path: Path):
    """A writer always names through `paths_for`, which always stamps an hour, so
    a name without one is not a name any recorder will open. It still gets
    repaired - off the blocking path, where a slow surprise costs nothing."""
    odd = tmp_path / "raw" / "binance" / "2026-08-02" / "trade_BTCUSDT_nohour.ndjson.zst"
    odd.parent.mkdir(parents=True, exist_ok=True)
    odd.write_bytes(b"")

    assert hour_of(odd) is None
    assert odd not in scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE)
    assert odd in scanned_paths(tmp_path, venue="binance", scope=Scope.ARCHIVE)


def test_the_hour_boundary_is_inclusive_at_the_hour_that_just_opened(tmp_path: Path):
    """An hour stamped exactly on the boundary is the one being opened. Off by
    one here skips the current hour on every pass, which is invisible until the
    one restart where it mattered."""
    now = datetime(2026, 8, 9, 12, 30, tzinfo=timezone.utc)
    boundary = now.replace(minute=0)
    raw = written_hour(tmp_path, symbol="EDGEUSDT",
                       t_recv_ns=int(boundary.timestamp()) * 1_000_000_000)

    assert hour_of(raw) == boundary
    assert raw in scanned_paths(tmp_path, venue="binance", scope=Scope.RESUMABLE, now=now)
    # And the hour before it is not - that one has rotated.
    earlier = written_hour(tmp_path, symbol="PRIORUSDT",
                           t_recv_ns=int((boundary - timedelta(hours=1)).timestamp()) * 1_000_000_000)
    assert earlier not in scanned_paths(tmp_path, venue="binance",
                                        scope=Scope.RESUMABLE, now=now)


def test_the_result_reports_what_the_pass_actually_looked_at(tmp_path: Path):
    """A scope bug reads as a quiet, healthy, empty result. The count is what
    makes the difference between "nothing was wrong" and "nothing was read"
    visible in the restart log."""
    hour_in_progress(tmp_path, symbol="BTCUSDT")
    written_hour(tmp_path, symbol="OLDUSDT")

    result = repair_archive(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    assert result["scanned"] == 1
    assert result["scope"] == {"venue": "binance", "hours": "resumable"}


def test_a_torn_hour_in_progress_is_still_repaired_by_the_blocking_pass(tmp_path: Path):
    """The scoping must not cost the thing repair exists for. A recorder refused
    by a torn current hour is the seventeen minutes of lost depth this module was
    written after."""
    raw = hour_in_progress(tmp_path, symbol="TORNUSDT")
    idx = raw.with_name(raw.name[: -len(".ndjson.zst")] + ".idx.zst")
    idx.write_bytes(b"not a zstd frame")

    result = repair_archive(tmp_path, venue="binance", scope=Scope.RESUMABLE)

    assert [r["path"] for r in result["repaired"]] == [str(raw)]
    assert result["failed"] == []
