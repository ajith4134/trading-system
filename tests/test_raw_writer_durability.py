"""Regression tests for a raw/index pair surviving opens, closes and repairs.

Every test here reproduces a defect the pre-existing suite did not constrain:
opens that truncated the hour they were meant to extend, a close order that
guaranteed the one direction reconcile_pair cannot repair, and a repair tool
that orphaned the inode a live writer was still filling.
"""
import builtins
import errno
import os
import stat
import subprocess
from pathlib import Path

import pytest
import zstandard

from capture.raw_writer import (
    RawWriter, read_pair, reconcile_pair, paths_for, _read_lines,
    TruncatedFrameFile, HourStillBeingWritten, HourFileNotAppendable,
    HourHeldByAnotherWriter, writing_marker_path,
)
from capture.frame_codec import IndexEntry, encode_index_entry

HOUR_05 = 1785648600_000_000_000        # 2026-08-02T05:30:00Z
HOUR_06 = 1785652200_000_000_000        # 2026-08-02T06:30:00Z


def _write_zst_lines(path: Path, lines: list[str]) -> None:
    cctx = zstandard.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines:
                w.write((line + "\n").encode("utf-8"))


def _chop_last_byte(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])


def _count_lines_tolerantly(path: Path) -> int:
    """Line count that survives a torn file, for asserting on a damaged pair."""
    if not path.exists():
        return 0
    try:
        return len(_read_lines(path))
    except TruncatedFrameFile as exc:
        return len(exc.recovered_lines)


def _count_zstd_frames(path: Path) -> int:
    """How many concatenated zstd frames the file holds.

    Each one is a boundary crash damage can stop at, so this is the thing the
    flush cadence is actually producing.
    """
    data = path.read_bytes()
    decompressor = zstandard.ZstdDecompressor()
    position = frames = 0
    while position < len(data):
        frame_reader = decompressor.decompressobj()
        frame_reader.decompress(data[position:])
        position += len(data) - position - len(frame_reader.unused_data)
        frames += 1
    return frames


def _write_seconds_apart(writer: RawWriter, count: int, seconds: float = 1.0,
                         start_ns: int = HOUR_05) -> None:
    for i in range(count):
        writer.append(f'{{"frame":{i}}}', t_recv_ns=start_ns + int(i * seconds * 1e9),
                      t_exch_ms=None, seq=None)

# --------------------------------------------------------------------------
# CRITICAL 1 - opening an hour file must never truncate it
# --------------------------------------------------------------------------

def test_restarting_mid_hour_appends_instead_of_truncating(tmp_path: Path):
    """The normal restart path: venue_recorder close()s on every run exit."""
    first = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        first.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    first.close()

    second = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    second.append('{"frame":5}', t_recv_ns=HOUR_05 + 5, t_exch_ms=None, seq=None)
    second.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(6)]
    # n keeps matching position across the restart, or every later entry would
    # describe the wrong frame.
    assert [p[1].n for p in pairs] == list(range(6))
    assert [p[1].t_recv_ns for p in pairs] == [HOUR_05 + i for i in range(6)]


def test_close_then_append_in_the_same_hour_keeps_earlier_frames(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "trades", "ETHUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()                              # close() clears _hour ...
    w.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)
    w.close()                              # ... so this append re-opens the hour

    raw, idx = paths_for(tmp_path, "binance", "trades", "ETHUSDT", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(4)]


def test_out_of_order_timestamp_back_across_an_hour_does_not_destroy_it(tmp_path: Path):
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append('{"frame":"05a"}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.append('{"frame":"05b"}', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    w.append('{"frame":"06a"}', t_recv_ns=HOUR_06, t_exch_ms=None, seq=None)
    # One late frame reaches back over the boundary and re-opens hour 05.
    w.append('{"frame":"05c"}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)
    w.close()

    r5, i5 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    assert [p[0] for p in read_pair(r5, i5)] == [
        '{"frame":"05a"}', '{"frame":"05b"}', '{"frame":"05c"}']
    r6, i6 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T06")
    assert [p[0] for p in read_pair(r6, i6)] == ['{"frame":"06a"}']


def test_a_failed_open_does_not_destroy_the_existing_hour(tmp_path: Path, monkeypatch):
    """An open that never succeeds must not have already destroyed the file."""
    first = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(4):
        first.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    first.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    assert len(raw_bytes_before) > 0

    real_open = builtins.open

    def open_failing_on_idx_append(file, mode="r", *args, **kwargs):
        if str(file).endswith(".idx.zst") and "a" in mode:
            raise OSError(errno.EMFILE, "Too many open files")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_failing_on_idx_append)
    second = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    with pytest.raises(OSError):
        second.append('{"frame":4}', t_recv_ns=HOUR_05 + 4, t_exch_ms=None, seq=None)
    monkeypatch.undo()

    assert raw.read_bytes() == raw_bytes_before, "a failed open truncated the raw file"
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(4)]


def test_resuming_into_a_misaligned_hour_is_refused(tmp_path: Path):
    """Resuming needs the frame count; a damaged pair must not be papered over."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}', '{"frame":1}'])
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    w = RawWriter(tmp_path, "v", "s", "SYM")
    with pytest.raises(HourFileNotAppendable):
        w.append('{"frame":2}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)

    # Refusing must not have altered the data it refused to extend.
    assert _read_lines(raw) == ['{"frame":0}', '{"frame":1}']


# --------------------------------------------------------------------------
# IMPORTANT 5 - close() must not leave the unrepairable direction
# --------------------------------------------------------------------------

def test_close_never_leaves_the_index_longer_than_the_raw_file(tmp_path: Path):
    """reconcile_pair rebuilds index entries from raw lines; the reverse is
    impossible. A close that fails on raw must therefore not go on to give the
    index a full tail."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)

    failing_raw_z = MagicMock(wraps=w._raw_z)
    failing_raw_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._raw_z = failing_raw_z

    with pytest.raises(OSError):
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    raw_count = _count_lines_tolerantly(raw)
    idx_count = _count_lines_tolerantly(idx)
    assert idx_count <= raw_count, (
        f"close() left idx={idx_count} lines against raw={raw_count}: "
        f"the one direction reconcile_pair cannot repair")


def test_close_still_releases_every_descriptor_when_raw_close_fails(tmp_path: Path):
    """Abandoning the index's frame must not mean leaking its handle."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    idx_fh, raw_fh = w._idx_fh, w._raw_fh

    failing_raw_z = MagicMock(wraps=w._raw_z)
    failing_raw_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._raw_z = failing_raw_z

    with pytest.raises(OSError):
        w.close()

    assert idx_fh.closed, "index file handle leaked"
    assert raw_fh.closed, "raw file handle leaked"
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert not writing_marker_path(raw).exists(), "writing marker leaked"


def test_close_leaves_a_repairable_pair_when_the_index_close_fails(tmp_path: Path):
    """The other direction: raw survives whole and repair can rebuild the index."""
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)

    failing_idx_z = MagicMock(wraps=w._idx_z)
    failing_idx_z.close.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._idx_z = failing_idx_z

    with pytest.raises(OSError):
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert _read_lines(raw) == [f'{{"frame":{i}}}' for i in range(5)]
    assert reconcile_pair(raw, idx).entries_rebuilt == 5
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(5)]


# --------------------------------------------------------------------------
# IMPORTANT 6 - repairing a live hour must not orphan the writer's inode
# --------------------------------------------------------------------------

def test_reconcile_refuses_an_hour_a_writer_still_holds_open(tmp_path: Path):
    """_write_lines swaps the inode; a live writer would keep filling the old one."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.flush()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    with pytest.raises(HourStillBeingWritten):
        reconcile_pair(raw, idx)

    for i in range(3, 6):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    # Every frame is still there: the refused repair created no mismatch.
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(6)]


def test_the_writing_marker_is_removed_once_the_hour_is_closed(tmp_path: Path):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert writing_marker_path(raw).exists()
    w.close()
    assert not writing_marker_path(raw).exists()
    assert reconcile_pair(raw, idx).entries_rebuilt == 0   # repair is allowed again


def test_a_stale_marker_from_a_dead_process_does_not_block_repair(tmp_path: Path):
    """A crash leaves the marker behind - exactly when repair is needed most."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    dead = subprocess.Popen(["true"])
    dead.wait()
    writing_marker_path(raw).write_text(str(dead.pid), encoding="utf-8")

    assert reconcile_pair(raw, idx).entries_rebuilt == 2
    assert len(read_pair(raw, idx)) == 3


def test_reconcile_rebuilds_an_index_whose_own_tail_is_torn(tmp_path: Path):
    """The index is rewritten wholesale, so its torn tail is salvageable."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(200):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(idx)
    with pytest.raises(TruncatedFrameFile):
        read_pair(raw, idx)

    outcome = reconcile_pair(raw, idx)
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(200)]
    # No raw frame was discarded to make the index fit.
    assert len(pairs) == 200
    assert outcome.entries_discarded == 0 and not outcome.raw_was_salvaged

    # Surface the real cost rather than resting on "repaired > 0". A torn zstd
    # frame loses a whole compressed block, and an hour written in one open is
    # one block: chopping a single byte off the index costs every timestamp in
    # it. Every frame's bytes survive; every frame's metadata does not.
    assert outcome.entries_rebuilt == 200, (
        "one chopped byte on the index costs all 200 entries, not a tail of them")
    assert all(p[1].kind == "recovered" for p in pairs)
    assert all(p[1].t_recv_ns == 0 for p in pairs)


# --------------------------------------------------------------------------
# CRITICAL - the advertised remedy must terminate: refuse -> repair -> resume
# --------------------------------------------------------------------------

def test_reconcile_repairs_a_torn_raw_file_so_appending_resumes(tmp_path: Path):
    """`HourFileNotAppendable` says "run reconcile_pair"; it has to work.

    Refusing a torn raw file left the operator with no exit at all: append
    refused, reconcile raised on the very file it was named as the remedy for,
    and append refused again. Since VenueRecorder.consume did not guard append,
    that was a permanent crash loop for the whole venue, not just the hour.

    Salvage keeps every complete line, quarantines the original bytes rather
    than destroying them, and leaves the pair appendable.
    """
    # Three sessions, so the hour holds three concatenated zstd frames and the
    # damage is confined to the last one. Salvage granularity is the zstd frame,
    # not the line - see test_a_single_block_hour_loses_all_of_it_to_one_byte.
    for session in range(3):
        w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
        for i in range(4):
            w.append(f'{{"session":{session},"i":{i}}}',
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    _chop_last_byte(raw)                       # the `kill -9` shape

    # 1. Refuse.
    blocked = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    with pytest.raises(HourFileNotAppendable):
        blocked.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)

    # 2. Repair - the step that used to raise TruncatedFrameFile.
    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    assert outcome.raw_frames_kept == 8, "the two intact zstd frames must survive"
    assert outcome.entries_discarded == 4, "the torn frame's 4 entries are unbacked"

    # The damaged original is kept verbatim, not destroyed.
    assert outcome.quarantined_paths
    quarantined_raw = next(p for p in outcome.quarantined_paths
                           if p.name.startswith(raw.name))
    assert quarantined_raw.read_bytes() == raw_bytes_before[:-1]

    # 3. Resume - and this is the assertion that makes the cycle terminate.
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)
    resumed.close()

    pairs = read_pair(raw, idx)
    assert len(pairs) == 9
    assert pairs[-1][0] == '{"frame":"new"}'
    assert [p[1].n for p in pairs] == list(range(9))
    # The salvaged frames are the intact prefix, in order, byte for byte, and
    # they kept the metadata they already had rather than being relabelled.
    assert [p[0] for p in pairs[:8]] == [
        f'{{"session":{s},"i":{i}}}' for s in range(2) for i in range(4)]
    assert [p[1].t_recv_ns for p in pairs[:8]] == [HOUR_05 + i for i in range(8)]


def test_a_single_block_hour_loses_all_of_it_to_one_byte(tmp_path: Path):
    """The honest limit of salvage: it is bounded by zstd's block, not the line.

    400 frames written in one open compress to a single ~370-byte block. Chopping
    one byte off it leaves NOTHING decodable - verified against zstandard 0.25.0
    for decompressobj, stream_reader and chunked stream_reader alike. Salvage
    cannot beat that; only writing more frequent zstd frames could bound it, and
    that is a capture-spec decision about storage cost, not a repair-tool one.

    What repair still guarantees is the part that was broken: the bytes are
    quarantined rather than destroyed, the loss is reported rather than implied,
    and the hour is appendable again.
    """
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    raw_bytes_before = raw.read_bytes()
    _chop_last_byte(raw)

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    assert outcome.raw_frames_kept == 0, "one block, so one byte costs the whole hour"
    assert outcome.entries_discarded == 400
    quarantined_raw = next(p for p in outcome.quarantined_paths
                           if p.name.startswith(raw.name))
    assert quarantined_raw.read_bytes() == raw_bytes_before[:-1], (
        "the unsalvageable bytes must be preserved, not deleted")

    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":400}', t_recv_ns=HOUR_05 + 400, t_exch_ms=None, seq=None)
    resumed.close()
    assert [p[0] for p in read_pair(raw, idx)] == ['{"frame":400}']


def test_reconcile_of_a_torn_raw_file_is_idempotent(tmp_path: Path):
    """A second repair must be a no-op, not another round of salvage."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    first = reconcile_pair(raw, idx)
    second = reconcile_pair(raw, idx)
    assert first.raw_was_salvaged and not second.raw_was_salvaged
    assert second.entries_rebuilt == 0 and second.entries_discarded == 0
    assert second.raw_frames_kept == first.raw_frames_kept
    assert not second.quarantined_paths, "a clean pair must not be quarantined again"


def test_reconcile_salvages_when_both_files_are_torn(tmp_path: Path):
    """The shape a single `kill -9` actually leaves: both tails cut."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)
    _chop_last_byte(idx)

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":400}', t_recv_ns=HOUR_05 + 400, t_exch_ms=None, seq=None)
    resumed.close()
    pairs = read_pair(raw, idx)
    assert pairs[-1][0] == '{"frame":400}'
    assert [p[1].n for p in pairs] == list(range(len(pairs)))


def test_a_repair_interrupted_halfway_can_still_be_finished(tmp_path: Path):
    """Repair runs after a crash, so it has to survive being crashed itself.

    The write order is load-bearing. Writing the salvaged raw file first and
    dying leaves a short raw file beside the original long index; the raw file is
    no longer torn, so the overrunning index is correctly refused as
    UnrepairableIndex - a dead end, which is the trap reconcile_pair exists to
    remove. Writing the index first leaves the torn raw file untouched, and the
    next run redoes the whole salvage.
    """
    import capture.raw_writer as raw_writer_module

    for session in range(3):
        w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
        for i in range(4):
            w.append(f'{{"session":{session},"i":{i}}}',
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    real_write_lines = raw_writer_module._write_lines
    calls = []

    def write_lines_dying_on_the_second_file(path, lines):
        calls.append(Path(path))
        if len(calls) == 2:
            raise OSError(errno.EIO, "crashed mid-repair")
        return real_write_lines(path, lines)

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(raw_writer_module, "_write_lines",
                          write_lines_dying_on_the_second_file)
    with pytest.raises(OSError):
        reconcile_pair(raw, idx)
    monkeypatched.undo()

    # The property that matters: the interrupted repair is finishable. Under the
    # other write order this raises UnrepairableIndex and the hour is stuck.
    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_frames_kept == 8
    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":"new"}', t_recv_ns=HOUR_05 + 100, t_exch_ms=None, seq=None)
    resumed.close()
    assert len(read_pair(raw, idx)) == 9

    # Why it is finishable: the raw file was still the torn original when the
    # crash landed, so the second run redid the whole salvage.
    assert calls[0] == idx and calls[1] == raw


def test_reconcile_rebuilds_an_index_that_is_missing_entirely(tmp_path: Path):
    """A missing index is the extreme of the damage repair exists to fix.

    `_open` refuses it the same way it refuses a torn pair, so refusing to
    repair it would be the same dead end.
    """
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(3):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    idx.unlink()

    blocked = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    with pytest.raises(HourFileNotAppendable):
        blocked.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)

    outcome = reconcile_pair(raw, idx)
    assert outcome.entries_rebuilt == 3

    resumed = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    resumed.append('{"frame":3}', t_recv_ns=HOUR_05 + 3, t_exch_ms=None, seq=None)
    resumed.close()
    assert [p[0] for p in read_pair(raw, idx)] == [f'{{"frame":{i}}}' for i in range(4)]


def test_reconcile_reports_a_missing_raw_file_as_a_capture_error(tmp_path: Path):
    """A bare FileNotFoundError escapes the documented base class."""
    from capture.raw_writer import MissingPairFile, RawCaptureError

    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    with pytest.raises(MissingPairFile) as exc_info:
        reconcile_pair(raw, idx)
    assert isinstance(exc_info.value, RawCaptureError)


def test_reconcile_of_an_absent_hour_is_a_no_op(tmp_path: Path):
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    outcome = reconcile_pair(raw, idx)
    assert not outcome and outcome.entries_rebuilt == 0


# --------------------------------------------------------------------------
# ROUND 3 - a periodic zstd frame boundary bounds crash loss to one interval
# --------------------------------------------------------------------------

def test_a_torn_hour_loses_only_the_frames_since_the_last_flush(tmp_path: Path):
    """The whole point of the cadence, asserted as an explicit frame count.

    Nothing decodes from a torn compressed block, so without periodic boundaries
    an hour written in one open is a single block and one chopped byte costs
    every frame in it - measured at 400/400 lost in round 2. A boundary every 30s
    of stream time is the only place the damage can stop.
    """
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    _write_seconds_apart(w, 100)                     # 1s cadence, 100 frames
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    with pytest.raises(TruncatedFrameFile) as exc_info:
        _read_lines(raw)
    recovered = exc_info.value.recovered_lines
    # Boundaries land after frames 30, 60 and 90, so 0-90 are on the safe side.
    assert len(recovered) == 91, (
        f"{len(recovered)} of 100 frames survived; without the cadence it is 0")
    assert recovered == [f'{{"frame":{i}}}' for i in range(91)]


def test_without_the_cadence_the_same_damage_costs_the_whole_hour(tmp_path: Path):
    """The contrast that makes the previous test mean something."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT",
                  flush_interval_seconds=10_000.0)     # effectively never
    _write_seconds_apart(w, 100)
    w.close()

    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert _count_zstd_frames(raw) == 1, "one open, one frame, one block"
    _chop_last_byte(raw)

    with pytest.raises(TruncatedFrameFile) as exc_info:
        _read_lines(raw)
    assert exc_info.value.recovered_lines == [], "a torn lone block yields nothing"


@pytest.mark.parametrize("interval_seconds,cadence_seconds,count", [
    (30.0, 1.0, 100),
    (10.0, 1.0, 100),
    (5.0, 1.0, 100),
    (30.0, 0.1, 400),
    (30.0, 8.0, 40),
])
def test_frames_at_risk_are_bounded_by_the_configured_interval(
        tmp_path: Path, interval_seconds: float, cadence_seconds: float, count: int):
    """Only frames since the last boundary are at risk, and the bound is the interval."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT",
                  flush_interval_seconds=interval_seconds)
    _write_seconds_apart(w, count, seconds=cadence_seconds)
    w.close()

    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)
    at_risk = count - _count_lines_tolerantly(raw)

    # One interval's worth of arrivals, plus the frame that triggers the boundary.
    bound = int(interval_seconds / cadence_seconds) + 1
    assert at_risk <= bound, f"{at_risk} frames at risk against a bound of {bound}"


def test_a_smaller_flush_interval_emits_more_frames(tmp_path: Path):
    """The parameter is honoured, on synthetic timestamps rather than real time."""
    counts = {}
    for interval in (5.0, 10.0, 30.0):
        root = tmp_path / f"interval_{interval}"
        w = RawWriter(root, "binance", "depth", "BTCUSDT",
                      flush_interval_seconds=interval)
        _write_seconds_apart(w, 100)
        w.close()
        raw, _ = paths_for(root, "binance", "depth", "BTCUSDT", "2026-08-02T05")
        counts[interval] = _count_zstd_frames(raw)

    # 100 frames at 1s: a boundary every `interval` seconds, plus the close.
    assert counts == {5.0: 20, 10.0: 10, 30.0: 4}, counts


def test_flush_keeps_raw_and_index_frame_aligned(tmp_path: Path):
    """Both files get the boundary, so damage costs the same range in each."""
    for interval in (5.0, 30.0):
        root = tmp_path / f"interval_{interval}"
        w = RawWriter(root, "binance", "depth", "BTCUSDT",
                      flush_interval_seconds=interval)
        _write_seconds_apart(w, 100)
        w.close()
        raw, idx = paths_for(root, "binance", "depth", "BTCUSDT", "2026-08-02T05")
        assert _count_zstd_frames(raw) == _count_zstd_frames(idx), interval
        assert _count_lines_tolerantly(raw) == _count_lines_tolerantly(idx)


def test_an_out_of_order_timestamp_does_not_emit_a_boundary(tmp_path: Path):
    """Late arrivals are ordinary on a live socket; each must not cost a frame.

    The cadence is measured against the frame's own timestamp, so a backward
    jump resets the reference rather than reading as "the interval elapsed".
    """
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    base = HOUR_05 + 600 * 10**9
    for offset_seconds in (0, -5, -10, -3, -8, -1):     # all within the same hour
        w.append(f'{{"late":{offset_seconds}}}',
                 t_recv_ns=base + offset_seconds * 10**9, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert _count_zstd_frames(raw) == 1, "a late frame emitted a boundary"
    assert len(read_pair(raw, idx)) == 6


def test_rotation_still_works_with_flushing_enabled(tmp_path: Path):
    """The cadence must not disturb hour rotation, or resume across it."""
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC", flush_interval_seconds=5.0)
    for i in range(60):
        w.append(f'{{"h5":{i}}}', t_recv_ns=HOUR_05 + i * 10**9, t_exch_ms=None, seq=None)
    for i in range(60):
        w.append(f'{{"h6":{i}}}', t_recv_ns=HOUR_06 + i * 10**9, t_exch_ms=None, seq=None)
    # One late frame reaches back over the boundary and re-opens hour 05.
    w.append('{"h5":60}', t_recv_ns=HOUR_05 + 60 * 10**9, t_exch_ms=None, seq=None)
    w.close()

    r5, i5 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    r6, i6 = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T06")
    hour_05 = read_pair(r5, i5)
    assert [p[0] for p in hour_05] == [f'{{"h5":{i}}}' for i in range(61)]
    assert [p[1].n for p in hour_05] == list(range(61))
    assert [p[0] for p in read_pair(r6, i6)] == [f'{{"h6":{i}}}' for i in range(60)]
    assert _count_zstd_frames(r5) > 1 and _count_zstd_frames(r6) > 1


def test_repair_and_resume_still_terminate_with_flushing_enabled(tmp_path: Path):
    """Round 2's cycle, now over a file the cadence has already split.

    This is the combination that matters in production: the crash shape, the
    boundary that bounds it, and the repair that has to make the hour appendable
    again - and the salvage now keeps 91 frames where it kept 0.
    """
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    _write_seconds_apart(w, 100)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    blocked = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    with pytest.raises(HourFileNotAppendable):
        blocked.append('{"frame":100}', t_recv_ns=HOUR_05 + 100 * 10**9,
                       t_exch_ms=None, seq=None)

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged
    assert outcome.raw_frames_kept == 91, "the cadence is what makes 91 recoverable"
    assert outcome.entries_discarded == 9

    resumed = RawWriter(tmp_path, "binance", "depth", "BTCUSDT",
                        flush_interval_seconds=30.0)
    resumed.append('{"frame":100}', t_recv_ns=HOUR_05 + 100 * 10**9,
                   t_exch_ms=None, seq=None)
    resumed.close()

    pairs = read_pair(raw, idx)
    assert len(pairs) == 92
    assert [p[1].n for p in pairs] == list(range(92))
    assert [p[0] for p in pairs[:91]] == [f'{{"frame":{i}}}' for i in range(91)]
    assert pairs[-1][0] == '{"frame":100}'
    # The salvaged frames kept the metadata they already had.
    assert [p[1].t_recv_ns for p in pairs[:91]] == [HOUR_05 + i * 10**9 for i in range(91)]


def test_a_failed_index_flush_drifts_only_in_the_repairable_direction(tmp_path: Path):
    """The one way the two files can fall out of frame alignment.

    Raw is flushed first, so if the index flush then fails (ENOSPC is the
    realistic cause) the raw file has a boundary the index does not, and a crash
    leaves the index holding FEWER complete entries than the raw file holds
    frames. That is the direction reconcile_pair repairs. The reverse cannot
    happen: the index is never flushed before raw, and an exception on the raw
    side stops the sequence before the index is touched.
    """
    from unittest.mock import MagicMock

    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    _write_seconds_apart(w, 40)                  # one boundary already emitted

    real_idx_z = w._idx_z
    failing_idx_z = MagicMock(wraps=real_idx_z)
    failing_idx_z.flush.side_effect = OSError(errno.ENOSPC, "No space left on device")
    w._idx_z = failing_idx_z

    with pytest.raises(OSError):
        _write_seconds_apart(w, 100)             # the next boundary fails on the index
    w._idx_z = real_idx_z                        # let close() finish normally

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    try:
        w.close()
    except OSError:
        pass

    raw_lines = _count_lines_tolerantly(raw)
    idx_lines = _count_lines_tolerantly(idx)
    assert idx_lines <= raw_lines, (
        f"idx={idx_lines} against raw={raw_lines}: the one direction "
        f"reconcile_pair cannot repair")

    # And the pair is still repairable back to appendable.
    outcome = reconcile_pair(raw, idx)
    resumed = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    resumed.append('{"after":1}', t_recv_ns=HOUR_05 + 500 * 10**9, t_exch_ms=None, seq=None)
    resumed.close()
    assert read_pair(raw, idx)[-1][0] == '{"after":1}'


# --------------------------------------------------------------------------
# IMPORTANT 5 - flush()'s ordering claim, constrained
# --------------------------------------------------------------------------

class _FailsTheSecondFlushOfAPair:
    """Wraps a zstd stream writer and fails the SECOND flush of the pair.

    Deliberately does not name a file. Which of the two files is flushed second
    IS the property under test, so a test that patched `_idx_z` by name would
    pass just as happily with the order reversed - which is exactly what the
    existing `test_a_failed_index_flush_drifts_only_in_the_repairable_direction`
    does, and why swapping `flush()` to index-first survived the whole suite.
    """

    def __init__(self, inner, calls: list[int]) -> None:
        self._inner = inner
        self._calls = calls

    def flush(self, *args, **kwargs):
        self._calls[0] += 1
        if self._calls[0] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return self._inner.flush(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_flush_never_leaves_the_index_ahead_of_the_raw_file(tmp_path: Path):
    """Raw must gain its frame boundary before the index gains its own.

    `reconcile_pair` rebuilds missing index entries from raw lines; nothing
    rebuilds raw frames from index entries. So when the second of the two
    flushes fails - ENOSPC is the realistic cause - the pair has to be left with
    the index holding FEWER complete entries than the raw file holds frames.

    Measured on this fixture: raw-first leaves raw=61 idx=31 and repairs to 61
    frames kept. Index-first leaves raw=31 idx=61, and repair refuses with
    UnrepairableIndex on a raw file that is completely intact - the frames are
    not recoverable and an operator has an unrepairable hour.
    """
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    _write_seconds_apart(w, 31)              # frame 30 emits a clean boundary in both

    calls = [0]
    real_raw_z, real_idx_z = w._raw_z, w._idx_z
    w._raw_z = _FailsTheSecondFlushOfAPair(real_raw_z, calls)
    w._idx_z = _FailsTheSecondFlushOfAPair(real_idx_z, calls)

    with pytest.raises(OSError):
        for i in range(31, 61):               # frame 60 triggers the failing boundary
            w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i * 10**9,
                     t_exch_ms=None, seq=None)
    assert calls[0] == 2, "the fixture did not reach the second flush"

    # The crash image: exactly the bytes that reached the OS before the failure.
    # Whatever is still in a Python buffer is gone, which is what a crash means.
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    crash_raw, crash_idx = paths_for(tmp_path / "after_the_crash", "binance",
                                     "depth", "BTCUSDT", "2026-08-02T05")
    crash_raw.parent.mkdir(parents=True, exist_ok=True)
    crash_raw.write_bytes(raw.read_bytes())
    crash_idx.write_bytes(idx.read_bytes())

    w._raw_z, w._idx_z = real_raw_z, real_idx_z         # let the live writer go
    try:
        w.close()
    except OSError:
        pass

    raw_count = _count_lines_tolerantly(crash_raw)
    idx_count = _count_lines_tolerantly(crash_idx)
    assert idx_count <= raw_count, (
        f"flush() left idx={idx_count} against raw={raw_count}: the one "
        f"direction reconcile_pair cannot repair")
    assert raw_count == 61, f"the raw file lost frames it had already taken: {raw_count}"

    # And the drift it did leave is repairable, with every raw frame kept.
    outcome = reconcile_pair(crash_raw, crash_idx)
    assert outcome.raw_frames_kept == 61
    assert outcome.entries_rebuilt == 61 - idx_count
    assert not outcome.raw_was_salvaged, "the raw file was intact; nothing to salvage"
    assert [p[0] for p in read_pair(crash_raw, crash_idx)] == [
        f'{{"frame":{i}}}' for i in range(61)]


# --------------------------------------------------------------------------
# IMPORTANT 7 - two capture processes must not write the same hour
# --------------------------------------------------------------------------

def test_a_second_writer_refuses_an_hour_a_live_writer_holds(tmp_path: Path):
    """The likely operator accident: `capture --venue binance` started twice.

    `is_hour_being_written()` and the `.writing` marker already existed and
    `reconcile_pair` consulted them, but `_open` did not. Both processes opened
    the pair "ab", both resumed `n` from the same count, and their zstd frames
    interleaved at arbitrary byte boundaries - corrupt at the container level,
    not merely misaligned. Verified: `read_pair` came back
    IndexPositionMismatch, and the second writer's `close()` silently deleted
    the first's marker while the first was still recording.
    """
    holder = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    holder.append('{"held":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)

    intruder = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    with pytest.raises(HourHeldByAnotherWriter) as exc_info:
        intruder.append('{"intruder":0}', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    assert exc_info.value.pid == os.getpid()

    # The refusal cost the holder nothing: its marker and its frames are intact.
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert writing_marker_path(raw).read_text(encoding="utf-8") == str(os.getpid())
    holder.append('{"held":1}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)
    holder.close()
    assert [p[0] for p in read_pair(raw, idx)] == ['{"held":0}', '{"held":1}']


def test_the_refusal_is_isolated_to_the_contended_hour(tmp_path: Path):
    """Contention over one hour's files is a `RawCaptureError` like any other
    damage to them, so `VenueRecorder` quarantines that hour and keeps every
    other stream recording - rather than taking the venue down."""
    from capture.raw_writer import RawCaptureError

    holder = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    holder.append('{"held":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)

    intruder = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    with pytest.raises(RawCaptureError):
        intruder.append('{"x":0}', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)

    # The next hour is a different pair of files and nobody holds it.
    intruder.append('{"x":1}', t_recv_ns=HOUR_06, t_exch_ms=None, seq=None)
    intruder.close()
    holder.close()
    r6, i6 = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T06")
    assert [p[0] for p in read_pair(r6, i6)] == ['{"x":1}']


def test_a_stale_marker_from_a_crashed_writer_does_not_block_recording(tmp_path: Path):
    """Refusing forever would be its own outage. A crash leaves the marker
    behind, and a restart is exactly when capture must resume - so a marker
    whose pid is gone is cleared and claimed rather than obeyed."""
    first = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    first.append('{"before":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    first.flush()
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")

    dead = subprocess.Popen(["true"])
    dead.wait()
    writing_marker_path(raw).write_text(str(dead.pid), encoding="utf-8")

    second = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    second.append('{"after":0}', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    assert writing_marker_path(raw).read_text(encoding="utf-8") == str(os.getpid())
    second.close()

    assert [p[0] for p in read_pair(raw, idx)] == ['{"before":0}', '{"after":0}']


def test_a_refused_open_leaves_no_marker_of_its_own(tmp_path: Path):
    """The claim is taken before the pair is read, so a damaged hour is refused
    with the marker already placed. Leaving it behind would make the writer's
    own live pid block the `reconcile_pair` that is the documented way out.
    """
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}', '{"frame":1}'])
    _write_zst_lines(idx, [encode_index_entry(IndexEntry(
        n=0, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None, kind="data", esc=False))])

    w = RawWriter(tmp_path, "v", "s", "SYM")
    with pytest.raises(HourFileNotAppendable):
        w.append('{"frame":2}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)

    assert not writing_marker_path(raw).exists(), (
        "the refused open left its own marker, which blocks the repair that "
        "is the only way out of the refusal")
    assert reconcile_pair(raw, idx).entries_rebuilt == 1


# --------------------------------------------------------------------------
# IMPORTANT 6 - the fsync posture
# --------------------------------------------------------------------------

def _record_fsyncs(monkeypatch) -> list[tuple[int, bool]]:
    """Record every fd fsynced, classified as file or directory while still open.

    Power loss cannot be simulated in-process: `kill -9` leaves the page cache
    intact, so nothing observable distinguishes a flushed file from a synced
    one. These tests therefore constrain the mechanism, and the reasoning for
    wanting it lives on `RawWriter.flush` and `CaptureLedger.record`.
    """
    calls: list[tuple[int, bool]] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        try:
            is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        except OSError:
            is_directory = False
        calls.append((fd, is_directory))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    return calls


def test_flush_makes_both_files_durable_raw_first(tmp_path: Path, monkeypatch):
    """`fh.flush()` only reaches the page cache. That covers `kill -9` and not a
    power loss or a hypervisor reset, so the 30s cadence bounded crash loss
    against process death only - against power loss the bound was whatever the
    kernel happened to have written back."""
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT", flush_interval_seconds=30.0)
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    raw_fd, idx_fd = w._raw_fh.fileno(), w._idx_fh.fileno()

    calls = _record_fsyncs(monkeypatch)
    w.flush()

    assert [fd for fd, _ in calls] == [raw_fd, idx_fd], (
        "raw must be made durable before the index gains its own boundary")
    w.close()


def test_close_makes_both_files_durable_before_releasing_them(tmp_path: Path,
                                                              monkeypatch):
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    raw_fd, idx_fd = w._raw_fh.fileno(), w._idx_fh.fileno()

    calls = _record_fsyncs(monkeypatch)
    w.close()

    assert [fd for fd, _ in calls] == [raw_fd, idx_fd]


def test_opening_an_hour_makes_its_directory_entry_durable(tmp_path: Path,
                                                           monkeypatch):
    """An fsync of a file's contents does not make its NAME durable. Without the
    directory fsync a power loss can leave the whole hour with no entry at all,
    which is the file's contents surviving in an inode nothing points at."""
    calls = _record_fsyncs(monkeypatch)
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    w.append('{"frame":0}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)

    assert any(is_directory for _, is_directory in calls), (
        "the hour's directory entry was never made durable")
    w.close()


def test_a_failed_raw_fsync_leaves_the_pair_repairable(tmp_path: Path, monkeypatch):
    """Raw not being safely down means the index must not gain a footer.

    Same rule as a failed raw close, now with one more way for raw to fail. The
    index's unfinished frame is abandoned deliberately, and `reconcile_pair`
    rebuilds it from the raw file - which is intact.
    """
    w = RawWriter(tmp_path, "binance", "depth", "BTCUSDT")
    for i in range(5):
        w.append(f'{{"frame":{i}}}', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    raw_fd = w._raw_fh.fileno()

    real_fsync = os.fsync

    def fsync_failing_on_the_raw_file(fd: int) -> None:
        if fd == raw_fd:
            raise OSError(errno.EIO, "I/O error")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync_failing_on_the_raw_file)
    with pytest.raises(OSError):
        w.close()
    monkeypatch.undo()

    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert _read_lines(raw) == [f'{{"frame":{i}}}' for i in range(5)]
    assert reconcile_pair(raw, idx).entries_rebuilt == 5, (
        "the index gained entries the failed raw side could not back")
