"""`iter_pair` must agree with `read_pair` on every verdict, and hold nothing.

The streaming reader exists for one reason - a universe-wide bar build peaked at
3.98 GB on a single hour of TUTUSDT because `read_pair` materialises the hour
before a trade is examined - and a reader that is cheaper but disagrees about
damage would trade an OOM for a silently wrong store. So most of these tests are
parity tests: the same pair, the same fault, the same exception.

The one deliberate difference is when the refusal lands. `read_pair` counts both
files before returning anything, so it refuses before a caller sees a frame;
`iter_pair` yields as it decodes, so frames ahead of the damage reach the caller
first. That is tested rather than hidden, because it is the property a caller has
to design around - see the note in `iter_pair` about appending after the loop.
"""
import json
from pathlib import Path

import pytest
import zstandard

from capture.raw_writer import (
    IndexPositionMismatch, PairLengthMismatch, RawWriter, TruncatedFrameFile,
    iter_lines, iter_pair, paths_for, read_pair, reconcile_pair,
)
from capture.frame_codec import IndexEntry, encode_index_entry

HOUR_05 = 1785648600_000_000_000        # 2026-08-02T05:30:00Z


def _write_zst_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines:
                w.write((line + "\n").encode("utf-8"))


def _chop_last_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-1])


def _entry(n: int, **over) -> str:
    fields = dict(n=n, t_recv_ns=HOUR_05 + n, t_exch_ms=None, seq={"id": n},
                  kind="data", esc=False)
    fields.update(over)
    return encode_index_entry(IndexEntry(**fields))


# --------------------------------------------------------------------------
# Parity on healthy pairs
# --------------------------------------------------------------------------

def test_streamed_frames_are_identical_to_the_buffered_ones(tmp_path: Path):
    """Byte-for-byte, entry-for-entry, in order."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(500):
        w.append(json.dumps({"seq": i}, separators=(",", ":")),
                 t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq={"id": i})
    w.close()
    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")

    assert list(iter_pair(raw, idx)) == read_pair(raw, idx)


def test_frames_written_across_several_opens_stream_as_one_run(tmp_path: Path):
    """Each open appends a new zstd frame; the reader walks all of them.

    A stream that stopped at the first frame's `eof` would return the first
    session's frames and call the file finished - a silent partial read, which is
    the failure mode the whole reader exists to refuse.
    """
    for session in range(3):
        w = RawWriter(tmp_path, "v", "s", "SYM")
        for i in range(4):
            w.append(json.dumps({"session": session, "i": i}, separators=(",", ":")),
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")

    streamed = list(iter_pair(raw, idx))
    assert len(streamed) == 12
    assert streamed == read_pair(raw, idx)


def test_an_empty_payload_frame_streams_as_an_empty_line(tmp_path: Path):
    """An empty websocket text frame is a frame, not a missing line."""
    w = RawWriter(tmp_path, "v", "s", "SYM")
    w.append('{"a":1}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.append("", t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    w.append('{"b":2}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")

    assert [payload for payload, _ in iter_pair(raw, idx)] == ['{"a":1}', "", '{"b":2}']


def test_a_payload_carrying_a_unicode_line_separator_is_not_split(tmp_path: Path):
    """Splitting is on \\n alone - never str.splitlines(), which eats U+2028.

    The stream splits on BYTES and decodes the pieces, so this also covers the
    boundary case that makes that safe: 0x0A cannot appear inside a multi-byte
    UTF-8 sequence, so a line break never lands mid-character.
    """
    payload = json.dumps({"text": "line break"}, separators=(",", ":"),
                         ensure_ascii=False)
    w = RawWriter(tmp_path, "v", "s", "SYM")
    w.append(payload, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")

    streamed = list(iter_pair(raw, idx))
    assert len(streamed) == 1
    assert streamed[0][0] == payload


def test_an_escaped_payload_is_unescaped_by_the_stream_too(tmp_path: Path):
    """A payload holding a newline round-trips through the escape, not past it."""
    payload = '{"text":"two\nlines"}'
    w = RawWriter(tmp_path, "v", "s", "SYM")
    w.append(payload, t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")

    streamed = list(iter_pair(raw, idx))
    assert streamed[0][1].esc is True
    assert streamed[0][0] == payload


def test_a_recovered_entry_streams_its_payload_as_stored(tmp_path: Path):
    """Escape state is unrecoverable for a rebuilt entry, so nothing is guessed."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ["\\n"])
    _write_zst_lines(idx, [_entry(0, kind="recovered", esc=False, seq=None)])

    streamed = list(iter_pair(raw, idx))
    assert streamed[0][0] == "\\n"
    assert streamed == read_pair(raw, idx)


def test_an_hour_holding_an_empty_zstd_stream_is_not_damage(tmp_path: Path):
    """A complete frame carrying no lines is a quiet hour, not a torn tail.

    Exactly the shape `store.cli` records meeting on 2026-08-08: a subscription
    that produced nothing writes a well-formed empty stream, and refusing it
    would fail a build over a venue that was simply silent.
    """
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, [])
    _write_zst_lines(idx, [])

    assert list(iter_pair(raw, idx)) == []
    assert read_pair(raw, idx) == []


# --------------------------------------------------------------------------
# Parity on damage - same exception, same precedence
# --------------------------------------------------------------------------

def test_a_short_index_is_refused_with_both_counts(tmp_path: Path):
    """The stream only meets the end of one file, and still reports both counts.

    Draining the longer side to count it is the whole reason `PairLengthMismatch`
    survives streaming with its diagnostics intact - `reconcile_pair` is chosen or
    not on those two numbers.
    """
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ['{"f":0}', '{"f":1}', '{"f":2}'])
    _write_zst_lines(idx, [_entry(0)])

    with pytest.raises(PairLengthMismatch) as exc_info:
        list(iter_pair(raw, idx))
    assert (exc_info.value.raw_count, exc_info.value.idx_count) == (3, 1)
    assert "reconcile_pair" in str(exc_info.value).lower()


def test_a_short_raw_file_is_refused_with_both_counts(tmp_path: Path):
    """The mirror case: the index outlived the frames it describes."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ['{"f":0}'])
    _write_zst_lines(idx, [_entry(0), _entry(1), _entry(2)])

    with pytest.raises(PairLengthMismatch) as exc_info:
        list(iter_pair(raw, idx))
    assert (exc_info.value.raw_count, exc_info.value.idx_count) == (1, 3)


def test_a_mispositioned_entry_in_an_equal_length_pair_is_refused(tmp_path: Path):
    """Equal counts are not enough: the entry must describe THIS line."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ['{"f":0}', '{"f":1}'])
    _write_zst_lines(idx, [_entry(0), _entry(2)])

    with pytest.raises(IndexPositionMismatch) as exc_info:
        list(iter_pair(raw, idx))
    assert (exc_info.value.position, exc_info.value.entry_n) == (1, 2)


def test_a_hole_in_the_index_reads_as_length_mismatch_not_position(tmp_path: Path):
    """Precedence, and it is not cosmetic.

    A frame whose index write failed leaves BOTH faults: the counts differ and
    the surviving entries no longer sit at their own positions. The buffered
    reader reports the length mismatch, and that is the one naming the repair. A
    stream meets the bad `n` first, so it has to drain both files before deciding
    which fault to report - this test is what holds it to that.
    """
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ['{"f":0}', '{"f":1}', '{"f":2}'])
    _write_zst_lines(idx, [_entry(0), _entry(2)])

    with pytest.raises(PairLengthMismatch):
        list(iter_pair(raw, idx))
    with pytest.raises(PairLengthMismatch):
        read_pair(raw, idx)

    assert reconcile_pair(raw, idx).entries_rebuilt == 1
    assert [payload for payload, _ in iter_pair(raw, idx)] == [
        '{"f":0}', '{"f":1}', '{"f":2}']


def test_a_torn_raw_tail_is_refused_rather_than_read_as_the_end(tmp_path: Path):
    """A decodable prefix is indistinguishable from success unless it is refused."""
    w = RawWriter(tmp_path, "v", "s", "SYM")
    for i in range(400):
        w.append(json.dumps({"seq": i}, separators=(",", ":")),
                 t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _chop_last_byte(raw)

    with pytest.raises(TruncatedFrameFile):
        list(iter_pair(raw, idx))


def test_a_complete_frame_ending_mid_line_is_refused(tmp_path: Path):
    """A whole zstd frame can still stop in the middle of a payload."""
    raw = tmp_path / "partial.ndjson.zst"
    cctx = zstandard.ZstdCompressor(level=3)
    with open(raw, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            w.write(b'{"frame":0}\n{"frame":1}\n{"fra')

    with pytest.raises(TruncatedFrameFile) as exc_info:
        list(iter_lines(raw))
    assert exc_info.value.reason == "final line has no terminating newline"


def test_the_damage_report_says_how_much_was_whole_without_holding_it(tmp_path: Path):
    """`recovered_lines` empty must not read as "nothing survived".

    The stream handed those lines to its caller as it decoded them, so it cannot
    hand them over a second time. It reports the count and says so, and
    `recovered_prefix_retained` is how the two cases are told apart.
    """
    raw = tmp_path / "partial.ndjson.zst"
    cctx = zstandard.ZstdCompressor(level=3)
    with open(raw, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            w.write(b'{"frame":0}\n{"frame":1}\n{"fra')

    with pytest.raises(TruncatedFrameFile) as exc_info:
        list(iter_lines(raw))
    damage = exc_info.value
    assert damage.recovered_count == 2
    assert damage.recovered_lines == []
    assert damage.recovered_prefix_retained is False
    assert "not retained" in str(damage)

    # The buffered reader is unchanged, and still carries the lines themselves.
    with pytest.raises(TruncatedFrameFile) as buffered_info:
        read_pair(raw, raw)
    assert buffered_info.value.recovered_lines == ['{"frame":0}', '{"frame":1}']
    assert buffered_info.value.recovered_prefix_retained is True


def test_frames_before_the_damage_reach_the_caller(tmp_path: Path):
    """The deliberate difference from `read_pair`, stated as a test.

    A caller that appends inside the loop would commit this prefix and then fail;
    every build using this reader appends after the loop, so the refusal
    abandons the build instead.
    """
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    _write_zst_lines(raw, ['{"f":0}', '{"f":1}', '{"f":2}'])
    _write_zst_lines(idx, [_entry(0), _entry(1)])

    seen = []
    with pytest.raises(PairLengthMismatch):
        for payload, _ in iter_pair(raw, idx):
            seen.append(payload)
    assert seen == ['{"f":0}', '{"f":1}']


# --------------------------------------------------------------------------
# The bound itself
# --------------------------------------------------------------------------

def test_stream_memory_does_not_grow_with_the_file(tmp_path: Path, monkeypatch):
    """The property, stated as the thing that actually has to hold.

    Not "under N megabytes": a stream's peak is one chunk of compressed input
    times whatever that expands to, so an absolute ceiling measures the test
    data's compression ratio rather than the reader. What must be true is that
    quadrupling the file barely moves the stream while it multiplies the
    buffered read - a list that grows per frame cannot pass that.

    Three things the first drafts of this test got wrong, all of which made it
    measure the fixture instead of the code: compressible padding, a file small
    enough to arrive inside a single chunk, and - the one that survived longest -
    remembering that the INDEX file is read the same way. Sequential `t_recv_ns`
    and null `seq` compress to almost nothing, so the whole sidecar landed in one
    chunk and the peak tracked the frame count no matter what the reader did. So
    both files are built from random bytes, and the chunk is shrunk until each
    spans many of them.

    Measured on the real tape rather than inferred here: one hour of TUTUSDT,
    5,634,232 frames, peaked at 3.98 GB through `read_pair` and 0.14 GB through
    `iter_pair`. This is the cheap standing version of that measurement.
    """
    import random
    import tracemalloc

    monkeypatch.setattr("capture.raw_writer.DECOMPRESS_CHUNK_BYTES", 64 * 1024)
    noise = random.Random(20260810)

    def peaks(frame_count: int) -> tuple[int, int]:
        raw, idx = paths_for(tmp_path / str(frame_count), "v", "s", "SYM",
                             "2026-08-02T05")
        payloads = [json.dumps({"seq": i, "pad": noise.randbytes(100).hex()},
                               separators=(",", ":"))
                    for i in range(frame_count)]
        _write_zst_lines(raw, payloads)
        _write_zst_lines(idx, [
            _entry(i, t_recv_ns=noise.randrange(2**62),
                   seq={"id": noise.randrange(2**62)})
            for i in range(frame_count)])

        tracemalloc.start()
        streamed = sum(1 for _ in iter_pair(raw, idx))
        _, streamed_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        buffered = read_pair(raw, idx)
        _, buffered_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert streamed == len(buffered) == frame_count
        return streamed_peak, buffered_peak

    small_streamed, small_buffered = peaks(10_000)
    large_streamed, large_buffered = peaks(40_000)

    assert large_buffered > 3 * small_buffered, (
        "the buffered reader is expected to scale with the file; if it stopped "
        "doing so, this test is no longer measuring what it claims")
    assert large_streamed < 1.5 * small_streamed
    assert large_streamed < large_buffered / 10
