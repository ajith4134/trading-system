"""Regression tests for reading and repairing a raw/index pair.

Every test here reproduces a defect the pre-existing suite did not constrain:
index entries paired by line position instead of by `n`, a torn zstd tail
accepted as complete data, and an empty payload - a legal websocket text frame -
collapsing two lines into one and bricking the hour permanently.
"""
import json
from pathlib import Path

import pytest
import zstandard

from capture.raw_writer import (
    RawWriter, read_pair, reconcile_pair, paths_for, _read_lines,
    RawCaptureError, TruncatedFrameFile, IndexPositionMismatch,
)
from capture.frame_codec import IndexEntry, encode_index_entry

HOUR_05 = 1785648600_000_000_000        # 2026-08-02T05:30:00Z


def _write_zst_lines(path: Path, lines: list[str]) -> None:
    cctx = zstandard.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for line in lines:
                w.write((line + "\n").encode("utf-8"))


def _chop_last_byte(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])


# --------------------------------------------------------------------------
# CRITICAL 2 - `n` is authoritative, not line position
# --------------------------------------------------------------------------

def test_read_pair_refuses_an_entry_that_does_not_match_its_position(tmp_path: Path):
    """Equal line counts are not enough: the entry must describe THIS line."""
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}', '{"frame":1}'])
    _write_zst_lines(idx, [
        encode_index_entry(IndexEntry(n=0, t_recv_ns=HOUR_05, t_exch_ms=None,
                                      seq={"id": 0}, kind="data", esc=False)),
        # n=2: this entry describes a frame further down, so pairing it against
        # line 1 would hand back frame 1's bytes with frame 2's metadata.
        encode_index_entry(IndexEntry(n=2, t_recv_ns=HOUR_05 + 2, t_exch_ms=None,
                                      seq={"id": 2}, kind="data", esc=False)),
    ])

    with pytest.raises(IndexPositionMismatch) as exc_info:
        read_pair(raw, idx)
    assert exc_info.value.position == 1
    assert exc_info.value.entry_n == 2


def test_reconcile_inserts_a_recovered_entry_at_the_hole_not_at_the_end(tmp_path: Path):
    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}', '{"frame":1}', '{"frame":2}', '{"frame":3}'])
    # Entries 1 and 2 are missing from the middle; entry 3 survived.
    _write_zst_lines(idx, [
        encode_index_entry(IndexEntry(n=0, t_recv_ns=HOUR_05, t_exch_ms=None,
                                      seq={"id": 0}, kind="data", esc=False)),
        encode_index_entry(IndexEntry(n=3, t_recv_ns=HOUR_05 + 3, t_exch_ms=None,
                                      seq={"id": 3}, kind="data", esc=False)),
    ])

    assert reconcile_pair(raw, idx).entries_rebuilt == 2

    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == [f'{{"frame":{i}}}' for i in range(4)]
    assert [p[1].n for p in pairs] == [0, 1, 2, 3]
    assert [p[1].kind for p in pairs] == ["data", "recovered", "recovered", "data"]
    # The surviving entry stayed on its own frame instead of sliding to line 1.
    assert pairs[3][1].t_recv_ns == HOUR_05 + 3
    assert pairs[3][1].seq == {"id": 3}
    assert pairs[1][1].t_recv_ns == 0 and pairs[2][1].t_recv_ns == 0


def test_reconcile_refuses_an_index_describing_frames_the_raw_file_lacks(tmp_path: Path):
    from capture.raw_writer import UnrepairableIndex

    raw, idx = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    _write_zst_lines(raw, ['{"frame":0}'])
    _write_zst_lines(idx, [
        encode_index_entry(IndexEntry(n=0, t_recv_ns=HOUR_05, t_exch_ms=None,
                                      seq=None, kind="data", esc=False)),
        encode_index_entry(IndexEntry(n=1, t_recv_ns=HOUR_05 + 1, t_exch_ms=None,
                                      seq=None, kind="data", esc=False)),
    ])
    with pytest.raises(UnrepairableIndex):
        reconcile_pair(raw, idx)


# --------------------------------------------------------------------------
# CRITICAL 3 - a torn zstd tail must never pass for complete data
# --------------------------------------------------------------------------

def test_torn_raw_tail_never_returns_a_partial_line_as_a_frame(tmp_path: Path):
    """`kill -9` shape: one byte short, so the last frame is unfinished.

    stream_reader() hands back the decodable prefix without complaint, and that
    prefix ends mid-payload - a half frame presented as a whole one.
    """
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(json.dumps({"seq": i, "px": 30000 + i}, separators=(",", ":")),
                 t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, _ = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    try:
        lines = _read_lines(raw)
    except RawCaptureError:
        return          # damage refused - the only acceptable outcome
    for line in lines:
        json.loads(line)  # a half payload is not parseable JSON
    pytest.fail("a truncated raw file was accepted as complete")


def test_torn_raw_tail_is_raised_with_the_recoverable_prefix(tmp_path: Path):
    """The frames written before the damaged one stay salvageable.

    Each open appends its own zstd frame, so a restart-heavy hour keeps its
    earlier frames readable even when the last one is torn. Within a single
    zstd frame the recoverable prefix can be empty - zstd emits nothing until a
    compressed block completes - which is why the damage is raised rather than
    reported as however much happened to decode.
    """
    for session in range(3):
        w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
        for i in range(4):
            w.append(json.dumps({"session": session, "seq": i}, separators=(",", ":")),
                     t_recv_ns=HOUR_05 + session * 4 + i, t_exch_ms=None, seq=None)
        w.close()

    raw, _ = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    assert len(_read_lines(raw)) == 12
    _chop_last_byte(raw)

    with pytest.raises(TruncatedFrameFile) as exc_info:
        _read_lines(raw)
    recovered = exc_info.value.recovered_lines
    assert len(recovered) == 8, "the two intact zstd frames should survive"
    for line in recovered:
        json.loads(line)      # every salvaged line is a whole frame


def test_matching_damage_in_both_files_is_still_detected(tmp_path: Path):
    """Equal surviving counts must not read as success."""
    w = RawWriter(tmp_path, "binance", "trades", "BTCUSDT")
    for i in range(400):
        w.append(json.dumps({"seq": i}, separators=(",", ":")),
                 t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "binance", "trades", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)
    _chop_last_byte(idx)

    with pytest.raises(TruncatedFrameFile):
        read_pair(raw, idx)


def test_a_final_line_without_its_newline_is_refused(tmp_path: Path):
    """A complete zstd frame can still end mid-line; that is not a whole frame."""
    raw, _ = paths_for(tmp_path, "v", "s", "SYM", "2026-08-02T05")
    raw.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=3)
    with open(raw, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            w.write(b'{"frame":0}\n{"frame":1}\n{"fra')

    with pytest.raises(TruncatedFrameFile) as exc_info:
        _read_lines(raw)
    assert exc_info.value.recovered_lines == ['{"frame":0}', '{"frame":1}']


# --------------------------------------------------------------------------
# CRITICAL 4 - an empty payload is a legal frame, not a missing line
# --------------------------------------------------------------------------

def test_empty_payload_frame_survives_as_an_empty_line(tmp_path: Path):
    """An empty websocket text frame is legal and must not brick the hour."""
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append('{"a":1}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.append('', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    w.append('{"b":2}', t_recv_ns=HOUR_05 + 2, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == ['{"a":1}', '', '{"b":2}']
    assert [p[1].n for p in pairs] == [0, 1, 2]
    assert pairs[1][1].t_recv_ns == HOUR_05 + 1


def test_trailing_empty_payload_does_not_create_a_mismatch(tmp_path: Path):
    """The worst case: the empty frame is last, so it is the newline stripped."""
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    w.append('{"a":1}', t_recv_ns=HOUR_05, t_exch_ms=None, seq=None)
    w.append('', t_recv_ns=HOUR_05 + 1, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    pairs = read_pair(raw, idx)       # used to raise PairLengthMismatch forever
    assert [p[0] for p in pairs] == ['{"a":1}', '']
    assert reconcile_pair(raw, idx).entries_rebuilt == 0


def test_consecutive_empty_payloads_each_keep_their_own_line(tmp_path: Path):
    w = RawWriter(tmp_path, "hyperliquid", "l2Book", "BTC")
    for i in range(3):
        w.append('', t_recv_ns=HOUR_05 + i, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "hyperliquid", "l2Book", "BTC", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == ['', '', '']
    assert [p[1].t_recv_ns for p in pairs] == [HOUR_05, HOUR_05 + 1, HOUR_05 + 2]
