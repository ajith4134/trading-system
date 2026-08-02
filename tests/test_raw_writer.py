import time
from pathlib import Path
import pytest
from capture.raw_writer import RawWriter, read_pair, hour_key, paths_for, PairLengthMismatch


def test_hour_key_is_utc(monkeypatch):
    """hour_key must read UTC, not the machine's local time.

    Asserting only the UTC answer passes by accident on a box whose TZ is
    already UTC - dropping `tz=timezone.utc` from hour_key survived the previous
    version of this test for exactly that reason. The process timezone is moved
    to a non-zero, non-integer offset so a naive fromtimestamp() gives a
    different hour, and a different date at the boundary.
    """
    monkeypatch.setenv("TZ", "Asia/Kolkata")  # UTC+05:30, no DST
    time.tzset()
    try:
        # 2026-08-02T05:30:00Z -> 11:00 local, so a naive read says hour 11.
        assert hour_key(1785648600_000_000_000) == "2026-08-02T05"
        # 2026-08-02T23:30:00Z -> 2026-08-03T05:00 local, so a naive read also
        # lands on the wrong date.
        assert hour_key(1785648600_000_000_000 + 64_800_000_000_000) == "2026-08-02T23"
    finally:
        monkeypatch.undo()
        time.tzset()


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


def test_payload_with_backslash_roundtrips_exactly(tmp_path: Path):
    """Regression test for finding 1: backslash should not be corrupted."""
    payload = '{"path":"C:\\\\Users\\\\data"}'
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert pairs[0][0] == payload  # byte-exact, with all backslashes intact
    assert pairs[0][1].esc is False


def test_payload_with_u2028_roundtrips_exactly(tmp_path: Path):
    """Regression test for finding 2: U+2028 should not cause over-split."""
    payload = '{"text":"before\u2028after"}'  # U+2028 line separator
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append(payload, t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")
    pairs = read_pair(raw, idx)
    assert len(pairs) == 1  # Should not split on U+2028
    assert pairs[0][0] == payload  # byte-exact roundtrip
    assert pairs[0][1].esc is False


def test_read_pair_detects_length_mismatch_truncated_idx(tmp_path: Path):
    """Test for finding 4: read_pair raises when idx file is truncated."""
    w = RawWriter(tmp_path, "test", "trades", "SYMBOL")
    w.append('{"frame":0}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"frame":1}', t_recv_ns=1785648600_000_000_001, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "test", "trades", "SYMBOL", "2026-08-02T05")

    # Truncate the idx file to simulate a mid-write failure
    # (delete it entirely to simulate the idx write failing completely)
    idx.unlink()
    # Write an empty compressed file to idx so it's valid but empty
    import zstandard
    cctx = zstandard.ZstdCompressor(level=3)
    with open(idx, "wb") as fh:
        with cctx.stream_writer(fh) as z:
            z.write(b"")  # Empty file

    # read_pair should raise PairLengthMismatch, not silently truncate
    with pytest.raises(PairLengthMismatch) as exc_info:
        read_pair(raw, idx)

    # Verify exception carries both line counts and file paths
    err = exc_info.value
    assert err.raw_count == 2
    assert err.idx_count == 0
    assert err.raw_path == raw
    assert err.idx_path == idx
    assert "2" in str(err)  # raw count in message
    assert "0" in str(err)  # idx count in message


def test_read_pair_exception_includes_counts_and_paths(tmp_path: Path):
    """Test that PairLengthMismatch exception provides actionable diagnostics."""
    w = RawWriter(tmp_path, "venue", "stream", "SYMBOL")
    w.append('{"data":1}', t_recv_ns=1785648600_000_000_000, t_exch_ms=None, seq=None)
    w.append('{"data":2}', t_recv_ns=1785648600_000_000_001, t_exch_ms=None, seq=None)
    w.append('{"data":3}', t_recv_ns=1785648600_000_000_002, t_exch_ms=None, seq=None)
    w.close()

    raw, idx = paths_for(tmp_path, "venue", "stream", "SYMBOL", "2026-08-02T05")

    # Create a truncated idx file with only 1 entry (simulates write failing after 1st frame)
    import zstandard
    cctx = zstandard.ZstdCompressor(level=3)
    with open(idx, "wb") as fh:
        with cctx.stream_writer(fh) as z:
            z.write(b'{"n":0,"t_recv_ns":1785648600000000000,"t_exch_ms":null,"seq":null,"kind":"data","esc":false}\n')

    with pytest.raises(PairLengthMismatch) as exc_info:
        read_pair(raw, idx)

    err = exc_info.value
    assert err.raw_count == 3
    assert err.idx_count == 1
    assert "reconcile_pair" in str(err).lower()  # Message should reference the repair function


def test_frame_after_a_failed_idx_write_keeps_its_own_receipt_time(tmp_path: Path):
    """A mid-stream index write failure must not relabel later frames.

    This test replaces `test_append_increments_n_after_raw_write_not_idx_write`,
    which asserted that `n` keeps incrementing so values are "not reused". That
    assertion was true but hollow: nothing ever read `n`, so the guarantee it
    claimed - that a frame is described by its own index entry - was not the one
    being verified. Positional pairing meant frame 1's bytes came back carrying
    frame 2's receipt time and sequence numbers. The n-increment behaviour is
    kept and still asserted here, but what is actually verified is the property
    it exists to provide.
    """
    from unittest.mock import MagicMock
    from capture.raw_writer import reconcile_pair

    base = 1785648600_000_000_000
    w = RawWriter(tmp_path, "test", "stream", "SYMBOL")

    assert w.append('{"frame":0}', t_recv_ns=base, t_exch_ms=None, seq={"id": 0}) == 0
    assert w._n == 1

    # Frame 1: raw write lands, index write fails. The hole is now in the MIDDLE
    # of the index, because frame 2 below writes its entry successfully.
    surviving_idx_z = w._idx_z
    failing_idx_z = MagicMock()
    failing_idx_z.write.side_effect = IOError("Simulated idx write failure")
    w._idx_z = failing_idx_z
    with pytest.raises(IOError):
        w.append('{"frame":1}', t_recv_ns=base + 1, t_exch_ms=None, seq={"id": 1})

    # n advanced with the raw file, so frame 2 is labelled n=2 and frame 1's n=1
    # is left unclaimed - that unclaimed value is how the hole is located.
    assert w._n == 2, "n must track raw lines written, even when the idx write fails"

    w._idx_z = surviving_idx_z
    assert w.append('{"frame":2}', t_recv_ns=base + 2, t_exch_ms=None,
                    seq={"id": 2}) == 2
    w.close()

    raw, idx = paths_for(tmp_path, "test", "stream", "SYMBOL", "2026-08-02T05")

    # 3 raw lines against 2 index entries: refused, not silently paired up.
    with pytest.raises(PairLengthMismatch):
        read_pair(raw, idx)

    assert reconcile_pair(raw, idx).entries_rebuilt == 1

    pairs = read_pair(raw, idx)
    assert [p[0] for p in pairs] == ['{"frame":0}', '{"frame":1}', '{"frame":2}']
    assert [p[1].n for p in pairs] == [0, 1, 2]

    # Frame 1 lost its entry, so it must read as unknown - never as frame 2's.
    assert pairs[1][1].kind == "recovered"
    assert pairs[1][1].t_recv_ns == 0
    assert pairs[1][1].seq is None

    # Frame 2 keeps its own receipt time and sequence numbers.
    assert pairs[2][1].kind == "data"
    assert pairs[2][1].t_recv_ns == base + 2
    assert pairs[2][1].seq == {"id": 2}
