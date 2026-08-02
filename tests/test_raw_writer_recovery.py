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
    assert repaired.entries_rebuilt == 1

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
    assert reconcile_pair(raw, idx).entries_rebuilt == 0


def test_recovered_entry_with_escaped_payload_returns_as_stored():
    """Verify the known limitation: recovered entries' escape state is unrecoverable.

    When a crash leaves an index entry missing, we cannot know whether the original
    payload was escaped (contained a real newline) or not (contained literal backslash-n).
    Both produce identical bytes on disk. Therefore, read_pair returns the raw line
    as stored, which may still be in escaped form. This documents the limitation so
    downstream can handle kind="recovered" entries explicitly.
    """
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        w = RawWriter(tmp_path, "test", "stream", "SYMBOL")
        # Write an escaped payload (contains a real newline)
        escaped_payload = '{"msg":"line1\nline2"}'
        w.append(escaped_payload, t_recv_ns=1785648600_000_000_000,
                 t_exch_ms=None, seq=None)
        w.close()

        raw, idx = paths_for(tmp_path, "test", "stream", "SYMBOL", "2026-08-02T05")
        _truncate_index_by_one(idx)

        repaired = reconcile_pair(raw, idx)
        assert repaired.entries_rebuilt == 1

        pairs = read_pair(raw, idx)
        assert len(pairs) == 1
        # The recovered entry's payload is returned as stored (still escaped)
        # because we cannot recover the escape state from the bytes alone
        recovered_payload, entry = pairs[0]
        assert entry.kind == "recovered"
        # Verify the payload is in escaped form (backslash-n, not real newline)
        assert "\\n" in recovered_payload
        assert "\n" not in recovered_payload.split("\\n")[0]  # no embedded newline before escape sequence
