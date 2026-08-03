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


def test_requesting_a_symbol_absent_from_an_existing_capture_raises(tmp_path):
    """A typo or the wrong stream name must be loud, not a silently smaller store.

    The venue/date folder exists here - a real capture ran that day - so a
    requested symbol matching nothing under it is almost certainly a mistake,
    not a legitimate empty day. Folding it into a normal-looking aggregate is
    exactly the silent loss `NoHourFilesForSymbol`'s docstring refuses.
    """
    from store.cli import NoHourFilesForSymbol

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    (source / "trade_BTCUSDT_2026-08-02T00.ndjson.zst").write_bytes(b"placeholder")
    (source / "trade_BTCUSDT_2026-08-02T00.idx.zst").write_bytes(b"placeholder")

    with pytest.raises(NoHourFilesForSymbol, match="ETHUSDT_TYPO"):
        build_bars_for_day(
            capture_root=tmp_path, store_root=tmp_path / "store", venue="binance",
            date="2026-08-02", symbols=["BTCUSDT", "ETHUSDT_TYPO"],
            interval_ns=60_000_000_000)


def test_per_symbol_breakdown_distinguishes_a_symbol_that_contributes_nothing(
    tmp_path, monkeypatch,
):
    """A symbol whose frames all parse to zero trades must stay visible.

    Two symbols are requested; both have hour files, so neither trips
    NoHourFilesForSymbol. One produces a real trade, the other only a non-trade
    frame. Without a per-symbol breakdown the two are indistinguishable in the
    aggregate totals - this is what makes a symbol contributing nothing provably
    different from one contributing everything.
    """
    from capture.frame_codec import IndexEntry
    from store import cli as store_cli

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        (source / f"trade_{symbol}_2026-08-02T00.ndjson.zst").write_bytes(b"placeholder")
        (source / f"trade_{symbol}_2026-08-02T00.idx.zst").write_bytes(b"placeholder")

    trade_frame = (
        '{"stream":"btcusdt@trade","data":{"e":"trade","T":1785685177439,'
        '"s":"BTCUSDT","p":"63113.20","q":"0.001"}}'
    )
    non_trade_frame = '{"stream":"ethusdt@trade","data":{"e":"kline"}}'
    entry = IndexEntry(n=0, t_recv_ns=1785685177508349176, t_exch_ms=1785685177439,
                       seq=None, kind="data", esc=False)

    def fake_read_pair(raw, idx):
        if "BTCUSDT" in raw.name:
            return [(trade_frame, entry)]
        return [(non_trade_frame, entry)]

    monkeypatch.setattr(store_cli, "read_pair", fake_read_pair)

    summary = store_cli.build_bars_for_day(
        capture_root=tmp_path, store_root=tmp_path / "store", venue="binance",
        date="2026-08-02", symbols=["BTCUSDT", "ETHUSDT"], interval_ns=60_000_000_000)

    assert summary["by_symbol"] == {
        "BTCUSDT": {"frames": 1, "trades": 1},
        "ETHUSDT": {"frames": 1, "trades": 0},
    }
    assert summary["frames"] == 2
    assert summary["trades"] == 1
