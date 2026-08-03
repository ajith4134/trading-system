# tests/test_store_cli.py
"""The build must refuse damaged input rather than quietly producing fewer bars."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from store.cli import build_bars_for_day

MINUTE_NS = 60_000_000_000
SECOND_NS = 1_000_000_000


def _midnight_ns(date: str) -> int:
    moment = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    return int(moment.timestamp()) * SECOND_NS


def _binance_trade_frame(symbol: str, price: float, size: float, event_ns: int) -> str:
    return json.dumps({
        "stream": f"{symbol.lower()}@trade",
        "data": {"e": "trade", "T": event_ns // 1_000_000, "s": symbol,
                 "p": f"{price}", "q": f"{size}"},
    })


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


def test_a_bar_whose_trades_span_two_day_folders_is_built_complete_not_partial(tmp_path):
    """The late arrival must extend day D's bar, never replace it with itself.

    `RawWriter` rotates hour files on RECEIVE time, so a trade with an event time
    of 23:59:59.9 on day D that lands 50 ms after midnight is filed under day
    D+1. Building each day from its own folder alone therefore made day D+1 emit
    a second bar for a day-D minute, built from that one late trade and carrying
    a later availability time - and the reader, which resolves corrections by
    (symbol, venue, event_time) with the latest version winning, treated the
    partial bar as a correction of the complete one. The complete bar became
    permanently unreachable, and nothing about the result looked wrong.

    Measured lag on the real archive reaches 141 s, so this corrupted the last
    bar of essentially every captured day.
    """
    from capture.raw_writer import RawWriter
    from store.clock_gated_reader import ClockGatedReader
    from store.temporal_schema import EVENT_TIME

    day_d, day_d1 = "2026-08-02", "2026-08-03"
    d_midnight = _midnight_ns(day_d)
    d1_midnight = _midnight_ns(day_d1)
    last_minute = d1_midnight - MINUTE_NS

    # (price, size, event_ns, receive_ns). The first three land in day D's 23:00
    # file; the fourth belongs to the same 23:59 minute but arrives after
    # midnight and is filed under day D+1; the fifth is a genuine day-D+1 trade.
    tape = [
        (100.0, 1.0, last_minute + 10 * SECOND_NS, last_minute + 10 * SECOND_NS + 100_000_000),
        (110.0, 2.0, last_minute + 20 * SECOND_NS, last_minute + 20 * SECOND_NS + 100_000_000),
        (105.0, 3.0, last_minute + 30 * SECOND_NS, last_minute + 30 * SECOND_NS + 100_000_000),
        (999.0, 0.001, d1_midnight - 100_000_000, d1_midnight + 50_000_000),
        (200.0, 4.0, d1_midnight + 5 * SECOND_NS, d1_midnight + 5 * SECOND_NS + 100_000_000),
    ]
    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    for price, size, event_ns, receive_ns in tape:
        writer.append(_binance_trade_frame("BTCUSDT", price, size, event_ns),
                      receive_ns, event_ns // 1_000_000, None)
    writer.close()

    store_root = tmp_path / "store"
    build = lambda date: build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=date,
        symbols=["BTCUSDT"], interval_ns=MINUTE_NS)

    day_d_summary = build(day_d)
    day_d1_summary = build(day_d1)

    # Day D reached into day D+1's folder for the late arrival; day D+1 read the
    # same trade and discarded it, because it belongs to day D's build.
    assert day_d_summary["lookahead_files"] == 1
    assert day_d1_summary["trades_outside_day"] == 1, (
        "the day-D trade sitting in day D+1's folder was not discarded, so day "
        "D+1 emitted a partial bar for a day-D minute")

    visible = ClockGatedReader(store_root, f"bars_{MINUTE_NS}ns").read_as_of(2**62)
    final = visible[visible[EVENT_TIME] == last_minute]
    assert len(final) == 1
    bar = final.iloc[0]
    assert bar["trades"] == 4, "the complete bar was replaced by the late-arrival-only bar"
    assert bar["volume"] == pytest.approx(6.001)
    assert bar["open"] == pytest.approx(100.0)
    assert bar["high"] == pytest.approx(999.0)
    assert bar["close"] == pytest.approx(999.0)

    # Day D+1 still built its own minute, so the discard cost nothing.
    assert set(visible[EVENT_TIME]) == {last_minute, d1_midnight}


def test_the_index_sidecar_path_follows_the_raw_writer_suffix_constants(monkeypatch):
    """A literal suffix here fails silently the moment the writer's changes.

    `_hour_files` globs on RAW_SUFFIX and the sidecar is derived from IDX_SUFFIX.
    Hardcoding either means a changed constant makes the glob match nothing - a
    build reporting a legitimate-looking zero - or points the sidecar at a file
    that does not exist, in the one module that refuses silent zeros everywhere
    else.
    """
    from store import cli as store_cli

    monkeypatch.setattr(store_cli, "RAW_SUFFIX", ".ndjson.lz4")
    monkeypatch.setattr(store_cli, "IDX_SUFFIX", ".idx.lz4")
    sidecar = store_cli._index_path_for(
        Path("/archive/trade_BTCUSDT_2026-08-02T23.ndjson.lz4"))
    assert sidecar.name == "trade_BTCUSDT_2026-08-02T23.idx.lz4"
