# tests/test_store_cli.py
"""The build must refuse damaged input rather than quietly producing fewer bars."""
from __future__ import annotations

import datetime as dt
import json
import os
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


def _append_until_compressed_bytes_reach_disk(writer, first_receive_ns: int) -> Path:
    """Feed a live `RawWriter` until zstd spills a partial frame to disk.

    A writer holding an hour open buffers inside the compressor, so a handful of
    frames leaves an EMPTY file that reads back clean and proves nothing. On a
    real stream the buffer fills within seconds and the hour on disk ends in an
    unfinished frame - which is what `read_pair` refuses. Feeding until the file
    is non-empty reproduces that state deterministically instead of depending on
    how much a fixed frame count happens to compress to.

    Receive times advance by a millisecond so the writer's 30 s flush cadence
    never fires: a frame boundary would leave the tail complete and readable,
    which is the one state this helper must not produce.
    """
    import random

    from capture.raw_writer import paths_for

    raw_path, _ = paths_for(writer._root, writer._venue, writer._stream, writer._symbol,
                            dt.datetime.fromtimestamp(
                                first_receive_ns / SECOND_NS,
                                tz=dt.timezone.utc).strftime("%Y-%m-%dT%H"))
    noise = random.Random(20260803)
    for index in range(200_000):
        receive_ns = first_receive_ns + index * 1_000_000
        event_ns = receive_ns
        writer.append(
            _binance_trade_frame("BTCUSDT", round(noise.random() * 100_000, 6),
                                 round(noise.random(), 8), event_ns),
            receive_ns, event_ns // 1_000_000, None)
        if raw_path.exists() and raw_path.stat().st_size > 0:
            return raw_path
    raise AssertionError("the writer never spilled compressed bytes to disk")


def test_a_lookahead_hour_a_live_writer_still_holds_open_is_skipped_not_read(tmp_path):
    """Building yesterday while capture runs must not abort on today's open hour.

    The lookahead reaches into the NEXT day's folder, and on a running capture
    that day's current hour is held open by `RawWriter` with an unfinished zstd
    frame on disk. `read_pair` refuses a torn file - correctly, for a closed hour
    - so selecting lookahead candidates on existence alone aborts the build of a
    day the operator did ask for, naming a file they did not, and the documented
    repair (`reconcile_pair`) refuses that file too because a writer holds it.
    Skipping the live hour costs nothing: those trades are still in the file and
    day D+1's own build reads them from its own folder once the hour is closed.
    """
    from capture.raw_writer import RawWriter, is_hour_being_written, read_pair

    day_d, day_d1 = "2026-08-02", "2026-08-03"
    d_midnight, d1_midnight = _midnight_ns(day_d), _midnight_ns(day_d1)

    settled = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    for offset in range(3):
        event_ns = d_midnight + offset * SECOND_NS
        settled.append(_binance_trade_frame("BTCUSDT", 100.0 + offset, 1.0, event_ns),
                       event_ns + 1_000_000, event_ns // 1_000_000, None)
    settled.close()

    live = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    try:
        open_hour = _append_until_compressed_bytes_reach_disk(live, d1_midnight)

        # The premise of the test, asserted rather than assumed: this hour is
        # both live and unreadable, which is exactly the pair of facts that made
        # the old lookahead abort.
        assert is_hour_being_written(open_hour)[0] is True
        with pytest.raises(Exception, match="incomplete|unreadable|newline"):
            read_pair(open_hour, open_hour.with_name(
                open_hour.name.replace(".ndjson.zst", ".idx.zst")))

        summary = build_bars_for_day(
            capture_root=tmp_path, store_root=tmp_path / "store", venue="binance",
            date=day_d, symbols=["BTCUSDT"], interval_ns=MINUTE_NS)
    finally:
        live.close()

    assert summary["bars"] == 1
    assert summary["lookahead_files"] == 0, "the live hour was read instead of skipped"
    assert [Path(path).name for path in summary["lookahead_files_skipped_live"]] == [
        "trade_BTCUSDT_2026-08-03T00.ndjson.zst"], (
        "a skipped lookahead hour that nothing reports is silent loss")


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


def test_a_rebuild_is_refused_before_a_single_frame_is_parsed(tmp_path, monkeypatch):
    """The refusal is the same one; what changes is when it costs.

    The collision used to be discovered inside `append_partition`, at the very
    end - so a re-run paid the full read before learning there was nothing to do.
    Measured on 2026-08-08: a 50-symbol rebuild cost 27.7s against 27.6s for the
    original. At 569 symbols on an hourly supervisor that is ~34 minutes of
    parsing every hour to rediscover "already built".

    The snapshot id is a digest of the input FILES, so it is knowable before any
    of them is decompressed. `read_pair` here refuses to be called at all, which
    is what makes this a test of the pre-check rather than of the outcome.
    """
    from capture.frame_codec import IndexEntry
    from store.parquet_partition import PartitionExistsError
    from store import cli as store_cli

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    (source / "trade_BTCUSDT_2026-08-02T00.ndjson.zst").write_bytes(b"placeholder")
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

    def refuse_to_read(raw, idx):
        raise AssertionError(f"parsed {raw} on a day already built")

    monkeypatch.setattr(store_cli, "read_pair", refuse_to_read)
    with pytest.raises(PartitionExistsError):
        build()


def test_a_batched_run_builds_every_symbol_without_holding_them_all(tmp_path, monkeypatch):
    """Peak memory is set by the batch, not by how many symbols were asked for.

    `build_bars_for_day` accumulates every trade of every requested symbol into
    one list before building. Measured 2026-08-08: peak RSS rose 30 -> 35 -> 44 MB
    per symbol across 50/100/150 symbols, so the 569-symbol universe extrapolates
    to ~25 GB for five hours of tape and far more for a full day. Batching is what
    makes the broad universe buildable at all, and each batch is its own snapshot
    because it reads its own set of files.
    """
    from capture.frame_codec import IndexEntry
    from store import cli as store_cli

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        (source / f"trade_{symbol}_2026-08-02T00.ndjson.zst").write_bytes(symbol.encode())
        (source / f"trade_{symbol}_2026-08-02T00.idx.zst").write_bytes(b"placeholder")

    entry = IndexEntry(n=0, t_recv_ns=1785685177508349176, t_exch_ms=1785685177439,
                       seq=None, kind="data", esc=False)

    def one_trade_per_file(raw, idx):
        symbol = Path(raw).name.split("_")[1]
        return [('{"stream":"x@trade","data":{"e":"trade","T":1785685177439,'
                 f'"s":"{symbol}","p":"100.0","q":"1.0"}}}}', entry)]

    monkeypatch.setattr(store_cli, "read_pair", one_trade_per_file)
    largest_batch = []
    real_build = store_cli.build_bars_for_day
    monkeypatch.setattr(store_cli, "build_bars_for_day",
                        lambda *a, **k: (largest_batch.append(len(a[4] if len(a) > 4
                                                                  else k["symbols"])),
                                         real_build(*a, **k))[1])

    code = store_cli.main([
        "--venue", "binance", "--date", "2026-08-02", "--symbols", "BTCUSDT,ETHUSDT",
        "--capture-root", str(tmp_path), "--store-root", str(tmp_path / "store"),
        "--batch-size", "1"])

    assert code == 0
    assert largest_batch == [1, 1]
    dataset = tmp_path / "store" / "bars_60000000000ns"
    assert sorted(p.name for p in dataset.iterdir()) == ["symbol=BTCUSDT", "symbol=ETHUSDT"]


def _two_symbol_capture(tmp_path, monkeypatch):
    from capture.frame_codec import IndexEntry
    from store import cli as store_cli

    source = tmp_path / "raw" / "binance" / "2026-08-02"
    source.mkdir(parents=True)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        (source / f"trade_{symbol}_2026-08-02T00.ndjson.zst").write_bytes(symbol.encode())
        (source / f"trade_{symbol}_2026-08-02T00.idx.zst").write_bytes(b"placeholder")

    entry = IndexEntry(n=0, t_recv_ns=1785685177508349176, t_exch_ms=1785685177439,
                       seq=None, kind="data", esc=False)

    def one_trade_per_file(raw, idx):
        symbol = Path(raw).name.split("_")[1]
        return [('{"stream":"x@trade","data":{"e":"trade","T":1785685177439,'
                 f'"s":"{symbol}","p":"100.0","q":"1.0"}}}}', entry)]

    monkeypatch.setattr(store_cli, "read_pair", one_trade_per_file)
    return store_cli


def test_a_batch_already_built_does_not_stop_the_batches_after_it(tmp_path, monkeypatch):
    """Batching makes a partly-built day possible, so a re-run must finish it.

    Unbatched, a day was all-or-nothing. Split into 114 batches, a run killed
    halfway leaves the first half built - and if the re-run stopped at the first
    collision the remaining symbols could never be built at all, with the archive
    aged out from under them seven days later.
    """
    store_cli = _two_symbol_capture(tmp_path, monkeypatch)
    common = ["--venue", "binance", "--date", "2026-08-02",
              "--capture-root", str(tmp_path), "--store-root", str(tmp_path / "store"),
              "--batch-size", "1"]

    assert store_cli.main([*common, "--symbols", "BTCUSDT"]) == 0
    assert store_cli.main([*common, "--symbols", "BTCUSDT,ETHUSDT"]) == 0

    dataset = tmp_path / "store" / "bars_60000000000ns"
    assert sorted(p.name for p in dataset.iterdir()) == ["symbol=BTCUSDT", "symbol=ETHUSDT"]


def test_a_run_whose_every_batch_was_already_built_still_reports_it(tmp_path, monkeypatch):
    """Nothing to do must stay distinguishable from something done.

    The supervisor reads a non-zero exit as "already built" and logs it as such.
    If a fully-redundant run exited 0, every hourly pass would read as a fresh
    build and the run log would stop being evidence of anything.
    """
    store_cli = _two_symbol_capture(tmp_path, monkeypatch)
    common = ["--venue", "binance", "--date", "2026-08-02", "--symbols", "BTCUSDT,ETHUSDT",
              "--capture-root", str(tmp_path), "--store-root", str(tmp_path / "store"),
              "--batch-size", "1"]

    assert store_cli.main(common) == 0
    assert store_cli.main(common) == store_cli.EXIT_ALREADY_BUILT


def test_rebuilding_a_day_after_more_next_day_trades_land_is_still_refused(tmp_path):
    """The build's identity is the day it builds, not the lookahead it consulted.

    Digesting the lookahead files into the snapshot id makes the id move whenever
    the NEXT day's folder grows - which it does continuously while capture runs.
    A rebuild of an unchanged day then computes a fresh id, sails past
    `append_partition`'s collision check and appends a duplicate part. Rows stay
    correct only because the reader de-duplicates; the guard against accidental
    re-runs is gone and parts accumulate on every run.
    """
    from capture.raw_writer import RawWriter
    from store.parquet_partition import PartitionExistsError

    day_d, day_d1 = "2026-08-02", "2026-08-03"
    d1_midnight = _midnight_ns(day_d1)
    last_minute = d1_midnight - MINUTE_NS

    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    writer.append(_binance_trade_frame("BTCUSDT", 100.0, 1.0, last_minute + SECOND_NS),
                  last_minute + 2 * SECOND_NS, (last_minute + SECOND_NS) // 1_000_000, None)
    writer.append(_binance_trade_frame("BTCUSDT", 200.0, 2.0, d1_midnight + SECOND_NS),
                  d1_midnight + 2 * SECOND_NS, (d1_midnight + SECOND_NS) // 1_000_000, None)
    writer.close()

    build = lambda: build_bars_for_day(
        capture_root=tmp_path, store_root=tmp_path / "store", venue="binance",
        date=day_d, symbols=["BTCUSDT"], interval_ns=MINUTE_NS)

    first = build()
    day_d_hour = (tmp_path / "raw" / "binance" / day_d /
                  f"trade_BTCUSDT_{day_d}T23.ndjson.zst")
    day_d_bytes = day_d_hour.read_bytes()

    later = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    later.append(_binance_trade_frame("BTCUSDT", 300.0, 3.0, d1_midnight + 30 * SECOND_NS),
                 d1_midnight + 31 * SECOND_NS,
                 (d1_midnight + 30 * SECOND_NS) // 1_000_000, None)
    later.close()

    assert day_d_hour.read_bytes() == day_d_bytes, "day D's own bytes were touched"
    with pytest.raises(PartitionExistsError):
        build()
    assert first["snapshot_id"] is not None


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
    assert day_d1_summary["trades_covered_by_previous_day"] == 1, (
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


def test_a_trade_stranded_beyond_the_lookahead_window_is_counted_and_quarantined(tmp_path):
    """A trade no build will ever pick up must leave a file behind, not a counter.

    A trade whose event time precedes the day being built is recoverable only if
    that earlier day's own lookahead would have reached its hour file. Beyond
    that window nothing reads it: the earlier day's build never looks that far,
    and this day's event-time filter discards it. Counting it alongside the
    ordinary deferrals - tens of thousands per day on this archive - buries the
    one number that means data is gone, so it is counted separately and every
    such trade is written to quarantine where it can still be recovered.
    """
    from capture.raw_writer import RawWriter

    day_d = "2026-08-02"
    d_midnight = _midnight_ns(day_d)
    stranded_event_ns = d_midnight - 30 * SECOND_NS          # belongs to day D-1
    stranded_receive_ns = d_midnight + 5 * 3600 * SECOND_NS  # filed in day D's hour 05

    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    writer.append(_binance_trade_frame("BTCUSDT", 100.0, 1.0, d_midnight + SECOND_NS),
                  d_midnight + 2 * SECOND_NS, (d_midnight + SECOND_NS) // 1_000_000, None)
    writer.append(_binance_trade_frame("BTCUSDT", 99.5, 0.25, stranded_event_ns),
                  stranded_receive_ns, stranded_event_ns // 1_000_000, None)
    writer.close()

    store_root = tmp_path / "store"
    summary = build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=day_d,
        symbols=["BTCUSDT"], interval_ns=MINUTE_NS, lookahead_hours=2)

    assert summary["trades_stranded"] == 1
    assert summary["trades_covered_by_previous_day"] == 0
    assert summary["trades_deferred_to_next_day"] == 0

    quarantine = (store_root / "quarantine" /
                  f"stranded-binance-{day_d}-{summary['snapshot_id']}.ndjson")
    assert Path(summary["quarantine_file"]) == quarantine
    records = [json.loads(line) for line in
               quarantine.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["symbol"] == "BTCUSDT"
    assert record["venue"] == "binance"
    assert record["price"] == pytest.approx(99.5)
    assert record["size"] == pytest.approx(0.25)
    assert record["event_time_ns"] == stranded_event_ns
    assert record["ingestion_time_ns"] == stranded_receive_ns
    assert record["source_file"].endswith(f"trade_BTCUSDT_{day_d}T05.ndjson.zst")


def test_a_rebuild_once_the_skipped_lookahead_hour_closes_serves_the_complete_bar(tmp_path):
    """A build that skipped a live hour must not lock the day into its partial bar.

    Two earlier fixes combine into permanent corruption if the snapshot id ignores
    what the build skipped. Skipping a live lookahead hour leaves day D's last bar
    built from its own hour 23 alone - three trades here, high 102.0, volume 3.0 -
    while the late arrival with a day-D event time sits unread in day D+1's open
    hour 00. Digesting only day D's OWN files then gives the complete rebuild the
    SAME id as the partial build, so `append_partition` refuses it, the partial bar
    is permanent in a store with no delete path, and day D+1's own build discards
    that trade by event time and reports `trades_covered_by_previous_day=1,
    trades_stranded=0` - a false all-clear over a bar that is quietly wrong.

    Folding the skipped hours' identities into the id makes the incomplete build a
    different build, which is what it is. The rebuild once the hour closes then
    appends, and because its late trade arrived later its bar carries a later
    availability time - so the reader's correction resolution returns the complete
    bar over the partial one, which is the bitemporal design working as designed
    rather than being worked around.
    """
    from capture.raw_writer import RawWriter
    from store.clock_gated_reader import ClockGatedReader
    from store.temporal_schema import EVENT_TIME

    day_d, day_d1 = "2026-08-02", "2026-08-03"
    d1_midnight = _midnight_ns(day_d1)
    last_minute = d1_midnight - MINUTE_NS

    settled = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    for price, offset in ((100.0, 10), (102.0, 20), (101.0, 30)):
        event_ns = last_minute + offset * SECOND_NS
        settled.append(_binance_trade_frame("BTCUSDT", price, 1.0, event_ns),
                       event_ns + 100_000_000, event_ns // 1_000_000, None)
    settled.close()

    # Event time 23:59:59.9 on day D, received 50 ms into day D+1: filed under day
    # D+1's hour 00, which a live writer still holds open.
    late_event_ns = d1_midnight - 100_000_000
    live = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    live.append(_binance_trade_frame("BTCUSDT", 999.0, 3.0, late_event_ns),
                d1_midnight + 50_000_000, late_event_ns // 1_000_000, None)

    store_root = tmp_path / "store"
    build = lambda: build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=day_d,
        symbols=["BTCUSDT"], interval_ns=MINUTE_NS)

    try:
        partial = build()
    finally:
        live.close()

    assert [Path(path).name for path in partial["lookahead_files_skipped_live"]] == [
        f"trade_BTCUSDT_{day_d1}T00.ndjson.zst"]
    assert partial["bars"] == 1

    # The hour is closed now, so the rebuild skips nothing and reads the late
    # trade. It is a different build and its id must say so.
    complete = build()
    assert complete["lookahead_files"] == 1
    assert complete["lookahead_files_skipped_live"] == []
    assert complete["snapshot_id"] != partial["snapshot_id"], (
        "a build that skipped a live hour and one that read it are not the same "
        "build, and an id that cannot tell them apart makes the partial bar permanent")

    visible = ClockGatedReader(store_root, f"bars_{MINUTE_NS}ns").read_as_of(2**62)
    final = visible[visible[EVENT_TIME] == last_minute]
    assert len(final) == 1, "the correction did not replace the partial bar"
    bar = final.iloc[0]
    assert bar["trades"] == 4, "the reader still serves the bar built without the late trade"
    assert bar["volume"] == pytest.approx(6.0)
    assert bar["open"] == pytest.approx(100.0)
    assert bar["high"] == pytest.approx(999.0)
    assert bar["low"] == pytest.approx(100.0)
    assert bar["close"] == pytest.approx(999.0)


def test_a_build_that_raises_leaves_no_quarantine_file_to_block_the_rerun(tmp_path):
    """A failed build must not brick the day it failed on.

    The quarantine file is named by the snapshot id, and that id derives from the
    raw files alone - it does not move when `interval_ns` does. So a build that
    raised AFTER writing quarantine left a file whose name refused every later
    attempt with `QuarantineExistsError`, including the operator's re-run with the
    correct interval. With no delete path in this store, the day could then only be
    built by hand-removing a file the error text never mentions.

    Writing the quarantine only once the build has otherwise succeeded keeps the
    record of unrecoverable trades exactly where it belongs - on runs that produced
    a store to be missing them from.
    """
    from capture.raw_writer import RawWriter
    from store.cli import BarOutsideBuildDay

    day_d = "2026-08-02"
    d_midnight = _midnight_ns(day_d)
    stranded_event_ns = d_midnight - 30 * SECOND_NS          # belongs to day D-1
    stranded_receive_ns = d_midnight + 5 * 3600 * SECOND_NS  # filed in day D's hour 05

    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    writer.append(_binance_trade_frame("BTCUSDT", 100.0, 1.0, d_midnight + SECOND_NS),
                  d_midnight + 2 * SECOND_NS, (d_midnight + SECOND_NS) // 1_000_000, None)
    writer.append(_binance_trade_frame("BTCUSDT", 99.5, 0.25, stranded_event_ns),
                  stranded_receive_ns, stranded_event_ns // 1_000_000, None)
    writer.close()

    store_root = tmp_path / "store"
    build = lambda interval_ns: build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=day_d,
        symbols=["BTCUSDT"], interval_ns=interval_ns, lookahead_hours=2)

    # Seven minutes does not divide a UTC day, so the day's first bar opens before
    # the day does and the stray-bar check refuses the whole build.
    with pytest.raises(BarOutsideBuildDay):
        build(7 * MINUTE_NS)
    assert not (store_root / "quarantine").exists(), (
        "a build that stored nothing left a quarantine file whose name refuses the re-run")

    summary = build(MINUTE_NS)
    assert summary["bars"] == 1
    assert summary["trades_stranded"] == 1
    assert Path(summary["quarantine_file"]).exists()


def test_a_days_old_trade_is_stranded_whichever_hour_it_was_filed_in(tmp_path):
    """The alarm must fire on the trade's event time, not on where it landed.

    `_is_within_previous_days_lookahead` tested only the hour file's position, so a
    trade three days stale - or one carrying `event_time_ns=0`, which
    `_extract_binance` will happily produce from a garbage `T` - read as covered by
    the previous day's build purely because it was filed in day D's hour 00. The
    identical trades filed in hour 05 were reported stranded. Whether data loss
    raised an alarm depended on which hour file the trade happened to land in.

    No build reaches either one: the previous day's lookahead reads day D's early
    hours but discards anything outside ITS day too, and this day discards them by
    event time. They are stranded, and they belong in quarantine.
    """
    from capture.raw_writer import RawWriter

    day_d = "2026-08-02"
    d_midnight = _midnight_ns(day_d)
    day_ns = 86_400 * SECOND_NS
    stale_event_ns = d_midnight - 3 * day_ns   # three days before the day being built
    inside_hour_00_ns = d_midnight + 10 * SECOND_NS

    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    writer.append(_binance_trade_frame("BTCUSDT", 100.0, 1.0, d_midnight + SECOND_NS),
                  d_midnight + 2 * SECOND_NS, (d_midnight + SECOND_NS) // 1_000_000, None)
    writer.append(_binance_trade_frame("BTCUSDT", 99.5, 0.25, stale_event_ns),
                  inside_hour_00_ns, stale_event_ns // 1_000_000, None)
    # A garbage venue timestamp reaches this code unmodified: nothing rejects T=0.
    writer.append(_binance_trade_frame("BTCUSDT", 98.0, 0.5, 0),
                  inside_hour_00_ns + SECOND_NS, 0, None)
    writer.close()

    store_root = tmp_path / "store"
    summary = build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=day_d,
        symbols=["BTCUSDT"], interval_ns=MINUTE_NS, lookahead_hours=2)

    assert summary["trades_covered_by_previous_day"] == 0, (
        "a trade older than the previous day was called recoverable because of the "
        "hour it was filed in")
    assert summary["trades_stranded"] == 2
    records = [json.loads(line) for line in
               Path(summary["quarantine_file"]).read_text(encoding="utf-8").splitlines()]
    assert sorted(record["event_time_ns"] for record in records) == [0, stale_event_ns]


def test_a_trade_deferred_to_the_next_day_is_neither_stranded_nor_quarantined(tmp_path):
    """The ordinary case must never trip the alarm the stranded count exists to be.

    A trade whose event time falls after the day being built is not lost: day D+1
    reads it from its own folder. `trades_outside_day` counted it identically to a
    genuinely unrecoverable trade and reached 47,401 on one real day, so it could
    never function as a signal. Deferrals stay a plain count and write nothing.
    """
    from capture.raw_writer import RawWriter

    day_d, day_d1 = "2026-08-02", "2026-08-03"
    d1_midnight = _midnight_ns(day_d1)
    last_minute = d1_midnight - MINUTE_NS

    writer = RawWriter(tmp_path, "binance", "trade", "BTCUSDT")
    writer.append(_binance_trade_frame("BTCUSDT", 100.0, 1.0, last_minute + SECOND_NS),
                  last_minute + 2 * SECOND_NS, (last_minute + SECOND_NS) // 1_000_000, None)
    writer.append(_binance_trade_frame("BTCUSDT", 200.0, 2.0, d1_midnight + SECOND_NS),
                  d1_midnight + 2 * SECOND_NS, (d1_midnight + SECOND_NS) // 1_000_000, None)
    writer.close()

    store_root = tmp_path / "store"
    summary = build_bars_for_day(
        capture_root=tmp_path, store_root=store_root, venue="binance", date=day_d,
        symbols=["BTCUSDT"], interval_ns=MINUTE_NS, lookahead_hours=2)

    assert summary["trades_deferred_to_next_day"] == 1
    assert summary["trades_stranded"] == 0
    assert summary["quarantine_file"] is None
    assert not (store_root / "quarantine").exists()


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


def test_the_module_runs_as_a_module(tmp_path):
    """Imported and executed are different, and only one of them is shipped.

    Every other test here calls `main()` after importing, which defines the whole
    module first. Running `python -m store.cli` executes top to bottom instead, so
    a helper defined below the `__main__` guard does not exist yet when `main()`
    reaches it. That happened on 2026-08-08: the pre-check helper was appended to
    the end of the file, the suite stayed green, and the first real invocation
    died with NameError. A subprocess is the only shape of test that can see it.
    """
    import subprocess
    import sys as _sys

    result = subprocess.run(
        [_sys.executable, "-m", "store.cli", "--venue", "binance",
         "--date", "2026-08-02", "--symbols", "BTCUSDT",
         "--capture-root", str(tmp_path), "--store-root", str(tmp_path / "store")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})

    assert result.returncode == 0, result.stderr
    assert "NameError" not in result.stderr
