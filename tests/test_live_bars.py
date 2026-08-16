"""A provisional bar that permanently shadows the complete one is the whole risk.

`build_bars` stamps availability from the DATA — `max(bar_close, ingestion)` —
so a live builder that stamped wall-clock time would produce rows the complete
build could never supersede, because its stamp would be earlier. In a store with
no delete path that is an incomplete bar becoming permanent.

`test_a_late_trade_supersedes_the_provisional_bar` is the one that matters. The
rest defend the other half: only closed minutes are emitted, and an unflushed
file is never read as a market with no trades.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from capture.raw_writer import RawWriter
from store.clock_gated_reader import ClockGatedReader
from store.live_bars import (
    BAR_INTERVAL_NS,
    DATASET,
    TRADE_STREAMS,
    build_live_hour,
    build_live_hour_for_venues,
    captured_symbols_this_hour,
    current_hour_utc,
    drop_open_minute,
)

_VENUE, _SYMBOL, _STREAM = "binance", "BTCUSDT", "trade"


def _now_in_hour(minute: int, second: int = 0) -> dt.datetime:
    """A clock inside a fixed hour, so a test never straddles a boundary."""
    return dt.datetime(2026, 8, 16, 12, minute, second, tzinfo=dt.timezone.utc)


def _write_trades(capture_root: Path, now: dt.datetime, minutes: list[int],
                  price: float = 100.0, symbol: str = _SYMBOL,
                  venue: str = _VENUE, stream: str = _STREAM) -> None:
    """One binance trade per minute listed, into that hour's live files."""
    writer = RawWriter(capture_root, venue, stream, symbol)
    hour_start = int(now.replace(minute=0, second=0, microsecond=0
                                 ).timestamp() * 1e9)
    for minute in minutes:
        at_ns = hour_start + minute * BAR_INTERVAL_NS + 1_000_000
        # The combined-stream wrapper the capture writes verbatim, because
        # `_extract_binance` reads `body["data"]` and a bare frame yields nothing
        # - which is exactly the "quiet market" shape these tests exist to catch.
        payload = json.dumps({
            "stream": f"{symbol.lower()}@trade",
            "data": {"e": "trade", "E": at_ns // 1_000_000,
                     "T": at_ns // 1_000_000, "s": symbol,
                     "p": str(price), "q": "1", "t": minute, "m": False}})
        writer.append(payload, t_recv_ns=at_ns, t_exch_ms=at_ns // 1_000_000,
                      seq=None)
    writer.flush()
    writer.close()


# --- the availability stamp, which is the whole risk ----------------------

def test_availability_is_derived_from_the_data_not_the_wall_clock(tmp_path):
    """The first version stamped `now`. `build_bars` stamps
    `max(bar_close, ingestion)`, so the complete build's stamp would be EARLIER
    and `drop_duplicates(keep="last")` would keep the provisional bar forever -
    an incomplete row permanently shadowing the complete one, in a store with no
    delete path."""
    now = _now_in_hour(30)
    _write_trades(tmp_path / "cap", now, minutes=[0, 1, 2])

    build_live_hour(tmp_path / "cap", tmp_path / "store", _VENUE, _STREAM,
                    [_SYMBOL], now=now)

    frame = pd.concat([pd.read_parquet(p) for p in
                       (tmp_path / "store" / DATASET / f"symbol={_SYMBOL}"
                        ).glob("*.parquet")])
    wall_clock_ns = int(now.timestamp() * 1e9)
    assert (frame.availability_time_ns < wall_clock_ns).all(), (
        "a stamp at or after the pass clock means wall-clock stamping is back")
    # Every bar becomes knowable at its close, or later if a trade arrived late.
    assert (frame.availability_time_ns >= frame.event_time_ns
            + BAR_INTERVAL_NS).all()


def test_a_late_trade_supersedes_the_provisional_bar(tmp_path):
    """The correction path, end to end through the real reader.

    A trade for minute 1 that arrives after minute 1 closed gives the complete
    build a LATER ingestion time, hence a later availability, hence the reader
    serves it over the provisional bar. The correction fires in exactly the case
    where the two bars differ, which is what the bitemporal store is for.
    """
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1], price=100.0)
    build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)

    provisional = ClockGatedReader(store, DATASET).read_as_of(2**62)
    minute_one = int(provisional[provisional.event_time_ns
                                 == provisional.event_time_ns.max()].close.iloc[0])
    assert minute_one == 100

    # Now the same minute is rebuilt with a trade that arrived late and moved it.
    hour_start = int(now.replace(minute=0, second=0, microsecond=0)
                     .timestamp() * 1e9)
    complete = pd.DataFrame([{
        "symbol": _SYMBOL, "venue": _VENUE,
        "event_time_ns": hour_start + BAR_INTERVAL_NS,
        "ingestion_time_ns": hour_start + 50 * BAR_INTERVAL_NS,
        "availability_time_ns": hour_start + 50 * BAR_INTERVAL_NS,
        "open": 100.0, "high": 900.0, "low": 100.0, "close": 900.0,
        "volume": 2.0, "trades": 2}])
    from store.parquet_partition import append_partition
    append_partition(store, DATASET, complete, "complete-rebuild")

    served = ClockGatedReader(store, DATASET).read_as_of(2**62)
    latest = served[served.event_time_ns == hour_start + BAR_INTERVAL_NS]
    assert len(latest) == 1, "the reader must serve one version of the bar"
    assert float(latest.close.iloc[0]) == 900.0, (
        "the complete bar must supersede the provisional one")


# --- only closed minutes -------------------------------------------------

def test_the_minute_containing_the_newest_trade_is_dropped():
    """The readable prefix ends somewhere inside the last flush interval, so that
    minute may be missing trades that already happened. Publishing it emits a bar
    that is wrong until the minute closes, and wrong invisibly."""
    frame = pd.DataFrame({"symbol": ["BTCUSDT"] * 3,
                          "event_time_ns": [0, BAR_INTERVAL_NS,
                                            2 * BAR_INTERVAL_NS]})
    newest = {"BTCUSDT": 2 * BAR_INTERVAL_NS + 30_000_000_000}  # mid 3rd minute

    kept, dropped = drop_open_minute(frame, newest)

    assert list(kept.event_time_ns) == [0, BAR_INTERVAL_NS]
    assert dropped == 1


def test_each_symbol_is_cut_at_its_own_readable_prefix():
    """The prefix is a property of a FILE, and every symbol has its own.

    Cutting a quiet symbol at a busy symbol's boundary publishes the quiet
    symbol's still-open minute as though it had closed - this function's whole
    purpose, defeated exactly for the symbols nobody checks.
    """
    frame = pd.DataFrame({
        "symbol": ["BUSY", "BUSY", "QUIET", "QUIET"],
        "event_time_ns": [BAR_INTERVAL_NS, 2 * BAR_INTERVAL_NS,
                          BAR_INTERVAL_NS, 2 * BAR_INTERVAL_NS]})
    # BUSY flushed a moment ago; QUIET's file has not been flushed since minute 2.
    newest = {"BUSY": 3 * BAR_INTERVAL_NS + 1,
              "QUIET": 2 * BAR_INTERVAL_NS + 1}

    kept, dropped = drop_open_minute(frame, newest)

    assert dropped == 1, "QUIET's minute 2 is still open at ITS prefix"
    assert sorted(map(tuple, kept[["symbol", "event_time_ns"]].values.tolist())) == [
        ("BUSY", BAR_INTERVAL_NS), ("BUSY", 2 * BAR_INTERVAL_NS),
        ("QUIET", BAR_INTERVAL_NS)]


def test_a_build_emits_only_closed_minutes(tmp_path):
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1, 2, 3])

    report = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)

    assert report.bars_written == 3, "minute 3 holds the newest trade"
    assert report.minutes_dropped_as_open == 1


def test_an_empty_frame_drops_nothing():
    kept, dropped = drop_open_minute(
        pd.DataFrame({"symbol": [], "event_time_ns": []}), {})

    assert kept.empty and dropped == 0


# --- an unflushed file is not a quiet market -----------------------------

def test_nothing_flushed_is_counted_apart_from_no_trades(tmp_path):
    """The documented trap: a file empty six and a half hours late "read as a
    market with no trades, in a build that reported success"."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    date, hour = current_hour_utc(now)
    folder = capture / "raw" / _VENUE / date
    folder.mkdir(parents=True)
    stem = f"{_STREAM}_{_SYMBOL}_{date}T{hour}"
    (folder / f"{stem}.ndjson.zst").write_bytes(b"")
    (folder / f"{stem}.idx.zst").write_bytes(b"")

    report = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)

    assert report.symbols_nothing_flushed == 1
    assert report.symbols_no_trades == 0
    assert report.bars_written == 0
    assert "nothing flushed" in report.describe()


def test_a_missing_file_is_not_counted_as_a_symbol_read(tmp_path):
    """A symbol with no file at all this hour has not been read, and counting it
    would make a venue that stopped capturing look like a quiet one."""
    report = build_live_hour(tmp_path / "cap", tmp_path / "store", _VENUE,
                             _STREAM, [_SYMBOL], now=_now_in_hour(30))

    assert report.symbols_read == 0
    assert report.symbols_nothing_flushed == 0


# --- not rewriting the hour every pass -----------------------------------

def test_a_second_pass_writes_only_new_minutes(tmp_path):
    """Without the high-water mark a pass every 60 seconds rewrites the whole
    hour, so by minute 59 the store holds 59 identical copies of minute 0."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1, 2])
    written: dict[str, int] = {}

    first = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL],
                            now=now, written_through=written)
    second = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL],
                             now=now, written_through=written)

    assert first.bars_written == 2
    assert second.bars_written == 0
    assert second.minutes_already_written == 2


def test_a_quiet_symbol_does_not_inherit_a_busy_symbols_high_water(tmp_path):
    """The mark is per symbol, so a symbol whose file flushes late still gets its
    minutes. One venue-wide mark set from the busiest symbol would suppress them
    permanently, and the daily build is the only thing that would ever notice."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1, 2, 3], symbol="BUSYUSDT")
    _write_trades(capture, now, minutes=[0, 1], symbol="QUIETUSDT")
    written: dict = {}
    build_live_hour(capture, store, _VENUE, _STREAM,
                    ["BUSYUSDT", "QUIETUSDT"], now=now, written_through=written)

    # QUIET's file is flushed further, carrying the minutes it was behind on.
    _write_trades(capture, now, minutes=[2, 3], symbol="QUIETUSDT")
    second = build_live_hour(capture, store, _VENUE, _STREAM,
                             ["BUSYUSDT", "QUIETUSDT"], now=now,
                             written_through=written)

    # Pass 1 wrote QUIET's minute 0 only - minute 1 was still open at its own
    # prefix - so minutes 1 and 2 are the newly readable ones.
    assert second.bars_written == 2
    assert all("QUIETUSDT" in str(part) for part in second.parts), (
        "BUSY was fully written in pass 1 and must contribute nothing")


def test_a_restart_re_reading_the_hour_does_not_kill_the_pass(tmp_path):
    """The mark lives in memory, so a restarted loop rebuilds minutes already in
    the store. `append_partition` refuses a rewrite by raising, and an uncaught
    raise takes down the loop the whole intraday path depends on."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1, 2])

    first = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)
    restarted = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL],
                                now=now)          # no mark, as after a restart

    assert first.bars_written == 2
    assert restarted.bars_written == 0
    assert restarted.minutes_already_written == 2, (
        "the identical snapshot must be counted, not raised")


def test_a_late_flush_into_the_last_minute_is_written_not_refused(tmp_path):
    """The trade count is in the snapshot id so a bar that CHANGED is not mistaken
    for one already written. Coverage alone would refuse it, and the corrected bar
    would never reach the store."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    _write_trades(capture, now, minutes=[0, 1, 2])
    build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)

    # A second trade lands in minute 1, at a different price, and minute 2 still
    # bounds the readable prefix - so coverage is unchanged and only the content moved.
    _write_trades(capture, now, minutes=[1], price=500.0)
    again = build_live_hour(capture, store, _VENUE, _STREAM, [_SYMBOL], now=now)

    assert again.bars_written == 2, "changed content must be written"
    served = ClockGatedReader(store, DATASET).read_as_of(2**62)
    hour_start = int(now.replace(minute=0, second=0, microsecond=0)
                     .timestamp() * 1e9)
    minute_one = served[served.event_time_ns == hour_start + BAR_INTERVAL_NS]
    assert len(minute_one) == 1
    assert float(minute_one.high.iloc[0]) == 500.0


def test_the_loop_entry_point_threads_the_high_water_mark(tmp_path, monkeypatch):
    """Reached by `main`, not only by tests.

    The first version created no mark in `main`, so every 60-second pass rewrote
    the whole hour while a green test asserted the mark worked - `tail_specs()`
    again, which this codebase's own CLAUDE.md names as the trap it keeps falling
    into.
    """
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    _write_trades(capture, now, minutes=[0, 1, 2])

    from store import live_bars

    passes: list[int] = []
    real_build = live_bars.build_live_hour_for_venues
    marks: list[int] = []

    def record(*args, written_through=None, **kwargs):
        marks.append(id(written_through))
        reports = real_build(*args, written_through=written_through, **kwargs)
        passes.append(sum(r.bars_written for r in reports))
        if len(passes) == 3:
            raise KeyboardInterrupt
        return reports

    monkeypatch.setattr(live_bars, "build_live_hour_for_venues", record)
    monkeypatch.setattr(live_bars, "print", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(KeyboardInterrupt):
        live_bars.main(["--capture-root", str(capture), "--store-root",
                        str(store), "--venue", f"{_VENUE}={_SYMBOL}",
                        "--interval-seconds", "1"])

    assert None not in marks and len(set(marks)) == 1, (
        "every pass must share ONE mark; a fresh dict per pass is no mark at all")
    assert passes[0] == 2, "the first pass writes the closed minutes"
    assert passes[1:] == [0, 0], (
        "later passes must write nothing, not another copy of the hour")


# --- the stream map is shared, not restated ------------------------------

def test_the_stream_map_is_the_one_store_cli_uses():
    """`store.cli`'s own comment records what a second copy costs: binance-spot
    absent for 1,363 symbols, and coinbase captured for a day with a working
    extractor the builder could not reach."""
    from store.cli import _TRADE_STREAMS

    assert TRADE_STREAMS is _TRADE_STREAMS


def test_all_reads_the_symbols_off_the_archive(tmp_path):
    """A hand-written list is how 99.6% of the tape went unbuilt in August: the
    capture subscribed 2,098 symbols and the supervisor asked for 9."""
    capture, store = tmp_path / "cap", tmp_path / "store"
    now = _now_in_hour(30)
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        _write_trades(capture, now, minutes=[0, 1, 2], symbol=symbol)

    reports = build_live_hour_for_venues(capture, store, {_VENUE: ["ALL"]},
                                         now=now)

    assert [r.symbols_read for r in reports] == [3]
    assert reports[0].bars_written == 6           # two closed minutes per symbol


def test_all_names_the_symbols_and_strips_the_stream_and_hour(tmp_path):
    now = _now_in_hour(30)
    _write_trades(tmp_path, now, minutes=[0], symbol="ETHUSDT")
    date, hour = current_hour_utc(now)

    found = captured_symbols_this_hour(tmp_path, _VENUE, _STREAM, date, hour)

    assert found == ["ETHUSDT"]


def test_all_on_a_venue_with_no_capture_this_hour_is_empty_not_an_error(tmp_path):
    date, hour = current_hour_utc(_now_in_hour(30))

    assert captured_symbols_this_hour(tmp_path, _VENUE, _STREAM, date, hour) == []


def test_a_venue_with_no_stream_mapping_is_skipped_not_guessed(tmp_path):
    """Inventing a stream name is how a venue gets captured and never built."""
    reports = build_live_hour_for_venues(
        tmp_path / "cap", tmp_path / "store",
        {"a-venue-nobody-mapped": [_SYMBOL]}, now=_now_in_hour(30))

    assert reports == []
