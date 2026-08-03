# src/store/cli.py
"""Builds the bitemporal store from the raw archive.

Reads through `capture.raw_writer.read_pair`, which refuses a torn file rather
than returning its readable prefix. That refusal is load-bearing here: silently
building from a truncated hour produces a store that is quietly missing trades,
and every statistic computed from it is wrong in a way nothing reports.

A day is built from EVENT time, never from which folder a frame landed in.
`RawWriter` rotates hour files on receive time, so a trade timestamped 23:59:59.9
that arrived 50 ms after midnight is filed under the next day. Building each day
from its own folder alone made day D+1 emit a second bar for a day-D minute -
built from that one late trade, carrying a later availability time - and the
reader resolves corrections by (symbol, venue, event_time) with the latest
version winning, so the partial bar replaced the complete one and the complete
one became unreachable. Measured arrival lag on this archive reaches 141 s, so
that corrupted the last bar of essentially every day. `build_bars_for_day`
therefore reads a bounded lookahead into the next day's folder and keeps only
the trades whose event time falls inside the day being built, which makes every
day's output complete and non-overlapping in event time.

The lookahead skips a next-day hour a live capture writer still holds open. That
refusal above is right for a CLOSED hour and wrong for one still being appended
to, and applying it there aborted the build of yesterday - the module's normal
operating configuration - over a file the operator never asked for.

Nothing discarded is left unaccounted for. A trade whose event time falls after
the day is deferred to that day's own build; one from before the day is
recoverable only while it sits inside the earlier day's lookahead window, and
beyond it no build will ever read it. Those last are counted apart as stranded
and written to `<store_root>/quarantine/`, because a trade lost with only a
counter to show for it is the failure this module opened by refusing.

    python -m store.cli --venue binance --date 2026-08-02 --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from capture.raw_writer import (
    IDX_SUFFIX, RAW_SUFFIX, _fsync_directory, is_hour_being_written, read_pair,
)
from store.parquet_partition import append_partition, compute_snapshot_id
from store.temporal_schema import EVENT_TIME
from store.trade_bars import Trade, build_bars, extract_trades

_TRADE_STREAMS = {"binance": "trade", "hyperliquid": "trades"}
DEFAULT_INTERVAL_NS = 60_000_000_000
DEFAULT_LOOKAHEAD_HOURS = 2

_NS_PER_SECOND = 1_000_000_000
_NS_PER_HOUR = 3_600 * _NS_PER_SECOND
_NS_PER_DAY = 86_400 * _NS_PER_SECOND


class NoHourFilesForSymbol(FileNotFoundError):
    """A requested symbol matched no captured hour files for this venue and day.

    Raised only when the day itself was captured (the venue/date folder exists).
    An entirely absent day is a legitimate zero - no capture has run yet, and
    `build_bars_for_day` reports that quietly by design. But inside a real
    capture, a symbol matching nothing is almost always a typo or the wrong
    stream name, and folding it into a normal-looking aggregate is exactly the
    silent loss this module refuses everywhere else: the caller asked for that
    symbol and must find out before the store quietly excludes it, not from a
    smaller total nobody thought to check.

    Judged on the day's OWN files, never on what the lookahead found. A symbol
    present only in the next day's folder is still absent from this day, and
    letting the lookahead satisfy the check would hide the typo it exists for.
    """


class BarOutsideBuildDay(ValueError):
    """A bar was about to be stored under a day it does not belong to.

    The whole point of the event-time filter is that no two builds ever emit the
    same (symbol, venue, event_time); a bar outside the day being built means
    the filter did not hold, and storing it anyway would reintroduce the
    partial-replacement corruption this module's docstring describes - silently,
    and only visible as a bar whose OHLCV is worse than the market's.

    Assumes `interval_ns` divides a UTC day, which every interval this store is
    built with does. A bar interval that straddles midnight would legitimately
    produce a bar open outside the day and would have to be handled before it
    could be used.
    """


class QuarantineExistsError(FileExistsError):
    """A quarantine file for this snapshot is already on disk.

    Refused for the same reason `append_partition` refuses a part that exists:
    the snapshot id is content-derived, so the same id means the same inputs, and
    overwriting the file would destroy the record of which trades an earlier run
    found unrecoverable. The quarantine file is new output, never a mutation -
    the only way it can already exist is a re-run of a build that produced it,
    which is precisely the accidental re-run the append-only rule exists to stop.
    """


def _index_path_for(raw_path: Path) -> Path:
    """The index sidecar beside a raw hour file, named by the writer's constants.

    Derived from RAW_SUFFIX and IDX_SUFFIX rather than spelled out: both suffixes
    belong to `capture.raw_writer`, and a literal copy here goes wrong silently
    the moment either changes - the glob in `_hour_files` matches nothing and the
    build reports a legitimate-looking zero, or the sidecar path names a file
    that does not exist. Silent zeros are the one failure this module refuses
    everywhere else.
    """
    name = raw_path.name
    stem = name[: -len(RAW_SUFFIX)] if name.endswith(RAW_SUFFIX) else name
    return raw_path.with_name(f"{stem}{IDX_SUFFIX}")


def _day_bounds_ns(date: str) -> tuple[int, int]:
    """The half-open [start, end) of one UTC calendar day, in integer nanoseconds.

    Half-open, so the instant of midnight belongs to exactly one day. An
    inclusive end would put a trade timestamped exactly at 00:00:00.000 into both
    days, and both builds would then emit a bar for it - the duplicate the event
    time filter exists to make impossible.
    """
    start = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    start_ns = int(start.timestamp()) * _NS_PER_SECOND
    return start_ns, start_ns + _NS_PER_DAY


def _next_date(date: str) -> str:
    moment = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    return (moment + dt.timedelta(days=1)).strftime("%Y-%m-%d")


def _hour_start_ns(raw_path: Path) -> int | None:
    """The instant an hour file's hour begins, from the writer's own naming.

    Returns None when the name carries no parseable hour key. Used only to decide
    whether a trade older than the day being built was still reachable by an
    earlier build, and an unparseable name means that question cannot be answered
    - which is treated as "no build will reach it" by the caller. Guessing the
    optimistic answer would silently downgrade an unrecoverable trade to a
    routine one, the exact conflation `trades_stranded` exists to end.
    """
    name = raw_path.name
    stem = name[: -len(RAW_SUFFIX)] if name.endswith(RAW_SUFFIX) else name
    hour_key = stem.rsplit("_", 1)[-1]
    try:
        moment = dt.datetime.strptime(hour_key, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return int(moment.timestamp()) * _NS_PER_SECOND


def _is_within_previous_days_lookahead(raw_path: Path, start_ns: int,
                                       lookahead_hours: int) -> bool:
    """Whether the build of an earlier day would have reached this hour file.

    The previous day's build reads its own folder plus `lookahead_hours` into
    this one, so a trade belonging to that day is recoverable exactly when its
    hour file sits inside that window - measured from the start of the day being
    built, which is where the previous day's lookahead begins. Assumes that build
    ran with the same `lookahead_hours` as this one; a build with a shorter
    window reached less far, and this would then call a stranded trade covered.

    Beyond the window no build ever reads the trade: the earlier day's does not
    look that far and this one discards it by event time. That is the difference
    between a trade deferred and a trade lost, and it is why the two are counted
    apart.
    """
    hour_start_ns = _hour_start_ns(raw_path)
    if hour_start_ns is None:
        return False
    return 0 <= hour_start_ns - start_ns < lookahead_hours * _NS_PER_HOUR


def _hour_files(capture_root: Path, venue: str, date: str,
                stream: str, symbol: str) -> list[Path]:
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{stream}_{symbol}_*{RAW_SUFFIX}"))


def _lookahead_hour_files(capture_root: Path, venue: str, date: str, stream: str,
                          symbol: str, lookahead_hours: int) -> tuple[list[Path], list[Path]]:
    """The next day's readable first hours, and the ones a writer still holds.

    Returns (readable, skipped_live). An absent next-day folder is normal, not an
    error: it is what building the most recent captured day looks like. It
    returns nothing and the count the caller reports says so, rather than the
    absence being invisible.

    An hour whose `.writing` marker is live is SKIPPED rather than read. This is
    the module's normal operating configuration - building yesterday's bars while
    capture keeps running - and in it the next day's current hour is open, with an
    unfinished zstd frame on disk. `read_pair` refuses that file, correctly for a
    CLOSED hour, and the refusal used to abort the build of a day the operator
    did ask for while naming a file they did not; the documented repair,
    `reconcile_pair`, then refuses the same file with `HourStillBeingWritten`, so
    the only way out was `--lookahead-hours 0`, which reinstates the partial-bar
    corruption the lookahead exists to prevent.

    Skipping costs no trades: they remain in the file, and day D+1's own build
    reads them from its own folder once the hour is closed. What it must not cost
    is visibility, so every skipped hour is returned and reported - an invisible
    skip is the silent loss this module refuses everywhere else.

    Only the LOOKAHEAD is tolerant. A live hour in the day's own folder means the
    operator is building today, and `read_pair`'s refusal is the correct and
    informative failure for that.
    """
    if lookahead_hours <= 0:
        return [], []
    following = _next_date(date)
    folder = Path(capture_root) / "raw" / venue / following
    if not folder.is_dir():
        return [], []
    found: list[Path] = []
    skipped_live: list[Path] = []
    for hour in range(min(lookahead_hours, 24)):
        candidate = folder / f"{stream}_{symbol}_{following}T{hour:02d}{RAW_SUFFIX}"
        if not candidate.exists():
            continue
        if is_hour_being_written(candidate)[0]:
            skipped_live.append(candidate)
            continue
        found.append(candidate)
    return found, skipped_live


def record_stranded_trades(store_root: Path, venue: str, date: str, snapshot_id: str,
                           stranded: Sequence[tuple[Trade, Path]]) -> Path:
    """Write every unrecoverable trade to a quarantine file and return its path.

    A counter is not enough. A stranded trade is one no build will ever pick up,
    and this module's rule - stated in its own docstring - is that a store quietly
    missing trades makes every statistic computed from it wrong in a way nothing
    reports. A number in a summary scrolls past; a file on disk names the trades,
    carries enough of each to reconstruct it, and says which hour file it came
    from, so the loss stays recoverable long after the run that found it.

    Created exclusively, never opened for writing over an existing name: see
    `QuarantineExistsError`. Contents are fsynced before the directory entry, the
    ordering `capture.raw_writer._write_lines` documents - a quarantine file whose
    name survives a power loss but whose bytes do not is worse than none, because
    it reads as a complete record of the damage.
    """
    folder = Path(store_root) / "quarantine"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"stranded-{venue}-{date}-{snapshot_id}.ndjson"
    try:
        handle = open(target, "x", encoding="utf-8")
    except FileExistsError as exc:
        raise QuarantineExistsError(
            f"{target} already exists; snapshot '{snapshot_id}' has already recorded its "
            f"stranded trades. Corrections are new snapshots, never rewrites") from exc
    try:
        with handle as sink:
            for trade, source in stranded:
                sink.write(json.dumps({
                    "symbol": trade.symbol,
                    "venue": trade.venue,
                    "price": trade.price,
                    "size": trade.size,
                    "event_time_ns": trade.event_time_ns,
                    "ingestion_time_ns": trade.ingestion_time_ns,
                    "source_file": str(source),
                }) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
    except BaseException:
        # A half-written quarantine file understates the damage while looking
        # like the complete record of it, and its name would then refuse the
        # re-run that would write it properly.
        target.unlink(missing_ok=True)
        raise
    _fsync_directory(folder)
    return target


def build_bars_for_day(capture_root: Path, store_root: Path, venue: str, date: str,
                       symbols: Sequence[str], interval_ns: int,
                       lookahead_hours: int = DEFAULT_LOOKAHEAD_HOURS) -> dict:
    """Read one venue-day of trades by event time and append the resulting bars.

    `lookahead_hours` bounds how far into the next day's folder the build reaches
    for trades whose event time still belongs to this day. Two hours is generous
    against a measured worst-case arrival lag of 141 s; the cost of reading too
    far is only a few extra files, while reading too little silently loses the
    tail of the day's last bar.

    A lookahead hour a live writer still holds open is skipped and reported
    (`lookahead_files_skipped_live`) rather than read; a live hour in the day's
    OWN folder is still a hard failure. See `_lookahead_hour_files`.

    Discarded trades are reported three ways - `trades_deferred_to_next_day`,
    `trades_covered_by_previous_day`, `trades_stranded` - because only the last
    of the three means a trade no build will ever read. Stranded trades are
    written to a quarantine file, never merely counted.
    """
    stream = _TRADE_STREAMS.get(venue)
    if stream is None:
        raise SystemExit(f"no trade stream known for venue '{venue}'")

    day_folder = Path(capture_root) / "raw" / venue / date
    symbol_files = {
        symbol: _hour_files(capture_root, venue, date, stream, symbol) for symbol in symbols
    }

    # Every requested symbol is checked before any frame is read, mirroring the
    # pre-check-then-write shape `append_partition` already uses: the failure
    # must arrive before work is done, not after some symbols were already read.
    # Gated on the day folder existing - see NoHourFilesForSymbol's docstring for
    # why an entirely absent day is not the same failure as a symbol missing from
    # a day that WAS captured.
    if day_folder.is_dir():
        missing = [symbol for symbol, files in symbol_files.items() if not files]
        if missing:
            raise NoHourFilesForSymbol(
                f"no {stream!r} hour files under {day_folder} for symbol(s) {missing} "
                f"(venue={venue!r}, date={date!r}); check for a typo or the wrong stream name")

    start_ns, end_ns = _day_bounds_ns(date)
    trades: list[Trade] = []
    # Only the day's OWN files, never the lookahead ones - see the snapshot id
    # computation below for why the distinction is load-bearing.
    day_sources: list[Path] = []
    frames = 0
    # Discards are split three ways because they are three different facts and one
    # counter made them indistinguishable. A trade whose event time falls after
    # this day is DEFERRED - day D+1 reads it from its own folder - and on this
    # archive that reached 47,401 in a single day, so it can never be an alarm. A
    # trade from before this day is recoverable only if the earlier day's own
    # lookahead reached its hour file; beyond that window nothing will ever read
    # it, and THAT is the alarm. Counting all three as one number meant the alarm
    # was always drowned by the routine case.
    trades_deferred_to_next_day = 0
    trades_covered_by_previous_day = 0
    stranded: list[tuple[Trade, Path]] = []
    lookahead_files = 0
    # `lookahead_files: 0` is how "the next day has not been captured yet" stays
    # visible instead of looking like a lookahead that ran, and the skipped list
    # is how "the next day's hour is still open" stays distinguishable from it.
    lookahead_files_skipped_live: list[Path] = []
    # Per symbol, not just a running total: a symbol that contributes zero frames
    # or zero trades must stay visible in the result rather than disappear into
    # an aggregate that looks identical to one where every symbol pulled its
    # weight.
    by_symbol: dict[str, dict[str, int]] = {}
    for symbol in symbols:
        symbol_frames = 0
        symbol_trades = 0
        ahead, skipped_live = _lookahead_hour_files(
            capture_root, venue, date, stream, symbol, lookahead_hours)
        lookahead_files += len(ahead)
        lookahead_files_skipped_live.extend(skipped_live)
        own_files = symbol_files[symbol]
        for raw_path in own_files + ahead:
            idx_path = _index_path_for(raw_path)
            for payload, entry in read_pair(raw_path, idx_path):
                frames += 1
                symbol_frames += 1
                for trade in extract_trades(payload, entry, venue, symbol):
                    # Event time decides the day, not the folder the frame landed
                    # in. A trade outside this day belongs to another day's build,
                    # which reads it from its own files - unless no build reaches
                    # it at all, which is what the three-way split establishes.
                    if start_ns <= trade.event_time_ns < end_ns:
                        trades.append(trade)
                        symbol_trades += 1
                    elif trade.event_time_ns >= end_ns:
                        trades_deferred_to_next_day += 1
                    elif _is_within_previous_days_lookahead(
                            raw_path, start_ns, lookahead_hours):
                        trades_covered_by_previous_day += 1
                    else:
                        stranded.append((trade, raw_path))
            if raw_path in own_files:
                day_sources.extend([raw_path, idx_path])
        by_symbol[symbol] = {"frames": symbol_frames, "trades": symbol_trades}

    # Digested over the day's OWN files alone. The lookahead decides WHICH trades
    # are selected, but the identity of the build is the day it builds: with the
    # lookahead files in the digest, the id moved every time the next day's folder
    # grew - which it does continuously while capture runs - so a rebuild of an
    # unchanged day computed a fresh id, sailed past `append_partition`'s
    # collision check and appended a duplicate part. Rows survived only because
    # the reader de-duplicates, while the guard against an accidental re-run was
    # gone and parts accumulated on every run. A rebuild of the same day must
    # collide; that refusal IS the append-only guarantee working.
    snapshot_id = compute_snapshot_id(day_sources)
    # Written before the bars, and in the empty-trades path too: a day that
    # produced no bars can still have stranded trades, and they are exactly the
    # ones nothing else will ever record.
    quarantine_file = (
        record_stranded_trades(store_root, venue, date, snapshot_id, stranded)
        if stranded else None)
    counts = {
        "trades_deferred_to_next_day": trades_deferred_to_next_day,
        "trades_covered_by_previous_day": trades_covered_by_previous_day,
        "trades_stranded": len(stranded),
        "lookahead_files": lookahead_files,
        "lookahead_files_skipped_live": [str(path) for path in lookahead_files_skipped_live],
        "quarantine_file": str(quarantine_file) if quarantine_file is not None else None,
    }

    if not trades:
        return {"frames": frames, "trades": 0, "bars": 0, "snapshot_id": None,
                "by_symbol": by_symbol, **counts}

    bars = build_bars(trades, interval_ns)
    stray = bars[(bars[EVENT_TIME] < start_ns) | (bars[EVENT_TIME] >= end_ns)]
    if not stray.empty:
        raise BarOutsideBuildDay(
            f"{len(stray)} bar(s) fall outside {date} (first event_time "
            f"{int(stray.iloc[0][EVENT_TIME])}, day is [{start_ns}, {end_ns})); "
            f"storing them would let two builds emit the same (symbol, venue, "
            f"event_time) and one replace the other")
    append_partition(store_root, f"bars_{interval_ns}ns", bars, snapshot_id)
    return {"frames": frames, "trades": len(trades), "bars": len(bars),
            "snapshot_id": snapshot_id, "by_symbol": by_symbol, **counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="store", description="Build the bitemporal store from captured frames.")
    parser.add_argument("--venue", required=True, choices=sorted(_TRADE_STREAMS))
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    parser.add_argument("--lookahead-hours", type=int, default=DEFAULT_LOOKAHEAD_HOURS,
                        help="how far into the next day's folder to look for trades "
                             "whose event time still belongs to this day")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    summary = build_bars_for_day(
        Path(args.capture_root), Path(args.store_root),
        args.venue, args.date, symbols, args.interval_ns, args.lookahead_hours)
    print(f"{summary['frames']} frames -> {summary['trades']} trades -> "
          f"{summary['bars']} bars (snapshot {summary['snapshot_id']})", file=sys.stderr)
    print(f"  {summary['lookahead_files']} next-day file(s) read ahead, "
          f"{summary['trades_deferred_to_next_day']} trade(s) deferred to the next day, "
          f"{summary['trades_covered_by_previous_day']} covered by the previous day's build",
          file=sys.stderr)
    for skipped in summary["lookahead_files_skipped_live"]:
        print(f"  skipped lookahead hour (a writer still holds it open): {skipped}",
              file=sys.stderr)
    for symbol, counts in summary["by_symbol"].items():
        print(f"  {symbol}: {counts['frames']} frames -> {counts['trades']} trades",
              file=sys.stderr)
    # Loud and last, on its own line naming the file. A stranded trade is one no
    # build will ever pick up, and a number folded into the summary above reads
    # exactly like the routine deferrals it sits beside - which is how this loss
    # stayed invisible in the first place.
    if summary["trades_stranded"]:
        print(f"STRANDED: {summary['trades_stranded']} trade(s) no build can recover; "
              f"quarantined at {summary['quarantine_file']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
