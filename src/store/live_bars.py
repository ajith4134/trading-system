"""Bars for the hour still being written, so an intraday system can see now.

Added 2026-08-16 after the user's §3a ruling — *"our entire trading crypto bot is
intraday on all segments spot, futures, options"* — made the store's freshness
the binding constraint on everything.

## The lag this removes, measured

Raw capture is live to the second: at clock 15:08:44 every venue was writing at
15:08:44. But `store.cli` builds only hours that have **closed**, in passes every
~35 minutes, so the newest bar the clock-gated reader served was **69 minutes
old**. For funding carry, settling 8-hourly, that is immaterial — which is why it
was never a problem and never a defect. For an intraday system it is
disqualifying.

## Why `store.cli` refuses the live hour, and why that refusal stays

Its reasoning is right and is not being overruled. From its own docstring:

> `RawWriter` rotates inside `append`, so a thin pair's finished hour stays
> claimed until it trades again, and its file can hold **ZERO BYTES** with every
> frame still inside the compressor. Zero bytes is an empty stream, not a torn
> one, so `read_pair` returns no frames and raises nothing.

Measured that day: 117 binance-spot hour files were claimed past their hour, and
one was empty six and a half hours late and **read as a market with no trades, in
a build that reported success.** That is the trap, and it is not about the data
being unreadable.

`store.cli` builds a **complete day** and must never emit an hour it only partly
read. This module builds something different — a deliberately partial, explicitly
provisional view of the current hour — so it can accept what that one must
refuse. The refusal there stays exactly as it is.

## What makes reading a live file safe

`capture.raw_writer` emits a **zstd frame boundary every 30 seconds of stream
time and fsyncs both files**. That cadence exists to bound crash loss, and it has
a second consequence: a live hour's file holds complete, decodable frames up to
the last boundary. Reading it is not reading a torn stream — it is reading a
shorter one.

## Only CLOSED minutes are emitted, and the newest one is dropped

The readable prefix ends somewhere unknown inside the last 30 seconds, so the
minute containing the newest readable trade **may be missing trades that already
happened**. Emitting it would publish a bar that is wrong until the minute
closes, and wrong in a direction nobody could see.

So the minute containing the newest readable trade is dropped, along with
anything after it. That is `store.cli`'s live-hour rule applied at minute
granularity, and it is the whole safety argument: **every bar this module emits
is a closed minute whose trades were all inside the readable prefix.**

Freshness is then `(now − newest readable trade) + up to 60s`, which measured
against a 30-second flush cadence is **roughly 30 to 90 seconds**.

## Every bar is PROVISIONAL, and the store already knows what to do with that

These bars are written with an availability time of *now*. When the hour closes,
`store.cli`'s ordinary build reads the whole hour and writes the same bars with a
**later** availability time. `store.clock_gated_reader` resolves that by keeping
the last visible version per (symbol, venue, event time) — so the complete bar
supersedes the provisional one automatically, and a backtest run afterwards sees
only the complete one.

That is the bitemporal store working as designed rather than a mechanism invented
here. What this module must not do is pretend otherwise, so the snapshot id says
`live` and names the last minute it emitted.

## An empty live file is NOT "no trades"

The trap above, closed explicitly. A file with no readable frames in an hour that
has not finished means **nothing has been flushed yet**, and it is reported as
`nothing_flushed` rather than folded into a bar count of zero. A silent zero here
is the failure this whole module is built around.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from capture.raw_writer import RAW_SUFFIX, IDX_SUFFIX, read_pair
# The one venue->stream map, imported rather than restated. `store.cli`'s own
# comment records what a second copy costs: binance-spot was absent for 1,363
# symbols, and coinbase was captured for a day with a working extractor the
# builder could not reach, because "the extractor existing is not the same as the
# builder reaching it". A test pins that map against the extractors; a private
# import here is cheaper than becoming the third instance of that bug.
from store.cli import _TRADE_STREAMS as TRADE_STREAMS
from store.parquet_partition import PartitionExistsError, append_partition
from store.trade_bars import BarAccumulator, extract_trades, is_tradeable

BAR_INTERVAL_NS = 60_000_000_000
_NS_PER_HOUR = 3_600_000_000_000

# The dataset the ordinary build writes to. The same one on purpose: a separate
# "live bars" dataset would make every consumer choose between two sources and
# get the choice wrong somewhere, and the correction machinery that supersedes
# these rows only works within one dataset.
DATASET = f"bars_{BAR_INTERVAL_NS}ns"

# The sentinel meaning "every symbol the archive holds for this venue-hour",
# spelled the same way `store.cli --symbols ALL` spells it. A symbol can never
# collide with it: venues name instruments BTCUSDT, BTC, BTC-USD, never ALL.
ALL_SYMBOLS = ["ALL"]


@dataclass
class LiveBuildReport:
    """What one live pass read and refused. Counts, never a bare success.

    `nothing_flushed` is separated from `no_trades` because they are different
    facts and conflating them is the documented failure this module exists
    around: an unflushed file read as a market with no trades, in a build that
    reported success.
    """
    venue: str
    hour: str
    symbols_read: int = 0
    symbols_nothing_flushed: int = 0
    symbols_no_trades: int = 0
    trades_read: int = 0
    bars_written: int = 0
    minutes_dropped_as_open: int = 0
    minutes_already_written: int = 0
    newest_trade_ns: int | None = None
    parts: list[Path] = field(default_factory=list)

    def describe(self) -> str:
        freshness = ""
        if self.newest_trade_ns is not None:
            age = (dt.datetime.now(dt.timezone.utc).timestamp() * 1e9
                   - self.newest_trade_ns) / 1e9
            freshness = f", newest readable trade {age:.0f}s old"
        return (f"{self.venue} hour {self.hour}: {self.bars_written} provisional "
                f"bar(s) from {self.trades_read} trade(s) across "
                f"{self.symbols_read} symbol(s){freshness}; "
                f"{self.minutes_dropped_as_open} open minute(s) dropped, "
                f"{self.minutes_already_written} already written, "
                f"{self.symbols_nothing_flushed} symbol(s) had nothing flushed "
                f"yet and {self.symbols_no_trades} genuinely had no trades")


def current_hour_utc(now: dt.datetime | None = None) -> tuple[str, str]:
    """The (date, hour) the writers are currently appending to."""
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%d"), now.strftime("%H")


def _raw_pair(capture_root: Path, venue: str, stream: str, symbol: str,
              date: str, hour: str) -> tuple[Path, Path]:
    folder = Path(capture_root) / "raw" / venue / date
    stem = f"{stream}_{symbol}_{date}T{hour}"
    return folder / f"{stem}{RAW_SUFFIX}", folder / f"{stem}{IDX_SUFFIX}"


def captured_symbols_this_hour(capture_root: Path, venue: str, stream: str,
                               date: str, hour: str) -> list[str]:
    """Every symbol with a capture file for this venue-hour, read off the archive.

    Read rather than listed, for the reason `store.cli` documents: capture
    subscribed 2,098 symbols on 2026-08-08 while the supervisor asked for 9, so
    99.6% of the tape was archived and never became a bar. The venues also name
    the same instrument differently - BTCUSDT, BTC, BTC-USD - so any hand-written
    list is really four lists, and all four were the core.
    """
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    prefix, suffix = f"{stream}_", f"_{date}T{hour}{RAW_SUFFIX}"
    return sorted(path.name[len(prefix):-len(suffix)]
                  for path in folder.glob(f"{prefix}*{suffix}"))


def drop_open_minute(frame: pd.DataFrame, newest_by_symbol: dict[str, int],
                     ) -> tuple[pd.DataFrame, int]:
    """Keep only bars whose minute closed before that SYMBOL's newest trade.

    The readable prefix ends somewhere inside the last flush interval, so the
    minute containing the newest readable trade may be missing trades that have
    already happened. Publishing it would emit a bar that is wrong until the
    minute closes, and wrong invisibly.

    Returns the surviving frame and how many bars were dropped, because a drop
    count of zero and a drop count of nine are different states and a caller that
    cannot tell them apart cannot see this rule working.
    """
    if frame.empty:
        return frame.reset_index(drop=True), 0
    # PER SYMBOL, because the readable prefix is a property of a FILE.
    #
    # The first version cut every symbol at the venue's newest trade. Each symbol
    # has its own capture file with its own last flush, so a symbol whose file was
    # flushed 90 seconds ago has a prefix ending 90 seconds back while a busy one
    # ends 5 seconds back. Cutting the quiet symbol at the busy symbol's boundary
    # publishes its final, still-open minute as though it had closed - the exact
    # thing this function exists to prevent, applied to the symbols most likely to
    # be mispriced by it.
    open_minute_start = frame["symbol"].map(
        lambda symbol: (newest_by_symbol[symbol] // BAR_INTERVAL_NS)
        * BAR_INTERVAL_NS)
    keep = frame["event_time_ns"] < open_minute_start
    return frame[keep].reset_index(drop=True), int((~keep).sum())


def build_live_hour(capture_root: Path, store_root: Path, venue: str,
                    stream: str, symbols: list[str],
                    now: dt.datetime | None = None,
                    written_through: dict[str, int] | None = None,
                    ) -> LiveBuildReport:
    """Build provisional bars for the hour still being written.

    Every bar written is a CLOSED minute whose trades were entirely inside the
    readable prefix. Nothing here is complete by construction, and the snapshot
    id says so.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    date, hour = current_hour_utc(now)
    report = LiveBuildReport(venue=venue, hour=f"{date}T{hour}")

    accumulator = BarAccumulator(BAR_INTERVAL_NS)
    # Keyed on the symbol the TRADE carries, not the one requested: binance
    # echoes `s` and coinbase `product_id`, and `BarAccumulator` groups on what
    # the trade says. A dict keyed on the request would miss every lookup.
    newest_by_symbol: dict[str, int] = {}
    for symbol in symbols:
        raw_path, idx_path = _raw_pair(capture_root, venue, stream, symbol,
                                       date, hour)
        if not raw_path.exists():
            continue
        report.symbols_read += 1
        try:
            pairs = read_pair(raw_path, idx_path)
        except Exception:                          # noqa: BLE001 - counted, not fatal
            # A torn tail on a file being appended to is the ordinary case here,
            # not corruption. One unreadable symbol must not cost the pass.
            report.symbols_nothing_flushed += 1
            continue
        if not pairs:
            # NOT "no trades". The hour has not finished, so an empty readable
            # prefix means nothing has been flushed yet - the exact conflation
            # that once read a market as having no trades inside a build that
            # reported success.
            report.symbols_nothing_flushed += 1
            continue

        trades = []
        for payload, entry in pairs:
            for trade in extract_trades(payload, entry, venue, symbol):
                if is_tradeable(trade):
                    trades.append(trade)
        if not trades:
            report.symbols_no_trades += 1
            continue
        report.trades_read += len(trades)
        for trade in trades:
            previous = newest_by_symbol.get(trade.symbol)
            if previous is None or trade.event_time_ns > previous:
                newest_by_symbol[trade.symbol] = trade.event_time_ns
        accumulator.extend(trades)

    frame = accumulator.to_frame()
    if frame.empty or not newest_by_symbol:
        return report

    report.newest_trade_ns = max(newest_by_symbol.values())
    frame, dropped = drop_open_minute(frame, newest_by_symbol)
    report.minutes_dropped_as_open = dropped

    # Only minutes this loop has not already written. Without it a pass every 60
    # seconds rewrites the whole hour every time - by minute 59 that is 59 copies
    # of minute 0, all identical, in a store with no delete path. The reader
    # dedupes them, so it is storage rather than correctness, but it is storage
    # that grows quadratically through the hour.
    #
    # Keyed PER SYMBOL for the same reason the open-minute cut is: one venue-wide
    # mark set from the busiest symbol would mark minutes as written for a quiet
    # symbol that had not produced them yet, and those minutes would then be
    # dropped forever once its file flushed - a permanent hole in the slow
    # symbols, which is the half of the universe least likely to be checked.
    if written_through is not None:
        before = len(frame)
        keep = frame.apply(
            lambda row: row["event_time_ns"] > written_through.get(
                (venue, row["symbol"]), -1), axis=1)
        frame = frame[keep].reset_index(drop=True)
        report.minutes_already_written = before - len(frame)
    if frame.empty:
        return report

    # Availability and ingestion are left EXACTLY as `BarAccumulator` stamped
    # them, and this is the most important line in the module.
    #
    # The first version overwrote availability with `now`. That is wrong twice
    # over and was caught before it wrote to the real store. `build_bars` stamps
    # `max(bar_close, ingestion_time)` - derived from the DATA - so the complete
    # build after the hour closes produces a stamp EARLIER than a wall-clock one,
    # and `drop_duplicates(keep="last")` would then keep the provisional bar
    # forever: an incomplete row permanently shadowing the complete one, in a
    # store with no delete path. It also breaks the clock gate's meaning, since
    # `availability_time` answers "when did this become knowable" and a wall
    # clock answers "when did this pass happen" - a backtest at a past clock
    # would miss bars it should have seen.
    #
    # Leaving the stamp alone makes the two cases behave correctly by themselves:
    #
    #   * no late trade  -> this bar IS the complete bar, byte for byte, same
    #     stamp, and the duplicate is harmless.
    #   * a late trade   -> the complete build's bar carries that trade's later
    #     ingestion time, so its availability is later and it supersedes this one
    #     through the ordinary correction resolution.
    #
    # The correction fires in exactly the case where the two bars differ, which
    # is what the bitemporal store was built to do.

    # Written ONE SYMBOL AT A TIME, with a snapshot id naming that symbol's own
    # coverage. `append_partition` writes a part per symbol under a single id, so
    # an id built from the venue-wide last minute names different content for
    # different symbols - and two passes collide the moment a quiet symbol's
    # newly-readable minutes end where a busy symbol's ended a pass earlier. That
    # is not a storage nuisance: `append_partition` raises, and an uncaught raise
    # kills the loop the whole intraday path depends on. Found by a test, before
    # it ran against the real store.
    #
    # The trade count is in the id because it is the content discriminator. After
    # a restart the mark is empty and the same minutes are rebuilt; if the tape
    # has not moved the id repeats and the write is correctly refused as a
    # duplicate, and if a late flush added trades to that last minute the id
    # differs, the bar is written, and its later ingestion time supersedes the
    # earlier one through the ordinary correction path.
    #
    # `live` is in the name so nothing downstream can read these as a finished hour.
    for symbol, group in frame.groupby("symbol", sort=True):
        last_minute = int(group["event_time_ns"].max())
        snapshot = (f"live-{venue}-{date}T{hour}-{symbol}"
                    f"-through-{last_minute}-t{int(group['trades'].sum())}")
        try:
            report.parts.extend(append_partition(Path(store_root), DATASET,
                                                 group.reset_index(drop=True),
                                                 snapshot))
        except PartitionExistsError:
            # Identical id means identical coverage AND identical trade count,
            # so the rows already in the store are these rows. Counted rather
            # than raised: this is the ordinary result of a restart re-reading an
            # hour, and one symbol must not cost the pass.
            report.minutes_already_written += len(group)
            continue
        report.bars_written += len(group)
        if written_through is not None:
            written_through[(venue, symbol)] = last_minute
    return report


def build_live_hour_for_venues(capture_root: Path, store_root: Path,
                               venues: dict[str, list[str]],
                               now: dt.datetime | None = None,
                               written_through: dict | None = None,
                               ) -> list[LiveBuildReport]:
    """One pass over several venues. A venue with no stream mapping is skipped.

    Skipped rather than guessed: `store.cli`'s map is the single record of which
    stream carries a venue's trades, and inventing one here is how a venue gets
    captured and never built.
    """
    date, hour = current_hour_utc(now)
    reports = []
    for venue, symbols in sorted(venues.items()):
        stream = TRADE_STREAMS.get(venue)
        if stream is None:
            continue
        if symbols == ALL_SYMBOLS:
            symbols = captured_symbols_this_hour(capture_root, venue, stream,
                                                 date, hour)
        reports.append(build_live_hour(capture_root, store_root, venue, stream,
                                       symbols, now=now,
                                       written_through=written_through))
    return reports


def main(argv: list[str] | None = None) -> int:
    """Build the live hour on a loop, so an intraday consumer sees ~30-90s data."""
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(
        prog="store.live_bars",
        description="Provisional bars for the hour still being written. Every "
                    "bar is a CLOSED minute inside the readable prefix, and the "
                    "complete build supersedes them when the hour ends.")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root",
                        default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--venue", action="append", required=True,
                        help="repeatable, as venue=SYM1,SYM2 or venue=ALL to "
                             "read the symbols off the archive")
    parser.add_argument("--interval-seconds", type=int, default=0,
                        help="0 builds once and exits; anything else loops")
    args = parser.parse_args(argv)

    venues: dict[str, list[str]] = {}
    for spec in args.venue:
        venue, _, symbols = spec.partition("=")
        venues[venue] = [s for s in symbols.split(",") if s]

    # Created ONCE, outside the loop, and threaded through every pass. Held here
    # rather than persisted because a restart re-reads the same hour and writes
    # the same rows with the same data-derived stamps - a duplicate the reader
    # already resolves - whereas a stale file would suppress minutes that were
    # never written.
    written_through: dict = {}

    while True:
        for report in build_live_hour_for_venues(
                Path(args.capture_root), Path(args.store_root), venues,
                written_through=written_through):
            print(report.describe(), flush=True)
        if args.interval_seconds <= 0:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
