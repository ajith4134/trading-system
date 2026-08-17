# src/store/cli.py
"""Builds the bitemporal store from the raw archive.

Reads through `capture.raw_writer.iter_pair`, which refuses a torn file rather
than returning its readable prefix. That refusal is load-bearing here: silently
building from a truncated hour produces a store that is quietly missing trades,
and every statistic computed from it is wrong in a way nothing reports.

`iter_pair` streams, and the refusal therefore arrives after the frames ahead of
the damage have already been counted into the accumulator. That is safe here for
one structural reason, and it has to stay true: this module appends to the store
AFTER the read loop, never inside it, so a refusal abandons the whole build
rather than committing its prefix.

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

Any hour a live capture writer still holds open is skipped, in the next day's
folder and in the day's own. That refusal above is right for a CLOSED hour and
wrong for one still being appended to, and applying it to the lookahead aborted
the build of yesterday - the module's normal operating configuration - over a
file the operator never asked for. A skipped hour is named in the snapshot id,
because a build that skipped one is missing trades and is therefore not the same
build as one that read everything. Naming it there is what lets the rebuild, once
the hour closes, append the complete bar instead of being refused as a duplicate
of the incomplete one.

The day's own hours were exempt from that check until 2026-08-08, on the belief
that the refusal above would catch them. It does not, and the belief was
load-bearing: `RawWriter` rotates inside `append`, so a thin pair's finished hour
stays claimed until it trades again, and its file can hold ZERO bytes with every
frame still inside the compressor. Zero bytes is an empty stream, not a torn one,
so `read_pair` returns no frames and raises nothing. Measured that day at 17:28,
117 binance-spot hour files were claimed past their hour;
`trade_ARBIDR_2026-08-08T11.ndjson.zst` was empty six and a half hours late and
read as a market with no trades, in a build that reported success.

Nothing discarded is left unaccounted for. A trade whose event time falls after
the day is deferred to that day's own build; one from before the day is
recoverable only when it belongs to the previous day AND sits inside that day's
lookahead window, and outside either bound no build will ever read it. Those last are counted apart as stranded
and written to `<store_root>/quarantine/`, because a trade lost with only a
counter to show for it is the failure this module opened by refusing.

    python -m store.cli --venue binance --date 2026-08-02 --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from capture.raw_writer import (
    IDX_SUFFIX, RAW_SUFFIX, _fsync_directory, is_hour_being_written, iter_pair,
)
from store.parquet_partition import (
    PartitionExistsError, append_partition, compute_snapshot_id,
)
from store.temporal_schema import EVENT_TIME
from store.trade_bars import BarAccumulator, Trade, extract_trades

# The capture stream carrying trades, per venue. A venue absent here cannot be
# built at all, and until 2026-08-08 spot was absent - 1,363 captured symbols the
# builder could not read, while the paper-engine design called tier 1 "2,123
# symbols". Nothing reported the gap, because a venue that is never asked for
# never refuses.
# Every venue whose trade tape can become bars, and the stream its trades arrive
# on. A venue missing here is captured and unbuildable, and nothing says so - the
# archive fills, the builder never looks, and the raw is evicted after seven
# days. That has now happened twice. binance-spot cost 1,363 symbols before it
# was noticed; coinbase was added 2026-08-10 with a working, tested extractor
# already registered in `trade_bars._EXTRACTORS` and no entry here, so
# `bars_supervisor.sh` asked for it every hour and got
# `invalid choice: 'coinbase'` while 5,208 `matches` files accumulated that day.
#
# The lesson both times is that the extractor existing is not the same as the
# builder reaching it, so `tests/test_store_cli.py` now asserts these two maps
# agree rather than leaving them to be compared by eye.
_TRADE_STREAMS = {"binance": "trade", "binance-spot": "trade",
                  "hyperliquid": "trades", "coinbase": "matches"}
DEFAULT_INTERVAL_NS = 60_000_000_000
DEFAULT_LOOKAHEAD_HOURS = 2

# `--symbols ALL`. A hand-maintained list is what kept the broad universe
# unbuilt: capture subscribed 2,098 symbols and the supervisor asked for three,
# so 99.6% of the tape was archived and never turned into a bar. The list also
# cannot be maintained - a symbol listing mid-day is captured immediately and
# would wait for a human to add it, and the raw it was captured from is evicted
# after seven days. Deliberately the same word capture already uses for the same
# idea (`--tail-symbols ALL`), so the two ends of the pipeline read alike.
SYMBOLS_ALL = "ALL"

# "There was nothing to do", distinct from both success and failure, so a caller
# can tell them apart without parsing text.
EXIT_ALREADY_BUILT = 4

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


class NoTradeTapeInCapture(FileNotFoundError):
    """A captured venue-day holds frames but not one file of the trade stream.

    The same reasoning as `NoHourFilesForSymbol`, one level up. An absent
    venue/date folder is a legitimate zero and `--symbols ALL` reports it as
    such. A folder that EXISTS means capture ran that day, and a run that
    recorded depth or funding while recording no trades at all is a broken
    subscription, not an idle market - `trade` is the one stream every venue in
    `_TRADE_STREAMS` subscribes for every symbol it touches.

    Enumerating to an empty list there would exit 0 with "nothing to build",
    which is indistinguishable from the day nobody captured. That is the shape
    this whole module refuses: the silence of a feed that never connected
    reading as the silence of a market that never traded.
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


def _is_within_previous_days_lookahead(raw_path: Path, event_time_ns: int,
                                       start_ns: int, lookahead_hours: int) -> bool:
    """Whether the build of the PREVIOUS day would have reached this trade.

    Two conditions, and both are necessary. The previous day's build reads its own
    folder plus `lookahead_hours` into this one, so the trade's hour file must sit
    inside that window - measured from the start of the day being built, which is
    where the previous day's lookahead begins. Assumes that build ran with the same
    `lookahead_hours` as this one; a build with a shorter window reached less far,
    and this would then call a stranded trade covered.

    And the trade's event time must fall inside the previous day, because that
    build keeps only its OWN day's trades however far its lookahead read. Testing
    the file's position alone made the alarm depend on where a trade happened to
    land: a three-day-stale trade - or one carrying `event_time_ns=0`, which
    `_extract_binance` produces from a garbage venue timestamp without complaint -
    read as covered in hour 00 and stranded in hour 05, identical trades, opposite
    verdicts. Anything older than the previous day is reachable by no build at all.

    Beyond either bound no build ever reads the trade: the earlier day's does not
    look that far or would discard it anyway, and this one discards it by event
    time. That is the difference between a trade deferred and a trade lost, and it
    is why the two are counted apart.
    """
    hour_start_ns = _hour_start_ns(raw_path)
    if hour_start_ns is None:
        return False
    if not 0 <= hour_start_ns - start_ns < lookahead_hours * _NS_PER_HOUR:
        return False
    return start_ns - _NS_PER_DAY <= event_time_ns


def _hour_files(capture_root: Path, venue: str, date: str,
                stream: str, symbol: str) -> list[Path]:
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{stream}_{symbol}_*{RAW_SUFFIX}"))


def _end_of_day_ns(date: str) -> int:
    """The last instant of a UTC day, for classifying that day's universe.

    End rather than start: a pair listed at 14:00 traded that day and its bars
    will be built, so classifying against 00:00 would leave it unclassified and
    silently excluded on its first day. Point-in-time is still preserved - the
    partition never sees a snapshot recorded after the day being built.
    """
    return (int(dt.datetime.fromisoformat(date)
                .replace(tzinfo=dt.timezone.utc).timestamp()) + 86_400) * 1_000_000_000 - 1


def _dollar_quoted_only(symbols: list[str], capture_root: Path, venue: str, date: str,
                        include_non_dollar: bool = False) -> list[str]:
    """Drop what is not priced in dollars, and say what was dropped.

    Ledger row DM-066 settled this in 2026-08-08 - *filter to dollar quotes, do
    not convert*, because conversion needs an FX rate the archive does not
    capture and a wrong rate corrupts a P&L silently. The library was written
    that day and **nothing called it**, so `--symbols ALL` kept building every
    captured pair. Measured on the live store 2026-08-09 before this: 536
    non-dollar symbols held 89,097 bars, 29.4% of binance-spot's, priced in TRY,
    EUR, JPY, IDR, BRL, BTC and ETH.

    That is not a storage problem, it is a correctness one, and it bites at the
    layer above: a cross-sectional strategy ranking those bars compares a lira
    price against a USDT price as if both were dollars.

    Only what is KNOWN not to be a dollar is dropped. Two other groups survive
    and are named instead:

    - **unknown** - in the quote map, quote asset not recognised. A new
      stablecoin lands here, and excluding it would shrink the tradeable universe
      on the day it listed with no symptom. See `QuotePartition`.
    - **unlisted** - captured, and absent from the snapshot entirely. A pair
      delisted since, or one captured before the universe was recorded.

    Both are the same epistemic state as each other and the opposite of a
    finding: not known to be a dollar is not known not to be. Dropping either
    would be assuming a denomination, which is the thing
    `QuoteAssetsNotRecorded` refuses to do three lines down. Building them costs
    storage; dropping them costs a market.
    """
    from store.quote_currency import QuoteAssetsNotRecorded, dollar_quoted_symbols

    if include_non_dollar:
        print(f"--include-non-dollar: building all {len(symbols)} captured symbol(s) "
              f"regardless of denomination", file=sys.stderr)
        return symbols
    try:
        partition = dollar_quoted_symbols(capture_root, venue, _end_of_day_ns(date))
    except QuoteAssetsNotRecorded:
        # A day older than the first snapshot that carried quote assets. Fall
        # forward to the earliest map there is, loudly.
        #
        # Justified by `UniverseTracker.record_snapshot`'s own contract: *"a
        # pair's quote currency is fixed, but which pairs exist is not"*. A map
        # read later therefore classifies an earlier day correctly for every pair
        # in both, and a pair that delisted in between is absent from the map -
        # which lands it in "unlisted" and builds it anyway, under the rule above.
        # So the fallback cannot mis-denominate anything; it can only fail to
        # classify, which is already handled.
        #
        # Found 2026-08-09 and it was costing days. Refusing outright killed the
        # bars pipeline for every day older than the first snapshot: 2026-08-08
        # failed on every supervisor pass, silently as far as the wall was
        # concerned, and raw is evicted after seven days. A guard that turns a
        # missing snapshot into permanent data loss is worse than the
        # mixed-currency store it was protecting against.
        import time as _time
        try:
            partition = dollar_quoted_symbols(capture_root, venue, _time.time_ns())
        except QuoteAssetsNotRecorded as refusal:
            # No snapshot at ALL. Now there is genuinely nothing to classify
            # against, and building everything would be the mixed-currency store.
            raise SystemExit(f"refusing to build {venue} {date}: {refusal}")
        print(f"no universe snapshot at or before {date}; classifying it against "
              f"the current map instead - a pair's quote currency is fixed, so a "
              f"later map is correct for any pair that existed on the day",
              file=sys.stderr)

    # Archive filenames are path-SAFE names; the venue's quote map keys are the
    # venue's own. They differ for any symbol carrying a character a filename
    # cannot: Binance lists 币安人生USDT, 龙虾USDT and 我踏马来了USDT, and
    # `_safe_path_token` base32-encodes them to `_b32_...` so nothing steers a
    # write out of the archive.
    #
    # Comparing the encoded form against the venue's map matched nothing, so
    # every one of them fell through to "unlisted" and was built. Measured
    # 2026-08-09: `币安人生U` is quoted in **U**, not dollars, and it was going
    # into the store anyway - the exact thing this filter exists to prevent,
    # defeated by a name the filter could not read.
    from capture.venue_recorder import decode_path_token

    venue_name = {symbol: decode_path_token(symbol) for symbol in symbols}
    known_non_dollar = set(partition.non_dollar)
    kept = [s for s in symbols if venue_name[s] not in known_non_dollar]
    dropped = sorted(s for s in symbols if venue_name[s] in known_non_dollar)
    if dropped:
        by_quote: dict[str, int] = {}
        for symbol in dropped:
            quote = partition.non_dollar[venue_name[symbol]]
            by_quote[quote] = by_quote.get(quote, 0) + 1
        top = ", ".join(f"{q} {n}" for q, n in
                        sorted(by_quote.items(), key=lambda kv: -kv[1])[:8])
        print(f"excluding {len(dropped)} non-dollar-quoted symbol(s) ({top}); "
              f"building {len(kept)}", file=sys.stderr)

    # Both reported, never folded into the exclusion count, and never silent -
    # a symbol built without a known denomination is a fact the operator owns.
    unknown = sorted(s for s in symbols if venue_name[s] in partition.unknown)
    unlisted = sorted(s for s in kept
                      if venue_name[s] not in partition.dollar and s not in unknown)
    if unknown:
        print(f"UNCLASSIFIED quote asset, built anyway: {unknown[:10]}"
              f"{' ...' if len(unknown) > 10 else ''}", file=sys.stderr)
    if unlisted:
        print(f"NOT IN THE UNIVERSE SNAPSHOT, built anyway: {unlisted[:10]}"
              f"{' ...' if len(unlisted) > 10 else ''}", file=sys.stderr)
    return kept


def captured_symbols(capture_root: Path, venue: str, date: str) -> list[str]:
    """Every symbol whose trade tape this venue-day actually holds, sorted.

    Read off the archive, not off the universe snapshot, and the difference is
    not cosmetic. The snapshot records what the venue LISTED; only the archive
    knows what was CAPTURED. A symbol listed at 09:00 whose subscription never
    connected appears in the snapshot and on no disk, and asking `build_bars_for_day`
    for it raises `NoHourFilesForSymbol` - which under batching would fail that
    whole batch of five over a symbol no build could ever have produced. The
    archive can only be the source of a list of what to read from the archive.

    Returns [] for a venue-day nobody captured; raises `NoTradeTapeInCapture`
    when the day was captured and carries no trade stream at all. Those two are
    the same empty list and opposite facts.

    The symbol is cut out with the date-and-hour tail anchored, not by splitting
    on '_': hyperliquid names instruments `0G` and `2Z` today and Binance keeps
    adding, so a symbol carrying an underscore is a listing away rather than
    impossible. The `.idx.zst` sibling of every hour file is excluded by the
    suffix and would be deduplicated anyway.
    """
    stream = _TRADE_STREAMS[venue]
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []

    hour_file = re.compile(
        rf"^{re.escape(stream)}_(?P<symbol>.+)_{re.escape(date)}T\d{{2}}"
        rf"{re.escape(RAW_SUFFIX)}$")
    symbols = {
        match.group("symbol")
        for match in (hour_file.match(path.name)
                      for path in folder.glob(f"{stream}_*{RAW_SUFFIX}"))
        if match is not None
    }
    if not symbols:
        raise NoTradeTapeInCapture(
            f"{folder} exists, so capture ran on {date}, but holds no "
            f"'{stream}_*{RAW_SUFFIX}' file for {venue}. Every symbol this venue "
            f"touches subscribes that stream, so this is a subscription that "
            f"never connected, not a day without trades - and reporting it as "
            f"nothing to build would make it look like the days no capture covered.")
    return sorted(symbols)


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
    unfinished zstd frame on disk. The reader refuses that file, correctly for a
    CLOSED hour, and the refusal used to abort the build of a day the operator
    did ask for while naming a file they did not; the documented repair,
    `reconcile_pair`, then refuses the same file with `HourStillBeingWritten`, so
    the only way out was `--lookahead-hours 0`, which reinstates the partial-bar
    corruption the lookahead exists to prevent.

    Skipping is NOT free, and it is returned rather than swallowed because of what
    it costs. The trades it passes over include exactly the ones the lookahead
    exists to rescue: a day-D trade filed under day D+1 because it arrived after
    midnight. Day D+1's own build will read that hour once it closes and then
    discard the trade by event time, so nothing else ever picks it up, and day D's
    last bar is left built from its own folder alone - complete-looking and short
    the late trades. That is why the caller folds the skipped hours into the
    snapshot id: it makes the incomplete build a different build, so the rebuild
    once the hour closes is allowed to append the complete bar, which carries a
    later availability time and wins the reader's correction resolution.

    Only the LOOKAHEAD is tolerant. A live hour in the day's own folder means the
    operator is building today, and the reader's refusal is the correct and
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

    A live hour is skipped and reported rather than read, in the lookahead
    (`lookahead_files_skipped_live`) and in the day's OWN folder
    (`day_files_skipped_live`) alike. Every skipped hour is named in `snapshot_id`,
    so rebuilding the day once those hours close is a new snapshot rather than a
    refused duplicate.

    This docstring used to say a live hour in the day's own folder was a hard
    failure, on the reasoning that `read_pair` refuses a torn zstd frame. Measured
    2026-08-08, it does not: rotation happens inside `RawWriter.append`, so a thin
    pair's finished hour stays claimed until it trades again, and its file can sit
    at ZERO bytes with every frame still inside the compressor. Zero bytes is an
    empty stream, not a torn one - `read_pair` returned 0 frames for
    `trade_ARBIDR_2026-08-08T11.ndjson.zst` and raised nothing, six and a half
    hours after hour 11 ended, while 116 other spot files were in the same state.
    A captured hour read as a market with no trades and the build reported
    complete. The claim is now checked directly instead of being inferred from a
    decompressor error.

    Skipped rather than raised, deliberately, even though the old docstring
    promised a failure: one claimed hour would otherwise cost the other 2,000
    symbols in the batch their bars, and the writer that holds it may not release
    it for hours. The trades are not lost - they are in the file, the file is named
    in this summary and in the snapshot id, and the next pass builds them.

    Discarded trades are reported three ways - `trades_deferred_to_next_day`,
    `trades_covered_by_previous_day`, `trades_stranded` - because only the last
    of the three means a trade no build will ever read. Stranded trades are
    written to a quarantine file, never merely counted - and written only once the
    rest of the build has succeeded, so a build that raises leaves no file behind
    to refuse the re-run that would fix it.
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

    # AFTER the missing-symbol check, never before. A symbol whose only hour is
    # still claimed HAS a file; folding it in with the typos would turn a writer
    # that has not flushed yet into "check for a typo".
    day_files_skipped_live: list[Path] = []
    for symbol, files in symbol_files.items():
        readable = []
        for raw_path in files:
            if is_hour_being_written(raw_path)[0]:
                day_files_skipped_live.append(raw_path)
            else:
                readable.append(raw_path)
        symbol_files[symbol] = readable

    start_ns, end_ns = _day_bounds_ns(date)
    # The bars themselves, folded as trades are read, rather than a list of
    # every trade in the batch. See `BarAccumulator` for the measurements -
    # including the one that says this is an improvement rather than a cure.
    bar_accumulator = BarAccumulator(interval_ns)
    trades_kept = 0
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

    # The lookahead sets are resolved for every symbol BEFORE any file is opened,
    # because the snapshot id depends on them and the id is what decides whether
    # there is any work to do at all. Resolving them costs directory listings and
    # stat calls; discovering the same answer after the read costs the read.
    lookahead_by_symbol: dict[str, list[Path]] = {}
    for symbol in symbols:
        ahead, skipped_live = _lookahead_hour_files(
            capture_root, venue, date, stream, symbol, lookahead_hours)
        lookahead_by_symbol[symbol] = ahead
        lookahead_files += len(ahead)
        lookahead_files_skipped_live.extend(skipped_live)
        for raw_path in symbol_files[symbol]:
            day_sources.extend([raw_path, _index_path_for(raw_path)])

    # Digested over the day's OWN files alone. The lookahead decides WHICH trades
    # are selected, but the identity of the build is the day it builds: with the
    # lookahead files in the digest, the id moved every time the next day's folder
    # grew - which it does continuously while capture runs - so a rebuild of an
    # unchanged day computed a fresh id, sailed past `append_partition`'s
    # collision check and appended a duplicate part. Rows survived only because
    # the reader de-duplicates, while the guard against an accidental re-run was
    # gone and parts accumulated on every run. A rebuild of the same day must
    # collide; that refusal IS the append-only guarantee working.
    #
    # The hours this build SKIPPED are part of its identity, by name and never by
    # content - see `compute_snapshot_id`. A build that skipped a live lookahead
    # hour is a different, incomplete build: its last bar is missing the late
    # trades in that hour, and no other build will ever supply them. Without the
    # skipped names in the id, the rebuild once the hour closes computes the same
    # id, `append_partition` refuses it as a re-run, and the partial bar is
    # permanent in a store with no delete path. With them, a re-run that skipped
    # the SAME hours still collides - the accidental-re-run guard is untouched -
    # while the rebuild that skips nothing appends a complete bar whose later
    # availability time makes the reader serve it over the partial one.
    #
    # Computed BEFORE the parse loop, not after. The id is a digest of the input
    # files, so it is knowable without decompressing any of them - and a re-run of
    # a day already built used to pay the whole read before `append_partition`
    # said there was nothing to do. Measured 2026-08-08: a 50-symbol rebuild cost
    # 27.7s against 27.6s for the original, which on an hourly supervisor over 569
    # symbols is ~34 minutes of parsing per hour to rediscover "already built".
    # The refusal is unchanged; only its cost moves.
    # Both kinds of skipped hour, by name and never by content - see
    # `compute_snapshot_id`. A day's own claimed hour is exactly the case its
    # docstring describes: bytes still changing, and the name is what says which
    # input this result is missing, so the rebuild once the writer releases it is
    # a new snapshot rather than a duplicate refused into permanence.
    snapshot_id = compute_snapshot_id(
        day_sources,
        [path.name for path in lookahead_files_skipped_live + day_files_skipped_live])
    _refuse_if_already_built(store_root, interval_ns, symbols, snapshot_id)

    for symbol in symbols:
        symbol_frames = 0
        symbol_trades = 0
        own_files = symbol_files[symbol]
        for raw_path in own_files + lookahead_by_symbol[symbol]:
            idx_path = _index_path_for(raw_path)
            for payload, entry in iter_pair(raw_path, idx_path):
                frames += 1
                symbol_frames += 1
                for trade in extract_trades(payload, entry, venue, symbol):
                    # Event time decides the day, not the folder the frame landed
                    # in. A trade outside this day belongs to another day's build,
                    # which reads it from its own files - unless no build reaches
                    # it at all, which is what the three-way split establishes.
                    if start_ns <= trade.event_time_ns < end_ns:
                        bar_accumulator.add(trade)
                        trades_kept += 1
                        symbol_trades += 1
                    elif trade.event_time_ns >= end_ns:
                        trades_deferred_to_next_day += 1
                    elif _is_within_previous_days_lookahead(
                            raw_path, trade.event_time_ns, start_ns, lookahead_hours):
                        trades_covered_by_previous_day += 1
                    else:
                        stranded.append((trade, raw_path))
        by_symbol[symbol] = {"frames": symbol_frames, "trades": symbol_trades}

    counts = {
        "trades_deferred_to_next_day": trades_deferred_to_next_day,
        "trades_covered_by_previous_day": trades_covered_by_previous_day,
        "trades_stranded": len(stranded),
        "lookahead_files": lookahead_files,
        "lookahead_files_skipped_live": [str(path) for path in lookahead_files_skipped_live],
        "day_files_skipped_live": [str(path) for path in day_files_skipped_live],
    }

    bars = bar_accumulator.to_frame() if trades_kept else None
    if bars is not None:
        stray = bars[(bars[EVENT_TIME] < start_ns) | (bars[EVENT_TIME] >= end_ns)]
        if not stray.empty:
            raise BarOutsideBuildDay(
                f"{len(stray)} bar(s) fall outside {date} (first event_time "
                f"{int(stray.iloc[0][EVENT_TIME])}, day is [{start_ns}, {end_ns})); "
                f"storing them would let two builds emit the same (symbol, venue, "
                f"event_time) and one replace the other")
        append_partition(store_root, f"bars_{interval_ns}ns", bars, snapshot_id)

    # LAST, once nothing else in this build can still fail. The quarantine file is
    # named by the snapshot id, and that id derives from the raw files alone - it
    # does not move when `interval_ns` does. So a build that wrote quarantine and
    # then raised left behind a name that refused every later attempt at the same
    # day with `QuarantineExistsError`, including the operator's corrected re-run,
    # and this store has no delete path: the day could only be built again by
    # hand-removing a file the error text never mentions. A run that stored no bars
    # has no trades to be missing, so it has nothing to quarantine.
    #
    # The no-bars path still records: a day whose every trade belongs elsewhere
    # produces no bars and can still strand trades, and nothing downstream of here
    # can fail on it.
    quarantine_file = (
        record_stranded_trades(store_root, venue, date, snapshot_id, stranded)
        if stranded else None)
    counts["quarantine_file"] = (
        str(quarantine_file) if quarantine_file is not None else None)

    if bars is None:
        return {"frames": frames, "trades": 0, "bars": 0, "snapshot_id": None,
                "by_symbol": by_symbol, **counts}
    return {"frames": frames, "trades": trades_kept, "bars": len(bars),
            "snapshot_id": snapshot_id, "by_symbol": by_symbol, **counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="store", description="Build the bitemporal store from captured frames.")
    parser.add_argument("--venue", required=True, choices=sorted(_TRADE_STREAMS))
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbols", required=True,
                        help=f"comma-separated, or {SYMBOLS_ALL} for every "
                             f"DOLLAR-QUOTED symbol whose trade tape this venue-day "
                             f"holds. A named list is built as given - the filter "
                             f"applies to {SYMBOLS_ALL}, which is where a universe "
                             f"gets chosen rather than stated")
    parser.add_argument("--include-non-dollar", action="store_true",
                        help=f"build every captured symbol under {SYMBOLS_ALL}, "
                             f"including pairs quoted in TRY, EUR, BTC and the rest. "
                             f"For archaeology on what was captured, not for a store "
                             f"anything ranks across")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    parser.add_argument("--lookahead-hours", type=int, default=DEFAULT_LOOKAHEAD_HOURS,
                        help="how far into the next day's folder to look for trades "
                             "whose event time still belongs to this day")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="build this many symbols per pass (0 = all at once). "
                             "Bounds how much work one collision throws away, not "
                             "peak memory - see BarAccumulator for that")
    args = parser.parse_args(argv)

    if args.batch_size < 0:
        parser.error("--batch-size cannot be negative")

    if args.symbols.strip().upper() == SYMBOLS_ALL:
        # Refusals propagate rather than being caught: an unreadable capture root
        # and a day nobody captured must not both exit 0 with nothing built.
        symbols = captured_symbols(Path(args.capture_root), args.venue, args.date)
        if not symbols:
            # The only path here is an absent venue/date folder - a captured day
            # with no trade tape raised above. Normal when building a day older
            # than the capture, and normal on the first pass after eviction.
            print(f"no capture on disk for {args.venue} {args.date}; nothing to build",
                  file=sys.stderr)
            return 0
        print(f"{len(symbols)} symbol(s) captured for {args.venue} {args.date}",
              file=sys.stderr)
        symbols = _dollar_quoted_only(
            symbols, Path(args.capture_root), args.venue, args.date,
            include_non_dollar=args.include_non_dollar)
        if not symbols:
            print(f"no dollar-quoted symbols captured for {args.venue} {args.date}; "
                  f"nothing to build", file=sys.stderr)
            return 0
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            parser.error("--symbols must name at least one symbol")

    # Batches bound how much work one collision throws away, and how much a
    # killed run loses. They are NOT the memory bound, whatever the numbers here
    # used to claim - and since 2026-08-10 nothing else is either. The reader
    # streams (`iter_pair`) and bars are folded as trades arrive, so a batch's
    # peak no longer tracks the largest hour file inside it: one hour of TUTUSDT
    # measured 3.98 GB read whole against 0.14 GB streamed. What is left in a
    # batch's peak is the bars it is accumulating, which is per symbol-minute
    # rather than per trade.
    #
    # Each batch reads its own set of files and therefore earns its own snapshot
    # id. That is correct rather than merely tolerable: partitions are per symbol,
    # and the id has always described the inputs a build actually read.
    size = args.batch_size or len(symbols)
    batches = [symbols[i:i + size] for i in range(0, len(symbols), size)]

    # A collision stops that batch, never the run. Unbatched, a day was
    # all-or-nothing; split into batches, a run killed halfway leaves the first
    # half built, and a re-run that stopped at the first collision could never
    # finish the rest - with the raw archive aged out from under it seven days
    # later. So each batch's refusal is recorded and the next one is attempted.
    summaries: list[dict] = []
    already_built: list[list[str]] = []
    for batch in batches:
        try:
            summaries.append(build_bars_for_day(
                Path(args.capture_root), Path(args.store_root),
                args.venue, args.date, batch, args.interval_ns, args.lookahead_hours))
        except PartitionExistsError as exc:
            already_built.append(batch)
            print(f"  already built, skipping {len(batch)} symbol(s): {exc}",
                  file=sys.stderr)

    # Nothing to do has to stay distinguishable from something done. The
    # supervisor reads a non-zero exit as "already built"; if a fully redundant
    # run exited 0, every hourly pass would read as a fresh build and the run log
    # would stop being evidence of anything.
    if not summaries:
        print(f"every batch was already built ({len(symbols)} symbol(s)); nothing to do",
              file=sys.stderr)
        # A dedicated code rather than 1, and rather than a string a caller has to
        # grep for. The supervisor has to tell "nothing to do" apart from "this
        # broke", and it used to do it by matching PartitionExistsError in the
        # traceback - which stopped appearing the moment main() started catching
        # the exception per batch. An exit code cannot drift out of sync with a
        # message that was never meant to be an interface.
        return EXIT_ALREADY_BUILT

    if already_built:
        print(f"  {sum(len(b) for b in already_built)} symbol(s) were already built; "
              f"built the remaining {sum(len(b) for b in batches) - sum(len(b) for b in already_built)}",
              file=sys.stderr)
    summary = _merge_summaries(summaries)
    print(f"{summary['frames']} frames -> {summary['trades']} trades -> "
          f"{summary['bars']} bars (snapshot {summary['snapshot_id']})", file=sys.stderr)
    # All three discard counters on one line, including a zero stranded count. A
    # reader must not have to infer a number from its absence: silence reads
    # equally well as "none" and as "this build does not report that", and those
    # are the two answers that must never look the same here.
    print(f"  {summary['lookahead_files']} next-day file(s) read ahead, "
          f"{summary['trades_deferred_to_next_day']} trade(s) deferred to the next day, "
          f"{summary['trades_covered_by_previous_day']} covered by the previous day's build, "
          f"{summary['trades_stranded']} stranded", file=sys.stderr)
    for skipped in summary["lookahead_files_skipped_live"]:
        print(f"  skipped lookahead hour (a writer still holds it open): {skipped}",
              file=sys.stderr)
    # Named individually and counted on its own line, because this is the one the
    # build is missing trades FROM. A skipped lookahead hour costs the tail of the
    # last bar; a skipped hour of the day being built costs that whole hour for
    # that symbol until a later pass, and a count with no names cannot be chased.
    if summary["day_files_skipped_live"]:
        print(f"  {len(summary['day_files_skipped_live'])} hour(s) of {args.date} "
              f"itself were skipped, still claimed by a live writer:", file=sys.stderr)
        for skipped in summary["day_files_skipped_live"]:
            print(f"    {skipped}", file=sys.stderr)
    for symbol, counts in summary["by_symbol"].items():
        print(f"  {symbol}: {counts['frames']} frames -> {counts['trades']} trades",
              file=sys.stderr)
    # Repeated loud and last, on its own line naming the file, when it is not
    # zero. The count above is there so a reader never has to infer it from
    # silence; this line is there because beside the routine deferrals it reads
    # exactly like one of them, which is how this loss stayed invisible in the
    # first place. The counter and the alarm are two different jobs.
    if summary["trades_stranded"]:
        print(f"STRANDED: {summary['trades_stranded']} trade(s) no build can recover; "
              f"quarantined at {summary['quarantine_file']}", file=sys.stderr)
    return 0


def _refuse_if_already_built(store_root: Path, interval_ns: int,
                             symbols: Sequence[str], snapshot_id: str) -> None:
    """Raise the collision now if every part this build would write already exists.

    Every part, not any: a build that would write four symbols and finds three
    already present still has work to do, and the fourth's `append_partition` call
    is what decides the rest. Only the fully-redundant case is short-circuited.

    Deliberately the same exception `append_partition` raises, because it is the
    same fact. A quieter return would turn "this was already built" into a success
    the supervisor could not distinguish from "this built one bar", and the
    append-only guarantee is worth more than a tidy exit code.
    """
    dataset = f"bars_{interval_ns}ns"
    root = Path(store_root) / dataset
    # The hour a part landed in is not knowable from the snapshot id - it comes
    # from the rows - so the question "is this symbol's part already written" is
    # asked of every hour directory rather than of one computed path. Glob rather
    # than a walk: the hour directories are the only level above symbol, and
    # readdir does not open a parquet file.
    targets = [any(root.glob(f"*/symbol={symbol}/part-{snapshot_id}.parquet"))
               for symbol in symbols]
    if targets and all(targets):
        raise PartitionExistsError(
            f"every part for snapshot {snapshot_id!r} already exists under "
            f"{Path(store_root) / dataset} for {len(targets)} symbol(s); this exact "
            f"build has been done. Refused before reading the day's files, which is "
            f"the whole saving - corrections are new snapshots, never rewrites")


def _merge_summaries(summaries: Sequence[dict]) -> dict:
    """Fold per-batch results into the one an unbatched run would have produced.

    Counters add. `snapshot_id` becomes a list, because there genuinely is one per
    batch and collapsing them to the first would name an id that accounts for a
    fraction of the output - the kind of tidy-looking summary that makes a partial
    build indistinguishable from a whole one.

    `quarantine_file` keeps only the batches that wrote one: a stranded trade is
    the alarm this module exists to raise, and it must not be averaged away.
    """
    if len(summaries) == 1:
        return summaries[0]

    merged: dict = {
        "frames": sum(s["frames"] for s in summaries),
        "trades": sum(s["trades"] for s in summaries),
        "bars": sum(s["bars"] for s in summaries),
        "trades_deferred_to_next_day": sum(
            s["trades_deferred_to_next_day"] for s in summaries),
        "trades_covered_by_previous_day": sum(
            s["trades_covered_by_previous_day"] for s in summaries),
        "trades_stranded": sum(s["trades_stranded"] for s in summaries),
        "lookahead_files": sum(s["lookahead_files"] for s in summaries),
        "lookahead_files_skipped_live": [
            name for s in summaries for name in s["lookahead_files_skipped_live"]],
        "day_files_skipped_live": [
            name for s in summaries for name in s["day_files_skipped_live"]],
        "snapshot_id": [s["snapshot_id"] for s in summaries],
        "quarantine_file": [s["quarantine_file"] for s in summaries
                            if s.get("quarantine_file")] or None,
        "by_symbol": {symbol: counts for s in summaries
                      for symbol, counts in s["by_symbol"].items()},
    }
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
