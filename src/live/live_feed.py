"""BF-01: the live market feed every segment bot trades on, and the age of what it hands over.

## Why this module exists at all

**Ruling RL-024, 2026-08-18, in the user's words:** *"sould te paper tradin sould be
done on live data on live crypto prices not on old data"*.

Until this module existed, `paper.forward_engine` took its prices from the parquet
store through `store.clock_gated_reader`. Three measurements from 2026-08-17 and
2026-08-18 say why that cannot be a trading clock:

* a cold filtered scan of the live bars dataset measured **185.6 seconds**,
* the running engine sat **14 minutes in uninterruptible IO without completing one
  poll**, on a box at load 20,
* an hour-pruned read filtered to three symbols was **killed at 300 seconds** having
  returned nothing.

And even a fast store read would be the wrong input. The store is a Layer-1 build
downstream of a raw tape that flushes a zstd frame every 30 seconds, which is itself
downstream of the venue. A bot polling it is backtesting on a delay while carrying the
name paper trading, and the P&L it produces looks like a result.

So: **no segment bot reads a price from the store.** The store keeps what it is good
at - the research and training corpus, deep history, reproducible reads - and this
module owns the trading clock.

## Two source kinds, because the venues are not alike

`websocket` and `rest_poll` are both live. They differ in how live, and the difference
is published rather than smoothed over.

| segment | venue | source | why |
|---|---|---|---|
| perp | binance futures | websocket | `trade` + `bookTicker`, sub-second |
| spot | binance spot | websocket | same streams, same latency |
| dated | bybit | rest_poll | dated futures ride the `linear` tickers REST call; `capture.venues.bybit` established this - the venue pushes nothing useful here |
| options | deribit | rest_poll | `get_book_summary_by_currency` is the whole chain in one request; `capture.venues.deribit` measured a websocket subscription as **1,502 channels** to fetch what one request covers |

A REST poll every few seconds is seconds old. The store is hours old. Both are "not a
websocket", and treating them as the same kind of staleness is how an options bot ends
up quietly trading yesterday.

## The age of a tick is part of the tick

Every tick carries `received_ns` - when THIS process saw it - alongside the venue's own
timestamp. `newest_age_ns()` reports the age of the freshest thing the feed holds, and
the engine writes it into its heartbeat.

The reason it is not optional: a websocket that has silently stopped delivering looks
exactly like a market with no trades. Both give an empty poll. Only the age separates
them, and one of the two means the bot is flying blind while its logs read normal. A
feed past `quiet_after_ns` reports QUIET, and QUIET is a refusal to trade rather than a
market with no opportunities in it.

## What this module will not do

* **It does not reconnect silently and forget.** Every reconnect is counted and the
  count is on the feed's description, because a feed reconnecting every 20 seconds is a
  broken feed that otherwise reads as healthy.
* **It does not fill gaps.** A tick that did not arrive is absent. Interpolating one
  would put a price the venue never printed in front of a brain.
* **It does not write to the store.** Capture owns that path and already has it. Two
  writers to one archive is a corruption source, and this process is not the one that
  should own durable history.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

# Sub-second on a websocket, and the REST pollers are bounded by their own interval.
# A feed quiet for this long is not a slow market; something is wrong with the
# connection, and the bots stop rather than trade on the last price they happen to
# hold.
DEFAULT_QUIET_AFTER_NS = 120_000_000_000

# How many ticks a feed buffers between polls. A bot polling every few seconds over a
# handful of symbols consumes far less; the headroom is for the first poll after a
# restart and for a burst. When it overflows the OLDEST are dropped and the drop is
# counted - dropping the newest would mean acting on a stale price while a fresh one
# was discarded, which is the wrong direction to fail in.
DEFAULT_BUFFER = 200_000

_WEBSOCKET = "websocket"
_REST_POLL = "rest_poll"

QUIET = "QUIET"
LIVE = "LIVE"
NEVER_DELIVERED = "NEVER_DELIVERED"


class FeedNotStarted(RuntimeError):
    """A poll arrived before the feed was started. Always a wiring fault."""


def _decimal_or_none(value) -> Decimal | None:
    """A price the venue did not send is None, never zero.

    Zero is a price. `bid_price` of null on a Deribit option with no bid means
    nobody is bidding; rendering that as 0 would let a brain read a free option and
    a fill model happily oblige.
    """
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed


@dataclass(frozen=True)
class LiveTick:
    """One observation from a venue, with when this process saw it.

    `venue_ts_ns` is the venue's own clock and may be absent or skewed - it is
    recorded, never trusted for staleness. `received_ns` is this process's monotonic
    view of arrival and is what `newest_age_ns` measures, because it is the only one
    of the two that cannot be wrong about whether the connection is alive.
    """

    venue: str
    symbol: str
    received_ns: int
    venue_ts_ns: int | None = None
    price: Decimal | None = None
    quantity: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: Decimal | None = None
    ask_size: Decimal | None = None
    # Segment-specific payload that has no place in a shared shape: an option's
    # expiry and mark IV, a dated future's delivery time. The brains that need it
    # know its keys; nothing generic reads it.
    extra: dict = field(default_factory=dict)

    @property
    def mid(self) -> Decimal | None:
        """The two-sided mid, or None. Never a one-sided fallback.

        A mid computed from a single side is not a mid, and the places that want one
        - fill modelling, feature frames - are exactly the places where quietly
        substituting the last trade would be invisible and wrong.
        """
        if self.bid is None or self.ask is None:
            return None
        if self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2

    @property
    def has_two_sided_quote(self) -> bool:
        return self.mid is not None


class _TickBuffer:
    """Thread-safe hand-off from a source thread to the engine's poll.

    A deque with a maxlen rather than a queue: the engine polls on its own cadence and
    a slow poll must never block the socket reader. `websockets` kills a slow consumer,
    which `capture.venue_recorder` already learned the hard way.
    """

    def __init__(self, capacity: int) -> None:
        self._lock = threading.Lock()
        self._ticks: deque[LiveTick] = deque(maxlen=capacity)
        self._dropped = 0
        self._delivered = 0
        self._newest_received_ns: int | None = None

    def put(self, tick: LiveTick) -> None:
        with self._lock:
            if len(self._ticks) == self._ticks.maxlen:
                self._dropped += 1
            self._ticks.append(tick)
            self._delivered += 1
            if (self._newest_received_ns is None
                    or tick.received_ns > self._newest_received_ns):
                self._newest_received_ns = tick.received_ns

    def drain(self) -> tuple[LiveTick, ...]:
        with self._lock:
            drained = tuple(self._ticks)
            self._ticks.clear()
        return drained

    @property
    def newest_received_ns(self) -> int | None:
        with self._lock:
            return self._newest_received_ns

    @property
    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self._delivered, self._dropped


class _SourceThread:
    """Shared lifecycle for both source kinds.

    Subclasses implement `_run_once`, which is expected to return on failure rather
    than raise its way out of the thread: a source thread that dies leaves a feed that
    reports LIVE forever with an ageing newest tick, which is the failure this whole
    module is written against.
    """

    def __init__(self, venue: str, buffer: _TickBuffer) -> None:
        self.venue = venue
        self._buffer = buffer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reconnects = 0
        self._last_error: str = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"live-feed-{self.venue}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def reconnects(self) -> int:
        return self._reconnects

    @property
    def last_error(self) -> str:
        return self._last_error

    def _loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._run_once()
                backoff = 1.0
            except Exception as exc:                      # noqa: BLE001
                # Deliberately broad. Every exception out of a venue connection is
                # the same decision here - record it, back off, reconnect - and a
                # narrower catch would let an unforeseen one kill the thread and
                # leave the feed silently dead.
                self._last_error = f"{type(exc).__name__}: {exc}"
            if self._stop.is_set():
                return
            self._reconnects += 1
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def _run_once(self) -> None:
        raise NotImplementedError


class WebsocketTickSource(_SourceThread):
    """Binance combined streams: `trade` for prints, `bookTicker` for the quote.

    Both are needed and neither substitutes for the other. `trade` says a trade
    happened and at what price - that is the tape a momentum feature reads.
    `bookTicker` says what is currently biddable and offerable - that is what a fill
    model needs to decide whether a resting order would have been hit.
    """

    def __init__(self, venue: str, url: str, buffer: _TickBuffer) -> None:
        super().__init__(venue, buffer)
        self._url = url

    def _run_once(self) -> None:
        import asyncio

        asyncio.run(self._consume())

    async def _consume(self) -> None:
        from websockets.asyncio.client import connect

        async with connect(self._url, max_queue=4096,
                           ping_interval=20, ping_timeout=20) as socket:
            while not self._stop.is_set():
                raw = await socket.recv()
                for tick in self._parse_many(raw):
                    self._buffer.put(tick)

    def _parse_many(self, raw) -> list:
        """One raw frame into zero or more ticks.

        The all-market `!markPrice@arr@1s` stream delivers an ARRAY of every symbol's
        mark and funding in a single frame. A parser that only understood objects
        would silently discard the entire funding feed for 570 symbols and leave the
        bear brain's crowded-short check reading None forever - which is the same
        silent-guard failure the per-symbol markPrice stream was added to fix.
        """
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return []
        payload = message.get("data", message) if isinstance(message, dict) else message
        if isinstance(payload, list):
            ticks = [self._parse_one(row) for row in payload if isinstance(row, dict)]
            return [tick for tick in ticks if tick is not None]
        if isinstance(payload, dict):
            tick = self._parse_one(payload)
            return [tick] if tick is not None else []
        return []

    def _parse(self, raw) -> LiveTick | None:
        ticks = self._parse_many(raw)
        return ticks[0] if ticks else None

    def _parse_one(self, data: dict) -> LiveTick | None:
        if not isinstance(data, dict):
            return None
        symbol = data.get("s")
        if not isinstance(symbol, str) or not symbol:
            return None
        received = time.time_ns()
        event = data.get("e")
        venue_ts_ms = data.get("T") or data.get("E")
        venue_ts = int(venue_ts_ms) * 1_000_000 if venue_ts_ms else None

        if event in ("trade", "aggTrade"):
            return LiveTick(
                venue=self.venue, symbol=symbol, received_ns=received,
                venue_ts_ns=venue_ts,
                price=_decimal_or_none(data.get("p")),
                quantity=_decimal_or_none(data.get("q")),
                # `m` is is-buyer-maker, so a True means the AGGRESSOR was a seller.
                # Recorded as the taker side because that is what order-flow
                # imbalance counts, and inverting it is the classic sign error.
                extra={"taker_side": "SELL" if data.get("m") else "BUY"})
        if event == "markPriceUpdate":
            return LiveTick(
                venue=self.venue, symbol=symbol, received_ns=received,
                venue_ts_ns=venue_ts,
                extra={"funding_rate": data.get("r"),
                       "mark_price": data.get("p"),
                       "index_price": data.get("i"),
                       "next_funding_ns": (int(data["T"]) * 1_000_000
                                           if data.get("T") else None)})
        # bookTicker carries no `e` on spot. Identified by its fields instead of by
        # a name the venue does not always send.
        if "b" in data and "a" in data:
            return LiveTick(
                venue=self.venue, symbol=symbol, received_ns=received,
                venue_ts_ns=venue_ts,
                bid=_decimal_or_none(data.get("b")),
                ask=_decimal_or_none(data.get("a")),
                bid_size=_decimal_or_none(data.get("B")),
                ask_size=_decimal_or_none(data.get("A")))
        return None


class RestPollTickSource(_SourceThread):
    """One request per interval, fanned out into one tick per instrument.

    Used where the venue's websocket is the wrong tool, which
    `capture.venues.bybit` and `capture.venues.deribit` each established on measured
    grounds rather than preference. The interval is the honest resolution of this
    feed and is published on the description.
    """

    def __init__(self, venue: str, url: str, buffer: _TickBuffer,
                 fan_out, interval_seconds: float, timeout_seconds: float = 15.0) -> None:
        super().__init__(venue, buffer)
        self._url = url
        self._fan_out = fan_out
        self._interval = interval_seconds
        self._timeout = timeout_seconds

    def _run_once(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            request = urllib.request.Request(
                self._url, headers={"User-Agent": "ajit-live-feed/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
            received = time.time_ns()
            for tick in self._fan_out(self.venue, payload, received):
                self._buffer.put(tick)
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self._interval - elapsed))


def fan_out_bybit_linear(venue: str, payload: dict, received_ns: int) -> list[LiveTick]:
    """Bybit `v5/market/tickers?category=linear` into one tick per instrument.

    Both perpetuals and dated futures arrive in this one response - 765 and 40
    respectively when `capture.venues.bybit` measured it. `deliveryTime` is what
    separates them: non-zero means the contract expires, and that is the dated
    universe. The separation is not done here; DB-01 owns admission and needs the
    field, so it is passed through rather than filtered on.
    """
    rows = (payload.get("result") or {}).get("list") or []
    ticks: list[LiveTick] = []
    for row in rows:
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            continue
        delivery_ms = row.get("deliveryTime") or "0"
        try:
            delivery_ns = int(delivery_ms) * 1_000_000
        except (TypeError, ValueError):
            delivery_ns = 0
        ticks.append(LiveTick(
            venue=venue, symbol=symbol, received_ns=received_ns,
            price=_decimal_or_none(row.get("lastPrice")),
            bid=_decimal_or_none(row.get("bid1Price")),
            ask=_decimal_or_none(row.get("ask1Price")),
            bid_size=_decimal_or_none(row.get("bid1Size")),
            ask_size=_decimal_or_none(row.get("ask1Size")),
            quantity=_decimal_or_none(row.get("volume24h")),
            extra={"delivery_ns": delivery_ns,
                   "funding_rate": row.get("fundingRate"),
                   "mark_price": row.get("markPrice"),
                   "index_price": row.get("indexPrice")}))
    return ticks


DERIBIT_CHAIN_URL = ("https://www.deribit.com/api/v2/public/"
                     "get_book_summary_by_currency?currency={currency}&kind=option")
BINANCE_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_BOOK_TICKER = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"


def binance_dated_fan_out(delivery_by_symbol: dict):
    """Build a fan-out for binance DATED contracts from `premiumIndex`.

    **Why this and not the all-market websocket.** `!bookTicker` carries quotes for
    every futures symbol including the dated ones, but the dated brains reason about
    the basis to the INDEX, and neither the index nor the delivery date is on that
    stream. `premiumIndex` returns `markPrice` and `indexPrice` for all 874 symbols in
    one request, which is precisely the pair `dated.segment_brains._annualised_basis`
    needs.

    The delivery date comes from discovery and is closed over here. It is a property
    of the CONTRACT rather than of the tick - it does not change between polls - so
    fetching it once and joining it in beats asking the venue for it every few seconds.
    """
    def fan_out(venue: str, payload, received_ns: int) -> list:
        rows = payload if isinstance(payload, list) else []
        ticks = []
        for row in rows:
            symbol = row.get("symbol")
            delivery_ns = delivery_by_symbol.get(symbol)
            if not delivery_ns:
                # Not a dated contract this bot was given. Perpetuals dominate this
                # response and belong to the perp bot.
                continue
            mark = _decimal_or_none(row.get("markPrice"))
            ticks.append(LiveTick(
                venue=venue, symbol=symbol, received_ns=received_ns,
                venue_ts_ns=(int(row["time"]) * 1_000_000 if row.get("time") else None),
                price=mark,
                extra={"delivery_ns": delivery_ns,
                       "mark_price": row.get("markPrice"),
                       "index_price": row.get("indexPrice"),
                       "funding_rate": row.get("lastFundingRate")}))
        return ticks
    return fan_out


def binance_dated_quote_fan_out(delivery_by_symbol: dict):
    """Best bid and ask for the dated contracts, from the REST book ticker.

    A separate call from `premiumIndex` because the two carry different things and
    neither is optional: without the quote there is no two-sided market and the frame
    refuses; without mark and index there is no basis and the brains decline.
    """
    def fan_out(venue: str, payload, received_ns: int) -> list:
        rows = payload if isinstance(payload, list) else []
        ticks = []
        for row in rows:
            symbol = row.get("symbol")
            if symbol not in delivery_by_symbol:
                continue
            ticks.append(LiveTick(
                venue=venue, symbol=symbol, received_ns=received_ns,
                bid=_decimal_or_none(row.get("bidPrice")),
                ask=_decimal_or_none(row.get("askPrice")),
                bid_size=_decimal_or_none(row.get("bidQty")),
                ask_size=_decimal_or_none(row.get("askQty")),
                extra={"delivery_ns": delivery_by_symbol[symbol]}))
        return ticks
    return fan_out


def fan_out_deribit_chain(venue: str, payload: dict, received_ns: int) -> list[LiveTick]:
    """Deribit `get_book_summary_by_currency` into one tick per option instrument.

    `capture.venues.deribit` measured this response at 363 KB and 0.36s for the whole
    BTC chain, and measured `high`, `low` and `price_change` null on 588 of 818
    instruments. So a null here is normal and expected, which is exactly why
    `_decimal_or_none` refuses to turn one into a zero: on an option chain, most
    fields being absent is the base case, and a zero bid on a call is a free option.
    """
    rows = payload.get("result") or []
    ticks: list[LiveTick] = []
    for row in rows:
        instrument = row.get("instrument_name")
        if not isinstance(instrument, str) or not instrument:
            continue
        ticks.append(LiveTick(
            venue=venue, symbol=instrument, received_ns=received_ns,
            price=_decimal_or_none(row.get("last")),
            bid=_decimal_or_none(row.get("bid_price")),
            ask=_decimal_or_none(row.get("ask_price")),
            quantity=_decimal_or_none(row.get("volume")),
            extra={"mark_price": row.get("mark_price"),
                   "mark_iv": row.get("mark_iv"),
                   "underlying_price": row.get("underlying_price"),
                   "open_interest": row.get("open_interest")}))
    return ticks


def binance_futures_url(symbols: list[str]) -> str:
    """Combined `trade` + `bookTicker` streams for the USD-M futures venue.

    **`aggTrade` is not used, and the reason is measured rather than stylistic.**
    Subscribed alongside the other two on 2026-08-18 over a 30-second window,
    `btcusdt@bookTicker` delivered 2,747 messages, `btcusdt@trade` delivered 296, and
    `btcusdt@aggTrade` delivered **zero**. Binance accepts the subscription and then
    serves nothing on it, so the failure is silent: a bot built on it would see quotes
    updating normally and simply never observe a trade, and every tape feature would
    read as a market where nothing prints.
    """
    return _binance_url("wss://fstream.binance.com/stream?streams=", symbols,
                        mark_price=True)


def binance_spot_url(symbols: list[str]) -> str:
    return _binance_url("wss://stream.binance.com:9443/stream?streams=", symbols)


# Binance caps a connection at 1024 streams. Kept well under so a universe that grows
# between a discovery call and a reconnect cannot silently cross it.
MAX_STREAMS_PER_CONNECTION = 480


def binance_futures_all_market_urls() -> list[str]:
    """Quotes for EVERY futures symbol, in one stream.

    `!bookTicker` carries the best bid and ask of every symbol the venue lists, so the
    whole board's quotes cost one connection. That is what makes RL-014's "consider
    everything" affordable rather than a fan-out of 570 sockets.

    **Funding is NOT here, and the reason is measured.** `!markPrice@arr@1s` and
    `!markPrice@arr` were both probed on 2026-08-18: the connection opens, the
    subscription is accepted, and nothing is ever delivered - a clean timeout after 14
    seconds on a stream documented to push every one to three seconds. Funding
    therefore comes from the `premiumIndex` REST poll, which returned all 874 symbols
    in one request. A stream that accepts a subscription and serves nothing is the
    same silent failure `aggTrade` had, and the same answer applies: measure what
    arrives, use what does.
    """
    return ["wss://fstream.binance.com/ws/!bookTicker"]


def binance_spot_all_market_url() -> str:
    """DEPRECATED BY THE VENUE - kept so its absence is documented, not rediscovered.

    Binance no longer serves an all-market `!bookTicker` on spot. Probed 2026-08-18:
    the connection opens and nothing arrives, exactly as with `!ticker@arr`. Spot
    quotes therefore come from per-symbol `@bookTicker` streams, sharded - six
    connections for 1,361 pairs instead of the one the futures board needs.
    """
    return "wss://stream.binance.com:9443/ws/!bookTicker"


def binance_quote_and_trade_shard_urls(base: str, symbols: list[str],
                                       per_connection: int = MAX_STREAMS_PER_CONNECTION
                                       ) -> list[str]:
    """Per-symbol `@bookTicker` AND `@trade`, sharded. For venues with no all-market quote."""
    streams = []
    for symbol in symbols:
        lowered = symbol.lower()
        streams.append(f"{lowered}@bookTicker")
        streams.append(f"{lowered}@trade")
    return [base + "/".join(streams[i:i + per_connection])
            for i in range(0, len(streams), per_connection)]


def binance_funding_fan_out(symbols=None):
    """Funding and mark for every perpetual, from the `premiumIndex` REST poll.

    Replaces the `!markPrice@arr` stream the venue accepts and never serves. The perp
    BEAR brain refuses a short into a crowded-short funding rate, and without this the
    guard reads None on every symbol and silently never fires.
    """
    wanted = set(symbols) if symbols else None

    def fan_out(venue: str, payload, received_ns: int) -> list:
        rows = payload if isinstance(payload, list) else []
        ticks = []
        for row in rows:
            symbol = row.get("symbol")
            if wanted is not None and symbol not in wanted:
                continue
            ticks.append(LiveTick(
                venue=venue, symbol=symbol, received_ns=received_ns,
                venue_ts_ns=(int(row["time"]) * 1_000_000 if row.get("time") else None),
                extra={"funding_rate": row.get("lastFundingRate"),
                       "mark_price": row.get("markPrice"),
                       "index_price": row.get("indexPrice")}))
        return ticks
    return fan_out


def composite_feed(*, venue: str, detail: str, websocket_urls=(), rest_endpoints=(),
                   rest_interval_seconds: float = 5.0,
                   capacity: int = DEFAULT_BUFFER,
                   quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    """One feed mixing websocket and REST sources over a shared buffer.

    The perp segment needs both: quotes and trades push over sockets, funding only
    comes back from a REST call because the venue's funding stream serves nothing.
    They are one market to the bot, so they are one buffer and one `poll()`.
    """
    buffer = _TickBuffer(capacity)
    sources = [WebsocketTickSource(venue, url, buffer) for url in websocket_urls]
    sources += [RestPollTickSource(venue, url, buffer, fan_out, rest_interval_seconds)
                for url, fan_out in rest_endpoints]
    if not sources:
        raise ValueError(f"{venue}: a feed with no source would report QUIET forever")
    return LiveFeed(venue=venue, sources=sources, buffer=buffer,
                    kind="composite", detail=detail, quiet_after_ns=quiet_after_ns)


def binance_trade_shard_urls(base: str, symbols: list[str],
                             per_connection: int = MAX_STREAMS_PER_CONNECTION) -> list[str]:
    """Per-symbol `@trade` streams, sharded across connections.

    **There is no all-market trade stream, and that is why this exists.** `!bookTicker`
    gives the whole market's quotes for free, but the AGGRESSOR SIDE only comes from
    per-symbol trade streams, and order-flow imbalance is the perp bull and bear
    brains' primary input. Dropping it would leave breadth with no flow feature - the
    universe would be wide and the brains would have less to reason with on every
    symbol in it.

    Sharding follows `capture.venues.shard_by_url_budget`'s reasoning rather than its
    code: that module budgets by URL bytes because its streams ride in the URL, and so
    do these, but the binding limit here is the venue's 1024-streams-per-connection cap.
    """
    urls = []
    for start in range(0, len(symbols), per_connection):
        shard = symbols[start:start + per_connection]
        urls.append(base + "/".join(f"{s.lower()}@trade" for s in shard))
    return urls


def _binance_url(base: str, symbols: list[str], *, mark_price: bool = False) -> str:
    streams = []
    for symbol in symbols:
        lowered = symbol.lower()
        streams.append(f"{lowered}@trade")
        streams.append(f"{lowered}@bookTicker")
        if mark_price:
            # Futures only, and the ONLY stream that carries the funding rate. The
            # perp BEAR brain refuses a short into a crowded-short funding rate, and
            # without this stream that check reads a field that is always None and
            # silently never fires - a guard that looks present and is not.
            streams.append(f"{lowered}@markPrice@1s")
    return base + "/".join(streams)


class LiveFeed:
    """What a segment bot holds. One venue, one source, ticks since the last poll."""

    def __init__(self, *, venue: str, sources, buffer: _TickBuffer,
                 kind: str, detail: str,
                 quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> None:
        self.venue = venue
        self.kind = kind
        self.detail = detail
        # **Several connections, one buffer.** Covering 570 perpetuals takes an
        # all-market quote stream, an all-market funding stream and two shards of
        # per-symbol trade streams - four sockets whose ticks are one market. They
        # share a buffer so the engine still makes one `poll()` call and cannot end up
        # holding a quote from one connection and a trade from another as if they were
        # separate feeds.
        self._sources = list(sources)
        self._buffer = buffer
        self._quiet_after_ns = quiet_after_ns
        self._started = False

    def start(self) -> LiveFeed:
        for source in self._sources:
            source.start()
        self._started = True
        return self

    def stop(self) -> None:
        for source in self._sources:
            source.stop()

    def poll(self) -> tuple[LiveTick, ...]:
        """Every tick that arrived since the last call, oldest first."""
        if not self._started:
            raise FeedNotStarted(
                f"{self.venue}: poll() before start(); the feed has no connection")
        return self._buffer.drain()

    def newest_age_ns(self, now_ns: int) -> int | None:
        """Age of the freshest tick this feed has EVER seen, or None if never any.

        Deliberately not reset by a poll. The question it answers is "is this
        connection alive", and draining the buffer is not evidence about that.
        """
        newest = self._buffer.newest_received_ns
        if newest is None:
            return None
        # Clamped at zero. The source thread keeps appending after the caller took
        # its `now_ns`, so a tick can be newer than the clock it is measured against
        # and the age comes out slightly negative - measured at -0.15s on the perp
        # feed 2026-08-18. Harmless as a race, corrosive as a display: a board
        # showing a negative age invites the reader to distrust every other number
        # on it. Zero is the honest floor, because "newer than the moment I asked"
        # is what actually happened.
        return max(0, now_ns - newest)

    def liveness(self, now_ns: int) -> str:
        age = self.newest_age_ns(now_ns)
        if age is None:
            return NEVER_DELIVERED
        return QUIET if age > self._quiet_after_ns else LIVE

    def describe(self, now_ns: int) -> dict:
        """What the heartbeat and the board tile publish about this feed."""
        delivered, dropped = self._buffer.counts
        age = self.newest_age_ns(now_ns)
        return {
            "venue": self.venue,
            "kind": self.kind,
            "detail": self.detail,
            "liveness": self.liveness(now_ns),
            "newest_age_ns": age,
            "newest_age_seconds": None if age is None else round(age / 1e9, 3),
            "ticks_delivered": delivered,
            "ticks_dropped": dropped,
            "connections": len(self._sources),
            # Summed across connections: one socket flapping while three are healthy
            # is still a broken feed, and an average would hide it.
            "reconnects": sum(s.reconnects for s in self._sources),
            "last_error": next((s.last_error for s in self._sources if s.last_error), ""),
        }


def websocket_feed(*, venue: str, detail: str, url: str | None = None,
                   urls=None, capacity: int = DEFAULT_BUFFER,
                   quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    """One venue, one or many sockets, one buffer."""
    if url is not None and urls is not None:
        raise ValueError("pass url or urls, not both")
    endpoints = [url] if url is not None else list(urls or ())
    if not endpoints:
        raise ValueError(f"{venue}: a feed with no endpoint would report QUIET forever")
    buffer = _TickBuffer(capacity)
    sources = [WebsocketTickSource(venue, endpoint, buffer) for endpoint in endpoints]
    return LiveFeed(venue=venue, sources=sources, buffer=buffer,
                    kind=_WEBSOCKET, detail=detail, quiet_after_ns=quiet_after_ns)


def multi_rest_poll_feed(*, venue: str, endpoints, interval_seconds: float,
                         detail: str, capacity: int = DEFAULT_BUFFER,
                         quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    """One feed over several REST endpoints, each with its own fan-out.

    The dated segment needs three: binance `premiumIndex` for mark and index, binance
    `ticker/bookTicker` for the two-sided quote, and bybit's tickers for its own board.
    They are one market to the bot and one buffer here, so `poll()` stays a single call
    and no caller has to know the segment spans two venues.
    """
    buffer = _TickBuffer(capacity)
    sources = [RestPollTickSource(venue, url, buffer, fan_out, interval_seconds)
               for url, fan_out in endpoints]
    return LiveFeed(venue=venue, sources=sources, buffer=buffer,
                    kind=_REST_POLL, detail=detail, quiet_after_ns=quiet_after_ns)


def rest_poll_feed(*, venue: str, url: str, fan_out, interval_seconds: float,
                   detail: str, capacity: int = DEFAULT_BUFFER,
                   quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    buffer = _TickBuffer(capacity)
    endpoints = [url] if isinstance(url, str) else list(url)
    sources = [RestPollTickSource(venue, endpoint, buffer, fan_out, interval_seconds)
               for endpoint in endpoints]
    return LiveFeed(venue=venue, sources=sources, buffer=buffer,
                    kind=_REST_POLL, detail=detail, quiet_after_ns=quiet_after_ns)
