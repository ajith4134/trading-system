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
                tick = self._parse(raw)
                if tick is not None:
                    self._buffer.put(tick)

    def _parse(self, raw) -> LiveTick | None:
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return None
        data = message.get("data", message)
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

    def __init__(self, *, venue: str, source: _SourceThread, buffer: _TickBuffer,
                 kind: str, detail: str,
                 quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> None:
        self.venue = venue
        self.kind = kind
        self.detail = detail
        self._source = source
        self._buffer = buffer
        self._quiet_after_ns = quiet_after_ns
        self._started = False

    def start(self) -> LiveFeed:
        self._source.start()
        self._started = True
        return self

    def stop(self) -> None:
        self._source.stop()

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
        return now_ns - newest

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
            "reconnects": self._source.reconnects,
            "last_error": self._source.last_error,
        }


def websocket_feed(*, venue: str, url: str, detail: str,
                   capacity: int = DEFAULT_BUFFER,
                   quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    buffer = _TickBuffer(capacity)
    source = WebsocketTickSource(venue, url, buffer)
    return LiveFeed(venue=venue, source=source, buffer=buffer,
                    kind=_WEBSOCKET, detail=detail, quiet_after_ns=quiet_after_ns)


def rest_poll_feed(*, venue: str, url: str, fan_out, interval_seconds: float,
                   detail: str, capacity: int = DEFAULT_BUFFER,
                   quiet_after_ns: int = DEFAULT_QUIET_AFTER_NS) -> LiveFeed:
    buffer = _TickBuffer(capacity)
    source = RestPollTickSource(venue, url, buffer, fan_out, interval_seconds)
    return LiveFeed(venue=venue, source=source, buffer=buffer,
                    kind=_REST_POLL, detail=detail, quiet_after_ns=quiet_after_ns)
