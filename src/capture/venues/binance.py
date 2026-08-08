"""Binance USDs-M perpetual futures. Spot is deliberately not captured - see spec 11 Q4."""
from __future__ import annotations

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

_WS_BASE = "wss://fstream.binance.com/stream?streams="
_INSTRUMENTS_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
_DEPTH_SNAPSHOT_URL = "https://fapi.binance.com/fapi/v1/depth"

# A depth diff is a changeset, not a book. Replaying `depthUpdate` frames into
# an order book requires an initial full snapshot to apply them onto, and
# without one the captured depth cannot be reconstructed at all - which is why
# the cost engine had to refuse every spread and impact question.
#
# Cadence and size are a request-weight decision, measured 2026-08-08 from the
# x-mbx-used-weight header rather than recalled: a limit=1000 snapshot costs 50
# on spot against a per-minute budget in the thousands, while premiumIndex costs
# 1. At the funding cadence of one per second the snapshot alone would spend the
# entire budget, so it carries its own interval. One per minute per core symbol
# also means every hourly file rotation contains several snapshots, so each hour
# stays independently replayable.
_DEPTH_SNAPSHOT_LIMIT = 1000
_DEPTH_SNAPSHOT_INTERVAL_SECONDS = 60.0
_DEPTH_SNAPSHOT_STREAM = "depthSnapshot"
# Measured 2026-08-08 from x-mbx-used-weight, not recalled.
_DEPTH_SNAPSHOT_WEIGHT = 50
_PREMIUM_INDEX_WEIGHT = 1

# `trade` rather than `aggTrade`, decided 2026-08-02 from live measurement:
# aggTrade delivers nothing at all to this host over the websocket (0 frames in
# 25s while depth and bookTicker flow normally, and REST /fapi/v1/aggTrades
# returns data - so the venue has it and a subset of websocket streams is
# silent). `trade` works, and it carries individual trades rather than
# aggregated ones, which is strictly more raw and the better fit for this layer.
# `markPrice@1s` was subscribed here until 2026-08-03 and removed after
# measurement, not suspicion. From this host, across three separate fstream edge
# IPs, it delivered zero frames in 35s while `trade` delivered 2001 on the same
# sockets - and so did `!markPrice@arr@1s`, which Binance guarantees at 1 Hz.
# The name is right (it matches Binance's own connector), the venue has the data
# (REST /fapi/v1/premiumIndex returns it), and COIN-M pushes the identical
# stream type normally. The feed now comes from `poll_specs` instead. Leaving
# the subscription in place would report it silent forever and bury the source
# that actually works.
#
# `forceOrder` stays, and the difference is deliberate: it is equally silent,
# but `allForceOrders` was withdrawn from the public REST API, so there is no
# replacement to move it to. An idle subscription costs nothing and is the only
# way this system would notice the venue starting to deliver liquidations. Until
# it does, the feed is genuinely unavailable and its tile is genuinely red.
_CORE_CHANNELS = ["depth@100ms", "trade", "forceOrder"]
_TAIL_CHANNELS = ["trade", "forceOrder"]

# Sampled once a second. Request weight is 1 per symbol against a 2400/minute
# budget, so three symbols spend 180/minute - the cadence is limited by what is
# worth storing, not by the venue's ceiling.
_POLL_STREAM = "premiumIndex"

# The stream name each event routes to. It must equal the `stream` on the
# StreamSpec that subscribed to it (`channel.split("@")[0]`), or one logical
# stream splits across two filenames. `aggTrade` stays mapped even though
# nothing subscribes to it now, so an archive captured earlier still routes.
_EVENT_TO_STREAM = {
    "depthUpdate": "depth",
    "trade": "trade",
    "aggTrade": "aggTrade",
    "markPriceUpdate": "markPrice",
    "forceOrder": "forceOrder",
}


class BinanceVenue:
    name = "binance"

    # Measured 2026-08-08 against the live endpoint, not taken from the docs:
    # 928 streams (16,338-byte URL) connect and deliver; 960 (16,886) return
    # HTTP 414. The documented 1024-stream cap is unreachable because the
    # request line dies first, and SUBSCRIBE over the socket - which would avoid
    # the URL entirely - is rejected with 1008 policy violation on fstream.
    # Full record: ~/research/binance-fstream-connection-limits.md
    #
    # 12,000 rather than something nearer the ceiling: the universe changes
    # daily, and the day a batch of long-named tokens lists must not be the day
    # capture discovers the limit. That headroom costs one extra connection.
    max_url_bytes = 12_000

    # Read by the recorder to pick a gap tracker. Futures chains depth updates
    # on `pu == prev.u`; spot chains on `U == prev.u + 1`. BinanceDepthTracker
    # handles both, and this is how it gets selected without the recorder
    # matching on a venue name.
    depth_is_binance_chained = True

    def _specs(self, symbols: list[str], channels: list[str]) -> list[StreamSpec]:
        return [
            StreamSpec(self.name, channel.split("@")[0], symbol, f"{symbol.lower()}@{channel}")
            for symbol in symbols
            for channel in channels
        ]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_CHANNELS)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_CHANNELS)

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """The feeds this venue will not push, fetched one symbol at a time.

        Per-symbol rather than the all-market form: omitting `symbol` returns
        every perpetual on the venue at request weight 10, and writing several
        hundred instruments to disk in order to read three of them is not a raw
        archive of what was asked for.
        """
        specs = [
            PollSpec(self.name, _POLL_STREAM, symbol,
                     f"{_PREMIUM_INDEX_URL}?symbol={symbol}",
                     weight=_PREMIUM_INDEX_WEIGHT)
            for symbol in symbols
        ]
        # Only the core symbols carry depth diffs, so only they need a snapshot
        # to replay those diffs onto. The tail subscribes trades alone.
        specs += [
            PollSpec(self.name, _DEPTH_SNAPSHOT_STREAM, symbol,
                     f"{_DEPTH_SNAPSHOT_URL}?symbol={symbol}"
                     f"&limit={_DEPTH_SNAPSHOT_LIMIT}",
                     interval_seconds=_DEPTH_SNAPSHOT_INTERVAL_SECONDS,
                     weight=_DEPTH_SNAPSHOT_WEIGHT)
            for symbol in symbols
        ]
        return specs

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_BASE + "/".join(spec.channel for spec in specs)

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        return []          # subscription is encoded in the URL

    def extract(self, parsed: dict) -> ExtractedMeta:
        if not isinstance(parsed, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        body = parsed.get("data", parsed)
        if not isinstance(body, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        event = body.get("e")
        if not isinstance(event, str):
            # A REST body has no event field - it is a bare object, not a
            # wrapped stream frame. `premiumIndex` is recognised by the fields
            # it is fetched for, so a polled response is routed as data rather
            # than dismissed as an unknown control frame.
            if isinstance(body.get("markPrice"), str) and isinstance(body.get("symbol"), str):
                t_poll_ms = body.get("time")
                return ExtractedMeta(
                    t_poll_ms if isinstance(t_poll_ms, int) else None,
                    None, "data", _POLL_STREAM, body["symbol"])
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        seq = None
        if event == "depthUpdate":
            seq = {k: body[k] for k in ("U", "u", "pu", "T") if k in body} or None

        t_exch_ms = body.get("E")
        if not isinstance(t_exch_ms, int):
            t_exch_ms = None

        symbol = body.get("s")
        if not isinstance(symbol, str) or not symbol:
            # Not every event puts the symbol at top level - forceOrder nests
            # order fields under "o". Fall back there before giving up.
            nested = body.get("o")
            symbol = nested.get("s") if isinstance(nested, dict) else None
            if not isinstance(symbol, str) or not symbol:
                symbol = "unknown"

        stream = _EVENT_TO_STREAM.get(event, event)
        return ExtractedMeta(t_exch_ms, seq, "data", stream, symbol)

    def instruments_request(self) -> tuple[str, str, dict | None]:
        return ("GET", _INSTRUMENTS_URL, None)

    def parse_instruments(self, payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return []

        result = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if item.get("contractType") != "PERPETUAL" or item.get("status") != "TRADING":
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                result.append(symbol)
        return sorted(result)

    def parse_quote_assets(self, payload: dict) -> dict[str, str]:
        """What each perpetual is priced in, from the venue's own field.

        Read live 2026-08-08 over the 569 PERPETUAL-and-TRADING pairs: 526 USDT,
        38 USDC, 2 USD1, 2 quoted in `U` and 1 in BTC. So futures is nearly all
        dollars and not entirely, and the three exceptions are exactly the kind
        that reads as a rounding error until one of them is in a P&L.

        Filtered the same way `parse_instruments` filters, deliberately: a quote
        map covering symbols the universe excludes would let a caller iterate the
        map and pick up a quarterly this venue never captures.
        """
        if not isinstance(payload, dict):
            return {}
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return {}

        quotes: dict[str, str] = {}
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if item.get("contractType") != "PERPETUAL" or item.get("status") != "TRADING":
                continue
            symbol, quote = item.get("symbol"), item.get("quoteAsset")
            if isinstance(symbol, str) and symbol and isinstance(quote, str) and quote:
                quotes[symbol] = quote
        return quotes
