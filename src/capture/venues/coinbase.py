"""Coinbase spot, captured without a key - which is the point of this module.

The build plan carried coinbase as BLOCKED: *"entry lacks `api_key` or
`api_secret` - blocked on user"*. That is true of the private API and it was
never true of the market data. Probed from this host 2026-08-10, keyless:

  * `GET /products` - 832 listings, 517 online, HTTP 200.
  * `matches` over `wss://ws-feed.exchange.coinbase.com` - 53 trades on BTC-USD
    in 12 seconds.
  * `level2_batch` - subscription accepted, answered as `level2_50`.
  * `heartbeat` - one per product per second, carrying a sequence number.

So the blocked half is trading, and the captured half was available all along. A
row left blocked on the wrong premise is a row nobody re-examines, which is why
this was measured rather than inherited.

## Three things that differ from the binance adapters, all measured

**The subscription rides the socket, not the URL.** One connect, one subscribe
message. Every one of the 517 online product ids together is 5,578 characters -
a quarter of what bybit's ceiling allows - so this venue declares no budget and
takes one connection, and that is a measurement rather than an omission.

**Trades are sequence-chained; depth is not.** A `match` frame carries
`sequence` and so does `heartbeat`, which is what makes a missed trade
detectable. An `l2update` carries `product_id` and `changes` and no sequence at
all, so depth here is staleness-only - the hyperliquid shape, not the binance
one. `depth_is_binance_chained` is therefore absent, and the recorder picks the
right tracker without matching on a venue name.

**The book snapshot is bigger than the default frame limit.** Measured at
1,209,067 bytes on BTC-USD, against the 1 MiB default in `websockets`. Left
alone it closes the connection with 1009 and no frames arrive - which reads
exactly like a quiet market. The recorder now sets an explicit ceiling; see the
note there.

## Symbols

Coinbase names a market `BTC-USD`, hyphen included, and that string is what goes
in the archive - not a normalised `BTCUSD`. Two venues naming the same
instrument differently is a fact about the venues, and Layer 1 already resolves
it: the store keys bars by (symbol, venue), and `store.quote_currency` reads the
quote from the venue's own field rather than off the symbol string.
"""
from __future__ import annotations

from datetime import datetime, timezone

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

_WS_URL = "wss://ws-feed.exchange.coinbase.com"
_INSTRUMENTS_URL = "https://api.exchange.coinbase.com/products"
# `level=2` is the aggregated book. Measured 2026-08-10 on BTC-USD: 18,976 bid
# levels and 27,038 ask levels, which is the whole book rather than the top 50
# the endpoint's documentation describes. The store keeps 20 levels a side, so
# the archive holds what the venue said and the dataset holds what is used.
_DEPTH_SNAPSHOT_URL_TEMPLATE = (
    "https://api.exchange.coinbase.com/products/{product}/book?level=2")
_DEPTH_SNAPSHOT_STREAM = "depthSnapshot"
_DEPTH_SNAPSHOT_INTERVAL_SECONDS = 300.0

# `matches` is the trade tape. `level2_batch` is depth, answered by the venue as
# `level2_50` - 50 levels, which is what `FEATURES.md` asks for. `heartbeat`
# earns its place: it is one frame per product per second carrying the current
# sequence, so a product that has simply not traded stays distinguishable from a
# product whose feed died. Without it, a quiet market and a dead subscription
# are the same silence.
_CORE_CHANNELS = ["matches", "level2_batch", "heartbeat"]
# The broad tail takes trades only, matching the other venues: depth on 517
# products is a different order of write volume and there is no all-market form
# of it.
_TAIL_CHANNELS = ["matches"]

# The venue's own type field, mapped to the stream name the archive files under.
# `last_match` is the one-off frame sent at subscribe carrying the most recent
# trade; it is filed with the trades because that is what it is, and dropping it
# would discard the only trade some quiet products publish all hour.
_TYPE_TO_STREAM = {
    "match": "matches",
    "last_match": "matches",
    "snapshot": "level2Snapshot",
    "l2update": "level2",
    "heartbeat": "heartbeat",
}


def _venue_time_ms(value) -> int | None:
    """Coinbase stamps an ISO8601 instant; everything downstream wants millis.

    Returns None rather than a guess when the field is missing or unparseable.
    A fabricated timestamp on a trade is worse than an absent one: the archive
    records receipt time regardless, and a wrong venue clock would silently
    reorder the tape against every other venue.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1000)


class CoinbaseVenue:
    name = "coinbase"

    def _specs(self, symbols: list[str], channels: list[str]) -> list[StreamSpec]:
        return [
            StreamSpec(self.name, channel, symbol, channel)
            for symbol in symbols
            for channel in channels
        ]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_CHANNELS)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_CHANNELS)

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """A periodic full book, which the socket does not provide.

        The websocket sends one `snapshot` when the subscription opens and diffs
        thereafter, so without this the book dataset would hold one row per
        recorder restart. `store.book_snapshots` builds from snapshots and does
        not replay diffs, deliberately - so a periodic snapshot is what a
        reference price can actually be built from.

        Five minutes, not one. This venue is reference-price only by
        `ARCHITECTURE.md` §3b - execution is binance and hyperliquid - and the
        cost is not the request, it is the archive: measured 2026-08-10, one
        BTC-USD book is 1,208,060 bytes raw and 355,923 compressed. Three
        symbols at 60s is 1.5 GB/day; at 300s it is 308 MB/day, which against
        seven-day retention is 2.2 GB held for a venue nothing trades on.
        `consolidated_price` judges staleness against each venue's own routine
        gap rather than a fixed bound, so a five-minute book is not read as a
        dead one.

        Nothing else to poll: spot has no funding.
        """
        return [
            PollSpec(self.name, _DEPTH_SNAPSHOT_STREAM, symbol,
                     f"{_DEPTH_SNAPSHOT_URL_TEMPLATE.format(product=symbol)}",
                     interval_seconds=_DEPTH_SNAPSHOT_INTERVAL_SECONDS)
            for symbol in symbols
        ]

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_URL

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        """One message. The venue takes product ids and channel names as lists,
        so a spec list of 517 products by 3 channels collapses to a single
        subscribe rather than one per pair."""
        products = sorted({spec.symbol for spec in specs})
        channels = sorted({spec.channel for spec in specs})
        if not products or not channels:
            return []
        return [{"type": "subscribe", "product_ids": products,
                 "channels": channels}]

    def extract(self, parsed: dict) -> ExtractedMeta:
        if not isinstance(parsed, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        message_type = parsed.get("type")
        if not isinstance(message_type, str):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        stream = _TYPE_TO_STREAM.get(message_type)
        if stream is None:
            # `subscriptions`, `error`, and anything the venue adds later.
            # Filed as control under the venue's own word for it, so a new
            # message type is visible in the archive rather than silently
            # renamed to something it is not.
            return ExtractedMeta(None, None, "control", message_type, "unknown")

        symbol = parsed.get("product_id")
        if not isinstance(symbol, str) or not symbol:
            symbol = "unknown"

        # Trades and heartbeats carry the venue's sequence; l2updates do not,
        # and that absence is the reason depth on this venue is staleness-only.
        # Recorded as whatever arrived rather than defaulted, so the archive
        # says what the venue said.
        sequence = parsed.get("sequence")
        seq = {"sequence": sequence} if isinstance(sequence, int) else None

        return ExtractedMeta(_venue_time_ms(parsed.get("time")), seq, "data",
                             stream, symbol)

    def instruments_request(self) -> tuple[str, str, dict | None]:
        return ("GET", _INSTRUMENTS_URL, None)

    def parse_instruments(self, payload) -> list[str]:
        """Every market currently online and not disabled.

        Measured 2026-08-10: 832 listings, of which 517 are `online` and 315 are
        `delisted` - and every delisted one also carries `trading_disabled`.
        Both conditions are checked rather than either, because a market can be
        halted without being delisted and subscribing to it would report a feed
        silent forever.

        Not filtered by quote currency. 482 are quoted in USD, 116 in USDT, 87
        in EUR and the rest in eight other currencies; which of those a strategy
        wants is the strategy layer's question, and a market excluded here is
        one that cannot be backfilled later.
        """
        if not isinstance(payload, list):
            return []
        result = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "online" or item.get("trading_disabled"):
                continue
            product = item.get("id")
            if isinstance(product, str) and product:
                result.append(product)
        return sorted(result)

    def parse_quote_assets(self, payload) -> dict[str, str]:
        """What each market is priced in, from the venue's own field.

        The field, never the symbol string - the same rule the binance adapters
        learned the hard way. `BTC-USD` looks parseable and `ETH-BTC` looks like
        a dollar pair to anything matching on a suffix.
        """
        if not isinstance(payload, list):
            return {}
        quotes = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            if item.get("status") != "online" or item.get("trading_disabled"):
                continue
            product, quote = item.get("id"), item.get("quote_currency")
            if isinstance(product, str) and product and isinstance(quote, str) and quote:
                quotes[product] = quote
        return quotes
