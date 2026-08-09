"""Bybit liquidations — the feed Binance withholds, from the venue that pushes it.

DM-020's blocker, measured 2026-08-03 in `binance-withheld-streams.md`: Binance
accepts a `forceOrder` subscription from this host and never sends a frame, and
withdrew `allForceOrders` from REST, so there was no liquidation source at all
and the wall's tile has been honestly FAILING since. Bybit is the second venue
that ruling said would end the wait.

Measured from this box, 2026-08-09, before a line of this module existed:
805 linear instruments subscribed over one socket, 3/3 subscribe acks
`success:true`, **16 liquidation frames in 90 seconds** with real payloads.
Frames are the only evidence that counts on a stream like this — the Binance
lesson is that a subscribe ack means nothing, and Bybit's acks at least carry a
`success` flag this module checks frames against.

Like `bybit.py` this venue is **captured, not traded** — `ARCHITECTURE.md` §3b's
venue ruling is about placing orders and stands untouched. It is a separate
venue process rather than a stream added to the funding poller because the two
fail differently: a websocket cut must not take the funding poll down with it,
and the funding poller's own docstring promises it subscribes to nothing.

## The stream, verified against the venue's docs and then against the wire

Topic `allLiquidation.{symbol}`, endpoint `wss://stream.bybit.com/v5/public/linear`.
Snapshot frames, pushed at most every 500ms per symbol, each carrying venue
timestamp `ts` (ms) and rows `{T, s, S, v, p}` — time, symbol, liquidated side,
size, bankruptcy price. `S: "Buy"` means a LONG was liquidated (the venue's
wording, kept as delivered; the store can translate, the archive must not).

## Sharding, and why it is by subscribe characters rather than URL bytes

This venue carries nothing in its URL — topics are subscribed over the socket,
so `shard_by_url_budget`'s measure never grows and would put the whole universe
on one connection. The documented budget is different here: at most 21,000
characters of `args` per public connection. The full universe measured 21,244
characters on 2026-08-09 — over the line already, and the universe only grows.
The venue therefore declares `max_subscribe_chars` and the capture CLI shards
on that instead. (The probe's single over-budget connection did work; building
to the documented limit costs one extra socket and removes the bet that
enforcement never arrives.)

## Keepalive, and the risk that is recorded rather than solved

The recorder sends protocol-level pings every 20s and the venue answered them
throughout the probe. Bybit's docs additionally describe an application-level
`{"op": "ping"}` and a 10-minute idle cutoff; whether protocol pongs count as
liveness against that cutoff is not knowable from the docs and was not settled
by a 90-second probe. If the venue does cut an idle connection, the recorder
dies, the supervisor restarts it, and the silence lands in the ledger — the
failure is visible and bounded, not silent. Liquidations across 800+ symbols
arrived every few seconds in the probe, so a 10-minute market-wide silence is
itself an anomaly worth a ledger entry.

## Quiet is not dead — and the recorder already knows

A single symbol's liquidation stream is quiet for hours; that is the market,
not a fault. The recorder's silence detection judges each stream against its
own measured cadence with a grace floor, so per-symbol quiet does not flood the
ledger. What WOULD be recorded is the whole venue going silent, which is
exactly the event worth recording.
"""
from __future__ import annotations

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

_WS_URL = "wss://stream.bybit.com/v5/public/linear"
_STREAM = "allLiquidation"
_INSTRUMENTS_URL = ("https://api.bybit.com/v5/market/instruments-info"
                    "?category=linear&limit=1000")
# The venue's documented ceiling: 21,000 characters of `args` per public
# connection. Not measured as binding — the probe exceeded it and lived — but a
# documented limit is a promise the venue may start keeping at any time, and
# the cost of honouring it is one extra socket.
_MAX_SUBSCRIBE_CHARS = 21_000
# Topics per subscribe request. The probe sent 300 per request and every ack
# came back success:true; kept, since nothing measured a finer limit.
_TOPICS_PER_REQUEST = 300


class UniverseTruncated(RuntimeError):
    """The instruments listing has more pages than the one request fetched.

    Raised rather than returning the first page, because a capture built from a
    truncated universe silently drops every instrument past the cut — the same
    shape as the dollar-quote filter defect, where absence on disk read as
    absence in the market. Refusing names the fix: page through
    `nextPageCursor` in `fetch_instruments`, which becomes necessary only when
    the linear universe outgrows one 1,000-row page (805 on 2026-08-09).
    """


class BybitLiquidationVenue:
    name = "bybit-liq"

    depth_is_binance_chained = False
    max_subscribe_chars = _MAX_SUBSCRIBE_CHARS
    # Silence is judged for the stream as a whole, never per symbol. One
    # symbol's liquidation stream is rightly quiet for hours; what can die and
    # matter is the whole feed. Without this the recorder's first 75-second
    # run recorded 800 of 805 symbols silent - byte-holders included.
    market_wide_streams = frozenset({_STREAM})

    def _specs(self, symbols: list[str]) -> list[StreamSpec]:
        return [StreamSpec(self.name, _STREAM, symbol, f"{_STREAM}.{symbol}")
                for symbol in symbols]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        """Identical to the core: liquidations have no expensive variant.

        The core/tail split exists for venues where depth is costly and the
        tail gets only cheap streams. Here every stream is the cheap one, so
        the tail IS the capture and the core is just the operator-named subset
        that gets its own socket.
        """
        return self._specs(symbols)

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        return []

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_URL

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        """Batched subscribes, several requests per connection.

        Batching is about request size; the per-connection total is the shard
        budget the CLI enforces via `max_subscribe_chars` before this is called.
        """
        topics = [spec.channel for spec in specs]
        return [{"op": "subscribe", "args": topics[i:i + _TOPICS_PER_REQUEST]}
                for i in range(0, len(topics), _TOPICS_PER_REQUEST)]

    def extract(self, parsed) -> ExtractedMeta:
        """Route a frame to its stream and symbol, or to control.

        Op-replies (subscribe acks, pongs) carry `op` and no `topic`; they are
        control. A subscribe ack with success:false is still filed as control
        rather than raised here — the recorder's silence detection is the
        mechanism that turns a refused subscription into a recorded fact,
        because a raise from extract() would kill every other stream on the
        shard for one symbol's refusal.
        """
        if not isinstance(parsed, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        topic = parsed.get("topic")
        if not isinstance(topic, str) or not topic.startswith(f"{_STREAM}."):
            op = parsed.get("op")
            stream = op if isinstance(op, str) and op else "unknown"
            return ExtractedMeta(None, None, "control", stream, "unknown")

        rows = parsed.get("data")
        first = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
        symbol = first.get("s")
        if not isinstance(symbol, str) or not symbol:
            # The topic names the symbol too; the row is preferred because it is
            # inside the stored object, but a malformed row must not orphan the
            # frame from the stream that produced it.
            symbol = topic.partition(".")[2] or "unknown"

        t_exch_ms = parsed.get("ts")
        if not isinstance(t_exch_ms, int):
            t_exch_ms = None

        return ExtractedMeta(t_exch_ms, None, "data", _STREAM, symbol)

    def instruments_request(self) -> tuple[str, str, dict | None]:
        return ("GET", _INSTRUMENTS_URL, None)

    def parse_instruments(self, payload: dict) -> list[str]:
        """Symbols currently trading, from the listing the request fetched.

        Instruments in `Delivering`/`Closed` states are skipped on the venue's
        own `status` field: a non-trading instrument cannot be liquidated into,
        and subscribing it spends budget on a stream that can only be silent.
        """
        result_block = self._result_or_refuse(payload)
        symbols = []
        for item in result_block.get("list", []):
            if not isinstance(item, dict) or item.get("status") != "Trading":
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                symbols.append(symbol)
        return sorted(symbols)

    def parse_quote_assets(self, payload: dict) -> dict[str, str]:
        """Per-symbol quote currency, from the venue's own `quoteCoin` field.

        Read rather than inferred from the symbol suffix: `SOLPERP` carries no
        suffix a rule could parse, and the suffix heuristic is exactly the
        guess the universe snapshot exists to make unnecessary.
        """
        result_block = self._result_or_refuse(payload)
        quotes = {}
        for item in result_block.get("list", []):
            if not isinstance(item, dict) or item.get("status") != "Trading":
                continue
            symbol, quote = item.get("symbol"), item.get("quoteCoin")
            if isinstance(symbol, str) and symbol and isinstance(quote, str) and quote:
                quotes[symbol] = quote
        return quotes

    def _result_or_refuse(self, payload) -> dict:
        if not isinstance(payload, dict) or payload.get("retCode") != 0:
            return {}
        result_block = payload.get("result")
        if not isinstance(result_block, dict):
            return {}
        cursor = result_block.get("nextPageCursor")
        if isinstance(cursor, str) and cursor:
            raise UniverseTruncated(
                "bybit's linear listing no longer fits one page; page through "
                "nextPageCursor in fetch_instruments before capturing, or the "
                "tail silently loses every instrument past row 1,000")
        return result_block
