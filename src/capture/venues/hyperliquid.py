"""Hyperliquid perps. l2Book carries no sequence numbers - staleness only."""
from __future__ import annotations

from capture.venues import QUOTE_USD, ExtractedMeta, PollSpec, StreamSpec

_WS_URL = "wss://api.hyperliquid.xyz/ws"
_INSTRUMENTS_URL = "https://api.hyperliquid.xyz/info"

_CORE_TYPES = ["l2Book", "trades"]
_TAIL_TYPES = ["trades"]

_CONTROL_CHANNELS = ("subscriptionResponse", "pong", "error")

# Funding, polled. `/info` is one POST with a type discriminator and no query
# string - there is no GET form and no websocket subscription that carries it.
_INFO_URL = "https://api.hyperliquid.xyz/info"
_META_AND_ASSET_CTXS_BODY = '{"type":"metaAndAssetCtxs"}'
_FUNDING_STREAM = "assetCtx"
# Names the request, never a file. Every coin is written under its own name.
_ALL_MARKET_SYMBOL = "ALL"
# Funding here is HOURLY on an oracle price and capped at 4%/hour - different
# economics from Binance's 8-hourly mark, and the reason this venue's carry is a
# distinct signal rather than a copy. Five seconds is far finer than an hourly
# rate needs, and the response also carries mark, oracle, premium and open
# interest, which do move continuously.
_FUNDING_INTERVAL_SECONDS = 5.0
# NOT measured against this venue's own limits, unlike the Binance weights.
# Hyperliquid's budget is consensus-bound and shaped differently, and nothing
# here has read a rate-limit header from it. Set to Binance's all-market cost as
# a placeholder that errs expensive; the cadence above is the real throttle.
_FUNDING_WEIGHT = 10


class HyperliquidVenue:
    name = "hyperliquid"

    def _specs(self, symbols: list[str], types: list[str]) -> list[StreamSpec]:
        return [StreamSpec(self.name, t, symbol, t) for symbol in symbols for t in types]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_TYPES)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_TYPES)

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """Funding, which this venue pushes to nobody.

        This returned nothing until 2026-08-09, on the note that "Hyperliquid
        pushes everything it is asked for". True of trades and books, and it left
        funding uncaptured entirely - so a venue whose funding is *hourly on an
        oracle price, capped at 4%/hour*, materially different economics from
        Binance's 8-hourly mark, had no carry history at all.

        One POST covers the market. `/info` answers no GET, which is why
        `PollSpec` carries a body.

        `symbols` is ignored on purpose: the response is the whole universe or
        nothing, so narrowing it would mean discarding data already fetched and
        paid for.
        """
        return [
            PollSpec(self.name, _FUNDING_STREAM, _ALL_MARKET_SYMBOL, _INFO_URL,
                     method="POST", body=_META_AND_ASSET_CTXS_BODY,
                     interval_seconds=_FUNDING_INTERVAL_SECONDS,
                     weight=_FUNDING_WEIGHT, fan_out=True),
        ]

    def fan_out_poll(self, spec: PollSpec, parsed) -> list[tuple[str, object]]:
        """Pair the universe with its contexts, which match by POSITION only.

        `metaAndAssetCtxs` returns `[{universe: [...]}, [ctx, ...]]`, and the
        halves are joined by index - `universe[i]` describes `ctxs[i]`. Nothing
        in a ctx names its coin, so an off-by-one here silently files BTC's
        funding under ETH and every downstream carry number is wrong while
        every file looks well-formed.

        `zip` is therefore refused: it truncates to the shorter side without a
        word, which is precisely the silent misalignment this must not do. A
        length mismatch means the venue changed shape and the whole response is
        dropped, which the recorder records as observation loss.

        The coin is merged INTO the stored object rather than left implied by
        the filename. A record that cannot be decoded without the half of the
        response that was not stored beside it is not an archive.
        """
        if not (isinstance(parsed, list) and len(parsed) == 2
                and isinstance(parsed[0], dict) and isinstance(parsed[1], list)):
            return []
        universe = parsed[0].get("universe")
        contexts = parsed[1]
        if not isinstance(universe, list) or len(universe) != len(contexts):
            return []

        pairs = []
        for entry, context in zip(universe, contexts):
            if not (isinstance(entry, dict) and isinstance(context, dict)):
                return []          # a shape change, not one bad row - drop the lot
            name = entry.get("name")
            if isinstance(name, str) and name:
                pairs.append((name, {"coin": name, **context}))
        return pairs

    def ws_url(self, specs: list[StreamSpec]) -> str:
        return _WS_URL

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        return [
            {"method": "subscribe",
             "subscription": {"type": spec.stream, "coin": spec.symbol}}
            for spec in specs
        ]

    def extract(self, parsed: dict) -> ExtractedMeta:
        if not isinstance(parsed, dict):
            return ExtractedMeta(None, None, "control", "unknown", "unknown")

        channel = parsed.get("channel")
        if not isinstance(channel, str) or channel in _CONTROL_CHANNELS:
            stream = channel if isinstance(channel, str) else "unknown"
            return ExtractedMeta(None, None, "control", stream, "unknown")

        data = parsed.get("data")
        if isinstance(data, list):
            item = data[0] if data and isinstance(data[0], dict) else {}
        elif isinstance(data, dict):
            item = data
        else:
            item = {}

        symbol = item.get("coin")
        if not isinstance(symbol, str) or not symbol:
            symbol = "unknown"

        t_exch_ms = item.get("time")
        if not isinstance(t_exch_ms, int):
            t_exch_ms = None

        return ExtractedMeta(t_exch_ms, None, "data", channel, symbol)

    def instruments_request(self) -> tuple[str, str, dict | None]:
        return ("POST", _INSTRUMENTS_URL, {"type": "meta"})

    def parse_instruments(self, payload: dict) -> list[str]:
        if not isinstance(payload, dict):
            return []
        universe = payload.get("universe")
        if not isinstance(universe, list):
            return []

        result = []
        for item in universe:
            if not isinstance(item, dict) or item.get("isDelisted", False):
                continue
            name = item.get("name")
            if isinstance(name, str) and name:
                result.append(name)
        return sorted(result)

    def parse_quote_assets(self, payload: dict) -> dict[str, str]:
        """`USD` for every listed perp, and said as a constant rather than a read.

        This venue publishes no per-symbol quote field. Measured live 2026-08-08,
        the keys on a `meta` universe entry are exactly isDelisted, marginMode,
        marginTableId, maxLeverage, name, onlyIsolated and szDecimals; the payload
        carries a top-level `collateralToken`, and its value is the integer 0 - a
        token index, not a name, resolvable only through a second endpoint.

        So the denomination here is a property of the venue, not of the listing:
        every perp is marked and settled in USD against USDC collateral. Returning
        the constant is honest and the docstring says which of the two it is. What
        would not be honest is a `quoteAsset` lookup that silently found nothing
        and left every hyperliquid symbol classified `unknown` - the same 232
        symbols dropped from a dollar-quoted universe they all belong to.
        """
        return {symbol: QUOTE_USD for symbol in self.parse_instruments(payload)}
