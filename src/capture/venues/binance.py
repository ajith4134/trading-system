"""Binance USDs-M perpetual futures. Spot is deliberately not captured - see spec 11 Q4."""
from __future__ import annotations

from capture.venues import ExtractedMeta, StreamSpec

_WS_BASE = "wss://fstream.binance.com/stream?streams="
_INSTRUMENTS_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# `trade` rather than `aggTrade`, decided 2026-08-02 from live measurement:
# aggTrade delivers nothing at all to this host over the websocket (0 frames in
# 25s while depth and bookTicker flow normally, and REST /fapi/v1/aggTrades
# returns data - so the venue has it and a subset of websocket streams is
# silent). `trade` works, and it carries individual trades rather than
# aggregated ones, which is strictly more raw and the better fit for this layer.
_CORE_CHANNELS = ["depth@100ms", "trade", "markPrice@1s", "forceOrder"]
_TAIL_CHANNELS = ["trade", "markPrice@1s", "forceOrder"]

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
