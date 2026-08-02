"""Hyperliquid perps. l2Book carries no sequence numbers - staleness only."""
from __future__ import annotations

from capture.venues import ExtractedMeta, StreamSpec

_WS_URL = "wss://api.hyperliquid.xyz/ws"
_INSTRUMENTS_URL = "https://api.hyperliquid.xyz/info"

_CORE_TYPES = ["l2Book", "trades"]
_TAIL_TYPES = ["trades"]

_CONTROL_CHANNELS = ("subscriptionResponse", "pong", "error")


class HyperliquidVenue:
    name = "hyperliquid"

    def _specs(self, symbols: list[str], types: list[str]) -> list[StreamSpec]:
        return [StreamSpec(self.name, t, symbol, t) for symbol in symbols for t in types]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_TYPES)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_TYPES)

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
