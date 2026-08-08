"""Binance spot. A separate venue from the USDⓈ-M futures adapter, deliberately.

The venue's `name` becomes a directory and (stream, symbol) becomes a filename
inside it, so sharing a name with the futures adapter would file spot BTCUSDT
and perpetual BTCUSDT as one instrument. Two different markets, merged, with
nothing reporting it - and the spot-perp basis this exists to enable is exactly
the difference between them.

Three things are genuinely different from futures here, each measured against
the live endpoint on 2026-08-08 rather than assumed from the futures adapter:

  * A different host. `stream.binance.com:9443`, not `fstream`.
  * No liquidations and no funding. Spot has neither, so subscribing to
    `forceOrder` or polling `premiumIndex` would report two feeds silent
    forever - which is how a real outage gets lost among feeds that were never
    coming.
  * Depth carries no `pu`. Futures chains updates on `pu == prev.u`; spot
    chains on `U == prev.u + 1`. `BinanceDepthTracker` already handles both, and
    `depth_is_binance_chained` is how it gets selected without the recorder
    matching on a venue name.
"""
from __future__ import annotations

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

_WS_BASE = "wss://stream.binance.com:9443/stream?streams="
_INSTRUMENTS_URL = "https://api.binance.com/api/v3/exchangeInfo"

# No `forceOrder`: spot has no liquidations. The futures adapter keeps that
# subscription deliberately, because there it is a real feed the venue withholds
# and the only way to notice it returning. Here there is nothing to notice.
_CORE_CHANNELS = ["depth@100ms", "trade"]
_TAIL_CHANNELS = ["trade"]

_EVENT_TO_STREAM = {
    "depthUpdate": "depth",
    "trade": "trade",
    "aggTrade": "aggTrade",
}


class BinanceSpotVenue:
    name = "binance-spot"

    # Measured 2026-08-08: 1,024 streams at 14,846 bytes connect and deliver on
    # this endpoint - unlike fstream, where the request line fails at ~16.3KB
    # before the documented 1,024-stream cap is reached. The budget is set below
    # both limits so neither is the thing that discovers the ceiling, and so a
    # batch of long-named listings cannot push a shard over it.
    max_url_bytes = 12_000

    # Read by the recorder to pick a gap tracker. A capability rather than a
    # name match: the recorder used to test `venue.name == "binance"`, which was
    # correct only while exactly one Binance venue existed.
    depth_is_binance_chained = True

    def _specs(self, symbols: list[str], channels: list[str]) -> list[StreamSpec]:
        return [
            StreamSpec(self.name, channel.split("@")[0], symbol,
                       f"{symbol.lower()}@{channel}")
            for symbol in symbols
            for channel in channels
        ]

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _CORE_CHANNELS)

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return self._specs(symbols, _TAIL_CHANNELS)

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """Nothing. Every feed this venue has, it pushes."""
        return []

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
            # `pu` is absent on spot and that absence is the continuity rule,
            # not a missing field: the tracker falls back to `u` when it is not
            # there. Recorded as whatever arrived rather than normalised, so the
            # archive says what the venue said.
            seq = {k: body[k] for k in ("U", "u", "pu", "T") if k in body} or None

        t_exch_ms = body.get("E")
        if not isinstance(t_exch_ms, int):
            t_exch_ms = None

        symbol = body.get("s")
        if not isinstance(symbol, str) or not symbol:
            symbol = "unknown"

        return ExtractedMeta(t_exch_ms, seq, "data",
                             _EVENT_TO_STREAM.get(event, event), symbol)

    def instruments_request(self) -> tuple[str, str, dict | None]:
        return ("GET", _INSTRUMENTS_URL, None)

    def parse_instruments(self, payload: dict) -> list[str]:
        """Every symbol currently tradeable, whatever it is quoted in.

        Not filtered to USDT pairs. The basis trade needs the quote currency
        that matches its perpetual, but deciding that is the strategy layer's
        job - Layer 0 records what was listed, and a pair excluded here is a
        pair that cannot be backfilled later.
        """
        if not isinstance(payload, dict):
            return []
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return []

        result = []
        for item in symbols:
            if not isinstance(item, dict) or item.get("status") != "TRADING":
                continue
            symbol = item.get("symbol")
            if isinstance(symbol, str) and symbol:
                result.append(symbol)
        return sorted(result)
