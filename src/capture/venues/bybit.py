"""Bybit linear perps — funding only, and deliberately nothing else.

Added 2026-08-09 for carry. `ARCHITECTURE.md` §3b settled the venue list on
2026-08-01 as *execution: Binance + Hyperliquid*, *reference-only: Kraken, OKX,
Coinbase*, *Bybit deferred*. That decision stands and this does not touch it:
its rationale is fee tiers, key scopes, integration cost and Bybit's Feb 2025
record ($1.5B stolen, $5.5B bank run in 24h) — every one of which is about
placing orders. **This venue is captured, not traded.** No key exists for it,
none is planned, and the endpoint below is public and unauthenticated.

What it buys is a third independent funding curve, on the same argument that
justified widening the other two: funding history cannot be backfilled into a
bitemporal archive after the fact, so an hour not captured today is an hour that
can only ever be reconstructed.

## Poll-only, and the plumbing already allows it

`core_specs` and `tail_specs` return nothing, so `run_capture` builds no
websocket source at all — it filters empty shards, and the poller becomes the
only source. `ws_url` and `subscribe_messages` exist to satisfy the venue
protocol and raise if anything ever calls them, which is better than returning a
URL that would quietly connect to a stream nobody meant to subscribe.

Trades and books here are deliberately not captured. They would double the
archive for a venue no order will ever reach, and the consolidated reference
price Layer 1 wants is a separate decision from carry.

## What one request returns, measured 2026-08-09

805 linear instruments: **765 perpetuals and 40 dated futures**. The dated ones
identify themselves — `fundingRate` is the empty string and `deliveryTime` is
set — so they are skipped by the funding reader on the venue's own say-so rather
than on a naming heuristic.

**Funding interval varies per symbol here**, which neither other venue does:
408 perps settle 4-hourly, 356 8-hourly and one hourly. Binance is uniformly
8-hourly and Hyperliquid hourly, so the interval could be left implicit for
them; here it cannot, and annualising a 4-hourly rate as though it were 8-hourly
is wrong by a factor of two in the direction that flatters the trade. The venue
publishes `fundingIntervalHour` and it is carried through to the store.
"""
from __future__ import annotations

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

# One request, unauthenticated, covering every linear instrument.
_TICKERS_URL = "https://api.bybit.com/v5/market/tickers?category=linear"
_FUNDING_STREAM = "linearTickers"
# Names the request, never a file - every instrument is written under its own
# symbol. Matches the word `--symbols ALL` uses at the other end of the pipeline.
_ALL_MARKET_SYMBOL = "ALL"
# Sixty seconds, for the reason the other two venues learned the hard way on
# this same day: the cost of a fan-out poll is not the request, it is the ~765
# files the response fans out into. At one second that killed binance capture
# outright - the event loop spent writing while the websocket keepalive went
# unanswered. There is no websocket here to lose, but the writes are the same
# and the archive is shared.
_FUNDING_INTERVAL_SECONDS = 60.0
# NOT measured against Bybit's published limits - nothing here has read a
# rate-limit header from this venue. `RateBudget` has no entry for it either and
# falls back to 600/minute, so one poll a minute at this weight sits far inside
# any plausible ceiling. The cadence is the real throttle.
_FUNDING_WEIGHT = 10


class BybitPushesNothingHere(RuntimeError):
    """Something asked this venue for a websocket. Nothing should."""


class BybitVenue:
    name = "bybit"

    # No depth is captured, so nothing chains depth updates.
    depth_is_binance_chained = False

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """One request for the whole linear market.

        `symbols` is ignored: the response is every instrument or none, so
        narrowing it would mean discarding data already fetched and paid for.
        """
        return [
            PollSpec(self.name, _FUNDING_STREAM, _ALL_MARKET_SYMBOL, _TICKERS_URL,
                     interval_seconds=_FUNDING_INTERVAL_SECONDS,
                     weight=_FUNDING_WEIGHT, fan_out=True),
        ]

    def fan_out_poll(self, spec: PollSpec, parsed) -> list[tuple[str, object]]:
        """Split the ticker list, and carry the venue's own clock onto each row.

        `retCode` is checked before the payload is read. Bybit answers HTTP 200
        with an error body, so a non-zero code is a failed request wearing a
        successful response, and splitting it would file an error object under
        every instrument that happened to be in the last good list.

        The venue's `time` is a TOP-LEVEL field and each ticker carries no clock
        of its own. It is merged into every element rather than left behind,
        because a record that cannot be dated without the envelope it arrived in
        is not an archive - the same reason Hyperliquid's coin name is merged in.
        """
        if not isinstance(parsed, dict) or parsed.get("retCode") != 0:
            return []
        result = parsed.get("result")
        if not isinstance(result, dict):
            return []
        tickers = result.get("list")
        if not isinstance(tickers, list):
            return []

        venue_time_ms = parsed.get("time")
        pairs = []
        for ticker in tickers:
            if not isinstance(ticker, dict):
                continue
            symbol = ticker.get("symbol")
            if isinstance(symbol, str) and symbol:
                pairs.append((symbol, {**ticker, "time": venue_time_ms}))
        return pairs

    def ws_url(self, specs: list[StreamSpec]) -> str:
        raise BybitPushesNothingHere(
            "this venue is polled for funding only; nothing subscribes a stream")

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        raise BybitPushesNothingHere(
            "this venue is polled for funding only; nothing subscribes a stream")

    def extract(self, parsed: dict) -> ExtractedMeta:
        """Only reachable if a non-fan-out frame ever arrives, which none does.

        Routed as control rather than guessed at: a frame this venue cannot
        explain must not be filed under a symbol invented to hold it.
        """
        return ExtractedMeta(None, None, "control", _FUNDING_STREAM, "unknown")
