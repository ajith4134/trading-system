"""Deribit options — the third segment, which had no data at all.

Added 2026-08-16. The user ruled the bot is intraday on *"all segments spot,
futures, options"*; §3a of the goal document records that options was not a phase
behind but a segment with **zero captured data**, and `FEATURES.md` §5b's own
header says its layer is *"required before any options position"*.

Deribit carries ~85-90% of BTC/ETH options volume, which the corpus already
recorded, and everything else is thin enough that a wide spread is the norm
rather than the exception.

## Why now rather than later, in one fact

**The chain cannot be backfilled.** Measured 2026-08-16: passing a `timestamp`
parameter to `get_book_summary_by_currency` is ignored and the response is the
current chain. There is no public historical option-chain endpoint. So an hour
not captured today is an hour that can only ever be reconstructed - the same
argument that widened funding capture to Bybit, and the reason this is worth a
process before any options feature exists to read it.

DVOL, by contrast, IS backfillable: `get_volatility_index_data` takes
`start_timestamp`/`end_timestamp` and returned 1,000 points for a single day when
probed. It is therefore deliberately NOT captured here. Recording a backfillable
series live costs archive weight to buy nothing, and the endpoint is written down
above so it can be pulled whenever a feature needs it.

## Polled, not subscribed, and the arithmetic is the reason

Deribit does have a websocket. Subscribing it would mean **1,502 channels** - one
per live BTC and ETH option - against ONE request that returns the whole chain in
363 KB and 0.36 seconds. `get_book_summary_by_currency` is the entire book
summary for a currency: mark price, mark IV, bid, ask, open interest and the
underlying, per instrument.

So `core_specs` and `tail_specs` return nothing and `run_capture` builds no
websocket at all, exactly as Bybit does. `ws_url` and `subscribe_messages` raise
rather than returning a URL that would quietly connect to a stream nobody meant.

## Cadence, and what actually costs something

Sixty seconds, and the constraint is NOT the request. One response fans out into
**1,502 files**, and this archive has already been taught what that costs twice:
857 funding writers rotating in one tick killed the binance recorder's websocket
at every hour boundary, and a one-second fan-out poll killed capture outright.
There is no websocket here to lose - which is the point of it being its own
process - but the writes are the same and the archive is shared.

The endpoint is edge-cached with `cache-control: public, max-age=1`, so polling
faster than one second cannot return anything newer regardless.

**Deribit publishes no rate-limit header on its public endpoints** - measured, not
assumed: the response carries no `x-ratelimit-*` or `retry-after` at all. So
`RateBudget` has no measured entry for this venue and falls back to its default,
and the cadence is the real throttle. Two requests a minute sits far inside any
plausible public ceiling, and the alternative - probing for the limit by hammering
an exchange - is how a ban is earned rather than avoided.

## Two currencies, because two is what exists

Measured 2026-08-16 across every currency Deribit lists: **BTC 818 option
instruments, ETH 684, and SOL, XRP and the other 47 currencies zero.** They are
listed currencies with no listed options. Naming BTC and ETH here is therefore a
statement of what the venue has, not a narrowing of it - and a venue that starts
listing SOL options will show up as a currency this file does not poll, which is
a visible gap rather than a silent one.

## What the frames carry that the reader will have to respect

**Every row carries its own clock.** `creation_timestamp` is the summary's
generation time, not the instrument's listing date - measured: 818 rows held 9
distinct values spanning 8 milliseconds, all within a second of the request. So
unlike Bybit and Hyperliquid, nothing has to be merged in from the envelope for a
row to be datable.

**Nulls are everywhere and they are real market structure**, not defects:
`high`, `low` and `price_change` are null on **588 of 818** instruments, `last`
on 139, and `bid_price` and `mid_price` on **89**. An option with no bid is an
option nobody will buy at any price, which is ordinary for far out-of-the-money
strikes.

They are recorded verbatim, because `raw_writer` archives the payload byte-exact.
**The extractor that reads them must refuse a null rather than coerce it.** The
prior-art Deribit client in `nse-crypto-bot-final` did `float(info.get(...) or
0.0)` on exactly these fields, which turns 89 missing bids into a bid of zero -
a valid-looking price that makes a worthless option look free, and one more
prior-art defect failing in the flattering direction.

**Options here are quoted in the BASE currency**, not in dollars: a BTC option's
`quote_currency` is `BTC` and its mark price is a fraction of a coin. Anything
that prices these in USD must multiply by the underlying, and `store.quote_currency`
will need to learn this venue rather than assume the dollar default.
"""
from __future__ import annotations

from capture.venues import ExtractedMeta, PollSpec, StreamSpec

_BASE = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"

# Backfillable, therefore deliberately not polled. Written down so the decision is
# visible rather than looking like an oversight:
#   https://www.deribit.com/api/v2/public/get_volatility_index_data
#       ?currency=BTC&start_timestamp=..&end_timestamp=..&resolution=60

# The currencies that actually list options, measured against the live venue
# rather than assumed from the currency list - 49 of Deribit's 51 currencies have
# none. A currency added here without that check would poll forever for an empty
# result and report a captured venue.
OPTION_CURRENCIES = ("BTC", "ETH")

_CHAIN_STREAM = "optionChain"
# Names the REQUEST, never a file. Every instrument is written under its own
# `instrument_name`, and this is the word the other end of the pipeline already
# uses for "one call covering the whole market".
_ALL_MARKET_SYMBOL = "ALL"

# Sixty seconds. See the module docstring: the cost is the 1,502 files this fans
# out into, not the two requests.
_CHAIN_INTERVAL_SECONDS = 60.0
# Deribit charges no published weight on public endpoints and returns no header to
# measure one from. One is the honest value - it means "one request" - rather than
# a number invented to look calibrated.
_CHAIN_WEIGHT = 1


class DeribitPushesNothingHere(RuntimeError):
    """Something asked this venue for a websocket. Nothing should.

    Deribit has one; this venue does not use it. Raising beats returning a URL
    that would subscribe 1,502 channels to fetch what one request already covers.
    """


class DeribitVenue:
    name = "deribit"

    # No depth is captured, so nothing chains depth updates.
    depth_is_binance_chained = False

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """One request per currency, each covering that currency's whole chain.

        `symbols` is ignored, as it is for every fan-out venue here: the response
        is every option on the currency or none, so narrowing it would mean
        discarding data already fetched and paid for.
        """
        return [
            PollSpec(self.name, _CHAIN_STREAM, _ALL_MARKET_SYMBOL,
                     f"{_BASE}?currency={currency}&kind=option",
                     interval_seconds=_CHAIN_INTERVAL_SECONDS,
                     weight=_CHAIN_WEIGHT, fan_out=True)
            for currency in OPTION_CURRENCIES
        ]

    def fan_out_poll(self, spec: PollSpec, parsed) -> list[tuple[str, object]]:
        """Split the chain into one record per option instrument.

        A JSON-RPC error body is checked for before the payload is read. Deribit
        answers a bad request with HTTP 200 and an `error` member instead of
        `result`, so an error is a failure wearing a successful response - and
        splitting it would file an error object under every instrument that
        happened to be in the last good chain.

        Rows are passed through UNCHANGED. Each already carries its own
        `creation_timestamp`, so unlike Bybit's tickers and Hyperliquid's trades
        there is no envelope clock to merge in, and merging one anyway would put a
        second, differently-derived timestamp beside the venue's own.
        """
        if not isinstance(parsed, dict):
            return []
        if parsed.get("error") is not None:
            return []
        rows = parsed.get("result")
        if not isinstance(rows, list):
            return []

        pairs = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            instrument = row.get("instrument_name")
            # An unnamed row cannot be filed. Skipped rather than given a
            # generated name, which would put a record in the archive under a
            # symbol that does not exist on the venue.
            if isinstance(instrument, str) and instrument:
                pairs.append((instrument, row))
        return pairs

    def ws_url(self, specs: list[StreamSpec]) -> str:
        raise DeribitPushesNothingHere(
            "this venue is polled for the option chain; nothing subscribes a stream")

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        raise DeribitPushesNothingHere(
            "this venue is polled for the option chain; nothing subscribes a stream")

    def extract(self, parsed: dict) -> ExtractedMeta:
        """Only reachable if a non-fan-out frame ever arrives, which none does.

        Routed as control rather than guessed at: a frame this venue cannot
        explain must not be filed under a symbol invented to hold it.
        """
        return ExtractedMeta(None, None, "control", _CHAIN_STREAM, "unknown")
