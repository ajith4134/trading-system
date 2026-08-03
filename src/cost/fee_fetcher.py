"""Fetches a venue's published fee schedule, where the venue will publish one.

Probed 2026-08-03 from this host:

    hyperliquid  POST /info {"type":"userFees"}   full schedule + VIP ladder,
                                                  unauthenticated, zero address
    binance      GET /fapi/v1/commissionRate      401 {"code":-2014} unsigned
                 GET /sapi/v1/asset/tradeFee      400
                 GET /fapi/v1/exchangeInfo        liquidationFee only

So one venue is fetchable by anyone and the other needs a signed request. The
Binance path is implemented rather than stubbed, because the only thing it was
ever missing is a key - and a function that raises "not implemented" would
invite a future reader to conclude the endpoint does not exist, which is not
what was measured.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from decimal import Decimal

from cost.fee_schedule import DECLARED_SCHEDULES, FeeRate, FeeSchedule, FeeSource
from cost.secret_store import CredentialsUnavailable, VenueCredentials, read_venue_credentials

_HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
_BINANCE_COMMISSION_URL = "https://fapi.binance.com/fapi/v1/commissionRate"

# Any address is accepted by `userFees` and the base schedule comes back
# regardless - measured with the zero address on 2026-08-03. This is what lets a
# schedule be read before an account exists.
_ANONYMOUS_ADDRESS = "0x0000000000000000000000000000000000000000"

_REQUEST_TIMEOUT_SECONDS = 20
_RECV_WINDOW_MS = 5_000


class FeeScheduleUnavailable(Exception):
    """The venue would not answer, and why. Never a silent fallback."""


def sign_query(secret: str, query_string: str) -> str:
    """HMAC-SHA256 over the exact query string, hex encoded.

    Binance signs the string as sent, so the caller must sign the same encoding
    it transmits: re-encoding parameters after signing changes the bytes and the
    venue rejects it. Verified against the worked example in Binance's own
    SIGNED-endpoint documentation - see `test_fee_fetcher.py`.
    """
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()


async def _get_json(url: str, headers: dict | None = None) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers or {}) as response:
            body = await response.text()
            if response.status != 200:
                raise FeeScheduleUnavailable(
                    f"{url.split('?')[0]} returned {response.status}: {body[:200]}")
            return json.loads(body)


async def _post_json(url: str, payload: dict) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise FeeScheduleUnavailable(
                    f"{url} returned {response.status}: {body[:200]}")
            return json.loads(body)


def parse_hyperliquid_fees(body: dict, now_ns: int) -> FeeSchedule:
    """`userFees` -> the perp schedule. Raises rather than defaulting."""
    schedule = body.get("feeSchedule")
    if not isinstance(schedule, dict):
        raise FeeScheduleUnavailable("userFees response carried no feeSchedule")
    add, cross = schedule.get("add"), schedule.get("cross")
    if not isinstance(add, str) or not isinstance(cross, str):
        raise FeeScheduleUnavailable("feeSchedule is missing add/cross rates")

    return FeeSchedule(
        venue="hyperliquid", instrument_kind="perp",
        # `add` is the maker (resting) rate, `cross` the taker (crossing) one.
        rate=FeeRate.from_unit_rates(add, cross),
        tier="base",
        source=FeeSource.VENUE_API,
        source_detail='POST /info {"type":"userFees"}',
        fetched_at_ns=now_ns)


async def fetch_hyperliquid_schedule(address: str = _ANONYMOUS_ADDRESS,
                                     now_ns: int | None = None,
                                     post=None) -> FeeSchedule:
    post = post or _post_json
    body = await post(_HYPERLIQUID_INFO_URL, {"type": "userFees", "user": address})
    return parse_hyperliquid_fees(body, now_ns if now_ns is not None else time.time_ns())


def parse_binance_commission(body: dict, now_ns: int) -> FeeSchedule:
    maker, taker = body.get("makerCommissionRate"), body.get("takerCommissionRate")
    if not isinstance(maker, str) or not isinstance(taker, str):
        raise FeeScheduleUnavailable(
            "commissionRate response is missing maker/taker rates")
    return FeeSchedule(
        venue="binance", instrument_kind="perp",
        rate=FeeRate.from_unit_rates(maker, taker),
        tier="account",              # the venue answers for THIS account's tier
        source=FeeSource.VENUE_API,
        source_detail="GET /fapi/v1/commissionRate (signed)",
        fetched_at_ns=now_ns)


async def fetch_binance_schedule(symbol: str = "BTCUSDT",
                                 credentials: VenueCredentials | None = None,
                                 now_ns: int | None = None,
                                 get=None) -> FeeSchedule:
    """The signed request. Needs a key with *reading* permission only.

    Raises `FeeScheduleUnavailable` when no credential exists, which is the
    state until a read-only key is issued. The caller's correct response is to
    fall back to the declared schedule and stay marked unverified - never to
    invent a number.
    """
    if credentials is None:
        try:
            credentials = read_venue_credentials("binance")
        except CredentialsUnavailable as error:
            raise FeeScheduleUnavailable(
                f"binance fee schedule needs a signed request: {error}") from error

    get = get or _get_json
    now_ns = now_ns if now_ns is not None else time.time_ns()
    query = urllib.parse.urlencode({
        "symbol": symbol,
        "recvWindow": _RECV_WINDOW_MS,
        "timestamp": now_ns // 1_000_000,
    })
    # Signed over exactly the string that is sent, then appended - not re-encoded.
    signed = f"{query}&signature={sign_query(credentials.api_secret, query)}"
    body = await get(f"{_BINANCE_COMMISSION_URL}?{signed}",
                     {"X-MBX-APIKEY": credentials.api_key})
    return parse_binance_commission(body, now_ns)


async def best_available_schedule(venue: str, instrument_kind: str = "perp",
                                  now_ns: int | None = None) -> FeeSchedule:
    """Fetch if the venue will answer; otherwise the declared rate, still marked
    unverified. The one thing this must never do is return a fetched-looking
    schedule built from a declared number."""
    try:
        if venue == "hyperliquid":
            return await fetch_hyperliquid_schedule(now_ns=now_ns)
        if venue == "binance":
            return await fetch_binance_schedule(now_ns=now_ns)
    except FeeScheduleUnavailable:
        pass

    declared = DECLARED_SCHEDULES.get((venue, instrument_kind))
    if declared is None:
        raise FeeScheduleUnavailable(
            f"no fetched or declared schedule for {venue}/{instrument_kind}")
    return declared
