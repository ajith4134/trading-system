"""Fetching a schedule, and refusing to invent one.

The Hyperliquid fixture is the real `userFees` body captured 2026-08-03 - the
shape of a real response is the thing under test, and the tier ladder in
particular is not something worth guessing at.
"""
from decimal import Decimal

import pytest

from cost.fee_fetcher import (
    FeeScheduleUnavailable, best_available_schedule, fetch_binance_schedule,
    fetch_hyperliquid_schedule, parse_binance_commission, parse_hyperliquid_fees,
    sign_query,
)
from cost.fee_schedule import FeeSource
from cost.secret_store import VenueCredentials

# Captured verbatim from POST https://api.hyperliquid.xyz/info
# {"type":"userFees","user":"0x000...000"} on 2026-08-03T18:00Z, trimmed to the
# fields this parser reads.
REAL_HYPERLIQUID_USER_FEES = {
    "feeSchedule": {
        "cross": "0.00045", "add": "0.00015",
        "spotCross": "0.0007", "spotAdd": "0.0004",
        "tiers": {"vip": [
            {"ntlCutoff": "5000000.0", "cross": "0.0004", "add": "0.00012"},
            {"ntlCutoff": "25000000.0", "cross": "0.00035", "add": "0.00008"},
        ]},
    },
    "userCrossRate": "0.00045", "userAddRate": "0.00015",
    "activeReferralDiscount": "0.0",
}

REAL_BINANCE_COMMISSION = {
    "symbol": "BTCUSDT",
    "makerCommissionRate": "0.000200",
    "takerCommissionRate": "0.000500",
}

NOW_NS = 1_785_780_000_000_000_000


# --------------------------------------------------------------------------
# signing - provable without a key
# --------------------------------------------------------------------------

def test_signing_matches_the_worked_example_in_binances_own_documentation():
    """The one part of a signed request that cannot be checked against the live
    venue without a key. Binance publishes a secret, a query string and the
    signature they produce; if this drifts, every signed request breaks with an
    error that blames the credential rather than the code."""
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    query = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1"
             "&price=0.1&recvWindow=5000&timestamp=1499827319559")
    assert sign_query(secret, query) == (
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")


async def test_the_signature_covers_the_exact_string_that_is_sent():
    """Signing one encoding and transmitting another is the classic way a
    signed request fails: the venue rejects a signature that was correct for a
    string it never received."""
    sent: list[str] = []

    async def get(url, headers):
        sent.append(url)
        return REAL_BINANCE_COMMISSION

    await fetch_binance_schedule(
        credentials=VenueCredentials("key", "secret"), now_ns=NOW_NS, get=get)

    query, _, signature = sent[0].split("?", 1)[1].rpartition("&signature=")
    assert sign_query("secret", query) == signature


async def test_the_api_key_travels_in_the_header_not_the_query():
    captured: dict = {}

    async def get(url, headers):
        captured.update(headers)
        return REAL_BINANCE_COMMISSION

    await fetch_binance_schedule(
        credentials=VenueCredentials("the-key", "secret"), now_ns=NOW_NS, get=get)
    assert captured["X-MBX-APIKEY"] == "the-key"


# --------------------------------------------------------------------------
# parsing real bodies
# --------------------------------------------------------------------------

def test_hyperliquid_fees_parse_to_the_live_measured_rates():
    schedule = parse_hyperliquid_fees(REAL_HYPERLIQUID_USER_FEES, NOW_NS)
    assert schedule.rate.maker_bps == Decimal("1.5")     # `add`, the resting side
    assert schedule.rate.taker_bps == Decimal("4.5")     # `cross`, the crossing side
    assert schedule.source is FeeSource.VENUE_API
    assert schedule.is_verified
    assert schedule.fetched_at_ns == NOW_NS


def test_binance_commission_parses_to_a_verified_schedule():
    schedule = parse_binance_commission(REAL_BINANCE_COMMISSION, NOW_NS)
    assert schedule.rate.maker_bps == Decimal("2.00")
    assert schedule.rate.taker_bps == Decimal("5.00")
    assert schedule.is_verified


@pytest.mark.parametrize("body", [{}, {"feeSchedule": {}},
                                  {"feeSchedule": {"add": "0.00015"}}, {"feeSchedule": []}])
def test_a_malformed_fee_body_raises_rather_than_defaulting(body):
    """The failure that would matter most: a parser that quietly returns zero
    fees makes every strategy look profitable."""
    with pytest.raises(FeeScheduleUnavailable):
        parse_hyperliquid_fees(body, NOW_NS)


def test_a_malformed_commission_body_raises_rather_than_defaulting():
    with pytest.raises(FeeScheduleUnavailable):
        parse_binance_commission({"symbol": "BTCUSDT"}, NOW_NS)


async def test_hyperliquid_is_fetched_anonymously_by_default():
    """The zero address is accepted (measured 2026-08-03), which is what lets a
    schedule be read before any account exists."""
    seen: dict = {}

    async def post(url, payload):
        seen.update(payload)
        return REAL_HYPERLIQUID_USER_FEES

    await fetch_hyperliquid_schedule(now_ns=NOW_NS, post=post)
    assert seen["type"] == "userFees"
    assert seen["user"] == "0x" + "0" * 40


# --------------------------------------------------------------------------
# refusing, rather than inventing
# --------------------------------------------------------------------------

async def test_binance_without_a_credential_refuses_and_says_why(monkeypatch):
    from cost import fee_fetcher
    from cost.secret_store import CredentialsUnavailable

    def no_credentials(venue):
        raise CredentialsUnavailable("still holds the shipped placeholder")

    monkeypatch.setattr(fee_fetcher, "read_venue_credentials", no_credentials)

    with pytest.raises(FeeScheduleUnavailable) as raised:
        await fetch_binance_schedule(now_ns=NOW_NS)
    assert "placeholder" in str(raised.value)


async def test_falling_back_to_declared_keeps_the_unverified_mark(monkeypatch):
    """The single most important property here. A fallback that produced a
    fetched-looking schedule would let a live strategy be gated on a number
    nobody measured, and nothing downstream could tell."""
    from cost import fee_fetcher

    async def unavailable(*args, **kwargs):
        raise FeeScheduleUnavailable("no key")

    monkeypatch.setattr(fee_fetcher, "fetch_binance_schedule", unavailable)

    schedule = await best_available_schedule("binance", "perp", now_ns=NOW_NS)
    assert schedule.source is FeeSource.DECLARED
    assert not schedule.is_verified
    assert schedule.is_stale(now_ns=NOW_NS, max_age_ns=10**18)


async def test_an_unknown_venue_raises_rather_than_returning_a_guess():
    with pytest.raises(FeeScheduleUnavailable):
        await best_available_schedule("kraken", "perp", now_ns=NOW_NS)
