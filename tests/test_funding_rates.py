"""Turning archived premiumIndex polls into a clock-gated funding dataset.

Funding is the carry the prime directive rests on, and it was the one cost the
engine had to refuse for want of a dataset. Two things have to be right:

  * A rate is knowable when we received it, never when the venue stamped it.
    Availability keyed on the venue's clock would let a backtest read a rate
    before it arrived - the exact leakage Layer 1 exists to make impossible.
  * Each settlement gets its own archived rate, not the latest one repeated.
"""
import json
from decimal import Decimal

import pytest

from store.funding_rates import (
    FundingObservation,
    build_funding_frame,
    extract_funding,
    rates_at_settlements,
)
from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    SYMBOL,
    VENUE,
    validate_temporal_frame,
)


class FakeEntry:
    """Stands in for the archive index entry, which carries receipt time."""

    def __init__(self, t_recv_ns: int) -> None:
        self.t_recv_ns = t_recv_ns


# A real frame, captured 2026-08-08 from the production archive.
REAL_PREMIUM_INDEX = json.dumps({
    "symbol": "BTCUSDT", "markPrice": "64971.30000000",
    "indexPrice": "64980.11000000", "estimatedSettlePrice": "64975.00000000",
    "lastFundingRate": "0.00006847", "interestRate": "0.00010000",
    "nextFundingTime": 1786204800000, "time": 1786179599006,
})


def test_a_poll_becomes_one_observation():
    got = extract_funding(REAL_PREMIUM_INDEX, FakeEntry(1_786_179_599_500_000_000),
                          venue="binance", symbol="BTCUSDT")
    assert len(got) == 1
    assert got[0].symbol == "BTCUSDT"
    assert got[0].funding_rate == Decimal("0.00006847")


def test_the_rate_keeps_full_decimal_precision():
    """Parsed straight from the string. A float here puts a representation
    error into the number that prices every carry trade."""
    got = extract_funding(REAL_PREMIUM_INDEX, FakeEntry(1), venue="binance",
                          symbol="BTCUSDT")
    assert isinstance(got[0].funding_rate, Decimal)
    assert str(got[0].funding_rate) == "0.00006847"


def test_mark_and_index_are_both_kept():
    """FEATURES.md marks mark-vs-index-vs-oracle [MISSED]: Hyperliquid funds on
    oracle, Binance on mark, and the difference is the whole basis trade.
    Dropping one here makes that unrecoverable."""
    got = extract_funding(REAL_PREMIUM_INDEX, FakeEntry(1), venue="binance",
                          symbol="BTCUSDT")[0]
    assert got.mark_price == Decimal("64971.30000000")
    assert got.index_price == Decimal("64980.11000000")


def test_a_frame_that_is_not_a_premium_index_poll_yields_nothing():
    """The archive holds several streams. Misreading a trade as a funding rate
    would quietly poison the carry cost."""
    trade = json.dumps({"data": {"e": "trade", "s": "BTCUSDT", "p": "1", "q": "1"}})
    assert extract_funding(trade, FakeEntry(1), venue="binance", symbol="BTCUSDT") == []


def test_a_rate_is_available_when_we_received_it_not_when_the_venue_stamped_it():
    """The leakage rule. The venue's `time` is 494ms before our receipt here; a
    backtest keyed on the venue clock would read the rate before it arrived."""
    venue_stamped_ns = 1_786_179_599_006_000_000
    received_ns = 1_786_179_599_500_000_000
    frame = build_funding_frame(extract_funding(
        REAL_PREMIUM_INDEX, FakeEntry(received_ns), venue="binance", symbol="BTCUSDT"))
    row = frame.iloc[0]
    assert row[EVENT_TIME] == venue_stamped_ns
    assert row[INGESTION_TIME] == received_ns
    assert row[AVAILABILITY_TIME] >= received_ns


def test_the_frame_satisfies_the_bitemporal_contract():
    frame = build_funding_frame(extract_funding(
        REAL_PREMIUM_INDEX, FakeEntry(1_786_179_599_500_000_000),
        venue="binance", symbol="BTCUSDT"))
    validate_temporal_frame(frame)
    for column in (SYMBOL, VENUE, EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME):
        assert column in frame.columns


def test_no_observations_is_an_empty_frame_not_a_zero_rate():
    """Charging zero funding to a carry strategy is not a conservative default,
    it is the optimistic one."""
    assert build_funding_frame([]).empty


# --- picking the rate that applied at each settlement ------------------------

def obs(rate: str, at_ns: int) -> FundingObservation:
    return FundingObservation(
        symbol="BTCUSDT", venue="binance", funding_rate=Decimal(rate),
        mark_price=Decimal("1"), index_price=Decimal("1"),
        next_funding_time_ns=0, event_time_ns=at_ns, ingestion_time_ns=at_ns)


def test_each_settlement_takes_the_last_rate_known_before_it():
    """Not the newest rate overall. A carry priced on a rate that had not yet
    been published at the settlement it is charged against is look-ahead."""
    observations = [obs("0.0001", 100), obs("0.0002", 200), obs("0.0003", 400)]
    got = rates_at_settlements(observations, settlements_ns=[150, 300])
    assert got == [Decimal("0.0001"), Decimal("0.0002")]


def test_a_settlement_with_no_prior_observation_refuses():
    """Returning zero, or borrowing a later rate, would price a carry against
    a number that did not exist yet."""
    with pytest.raises(LookupError):
        rates_at_settlements([obs("0.0001", 500)], settlements_ns=[100])
