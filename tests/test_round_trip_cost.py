"""The gate every signal passes before it may become an order.

Two properties are load-bearing and everything else here supports them:

  * A refusal is a value, not an exception to swallow. It cannot be coerced to
    a number, so a caller cannot accidentally treat "we do not know" as "zero".
  * A quote built on an unverified input is itself unverified, all the way out.
    Provenance does not get lost in the arithmetic.
"""
from decimal import Decimal

import pytest

from cost.round_trip_cost import (
    CostQuote,
    CostRefused,
    is_signal_viable,
    quote_round_trip_cost,
)

AT_NS = 1_785_648_600_000_000_000


def test_a_binance_spot_taker_round_trip_prices_at_the_documented_20_bps():
    """ARCHITECTURE.md Layer 1 states Binance spot taker round-trip is ~20 bps
    and builds the whole "fees dominate breakeven by 5-10x" argument on it. If
    the engine and the design document ever disagree, one of them is wrong and
    nobody would notice - so the number is pinned here."""
    quote = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="spot")
    assert isinstance(quote, CostQuote)
    assert quote.fee_bps == Decimal("20.0")


def test_a_maker_round_trip_costs_less_than_a_taker_one():
    maker = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="maker", at_ns=AT_NS,
                                  instrument_kind="perp")
    taker = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="perp")
    assert maker.fee_bps < taker.fee_bps


def test_a_quote_built_on_a_declared_fee_is_not_verified():
    """Binance will not serve its schedule without an API key, so its rates are
    a human's reading of a fee page. A quote resting on that must say so, or the
    unverified number silently becomes a trusted one."""
    quote = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="perp")
    assert quote.is_verified is False
    assert any(not source.verified for source in quote.inputs)


def test_every_component_names_its_source():
    """Rule 8 applied to a number: a quote is a measurement with provenance, not
    a figure. A component whose origin cannot be named is not a component."""
    quote = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="perp")
    assert quote.inputs
    for source in quote.inputs:
        assert source.name and source.detail


def test_an_unknown_venue_is_refused_not_priced():
    refusal = quote_round_trip_cost("kraken", "BTCUSD", Decimal("10000"),
                                    order_type="taker", at_ns=AT_NS,
                                    instrument_kind="spot")
    assert isinstance(refusal, CostRefused)
    assert "kraken" in refusal.reason


def test_a_refusal_cannot_be_mistaken_for_a_cost():
    """The failure this guards against is a caller doing `quote.breakeven_bps`
    on a refusal and getting something arithmetic. There is no such attribute,
    and no numeric fallback anywhere to reach for."""
    refusal = quote_round_trip_cost("kraken", "BTCUSD", Decimal("10000"),
                                    order_type="taker", at_ns=AT_NS,
                                    instrument_kind="spot")
    assert not hasattr(refusal, "breakeven_bps")
    with pytest.raises(TypeError):
        float(refusal)


def test_holding_a_position_across_settlements_is_refused_until_funding_exists():
    """Funding is not yet a clock-gated dataset. A carry quote that quietly
    omitted it would understate the cost of exactly the family the prime
    directive rests on."""
    result = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                   order_type="taker", at_ns=AT_NS,
                                   instrument_kind="perp",
                                   holding_ns=24 * 3_600_000_000_000)
    assert isinstance(result, CostRefused)
    assert "funding" in result.reason.lower()


def test_an_intraday_quote_does_not_need_funding():
    """Held across no settlement, so there is nothing to refuse over."""
    result = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                   order_type="taker", at_ns=AT_NS,
                                   instrument_kind="perp",
                                   holding_ns=60_000_000_000)
    assert isinstance(result, CostQuote)


# --- the gate ---------------------------------------------------------------

def test_an_edge_below_breakeven_is_not_viable():
    quote = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="spot")
    assert is_signal_viable(Decimal("5"), quote) is False


def test_an_edge_clearing_breakeven_is_viable():
    quote = quote_round_trip_cost("binance", "BTCUSDT", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="spot")
    assert is_signal_viable(Decimal("50"), quote) is True


def test_a_refusal_is_never_viable_however_large_the_claimed_edge():
    """The one behaviour that makes this a gate rather than a suggestion. An
    unpriceable trade is not a profitable trade, no matter what the strategy
    claims for it."""
    refusal = quote_round_trip_cost("kraken", "BTCUSD", Decimal("10000"),
                                    order_type="taker", at_ns=AT_NS,
                                    instrument_kind="spot")
    assert is_signal_viable(Decimal("100000"), refusal) is False


def test_a_fetched_schedule_makes_the_quote_verified():
    """Hyperliquid publishes its full schedule unauthenticated - it is the one
    venue this system can actually verify. A quote resting on a live fetch must
    say so, or the distinction between verified and declared buys nothing."""
    from cost.fee_fetcher import parse_hyperliquid_fees

    # The real response shape: base perp rates measured live 2026-08-03 at
    # 1.5 bps maker / 4.5 bps taker.
    schedule = parse_hyperliquid_fees(
        {"feeSchedule": {"add": "0.00015", "cross": "0.00045"}}, now_ns=AT_NS)
    quote = quote_round_trip_cost("hyperliquid", "BTC", Decimal("10000"),
                                  order_type="taker", at_ns=AT_NS,
                                  instrument_kind="perp", schedule=schedule)
    assert isinstance(quote, CostQuote)
    assert quote.fee_bps == Decimal("9.0")          # 4.5 bps each leg, measured live
    assert any(s.name == "fee" and s.verified for s in quote.inputs)
