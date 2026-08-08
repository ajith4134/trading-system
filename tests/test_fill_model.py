"""The paper fill model, scored under both accountings.

`ARCHITECTURE.md:92` — *"paper mode manufactures edge"*. This module is the place
that would do it. Every test here exists to make one optimistic assumption
impossible to hold silently:

- a fill needs a print that actually went through the resting price
- a fill can never exceed the volume that printed
- a fill can never exceed the share of that volume the queue would plausibly give
- and the pessimistic accounting can never come out ahead of the optimistic one

The last is the load-bearing invariant, tested as a property over generated event
sequences: if pessimistic P&L can beat optimistic P&L for any strategy on any
sequence, then "optimistic" is conservative somewhere, and gating promotion on the
pessimistic number proves nothing.

Design: `docs/superpowers/specs/2026-08-08-paper-execution-engine-design.md`.
"""
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from paper.fill_model import (
    MarketEvent,
    Participation,
    PaperFill,
    RestingOrder,
    signed_cash,
    simulate_fills,
)


def a_resting_buy(limit="100.00", remaining="10"):
    return RestingOrder(side="BUY", limit_price=Decimal(limit),
                        remaining=Decimal(remaining))


def a_print(price, quantity, bid="99.99", ask="100.01"):
    return MarketEvent(trade_price=Decimal(price), trade_quantity=Decimal(quantity),
                       best_bid=Decimal(bid), best_ask=Decimal(ask))


def all_of_the_queue():
    """Participation 1.0: everything that printed through is ours.

    Not a realistic calibration - it is the setting that isolates the other rules
    from the queue-share rule, so a test about volume capping fails for the reason
    it names.
    """
    return Participation(fraction=Decimal("1"), calibrated=True)


# --- a fill needs a print that went through ---------------------------------

def test_a_resting_buy_does_not_fill_when_the_market_never_trades_through_it():
    """A trade printing at or above a resting bid did not take that bid.

    Filling here is the assumption that manufactures edge: it awards a maker fill
    for a price the market never actually reached down to.
    """
    order = a_resting_buy(limit="100.00")
    events = (a_print(price="100.01", quantity="40"),)

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert outcome.optimistic == ()
    assert outcome.pessimistic == ()


def test_a_print_through_fills_the_resting_buy_as_a_maker_at_its_own_price():
    """The trade went below our bid, so the book had to come through us.

    Price is our limit, not the print: a resting order that fills is filled at the
    price it rested at. Paying the print price would be a better fill than the
    order asked for, which is edge from nowhere.
    """
    order = a_resting_buy(limit="100.00", remaining="10")
    events = (a_print(price="99.98", quantity="40"),)

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert outcome.optimistic == (
        PaperFill(quantity=Decimal("10"), price=Decimal("100.00"),
                  liquidity="maker"),
    )


# --- never more than the volume that printed --------------------------------

def test_a_fill_never_exceeds_the_volume_that_printed_through():
    """One lot printing through a hundred-lot order is not a hundred-lot fill.

    This is the flaw that makes plain trade-through fills optimistic: it awards the
    whole resting size on evidence of one contract's worth of trading.
    """
    order = a_resting_buy(limit="100.00", remaining="100")
    events = (a_print(price="99.98", quantity="40"),)

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert outcome.optimistic[0].quantity == Decimal("40")


# --- never more than the queue would have given us ---------------------------

def test_a_fill_takes_only_the_calibrated_share_of_the_printed_volume():
    """Others are in the queue ahead of us, and the data cannot say how many.

    Taking all of the printed volume assumes queue priority the archive never
    shows. The share is measured from the depth archive and carried here as a
    number with provenance, rather than invented at the point of use.
    """
    order = a_resting_buy(limit="100.00", remaining="100")
    events = (a_print(price="99.98", quantity="40"),)
    quarter_of_the_queue = Participation(fraction=Decimal("0.25"), calibrated=True)

    outcome = simulate_fills(order, events, participation=quarter_of_the_queue)

    assert outcome.optimistic[0].quantity == Decimal("10")


def test_fills_stop_once_the_order_is_exhausted():
    """Remaining has to come down as fills land, or every print refills the order.

    The failure this catches is a paper book that trades the same size repeatedly
    and reports the P&L of a position many times larger than was ever intended.
    """
    order = a_resting_buy(limit="100.00", remaining="10")
    events = (a_print(price="99.98", quantity="6"),
              a_print(price="99.98", quantity="6"),
              a_print(price="99.98", quantity="6"))

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert [f.quantity for f in outcome.optimistic] == [Decimal("6"), Decimal("4")]


# --- the same events, scored the other way -----------------------------------

def test_the_pessimistic_accounting_crosses_the_spread_as_a_taker():
    """The same fill, priced as though the passive assumption were wrong.

    Same quantity on purpose: the two accountings differ in what a fill *cost*,
    not in whether it happened. Letting them differ in schedule would make the gap
    between them a mixture of two effects and measure neither.
    """
    order = a_resting_buy(limit="100.00", remaining="10")
    events = (a_print(price="99.98", quantity="40", bid="99.97", ask="100.05"),)

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert outcome.pessimistic == (
        PaperFill(quantity=Decimal("10"), price=Decimal("100.05"),
                  liquidity="taker"),
    )


def test_the_pessimistic_price_is_never_better_than_the_order_asked_for():
    """A print through usually leaves the touch *inside* our limit.

    Taking at that touch would buy cheaper than the resting order ever asked to —
    the pessimistic accounting coming out ahead by construction, which is a free
    lunch conjured by the accounting rather than earned by the strategy. Floored
    at the limit: if crossing were genuinely better, the order should have crossed.
    """
    order = a_resting_buy(limit="100.00", remaining="10")
    events = (a_print(price="99.98", quantity="40", bid="99.97", ask="99.99"),)

    outcome = simulate_fills(order, events, participation=all_of_the_queue())

    assert outcome.pessimistic[0].price == Decimal("100.00")


# --- the other side, which is not free ---------------------------------------

def a_resting_sell(limit="100.00", remaining="10"):
    return RestingOrder(side="SELL", limit_price=Decimal(limit),
                        remaining=Decimal(remaining))


def test_a_resting_sell_fills_only_when_a_trade_prints_above_it():
    """Mirrored, and tested rather than assumed.

    A sell is taken when the market trades *up* through the offer. Reusing the
    buy's comparison would fill every sell on every downtick — the direction of
    the inequality is the whole rule.
    """
    order = a_resting_sell(limit="100.00", remaining="10")
    above = a_print(price="100.02", quantity="40", bid="99.95", ask="100.03")
    below = a_print(price="99.98", quantity="40", bid="99.95", ask="100.03")

    assert simulate_fills(order, (below,),
                          participation=all_of_the_queue()).optimistic == ()
    assert simulate_fills(order, (above,),
                          participation=all_of_the_queue()).optimistic == (
        PaperFill(quantity=Decimal("10"), price=Decimal("100.00"),
                  liquidity="maker"),
    )


def test_a_pessimistic_sell_hits_the_bid_and_is_never_better_than_the_offer():
    """Selling as a taker means hitting the bid, which is below the offer.

    Floored the other way for the same reason as the buy: if the bid were above
    our offer, crossing would have been the better trade and the order would not
    have rested.
    """
    order = a_resting_sell(limit="100.00", remaining="10")
    worse = a_print(price="100.02", quantity="40", bid="99.90", ask="100.03")
    better = a_print(price="100.02", quantity="40", bid="100.04", ask="100.05")

    assert simulate_fills(order, (worse,),
                          participation=all_of_the_queue()).pessimistic[0] == (
        PaperFill(quantity=Decimal("10"), price=Decimal("99.90"),
                  liquidity="taker"))
    assert simulate_fills(order, (better,),
                          participation=all_of_the_queue()
                          ).pessimistic[0].price == Decimal("100.00")


# --- an assumed number must not read as a measured one -----------------------

def test_the_outcome_carries_the_participation_it_used_and_whether_it_was_measured():
    """A defaulted share and a calibrated share are different claims.

    Without the flag travelling with the result, a conservative default applied
    because no depth existed is indistinguishable downstream from a fraction
    measured off the archive — and tier-2 promotion turns on exactly that
    difference.
    """
    order = a_resting_buy()
    events = (a_print(price="99.98", quantity="40"),)
    guessed = Participation(fraction=Decimal("0.1"), calibrated=False)

    outcome = simulate_fills(order, events, participation=guessed)

    assert outcome.participation == Decimal("0.1")
    assert outcome.uncalibrated is True


# --- what a set of fills was worth, fees included ----------------------------

def test_buying_costs_the_notional_plus_the_fee_for_that_liquidity():
    """Signed cash, so both sides compare with one inequality.

    Fees are charged at the rate for the liquidity each fill actually took —
    charging one blended rate is how a taker fill gets quietly priced as a maker.
    """
    fills = (PaperFill(quantity=Decimal("10"), price=Decimal("100.00"),
                       liquidity="maker"),)

    cash = signed_cash(fills, side="BUY", maker_bps=Decimal("2"),
                       taker_bps=Decimal("5"))

    assert cash == Decimal("-1000.2000")


# --- the load-bearing invariant ----------------------------------------------

def _prices(low=1_00, high=1000_00):
    """Two-decimal prices as exact Decimals, never floats."""
    return st.integers(min_value=low, max_value=high).map(
        lambda cents: Decimal(cents) / Decimal("100"))


def _quantities():
    return st.integers(min_value=1, max_value=100_000).map(
        lambda units: Decimal(units) / Decimal("100"))


@st.composite
def _market_events(draw):
    """An uncrossed book with a print, which is the only kind that ever exists."""
    bid = draw(_prices())
    spread = draw(st.integers(min_value=0, max_value=500).map(
        lambda cents: Decimal(cents) / Decimal("100")))
    return MarketEvent(trade_price=draw(_prices()), trade_quantity=draw(_quantities()),
                       best_bid=bid, best_ask=bid + spread)


@given(side=st.sampled_from(["BUY", "SELL"]),
       limit_price=_prices(),
       remaining=_quantities(),
       events=st.lists(_market_events(), max_size=8).map(tuple),
       participation_pct=st.integers(min_value=1, max_value=100),
       maker_bps=st.integers(min_value=0, max_value=10),
       extra_taker_bps=st.integers(min_value=0, max_value=10))
@settings(max_examples=400, deadline=None)
def test_the_pessimistic_accounting_never_beats_the_optimistic_one(
        side, limit_price, remaining, events, participation_pct, maker_bps,
        extra_taker_bps):
    """The claim the whole promotion argument rests on.

    Promotion gates on the pessimistic number. If pessimistic cash can exceed
    optimistic cash for *any* order on *any* event sequence, then "optimistic" is
    the conservative one somewhere, and passing the pessimistic gate stops meaning
    what it is read to mean.

    Stated on signed cash so one inequality covers both sides: for a buy, worse is
    paying more; for a sell, worse is receiving less; both are smaller numbers.
    """
    order = RestingOrder(side=side, limit_price=limit_price, remaining=remaining)
    participation = Participation(
        fraction=Decimal(participation_pct) / Decimal("100"), calibrated=True)
    fees = {"maker_bps": Decimal(maker_bps),
            "taker_bps": Decimal(maker_bps + extra_taker_bps)}

    outcome = simulate_fills(order, events, participation=participation)

    assert (signed_cash(outcome.pessimistic, side=side, **fees)
            <= signed_cash(outcome.optimistic, side=side, **fees))
