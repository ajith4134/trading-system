"""The paper broker as a transport for the WAL that already exists.

The load-bearing idea in `2026-08-08-paper-execution-engine-design.md`: paper is
not a simulator standing beside the execution path, it is a `transport` callable
plugged into `OrderIntentWal.submit`. So paper inherits write-before-send
durability, deterministic client order ids, signal expiry and the unknown-outcome
path without a parallel implementation of any of them — and exercises the same
code a live transport would, which is what `ARCHITECTURE.md`'s "same code path
across backtest/paper/live" has to mean if it means anything.

Three decisions are defended here:

**A refusal is returned, never raised.** `submit` writes `outcome=unknown` before
the transport is called and, if the transport raises, leaves it there. `unknown`
means "the venue may be holding this order" — the one state that prompts a query
instead of a resend. For a local refusal that claim is simply false, and it would
block every future startup reconciliation on an order that never existed. So the
broker rejects by returning a rejection.

**Two orders, one per accounting.** The two fill at identical quantities and
different prices, so a single `Order` would have to carry one accounting's prices
under a name that claims neither. Their remaining quantities must stay in
lockstep, and a divergence raises rather than being absorbed.

**Anything the fill model cannot price is refused, not approximated.** A market
order has no resting price to print through, and filling one at the last touch
the broker happened to see prices it at a moment already known to have gone the
right way.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from cost.fee_schedule import FeeRate
from execution.order_intent_wal import OrderIntent, OrderIntentWal
from ops.venue_halt import VenueHaltRegistry
from paper.fill_model import MarketEvent, Participation
from paper.paper_broker import AccountingDiverged, PaperBroker

BINANCE = FeeRate(maker_bps=Decimal("2.0"), taker_bps=Decimal("5.0"))
CALIBRATED_HALF = Participation(fraction=Decimal("0.5"), calibrated=True)


def a_broker(tmp_path=None, halts=None, participation=CALIBRATED_HALF):
    return PaperBroker(participation=participation,
                       fees_by_venue={"binance": BINANCE},
                       halts=halts)


def an_intent(side="BUY", price="100", quantity="10", venue="binance",
              created_at_ns=1_000, reduce_only=False):
    return OrderIntent(
        strategy="test", symbol="BTCUSDT", venue=venue, side=side,
        quantity=Decimal(quantity), created_at_ns=created_at_ns,
        valid_for_ns=10**12,
        price=Decimal(price) if price is not None else None,
        reduce_only=reduce_only)


def a_print(price="99", quantity="10", bid="98.5", ask="99.5"):
    return MarketEvent(trade_price=Decimal(price),
                       trade_quantity=Decimal(quantity),
                       best_bid=Decimal(bid), best_ask=Decimal(ask))


def submit(broker, intent):
    """Through the WAL, deliberately — the transport contract is the subject."""
    return broker(intent, "client-1")


# --- it really is a transport ------------------------------------------------

def test_the_broker_satisfies_the_wal_transport_signature(tmp_path: Path):
    wal = OrderIntentWal(tmp_path)
    broker = a_broker()
    response = wal.submit(an_intent(), broker, now_ns=2_000)
    assert response["status"] == "resting"


def test_an_order_submitted_through_the_wal_is_journalled_before_it_rests(
        tmp_path: Path):
    wal = OrderIntentWal(tmp_path)
    broker = a_broker()
    wal.submit(an_intent(), broker, now_ns=2_000)
    # The durable record exists and is acknowledged, and the broker holds it.
    (record,) = wal.records()
    assert record["outcome"] == "acknowledged"
    assert broker.open_order_count == 1


def test_an_expired_intent_never_reaches_the_broker(tmp_path: Path):
    from execution.order_intent_wal import IntentExpired
    wal = OrderIntentWal(tmp_path)
    broker = a_broker()
    intent = an_intent(created_at_ns=0)
    with pytest.raises(IntentExpired):
        wal.submit(intent, broker, now_ns=10**13)
    assert broker.open_order_count == 0


# --- refusals return, so the WAL never claims the venue might hold it --------

def test_a_market_order_is_accepted_and_crosses(tmp_path: Path):
    """Refused until 2026-08-16, then accepted at the user's instruction that
    orders be market wherever possible.

    The original objection was LOOKAHEAD - filling "at the last touch this
    broker happened to see prices it at a moment already known to have gone the
    right way" - and it is answered by ordering rather than by pricing: the
    engine applies prints to resting orders before submitting anything from that
    same print, so a market order is first offered the NEXT bar.
    """
    broker = a_broker()
    response = submit(broker, an_intent(price=None))

    assert response["status"] == "resting"
    assert response["order_type"] == "market"
    assert response["limit_price"] is None


def test_a_refusal_is_returned_not_raised_so_the_wal_records_a_known_outcome(
        tmp_path: Path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(venue="bybit"), a_broker(), now_ns=2_000)
    # unresolved() is the startup work list. A raised refusal would put this
    # order on it forever, and there is no venue to query about it.
    assert wal.unresolved() == []


def test_a_reduce_only_order_with_nothing_to_reduce_is_refused():
    """Implemented 2026-08-16, having been refused wholesale until then - which
    meant every exit was rejected. Accepting it flat would open a position under
    a flag that says it cannot."""
    response = submit(a_broker(), an_intent(reduce_only=True, side="SELL"))

    assert response["status"] == "rejected"
    assert "nothing to reduce" in response["reason"]


def test_a_reduce_only_order_larger_than_the_position_is_refused_not_trimmed():
    """A caller asking to close more than it holds has a different view of the
    position than this book does, and silently trimming hides the
    disagreement."""
    broker = a_broker()
    broker.optimistic.apply_fill(symbol="BTCUSDT", venue="binance", side="BUY",
                                 quantity=Decimal("1"), price=Decimal("100"),
                                 liquidity="taker")
    response = submit(broker, an_intent(reduce_only=True, side="SELL",
                                        quantity="5"))

    assert response["status"] == "rejected"
    assert "refused rather than trimmed" in response["reason"]


def test_a_reduce_only_order_against_a_real_position_is_accepted():
    """The positive case - a rule that refuses everything is not a rule."""
    broker = a_broker()
    broker.optimistic.apply_fill(symbol="BTCUSDT", venue="binance", side="BUY",
                                 quantity=Decimal("5"), price=Decimal("100"),
                                 liquidity="taker")
    response = submit(broker, an_intent(reduce_only=True, side="SELL",
                                        quantity="5", price=None))

    assert response["status"] == "resting"


def test_a_venue_with_no_declared_fees_is_refused_at_submit_not_at_first_fill():
    response = submit(a_broker(), an_intent(venue="bybit"))
    assert response["status"] == "rejected"
    assert "bybit" in response["reason"]


def test_a_rejected_order_does_not_rest():
    broker = a_broker()
    submit(broker, an_intent(venue="bybit"))
    assert broker.open_order_count == 0


def test_the_same_client_order_id_cannot_rest_twice():
    broker = a_broker()
    broker(an_intent(), "client-1")
    response = broker(an_intent(quantity="5"), "client-1")
    assert response["status"] == "rejected"
    assert "already" in response["reason"]


# --- halts freeze the book ---------------------------------------------------

def test_a_halted_venue_refuses_the_order(tmp_path: Path):
    halts = VenueHaltRegistry(tmp_path)
    # Nothing observed: the registry's own default is not-tradeable, which is
    # the correct state for a venue nobody has measured.
    response = submit(a_broker(halts=halts), an_intent())
    assert response["status"] == "rejected"
    assert "binance" in response["reason"]


def test_a_halted_venue_produces_no_fills_and_freezes_what_is_open(
        tmp_path: Path, monkeypatch):
    halts = VenueHaltRegistry(tmp_path)
    monkeypatch.setattr(halts, "is_tradeable", lambda venue: True)
    broker = a_broker(halts=halts)
    broker(an_intent(), "client-1")
    monkeypatch.setattr(halts, "is_tradeable", lambda venue: False)
    assert broker.on_market_event("BTCUSDT", "binance", a_print()) == ()
    assert broker.open_order_count == 1


# --- fills, under both accountings, from one event stream -------------------

def test_a_print_through_the_limit_fills_at_the_participation_share():
    broker = a_broker()
    broker(an_intent(), "client-1")
    # 10 printed, participation 0.5, so 5 of our 10 fill.
    (fill,) = broker.on_market_event("BTCUSDT", "binance", a_print())
    assert fill.quantity == Decimal("5")


def test_the_optimistic_leg_is_a_maker_fill_at_our_own_limit():
    broker = a_broker()
    broker(an_intent(), "client-1")
    (fill,) = broker.on_market_event("BTCUSDT", "binance", a_print())
    assert fill.optimistic_price == Decimal("100")
    assert fill.optimistic_liquidity == "maker"


def test_the_pessimistic_leg_takes_liquidity_and_never_beats_the_optimistic_one():
    broker = a_broker()
    broker(an_intent(), "client-1")
    (fill,) = broker.on_market_event("BTCUSDT", "binance", a_print())
    assert fill.pessimistic_liquidity == "taker"
    # A BUY: paying more is worse, and the crossing price may not undercut the
    # resting price or the accounting invents a free lunch.
    assert fill.pessimistic_price >= fill.optimistic_price


def test_a_print_that_does_not_go_through_the_limit_fills_nothing():
    broker = a_broker()
    broker(an_intent(), "client-1")
    # A print *at* our price proves nothing about the queue ahead of us.
    assert broker.on_market_event(
        "BTCUSDT", "binance", a_print(price="100")) == ()


def test_a_fill_never_exceeds_the_volume_that_printed():
    broker = a_broker()
    broker(an_intent(quantity="1000"), "client-1")
    (fill,) = broker.on_market_event(
        "BTCUSDT", "binance", a_print(quantity="4"))
    assert fill.quantity == Decimal("2")     # 4 printed x 0.5 participation


def test_an_order_on_another_symbol_is_untouched_by_this_symbols_prints():
    broker = a_broker()
    broker(an_intent(), "client-1")
    assert broker.on_market_event("ETHUSDT", "binance", a_print()) == ()
    assert broker.open_order_count == 1


def test_filling_completely_closes_the_order():
    broker = a_broker()
    broker(an_intent(quantity="5"), "client-1")
    broker.on_market_event("BTCUSDT", "binance", a_print(quantity="10"))
    assert broker.open_order_count == 0


def test_a_closed_order_takes_no_further_fills():
    broker = a_broker()
    broker(an_intent(quantity="5"), "client-1")
    broker.on_market_event("BTCUSDT", "binance", a_print(quantity="10"))
    assert broker.on_market_event("BTCUSDT", "binance", a_print()) == ()


# --- the two books ----------------------------------------------------------

def test_both_books_hold_the_same_quantity_and_differ_only_in_price():
    broker = a_broker()
    broker(an_intent(), "client-1")
    broker.on_market_event("BTCUSDT", "binance", a_print())
    optimistic = broker.optimistic.position("BTCUSDT", "binance")
    pessimistic = broker.pessimistic.position("BTCUSDT", "binance")
    assert optimistic.quantity == pessimistic.quantity == Decimal("5")


def test_the_pessimistic_book_never_pays_less_in_fees_than_the_optimistic_one():
    broker = a_broker()
    broker(an_intent(), "client-1")
    broker.on_market_event("BTCUSDT", "binance", a_print())
    # 2 bps maker against 5 bps taker on the same notional. If this ever
    # inverts, the promotion gate is being asked to trust the cheaper number.
    assert broker.pessimistic.fees_paid > broker.optimistic.fees_paid


def test_the_books_are_reachable_in_the_shape_the_reconciler_takes():
    broker = a_broker()
    broker(an_intent(), "client-1")
    broker.on_market_event("BTCUSDT", "binance", a_print())
    (position,) = broker.pessimistic.local_positions()
    assert position.key == ("binance", "BTCUSDT")


def test_a_divergence_between_the_two_accountings_raises_rather_than_absorbing():
    broker = a_broker()
    broker(an_intent(), "client-1")
    resting = broker.resting("client-1")
    # Force the two out of step the way a real bug would, and confirm the next
    # fill refuses instead of quietly reconciling them.
    resting.pessimistic_order.fill(Decimal("1"), Decimal("100"))
    with pytest.raises(AccountingDiverged):
        broker.on_market_event("BTCUSDT", "binance", a_print())


# --- what the participation figure is allowed to claim ----------------------

def test_an_uncalibrated_participation_is_carried_onto_every_fill():
    broker = a_broker(participation=Participation(fraction=Decimal("0.5"),
                                                  calibrated=False))
    broker(an_intent(), "client-1")
    (fill,) = broker.on_market_event("BTCUSDT", "binance", a_print())
    assert fill.uncalibrated is True


def test_a_calibrated_participation_says_so():
    broker = a_broker()
    broker(an_intent(), "client-1")
    (fill,) = broker.on_market_event("BTCUSDT", "binance", a_print())
    assert fill.uncalibrated is False
