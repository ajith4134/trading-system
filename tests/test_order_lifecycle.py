"""Order lifecycle by remaining quantity — EX-002, ported from real prior art.

`FEATURES.md` §5: partial-fill tracking by remaining quantity, **never a binary
filled flag**. The ledger records the bug class this prevents: `if status ==
"FILLED"` mishandles `PARTIALLY_FILLED` and double-counts.

**This one is a genuine port.** `nse-crypto-bot-final/trading/execution/
order_state.py` (ledger EX-049) was fetched raw like the others and is the first
donor in this audit that holds up: the transition guards are real, over-fill is
caught, the average price is size-weighted, and `remaining` is a property rather
than a flag. Credit where it is due - most of this file's structure is theirs.

Two changes, both deliberate:

**`Decimal`, not `float` with an epsilon.** The donor compares against `EPS = 1e-9`.
Exchange quantities have defined step sizes, and `Decimal` makes "is this order
complete" exact instead of nearly exact. `CLAUDE.md` already requires Decimal for
fees; the same argument applies to the quantity that decides whether an order is
done.

**A rejection is not a fill.** The donor does `self.fills.append({"rejected":
reason})`, putting a record with no `qty` into the list everything else sums over.
Rejections get their own field.
"""
from decimal import Decimal

import pytest

from execution.order_lifecycle import (
    InvalidTransition,
    Order,
    OrderStatus,
)


def an_order(quantity="10", side="BUY"):
    return Order(client_order_id="c-1", symbol="BTCUSDT", venue="binance",
                 side=side, quantity=Decimal(quantity))


# --- remaining quantity, never a flag ---------------------------------------

def test_a_new_order_is_pending_with_everything_remaining():
    order = an_order()
    assert order.status is OrderStatus.PENDING
    assert order.remaining == Decimal("10")
    assert order.filled_quantity == Decimal("0")


def test_a_partial_fill_leaves_the_order_open_with_the_rest_remaining():
    """The bug this exists to prevent: treating a partially filled order as done
    abandons the unfilled remainder while believing the position is on."""
    order = an_order()
    order.fill(Decimal("3"), Decimal("50000"))
    assert order.status is OrderStatus.PARTIAL
    assert order.remaining == Decimal("7")
    assert order.is_open


def test_filling_the_remainder_completes_the_order():
    order = an_order()
    order.fill(Decimal("3"), Decimal("50000"))
    order.fill(Decimal("7"), Decimal("50100"))
    assert order.status is OrderStatus.FILLED
    assert order.remaining == Decimal("0")
    assert not order.is_open


def test_decimal_fills_that_sum_exactly_complete_the_order():
    """The case that matters, and the one float gets wrong: 0.1 + 0.2 + 0.7 is
    exactly 1 in Decimal and is not in binary float, where the donor needed
    `qty > remaining + 1e-9` to paper over it. Exchange step sizes are decimal
    quantities, so this is the normal path, not a corner."""
    order = an_order(quantity="1")
    for part in ("0.1", "0.2", "0.7"):
        order.fill(Decimal(part), Decimal("100"))
    assert order.remaining == Decimal("0")
    assert order.status is OrderStatus.FILLED


def test_a_quantity_step_absorbs_sub_precision_residue():
    """Decimal is exact for representable values, but `Decimal(1)/Decimal(3)` is
    not one - it carries 28 digits, so three of them leave 1E-28 remaining and the
    order would sit PARTIAL forever. The answer is not an epsilon: it is instrument
    precision, which RX-005 already requires ("qty to instrument precision"). Below
    one step there is no order an exchange could accept, so it is complete."""
    order = Order(client_order_id="c-1", symbol="BTCUSDT", venue="binance",
                  side="BUY", quantity=Decimal("1"),
                  quantity_step=Decimal("0.001"))
    for _ in range(3):
        order.fill(Decimal("1") / Decimal("3"), Decimal("100"))
    assert order.remaining == Decimal("0")
    assert order.status is OrderStatus.FILLED


def test_without_a_quantity_step_residue_is_reported_not_hidden():
    """The limitation, stated by test rather than left to be discovered. With no
    declared step the order stays PARTIAL on sub-precision residue - visible and
    wrong-in-the-safe-direction, rather than silently rounded to complete."""
    order = an_order(quantity="1")
    for _ in range(3):
        order.fill(Decimal("1") / Decimal("3"), Decimal("100"))
    assert order.remaining > Decimal("0")
    assert order.status is OrderStatus.PARTIAL


def test_the_average_price_is_size_weighted():
    order = an_order()
    order.fill(Decimal("2"), Decimal("100"))
    order.fill(Decimal("8"), Decimal("200"))
    assert order.average_fill_price == Decimal("180")


def test_the_average_price_of_an_unfilled_order_is_none_not_zero():
    """Zero is a price. An unfilled order has no average price, and reporting 0.0
    puts a number into P&L arithmetic that means "no data"."""
    assert an_order().average_fill_price is None


# --- illegal transitions raise ----------------------------------------------

def test_over_filling_is_refused():
    order = an_order()
    order.fill(Decimal("9"), Decimal("100"))
    with pytest.raises(InvalidTransition):
        order.fill(Decimal("2"), Decimal("100"))


def test_filling_a_cancelled_order_is_refused():
    """A late fill arriving after cancellation is exactly the race this guards.
    Applying it would create a position nothing believes it holds."""
    order = an_order()
    order.cancel()
    with pytest.raises(InvalidTransition):
        order.fill(Decimal("1"), Decimal("100"))


def test_filling_a_filled_order_is_refused():
    order = an_order()
    order.fill(Decimal("10"), Decimal("100"))
    with pytest.raises(InvalidTransition):
        order.fill(Decimal("1"), Decimal("100"))


def test_cancelling_a_terminal_order_is_refused():
    order = an_order()
    order.fill(Decimal("10"), Decimal("100"))
    with pytest.raises(InvalidTransition):
        order.cancel()


def test_a_partially_filled_order_can_still_be_cancelled():
    """Cancelling keeps the fills that already happened. The position is real."""
    order = an_order()
    order.fill(Decimal("4"), Decimal("100"))
    order.cancel()
    assert order.status is OrderStatus.CANCELLED
    assert order.filled_quantity == Decimal("4")


def test_only_a_pending_order_can_be_rejected():
    order = an_order()
    order.fill(Decimal("1"), Decimal("100"))
    with pytest.raises(InvalidTransition):
        order.reject("too late")


def test_a_zero_or_negative_fill_is_refused():
    for bad in (Decimal("0"), Decimal("-1")):
        with pytest.raises(ValueError):
            an_order().fill(bad, Decimal("100"))


def test_a_non_positive_order_quantity_is_refused():
    with pytest.raises(ValueError):
        Order(client_order_id="c", symbol="S", venue="v", side="BUY",
              quantity=Decimal("0"))


def test_an_unknown_side_is_refused():
    with pytest.raises(ValueError):
        Order(client_order_id="c", symbol="S", venue="v", side="SIDEWAYS",
              quantity=Decimal("1"))


# --- a rejection is not a fill ----------------------------------------------

def test_a_rejection_is_recorded_separately_from_the_fills():
    """The donor appended `{"rejected": reason}` to `fills`, putting a record with
    no `qty` into the list everything else sums over. Anything totalling filled
    quantity from that list hits a KeyError or silently skips."""
    order = an_order()
    order.reject("insufficient margin")
    assert order.status is OrderStatus.REJECTED
    assert order.fills == ()
    assert order.rejection_reason == "insufficient margin"
    assert order.filled_quantity == Decimal("0")


def test_every_fill_is_retained_in_order():
    order = an_order()
    order.fill(Decimal("2"), Decimal("100"))
    order.fill(Decimal("3"), Decimal("110"))
    assert [f.quantity for f in order.fills] == [Decimal("2"), Decimal("3")]
    assert [f.price for f in order.fills] == [Decimal("100"), Decimal("110")]


# --- serialisation, for the WAL ---------------------------------------------

def test_an_order_round_trips_through_its_serialised_form():
    """The WAL replays orders from disk, so the serialised form has to reconstruct
    the exact state - including partial fills. A round trip that loses the fills
    rebuilds an order that looks pending and gets sent twice."""
    order = an_order()
    order.fill(Decimal("3"), Decimal("50000.5"))

    restored = Order.from_dict(order.to_dict())
    assert restored.status is OrderStatus.PARTIAL
    assert restored.filled_quantity == Decimal("3")
    assert restored.remaining == Decimal("7")
    assert restored.average_fill_price == Decimal("50000.5")
    assert restored.client_order_id == order.client_order_id


def test_the_serialised_form_carries_decimals_as_strings():
    """Floats in JSON lose precision, and the WAL is the record used to decide
    whether an order still needs sending."""
    order = an_order()
    order.fill(Decimal("1") / Decimal("3"), Decimal("100"))
    payload = order.to_dict()
    assert isinstance(payload["filled_quantity"], str)
    assert isinstance(payload["quantity"], str)
