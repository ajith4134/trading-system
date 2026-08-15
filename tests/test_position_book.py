"""The local position model — what we believe we hold, and what it cost.

This is source 2 of the three `execution.state_recovery` reconciles: the WAL says
what we intended, the venue says what is real, and this says what we believe. The
whole point of a separate believer is that it can be *wrong*, and the reconciler
catches it — so the contract it hands over matters more than its internals.

Two decisions are defended here rather than assumed, because both are the kind
that read as harmless and are not:

**A flat position is not reported.** `recover_and_reconcile` treats a local
position with no venue counterpart as an unmanaged position — "we believe we hold
X and the venue reports nothing". A book that emitted every symbol it had ever
touched, at quantity zero, would manufacture one of those mismatches per closed
trade and block every start. Flat is absence, not a zero-quantity row.

**One book per accounting, never one book with two numbers.** The paper engine
scores optimistic and pessimistic fills separately and gates promotion on the
pessimistic one. A book holding both invites an average of them, and an average
of the two is exactly the manufactured edge `ARCHITECTURE.md:92` names.
"""
from decimal import Decimal

import pytest

from cost.fee_schedule import FeeRate
from execution.state_recovery import VenuePosition
from paper.position_book import (
    Accounting,
    PositionBook,
    UndeclaredFeeSchedule,
)

BINANCE = FeeRate(maker_bps=Decimal("2.0"), taker_bps=Decimal("5.0"))
HYPERLIQUID = FeeRate(maker_bps=Decimal("1.5"), taker_bps=Decimal("4.5"))


def a_book(accounting=Accounting.PESSIMISTIC):
    return PositionBook(accounting=accounting,
                        fees_by_venue={"binance": BINANCE,
                                       "hyperliquid": HYPERLIQUID})


def buy(book, quantity, price, symbol="BTCUSDT", venue="binance",
        liquidity="taker"):
    return book.apply_fill(symbol=symbol, venue=venue, side="BUY",
                           quantity=Decimal(quantity), price=Decimal(price),
                           liquidity=liquidity)


def sell(book, quantity, price, symbol="BTCUSDT", venue="binance",
         liquidity="taker"):
    return book.apply_fill(symbol=symbol, venue=venue, side="SELL",
                           quantity=Decimal(quantity), price=Decimal(price),
                           liquidity=liquidity)


# --- an empty book asserts nothing ------------------------------------------

def test_a_new_book_holds_no_positions():
    assert a_book().local_positions() == []


def test_a_new_book_has_realized_nothing():
    assert a_book().realized_pnl_after_fees == Decimal("0")


# --- opening, averaging, and the sign convention ----------------------------

def test_a_buy_opens_a_long_at_the_fill_price():
    book = a_book()
    buy(book, "2", "100")
    position = book.position("BTCUSDT", "binance")
    assert position.quantity == Decimal("2")
    assert position.average_entry_price == Decimal("100")


def test_a_sell_opens_a_short_carried_as_a_negative_quantity():
    book = a_book()
    sell(book, "2", "100")
    assert book.position("BTCUSDT", "binance").quantity == Decimal("-2")


def test_adding_to_a_long_averages_the_entry_by_size():
    book = a_book()
    buy(book, "1", "100")
    buy(book, "3", "200")
    # (1*100 + 3*200) / 4 = 175, and it is exact in Decimal.
    assert book.position("BTCUSDT", "binance").average_entry_price == Decimal("175")


def test_the_average_entry_is_exact_where_binary_float_would_drift():
    book = a_book()
    buy(book, "0.1", "100")
    buy(book, "0.2", "100")
    assert book.position("BTCUSDT", "binance").quantity == Decimal("0.3")


# --- reducing keeps the entry; it does not re-average -----------------------

def test_reducing_a_long_leaves_the_average_entry_untouched():
    book = a_book()
    buy(book, "4", "100")
    sell(book, "1", "150")
    position = book.position("BTCUSDT", "binance")
    assert position.quantity == Decimal("3")
    assert position.average_entry_price == Decimal("100")


def test_reducing_realizes_the_gain_on_the_closed_part_only():
    book = a_book()
    buy(book, "4", "100", liquidity="maker")
    sell(book, "1", "150", liquidity="maker")
    # Gross gain 1 * (150 - 100) = 50. Fees at 2 bps maker on both legs:
    #   entry 4 * 100 * 0.0002 = 0.08, exit 1 * 150 * 0.0002 = 0.03.
    # Entry fees are paid when paid, not amortised over the closed fraction.
    assert book.realized_pnl_after_fees == Decimal("50") - Decimal("0.08") - Decimal("0.03")


def test_fees_are_charged_at_the_rate_for_the_liquidity_the_fill_took():
    maker_book, taker_book = a_book(), a_book()
    buy(maker_book, "1", "100", liquidity="maker")
    buy(taker_book, "1", "100", liquidity="taker")
    # 2 bps against 5 bps on the same notional; a blended rate would hide this.
    assert maker_book.fees_paid == Decimal("0.02")
    assert taker_book.fees_paid == Decimal("0.05")


# --- closing flat, and why flat disappears ----------------------------------

def test_closing_a_long_completely_leaves_no_position_to_report():
    book = a_book()
    buy(book, "2", "100")
    sell(book, "2", "100")
    assert book.local_positions() == []


def test_a_flat_position_has_no_average_entry_price_because_zero_is_a_price():
    book = a_book()
    buy(book, "2", "100")
    sell(book, "2", "100")
    assert book.position("BTCUSDT", "binance").average_entry_price is None


def test_a_closed_round_trip_still_reports_what_it_realized():
    book = a_book()
    buy(book, "2", "100", liquidity="maker")
    sell(book, "2", "110", liquidity="maker")
    # 20 gross, less 2*100*0.0002 = 0.04 and 2*110*0.0002 = 0.044.
    assert book.realized_pnl_after_fees == Decimal("20") - Decimal("0.084")


# --- flipping through zero --------------------------------------------------

def test_selling_through_a_long_flips_it_short():
    book = a_book()
    buy(book, "2", "100")
    sell(book, "5", "120")
    assert book.position("BTCUSDT", "binance").quantity == Decimal("-3")


def test_a_flip_prices_the_new_leg_at_the_flipping_fill_alone():
    book = a_book()
    buy(book, "2", "100")
    sell(book, "5", "120")
    # The remaining 3 short were opened at 120. Carrying any part of the old
    # long's 100 into the short's entry would misstate every later P&L.
    assert book.position("BTCUSDT", "binance").average_entry_price == Decimal("120")


def test_a_flip_realizes_only_the_part_that_actually_closed():
    book = a_book()
    buy(book, "2", "100", liquidity="maker")
    sell(book, "5", "120", liquidity="maker")
    # Closed 2 at +20 each = 40 gross. Fees: 2*100*0.0002 = 0.04 on entry,
    # 5*120*0.0002 = 0.12 on the whole exit fill (all of it was traded).
    assert book.realized_pnl_after_fees == Decimal("40") - Decimal("0.04") - Decimal("0.12")


# --- keys: spot and perp are not the same instrument ------------------------

def test_the_same_symbol_on_two_venues_is_two_positions():
    book = a_book()
    buy(book, "1", "100", venue="binance")
    buy(book, "1", "200", venue="hyperliquid")
    assert {(p.venue, p.quantity) for p in book.local_positions()} == {
        ("binance", Decimal("1")), ("hyperliquid", Decimal("1"))}


def test_a_venue_with_no_declared_fee_schedule_is_refused_not_defaulted():
    book = a_book()
    with pytest.raises(UndeclaredFeeSchedule, match="bybit"):
        buy(book, "1", "100", venue="bybit")


def test_a_refused_fill_leaves_the_book_unchanged():
    book = a_book()
    with pytest.raises(UndeclaredFeeSchedule):
        buy(book, "1", "100", venue="bybit")
    assert book.local_positions() == []
    assert book.fees_paid == Decimal("0")


# --- the handover to reconciliation -----------------------------------------

def test_local_positions_hands_over_the_type_the_reconciler_takes():
    book = a_book()
    buy(book, "2", "100")
    (position,) = book.local_positions()
    assert isinstance(position, VenuePosition)
    assert position.key == ("binance", "BTCUSDT")
    assert position.average_price == Decimal("100")


def test_the_book_names_its_own_accounting_so_a_journal_row_cannot_be_misread():
    assert a_book(Accounting.OPTIMISTIC).accounting is Accounting.OPTIMISTIC
    assert a_book(Accounting.PESSIMISTIC).accounting is Accounting.PESSIMISTIC


# --- refusals over defaults -------------------------------------------------

def test_a_fill_of_zero_is_rejected_rather_than_recorded_as_a_no_op():
    book = a_book()
    with pytest.raises(ValueError):
        buy(book, "0", "100")


def test_an_unknown_liquidity_is_refused_rather_than_priced_as_maker():
    book = a_book()
    with pytest.raises(ValueError, match="liquidity"):
        buy(book, "1", "100", liquidity="unknown")
