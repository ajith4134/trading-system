"""Rebuild on startup, reconcile against venue truth, refuse to start on mismatch.

`FEATURES.md` §6 / `DECISIONS.md` §6, RX-005: *"Rebuild local state on every startup;
**refuse to start if reconciliation fails**"* - with NautilusTrader's invariant
tolerance: quantity to instrument precision, average price within 0.01%.

The ledger flags the prior art as imperfect and says exactly how:
`live_trader.py` reconciles positions at startup but *"its periodic 'reconcile' loop
**only re-syncs capital, not positions**, despite the name - flagged in-source as
'narrower than its name implies'."* And it is **not a refuse-to-start gate**: it
adopts or closes mismatches on boot instead of stopping.

That difference is the requirement. Adopting a mismatch means the system decides, on
its own, that the venue is right and continues - which is reasonable exactly until
the mismatch is a symptom of something that will keep producing mismatches. Refusing
to start is recoverable by a human in minutes; trading on a position model that is
already known to be wrong is not.

Three sources have to agree before trading resumes:

  1. the **WAL** - what we intended, including intents of unknown fate
  2. the **local position model** - what we believe we hold
  3. **venue truth** - what the exchange says

Any disagreement outside tolerance halts. Unknown-fate intents halt regardless of
whether the positions happen to match, because an order that may be live is a
position that may be about to exist.
"""
from decimal import Decimal

import pytest

from execution.order_intent_wal import OrderIntent, OrderIntentWal
from execution.state_recovery import (
    ReconciliationFailed,
    VenuePosition,
    recover_and_reconcile,
)

NOW = 1_800_000_000_000_000_000
SECOND = 1_000_000_000


def an_intent(symbol="BTCUSDT", quantity="0.5", created_at_ns=NOW):
    return OrderIntent(strategy="momentum-v1", symbol=symbol, venue="binance",
                       side="BUY", quantity=Decimal(quantity),
                       created_at_ns=created_at_ns, valid_for_ns=30 * SECOND)


def venue(symbol="BTCUSDT", quantity="0.5", price="50000"):
    return VenuePosition(symbol=symbol, venue="binance",
                         quantity=Decimal(quantity),
                         average_price=Decimal(price))


# --- the happy path ---------------------------------------------------------

def test_matching_state_reconciles_and_permits_start(tmp_path):
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"ok": 1}, now_ns=NOW)

    report = recover_and_reconcile(
        wal=wal,
        local_positions=[venue()],
        venue_positions=[venue()],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert report.may_start
    assert report.mismatches == ()


def test_an_empty_system_reconciles(tmp_path):
    report = recover_and_reconcile(wal=OrderIntentWal(tmp_path),
                                   local_positions=[], venue_positions=[],
                                   quantity_steps={})
    assert report.may_start


def test_the_report_names_what_it_checked(tmp_path):
    """Rule 8. "Reconciliation passed" with no visible inputs is indistinguishable
    from a reconciliation that compared two empty lists."""
    report = recover_and_reconcile(wal=OrderIntentWal(tmp_path),
                                   local_positions=[venue()],
                                   venue_positions=[venue()],
                                   quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert report.n_positions_checked == 1
    assert report.n_unresolved_intents == 0
    assert "1" in report.detail


# --- quantity mismatch ------------------------------------------------------

def test_a_quantity_mismatch_beyond_instrument_precision_refuses_start(tmp_path):
    """The core gate. Believing we hold 0.5 while the venue holds 0.7 means every
    risk limit downstream is computed against a fiction."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(quantity="0.5")],
        venue_positions=[venue(quantity="0.7")],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert not report.may_start
    assert any("quantity" in m for m in report.mismatches)


def test_a_difference_below_instrument_precision_is_tolerated(tmp_path):
    """NautilusTrader's invariant: quantity to instrument precision. A difference
    smaller than the venue's own step size is not a difference it could represent."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(quantity="0.5")],
        venue_positions=[venue(quantity="0.5000001")],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert report.may_start


def test_an_undeclared_instrument_step_refuses_rather_than_guessing(tmp_path):
    """Fail closed. Without a step there is no defined tolerance, and picking one
    silently would decide how wrong the position model is allowed to be."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue()], venue_positions=[venue()],
        quantity_steps={})
    assert not report.may_start
    assert any("precision" in m or "step" in m for m in report.mismatches)


# --- price mismatch ---------------------------------------------------------

def test_an_average_price_gap_beyond_one_basis_point_refuses_start(tmp_path):
    """0.01% is NautilusTrader's tolerance. A wrong entry price is a wrong unrealised
    P&L, which is what the drawdown circuit breakers read."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(price="50000")],
        venue_positions=[venue(price="50100")],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert not report.may_start
    assert any("price" in m for m in report.mismatches)


def test_an_average_price_gap_within_tolerance_is_accepted(tmp_path):
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(price="50000")],
        venue_positions=[venue(price="50002")],   # 0.004%
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert report.may_start


# --- positions the other side does not have ---------------------------------

def test_a_position_the_venue_does_not_have_refuses_start(tmp_path):
    """We think we hold something that is not there. Every exit order against it
    would be rejected, and every risk calculation counts capital that is not at
    risk."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(symbol="ETHUSDT")], venue_positions=[],
        quantity_steps={"ETHUSDT": Decimal("0.01")})
    assert not report.may_start
    assert any("ETHUSDT" in m for m in report.mismatches)


def test_a_venue_position_we_do_not_know_about_refuses_start(tmp_path):
    """The more dangerous direction: an unmanaged live position. Nothing will size
    it, stop it, or close it, because nothing knows it exists."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[], venue_positions=[venue(symbol="SOLUSDT")],
        quantity_steps={"SOLUSDT": Decimal("0.1")})
    assert not report.may_start
    assert any("SOLUSDT" in m for m in report.mismatches)


def test_the_prior_art_behaviour_of_adopting_a_mismatch_is_not_available(tmp_path):
    """The donor adopted or closed mismatches on boot. There is deliberately no
    parameter here that does that: a system that silently accepts venue truth on
    startup cannot tell a benign restart from a bug that will keep producing
    mismatches."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[], venue_positions=[venue()],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert not report.may_start
    with pytest.raises(TypeError):
        recover_and_reconcile(
            wal=OrderIntentWal(tmp_path), local_positions=[],
            venue_positions=[venue()], quantity_steps={},
            adopt_mismatches=True)


# --- unresolved intents halt regardless -------------------------------------

def test_an_unresolved_intent_refuses_start_even_when_positions_match(tmp_path):
    """An order of unknown fate is a position that may be about to exist. Matching
    positions now says nothing about the moment that order fills."""
    wal = OrderIntentWal(tmp_path)
    with pytest.raises(TimeoutError):
        wal.submit(an_intent(), lambda i, c: (_ for _ in ()).throw(TimeoutError()),
                   now_ns=NOW)

    report = recover_and_reconcile(
        wal=wal, local_positions=[venue()], venue_positions=[venue()],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert not report.may_start
    assert report.n_unresolved_intents == 1
    assert any("unresolved" in m for m in report.mismatches)


def test_resolving_the_intent_then_permits_start(tmp_path):
    """The recovery path actually completing: query the venue, resolve, restart."""
    wal = OrderIntentWal(tmp_path)
    with pytest.raises(TimeoutError):
        wal.submit(an_intent(), lambda i, c: (_ for _ in ()).throw(TimeoutError()),
                   now_ns=NOW)
    wal.resolve(wal.unresolved()[0]["client_order_id"], outcome="rejected",
                detail="venue had no such order on query")

    report = recover_and_reconcile(
        wal=wal, local_positions=[venue()], venue_positions=[venue()],
        quantity_steps={"BTCUSDT": Decimal("0.001")})
    assert report.may_start


def test_a_corrupt_wal_refuses_start(tmp_path):
    """Not a mismatch report - a raise. If the WAL cannot be read, the list of orders
    that might be live is unknown, and there is nothing to reconcile against."""
    from execution.order_intent_wal import WalCorrupt
    wal = OrderIntentWal(tmp_path)
    wal.submit(an_intent(), lambda i, c: {"ok": 1}, now_ns=NOW)
    (tmp_path / "order_intents.ndjson").write_text("{ broken\n", encoding="utf-8")

    with pytest.raises(WalCorrupt):
        recover_and_reconcile(wal=OrderIntentWal(tmp_path), local_positions=[],
                              venue_positions=[], quantity_steps={})


# --- every mismatch is reported, not just the first -------------------------

def test_all_mismatches_are_reported_together(tmp_path):
    """A gate that stops at the first problem turns one restart into five. The
    operator needs the whole list to fix it once."""
    report = recover_and_reconcile(
        wal=OrderIntentWal(tmp_path),
        local_positions=[venue(symbol="BTCUSDT", quantity="0.5"),
                         venue(symbol="ETHUSDT", quantity="2")],
        venue_positions=[venue(symbol="BTCUSDT", quantity="0.9"),
                         venue(symbol="SOLUSDT", quantity="10")],
        quantity_steps={"BTCUSDT": Decimal("0.001"), "ETHUSDT": Decimal("0.01"),
                        "SOLUSDT": Decimal("0.1")})
    assert not report.may_start
    assert len(report.mismatches) >= 3


def test_raising_on_failure_is_available_for_a_startup_path(tmp_path):
    """The report form suits a status wall; a startup path wants an exception it
    cannot accidentally ignore by not checking a boolean."""
    with pytest.raises(ReconciliationFailed) as caught:
        recover_and_reconcile(
            wal=OrderIntentWal(tmp_path), local_positions=[],
            venue_positions=[venue()],
            quantity_steps={"BTCUSDT": Decimal("0.001")}, raise_on_failure=True)
    assert "BTCUSDT" in str(caught.value)
