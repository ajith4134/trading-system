"""Nothing said no, and the engine opened 363 positions.

That is the gap this closes, and it is why the tests here are mostly about
refusing. The two rules that carry the module are: **refuse, never trim** - a
silently reduced order is a different order from the one the strategy asked for,
and its expectancy was computed on the size it asked for - and **report every
breach**, because a gate that returns on the first one makes a caller discover a
rate limit one rejected order at a time.
"""
from decimal import Decimal

import pytest

from risk.pre_trade_gate import (
    BUY,
    SELL,
    GateLimits,
    NavNotSupplied,
    OrderRateWindow,
    PreTradeGate,
    RejectionReason,
)

_NAV = Decimal("100000")
_PRICE = Decimal("100")
_SECOND = 1_000_000_000


def _limits(**over):
    base = dict(
        max_order_notional=Decimal("10000"),
        max_position_notional=Decimal("20000"),
        max_position_nav_fraction=Decimal("0.20"),
        max_aggregate_nav_fraction=Decimal("0.80"),
        max_open_instruments=5,
        max_orders_per_window=10,
        order_rate_window_ns=60 * _SECOND,
        price_collar_fraction=Decimal("0.05"),
    )
    base.update(over)
    return GateLimits(**base)


def _gate(**over):
    return PreTradeGate(_limits(**over))


def _evaluate(gate, *, quantity="1", side=BUY, limit_price=None,
              open_positions=None, now_ns=1_000, reference=_PRICE, nav=_NAV):
    return gate.evaluate(
        venue="binance", symbol="BTCUSDT", side=side,
        quantity=Decimal(quantity), reference_price=reference,
        limit_price=limit_price, nav=nav,
        open_positions=open_positions or {}, now_ns=now_ns)


# --- the shape adopted from the donor -------------------------------------

def test_an_oversized_order_is_refused_not_trimmed():
    """The donor's best rule, generalised: "a trade the risk budget can't afford
    at least 1 lot of is rejected, not rounded up". A trimmed order is a
    different order from the one whose expectancy was computed."""
    decision = _evaluate(_gate(), quantity="200")      # 20,000 > 10,000

    assert not decision.approved
    assert decision.approved_quantity == Decimal(0)
    assert RejectionReason.ORDER_NOTIONAL_TOO_LARGE in decision.reasons


def test_every_breach_is_reported_not_just_the_first():
    """A gate that returns on the first violation makes a caller fix one thing,
    resubmit, and discover the next - which on a rate limit means learning about
    it one rejected order at a time."""
    gate = _gate(max_order_notional=Decimal("100"),
                 max_position_notional=Decimal("100"),
                 max_position_nav_fraction=Decimal("0.0001"))
    decision = _evaluate(gate, quantity="50")

    assert len(decision.reasons) >= 3
    assert RejectionReason.ORDER_NOTIONAL_TOO_LARGE in decision.reasons
    assert RejectionReason.POSITION_CAP_EXCEEDED in decision.reasons
    assert RejectionReason.POSITION_NAV_FRACTION_EXCEEDED in decision.reasons


def test_an_order_inside_every_limit_is_approved():
    """A gate that refuses everything is not a gate."""
    decision = _evaluate(_gate())

    assert decision.approved
    assert decision.approved_quantity == Decimal(1)
    assert decision.reasons == ()


# --- the caps -------------------------------------------------------------

def test_a_position_cap_counts_what_the_order_would_make_it():
    """Judging the order alone lets a book be built one compliant order at a
    time, which is exactly how 363 positions appeared."""
    gate = _gate(max_position_notional=Decimal("500"))
    decision = _evaluate(gate, quantity="1",
                         open_positions={("binance", "BTCUSDT"): Decimal("10")})

    assert RejectionReason.POSITION_CAP_EXCEEDED in decision.reasons


def test_a_reducing_order_is_not_blocked_by_the_position_cap():
    """A SELL against a long makes the position smaller. Blocking it would trap
    a book at its own limit - the one state where an exit matters most."""
    gate = _gate(max_position_notional=Decimal("500"))
    decision = _evaluate(gate, side=SELL, quantity="1",
                         open_positions={("binance", "BTCUSDT"): Decimal("10")})

    assert RejectionReason.POSITION_CAP_EXCEEDED not in decision.reasons


def test_aggregate_exposure_counts_every_instrument():
    gate = _gate(max_aggregate_nav_fraction=Decimal("0.01"))
    decision = _evaluate(gate, open_positions={
        ("binance", "ETHUSDT"): Decimal("5"),
        ("binance", "SOLUSDT"): Decimal("5")})

    assert RejectionReason.AGGREGATE_EXPOSURE_EXCEEDED in decision.reasons


def test_a_new_instrument_is_refused_once_the_book_is_full():
    """The concrete gap: nothing stopped the engine opening a 363rd position."""
    gate = _gate(max_open_instruments=2)
    decision = _evaluate(gate, open_positions={
        ("binance", "ETHUSDT"): Decimal("1"),
        ("binance", "SOLUSDT"): Decimal("1")})

    assert RejectionReason.TOO_MANY_OPEN_INSTRUMENTS in decision.reasons


def test_adding_to_an_instrument_already_held_is_not_a_new_instrument():
    """The cap is on breadth, not on trading. A full book must still be able to
    manage what it holds."""
    gate = _gate(max_open_instruments=1)
    decision = _evaluate(gate, open_positions={
        ("binance", "BTCUSDT"): Decimal("1")})

    assert RejectionReason.TOO_MANY_OPEN_INSTRUMENTS not in decision.reasons


# --- the rate limit -------------------------------------------------------

def test_the_order_rate_is_capped_per_instrument():
    """A control against a runaway loop, which is the failure a paper engine
    reaching a broken tape actually has."""
    gate = _gate(max_orders_per_window=3)
    for _ in range(3):
        assert _evaluate(gate).approved
    decision = _evaluate(gate)

    assert RejectionReason.ORDER_RATE_EXCEEDED in decision.reasons


def test_the_rate_window_slides():
    gate = _gate(max_orders_per_window=2, order_rate_window_ns=10 * _SECOND)
    _evaluate(gate, now_ns=0)
    _evaluate(gate, now_ns=_SECOND)

    assert not _evaluate(gate, now_ns=2 * _SECOND).approved
    assert _evaluate(gate, now_ns=20 * _SECOND).approved


def test_the_rate_is_counted_in_aggregate_too():
    """A loop that hits one symbol and a loop that sprays the universe are the
    same defect wearing different shapes."""
    window = OrderRateWindow(window_ns=10 * _SECOND)
    for i in range(5):
        window.record("*", i)

    assert window.count("*", 5) == 5
    assert window.count("*", 100 * _SECOND) == 0


# --- the collar, and the case it cannot cover -----------------------------

def test_a_fat_finger_limit_is_refused():
    decision = _evaluate(_gate(), limit_price=Decimal("200"))

    assert RejectionReason.PRICE_OUTSIDE_COLLAR in decision.reasons


def test_a_limit_inside_the_collar_passes():
    decision = _evaluate(_gate(), limit_price=Decimal("102"))

    assert decision.approved


def test_a_market_order_has_no_price_to_collar():
    """Stated rather than silent: a collar that skips every market order is a
    control that reads as active and is not. The guard for one is
    `max_order_notional`."""
    decision = _evaluate(_gate(), limit_price=None, quantity="200")

    assert RejectionReason.PRICE_OUTSIDE_COLLAR not in decision.reasons
    assert RejectionReason.ORDER_NOTIONAL_TOO_LARGE in decision.reasons


# --- refusals rather than assumptions -------------------------------------

def test_a_missing_nav_is_refused_rather_than_invented():
    """Every fractional limit is a fraction of it, and a gate that invented one
    computes fractions of a number nobody set."""
    with pytest.raises(NavNotSupplied):
        _evaluate(_gate(), nav=None)


def test_an_unpriceable_order_is_refused_rather_than_waved_through():
    """Every remaining rule is about notional. A gate that lets an order pass
    because it could not price it is not a gate."""
    decision = _evaluate(_gate(), reference=None)

    assert not decision.approved
    assert RejectionReason.NO_REFERENCE_PRICE in decision.reasons


def test_a_non_positive_reference_is_treated_as_no_price():
    """The placeholder-price defect this store has actually shipped."""
    decision = _evaluate(_gate(), reference=Decimal(0))

    assert RejectionReason.NO_REFERENCE_PRICE in decision.reasons


# --- what is deliberately elsewhere ---------------------------------------

def test_the_gate_owns_no_daily_loss_or_drawdown_rule():
    """They belong to `risk.tail_cap`, which reads a ceiling the user owns and
    that no code may raise. Two homes for one §6 limit means one that drifts."""
    assert not any("daily" in r.value or "drawdown" in r.value
                   for r in RejectionReason)


def test_refusals_are_counted_by_reason():
    """A board reporting "12 orders blocked" cannot be acted on; one reporting
    which rule blocked them can."""
    gate = _gate(max_order_notional=Decimal("10"))
    _evaluate(gate, quantity="5")
    _evaluate(gate, quantity="5")

    assert gate.refusals_by_reason["order_notional_too_large"] == 2
    assert "order_notional_too_large 2" in gate.describe()


# --- the limits live in a file the user owns ------------------------------

def test_the_limits_are_seeded_once_and_never_overwritten(tmp_path):
    """The same refusal `risk.tail_cap.seed_ceiling` makes: no code path in this
    system may widen its own limit. Editing the file is how it changes, which
    keeps the decision with whoever owns the capital."""
    from risk.pre_trade_gate import read_gate_limits, seed_gate_limits

    seed_gate_limits(tmp_path, {"max_open_instruments": "3"})
    seed_gate_limits(tmp_path, {"max_open_instruments": "9999"})
    limits, _nav = read_gate_limits(tmp_path)

    assert limits.max_open_instruments == 3


def test_missing_limits_raise_rather_than_defaulting(tmp_path):
    """So a caller decides between running gated and running ungated, out loud,
    rather than by accident."""
    from risk.pre_trade_gate import LimitsNotSet, read_gate_limits

    with pytest.raises(LimitsNotSet, match="not by accident"):
        read_gate_limits(tmp_path)


def test_a_non_positive_nav_is_refused(tmp_path):
    """It would make every fractional limit zero and refuse the first order -
    a gate that blocks everything is as useless as one that blocks nothing."""
    from risk.pre_trade_gate import LimitsNotSet, read_gate_limits, seed_gate_limits

    seed_gate_limits(tmp_path, {"nav": "0"})

    with pytest.raises(LimitsNotSet, match="non-positive NAV"):
        read_gate_limits(tmp_path)


def test_the_seeded_file_says_it_is_a_proposal(tmp_path):
    """A seeded number read as a decision is the failure `risk.tail_cap` already
    records: its ceiling has sat unconfirmed since 2026-08-09."""
    import json

    from risk.pre_trade_gate import LIMITS_FILE, seed_gate_limits

    seed_gate_limits(tmp_path)
    body = json.loads((tmp_path / LIMITS_FILE).read_text(encoding="utf-8"))

    assert "PROPOSAL" in body["_note"]
    assert "tail-cap.json" in body["_note"], (
        "and it must point at where the daily-loss and drawdown kills live, so "
        "nobody looks for them here")
