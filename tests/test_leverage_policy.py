"""CL-06: leverage is chosen, bounded, survivable, and charged for — RL-041."""

from decimal import Decimal

import pytest

from segment.capital_declaration import parse_declaration
from segment.leverage_policy import (
    FIXED,
    STOP_OUTSIDE_LIQUIDATION,
    VOLATILITY_TARGETED,
    annualise_window_volatility,
    borrow_interest,
    choose_leverage,
    fit_leverage_to_stop,
    liquidation_distance,
)

WINDOW_NS = 180_000_000_000          # the live feature window


def _declaration(rule=VOLATILITY_TARGETED, floor="1", perp="20", spot="5",
                 target="40", maintenance="0.005"):
    return parse_declaration({
        "declared_at": "2026-08-19T00:00:00Z",
        "portfolio_usdt": "2000",
        "per_bot_cap_fraction": "0.60",
        "min_margin_per_trade_usdt": "5",
        "max_margin_per_trade_usdt": "50",
        "leverage": {"rule": rule, "floor": floor,
                     "ceiling": {"perp": perp, "spot": spot},
                     "target_annual_vol_pct": target},
        "spot_borrow_annual_pct": "8",
        "maintenance_margin_rate": maintenance,
    })


# ------------------------------------------------------------- the conversion


def test_a_window_volatility_annualises_by_root_of_time():
    # 180-second windows: sqrt(31,536,000 / 180) = 418.57 of them in a year.
    annual = annualise_window_volatility(Decimal("0.001"), WINDOW_NS)

    assert float(annual) == pytest.approx(0.41857, rel=1e-4)


def test_no_estimate_is_distinguishable_from_an_estimate_of_zero():
    # A zero would divide into infinite leverage, so the two must not collapse.
    assert annualise_window_volatility(None, WINDOW_NS) is None
    assert annualise_window_volatility(Decimal("0"), WINDOW_NS) is None
    assert annualise_window_volatility(Decimal("0.001"), 0) is None


# ------------------------------------------------------------- choosing


def test_a_quiet_instrument_earns_more_leverage_than_a_violent_one():
    declaration = _declaration()
    quiet = choose_leverage(declaration=declaration, segment="perp",
                            window_volatility=Decimal("0.0002"), window_ns=WINDOW_NS)
    violent = choose_leverage(declaration=declaration, segment="perp",
                              window_volatility=Decimal("0.004"), window_ns=WINDOW_NS)

    assert quiet.leverage > violent.leverage
    assert quiet.rule == VOLATILITY_TARGETED


def test_leverage_is_clamped_to_the_segment_ceiling():
    # An almost motionless instrument would otherwise ask for hundreds of x.
    choice = choose_leverage(declaration=_declaration(), segment="spot",
                             window_volatility=Decimal("0.000001"),
                             window_ns=WINDOW_NS)

    assert choice.leverage == Decimal("5")


def test_leverage_is_clamped_to_the_floor_for_a_violent_instrument():
    choice = choose_leverage(declaration=_declaration(), segment="perp",
                             window_volatility=Decimal("1"), window_ns=WINDOW_NS)

    assert choice.leverage == Decimal("1")


def test_an_instrument_with_no_volatility_estimate_is_floored_not_ceilinged():
    # Unknown risk is not an argument for more leverage. Defaulting to the ceiling
    # would hand the most leverage to the instruments least understood.
    choice = choose_leverage(declaration=_declaration(), segment="perp",
                             window_volatility=None, window_ns=WINDOW_NS)

    assert choice.leverage == Decimal("1")
    assert "floored rather than assumed" in choice.reason


def test_the_fixed_rule_takes_the_ceiling_and_ignores_volatility():
    choice = choose_leverage(declaration=_declaration(rule=FIXED), segment="perp",
                             window_volatility=Decimal("0.5"), window_ns=WINDOW_NS)

    assert choice.leverage == Decimal("20")
    assert choice.rule == FIXED


def test_the_choice_carries_the_numbers_it_was_made_from():
    choice = choose_leverage(declaration=_declaration(), segment="perp",
                             window_volatility=Decimal("0.001"), window_ns=WINDOW_NS)

    assert "annualised_volatility_pct" in choice.evidence
    assert choice.as_dict()["leverage_rule"] == VOLATILITY_TARGETED


# ------------------------------------------------------------- liquidation


def test_the_liquidation_distance_shrinks_as_leverage_rises():
    assert liquidation_distance(Decimal("1"), Decimal("0.005")) == Decimal("0.995")
    assert liquidation_distance(Decimal("20"), Decimal("0.005")) == Decimal("0.045")


def test_a_stop_that_fits_leaves_the_leverage_alone():
    declaration = _declaration(rule=FIXED, perp="10")
    choice = choose_leverage(declaration=declaration, segment="perp")

    fitted = fit_leverage_to_stop(choice, stop_fraction=Decimal("0.01"),
                                  declaration=declaration)

    assert fitted.leverage == Decimal("10")
    assert fitted.reduced_from is None
    assert not fitted.refused


def test_a_stop_wider_than_liquidation_reduces_the_leverage_rather_than_refusing():
    # A 5% stop at 20x is a stop that can never fire: liquidation is at 4.5%.
    # The trade is fine at lower leverage, so the LEVERAGE is what did not fit.
    declaration = _declaration(rule=FIXED, perp="20")
    choice = choose_leverage(declaration=declaration, segment="perp")

    fitted = fit_leverage_to_stop(choice, stop_fraction=Decimal("0.05"),
                                  declaration=declaration)

    assert not fitted.refused
    assert fitted.leverage < Decimal("20")
    assert fitted.reduced_from == Decimal("20")
    allowed = liquidation_distance(fitted.leverage,
                                   declaration.maintenance_margin_rate) * Decimal("0.8")
    assert Decimal("0.05") < allowed


def test_a_stop_that_does_not_fit_even_at_the_floor_is_refused_with_its_numbers():
    # An 90% stop cannot sit inside 0.8 x 99.5% of the price. Nothing survives it.
    declaration = _declaration(rule=FIXED, perp="20")
    choice = choose_leverage(declaration=declaration, segment="perp")

    fitted = fit_leverage_to_stop(choice, stop_fraction=Decimal("0.9"),
                                  declaration=declaration)

    assert fitted.refused
    assert fitted.refusal == STOP_OUTSIDE_LIQUIDATION
    assert "allowed_stop_fraction_at_floor" in fitted.evidence


def test_every_reduced_leverage_leaves_the_stop_strictly_inside_liquidation():
    # The invariant probe_leverage_declared_and_bounded asserts on live journals.
    declaration = _declaration(rule=FIXED, perp="20")
    for stop in ("0.001", "0.01", "0.02", "0.03", "0.04"):
        choice = choose_leverage(declaration=declaration, segment="perp")
        fitted = fit_leverage_to_stop(choice, stop_fraction=Decimal(stop),
                                      declaration=declaration)
        assert not fitted.refused
        distance = liquidation_distance(fitted.leverage,
                                        declaration.maintenance_margin_rate)
        assert Decimal(stop) < distance, f"stop {stop} at {fitted.leverage}x"


# ------------------------------------------------------------- borrow cost


def test_an_unleveraged_position_borrows_nothing_and_pays_exactly_zero():
    # Exactly zero, not a small number that reads as a rounding artefact.
    assert borrow_interest(notional_usdt=Decimal("100"), margin_usdt=Decimal("100"),
                           annual_pct=Decimal("8"), held_ns=86_400_000_000_000) == 0


def test_interest_is_charged_on_the_borrowed_part_not_the_whole_notional():
    # 5x on 100 margin borrows 400, not 500. Charging the whole notional would
    # bill the trader interest on their own capital.
    one_year = int(365 * 24 * 3600 * 1e9)
    interest = borrow_interest(notional_usdt=Decimal("500"),
                               margin_usdt=Decimal("100"),
                               annual_pct=Decimal("8"), held_ns=one_year)

    assert float(interest) == pytest.approx(32.0, rel=1e-9)


def test_interest_scales_with_how_long_the_position_was_held():
    day = 86_400_000_000_000
    one = borrow_interest(notional_usdt=Decimal("500"), margin_usdt=Decimal("100"),
                          annual_pct=Decimal("8"), held_ns=day)
    five = borrow_interest(notional_usdt=Decimal("500"), margin_usdt=Decimal("100"),
                           annual_pct=Decimal("8"), held_ns=5 * day)

    assert float(five) == pytest.approx(float(one) * 5, rel=1e-12)
    # 5x spot held five days at 8% is ~44 bp of the margin - not a rounding error
    # at any horizon, and the reason it cannot be left unmodelled.
    assert float(five / Decimal("100")) == pytest.approx(0.0044, rel=1e-2)
