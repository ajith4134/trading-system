"""CL-01: the declared capital budget parses, refuses, and reloads — RL-040, RL-041."""

import json
import os
from decimal import Decimal

import pytest

from segment.capital_declaration import (
    NO_DECLARATION,
    CapitalDeclaration,
    DeclarationRejected,
    DeclarationReloader,
    declaration_path,
    parse_declaration,
)


def _payload(**overrides):
    """A declaration that is valid, so each test can break exactly one thing."""
    payload = {
        "declared_at": "2026-08-19T14:30:00Z",
        "portfolio_usdt": "2000",
        "per_bot_cap_fraction": "0.60",
        "min_margin_per_trade_usdt": "5",
        "max_margin_per_trade_usdt": "50",
        "leverage": {
            "rule": "volatility_targeted",
            "ceiling": {"perp": "20", "spot": "5"},
            "floor": "1",
            "target_annual_vol_pct": "40",
        },
        "spot_borrow_annual_pct": "8.0",
        "maintenance_margin_rate": "0.005",
    }
    for key, value in overrides.items():
        if key.startswith("leverage."):
            payload["leverage"][key.split(".", 1)[1]] = value
        elif value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def _write(path, payload, *, mtime_ns=None):
    path.write_text(json.dumps(payload))
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


# ---------------------------------------------------------------- parsing


def test_a_valid_declaration_parses_every_field_as_decimal():
    declaration = parse_declaration(_payload())

    assert declaration.portfolio_usdt == Decimal("2000")
    assert declaration.per_bot_cap_fraction == Decimal("0.60")
    assert declaration.min_margin_per_trade_usdt == Decimal("5")
    assert declaration.max_margin_per_trade_usdt == Decimal("50")
    assert declaration.leverage_rule == "volatility_targeted"
    assert declaration.leverage_ceiling == {"perp": Decimal("20"), "spot": Decimal("5")}
    assert isinstance(declaration.portfolio_usdt, Decimal)


def test_the_per_bot_cap_is_the_fraction_of_the_portfolio():
    declaration = parse_declaration(_payload())

    # 2000 x 0.60. Both bots get the same cap; the pool is what they share.
    assert declaration.cap_for("perp") == Decimal("1200")
    assert declaration.cap_for("spot") == Decimal("1200")


def test_an_ungoverned_segment_is_refused_rather_than_given_a_cap():
    # dated and options are switched off (RL-039). Answering for them would be
    # inventing a budget for a bot nobody declared one for.
    declaration = parse_declaration(_payload())

    with pytest.raises(DeclarationRejected):
        declaration.cap_for("dated")
    with pytest.raises(DeclarationRejected):
        declaration.ceiling_for("options")


def test_a_json_float_is_refused_rather_than_rounded():
    # 0.1 is not 0.1. A budget is the last place to discover binary float error,
    # so the value that was typed must be the value that is used.
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(portfolio_usdt=2000.0))

    assert caught.value.field == "portfolio_usdt"
    assert "float" in caught.value.detail


def test_a_missing_field_names_itself():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(min_margin_per_trade_usdt=None))

    assert caught.value.field == "min_margin_per_trade_usdt"


def test_a_declaration_with_no_timestamp_is_refused():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(declared_at=None))

    assert caught.value.field == "declared_at"


@pytest.mark.parametrize("fraction", ["0", "-0.1", "1.5"])
def test_a_cap_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(per_bot_cap_fraction=fraction))

    assert caught.value.field == "per_bot_cap_fraction"


def test_a_zero_minimum_margin_is_refused():
    # This is the defect being fixed: a floor of zero is what the system does
    # today, and it is why the median trade used 0.00014 USDT.
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(min_margin_per_trade_usdt="0"))

    assert caught.value.field == "min_margin_per_trade_usdt"


def test_a_maximum_below_the_minimum_is_refused():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(min_margin_per_trade_usdt="50",
                                   max_margin_per_trade_usdt="5"))

    assert caught.value.field == "max_margin_per_trade_usdt"


def test_a_maximum_above_the_per_bot_cap_is_refused():
    # 2000 x 0.60 = 1200. A 1500 maximum means no trade could ever be opened,
    # which would render as a bot that simply never finds anything.
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(max_margin_per_trade_usdt="1500"))

    assert caught.value.field == "max_margin_per_trade_usdt"
    assert "1200" in caught.value.detail


def test_an_unimplemented_leverage_rule_is_refused():
    # A declared rule nothing implements reads as a decision that was made.
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(**{"leverage.rule": "confidence_scaled"}))

    assert caught.value.field == "leverage.rule"
    assert "volatility_targeted" in caught.value.detail


def test_the_fixed_leverage_rule_is_accepted():
    declaration = parse_declaration(_payload(**{"leverage.rule": "fixed"}))

    assert declaration.leverage_rule == "fixed"


def test_a_leverage_floor_below_one_is_refused():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(**{"leverage.floor": "0.5"}))

    assert caught.value.field == "leverage.floor"


def test_a_ceiling_below_the_floor_is_refused():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(**{"leverage.floor": "10",
                                      "leverage.ceiling": {"perp": "20", "spot": "5"}}))

    assert caught.value.field == "leverage.ceiling.spot"


def test_a_missing_ceiling_for_a_governed_segment_is_refused():
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(**{"leverage.ceiling": {"perp": "20"}}))

    assert caught.value.field == "spot"


def test_zero_spot_borrow_with_leveraged_spot_is_refused():
    # Leveraged spot is borrowed money. A zero rate makes it a free loan that
    # inflates spot's returns MORE the more leverage it takes - a bug that
    # rewards recklessness (RL-041).
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(spot_borrow_annual_pct="0"))

    assert caught.value.field == "spot_borrow_annual_pct"
    assert "free loan" in caught.value.detail


def test_zero_spot_borrow_is_allowed_when_spot_is_unleveraged():
    declaration = parse_declaration(
        _payload(spot_borrow_annual_pct="0",
                 **{"leverage.ceiling": {"perp": "20", "spot": "1"}}))

    assert declaration.spot_borrow_annual_pct == Decimal("0")


def test_a_ceiling_whose_liquidation_distance_vanishes_is_refused():
    # At 200x with 0.5% maintenance margin the liquidation distance is exactly
    # zero: the position is liquidated on open and no hard stop could ever fire
    # first, so the risk gate would believe in protection it does not have.
    with pytest.raises(DeclarationRejected) as caught:
        parse_declaration(_payload(**{"leverage.ceiling": {"perp": "200", "spot": "5"}}))

    assert caught.value.field == "leverage.ceiling.perp"
    assert "liquidated on open" in caught.value.detail


def test_a_non_object_declaration_is_refused():
    with pytest.raises(DeclarationRejected):
        parse_declaration(["not", "an", "object"])


# ---------------------------------------------------------------- reloading


def test_the_reloader_loads_the_file_once_and_not_again(tmp_path):
    path = _write(tmp_path / "capital.json", _payload())
    reloader = DeclarationReloader(path)

    first = reloader.poll()
    second = reloader.poll()

    assert first is second
    assert reloader.reloads == 1
    assert reloader.rejection is None


def test_the_reloader_picks_up_a_changed_file_without_a_restart(tmp_path):
    path = _write(tmp_path / "capital.json", _payload(), mtime_ns=1_000_000_000)
    reloader = DeclarationReloader(path)
    assert reloader.poll().portfolio_usdt == Decimal("2000")

    _write(path, _payload(portfolio_usdt="5000"), mtime_ns=2_000_000_000)

    assert reloader.poll().portfolio_usdt == Decimal("5000")
    assert reloader.reloads == 2


def test_a_missing_file_yields_no_declaration_and_says_where_to_write_one(tmp_path):
    reloader = DeclarationReloader(tmp_path / "absent.json")

    assert reloader.poll() is None
    assert "write_capital_declaration" in reloader.rejection


def test_unreadable_json_yields_no_declaration_and_names_the_file(tmp_path):
    path = tmp_path / "capital.json"
    path.write_text("{not json")
    reloader = DeclarationReloader(path)

    assert reloader.poll() is None
    assert "not readable JSON" in reloader.rejection


def test_a_rejected_edit_drops_the_previous_declaration_rather_than_keeping_it(tmp_path):
    # The deliberate choice: an edit that fails to parse is an edit someone MEANT.
    # Continuing on the superseded numbers would be trading against a stated
    # intention while the board showed no problem.
    path = _write(tmp_path / "capital.json", _payload(), mtime_ns=1_000_000_000)
    reloader = DeclarationReloader(path)
    assert reloader.poll() is not None

    _write(path, _payload(min_margin_per_trade_usdt="0"), mtime_ns=2_000_000_000)

    assert reloader.poll() is None
    assert reloader.current is None
    assert "min_margin_per_trade_usdt" in reloader.rejection


def test_a_broken_file_is_not_reparsed_every_poll(tmp_path):
    # A file that is broken and NOT being edited must not cost a parse every six
    # seconds for as long as it stays broken.
    path = _write(tmp_path / "capital.json", _payload(portfolio_usdt="0"),
                  mtime_ns=1_000_000_000)
    reloader = DeclarationReloader(path)

    for _ in range(5):
        assert reloader.poll() is None

    assert reloader.reloads == 1


def test_a_declaration_appearing_after_an_absence_is_read(tmp_path):
    # The mtime is cleared when the file is gone, so the next file to appear is
    # always read even if it lands with an older timestamp than a previous one.
    path = tmp_path / "capital.json"
    reloader = DeclarationReloader(path)
    assert reloader.poll() is None

    _write(path, _payload(), mtime_ns=1_000_000_000)

    assert reloader.poll() is not None
    assert reloader.rejection is None


def test_the_loaded_declaration_carries_its_own_provenance(tmp_path):
    path = _write(tmp_path / "capital.json", _payload(), mtime_ns=1_234_000_000)
    declaration = DeclarationReloader(path).poll()

    assert declaration.source_path == str(path)
    assert declaration.source_mtime_ns == 1_234_000_000
    assert declaration.as_dict()["declared_at"] == "2026-08-19T14:30:00Z"


def test_the_default_path_sits_beside_the_segment_state(tmp_path):
    assert declaration_path(tmp_path) == tmp_path / "capital.json"


def test_the_refusal_reason_is_a_named_constant():
    # A bot that is broke must render as broke, never as a bot that found nothing.
    assert NO_DECLARATION == "NO_CAPITAL_DECLARATION"


def test_as_dict_round_trips_through_parse(tmp_path):
    original = parse_declaration(_payload())
    payload = original.as_dict()
    payload["leverage"] = {
        "rule": payload.pop("leverage_rule"),
        "floor": payload.pop("leverage_floor"),
        "ceiling": payload.pop("leverage_ceiling"),
        "target_annual_vol_pct": payload.pop("target_annual_vol_pct"),
    }

    reparsed = parse_declaration(payload)

    assert isinstance(reparsed, CapitalDeclaration)
    assert reparsed.portfolio_usdt == original.portfolio_usdt
    assert reparsed.leverage_ceiling == original.leverage_ceiling
