"""The tail-loss ceiling: enforced on real money, measured on paper.

§0 makes the cap part of the objective rather than a safety tax - "maximise
green days" on its own is the objective that selects for selling insurance, and
the cap is what makes it safe to optimise against. §6 makes the ceiling a human
gate: the system may move its limits inside it and may never raise it.

Ruled 2026-08-09: the caps bind only on real money. Paper trades fake capital,
and capital that cannot be lost does not need protecting - constraining
exploration there would cost the thing §0 calls the asset.
"""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from risk.tail_cap import (
    CEILING_FILE, DEFAULT_DAILY_LOSS_FRACTION, DEFAULT_DRAWDOWN_FRACTION,
    LIVE, PAPER, CeilingNotSet, TailCap, assess, enforce, read_ceiling,
    seed_ceiling,
)

CAP = TailCap(daily_loss_fraction=Decimal("0.010"),
              drawdown_fraction=Decimal("0.030"))


class _Watchdog:
    def __init__(self):
        self.trips = []

    def trip(self, reason, detail, pids=()):
        self.trips.append({"reason": reason, "detail": detail})
        return self.trips[-1]


# --------------------------------------------------------------------------
# the ceiling is the user's
# --------------------------------------------------------------------------

def test_a_missing_ceiling_refuses_rather_than_defaulting(tmp_path):
    """Not zero appetite, and not the default either - a question nobody
    answered. Answering it on the user's behalf is what §6 reserves to them."""
    with pytest.raises(CeilingNotSet) as refusal:
        read_ceiling(tmp_path)
    assert "§6" in str(refusal.value)


def test_seeding_never_overwrites_a_ceiling_that_exists(tmp_path):
    """The whole guarantee. This is the only function that writes the file, and
    it will not widen one already set - so no code path in this system can raise
    its own limit."""
    seed_ceiling(tmp_path, Decimal("0.005"), Decimal("0.020"))
    seed_ceiling(tmp_path, Decimal("0.500"), Decimal("0.900"))

    cap = read_ceiling(tmp_path)
    assert cap.daily_loss_fraction == Decimal("0.005")
    assert cap.drawdown_fraction == Decimal("0.020")


def test_nothing_in_the_module_raises_a_ceiling():
    """Enforced by reading the source, because the absence IS the mechanism - a
    permission check is something that can be argued past."""
    source = Path("src/risk/tail_cap.py").read_text(encoding="utf-8")
    writes = [line for line in source.splitlines()
              if "write_text" in line and "def " not in line]
    assert len(writes) == 1, f"more than one writer of the ceiling: {writes}"


def test_a_non_positive_ceiling_is_refused(tmp_path):
    """Zero would halt on the first tick rather than bound anything."""
    (tmp_path / CEILING_FILE).write_text(json.dumps(
        {"daily_loss_fraction": "0", "drawdown_fraction": "0.03"}), encoding="utf-8")
    with pytest.raises(CeilingNotSet):
        read_ceiling(tmp_path)


def test_the_defaults_sit_well_inside_a_year_undone():
    """§0's own table puts market-neutral carry at ~2.8% annualised, so 2.8% in
    a day IS a year undone - the ceiling on any sane answer, not a candidate."""
    a_year_undone = Decimal("0.028")
    assert DEFAULT_DAILY_LOSS_FRACTION < a_year_undone / 2
    assert DEFAULT_DRAWDOWN_FRACTION > DEFAULT_DAILY_LOSS_FRACTION


# --------------------------------------------------------------------------
# two caps, because they fail differently
# --------------------------------------------------------------------------

def test_a_single_bad_session_trips_the_daily_cap():
    a = assess(CAP, nav=Decimal("9880"), day_open_nav=Decimal("10000"),
               peak_nav=Decimal("10000"), mode=LIVE)
    assert a.daily_breached
    assert "day down" in a.reason()


def test_a_slow_bleed_trips_the_drawdown_the_daily_cap_never_sees():
    """Twenty consecutive 0.4% days is 8% gone with no single breach - close to
    the shape of the prior bot's net -$837 over 10,240 trades, where no
    individual day was dramatic."""
    a = assess(CAP, nav=Decimal("9600"), day_open_nav=Decimal("9620"),
               peak_nav=Decimal("10000"), mode=LIVE)

    assert not a.daily_breached, "the day alone looks fine, which is the point"
    assert a.drawdown_breached
    assert a.breached


def test_a_quiet_day_inside_both_breaches_nothing():
    a = assess(CAP, nav=Decimal("9990"), day_open_nav=Decimal("10000"),
               peak_nav=Decimal("10000"), mode=LIVE)
    assert not a.breached
    assert a.reason() == "within the ceiling"


def test_losses_are_fractions_of_nav_not_currency():
    """§4: every size is a fraction of NAV. A cap in currency means something
    different at each end of a dial that runs $2k to $100k+."""
    small = assess(CAP, Decimal("1980"), Decimal("2000"), Decimal("2000"), LIVE)
    large = assess(CAP, Decimal("99000"), Decimal("100000"), Decimal("100000"), LIVE)

    assert small.daily_loss_fraction == large.daily_loss_fraction
    assert small.daily_breached and large.daily_breached


# --------------------------------------------------------------------------
# the ruling: paper is free, live is not
# --------------------------------------------------------------------------

def test_paper_is_not_stopped_however_badly_it_breaches():
    """Fake capital does not need protecting, and constraining exploration
    there would cost the thing §0 calls the asset."""
    watchdog = _Watchdog()
    a = assess(CAP, nav=Decimal("5000"), day_open_nav=Decimal("10000"),
               peak_nav=Decimal("10000"), mode=PAPER)

    assert a.breached
    assert enforce(a, watchdog) is None
    assert watchdog.trips == [], "paper trading was halted"


def test_a_paper_breach_is_still_recorded_as_one():
    """A candidate tuned with no ceiling was optimised against a different
    objective from the one it must satisfy with money behind it. Promoting it
    hands capital to a strategy never tested against its own constraint."""
    a = assess(CAP, nav=Decimal("5000"), day_open_nav=Decimal("10000"),
               peak_nav=Decimal("10000"), mode=PAPER)

    assert a.would_have_breached is True
    assert a.enforced is False


def test_a_live_breach_trips_the_kill():
    """And this is the trigger the watchdog has never had - DECISIONS.md §13
    records the trip side unarmed because "a hard cap needs capital to have a
    hard cap over"."""
    watchdog = _Watchdog()
    a = assess(CAP, nav=Decimal("9880"), day_open_nav=Decimal("10000"),
               peak_nav=Decimal("10000"), mode=LIVE)

    outcome = enforce(a, watchdog)

    assert outcome is not None
    assert watchdog.trips[0]["reason"] == "tail_cap"
    assert "day down" in watchdog.trips[0]["detail"]


def test_a_live_run_inside_the_ceiling_trips_nothing():
    watchdog = _Watchdog()
    a = assess(CAP, Decimal("9990"), Decimal("10000"), Decimal("10000"), LIVE)

    assert enforce(a, watchdog) is None
    assert watchdog.trips == []


def test_a_live_breach_reaches_the_real_kill_file(tmp_path):
    """End to end against `ops.watchdog`, because the whole point is that the
    kill file every trading path already consults gets written - and
    `prove_plumbing` already refuses to start while it exists."""
    from ops.watchdog import Watchdog, is_killed

    a = assess(CAP, Decimal("9800"), Decimal("10000"), Decimal("10000"), LIVE)
    enforce(a, Watchdog(tmp_path, identity_path=tmp_path / "absent"))

    assert is_killed(tmp_path), "a breached tail cap left trading enabled"
