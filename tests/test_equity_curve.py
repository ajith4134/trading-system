"""A cap that governs nothing is a cap nobody has to obey.

`risk.paper_tail_cap` was built on the user's instruction that the paper ceiling
adapt, and nothing fed it. This is what makes it operative, and the tests are
mostly about the two things that would quietly make it lie: an equity curve built
from the flattering accounting, and unrealised P&L folded in as zero when it is
actually unknown.
"""
from decimal import Decimal

import pytest

from paper.equity_curve import EquityCurve
from risk.paper_tail_cap import MIN_OBSERVATIONS
from risk.tail_cap import PAPER, TailCap

_NAV = Decimal("10000")
_CEILING = TailCap(daily_loss_fraction=Decimal("0.010"),
                   drawdown_fraction=Decimal("0.030"))
_SECOND = 1_000_000_000


def _curve(nav=_NAV):
    return EquityCurve(nav=nav, live_ceiling=_CEILING)


def _walk(curve, deltas, unmarked=0):
    """Record a series of realised-P&L levels, both accountings."""
    for i, (optimistic, pessimistic) in enumerate(deltas):
        curve.record(at_ns=i * _SECOND,
                     realised_optimistic=Decimal(str(optimistic)),
                     realised_pessimistic=Decimal(str(pessimistic)),
                     unmarked_positions=unmarked)
    return curve


# --- both accountings, never blended -------------------------------------

def test_equity_is_tracked_under_both_accountings():
    """A single curve would have to pick one, which on an uncalibrated strategy
    is picking how good the answer is allowed to look."""
    curve = _walk(_curve(), [(100, 50)])
    sample = curve.samples[0]

    assert sample.optimistic == _NAV + 100
    assert sample.pessimistic == _NAV + 50


def test_the_returns_series_is_the_pessimistic_one():
    """The cap's purpose is to bound the bad case. A limit derived from the
    optimistic curve is set around a book that never existed."""
    curve = _walk(_curve(), [(0, 0), (1000, 100)])

    assert curve.returns() == [pytest.approx(0.01)]     # 100/10000, not 1000


# --- unrealised is unknown, not zero -------------------------------------

def test_an_unmarked_position_is_reported_rather_than_folded_in_as_zero():
    """Folding it in as zero reports a flat book when the truth is an unmeasured
    one - the same distinction `paper.blotter` makes."""
    curve = _walk(_curve(), [(0, 0)], unmarked=7)

    assert curve.samples[0].unmarked_positions == 7
    assert "realised P&L only" in curve.describe()


def test_a_fully_marked_curve_does_not_carry_the_caveat():
    """A caveat that always appears is a caveat nobody reads."""
    curve = _walk(_curve(), [(0, 0)], unmarked=0)

    assert "realised P&L only" not in curve.describe()


# --- the adaptive cap ----------------------------------------------------

def test_too_little_history_passes_the_live_ceiling_through():
    """The early state of every paper run, and it must render without a try
    block."""
    derived = _walk(_curve(), [(0, 0), (1, 1)]).paper_cap(n_samples=50)

    assert not derived.is_derived
    assert derived.cap == _CEILING
    assert derived.mode == PAPER


def test_a_long_quiet_curve_derives_a_cap():
    """The positive case - a module that only ever passes through is not
    adaptive."""
    deltas = [(i * 2, i) for i in range(MIN_OBSERVATIONS + 10)]
    derived = _walk(_curve(), deltas).paper_cap(n_samples=100)

    assert derived.is_derived
    assert derived.observations >= MIN_OBSERVATIONS


def test_a_violent_curve_cannot_widen_the_cap_without_limit():
    """The danger the bounds exist for: a limit derived from recent realised
    risk rises exactly when risk rises."""
    import math

    deltas = []
    level = 0.0
    for i in range(MIN_OBSERVATIONS + 10):
        level += 400 * math.sin(i * 1.7)      # large, oscillating
        deltas.append((level, level))
    derived = _walk(_curve(), deltas).paper_cap(n_samples=100)

    assert derived.is_derived
    assert derived.cap.drawdown_fraction <= (
        _CEILING.drawdown_fraction * Decimal("3"))


# --- breaches are recorded, never enforced -------------------------------

def test_a_drawdown_past_the_cap_is_recorded_as_would_have_breached():
    """Paper mode, by the design's ruling. A run that could not have lived
    inside its ceiling has not earned capital, and the record is the
    deliverable."""
    curve = _walk(_curve(), [(0, 0), (0, -1000)])       # -10% of NAV

    judged = curve.assess_latest(cap=_CEILING)

    assert judged is not None
    assert judged.mode == PAPER
    assert judged.would_have_breached
    assert not judged.enforced
    assert curve.would_have_breached == 1


def test_a_breach_is_recorded_with_its_reason():
    """"12 breaches" cannot be acted on; "12 peak-to-trough breaches" can."""
    curve = _walk(_curve(), [(0, 0), (0, -1000)])
    curve.assess_latest(cap=_CEILING)

    assert curve.breaches_by_reason
    assert "would-have-breach" in curve.describe()


def test_a_curve_inside_the_ceiling_records_nothing():
    """A recorder that fires on everything records nothing useful."""
    curve = _walk(_curve(), [(0, 0), (0, -10)])         # -0.1%
    judged = curve.assess_latest(cap=_CEILING)

    assert not judged.would_have_breached
    assert curve.would_have_breached == 0
    assert "no breach recorded" in curve.describe()


def test_nothing_here_enforces_or_halts():
    """Enforcing would destroy the evidence by preventing the breach it exists
    to observe. Pinned as an import check, since the way this erodes is a
    convenience call to the watchdog."""
    import ast

    import paper.equity_curve as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    assert not any("watchdog" in name for name in imported)
    assert "enforce" not in called


def test_the_live_ceiling_cannot_be_produced_here():
    """`risk.paper_tail_cap` cannot make one and this module cannot ask for
    one - §6 reserves the live ceiling to the user."""
    import ast

    import paper.equity_curve as module

    source = ast.parse(open(module.__file__, encoding="utf-8").read())
    called = {n.func.attr for n in ast.walk(source)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    called |= {n.func.id for n in ast.walk(source)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    assert "seed_ceiling" not in called
    assert "read_ceiling" not in called


# --- the empty case -------------------------------------------------------

def test_an_unobserved_book_says_so():
    curve = _curve()

    assert curve.assess_latest() is None
    assert "has not been observed" in curve.describe()


def test_the_peak_only_ever_rises():
    """A peak that could fall would make every drawdown shallower than it was."""
    curve = _walk(_curve(), [(0, 100), (0, 500), (0, 200), (0, 50)])
    judged = curve.assess_latest(cap=_CEILING)

    # Peak 10,500; latest 10,050 -> a 4.3% drawdown, past the 3% ceiling.
    assert judged.would_have_breached
