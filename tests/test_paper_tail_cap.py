"""An adaptive risk limit rises exactly when risk rises. That is the danger.

The user asked for a cap that adapts on paper, because paper is for practising
and a limit that halts every experiment teaches nothing. The reason this file is
mostly about BOUNDS is that the naive version of that request is how accounts
die: derive a limit from recent realised risk and it stops binding at the moment
it was for.

The other half is separation. There is no path here that produces a live cap,
and `risk.tail_cap` keeps its single writer and its absolute meaning - §6 says no
code in this system raises the user's ceiling, and an adaptive function living
next to that invariant is how it would eventually be blurred.
"""
import json
from decimal import Decimal

import numpy as np
import pytest

from risk.paper_tail_cap import (
    HISTORY_FILE,
    MAX_WIDENING_MULTIPLE,
    MIN_OBSERVATIONS,
    MIN_TIGHTENING_MULTIPLE,
    derive_paper_cap,
    read_paper_cap_history,
    record_paper_cap,
)
from risk.tail_cap import LIVE, PAPER, TailCap

_LIVE = TailCap(daily_loss_fraction=Decimal("0.010"),
                drawdown_fraction=Decimal("0.030"))


def _returns(n, scale, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.normal(0.0, scale, n))


# --- the bounds, which are the whole safety argument ---------------------

def test_a_violent_series_cannot_widen_the_cap_without_limit():
    """The danger, stated: a limit derived from recent realised risk rises
    exactly when risk rises, so it stops binding at the moment it was for."""
    derived = derive_paper_cap(_LIVE, _returns(2000, 0.05), n_samples=200)

    assert derived.is_derived
    assert derived.widening_bound_hit
    assert derived.cap.drawdown_fraction == (
        _LIVE.drawdown_fraction * MAX_WIDENING_MULTIPLE)


def test_hitting_the_widening_bound_is_reported_not_silently_granted():
    """"The measured drawdown wants more room than we will give it" is a
    finding, and it is the one a caller most needs to see."""
    derived = derive_paper_cap(_LIVE, _returns(2000, 0.05), n_samples=200)

    assert "widening bound bit" in derived.describe()
    assert derived.measured_drawdown_fraction > derived.cap.drawdown_fraction


def test_a_calm_series_cannot_collapse_the_cap_to_nothing():
    """A cap derived from an unusually quiet stretch would halt on the first
    ordinary day, which teaches the same nothing as halting on the first tick."""
    derived = derive_paper_cap(_LIVE, _returns(2000, 1e-6), n_samples=200)

    assert derived.tightening_bound_hit
    assert derived.cap.drawdown_fraction == (
        _LIVE.drawdown_fraction * MIN_TIGHTENING_MULTIPLE)


def test_an_ordinary_series_lands_between_the_bounds():
    """Neither bound firing is the case the module exists for - if every input
    hit a bound, the derivation would be decoration on a constant.

    The series is short and quiet on purpose: 2,000 draws at 0.4% each compound
    into a bootstrapped p90 max drawdown far above 9%, so a "normal-looking"
    series of that length hits the widening bound. That is the module working,
    and it is why this test needed a measured fixture rather than a guessed one.
    """
    derived = derive_paper_cap(_LIVE, _returns(300, 0.0012), n_samples=200,
                               block_size=25)

    assert derived.is_derived
    assert not derived.widening_bound_hit
    assert not derived.tightening_bound_hit
    assert (_LIVE.drawdown_fraction * MIN_TIGHTENING_MULTIPLE
            < derived.cap.drawdown_fraction
            < _LIVE.drawdown_fraction * MAX_WIDENING_MULTIPLE)


def test_the_bounds_are_declared_and_not_symmetric():
    """Widening is the dangerous direction and is allowed less freely in
    proportion than tightening."""
    assert MAX_WIDENING_MULTIPLE > 1
    assert MIN_TIGHTENING_MULTIPLE < 1


# --- too little history ---------------------------------------------------

def test_too_little_history_passes_the_live_ceiling_through_unchanged():
    """A cap derived from twenty paper minutes is not adaptive, it is random -
    and the tidy percentile it produces is what would make it convincing."""
    derived = derive_paper_cap(_LIVE, _returns(MIN_OBSERVATIONS - 10, 0.01))

    assert not derived.is_derived
    assert derived.cap == _LIVE
    assert "random rather than adaptive" in derived.reason


def test_a_passed_through_cap_is_distinguishable_from_a_derived_one():
    """They are the same numbers in the common early case and mean entirely
    different things, so a caller that cannot tell them apart will report a
    proposal as a measurement."""
    passed = derive_paper_cap(_LIVE, _returns(10, 0.01))
    derived = derive_paper_cap(_LIVE, _returns(2000, 0.004), n_samples=200)

    assert not passed.is_derived and derived.is_derived
    assert "passed through" in passed.describe()


def test_it_returns_rather_than_raises_on_too_little_history():
    """The early state of every paper run is 'not enough yet', and a board
    should render that without a try block."""
    result = derive_paper_cap(_LIVE, [])

    assert result.observations == 0
    assert not result.is_derived


# --- paper only -----------------------------------------------------------

def test_the_derived_cap_is_always_paper_mode():
    for returns in ([], _returns(2000, 0.004)):
        assert derive_paper_cap(_LIVE, returns, n_samples=100).mode == PAPER


def test_no_function_here_produces_a_live_cap():
    """§6: no code in this system raises the user's ceiling. This module is
    where that would erode, so the absence is pinned rather than trusted."""
    import ast

    import risk.paper_tail_cap as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for function in functions:
        arguments = {a.arg for a in function.args.args
                     + function.args.kwonlyargs}
        assert "mode" not in arguments, (
            f"{function.name} takes a mode argument, which is the shape that "
            f"lets a live cap be requested from here")
    # Checked against the parsed source, not the text: the docstring names
    # `seed_ceiling` and LIVE on purpose, to say what this module must not do.
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    called |= {node.func.id for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "seed_ceiling" not in called, "this module must never write the ceiling"
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "LIVE" not in names, "no code path here may reference live mode"


def test_the_live_ceiling_module_still_refuses_to_be_overwritten(tmp_path):
    """The invariant this module is separate in order to protect. A regression
    here would not show up in any test of the adaptive path."""
    from risk.tail_cap import read_ceiling, seed_ceiling

    seed_ceiling(tmp_path, Decimal("0.01"), Decimal("0.03"))
    seed_ceiling(tmp_path, Decimal("0.50"), Decimal("0.90"))   # must not widen

    assert read_ceiling(tmp_path).drawdown_fraction == Decimal("0.03")


# --- the record -----------------------------------------------------------

def test_every_derivation_is_appended_with_what_moved_it(tmp_path):
    """"The strategy never breached" is meaningless unless the limit it was
    measured against that day is recoverable."""
    first = derive_paper_cap(_LIVE, _returns(2000, 0.004, seed=1), n_samples=200)
    record_paper_cap(tmp_path, first, previous=_LIVE)
    second = derive_paper_cap(_LIVE, _returns(2000, 0.02, seed=2), n_samples=200)
    record_paper_cap(tmp_path, second, previous=first.cap)

    history = read_paper_cap_history(tmp_path)
    assert len(history) == 2
    assert history[1]["previous_drawdown_fraction"] == str(
        first.cap.drawdown_fraction)
    assert history[1]["observations"] == 2000
    assert history[0]["mode"] == PAPER


def test_the_history_is_append_only(tmp_path):
    derived = derive_paper_cap(_LIVE, _returns(2000, 0.004), n_samples=200)
    record_paper_cap(tmp_path, derived)
    after_first = (tmp_path / HISTORY_FILE).read_text(encoding="utf-8")
    record_paper_cap(tmp_path, derived)

    assert (tmp_path / HISTORY_FILE).read_text(
        encoding="utf-8").startswith(after_first)


def test_a_torn_final_line_does_not_hide_the_record(tmp_path):
    """A process killed mid-append must not make unreadable the file that
    explains what a past paper run was measured against."""
    record_paper_cap(tmp_path, derive_paper_cap(_LIVE, _returns(2000, 0.004),
                                                n_samples=200))
    with open(tmp_path / HISTORY_FILE, "a", encoding="utf-8") as handle:
        handle.write('{"recorded_at')

    assert len(read_paper_cap_history(tmp_path)) == 1


def test_the_record_says_whether_a_bound_bit(tmp_path):
    """A cap at its bound and a cap the data chose are different states, and the
    history is where a past run's state is read from."""
    derived = derive_paper_cap(_LIVE, _returns(2000, 0.05), n_samples=200)
    record_paper_cap(tmp_path, derived)

    assert json.loads((tmp_path / HISTORY_FILE).read_text(
        encoding="utf-8").splitlines()[0])["widening_bound_hit"] is True


def test_an_empty_history_reads_as_empty_rather_than_raising(tmp_path):
    assert read_paper_cap_history(tmp_path) == []


def test_the_floor_is_the_bootstrap_s_own():
    """This module declared 100 while the bootstrap it calls requires 250, so a
    caller with 150 observations cleared this threshold and was then refused by
    machinery it had not called, with a message about a different number. A floor
    a module cannot honour reads as a decision and is not one."""
    from risk.drawdown_distribution import _MIN_OBSERVATIONS

    assert MIN_OBSERVATIONS == _MIN_OBSERVATIONS
