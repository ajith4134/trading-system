"""The labels arrive late, and every defect here comes from pretending they do not.

Three populations exist at any instant - matured, pending, and matured-but-nobody
-wrote-the-outcome-down - and the naive implementation collapses them. Scoring
pending decisions as wrong punishes whichever model made more recent predictions,
which is always the challenger. Dropping them silently makes the comparison shrink
whenever the pipeline is behind, and nothing says so.
"""
import json

import pytest

from models.champion_challenger import (
    LEDGER_FILE,
    MIN_MATURED_DECISIONS,
    DecisionAlreadySettled,
    NoVerdict,
    ShadowLedger,
    SwapVerdict,
    UnknownDecision,
)

_HOUR = 3_600_000_000_000
_START = 1_700_000_000_000_000_000


def _ledger(tmp_path):
    return ShadowLedger(tmp_path / "shadow")


def _record(ledger, index, champion, challenger, *, made_at=None,
            horizon_ns=_HOUR):
    made = _START + index * 60_000_000_000 if made_at is None else made_at
    return ledger.record_decision(
        f"d{index:04d}", made_at_ns=made, matures_at_ns=made + horizon_ns,
        champion_prediction=champion, challenger_prediction=challenger)


def _populate(ledger, n, champion_correct, challenger_correct, *, actual=1):
    """`n` decisions where each model is right on the given fraction of them."""
    for i in range(n):
        champion = actual if i < int(n * champion_correct) else 1 - actual
        challenger = actual if i < int(n * challenger_correct) else 1 - actual
        _record(ledger, i, champion, challenger)
        ledger.settle(f"d{i:04d}", actual, settled_at_ns=_START + 10 * _HOUR)


def _long_after(n):
    """A clock well past every decision's maturity."""
    return _START + n * 60_000_000_000 + 10 * _HOUR


# --- the three populations ------------------------------------------------

def test_a_pending_decision_is_not_a_loss(tmp_path):
    """Scoring it as wrong punishes whichever model made more recent
    predictions - which is always the challenger, by construction."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 40, 0.5, 0.9)
    # One more, made now, maturing in an hour, and settled optimistically.
    _record(ledger, 999, 0, 1, made_at=_START + 100 * _HOUR)

    verdict = ledger.compare(now_ns=_long_after(40))

    assert verdict.pending_decisions == 1
    assert verdict.matured_decisions == 40


def test_a_matured_but_unsettled_decision_is_counted_apart(tmp_path):
    """It means something upstream stopped writing outcomes, which is an
    operational fault and not a result - and it looks identical to `pending`
    unless the two are separated."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 35, 0.5, 0.9)
    _record(ledger, 500, 1, 1)                      # matured, never settled

    verdict = ledger.compare(now_ns=_long_after(500))

    assert verdict.unsettled_past_maturity == 1
    assert verdict.pending_decisions == 0
    assert verdict.matured_decisions == 35


def test_every_decision_lands_in_exactly_one_population(tmp_path):
    ledger = _ledger(tmp_path)
    _populate(ledger, 35, 0.6, 0.7)
    _record(ledger, 500, 1, 1)                          # matured, unsettled
    _record(ledger, 999, 1, 1, made_at=_START + 100 * _HOUR)   # pending

    verdict = ledger.compare(now_ns=_long_after(500))

    assert (verdict.matured_decisions + verdict.pending_decisions
            + verdict.unsettled_past_maturity) == 37


# --- the comparison -------------------------------------------------------

def test_a_clearly_better_challenger_is_recommended(tmp_path):
    """The only test here that can say swap. Without it every refusal below is
    satisfied by a module that never recommends anything."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 200, champion_correct=0.45, challenger_correct=0.85)

    verdict = ledger.compare(now_ns=_long_after(200))

    assert isinstance(verdict, SwapVerdict)
    assert verdict.should_swap
    assert verdict.challenger_accuracy > verdict.champion_accuracy
    assert "SWAP" in verdict.detail


def test_a_marginally_better_challenger_is_not_swapped_in(tmp_path):
    """On a few hundred decisions a two-point difference is routinely noise, and
    swapping the model that prices real positions on noise is how a system
    churns."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 200, champion_correct=0.60, challenger_correct=0.62)

    verdict = ledger.compare(now_ns=_long_after(200))

    assert verdict.challenger_accuracy > verdict.champion_accuracy
    assert not verdict.should_swap


def test_an_identical_challenger_is_reported_as_nothing_to_swap_to(tmp_path):
    """The bootstrap has no variance to work with, and 'indistinguishable' is
    the answer rather than an exception."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 60, champion_correct=0.7, challenger_correct=0.7)

    verdict = ledger.compare(now_ns=_long_after(60))

    assert not verdict.should_swap
    assert verdict.p_value is None
    assert "nothing to swap to" in verdict.detail


def test_too_few_matured_decisions_is_its_own_finding(tmp_path):
    """"Nobody has looked yet" and "the challenger is not better" read the same
    to a caller, and only one of them means the challenger might be."""
    ledger = _ledger(tmp_path)
    _populate(ledger, MIN_MATURED_DECISIONS - 5, 0.4, 0.9)

    verdict = ledger.compare(now_ns=_long_after(MIN_MATURED_DECISIONS))

    assert isinstance(verdict, NoVerdict)
    assert not isinstance(verdict, SwapVerdict)
    assert "Nobody has looked yet" in verdict.detail


def test_the_window_span_is_reported_so_a_calm_stretch_is_visible(tmp_path):
    """A challenger that beat the champion across six calm hours has beaten it
    across six calm hours. Judging that is MD-022, which does not exist - saying
    how long the window was is the honest half that can be built today."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 200, 0.45, 0.85)

    verdict = ledger.compare(now_ns=_long_after(200))

    assert verdict.window_span_ns > 0
    assert "MD-022 does not exist yet" in verdict.detail


def test_both_models_are_scored_on_the_same_decisions(tmp_path):
    """A challenger that started shadowing on Tuesday has no opinion about
    Monday, and letting each model be scored on 'whatever it has' compares two
    different weeks and calls it a model difference."""
    ledger = _ledger(tmp_path)
    _populate(ledger, 50, 0.5, 0.9)

    verdict = ledger.compare(now_ns=_long_after(50))

    # Every scored decision carries both predictions by construction: there is
    # no path through `record_decision` that stores one without the other.
    stored = ledger.decisions()
    scored = [d for d in stored.values() if d.is_settled]
    assert len(scored) == verdict.matured_decisions
    assert all(d.champion_prediction is not None
               and d.challenger_prediction is not None for d in scored)


# --- the record -----------------------------------------------------------

def test_the_record_survives_a_reopened_ledger(tmp_path):
    """A comparison spans days. Held in memory it would be a session, and the
    first restart would reset the evidence to nothing while the code kept
    working."""
    _populate(_ledger(tmp_path), 40, 0.5, 0.9)

    reopened = _ledger(tmp_path).compare(now_ns=_long_after(40))

    assert reopened.matured_decisions == 40


def test_a_settlement_is_appended_never_written_over_the_decision(tmp_path):
    ledger = _ledger(tmp_path)
    _record(ledger, 0, 1, 1)
    ledger.settle("d0000", 1, settled_at_ns=_START + 5 * _HOUR)

    lines = (tmp_path / "shadow" / LEDGER_FILE).read_text(
        encoding="utf-8").splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["decision", "settlement"]


def test_settling_twice_is_refused(tmp_path):
    """A second write is either a duplicate worth seeing or a revision, and a
    revision is the negotiation an append-only record exists to prevent."""
    ledger = _ledger(tmp_path)
    _record(ledger, 0, 1, 1)
    ledger.settle("d0000", 1)

    with pytest.raises(DecisionAlreadySettled):
        ledger.settle("d0000", 0)


def test_settling_an_unrecorded_decision_is_refused(tmp_path):
    """It means the two halves of this record were written by processes that
    disagree about what happened."""
    with pytest.raises(UnknownDecision):
        _ledger(tmp_path).settle("never-happened", 1)


def test_a_label_available_at_decision_time_is_refused(tmp_path):
    """It is not a delayed label, and the delay is what this module is about."""
    with pytest.raises(ValueError, match="delayed label"):
        _ledger(tmp_path).record_decision(
            "d0", made_at_ns=_START, matures_at_ns=_START,
            champion_prediction=1, challenger_prediction=1)


def test_a_torn_final_line_does_not_hide_the_record(tmp_path):
    ledger = _ledger(tmp_path)
    _populate(ledger, 35, 0.5, 0.9)
    with open(tmp_path / "shadow" / LEDGER_FILE, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "settle')

    assert ledger.compare(now_ns=_long_after(35)).matured_decisions == 35


def test_an_empty_ledger_says_nobody_has_looked(tmp_path):
    verdict = _ledger(tmp_path).compare(now_ns=_START)

    assert isinstance(verdict, NoVerdict)
    assert verdict.matured_decisions == 0


def test_this_module_does_not_move_an_alias(tmp_path):
    """The comparison is a fact about two models; what to do about it is a
    decision about capital, and it lives in the caller. Pinned as an import
    check because the likeliest way this erodes is a convenience helper."""
    import ast

    import models.champion_challenger as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}

    # Checked against the parsed source, not the text: the docstring names the
    # registry on purpose, to say that moving the alias is somebody else's act.
    assert not any("model_registry" in name for name in imported), imported
    assert "assign_alias" not in called
