"""The first caller `validation/` has ever had, and the gates that refuse.

Trial Registry, Holdout Custodian, purged CPCV, deflated Sharpe, MinBTL, PBO and
SPA were written complete and reached by nothing - `promotion_gate` had zero
callers and every other module was reached only through it.

**A refusal is this working.** The tests below mostly check that the machinery
says no when it should, because saying yes is the easy half and the expensive
mistake.
"""
import numpy as np
import pandas as pd
import pytest

from validation.promotion_pipeline import (
    MAX_PBO, OBSERVED_FUNDING, SPA_ALPHA, CarryCandidate, candidate_returns,
    daily_carry, fold_returns, run, sealed_reader,
)
from validation.trial_registry import TrialRegistry

DAY_NS = 86_400_000_000_000
BASE = 1_700_000_000_000_000_000 // DAY_NS * DAY_NS


def _frame(days: int, symbols: int, seed: int = 1, edge: float = 0.0):
    """A funding frame with `days` of history for `symbols` symbols."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(symbols):
        for d in range(days):
            rate = rng.normal(loc=edge, scale=0.001)
            rows.append({"symbol": f"S{s}", "event_time_ns": BASE + d * DAY_NS,
                         "funding_rate": str(rate)})
    return pd.DataFrame(rows)


def _grid():
    return [CarryCandidate(lookback_days=lb, percentile=p)
            for lb in (10, 20) for p in (0.80, 0.90)]


# --------------------------------------------------------------------------
# it must not read the dataset that cannot be backtested
# --------------------------------------------------------------------------

def test_the_pipeline_reads_the_observed_dataset_not_the_reconstructed_one():
    """`funding_reconstructed` has an availability time of the fetch, so a
    backtest sees nothing in it. Naming the observed dataset as a constant makes
    the wrong one unreachable by editing a string at a call site."""
    from store.funding_backfill import DATASET as RECONSTRUCTED
    assert OBSERVED_FUNDING == "funding"
    assert OBSERVED_FUNDING != RECONSTRUCTED


# --------------------------------------------------------------------------
# the holdout, finally attached to something
# --------------------------------------------------------------------------

def test_a_read_inside_the_sealed_holdout_raises(tmp_path):
    """The guard existed and guarded nothing - no caller had ever passed a
    custodian to the reader. This is the difference between a rule and a
    mechanism."""
    from validation.holdout_custodian import HoldoutSealed

    reader, custodian = sealed_reader(tmp_path / "store", tmp_path / "trials",
                                      BASE, BASE + 30 * DAY_NS)
    assert custodian.covers(BASE + DAY_NS)
    with pytest.raises(HoldoutSealed):
        reader.read_as_of(BASE + DAY_NS)


def test_a_read_before_the_holdout_is_permitted(tmp_path):
    """Selection happens on history. The seal must not make the pipeline
    unrunnable, only make the future unreadable."""
    reader, _ = sealed_reader(tmp_path / "store", tmp_path / "trials",
                              BASE, BASE + 30 * DAY_NS)
    assert reader.read_as_of(BASE - 1).empty      # no store, but no refusal


# --------------------------------------------------------------------------
# the candidate obeys §5a.5
# --------------------------------------------------------------------------

def test_standing_aside_is_scored_as_a_zero_day_not_dropped():
    """Dropping the days it chose not to act scores the strategy only on the
    days it acted - which is how a rule that trades twice a year reports a
    wonderful Sharpe."""
    matrix = daily_carry(_frame(days=60, symbols=5))
    series = candidate_returns(matrix, CarryCandidate(lookback_days=10, percentile=0.9))

    assert len(series) == len(matrix), "days were dropped"
    assert (series == 0).any(), "no day stood aside, so nothing was tested"


def test_a_spike_cannot_pay_on_the_day_it_happened():
    """The lookahead this pipeline shipped with, and the test that encoded it.

    The signal is YESTERDAY's carry; the return is TODAY's rate. Comparing
    today's funding to a past threshold and then collecting today's funding is
    deciding to have been positioned for a rate that had not settled yet.

    Measured before the fix, on 400 days of PURE NOISE across 30 symbols:
    observed Sharpe 3.90, with the deflated Sharpe, MinBTL and purge-retention
    gates all passing it. Selecting the top decile of a symmetric distribution
    and averaging returns a positive number every single day - a selection
    artefact of the return definition wearing the shape of an edge.
    """
    matrix = pd.DataFrame({"A": [0.001] * 10 + [99.0, 0.002]})
    paid = candidate_returns(matrix, CarryCandidate(lookback_days=5, percentile=0.9))

    assert paid.iloc[10] == pytest.approx(0.0), "the spike paid itself"
    # It fires the day AFTER, and collects that day's rate rather than the spike.
    assert paid.iloc[11] == pytest.approx(0.002)


def test_pure_noise_does_not_manufacture_a_sharpe(tmp_path):
    """The regression that matters. Before the shift was fixed this returned
    3.90 out of nothing; the honest answer is indistinguishable from zero."""
    result = run(_frame(days=400, symbols=30, seed=7), TrialRegistry(tmp_path), _grid())
    assert abs(result.observed_sharpe) < 0.5, result.observed_sharpe
    assert result.promoted is False


def _cap():
    from decimal import Decimal
    from risk.tail_cap import TailCap
    return TailCap(Decimal("0.010"), Decimal("0.030"))


def test_a_real_edge_is_promoted(tmp_path):
    """A gate stack that refuses everything is not a gate stack. The same
    machinery that rejects noise has to pass a signal, or it is measuring
    nothing."""
    result = run(_frame(days=400, symbols=30, seed=7, edge=0.0004),
                 TrialRegistry(tmp_path), _grid(), cap=_cap(),
                 baseline=_a_passing_baseline())
    assert result.observed_sharpe > 0.5
    assert result.promoted is True, result.refusals


def test_the_pipeline_refuses_a_candidate_with_no_baseline_comparison(tmp_path):
    """The same edge, without MD-001's comparison. The pipeline does not compute
    one for itself: a gate that manufactured its own evidence is a gate marking
    its own homework."""
    result = run(_frame(days=400, symbols=30, seed=7, edge=0.0004),
                 TrialRegistry(tmp_path), _grid(), cap=_cap())
    assert result.promoted is False
    assert any("naive_baseline" in r for r in result.refusals)


def test_a_missing_ceiling_refuses_rather_than_disappearing(tmp_path):
    """§6 reserves the number to the user and says nothing trades real money
    without one. A gate that quietly vanished when unset is absence of evidence
    rendering as green."""
    result = run(_frame(days=400, symbols=30, seed=7, edge=0.0004),
                 TrialRegistry(tmp_path), _grid(), cap=None)

    tail = next(g for g in result.gates if g["name"] == "tail_cap")
    assert tail["passed"] is False
    assert result.promoted is False


def test_a_candidate_that_would_have_breached_is_not_promoted(tmp_path):
    """Paper is unconstrained by ruling, so nothing stopped it at the time. This
    is the record of whether it could have lived inside the ceiling - promoting
    without it hands capital to a strategy never tested against its own
    constraint."""
    from decimal import Decimal
    from risk.tail_cap import TailCap

    airless = TailCap(Decimal("0.00001"), Decimal("0.00001"))
    result = run(_frame(days=400, symbols=30, seed=7, edge=0.0004),
                 TrialRegistry(tmp_path), _grid(), cap=airless)

    assert result.promoted is False
    assert any("would have breached" in r for r in result.refusals)


def test_a_fold_is_scored_on_its_test_blocks_only():
    from validation.purged_cross_validation import combinatorial_purged_folds

    series = pd.Series(range(100), dtype=float)
    fold = combinatorial_purged_folds(100, "carry", n_groups=5, k_test=2)[0]
    scored = fold_returns(series, fold)

    expected = [float(v) for start, end in fold.test_blocks for v in range(start, end)]
    assert scored == expected


# --------------------------------------------------------------------------
# the gates refuse
# --------------------------------------------------------------------------

def test_too_little_history_refuses_before_it_scores_anything(tmp_path):
    """The live position on 2026-08-09: the observed record is two days old, the
    trailing windows never open, and the honest answer is to say so rather than
    score noise."""
    result = run(_frame(days=3, symbols=10), TrialRegistry(tmp_path), _grid())

    assert result.promoted is False
    assert any("not enough history" in r for r in result.refusals)


def test_pure_noise_is_refused_with_the_gates_named(tmp_path):
    """A strategy with no edge and plenty of data must be refused BY a gate,
    with the number that refused it - not by running out of rows."""
    registry = TrialRegistry(tmp_path)
    result = run(_frame(days=400, symbols=30, seed=7), registry, _grid())

    assert result.promoted is False
    assert result.refusals, "noise was not refused by anything"
    named = {g["name"] for g in result.gates}
    assert {"deflated_sharpe", "min_backtest_length", "purge_retention"} <= named


def test_every_configuration_raises_the_bar_for_the_winner(tmp_path):
    """The N that deflates the winner is the registry's cumulative count, so a
    wider search makes its own result harder to promote. That is the registry's
    whole purpose and the reason the search runs inside it."""
    registry = TrialRegistry(tmp_path)
    run(_frame(days=300, symbols=20), registry, _grid())
    after_small = registry.cumulative_count()

    run(_frame(days=300, symbols=20), registry,
        [CarryCandidate(lookback_days=lb, percentile=p)
         for lb in (10, 15, 20, 25) for p in (0.75, 0.80, 0.90)])

    assert registry.cumulative_count() > after_small


def test_pbo_and_spa_both_run_when_there_is_a_search_to_judge(tmp_path):
    """Both need the OTHER candidates, which `evaluate_for_promotion` never
    sees - it scores one candidate and knows nothing about the search around
    it."""
    result = run(_frame(days=400, symbols=30, seed=3), TrialRegistry(tmp_path), _grid())

    gates = {g["name"]: g for g in result.gates}
    assert "pbo" in gates and "spa" in gates
    assert gates["pbo"]["threshold"] == MAX_PBO
    assert gates["spa"]["threshold"] == SPA_ALPHA


def test_a_single_candidate_runs_no_pbo_because_there_is_no_selection(tmp_path):
    """CSCV measures a SELECTION procedure. With one configuration nothing was
    selected, and reporting a number would be inventing one."""
    result = run(_frame(days=300, symbols=20), TrialRegistry(tmp_path),
                 [CarryCandidate(lookback_days=10, percentile=0.9)])

    assert "pbo" not in {g["name"] for g in result.gates}


def test_the_verdict_is_not_promoted_when_any_gate_refuses(tmp_path):
    """Composed with AND. A gate that can be outvoted is decoration."""
    result = run(_frame(days=400, symbols=30, seed=11), TrialRegistry(tmp_path), _grid())

    if result.refusals:
        assert result.promoted is False
    failed = [g for g in result.gates if not g["passed"]]
    if failed:
        assert result.promoted is False


def test_an_empty_frame_refuses_rather_than_raising(tmp_path):
    result = run(pd.DataFrame(), TrialRegistry(tmp_path), _grid())
    assert result.promoted is False
    assert result.refusals == ["no funding rows to score"]


# --------------------------------------------------------------------------
# §5a.5's second axis: effective sample size
#
# "Every scan counts in the Trial Registry - corrected on both axes, trial count
# AND effective sample size." Counting calendar days demanded ~6.1 years of
# history for a Sharpe-1.0 claim at N=21; the independent bets to support it
# arrive far faster than one a day. This multiplier only ever makes promotion
# EASIER, which is the direction every defect this project has found failed in,
# so what it may not do is the part worth testing.
# --------------------------------------------------------------------------

def test_a_short_window_falls_back_to_counting_days(tmp_path):
    """A correlation matrix over a handful of rows reports whatever it likes.
    The uncertain case takes the conservative side."""
    from validation.promotion_pipeline import (
        MIN_DAYS_FOR_BREADTH, daily_carry, effective_sample_multiplier)

    matrix = daily_carry(_frame(days=MIN_DAYS_FOR_BREADTH - 1, symbols=40))
    assert effective_sample_multiplier(matrix, CarryCandidate(10, 0.9)) == 1.0


def test_a_wide_independent_cross_section_is_worth_more_than_one_a_day(tmp_path):
    from validation.promotion_pipeline import daily_carry, effective_sample_multiplier

    matrix = daily_carry(_frame(days=300, symbols=40, seed=5))
    multiplier = effective_sample_multiplier(matrix, CarryCandidate(20, 0.9))

    assert multiplier > 1.0
    assert multiplier <= 40, "more independent bets than symbols"


def test_the_multiplier_can_never_exceed_the_symbol_count(tmp_path):
    """Whatever the arithmetic says. 850 correlated streams are not 850 samples,
    and they are certainly not more than 850."""
    from validation.promotion_pipeline import daily_carry, effective_sample_multiplier

    matrix = daily_carry(_frame(days=300, symbols=5, seed=2))
    assert effective_sample_multiplier(matrix, CarryCandidate(20, 0.9)) <= 5


def test_the_multiplier_is_named_in_the_verdict_not_folded_into_a_year_count(tmp_path):
    """It makes promotion easier, so it has to be readable in the gate's own
    words rather than showing up as a year count nobody can reproduce."""
    result = run(_frame(days=400, symbols=30, seed=7, edge=0.0004),
                 TrialRegistry(tmp_path), _grid(), cap=_cap())

    minbtl = next(g for g in result.gates if g["name"] == "min_backtest_length")
    assert "effective breadth" in minbtl["detail"]


def test_the_gate_defaults_to_calendar_days_when_nobody_passes_a_multiplier(tmp_path):
    """`evaluate_for_promotion` is callable directly. Its default must be the
    conservative basis, not the permissive one."""
    from validation.promotion_gate import evaluate_for_promotion
    from validation.trial_registry import TrialSpec

    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.001, 0.01, 400))
    verdict = evaluate_for_promotion(
        TrialRegistry(tmp_path), TrialSpec("t", "carry", {}),
        backtest_fold=lambda fold: fold_returns(series, fold),
        n_rows=len(series), periods_per_year=365)

    minbtl = next(g for g in verdict.gates if g.name == "min_backtest_length")
    assert "x 1.0 effective breadth" in minbtl.detail


def test_readiness_reports_the_conservative_bound_before_breadth_is_measurable(tmp_path):
    """Under 30 days the multiplier fails closed to 1.0, so the projection is
    the conservative bound rather than the expectation - and a reader who does
    not know that will misread a fall from 2,220 days to 46 as a bug."""
    from validation.promotion_pipeline import readiness
    from validation.trial_registry import TrialSpec

    # A search has to have happened for MinBTL to demand anything: at N=1,
    # 2*ln(1)/SR^2 is zero and one look at the data needs no history to justify.
    registry = TrialRegistry(tmp_path)
    for i in range(20):
        registry.pre_register(TrialSpec(f"scan {i}", "carry", {}))

    state = readiness(_frame(days=2, symbols=900), registry)

    assert state["effective_breadth"] == 1.0, "breadth was estimated from 2 days"
    assert state["days_observed"] == 2
    assert state["ready"] is False
    assert state["days_required"] > 1000, state["days_required"]


def test_readiness_shortens_once_breadth_can_be_measured(tmp_path):
    """The same trial count, the same target - only a window long enough to
    measure the cross-section, and the requirement collapses."""
    from validation.promotion_pipeline import readiness

    registry = TrialRegistry(tmp_path)
    short = readiness(_frame(days=20, symbols=40, seed=4), registry)
    long_enough = readiness(_frame(days=300, symbols=40, seed=4), registry)

    assert short["effective_breadth"] == 1.0
    assert long_enough["effective_breadth"] > 1.0
    assert long_enough["days_required"] < short["days_required"]


def test_readiness_recedes_as_the_trial_count_grows(tmp_path):
    """N only rises and MinBTL rises with it, so the finish line moves away with
    every scan. Honest, and worth being able to watch."""
    from validation.promotion_pipeline import readiness
    from validation.trial_registry import TrialSpec

    registry = TrialRegistry(tmp_path)
    frame = _frame(days=300, symbols=40, seed=4)
    before = readiness(frame, registry)
    for i in range(30):
        registry.pre_register(TrialSpec(f"scan {i}", "carry", {}))
    after = readiness(frame, registry)

    assert after["n_trials"] > before["n_trials"]
    assert after["days_required"] > before["days_required"]


def _a_passing_baseline():
    """A real BaselineVerdict from a model that genuinely beats the random walk.

    Computed rather than hand-constructed, so this fixture cannot drift out of
    agreement with what `judge_against_naive_baseline` actually returns - a
    hand-built dataclass would keep passing after the real one started failing.
    """
    from models.naive_baseline import judge_against_naive_baseline
    prices, price = [], 100.0
    for i in range(240):
        price += 0.5 if i % 3 else -0.4
        prices.append(price)
    actual = prices[1:]
    skilled = [a + (0.02 if i % 2 else -0.02) for i, a in enumerate(actual)]
    verdict = judge_against_naive_baseline(actual, skilled)
    assert verdict.beats_naive, "fixture must actually pass, or it proves nothing"
    return verdict
