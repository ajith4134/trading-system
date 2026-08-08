"""The composition — the gate that actually calls the six pieces.

`CLAUDE.md` names this codebase's recurring failure: `tail_specs()` built, tested,
called by nothing; five circuit breakers living only in docstrings. Six correct
statistical modules that no single code path invokes would be the same defect at a
larger scale, so this is the harness that composes them into one decision:

    TrialRegistry (honest N) → CPCV folds (purged) → path Sharpes → DSR → MinBTL
    → PBO → a verdict that carries its own evidence

**The one thing this must not get wrong.** A candidate evaluated over C(6,2)=15
CPCV paths is **one trial, not fifteen**. `nse-botonly`'s gate passed the path
count in as the trial count, so a candidate selected from thousands was deflated as
though 15 things had been tried - rigorous-looking and nearly toothless. The
distinction is tested directly below, because it is the single easiest place for
the same error to reappear.
"""
import random

import pytest

from validation.promotion_gate import (
    PromotionVerdict,
    evaluate_for_promotion,
)
from validation.trial_registry import TrialRegistry, TrialSpec


def a_spec(name="momentum-v1", family="trend"):
    return TrialSpec(name=name, family=family, params={"lookback": 20})


def constant_edge(seed=1, edge=0.004, noise=0.01, n=2000):
    """A strategy with a real, persistent edge."""
    rng = random.Random(seed)
    return lambda fold: [rng.gauss(edge, noise) for _ in range(n // 6)]


def pure_noise(seed=1, noise=0.01, n=2000):
    rng = random.Random(seed)
    return lambda fold: [rng.gauss(0.0, noise) for _ in range(n // 6)]


# --- N is counted once per candidate, not once per fold ----------------------

def test_a_candidate_evaluated_over_fifteen_paths_counts_as_one_trial(tmp_path):
    """The donor's error, pinned. Fifteen CPCV paths are fifteen resamples of one
    candidate's returns, not fifteen looks at fifteen ideas. Counting them as
    trials inflates N with resampling noise; counting the candidate once is the
    only figure that means what DSR needs it to mean."""
    registry = TrialRegistry(tmp_path)
    evaluate_for_promotion(registry, a_spec(), constant_edge(),
                           n_rows=2400, n_groups=6, k_test=2)
    assert registry.cumulative_count() == 1, (
        "each CPCV path was counted as a separate trial - this is the "
        "number_of_paths-as-trial-count bug from nse-botonly")


def test_the_deflation_uses_the_registry_count_not_the_fold_count(tmp_path):
    """Two candidates evaluated means N=2 for the second, and its DSR must reflect
    that rather than the 15 paths it ran."""
    registry = TrialRegistry(tmp_path)
    for i in range(2):
        verdict = evaluate_for_promotion(registry, a_spec(f"c{i}"), constant_edge(seed=i),
                                         n_rows=2400, n_groups=6, k_test=2)
    assert verdict.n_trials == 2
    assert verdict.n_cpcv_paths == 15


def test_evaluating_many_candidates_makes_the_gate_harder(tmp_path):
    """The property that makes DSR worth using in-loop: the same candidate is
    harder to promote after a wide search than after a narrow one."""
    def dsr_after(n_prior):
        registry = TrialRegistry(tmp_path / f"r{n_prior}")
        for i in range(n_prior):
            evaluate_for_promotion(registry, a_spec(f"junk{i}"), pure_noise(seed=100 + i),
                                   n_rows=2400, n_groups=6, k_test=2)
        return evaluate_for_promotion(registry, a_spec("real"), constant_edge(),
                                      n_rows=2400, n_groups=6, k_test=2).deflated_sharpe

    assert dsr_after(30) <= dsr_after(2)


# --- the verdict carries its evidence ---------------------------------------

def test_the_verdict_reports_every_gate_with_its_measurement(tmp_path):
    """Rule 8 on a promotion decision. A verdict that says only pass or fail
    cannot be argued with, and a gate nobody can audit is one nobody will trust
    enough to act on."""
    verdict = evaluate_for_promotion(TrialRegistry(tmp_path), a_spec(), constant_edge(),
                                     n_rows=2400, n_groups=6, k_test=2)
    assert isinstance(verdict, PromotionVerdict)
    names = {g.name for g in verdict.gates}
    assert {"deflated_sharpe", "min_backtest_length", "purge_retention"} <= names
    for gate in verdict.gates:
        assert gate.detail, f"gate {gate.name} passed no evidence"
        assert gate.measured is not None


def test_a_genuine_edge_is_promoted_when_few_things_were_tried(tmp_path):
    verdict = evaluate_for_promotion(TrialRegistry(tmp_path), a_spec(), constant_edge(),
                                     n_rows=2400, n_groups=6, k_test=2,
                                     min_deflated_sharpe=0.95)
    assert verdict.promoted, f"a real edge was rejected: {verdict.rejection_reasons()}"


def test_pure_noise_is_not_promoted(tmp_path):
    """The whole point. A candidate with no edge must fail whichever gate catches
    it - the test does not care which, only that the gate holds."""
    verdict = evaluate_for_promotion(TrialRegistry(tmp_path), a_spec(), pure_noise(),
                                     n_rows=2400, n_groups=6, k_test=2,
                                     min_deflated_sharpe=0.95)
    assert not verdict.promoted


def test_the_best_of_many_noise_candidates_is_not_promoted(tmp_path):
    """The realistic version: mine noise, then submit the luckiest. The registry
    has already counted every miner, so the winner faces the N it earned."""
    registry = TrialRegistry(tmp_path)
    verdicts = [evaluate_for_promotion(registry, a_spec(f"n{i}"), pure_noise(seed=i),
                                       n_rows=2400, n_groups=6, k_test=2,
                                       min_deflated_sharpe=0.95)
                for i in range(60)]
    best = max(verdicts, key=lambda v: v.observed_sharpe)
    assert not best.promoted, (
        f"the luckiest of 60 noise candidates was promoted with observed Sharpe "
        f"{best.observed_sharpe:.3f} and DSR {best.deflated_sharpe:.3f}")


# --- refusals rather than defaults ------------------------------------------

def test_a_backtest_that_raises_still_counts_the_trial(tmp_path):
    """It consumed a look at the data. If the gate swallowed it, a search could
    reduce its own N by writing candidates that crash."""
    registry = TrialRegistry(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate_for_promotion(registry, a_spec(),
                               lambda fold: (_ for _ in ()).throw(RuntimeError("x")),
                               n_rows=2400, n_groups=6, k_test=2)
    assert registry.cumulative_count() == 1


def test_an_undeclared_family_is_refused_before_any_evaluation(tmp_path):
    """The purge horizon comes from the family, so an unknown family cannot be
    validated at all - and must not consume a trial pretending otherwise."""
    from validation.purged_cross_validation import UnknownStrategyFamily
    registry = TrialRegistry(tmp_path)
    with pytest.raises(UnknownStrategyFamily):
        evaluate_for_promotion(registry, a_spec(family="brand-new"), constant_edge(),
                               n_rows=2400, n_groups=6, k_test=2)
    assert registry.cumulative_count() == 0, (
        "a candidate that could never be validated still consumed a trial")


def test_a_fold_whose_purge_ate_all_the_training_data_is_reported(tmp_path):
    """A long horizon against a short series can purge everything. The gate must
    say so rather than returning a Sharpe distribution built on nothing."""
    verdict = evaluate_for_promotion(TrialRegistry(tmp_path),
                                     a_spec(family="trend"), constant_edge(),
                                     n_rows=12_000, n_groups=6, k_test=2,
                                     min_train_retention=0.9)
    retention = next(g for g in verdict.gates if g.name == "purge_retention")
    assert not retention.passed
    assert not verdict.promoted
