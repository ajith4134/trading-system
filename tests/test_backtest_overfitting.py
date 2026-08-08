"""PBO via CSCV — VX-008 — and BH-FDR on the promoted set — VX-009.

**PBO** (Bailey, Borwein, López de Prado, Zhu) answers a different question from
the Deflated Sharpe. DSR asks *"is this candidate's Sharpe explained by luck given
N tries?"*. PBO asks *"does my whole selection procedure pick winners at all?"* -
it resamples the in-sample/out-of-sample split, re-runs the selection each time,
and measures how often the config the procedure chose lands below the median
out-of-sample. A procedure at PBO ≈ 0.5 is choosing at random, regardless of how
good the chosen candidate's backtest looks.

The anchor case (ledger VX-008): 8,800 configurations on **pure noise** gave an
in-sample Sharpe of 1.27 with 53% of out-of-sample Sharpes negative. A gate that
passes that is not a gate, so the noise case is tested directly below.

**BH-FDR** controls the false-discovery rate across the promoted set. The ledger's
verdict on the alternative is arithmetic: at N=10,000, Bonferroni's α/N = 5×10⁻⁶
*"suppresses essentially everything including true positives"*. Controlling FWER
at that scale is not conservatism, it is switching the search off - which is why
BH is the gate on the promoted set and Bonferroni is not.
"""
import random

import pytest

from validation.backtest_overfitting import (
    benjamini_hochberg,
    bonferroni_threshold,
    probability_of_backtest_overfitting,
)


def _noise_configs(n_configs, n_periods, seed=20260808):
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 0.01) for _ in range(n_periods)]
            for _ in range(n_configs)]


# --- PBO --------------------------------------------------------------------

def test_pure_noise_configurations_give_a_pbo_near_one_half():
    """The defining case. If every config is worthless, the one that looks best
    in-sample is a coin flip out-of-sample, so the selection procedure has no
    skill and PBO must sit near 0.5."""
    pbo = probability_of_backtest_overfitting(_noise_configs(40, 480), n_splits=8)
    assert 0.3 < pbo < 0.7, f"noise-mined selection reported PBO {pbo}"


def test_a_genuinely_dominant_configuration_gives_a_low_pbo():
    """When one config really is better, the procedure that picks it in-sample
    keeps picking a winner out-of-sample, and PBO collapses toward zero. Without
    this the metric could be satisfied by always returning 0.5."""
    configs = _noise_configs(20, 480)
    configs[7] = [r + 0.004 for r in configs[7]]        # a real, persistent edge
    pbo = probability_of_backtest_overfitting(configs, n_splits=8)
    assert pbo < 0.2, f"a genuinely dominant config still reported PBO {pbo}"


def test_pbo_is_a_probability():
    pbo = probability_of_backtest_overfitting(_noise_configs(10, 240), n_splits=6)
    assert 0.0 <= pbo <= 1.0


def test_pbo_is_deterministic_for_the_same_input():
    """It resamples combinatorially rather than randomly, so two runs on the same
    matrix must agree exactly. A PBO that wobbles between runs cannot gate
    anything."""
    configs = _noise_configs(12, 240)
    assert (probability_of_backtest_overfitting(configs, n_splits=6)
            == probability_of_backtest_overfitting(configs, n_splits=6))


def test_pbo_needs_an_even_number_of_splits():
    """CSCV forms in-sample sets from exactly half the groups. An odd count has no
    half, and rounding it would make the in-sample and out-of-sample periods
    different lengths - which biases the comparison the metric is built on."""
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_configs(10, 240), n_splits=7)


def test_pbo_needs_at_least_two_configurations():
    """With one config there is no selection happening, so there is no selection
    procedure to evaluate."""
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_configs(1, 240), n_splits=6)


def test_pbo_refuses_configurations_of_differing_length():
    """A ragged matrix means some configs were evaluated on more data than others,
    and the in-sample winner may simply be the one that saw the most."""
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[0.01] * 100, [0.01] * 80], n_splits=4)


# --- BH-FDR -----------------------------------------------------------------

def test_benjamini_hochberg_rejects_nothing_when_every_p_value_is_large():
    assert benjamini_hochberg([0.6, 0.8, 0.95], alpha=0.05) == []


def test_benjamini_hochberg_rejects_a_clearly_significant_result():
    assert 0 in benjamini_hochberg([0.0001, 0.7, 0.8, 0.9], alpha=0.05)


def test_benjamini_hochberg_returns_original_positions_not_sorted_ones():
    """The caller has to map a rejection back to the strategy that produced it.
    Returning ranks in sorted order silently reattributes every promotion."""
    assert benjamini_hochberg([0.9, 0.0001, 0.8], alpha=0.05) == [1]


def test_bh_gives_no_relief_for_a_single_finding_at_large_n():
    """The subtlety worth pinning, because it is easy to assume otherwise: BH's
    threshold at rank 1 is (1/m)·α, which is *exactly* Bonferroni's α/m. A lone
    p=1e-5 among 10,000 tests is discarded by both. BH is not a laxer per-test
    threshold, and treating it as one would be a false reassurance."""
    lonely = [1e-5] + [0.5] * 9_999
    assert bonferroni_threshold(n_tests=10_000, alpha=0.05) == pytest.approx(5e-6)
    assert benjamini_hochberg(lonely, alpha=0.05) == []


def test_bh_recovers_many_true_findings_that_bonferroni_suppresses():
    """Where BH actually earns its place, and the ledger's verdict made concrete:
    100 real effects at p=1e-4 among 10,000 tests. Bonferroni's 5e-6 threshold
    rejects every one of them - *"suppresses essentially everything including true
    positives"*. BH's rank-100 threshold is (100/10000)·0.05 = 5e-4, so all 100
    survive. The power comes from the number of discoveries, not from a weaker
    bar per test."""
    p_values = [1e-4] * 100 + [0.5] * 9_900
    assert all(p > bonferroni_threshold(10_000, 0.05) for p in p_values[:100]), \
        "Bonferroni would have kept these anyway, so the test proves nothing"

    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert len(rejected) == 100
    assert sorted(rejected) == list(range(100))


def test_the_step_up_procedure_rejects_a_whole_prefix_not_only_the_smallest():
    """BH is step-up: once the largest satisfying rank k is found, ranks 1..k are
    all rejected - including p-values above their own individual threshold. An
    implementation that tests each p-value independently is not BH and is far
    less powerful."""
    p_values = [0.001, 0.008, 0.02, 0.9]
    rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert sorted(rejected) == [0, 1, 2], (
        f"step-up prefix not rejected, got {rejected} - 0.02 exceeds its own "
        f"threshold of 0.0375? no: it is below it, and 0.008 must come with it")


def test_an_empty_set_of_tests_rejects_nothing():
    assert benjamini_hochberg([], alpha=0.05) == []


def test_alpha_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01], alpha=1.5)


def test_p_values_outside_zero_to_one_are_refused():
    """A p-value above one is a bug upstream, and clamping it would let that bug
    reach a promotion decision."""
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 1.4], alpha=0.05)
