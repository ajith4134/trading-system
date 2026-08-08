"""Deflated Sharpe as in-loop fitness — VX-005, and MinBTL — VX-006.

`FEATURES.md` §8 is specific that the Deflated Sharpe is **the search loop's
fitness function, not a report on the winner**. The distinction is the whole
value: a raw-Sharpe search climbs toward whichever candidate got luckiest, and
deflating afterwards only tells you the winner was luck. Deflating *inside* the
loop stops the climb from selecting for luck in the first place.

The empirical anchor for why (ledger VX-008): on **pure noise**, 8,800
configurations produced an in-sample Sharpe of **1.27** with 53% of out-of-sample
Sharpes negative. Any gate that would pass that is not a gate.

Two formulas, both from Bailey & López de Prado:

  PSR(SR*) = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]

  DSR      = PSR(SR₀),  SR₀ = √V[SR] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

with γ the Euler–Mascheroni constant, V[SR] the cross-sectional variance of the
Sharpes actually tried, and N the cumulative trial count.

MinBTL ≈ 2·ln(N) / E[max SR]² — the backtest length, in years, below which an
in-sample Sharpe of that size is expected from noise alone. The ledger's stated
consequence is the arithmetic check on the implementation: at N=45,
2·ln(45)/1.0² = 7.6 years, so five years of data cannot support a Sharpe-1.0
claim after 45 configurations.

**Verification honesty:** the formulas are implemented from their published form
and pinned below by properties (monotonicity in N, the N=1 identity, rejection of
noise-mined winners), not by reproducing a published worked example. No worked
numeric example from the papers was re-derived here.
"""
import math

import pytest

from validation.deflated_sharpe import (
    NotEnoughTrials,
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    min_backtest_length_years,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_ratio,
)


# --- the plain Sharpe, so the rest has a foundation --------------------------

def test_sharpe_of_a_constant_positive_return_is_undefined_not_infinite(tmp_path):
    """Zero variance means the Sharpe has no denominator. Returning a huge number
    would make a broken or leaked backtest look like the best strategy ever
    found - which is exactly how leakage gets promoted."""
    with pytest.raises(ZeroDivisionError):
        sharpe_ratio([0.01] * 50)


def test_sharpe_is_annualised_from_the_stated_periods_per_year():
    daily = [0.001, -0.002, 0.003, 0.0005, -0.001] * 50
    annual = sharpe_ratio(daily, periods_per_year=252)
    raw = sharpe_ratio(daily, periods_per_year=1)
    assert annual == pytest.approx(raw * math.sqrt(252), rel=1e-9)


# --- PSR --------------------------------------------------------------------

def test_psr_is_one_half_when_the_observed_sharpe_equals_the_benchmark():
    """At SR = SR* the probability of truly exceeding it is exactly a coin flip,
    whatever the sample size. A closed-form anchor the implementation must hit."""
    assert probabilistic_sharpe_ratio(
        observed_sharpe=1.0, benchmark_sharpe=1.0, n_observations=500,
        skew=0.0, kurtosis=3.0) == pytest.approx(0.5, abs=1e-12)


def test_psr_rises_with_more_observations():
    """The same Sharpe measured over a longer record is better evidence."""
    short = probabilistic_sharpe_ratio(1.0, 0.0, 100, 0.0, 3.0)
    long = probabilistic_sharpe_ratio(1.0, 0.0, 2000, 0.0, 3.0)
    assert long > short


def test_psr_is_punished_by_negative_skew_and_fat_tails():
    """A Sharpe earned by selling tails is worth less than the same number earned
    symmetrically. Ignoring the higher moments is how short-vol strategies pass
    a Sharpe gate right up until they don't."""
    # Short record, modest Sharpe. At T=500 and SR=1.0 the z-score is ~18 and
    # both answers are 1.0 to float precision - a real saturation of the
    # statistic, not a bug, but useless for telling the two apart.
    gaussian = probabilistic_sharpe_ratio(0.5, 0.0, 30, skew=0.0, kurtosis=3.0)
    nasty = probabilistic_sharpe_ratio(0.5, 0.0, 30, skew=-1.5, kurtosis=9.0)
    assert nasty < gaussian


# --- the expected maximum under the null ------------------------------------

def test_the_expected_max_sharpe_grows_with_the_number_of_trials():
    """This is the entire mechanism: try more things, and the best of them looks
    better even when none of them work."""
    variance = 0.25
    assert (expected_max_sharpe_under_null(n_trials=10, trial_variance=variance)
            < expected_max_sharpe_under_null(n_trials=1000, trial_variance=variance))


def test_the_expected_max_sharpe_is_zero_when_the_trials_never_differed():
    """No dispersion across trials means selection had nothing to select on, so
    it conferred no advantage to deflate away."""
    assert expected_max_sharpe_under_null(
        n_trials=500, trial_variance=0.0) == pytest.approx(0.0)


def test_a_single_trial_needs_no_deflation():
    """One look at the data is not multiple testing. The hurdle must collapse to
    zero, or a system that has only ever tried one thing is penalised for it."""
    assert expected_max_sharpe_under_null(
        n_trials=1, trial_variance=0.25) == pytest.approx(0.0)


def test_deflating_against_fewer_trials_than_were_run_is_refused():
    """`cpcv_botonly.py` passed the number of CPCV resample paths of a *single*
    strategy as the trial count, so a candidate chosen from thousands was
    deflated by N=15. The gate read as rigorous and was nearly toothless. N below
    one is not a smaller correction - it is a broken caller."""
    with pytest.raises(NotEnoughTrials):
        expected_max_sharpe_under_null(n_trials=0, trial_variance=0.25)


# --- DSR --------------------------------------------------------------------

def test_dsr_equals_psr_against_zero_when_only_one_trial_was_run():
    """With N=1 the deflated hurdle is zero, so the DSR must reduce *exactly* to
    the PSR against a zero benchmark. Any gap means the two disagree about what
    'no multiple testing' means, and the DSR is not a strict generalisation of
    the PSR the way the paper states."""
    returns = [0.002, -0.001, 0.004, 0.0, -0.003, 0.005] * 90
    observed = sharpe_ratio(returns, periods_per_year=1)
    skew, kurtosis = return_moments(returns)

    assert deflated_sharpe_ratio(
        returns, n_trials=1, trial_sharpes=[observed], periods_per_year=1
    ) == pytest.approx(
        probabilistic_sharpe_ratio(observed, 0.0, len(returns), skew, kurtosis),
        abs=1e-12)


def test_the_moments_of_a_symmetric_sample_are_zero_skew_and_gaussian_kurtosis():
    """Pins the moment convention: kurtosis is reported *raw* (3.0 for a normal),
    not excess. The PSR formula's ((γ₄−1)/4) term assumes raw, and feeding excess
    kurtosis in silently understates the tail penalty."""
    symmetric = [-2.0, -1.0, 0.0, 1.0, 2.0] * 200
    skew, kurtosis = return_moments(symmetric)
    assert skew == pytest.approx(0.0, abs=1e-9)
    assert 1.5 < kurtosis < 3.0, f"kurtosis {kurtosis} is not on the raw scale"


def test_dsr_falls_as_the_trial_count_rises():
    """The same strategy, the same returns, more things tried before settling on
    it - and less reason to believe it. This is the property that makes DSR worth
    using as in-loop fitness."""
    returns = [0.003, -0.001, 0.002, 0.004, -0.002] * 100
    trials = [0.2, 0.5, 0.9, 1.4, 1.1, 0.3]
    few = deflated_sharpe_ratio(returns, n_trials=5, trial_sharpes=trials,
                                periods_per_year=1)
    many = deflated_sharpe_ratio(returns, n_trials=5000, trial_sharpes=trials,
                                 periods_per_year=1)
    assert many < few


def test_dsr_rejects_the_best_of_many_noise_strategies(tmp_path):
    """The anchor case from ledger VX-008, reproduced in miniature: mine pure
    noise with many configurations, take the luckiest, and the gate must not
    pass it. A raw-Sharpe search would rank this candidate first."""
    import random
    rng = random.Random(20260808)
    n_configs = 400
    per_config = [[rng.gauss(0.0, 0.01) for _ in range(500)]
                  for _ in range(n_configs)]
    sharpes = [sharpe_ratio(r, periods_per_year=1) for r in per_config]
    best_index = max(range(n_configs), key=lambda i: sharpes[i])

    assert sharpes[best_index] > 0.10, "the mining did not find a lucky winner"

    dsr = deflated_sharpe_ratio(per_config[best_index], n_trials=n_configs,
                                trial_sharpes=sharpes, periods_per_year=1)
    assert dsr < 0.95, f"a noise-mined winner passed the gate with DSR {dsr}"


def test_dsr_is_a_probability():
    returns = [0.002, -0.001, 0.003, 0.001, -0.002] * 100
    dsr = deflated_sharpe_ratio(returns, n_trials=50,
                                trial_sharpes=[0.1, 0.9, 0.4], periods_per_year=1)
    assert 0.0 <= dsr <= 1.0


def test_dsr_refuses_when_no_trial_sharpes_are_supplied():
    """The trial variance has to come from the trials actually run. Substituting
    an assumed variance is how a deflation gate becomes decoration - and the
    registry exists precisely so the real ones are available."""
    with pytest.raises(NotEnoughTrials):
        deflated_sharpe_ratio([0.001] * 200, n_trials=100, trial_sharpes=[],
                              periods_per_year=1)


# --- MinBTL -----------------------------------------------------------------

def test_min_backtest_length_matches_the_documented_45_variant_case():
    """The ledger's stated consequence, used as the arithmetic check: at N=45 an
    in-sample Sharpe of 1.0 needs ~7.6 years, so five years of daily data cannot
    support the claim. 2·ln(45)/1.0² = 7.61."""
    assert min_backtest_length_years(
        n_trials=45, target_sharpe=1.0) == pytest.approx(7.612, abs=0.01)


def test_min_backtest_length_grows_with_the_number_of_trials():
    assert (min_backtest_length_years(45, 1.0)
            < min_backtest_length_years(10_000, 1.0))


def test_a_higher_claimed_sharpe_needs_less_data_to_be_credible():
    """A Sharpe of 2 is harder for noise to fake than a Sharpe of 1, so it clears
    the bar with a shorter record. The relationship is inverse-square."""
    assert min_backtest_length_years(1000, 2.0) < min_backtest_length_years(1000, 1.0)
    assert min_backtest_length_years(1000, 2.0) == pytest.approx(
        min_backtest_length_years(1000, 1.0) / 4.0, rel=1e-9)


def test_min_backtest_length_needs_a_positive_target_sharpe():
    """A claim of zero Sharpe needs infinite data to distinguish from noise, which
    is a refusal rather than a number."""
    with pytest.raises(ValueError):
        min_backtest_length_years(n_trials=100, target_sharpe=0.0)


def test_a_single_trial_still_needs_some_data():
    """N=1 gives ln(1)=0 and so a zero-year requirement, which is wrong: one
    backtest on one week of data is not credible either. The floor exists so the
    gate cannot be passed by a record too short to mean anything."""
    assert min_backtest_length_years(n_trials=1, target_sharpe=1.0) > 0.0
