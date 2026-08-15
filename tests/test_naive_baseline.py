"""MD-001 and VX-121 — the cheapest overfitting check that exists.

`FEATURES.md` §3, marked MISSED: *"Linear / naive baseline, mandatory — every
model must beat it before promotion."* Ledger VX-121 names the specific failure
it catches: *"Compare a trained model's held-out MSE against a naive 'predict
last observed price' baseline; if within noise, the model reproduced the trap."*

**The trap.** A model trained to predict a price level can reach a spectacular
R² and a tiny MSE by learning to output approximately the last price it saw. The
error looks small because prices are persistent, not because anything was
learned. Plotted, the forecast tracks the actual beautifully and one bar late.
Whole projects have shipped on that chart.

The only defence is a benchmark that does exactly the same trick honestly — the
random walk — and a test of whether the difference is real rather than noise.

Two distinct failure modes, reported apart, because they call for different
work:

  * **the lag-one trap** — the model's forecasts are a hair from the naive one.
    It learned to copy. Nothing about the features matters yet.
  * **different but no better** — the model forecasts something genuinely its
    own and is not measurably better. The features may be wrong, or the horizon.

Collapsing both into "failed" loses the diagnosis, and the diagnosis is the
whole value of running the cheapest check first.
"""
import math

import pytest

from models.naive_baseline import (
    ForecastLengthMismatch,
    judge_against_naive_baseline,
    random_walk_forecast,
    squared_losses,
)


def _drifting_prices(n=240, start=100.0, step=0.5):
    """A deterministic zig-zag with drift. No RNG: `Math.random` in a test is a
    test whose failures cannot be reproduced."""
    prices, price = [], start
    for i in range(n):
        price += step if i % 3 else -step * 0.8
        prices.append(price)
    return prices


# --- the naive forecast itself ----------------------------------------------

def test_the_random_walk_predicts_the_last_observed_price():
    assert random_walk_forecast([10.0, 11.0, 12.0]) == [10.0, 11.0]


def test_the_naive_forecast_is_one_shorter_than_the_series_it_forecasts():
    """There is no forecast for the first observation - nothing preceded it.
    An implementation that returned an equal-length list has invented one."""
    prices = _drifting_prices(50)
    assert len(random_walk_forecast(prices)) == len(prices) - 1


def test_a_series_too_short_to_forecast_yields_nothing():
    assert random_walk_forecast([10.0]) == []


def test_squared_losses_are_per_observation_not_summarised():
    assert squared_losses([1.0, 2.0], [1.5, 2.0]) == [0.25, 0.0]


# --- the trap, caught -------------------------------------------------------

def test_a_model_that_merely_copies_the_last_price_does_not_pass():
    """The lag-one trap in its purest form: forecasts identical to the naive
    benchmark. Its MSE is exactly the benchmark's, which on a persistent series
    looks impressively small in isolation."""
    prices = _drifting_prices()
    actual = prices[1:]
    verdict = judge_against_naive_baseline(actual, random_walk_forecast(prices))
    assert verdict.beats_naive is False


def test_an_exact_copy_is_named_as_the_lag_one_trap_not_merely_as_no_better():
    prices = _drifting_prices()
    verdict = judge_against_naive_baseline(prices[1:], random_walk_forecast(prices))
    assert verdict.reproduced_lag_one_trap is True
    assert "lag-one" in verdict.detail.lower()


def test_a_near_copy_is_still_caught_as_the_trap():
    """Nobody ships an exact copy. They ship a model that has effectively
    learned one, whose forecasts sit a fraction of a tick from the naive."""
    prices = _drifting_prices()
    naive = random_walk_forecast(prices)
    sneaky = [p * 1.000001 for p in naive]
    verdict = judge_against_naive_baseline(prices[1:], sneaky)
    assert verdict.beats_naive is False
    assert verdict.reproduced_lag_one_trap is True


def test_a_model_that_is_different_and_no_better_is_not_called_the_trap():
    """A genuinely different forecast that fails is a different diagnosis: the
    features or the horizon are wrong, not that it learned to copy."""
    prices = _drifting_prices()
    naive = random_walk_forecast(prices)
    # Off by a wandering amount, so it is nobody's copy and no better either.
    wrong = [p + (2.0 if i % 2 else -2.0) for i, p in enumerate(naive)]
    verdict = judge_against_naive_baseline(prices[1:], wrong)
    assert verdict.beats_naive is False
    assert verdict.reproduced_lag_one_trap is False


# --- a model with real skill passes -----------------------------------------

def test_a_model_that_genuinely_forecasts_better_passes():
    """Given the actual next value with a little noise, the model should clear
    the benchmark. If this fails, the test has no power and every pass above is
    meaningless."""
    prices = _drifting_prices()
    actual = prices[1:]
    skilled = [a + (0.02 if i % 2 else -0.02) for i, a in enumerate(actual)]
    verdict = judge_against_naive_baseline(actual, skilled)
    assert verdict.beats_naive is True
    assert verdict.p_value < 0.05


def test_the_verdict_carries_both_mean_losses_so_it_can_be_argued_with():
    prices = _drifting_prices()
    actual = prices[1:]
    skilled = [a + 0.01 for a in actual]
    verdict = judge_against_naive_baseline(actual, skilled)
    assert verdict.mean_loss_model < verdict.mean_loss_naive
    assert verdict.n_observations == len(actual)


def test_the_loss_reduction_is_reported_as_a_share_of_the_benchmark():
    prices = _drifting_prices()
    actual = prices[1:]
    skilled = [a + 0.01 for a in actual]
    verdict = judge_against_naive_baseline(actual, skilled)
    expected = (1 - verdict.mean_loss_model / verdict.mean_loss_naive) * 100
    assert math.isclose(verdict.loss_reduction_pct, expected, rel_tol=1e-9)


# --- refusals over quiet passes ---------------------------------------------

def test_a_mismatched_forecast_length_is_refused_not_trimmed():
    """Trimming to the shorter would hide that the two were never aligned, and
    an unaligned comparison is not a weaker comparison - it is a different one."""
    prices = _drifting_prices(60)
    with pytest.raises(ForecastLengthMismatch):
        judge_against_naive_baseline(prices, prices[:-1])


def test_too_few_observations_refuses_rather_than_reporting_a_pass():
    """Inherited from the SPA machinery, deliberately: a gate that cannot test
    must not report a pass. This is the fail-open inversion the corpus records
    four separate systems getting wrong in the flattering direction."""
    from validation.superior_predictive_ability import NotEnoughPaths
    with pytest.raises(NotEnoughPaths):
        judge_against_naive_baseline([1.0, 2.0] * 3, [1.0, 2.0] * 3)


def test_the_minimum_sample_matches_the_test_that_enforces_it():
    """Two constants for one requirement drift, and the one that drifts is the
    one nobody re-derived. Pinned so a change to either is a failing test."""
    from validation.superior_predictive_ability import _MIN_PATHS
    from models.naive_baseline import MIN_OBSERVATIONS
    assert MIN_OBSERVATIONS == _MIN_PATHS


def test_a_flat_series_with_nothing_to_forecast_is_refused(monkeypatch):
    """Zero benchmark loss means the naive forecast was perfect, and the
    percentage reduction against it is a division by zero. Refused rather than
    reported as an infinite improvement."""
    from models.naive_baseline import DegenerateBenchmark
    flat = [100.0] * 60
    with pytest.raises(DegenerateBenchmark):
        judge_against_naive_baseline(flat[1:], random_walk_forecast(flat))


# --- the comparison is reproducible -----------------------------------------

def test_the_same_inputs_give_the_same_verdict():
    """The bootstrap is seeded. A gate whose answer moves between runs is one
    people re-run until it agrees with them."""
    prices = _drifting_prices()
    actual = prices[1:]
    model = [a + 0.01 for a in actual]
    first = judge_against_naive_baseline(actual, model)
    second = judge_against_naive_baseline(actual, model)
    assert first.p_value == second.p_value
