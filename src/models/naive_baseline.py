"""The benchmark every model has to beat, and the trap it exists to catch.

`FEATURES.md` §3 marks this MISSED: *"Linear / naive baseline, mandatory — every
model must beat it before promotion. Cheapest overfitting check that exists."*
Ledger MD-001 carries the same requirement; ledger VX-121 names the specific
failure — *"compare a trained model's held-out MSE against a naive 'predict last
observed price' baseline; if within noise, the model reproduced the trap."*

**The trap.** A model trained to predict a price *level* can reach a superb R²
and a tiny MSE by learning to emit approximately the last price it saw. The error
is small because prices are persistent, not because anything was learned. Plotted
against the actual series it looks near-perfect, one bar late. The chart is
convincing and the model is worthless, and it is worthless in the direction that
gets it promoted.

The defence is a benchmark doing that trick honestly — the random walk — plus a
test of whether any difference is real. A raw MSE comparison is not enough: on
20,000 observations a 0.2% improvement can be pure sampling noise, and reporting
it as "beat the baseline" is how the trap survives the check meant to catch it.

**The significance test is not reimplemented here.** `validation.
superior_predictive_ability` already runs a studentised stationary-bootstrap test
on a performance differential, and it already preserves the serial dependence
that makes a standard error honest on time-series data. Comparing one model
against one benchmark is that test with a single challenger. A second statistical
machinery for the same question is two things to keep in agreement, and the one
that drifts is always the one nobody re-derived.

Two failure modes, reported apart, because they call for different work:

  * `reproduced_lag_one_trap` — the forecasts sit a hair from the naive ones. The
    model learned to copy, and nothing about the features has been tested yet.
  * beaten but not a copy — a genuinely independent forecast that is not
    measurably better. The features or the horizon are wrong.

Collapsing both into "failed" throws away the diagnosis, which is most of what
running the cheapest check first buys you.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from validation.superior_predictive_ability import (
    NotEnoughPaths, superior_predictive_ability,
)

# How close a model's forecasts must sit to the naive ones, relative to the
# movement the naive forecast itself is trying to track, before this is called a
# copy rather than an independent forecast that happened to fail.
#
# Measured against the thing that matters: the mean absolute step of the series.
# An absolute tolerance would be meaningless across instruments - a hair on
# BTCUSDT is a mile on a token priced in cents.
_COPY_FRACTION_OF_STEP = 0.05

# Mirrors `superior_predictive_ability`'s own minimum. Checked HERE, before
# anything is computed, so that too-little-data is diagnosed ahead of every other
# complaint the input could raise: on five flat observations both "too few to
# test" and "the benchmark never moves" are true, and only the first tells the
# caller what to do about it. `test_naive_baseline` pins this equal to SPA's
# constant, so the two cannot drift apart unnoticed.
MIN_OBSERVATIONS = 20


class ForecastLengthMismatch(ValueError):
    """Model forecasts and actuals are not aligned.

    Refused rather than trimmed to the shorter. An unaligned comparison is not a
    weaker comparison, it is a different one, and truncating hides that it was
    never valid - the same reasoning `superior_predictive_ability` applies to
    candidates scored over different periods.
    """


class DegenerateBenchmark(ValueError):
    """The naive forecast had zero loss, so there is nothing to improve on.

    A perfectly flat series. The percentage improvement over it is a division by
    zero, and reporting that as an infinite improvement would turn the most
    trivial possible input into the most impressive possible result.
    """


@dataclass(frozen=True)
class BaselineVerdict:
    """Whether the model beat the random walk, and the evidence either way.

    `mean_loss_model` and `mean_loss_naive` are both carried, for the reason
    `promotion_gate.GateResult` carries `measured` and `threshold`: a verdict that
    reports only pass or fail cannot be argued with, and one nobody can audit is
    one nobody will act on.
    """

    beats_naive: bool
    p_value: float
    test_statistic: float
    mean_loss_model: float
    mean_loss_naive: float
    loss_reduction_pct: float
    mean_abs_deviation_from_naive: float
    reproduced_lag_one_trap: bool
    n_observations: int
    detail: str


def random_walk_forecast(prices: list[float]) -> list[float]:
    """The honest version of the trap: tomorrow's forecast is today's price.

    Returns one fewer value than it was given, and that is the contract rather
    than an off-by-one: nothing precedes the first observation, so there is no
    forecast for it. An implementation returning an equal-length list has
    invented one, and the invented value lands at the start of every comparison
    made afterwards.
    """
    return list(prices[:-1])


def squared_losses(actual: list[float], predicted: list[float]) -> list[float]:
    """Per-observation squared error. Not summarised, because the significance
    test needs the series - a mean has already thrown away the dependence
    structure the bootstrap exists to preserve."""
    if len(actual) != len(predicted):
        raise ForecastLengthMismatch(
            f"{len(actual)} actual(s) against {len(predicted)} prediction(s)")
    return [(a - p) ** 2 for a, p in zip(actual, predicted)]


def judge_against_naive_baseline(
    actual: list[float],
    model_forecast: list[float],
    naive_forecast: list[float] | None = None,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> BaselineVerdict:
    """Did this model beat 'predict the last price', beyond sampling noise?

    `naive_forecast` defaults to the random walk implied by `actual` itself -
    each observation's own predecessor - which is the benchmark VX-121 names.
    Supplying one explicitly is for the case where the model forecasts something
    other than the level it was fed.

    Raises `NotEnoughPaths` from the SPA machinery when there are too few
    observations to test. Refused rather than passed: a gate that cannot test
    must not report a pass, which is the fail-open inversion this corpus records
    four separate prior systems getting wrong in the flattering direction.
    """
    if len(actual) < MIN_OBSERVATIONS:
        raise NotEnoughPaths(
            f"need >= {MIN_OBSERVATIONS} observations to test a model against "
            f"the random walk, got {len(actual)}. Refused rather than passed: a "
            f"gate that cannot test must not report a pass")
    if len(actual) != len(model_forecast):
        raise ForecastLengthMismatch(
            f"{len(actual)} actual(s) against {len(model_forecast)} model "
            f"forecast(s). Refused rather than trimmed - an unaligned comparison "
            f"is a different comparison, not a weaker one")

    if naive_forecast is None:
        # actual[i-1] is the last value observed before actual[i]. The first
        # observation has no predecessor, so it carries its own value and
        # contributes zero loss to BOTH sides - it cannot tilt the comparison.
        naive_forecast = [actual[0]] + list(actual[:-1])
    if len(naive_forecast) != len(actual):
        raise ForecastLengthMismatch(
            f"{len(actual)} actual(s) against {len(naive_forecast)} naive "
            f"forecast(s)")

    model_losses = squared_losses(actual, model_forecast)
    naive_losses = squared_losses(actual, naive_forecast)
    mean_model, mean_naive = mean(model_losses), mean(naive_losses)

    if mean_naive <= 0.0:
        raise DegenerateBenchmark(
            "the naive forecast has zero mean loss - the series does not move, "
            "so there is nothing to forecast and no improvement to measure. "
            "Refused rather than reported as an infinite improvement")

    # Higher is better for SPA, so performance is negative loss. d_t is then
    # naive_loss - model_loss: positive exactly when the model did better.
    result = superior_predictive_ability(
        incumbent=[-loss for loss in naive_losses],
        challengers={"model": [-loss for loss in model_losses]},
        n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)

    deviation = mean(abs(m - n) for m, n in zip(model_forecast, naive_forecast))
    # Scaled by how far the series actually moves. An absolute tolerance is
    # meaningless across instruments priced decades apart.
    typical_step = mean(abs(a - n) for a, n in zip(actual, naive_forecast))
    is_copy = (typical_step > 0
               and deviation <= _COPY_FRACTION_OF_STEP * typical_step)

    beats = "model" in result.superior
    reduction = (1 - mean_model / mean_naive) * 100

    if beats:
        detail = (f"beat the random walk: mean squared loss {mean_model:.6g} "
                  f"against {mean_naive:.6g}, {reduction:.2f}% lower, "
                  f"p={result.p_value:.4f} over {len(actual)} observations")
    elif is_copy:
        detail = (f"REPRODUCED THE LAG-ONE TRAP: forecasts sit "
                  f"{deviation:.6g} from the naive benchmark against a typical "
                  f"move of {typical_step:.6g}, so the model learned to emit the "
                  f"last price rather than to forecast. Its {mean_model:.6g} mean "
                  f"squared loss looks small because prices persist, not because "
                  f"anything was learned (p={result.p_value:.4f})")
    else:
        detail = (f"did not beat the random walk: mean squared loss "
                  f"{mean_model:.6g} against {mean_naive:.6g}, p="
                  f"{result.p_value:.4f}. The forecast is its own - "
                  f"{deviation:.6g} from the naive against a typical move of "
                  f"{typical_step:.6g} - so this is a features or horizon "
                  f"problem, not a copied benchmark")

    return BaselineVerdict(
        beats_naive=beats,
        p_value=result.p_value,
        test_statistic=result.test_statistic,
        mean_loss_model=mean_model,
        mean_loss_naive=mean_naive,
        loss_reduction_pct=reduction,
        mean_abs_deviation_from_naive=deviation,
        # A model that beat the benchmark did not copy it, whatever the distance
        # says. Reporting both would be a contradiction on the same verdict.
        reproduced_lag_one_trap=is_copy and not beats,
        n_observations=len(actual),
        detail=detail)
