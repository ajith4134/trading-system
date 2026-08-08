"""Bootstrapped max-drawdown distribution — VX-011, and the ladder it anchors.

`FEATURES.md` §8 wants circuit-breaker thresholds that are *"non-arbitrary"*. The
corpus is blunt about why the usual practice is not: *"Most industry thresholds are
round numbers and committee judgment"* (`allocation-and-regime.md`). And VX-119:
**the backtest's max drawdown is a lower bound, not an expectation** - it is one
draw from a distribution, and the single worst thing that happened to happen.

The prescription is specific and the reason is mechanical:

> *"Bootstrap the max-drawdown DISTRIBUTION (**block bootstrap to preserve
> autocorrelation**) rather than quoting one realized worst case. A few dozen
> lines, cheap, directly actionable, **underused relative to its value** - and it
> is what makes circuit-breaker thresholds non-arbitrary."*

**Why the parenthesis is the whole thing.** Drawdowns are created by *runs* of
losses. An i.i.d. bootstrap resamples single returns, which destroys exactly the
serial correlation that produces runs, so it reports a shallower drawdown
distribution than the data supports - and it does so while looking like a rigorous
Monte Carlo. Underestimating the drawdown budget is the direction that gets an
account closed, so that failure mode is tested directly below.

Prior art, checked: the nearest relative is RX-027, a 1,000-scenario forward Monte
Carlo of P(20% drawdown) that soft-reduces concurrent trades. Different mechanism -
it simulates forward from assumed trade statistics rather than resampling realized
history - so it is not a donor for this. OGE-028 records the bootstrapped
circuit-breaker calibration as *"design only - not found as working code anywhere"*.
"""
import pytest

from risk.drawdown_distribution import (
    CircuitBreakerLadder,
    NotEnoughHistory,
    block_bootstrap_max_drawdowns,
    circuit_breaker_ladder,
    max_drawdown,
)


def autocorrelated_returns(n=1500, phi=0.35, scale=0.01, seed=7):
    """AR(1) returns. Positive phi produces streaks, which produce drawdowns."""
    import random
    rng = random.Random(seed)
    out, previous = [], 0.0
    for _ in range(n):
        shock = rng.gauss(0.0, scale)
        previous = phi * previous + shock
        out.append(previous)
    return out


# --- the measurement itself -------------------------------------------------

def test_a_monotonically_rising_curve_has_no_drawdown():
    assert max_drawdown([0.01] * 50) == pytest.approx(0.0)


def test_max_drawdown_is_the_worst_peak_to_trough_not_the_last_decline():
    """A later, shallower dip must not overwrite a deeper earlier one. Tracking
    only the running decline is the classic version of this bug."""
    returns = [0.5, -0.5, 0.5, -0.1]      # deep dip first, shallow one after
    assert max_drawdown(returns) == pytest.approx(0.5)


def test_max_drawdown_compounds_rather_than_sums():
    """Equity is multiplicative. Summing returns overstates recovery after a large
    loss - down 50% then up 50% is not flat, it is down 25%."""
    assert max_drawdown([-0.5, 0.5]) == pytest.approx(0.5)


def test_max_drawdown_is_a_positive_fraction():
    dd = max_drawdown(autocorrelated_returns())
    assert 0.0 < dd < 1.0


def test_max_drawdown_of_a_total_loss_is_one():
    assert max_drawdown([-1.0]) == pytest.approx(1.0)


# --- the reason block bootstrap exists --------------------------------------

def test_an_iid_bootstrap_understates_the_drawdown_of_autocorrelated_returns():
    """**The load-bearing test.** Drawdowns come from runs of losses. Resampling
    single returns destroys the serial correlation that produces runs, so the i.i.d.
    distribution is shallower than the data supports - while still looking like a
    rigorous Monte Carlo. Underestimating a drawdown budget is the direction that
    closes an account."""
    returns = autocorrelated_returns(phi=0.4)
    blocked = block_bootstrap_max_drawdowns(returns, n_samples=400, block_size=50,
                                            seed=1)
    iid = block_bootstrap_max_drawdowns(returns, n_samples=400, block_size=1,
                                        seed=1)
    blocked_p90 = sorted(blocked)[int(0.9 * len(blocked))]
    iid_p90 = sorted(iid)[int(0.9 * len(iid))]

    assert iid_p90 < blocked_p90, (
        f"the i.i.d. bootstrap ({iid_p90:.4f}) did not understate the block "
        f"bootstrap ({blocked_p90:.4f}) - either the block resampling is not "
        f"preserving autocorrelation, or the test data has none")


def test_the_realized_drawdown_sits_below_the_bootstrap_upper_percentiles():
    """VX-119's point, made measurable: the realized worst case is one draw. Sizing
    a kill-switch budget to it is sizing to the median of a distribution whose
    upper tail is where the account dies."""
    returns = autocorrelated_returns()
    realized = max_drawdown(returns)
    samples = sorted(block_bootstrap_max_drawdowns(returns, n_samples=500,
                                                   block_size=50, seed=3))
    p90 = samples[int(0.9 * len(samples))]
    assert realized < p90, (
        f"realized {realized:.4f} was not below the bootstrap p90 {p90:.4f}")


def test_the_bootstrap_is_deterministic_for_a_given_seed():
    """A drawdown budget that changes between runs cannot be a threshold."""
    returns = autocorrelated_returns()
    assert (block_bootstrap_max_drawdowns(returns, n_samples=50, block_size=20, seed=9)
            == block_bootstrap_max_drawdowns(returns, n_samples=50, block_size=20, seed=9))


def test_different_seeds_give_different_draws():
    returns = autocorrelated_returns()
    a = block_bootstrap_max_drawdowns(returns, n_samples=50, block_size=20, seed=1)
    b = block_bootstrap_max_drawdowns(returns, n_samples=50, block_size=20, seed=2)
    assert a != b


def test_the_stationary_bootstrap_is_available_and_also_beats_iid():
    """Politis & Romano's stationary bootstrap draws geometric block lengths, so
    the resampled series is stationary by construction rather than carrying the
    fixed-block scheme's periodicity artefacts. VX-152 names both; both must
    preserve the autocorrelation that matters."""
    returns = autocorrelated_returns(phi=0.4)
    stationary = block_bootstrap_max_drawdowns(returns, n_samples=400, block_size=50,
                                               seed=1, method="stationary")
    iid = block_bootstrap_max_drawdowns(returns, n_samples=400, block_size=1, seed=1)
    assert sorted(stationary)[int(0.9 * 400)] > sorted(iid)[int(0.9 * 400)]


def test_an_unknown_bootstrap_method_is_refused():
    with pytest.raises(ValueError):
        block_bootstrap_max_drawdowns([0.01, -0.01] * 100, n_samples=10,
                                      block_size=10, method="magic")


def test_a_block_longer_than_the_series_is_refused():
    """It would return the same series every draw, so the "distribution" would be
    one point repeated - a fixed number wearing a Monte Carlo's clothes."""
    with pytest.raises(ValueError):
        block_bootstrap_max_drawdowns([0.01] * 20, n_samples=10, block_size=50)


def test_too_short_a_history_is_refused_rather_than_bootstrapped():
    """A bootstrap cannot manufacture information the sample does not contain. Fifty
    observations resampled a thousand times is still fifty observations, and the
    confident-looking percentile that comes out is the danger."""
    with pytest.raises(NotEnoughHistory):
        block_bootstrap_max_drawdowns([0.01, -0.02] * 10, n_samples=100,
                                      block_size=5)


# --- the ladder -------------------------------------------------------------

def test_the_ladder_rungs_come_from_measured_percentiles():
    """Rule 8 on a risk limit: each rung names the percentile and the number it
    came from. A threshold whose provenance cannot be stated is committee
    judgement with extra steps."""
    ladder = circuit_breaker_ladder(autocorrelated_returns(), seed=5)
    assert isinstance(ladder, CircuitBreakerLadder)
    for rung in ladder.rungs:
        assert rung.percentile in (0.75, 0.90, 0.99)
        assert rung.drawdown_threshold > 0.0
        assert str(rung.percentile) in rung.provenance
        assert "block bootstrap" in rung.provenance


def test_the_ladder_is_monotonic_in_both_threshold_and_severity():
    """A deeper drawdown must never trigger a milder response. `allocation-and-
    regime.md`'s graduated ladder: -5% cut gross 25%, -10% cut 50%, -15% flat."""
    ladder = circuit_breaker_ladder(autocorrelated_returns(), seed=5)
    thresholds = [r.drawdown_threshold for r in ladder.rungs]
    cuts = [r.gross_reduction for r in ladder.rungs]
    assert thresholds == sorted(thresholds)
    assert cuts == sorted(cuts)


def test_the_deepest_rung_goes_flat():
    """The p99 rung is not a reduction. Past there the assumption the ladder was
    built on is the thing that failed."""
    ladder = circuit_breaker_ladder(autocorrelated_returns(), seed=5)
    assert ladder.rungs[-1].gross_reduction == pytest.approx(1.0)


def test_the_ladder_records_the_realized_drawdown_alongside_its_thresholds():
    """So the gap between "the worst that happened" and "the worst to plan for" is
    visible at the point of decision rather than requiring a second query."""
    returns = autocorrelated_returns()
    ladder = circuit_breaker_ladder(returns, seed=5)
    assert ladder.realized_max_drawdown == pytest.approx(max_drawdown(returns))
    assert ladder.realized_max_drawdown < ladder.rungs[-1].drawdown_threshold


def test_the_ladder_states_how_many_samples_backed_it():
    """A p99 taken from 100 draws is one observation. The count has to travel with
    the number so nobody reads more precision into it than it has."""
    ladder = circuit_breaker_ladder(autocorrelated_returns(), n_samples=1000, seed=5)
    assert ladder.n_bootstrap_samples == 1000


def test_a_percentile_too_extreme_for_the_sample_count_is_refused():
    """p99 from 50 draws interpolates between the top two observations and reads as
    a precise tail estimate. Refused rather than reported."""
    with pytest.raises(ValueError):
        circuit_breaker_ladder(autocorrelated_returns(), n_samples=50, seed=5)
