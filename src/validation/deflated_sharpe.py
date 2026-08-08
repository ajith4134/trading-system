"""Deflated Sharpe as the search loop's fitness, and the MinBTL hard gate.

`FEATURES.md` §8 requires the Deflated Sharpe **as in-loop fitness, not a report
on the winner**. That distinction is the whole point. A search that ranks by raw
Sharpe climbs toward whichever candidate got luckiest, and deflating at the end
only tells you afterwards that it did. Deflating inside the loop removes the
incentive: a candidate that is only ahead because more things were tried does not
score ahead.

The empirical anchor (ledger VX-008): on **pure noise**, 8,800 configurations
produced an in-sample Sharpe of **1.27**, with 53% of out-of-sample Sharpes
negative. And `DECISIONS.md` §5: five years of daily data, 45+ variations, and
the selected best is more likely than not to have a true OOS Sharpe of zero.

Both formulas are Bailey & López de Prado's:

    PSR(SR*) = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]

    SR₀ = √V[SR] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
    DSR = PSR(SR₀)

γ is Euler–Mascheroni, V[SR] the cross-sectional variance of the Sharpes actually
tried, N the cumulative trial count from `trial_registry`.

    MinBTL ≈ 2·ln(N) / SR²   (years)

**Two things this module refuses to do**, both because the corpus already shipped
them wrong:

*It will not accept a trial count below one.* `nse-botonly`'s
`combinatorial_purged_cross_validation.py` passed `number_of_strategy_trials =
number_of_paths` - the count of CPCV resample paths of a *single* strategy, C(6,2)
= 15. A candidate selected from thousands was deflated as though 15 things had
been tried. The gate read as rigorous and was close to toothless. A wrong N is
not a smaller correction; it is a false one.

*It will not invent a trial variance.* V[SR] comes from the trials that really
ran. With no trials supplied it raises, because an assumed dispersion turns the
deflation into decoration.

**Verification status, stated plainly:** these are implemented from the published
form of the formulas and pinned by properties in `tests/test_deflated_sharpe.py` -
the N=1 PSR identity, monotonicity in N, the documented 45-variant MinBTL
arithmetic, and rejection of a noise-mined winner. No worked numeric example from
the papers was re-derived against this code.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

# Euler-Mascheroni. Appears in the expected-maximum-of-N-draws approximation.
_EULER_MASCHERONI = 0.5772156649015329

# MinBTL floor. ln(1) = 0, so a literal reading gives a zero-year requirement for
# a single backtest - and one backtest on one week of data is not credible either.
# Half a year is the shortest record this system will call evidence of anything.
_MIN_BACKTEST_FLOOR_YEARS = 0.5

_TRADING_DAYS_PER_YEAR = 252


class NotEnoughTrials(ValueError):
    """The trial count or trial dispersion needed for deflation is not available.

    A refusal rather than a default, because every default here makes the gate
    weaker, and a weaker gate is invisible until money is lost.
    """


def sharpe_ratio(returns: list[float], periods_per_year: int = 1) -> float:
    """Annualised Sharpe of a return series.

    Raises on zero variance rather than returning an enormous number. A constant
    return series is a broken or leaked backtest, and the flattering answer would
    rank it first in the search.
    """
    if len(returns) < 2:
        raise ValueError(f"need >= 2 returns for a Sharpe, got {len(returns)}")
    dispersion = pstdev(returns)
    if dispersion == 0.0:
        raise ZeroDivisionError(
            "return series has zero variance, so its Sharpe is undefined - this "
            "is a broken or leaked backtest, not an infinitely good strategy")
    return (mean(returns) / dispersion) * math.sqrt(periods_per_year)


def return_moments(returns: list[float]) -> tuple[float, float]:
    """(skew, kurtosis) of a return series, kurtosis on the **raw** scale.

    Raw, not excess: the PSR's ((γ₄−1)/4) term is written for raw kurtosis, where
    a normal sample is 3.0. Passing excess kurtosis silently shrinks the tail
    penalty, which is the wrong direction for a gate.
    """
    n = len(returns)
    if n < 2:
        raise ValueError(f"need >= 2 returns for moments, got {n}")
    mu = mean(returns)
    sigma = pstdev(returns)
    if sigma == 0.0:
        raise ZeroDivisionError("cannot compute moments of a zero-variance series")
    skew = sum(((r - mu) / sigma) ** 3 for r in returns) / n
    kurtosis = sum(((r - mu) / sigma) ** 4 for r in returns) / n
    return skew, kurtosis


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _standard_normal_ppf(p: float) -> float:
    """Inverse standard normal CDF, via Acklam's rational approximation.

    Written out rather than pulled from scipy because this is the only scipy
    function the validation stack would need, and the approximation's error
    (~1e-9 in the relevant range) is far below the uncertainty in the inputs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def probabilistic_sharpe_ratio(observed_sharpe: float, benchmark_sharpe: float,
                               n_observations: int, skew: float,
                               kurtosis: float) -> float:
    """Probability the true Sharpe exceeds `benchmark_sharpe`.

    The higher moments are in the denominator on purpose: a Sharpe earned by
    selling tails is weaker evidence than the same number earned symmetrically,
    and a gate blind to that passes short-vol strategies right up until it
    doesn't.
    """
    if n_observations < 2:
        raise ValueError(f"need >= 2 observations, got {n_observations}")
    variance_term = (1.0
                     - skew * observed_sharpe
                     + ((kurtosis - 1.0) / 4.0) * observed_sharpe ** 2)
    if variance_term <= 0.0:
        raise ValueError(
            f"the Sharpe estimator's variance term is non-positive "
            f"({variance_term:.4f}) at SR={observed_sharpe}, skew={skew}, "
            f"kurtosis={kurtosis} - the moments are inconsistent with the Sharpe "
            f"and no probability can be computed from them")
    z = ((observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1)
         / math.sqrt(variance_term))
    return _standard_normal_cdf(z)


def expected_max_sharpe_under_null(n_trials: int, trial_variance: float) -> float:
    """The Sharpe the *best of N* worthless strategies is expected to show.

    This is the hurdle a candidate has to clear to be interesting. It rises with
    N - the mechanism the whole module exists for - and collapses to zero at N=1,
    because one look at the data is not multiple testing.

    It is also zero when the trials never differed: no dispersion means selection
    had nothing to select on, so it conferred no advantage to deflate away.
    """
    if n_trials < 1:
        raise NotEnoughTrials(
            f"n_trials must be >= 1, got {n_trials}. A trial count below one is "
            f"not a gentler correction, it is a broken caller - and the count "
            f"must be the number of candidates evaluated (trial_registry's "
            f"cumulative N), never the number of resample paths of one candidate")
    if trial_variance < 0.0:
        raise ValueError(f"trial variance cannot be negative, got {trial_variance}")
    if n_trials == 1 or trial_variance == 0.0:
        return 0.0

    gamma = _EULER_MASCHERONI
    term_a = _standard_normal_ppf(1.0 - 1.0 / n_trials)
    term_b = _standard_normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(trial_variance) * ((1.0 - gamma) * term_a + gamma * term_b)


def deflated_sharpe_ratio(returns: list[float], n_trials: int,
                          trial_sharpes: list[float],
                          periods_per_year: int = 1) -> float:
    """Probability this strategy's Sharpe survives the fact that N were tried.

    Use as the search's fitness. `n_trials` is the registry's cumulative N and
    `trial_sharpes` the Sharpes those trials produced - both from the same
    `TrialRegistry`, so that the count and the dispersion describe the same set
    of looks at the data.
    """
    if not trial_sharpes:
        raise NotEnoughTrials(
            "no trial Sharpes supplied, so V[SR] would have to be assumed. An "
            "assumed dispersion makes the deflation decoration - take them from "
            "TrialRegistry.trial_sharpes()")
    observed = sharpe_ratio(returns, periods_per_year=periods_per_year)
    skew, kurtosis = return_moments(returns)
    variance = pstdev(trial_sharpes) ** 2 if len(trial_sharpes) > 1 else 0.0
    hurdle = expected_max_sharpe_under_null(n_trials, variance)
    return probabilistic_sharpe_ratio(observed, hurdle, len(returns),
                                      skew, kurtosis)


def min_backtest_length_years(n_trials: int, target_sharpe: float) -> float:
    """Years of data needed before an in-sample Sharpe of this size means anything.

    MinBTL ≈ 2·ln(N) / SR². Cheap, closed-form, and a hard gate: below this much
    data, an in-sample Sharpe that large is *expected* from noise alone after N
    trials, so it carries no information.

    The documented consequence, which is also the arithmetic check on this code:
    at N=45 and SR=1.0 the requirement is 7.6 years, so five years of daily data
    cannot support a Sharpe-1.0 claim after 45 configurations.
    """
    if n_trials < 1:
        raise NotEnoughTrials(f"n_trials must be >= 1, got {n_trials}")
    if target_sharpe <= 0.0:
        raise ValueError(
            f"target_sharpe must be positive, got {target_sharpe}. A claim of "
            f"zero Sharpe needs unbounded data to separate from noise, which is "
            f"a refusal rather than a number")
    # The floor is not cosmetic: at N=1, ln(1)=0 gives a zero-year requirement,
    # which would let a one-week backtest pass a gate about data sufficiency.
    return max(_MIN_BACKTEST_FLOOR_YEARS,
               2.0 * math.log(n_trials) / (target_sharpe ** 2))


def has_enough_history(n_trials: int, target_sharpe: float,
                       n_observations: int,
                       periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> bool:
    """Whether the record is long enough for this Sharpe claim at this N."""
    return (n_observations / periods_per_year
            >= min_backtest_length_years(n_trials, target_sharpe))
