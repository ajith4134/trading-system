"""Does the selection procedure pick winners at all? PBO, and FDR on promotions.

Two gates that answer questions `deflated_sharpe` cannot.

**PBO** (Bailey, Borwein, López de Prado, Zhu - Combinatorially Symmetric Cross
Validation). DSR asks whether *this candidate's* Sharpe survives N tries. PBO asks
whether *the procedure* has any skill: resample the in-sample/out-of-sample split,
re-run the whole selection each time, and measure how often the config the
procedure chose lands below the out-of-sample median. At PBO ≈ 0.5 the procedure
is choosing at random - and that verdict is independent of how good the chosen
candidate's backtest looks, which is why a DSR gate alone can be passed by a
search that is fundamentally not working.

The anchor (ledger VX-008): 8,800 configurations on **pure noise** produced an
in-sample Sharpe of 1.27 with 53% of OOS Sharpes negative. That case is a test.

**BH-FDR** on the promoted set. The alternative is settled by arithmetic rather
than preference: at N=10,000 Bonferroni's α/N = 5×10⁻⁶ *"suppresses essentially
everything including true positives"* (ledger VX-009). Controlling family-wise
error at that scale switches the search off. Benjamini-Hochberg controls the
*proportion* of promotions that are false, which is the quantity actually worth
bounding when the whole point is to promote several strategies.

Both refuse rather than coerce their inputs. A ragged performance matrix, an odd
split count, a p-value above one - each is a bug upstream, and each would
otherwise reach a promotion decision looking like a number.
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import mean, pstdev


def _sharpe(returns: list[float]) -> float:
    """Per-period Sharpe. Zero-variance segments score zero, not infinity.

    Unlike `deflated_sharpe.sharpe_ratio`, which raises: here a single degenerate
    CSCV segment among hundreds should not abort the whole estimate, and scoring it
    zero keeps it from being selected. The asymmetry is deliberate - a degenerate
    *strategy* is a bug worth stopping for, a degenerate *slice* is not.
    """
    if len(returns) < 2:
        return 0.0
    dispersion = pstdev(returns)
    return 0.0 if dispersion == 0.0 else mean(returns) / dispersion


def probability_of_backtest_overfitting(config_returns: list[list[float]],
                                        n_splits: int = 16) -> float:
    """Probability the selection procedure picks a below-median performer.

    `config_returns` is one return series per configuration, all the same length -
    the whole search's results, not just the finalists' - and `n_splits` the number
    of contiguous groups the period is cut into. CSCV then takes every combination
    of exactly half those groups as in-sample, the complement as out-of-sample,
    picks the in-sample winner, and records where that winner ranks
    out-of-sample.

    Deterministic: the resampling is combinatorial, not random. A PBO that moves
    between runs on the same matrix cannot gate anything.
    """
    n_configs = len(config_returns)
    if n_configs < 2:
        raise ValueError(
            f"PBO needs >= 2 configurations, got {n_configs}. With one config no "
            f"selection happens, so there is no procedure to evaluate")
    if n_splits % 2 != 0:
        raise ValueError(
            f"n_splits must be even, got {n_splits}. CSCV forms the in-sample set "
            f"from exactly half the groups; rounding an odd count would make the "
            f"in-sample and out-of-sample periods different lengths and bias the "
            f"comparison the metric rests on")
    if n_splits < 4:
        raise ValueError(f"n_splits must be >= 4 for a meaningful CSCV, got {n_splits}")

    lengths = {len(r) for r in config_returns}
    if len(lengths) != 1:
        raise ValueError(
            f"all configurations must have the same number of periods, got "
            f"{sorted(lengths)} - a ragged matrix means some configs saw more data "
            f"than others, and the in-sample winner may just be the one that saw "
            f"the most")
    n_periods = lengths.pop()
    if n_periods < n_splits * 2:
        raise ValueError(
            f"need >= {n_splits * 2} periods for {n_splits} splits, got {n_periods}")

    base, remainder = divmod(n_periods, n_splits)
    bounds, cursor = [], 0
    for index in range(n_splits):
        size = base + (1 if index < remainder else 0)
        bounds.append((cursor, cursor + size))
        cursor += size

    half = n_splits // 2
    below_median = 0
    total = 0
    for in_sample_groups in combinations(range(n_splits), half):
        in_set = set(in_sample_groups)
        out_groups = [g for g in range(n_splits) if g not in in_set]

        def concat(groups, series):
            return [series[i] for g in groups for i in range(*bounds[g])]

        in_scores = [_sharpe(concat(in_sample_groups, s)) for s in config_returns]
        out_scores = [_sharpe(concat(out_groups, s)) for s in config_returns]

        chosen = max(range(n_configs), key=lambda i: in_scores[i])
        # Relative rank of the chosen config among all configs' OOS scores.
        # Bailey et al. take the logit of this and measure P(logit < 0); the logit
        # is monotone in the rank, so counting rank below the midpoint is the same
        # test without the numerical trouble at the ends.
        n_worse = sum(1 for score in out_scores if score < out_scores[chosen])
        relative_rank = (n_worse + 1) / (n_configs + 1)
        if relative_rank <= 0.5:
            below_median += 1
        total += 1

    return below_median / total


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """α/N. Provided so the comparison against BH is computable, not as a gate.

    At N=10,000 this is 5×10⁻⁶. The ledger's verdict stands: that suppresses
    essentially everything, including true positives.
    """
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha / n_tests


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[int]:
    """Indices of the hypotheses rejected while holding the FDR at `alpha`.

    Step-up: sort ascending, find the largest rank k with p₍k₎ ≤ (k/m)·α, and
    reject ranks 1..k. Rejecting the whole prefix is what makes BH more powerful
    than testing each p-value against its own threshold - a p-value above its own
    bound is still rejected when a larger one satisfies its bound.

    Returns positions in the **original** order. The caller has to map a rejection
    back to the strategy that produced it, and returning sorted ranks would
    silently reattribute every promotion.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for index, p in enumerate(p_values):
        if not 0.0 <= p <= 1.0 or math.isnan(p):
            raise ValueError(
                f"p_values[{index}] = {p} is not a probability. Clamping it would "
                f"let an upstream bug reach a promotion decision")
    m = len(p_values)
    if m == 0:
        return []

    ranked = sorted(range(m), key=lambda i: p_values[i])
    largest_satisfying_rank = 0
    for rank, index in enumerate(ranked, start=1):
        if p_values[index] <= (rank / m) * alpha:
            largest_satisfying_rank = rank
    return sorted(ranked[:largest_satisfying_rank])
