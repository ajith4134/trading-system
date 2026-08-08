"""What drawdown to plan for — a distribution, not the one that happened.

`FEATURES.md` §8 asks for circuit-breaker thresholds that are non-arbitrary. The
corpus states the problem with the usual practice directly: *"Most industry
thresholds are round numbers and committee judgment"* (`allocation-and-regime.md`).
And the sharper version, VX-119: **the backtest's max drawdown is a lower bound,
not an expectation.** It is a single draw - the worst thing that happened to
happen - and sizing a kill-switch budget to it means sizing to roughly the middle
of the distribution whose upper tail is where the account dies.

The prescription, verbatim:

> *"Bootstrap the max-drawdown DISTRIBUTION (**block bootstrap to preserve
> autocorrelation**) rather than quoting one realized worst case. A few dozen
> lines, cheap, directly actionable, **underused relative to its value** - and it
> is what makes circuit-breaker thresholds non-arbitrary."*

**The parenthesis is the entire correctness argument.** Drawdowns are produced by
*runs* of losses. An i.i.d. bootstrap resamples individual returns and destroys the
serial correlation that creates runs, so it reports a systematically shallower
distribution than the data supports - while presenting as a rigorous thousand-path
Monte Carlo. The error direction is the dangerous one: too small a drawdown budget
is what closes an account. `block_size=1` reduces to i.i.d. and is retained only so
that failure can be demonstrated in a test.

Two block schemes, both named in VX-152:

  * **moving block**, circular - fixed-length blocks drawn from a wrapped series, so
    the first and last observations are not sampled less often than the middle.
  * **stationary** (Politis & Romano 1994) - geometric block lengths, so the
    resampled series is stationary by construction rather than carrying the
    fixed-block scheme's periodicity.

Prior art checked before writing. RX-027 is the nearest: a vectorised
1,000-scenario forward Monte Carlo of P(20% drawdown) that soft-reduces concurrent
trades. Different mechanism - it simulates forward from assumed trade statistics
rather than resampling realized history - so not a donor. OGE-028 records the
bootstrapped circuit-breaker calibration itself as *"design only - not found as
working code anywhere"*.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Below this, resampling manufactures confidence rather than information. Fifty
# observations bootstrapped a thousand times is still fifty observations, and the
# tidy percentile that comes out is exactly what makes it dangerous.
_MIN_OBSERVATIONS = 250

# A percentile needs enough draws that it is not interpolating between the top two
# observations. p99 from 100 samples is one observation wearing a tail estimate's
# clothes.
_MIN_SAMPLES_PER_TAIL_PERCENTILE = 200

_LADDER_PERCENTILES = (0.75, 0.90, 0.99)


class NotEnoughHistory(ValueError):
    """The sample is too short for a bootstrap to mean anything."""


def max_drawdown(returns: list[float]) -> float:
    """Worst peak-to-trough decline of the compounded equity curve, as a fraction.

    Compounded, not summed: equity is multiplicative, and summing overstates the
    recovery after a large loss - down 50% then up 50% is down 25%, not flat.

    Tracks the running peak rather than the running decline. Watching only the
    current decline lets a later shallow dip overwrite a deeper earlier one, which
    is the usual form of this bug and always understates.
    """
    if not returns:
        raise ValueError("cannot compute a drawdown of an empty return series")
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def _resample_moving_block(returns: list[float], block_size: int,
                           rng: random.Random) -> list[float]:
    """Circular moving-block resample of the same length as the input.

    Circular on purpose: with non-circular blocks the first and last observations
    can only appear in one block position each, so the ends are undersampled and
    the resampled distribution is biased toward the middle of the record.
    """
    n = len(returns)
    out: list[float] = []
    while len(out) < n:
        start = rng.randrange(n)
        out += [returns[(start + i) % n] for i in range(block_size)]
    return out[:n]


def _resample_stationary(returns: list[float], mean_block: int,
                         rng: random.Random) -> list[float]:
    """Politis & Romano stationary bootstrap: geometric block lengths.

    Each step either continues the current block or starts a new one at a random
    index, with continuation probability 1 - 1/mean_block. Block lengths are
    therefore geometric with mean `mean_block`, and the resampled series is
    stationary rather than inheriting a fixed block period.
    """
    n = len(returns)
    p_new = 1.0 / mean_block
    out: list[float] = []
    index = rng.randrange(n)
    for _ in range(n):
        out.append(returns[index])
        index = rng.randrange(n) if rng.random() < p_new else (index + 1) % n
    return out


def block_bootstrap_max_drawdowns(returns: list[float], n_samples: int = 1000,
                                  block_size: int = 50, seed: int = 0,
                                  method: str = "moving") -> list[float]:
    """`n_samples` max drawdowns from resampled paths of the same length.

    `block_size=1` is i.i.d. resampling. It is permitted only so its shortfall can
    be demonstrated - it destroys the autocorrelation that produces drawdowns and
    understates the distribution.
    """
    if len(returns) < _MIN_OBSERVATIONS:
        raise NotEnoughHistory(
            f"need >= {_MIN_OBSERVATIONS} observations to bootstrap a drawdown "
            f"distribution, got {len(returns)}. Resampling a short sample "
            f"manufactures confidence rather than information, and the tidy "
            f"percentile it produces is what makes it dangerous")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if block_size > len(returns):
        raise ValueError(
            f"block_size {block_size} exceeds the series length {len(returns)}; "
            f"every draw would be the same series, so the 'distribution' would be "
            f"one point repeated")
    if method not in ("moving", "stationary"):
        raise ValueError(
            f"unknown bootstrap method {method!r}; use 'moving' or 'stationary'")

    rng = random.Random(seed)
    resample = (_resample_moving_block if method == "moving"
                else _resample_stationary)
    return [max_drawdown(resample(returns, block_size, rng))
            for _ in range(n_samples)]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than interpolated: interpolating between the top two draws
    of a bootstrap invents a value the resampling never produced, and does it
    precisely in the tail where the number is being trusted most.
    """
    index = min(len(sorted_values) - 1,
                max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


@dataclass(frozen=True)
class LadderRung:
    """One circuit-breaker step, and where its number came from."""

    percentile: float
    drawdown_threshold: float
    gross_reduction: float
    provenance: str


@dataclass(frozen=True)
class CircuitBreakerLadder:
    """A graduated ladder whose every rung traces to a measured percentile.

    `realized_max_drawdown` travels with it so the gap between "the worst that
    happened" and "the worst to plan for" is visible where the decision is made.
    """

    rungs: tuple[LadderRung, ...]
    realized_max_drawdown: float
    n_bootstrap_samples: int
    block_size: int
    method: str


def circuit_breaker_ladder(returns: list[float], n_samples: int = 1000,
                           block_size: int = 50, seed: int = 0,
                           method: str = "moving") -> CircuitBreakerLadder:
    """Build the graduated ladder from the bootstrapped drawdown distribution.

    Shape from `allocation-and-regime.md`: cut gross progressively, go flat at the
    deepest rung. The *levels* come from the bootstrap rather than from round
    numbers, which is the whole point - past the p99 rung the assumption the ladder
    was built on is the thing that has failed, so the response is not a reduction.
    """
    if n_samples < _MIN_SAMPLES_PER_TAIL_PERCENTILE:
        raise ValueError(
            f"n_samples={n_samples} is too few for a p99 rung - it would "
            f"interpolate between the top draws and read as a precise tail "
            f"estimate. Use >= {_MIN_SAMPLES_PER_TAIL_PERCENTILE}")

    samples = sorted(block_bootstrap_max_drawdowns(
        returns, n_samples=n_samples, block_size=block_size, seed=seed,
        method=method))

    # Progressive de-risking, flat at the top. Paired with the percentiles in
    # order, so a deeper drawdown can never map to a milder response.
    reductions = (0.25, 0.50, 1.0)
    rungs = tuple(
        LadderRung(
            percentile=q,
            drawdown_threshold=_percentile(samples, q),
            gross_reduction=reduction,
            provenance=(
                f"p{q} = {_percentile(samples, q):.4f} from {n_samples} "
                f"{method} block bootstrap draws (block_size={block_size}) over "
                f"{len(returns)} observations"),
        )
        for q, reduction in zip(_LADDER_PERCENTILES, reductions)
    )
    return CircuitBreakerLadder(
        rungs=rungs,
        realized_max_drawdown=max_drawdown(returns),
        n_bootstrap_samples=n_samples,
        block_size=block_size,
        method=method,
    )
