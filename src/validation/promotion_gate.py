"""One decision, composed from the six gates — so that they are actually called.

`CLAUDE.md` names the trap this file exists to avoid: `tail_specs()` built, tested,
called by nothing; five circuit breakers living only in docstrings. Six correct
statistical modules with no code path invoking them would be the same defect, just
harder to notice because each part has passing tests.

The composition:

    TrialRegistry            honest cumulative N, including abandoned runs
      -> CPCV folds          purged both sides, embargoed, per-family horizon
      -> path Sharpes        the trial distribution, not one number
      -> Deflated Sharpe     against the registry's N and the real dispersion
      -> MinBTL              is the record even long enough for this claim
      -> purge retention     did the folds keep enough training data to mean anything
      -> a verdict           carrying every measurement and its threshold

**The error this file is most likely to reintroduce, so it is stated here and
tested:** a candidate evaluated over C(6,2)=15 CPCV paths is **one trial**. The
paths are resamples of one candidate's returns; they are not fifteen ideas. The
corpus's own gate (`nse-botonly`) passed the path count in as the trial count, so a
candidate selected from thousands was deflated as though 15 things had been tried.
`registry.evaluate` is therefore called **once per candidate**, with all the folds
running inside that single trial.

Everything refuses rather than defaults. A verdict is only as honest as its N, and
every silent fallback available here - a default label horizon, an assumed trial
variance, a swallowed exception - moves the answer toward promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable

from validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    min_backtest_length_years,
    sharpe_ratio,
)
from validation.purged_cross_validation import (
    CpcvFold,
    FAMILY_LABEL_HORIZONS,
    UnknownStrategyFamily,
    combinatorial_purged_folds,
)
from validation.trial_registry import TrialRegistry, TrialSpec

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with the number it was based on.

    `measured` and `threshold` are separate fields on purpose: a gate that reports
    only pass/fail cannot be argued with, and a promotion decision nobody can audit
    is one nobody will act on.
    """

    name: str
    passed: bool
    measured: float
    threshold: float
    detail: str


@dataclass(frozen=True)
class PromotionVerdict:
    """Whether to promote, and the whole evidentiary basis for it."""

    promoted: bool
    n_trials: int
    n_cpcv_paths: int
    observed_sharpe: float
    deflated_sharpe: float
    path_sharpes: tuple[float, ...]
    gates: tuple[GateResult, ...] = field(default_factory=tuple)

    def rejection_reasons(self) -> list[str]:
        return [f"{g.name}: {g.detail}" for g in self.gates if not g.passed]


def evaluate_for_promotion(
    registry: TrialRegistry,
    spec: TrialSpec,
    backtest_fold: Callable[[CpcvFold], list[float]],
    *,
    n_rows: int,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_pct: float = 0.01,
    label_horizon: int | None = None,
    min_deflated_sharpe: float = 0.95,
    min_train_retention: float = 0.25,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> PromotionVerdict:
    """Run one candidate through every gate and return the composed verdict.

    `backtest_fold` is called once per CPCV path and returns that path's
    out-of-sample returns. It must not see the test blocks' data during training -
    the fold it receives already carries purged, embargoed train blocks, which is
    the point of receiving a fold rather than a row range.

    The candidate is registered as exactly **one** trial. It counts even if the
    backtest raises, because it consumed a look at the data; a search that could
    shed trials by crashing could shrink its own N at will.
    """
    # Fold construction first, and deliberately outside the trial: a candidate
    # whose family has no declared label horizon can never be validated, and must
    # not consume a trial pretending otherwise.
    if label_horizon is None and spec.family not in FAMILY_LABEL_HORIZONS:
        raise UnknownStrategyFamily(
            f"no label horizon declared for family {spec.family!r}; purging cannot "
            f"be correct without one. Known: {sorted(FAMILY_LABEL_HORIZONS)}")

    folds = combinatorial_purged_folds(
        n_rows, spec.family, n_groups=n_groups, k_test=k_test,
        embargo_pct=embargo_pct, label_horizon=label_horizon)

    def run_every_fold(_spec: TrialSpec) -> dict:
        """All C(n_groups, k_test) paths, inside a single trial.

        This closure is why the path count never reaches the trial count: the whole
        CPCV sweep happens within one `registry.evaluate`, so N increments once.
        """
        per_path: list[float] = []
        pooled: list[float] = []
        for fold in folds:
            returns = backtest_fold(fold)
            if len(returns) >= 2:
                per_path.append(sharpe_ratio(returns, periods_per_year=1))
                pooled += returns
        if not pooled:
            raise ValueError(
                f"every CPCV path returned fewer than 2 out-of-sample returns for "
                f"{_spec.name!r}; there is nothing to score")
        return {"sharpe": sharpe_ratio(pooled, periods_per_year=1),
                "path_sharpes": per_path,
                "pooled_returns": pooled}

    outcome = registry.evaluate(spec, run_every_fold)

    pooled = outcome["pooled_returns"]
    path_sharpes = outcome["path_sharpes"]
    observed = outcome["sharpe"]

    n_trials = registry.cumulative_count()
    # The dispersion comes from the trials the registry actually holds. When this
    # is the first candidate there is only its own Sharpe, dispersion is zero, and
    # the deflation hurdle collapses to zero - correctly, because one look at the
    # data is not multiple testing.
    trial_sharpes = registry.trial_sharpes() or [observed]
    dsr = deflated_sharpe_ratio(pooled, n_trials=n_trials,
                                trial_sharpes=trial_sharpes, periods_per_year=1)

    required_years = min_backtest_length_years(
        n_trials, target_sharpe=max(observed, 1e-9))
    available_years = len(pooled) / periods_per_year
    retention = mean(f.train_fraction_retained for f in folds) if folds else 0.0

    gates = (
        GateResult(
            name="deflated_sharpe", passed=dsr >= min_deflated_sharpe,
            measured=dsr, threshold=min_deflated_sharpe,
            detail=(f"DSR {dsr:.4f} vs required {min_deflated_sharpe} at N={n_trials} "
                    f"trials, observed Sharpe {observed:.4f} over "
                    f"{len(path_sharpes)} CPCV paths"),
        ),
        GateResult(
            name="min_backtest_length",
            passed=available_years >= required_years,
            measured=available_years, threshold=required_years,
            detail=(f"{available_years:.2f} years of out-of-sample data vs MinBTL "
                    f"{required_years:.2f} years for a Sharpe-{observed:.2f} claim "
                    f"at N={n_trials} (2*ln(N)/SR^2)"),
        ),
        GateResult(
            name="purge_retention", passed=retention >= min_train_retention,
            measured=retention, threshold=min_train_retention,
            detail=(f"folds retained {retention:.1%} of their training data after "
                    f"purge+embargo for family {spec.family!r}; below the floor the "
                    f"path Sharpes are computed against a model fit on too little"),
        ),
    )

    return PromotionVerdict(
        promoted=all(g.passed for g in gates),
        n_trials=n_trials,
        n_cpcv_paths=len(folds),
        observed_sharpe=observed,
        deflated_sharpe=dsr,
        path_sharpes=tuple(path_sharpes),
        gates=gates,
    )
