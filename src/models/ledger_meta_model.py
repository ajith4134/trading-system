"""What worked in which regime — summarised always, modelled only when it can be.

`FEATURES.md` §3 (P2): *"Meta-model over the experiment ledger — learns which
strategies work in which regime"*. Ledger MD-034.

## The thing that makes this row different from the rest of §3

Every other model in this package is fitted on market data, of which there is a
lot. This one is fitted on the **experiment ledger**, of which there are
currently a few dozen rows and will be a few hundred for a long time. A
"meta-model" over thirty trials is a lookup table with a p-value it has not
earned, and the fact that it produces coefficients is not evidence that it
learned anything.

So the module does two separable things and never confuses them:

* `summarise` — conditional success rates by (family, regime) **with their
  counts**, available from the first trial. Descriptive, honest at any N.
* `fit` — the actual meta-model, **refused** below `MIN_TRIALS_TO_FIT` with the
  number it has and the number it needs. It becomes available on its own, when
  the ledger is big enough, without anyone changing this file.

`FitRefused` is a returned value rather than an exception, because "not yet" is
the expected state for months and a caller should be able to render it on a board
without a try block.

## Two biases the ledger has, stated because they do not go away with N

**Selection.** The ledger contains what somebody chose to try. A meta-model over
it learns the searcher's habits at least as much as the market's: if 80% of
trials are carry strategies, the model's confident statement about carry is a
statement about where attention went. `largest_family_share` is reported for
exactly this reason, and it does not improve as trials accumulate — it improves
only if the search broadens.

**Abandonment.** A trial that raised has no Sharpe and no outcome, so
conditioning on completed trials conditions on completion. `abandoned` is
reported beside every summary. This is the same shape as the survivorship problem
`features.cross_sectional` closes structurally, and here it cannot be closed
structurally — an abandoned trial genuinely has no outcome — so it is made
visible instead.

## The regime is supplied, never invented here

A trial does not know what regime it ran in. `features.volatility_regime` reports
a decile per (venue, symbol) per clock, and the caller is the thing that knows
which key and which window a trial covered. Deriving a regime inside this module
would mean guessing that mapping, and a wrong guess would relabel every row while
looking entirely normal.

## The split is chronological, and that is the only honest one

When the model can be fitted, it is scored on a forward split: fitted on the
earlier trials, tested on the later ones. Random K-fold over the ledger would
train on trials run after the ones it tests, and the whole question here is
whether past experiments predict future ones.

One split, not a walk-forward. `models.walk_forward` exists and is the right tool
over a bar series; over a few hundred ledger rows it would produce windows of
tens of trials each, and a decay statistic computed on those is a statistic about
sampling noise. One boundary, stated.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.stacked_ensemble import META_RIDGE, _sigmoid, fit_logistic_ridge

# Below this the model is refused. Four one-hot columns and an intercept over
# thirty rows is a fit with six observations a parameter, which is the point at
# which the coefficients start describing the sample rather than the world.
# Declared here so the refusal has a number a reader can argue with.
MIN_TRIALS_TO_FIT = 120

# The forward split. Two thirds is not a tuned number - it is the conventional
# split, chosen so that the earlier and later halves are both large enough to
# say anything, and it is declared rather than searched.
TRAIN_FRACTION = 2 / 3

# Below this many test trials the forward split cannot support a score, whatever
# the total is - a 200-trial ledger with 5 distinct later trials is still 5.
MIN_TEST_TRIALS = 30


@dataclass(frozen=True)
class TrialOutcome:
    """One completed experiment, with the regime it ran in.

    `regime` is supplied by the caller - see the module docstring on why this
    module must not derive it. `succeeded` is the caller's threshold too: what
    counts as a success is a promotion question, and `validation.
    promotion_pipeline` owns those.
    """
    trial_id: int
    family: str
    regime: str
    succeeded: bool
    ran_at_ns: int


@dataclass(frozen=True)
class CellSummary:
    """One (family, regime) cell: how often it worked, out of how many."""
    family: str
    regime: str
    trials: int
    successes: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials


@dataclass(frozen=True)
class LedgerSummary:
    """The descriptive half, honest at any N.

    `largest_family_share` and `abandoned` are carried because they are the two
    numbers that say how much the rest of this can be trusted, and neither is
    improved by waiting.
    """
    cells: list[CellSummary]
    total_trials: int
    abandoned: int
    distinct_families: int
    distinct_regimes: int
    largest_family_share: float

    def describe(self) -> str:
        return (f"{self.total_trials} completed trial(s) across "
                f"{self.distinct_families} family(ies) and "
                f"{self.distinct_regimes} regime(s) in {len(self.cells)} cell(s); "
                f"{self.abandoned} abandoned and therefore outside every rate "
                f"below; largest family holds "
                f"{self.largest_family_share:.0%} of the record")


@dataclass(frozen=True)
class FitRefused:
    """The model cannot be fitted yet, with the numbers behind that.

    A value rather than an exception: "not yet" is the expected state for months,
    and a board should be able to render it without a try block.
    """
    reason: str
    trials: int
    required: int


@dataclass(frozen=True)
class MetaModelFit:
    """The fitted meta-model and its forward-split score.

    `beats_base_rate` is the only claim worth making about a model this small,
    and it is reported with the base rate beside it.
    """
    coefficients: dict[str, float]
    intercept: float
    converged: bool
    train_trials: int
    test_trials: int
    test_accuracy: float
    test_base_rate: float
    beats_base_rate: bool

    def describe(self) -> str:
        verdict = "beats" if self.beats_base_rate else "does NOT beat"
        note = "" if self.converged else " [DID NOT CONVERGE]"
        return (f"fitted on {self.train_trials} trial(s), tested on "
                f"{self.test_trials}: accuracy {self.test_accuracy:.4f} "
                f"{verdict} the {self.test_base_rate:.4f} base rate{note}")


def summarise(outcomes: Sequence[TrialOutcome], *,
              abandoned: int = 0) -> LedgerSummary:
    """Conditional success rates by (family, regime), with their counts.

    Every cell carries its `trials`. A cell that succeeded once out of one is
    reported as 100%, and the count beside it is the only thing that stops that
    number being read as a finding.
    """
    counts: Counter = Counter()
    successes: Counter = Counter()
    for outcome in outcomes:
        key = (outcome.family, outcome.regime)
        counts[key] += 1
        successes[key] += int(outcome.succeeded)

    cells = [CellSummary(family=family, regime=regime, trials=counts[(family, regime)],
                         successes=successes[(family, regime)])
             for family, regime in sorted(counts)]
    families = Counter(o.family for o in outcomes)
    total = len(outcomes)
    return LedgerSummary(
        cells=cells, total_trials=total, abandoned=abandoned,
        distinct_families=len(families),
        distinct_regimes=len({o.regime for o in outcomes}),
        largest_family_share=(max(families.values()) / total if total else 0.0))


def _design_matrix(outcomes: Sequence[TrialOutcome], families: list[str],
                   regimes: list[str]) -> tuple[np.ndarray, list[str]]:
    """One-hot family and regime, with the first level of each dropped.

    Dropping a level is not cosmetic: keeping every level alongside an intercept
    makes the design rank-deficient, and a ridge solve on a rank-deficient design
    returns coefficients that are a valid answer to an ill-posed question - they
    look ordinary and they are not comparable between fits.
    """
    names = ([f"family={f}" for f in families[1:]]
             + [f"regime={r}" for r in regimes[1:]])
    rows = []
    for outcome in outcomes:
        row = ([1.0 if outcome.family == f else 0.0 for f in families[1:]]
               + [1.0 if outcome.regime == r else 0.0 for r in regimes[1:]])
        rows.append(row)
    matrix = np.asarray(rows, dtype=float) if rows else np.empty((0, len(names)))
    return matrix, names


def fit(outcomes: Sequence[TrialOutcome], *, ridge: float = META_RIDGE,
        ) -> MetaModelFit | FitRefused:
    """Fit the meta-model on the earlier trials and score it on the later ones.

    Returns `FitRefused` rather than raising when the ledger is too small. The
    refusal carries the count it has and the count it needs, so a board can
    render the progress toward being able to answer rather than an error.
    """
    ordered = sorted(outcomes, key=lambda o: (o.ran_at_ns, o.trial_id))
    if len(ordered) < MIN_TRIALS_TO_FIT:
        return FitRefused(
            reason=(f"a meta-model over {len(ordered)} trial(s) is a lookup "
                    f"table with a p-value it has not earned"),
            trials=len(ordered), required=MIN_TRIALS_TO_FIT)

    split = int(len(ordered) * TRAIN_FRACTION)
    train, test = ordered[:split], ordered[split:]
    if len(test) < MIN_TEST_TRIALS:
        return FitRefused(
            reason=(f"{len(test)} trial(s) after the forward split, need >= "
                    f"{MIN_TEST_TRIALS} to score on"),
            trials=len(ordered), required=MIN_TRIALS_TO_FIT)

    # Levels are taken from the TRAINING half only. Taking them from the whole
    # ledger would let a family that appears only in the test half create a
    # column the fit has never seen, which is a lookahead through the schema
    # rather than through the values.
    families = sorted({o.family for o in train})
    regimes = sorted({o.regime for o in train})

    train_matrix, names = _design_matrix(train, families, regimes)
    train_design = np.column_stack([np.ones(len(train)), train_matrix])
    train_targets = np.array([float(o.succeeded) for o in train])
    weights, converged, _steps = fit_logistic_ridge(
        train_design, train_targets, ridge=ridge)

    test_matrix, _ = _design_matrix(test, families, regimes)
    test_design = np.column_stack([np.ones(len(test)), test_matrix])
    predicted = (_sigmoid(test_design @ weights) >= 0.5).astype(int)
    actual = np.array([int(o.succeeded) for o in test])
    accuracy = float((predicted == actual).mean())
    majority = 1 if actual.sum() * 2 >= len(actual) else 0
    base_rate = float((actual == majority).mean())

    return MetaModelFit(
        coefficients={name: float(w) for name, w in zip(names, weights[1:])},
        intercept=float(weights[0]), converged=converged,
        train_trials=len(train), test_trials=len(test),
        test_accuracy=accuracy, test_base_rate=base_rate,
        # Strictly greater. Equal means the model reproduced the majority class,
        # which on a ledger this sparse is the likeliest outcome and is not a
        # result worth calling a pass.
        beats_base_rate=accuracy > base_rate)
