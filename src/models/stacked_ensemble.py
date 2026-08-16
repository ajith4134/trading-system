"""Stacking, with the one trap that makes it look brilliant closed by construction.

`FEATURES.md` §3 (P2): *"Stacked ensemble"*. Ledger MD-033.

## The trap, and why this module takes fitters rather than predictions

A stacked ensemble trains base models, then trains a **meta-learner** on their
predictions. The near-universal defect is training the meta-learner on the base
models' **in-sample** predictions: on the training set a boosted tree is nearly
perfect, so the stacker sees a column that is nearly the answer, learns to trust
whichever base model overfit hardest, and reports a cross-validated score that is
outstanding and unreachable in production. It fails in the flattering direction,
it is invisible in the output, and it is what most stacking code does.

The obvious API — `stack(base_predictions, targets)` — cannot prevent it. Whoever
calls it decides whether those predictions were out of fold, and a flag saying
*"yes, these are out-of-fold, I promise"* is a comment with a colon in it.

So this module takes **fitters**, not predictions. `BaseLearner.fit_predict` is
called by this module, on folds this module built, and the meta-learner is
trained on the resulting out-of-fold matrix. There is no argument through which
in-sample base predictions can arrive.

## The folds are purged, and the same folds are used twice on purpose

`validation.purged_cross_validation.combinatorial_purged_folds` builds them, so
the base models' out-of-fold predictions carry the same purge and embargo every
other fit in this project does. The meta-learner is then scored on the **same**
fold structure, which means its score is out of sample with respect to both
levels — the base fit and the stack fit — rather than only the second.

A stack scored on fresh random folds over the out-of-fold matrix would look
better and would be wrong: those "out-of-fold" columns were produced by models
that saw most of the rest of the series.

## The meta-learner is deliberately the simplest thing that can combine them

Ridge-regularised logistic regression, solved by IRLS, over `k` probability
columns and an intercept. Not another boosted tree.

A tree stacker over three probability columns has enough capacity to memorise
which fold a row came from — the base models' calibration differs slightly per
fold, and a tree will find that and use it. The stacker must be simpler than the
things it combines or it becomes one more place to overfit, and the whole point
of stacking is that the base models are where the capacity lives. The ridge term
is not optional for the same reason: base predictions are highly correlated with
each other, and an unregularised logistic fit on collinear columns produces
enormous opposing coefficients that are exquisitely tuned to this sample.

## It has to beat the best single base model, not the majority class

A stack that beats the base rate has proved nothing — its base models already do.
The question stacking has to answer is whether combining them beat the best one
alone, and `beats_best_base` is that question, tested with
`validation.superior_predictive_ability` rather than compared as two numbers.

`base_accuracies` rides the result so the comparison is auditable. A stack
reported without the numbers it beat is a claim nobody can check, and on this
particular technique the honest answer is often no: three correlated models
combine to approximately one of them.

## Determinism

IRLS is deterministic given the data. The iteration cap and the convergence
tolerance are declared, and `converged` is reported rather than assumed — a
meta-learner that hit the cap has coefficients that are wherever the last step
left them, and a caller comparing two stacks needs to know that one of them did
not finish.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from validation.purged_cross_validation import combinatorial_purged_folds
from validation.superior_predictive_ability import (
    NotEnoughPaths, superior_predictive_ability,
)
from validation.trial_registry import TrialRegistry, TrialSpec

# Ridge strength on the meta-learner. Not optional: base predictions are highly
# correlated with each other, and an unregularised logistic fit on collinear
# columns produces enormous opposing coefficients tuned to this sample. Declared
# rather than searched - searching it would make the stacker's regularisation a
# hyperparameter fitted on the same data it is protecting.
META_RIDGE = 1.0

# IRLS converges in a handful of steps on a problem this small. The cap exists so
# a pathological input terminates; hitting it is REPORTED, never absorbed.
MAX_IRLS_ITERATIONS = 50
IRLS_TOLERANCE = 1e-8

# Fewer base models than this is not an ensemble - it is one model with extra
# steps, and the stacker would be fitting a monotone transform of a single column.
MIN_BASE_LEARNERS = 2

MIN_SCORED_ROWS = 20


class NotAnEnsemble(ValueError):
    """Fewer than two base learners.

    Refused rather than passed through: with one column the meta-learner fits a
    monotone transform of that column, which cannot beat it and cannot fail to
    match it, so every number the module reports would be a tautology.
    """


class MetaLearnerDidNotConverge(RuntimeError):
    """IRLS hit its iteration cap.

    Raised only when a caller asks for it; the default is to report `converged`
    on the result. Coefficients from an unfinished solve are wherever the last
    step left them, and two stacks are not comparable when one of them did not
    finish.
    """


@dataclass(frozen=True)
class BaseLearner:
    """One base model, as a name and a fitter this module calls itself.

    `fit_predict(train_rows, test_rows) -> probabilities`. Taking the fitter
    rather than its predictions is what makes in-sample base predictions
    unreachable - see the module docstring.
    """
    name: str
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MetaLearner:
    """The fitted combination, with the diagnostic that says it finished."""
    intercept: float
    coefficients: dict[str, float]
    converged: bool
    iterations: int

    def predict(self, columns: dict[str, Sequence[float]]) -> np.ndarray:
        names = list(self.coefficients)
        matrix = np.column_stack([np.asarray(columns[name], dtype=float)
                                  for name in names])
        weights = np.array([self.coefficients[name] for name in names])
        return _sigmoid(self.intercept + matrix @ weights)


@dataclass(frozen=True)
class StackResult:
    """What the stack scored, and what it had to beat.

    `base_accuracies` rides the result because a stack reported without the
    numbers it beat is a claim nobody can check.
    """
    stack_accuracy: float
    base_accuracies: dict[str, float]
    best_base: str
    beats_best_base: bool
    p_value: float | None
    n_scored: int
    meta_learner: MetaLearner

    def as_registry_result(self) -> dict:
        return {
            "sharpe": None,
            "stack_accuracy": self.stack_accuracy,
            "base_accuracies": self.base_accuracies,
            "best_base": self.best_base,
            "beats_best_base": self.beats_best_base,
            "p_value": self.p_value,
            "n_scored": self.n_scored,
            "meta_coefficients": self.meta_learner.coefficients,
            "meta_intercept": self.meta_learner.intercept,
            "meta_converged": self.meta_learner.converged,
        }

    def describe(self) -> str:
        verdict = "beats" if self.beats_best_base else "does NOT beat"
        p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
        converged = "" if self.meta_learner.converged else " [META DID NOT CONVERGE]"
        return (f"stack {self.stack_accuracy:.4f} {verdict} the best base "
                f"{self.best_base} at "
                f"{self.base_accuracies[self.best_base]:.4f} over "
                f"{self.n_scored} row(s), p={p}{converged}")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so neither branch overflows: exp of a large positive number
    # is inf, and inf/inf is nan, which would silently poison the whole column.
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def fit_logistic_ridge(design: np.ndarray, targets: np.ndarray, *,
                       ridge: float = META_RIDGE,
                       max_iterations: int = MAX_IRLS_ITERATIONS,
                       tolerance: float = IRLS_TOLERANCE,
                       ) -> tuple[np.ndarray, bool, int]:
    """Ridge logistic regression by IRLS. Returns (weights, converged, steps).

    The intercept is the first column and is **not** penalised: shrinking it
    toward zero would pull every prediction toward 0.5 in proportion to how
    imbalanced the labels are, which is a bias introduced by the regulariser
    rather than by the data.
    """
    n, d = design.shape
    weights = np.zeros(d)
    penalty = np.full(d, ridge)
    penalty[0] = 0.0                        # the intercept is not penalised
    for step in range(1, max_iterations + 1):
        probabilities = _sigmoid(design @ weights)
        # Floored away from 0 and 1: a perfectly separated column drives the
        # IRLS weight to zero and the update to a singular solve, and the
        # answer there is a very large coefficient, not an exception.
        variance = np.clip(probabilities * (1 - probabilities), 1e-10, None)
        gradient = design.T @ (probabilities - targets) + penalty * weights
        hessian = (design.T * variance) @ design + np.diag(penalty)
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:       # pragma: no cover - ridge prevents it
            return weights, False, step
        weights = weights - update
        if np.max(np.abs(update)) < tolerance:
            return weights, True, step
    return weights, False, max_iterations


def out_of_fold_matrix(base_learners: Sequence[BaseLearner], n_rows: int,
                       folds) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each base learner's out-of-fold probability for every row it was tested on.

    Rows tested by more than one fold have their probabilities averaged, for the
    reason `models.gradient_boosted_trees` averages them: with `k_test > 1` a row
    appears in several test blocks, and keeping the copies would give the
    meta-learner the same row several times and let every downstream statistic
    treat repetition as sample size.
    """
    sums = {learner.name: np.zeros(n_rows) for learner in base_learners}
    counts = np.zeros(n_rows, dtype=int)
    for fold in folds:
        train_rows = np.concatenate(
            [np.arange(start, end) for start, end in fold.train_blocks])
        test_rows = np.concatenate(
            [np.arange(start, end) for start, end in fold.test_blocks])
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue
        for learner in base_learners:
            sums[learner.name][test_rows] += np.asarray(
                learner.fit_predict(train_rows, test_rows), dtype=float)
        counts[test_rows] += 1

    scored = np.flatnonzero(counts > 0)
    columns = {name: total[scored] / counts[scored]
               for name, total in sums.items()}
    return columns, scored


def stack(labels, base_learners: Sequence[BaseLearner], *, family: str,
          registry: TrialRegistry, trial_name: str, n_groups: int = 6,
          k_test: int = 2, ridge: float = META_RIDGE,
          seed: int = 0) -> StackResult:
    """Fit base models out of fold, combine them, and test the combination.

    The whole stack is ONE trial. It answers one question - does combining these
    base models beat the best of them - and registering one trial per base model
    would inflate N by a search nobody performed, the same accounting
    `models.walk_forward` uses for its windows.
    """
    if len(base_learners) < MIN_BASE_LEARNERS:
        raise NotAnEnsemble(
            f"{len(base_learners)} base learner(s), need >= "
            f"{MIN_BASE_LEARNERS}. With one column the meta-learner fits a "
            f"monotone transform of that column, so every number reported would "
            f"be a tautology")
    names = [learner.name for learner in base_learners]
    if len(set(names)) != len(names):
        raise ValueError(
            f"base learner names must be unique, got {names} - a duplicate name "
            f"would silently drop one learner's column")

    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    if len(classes) != 2:
        raise ValueError(f"expected exactly two label classes, got {classes}")
    targets = (labels == classes[1]).astype(int)

    folds = combinatorial_purged_folds(len(targets), family, n_groups=n_groups,
                                       k_test=k_test)

    def evaluate(_spec: TrialSpec) -> dict:
        return _run(targets, base_learners, folds, ridge, seed
                    ).as_registry_result()

    spec = TrialSpec(
        name=trial_name, family=family,
        params={"base_learners": names, "ridge": ridge, "n_groups": n_groups,
                "k_test": k_test, "seed": seed})
    registry.evaluate(spec, evaluate)
    return _run(targets, base_learners, folds, ridge, seed)


def _run(targets: np.ndarray, base_learners: Sequence[BaseLearner], folds,
         ridge: float, seed: int) -> StackResult:
    columns, scored = out_of_fold_matrix(base_learners, len(targets), folds)
    if len(scored) < MIN_SCORED_ROWS:
        raise ValueError(
            f"{len(scored)} row(s) received an out-of-fold prediction, need >= "
            f"{MIN_SCORED_ROWS}. Refused rather than scored")
    actual = targets[scored]

    names = [learner.name for learner in base_learners]
    matrix = np.column_stack([columns[name] for name in names])
    design = np.column_stack([np.ones(len(scored)), matrix])
    weights, converged, iterations = fit_logistic_ridge(
        design, actual.astype(float), ridge=ridge)
    meta = MetaLearner(intercept=float(weights[0]),
                       coefficients={name: float(w)
                                     for name, w in zip(names, weights[1:])},
                       converged=converged, iterations=iterations)

    stack_probabilities = _sigmoid(design @ weights)
    stack_correct = [1.0 if p == a else 0.0
                     for p, a in zip((stack_probabilities >= 0.5).astype(int),
                                     actual)]
    base_correct = {
        name: [1.0 if p == a else 0.0
               for p, a in zip((columns[name] >= 0.5).astype(int), actual)]
        for name in names
    }
    base_accuracies = {name: sum(series) / len(series)
                       for name, series in base_correct.items()}
    best_base = max(base_accuracies, key=lambda name: base_accuracies[name])

    p_value: float | None = None
    beats = False
    if stack_correct != base_correct[best_base]:
        try:
            spa = superior_predictive_ability(
                incumbent=base_correct[best_base],
                challengers={"stack": stack_correct}, seed=seed)
            beats = "stack" in spa.superior
            p_value = spa.p_value
        except (NotEnoughPaths, ValueError):
            p_value = None
    # Identical to the best base means the stack learned to be that base model -
    # a real outcome for three correlated inputs, and not one to dress up as a
    # non-significant improvement.

    return StackResult(
        stack_accuracy=sum(stack_correct) / len(stack_correct),
        base_accuracies=base_accuracies, best_base=best_base,
        beats_best_base=beats, p_value=p_value, n_scored=len(scored),
        meta_learner=meta)
