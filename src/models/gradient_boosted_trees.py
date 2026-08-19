"""LightGBM, trained the only way this project allows a model to be trained.

`FEATURES.md` §3 (P1): *"Gradient-boosted trees — primary workhorse; CPU-native"*.
Ledger MD-010. `DECISIONS.md` §8 settles the choice rather than leaving it open:
*"LightGBM is the default. Every neural component must beat LightGBM and a linear
baseline on our own data before it ships."* — so this is the thing other models
are measured against, and it has to be trained honestly before that comparison
means anything.

## This module is mostly wiring, and the wiring is the point

Calling `lightgbm.train` is four lines. Everything else here exists because this
repo already owns the machinery that makes a fitted number trustworthy, and until
now nothing called it:

* `validation.trial_registry` — **every fit is counted.** Training runs inside
  `registry.evaluate`, which pre-registers the trial on disk *before* the fit and
  settles it after, so a sweep of twenty parameter sets raises its own bar
  twentyfold and a process killed mid-fit leaves a counted, visibly unfinished
  trial rather than a silent gap.
* `validation.purged_cross_validation` — folds are **purged and embargoed by the
  strategy family's own label horizon.** A carry label spans 480 bars; training on
  a row whose label window overlaps a test block leaks that block's outcome into
  the fit, and the leak flatters.
* `features.sample_uniqueness` — **sample weights are average uniqueness**, not
  1.0. This is that module's first consumer.
* `validation.superior_predictive_ability` — the out-of-fold score is compared
  against the majority-class benchmark with a studentised stationary bootstrap,
  so "beat the baseline" is a test rather than a difference.

Three of those were built and reached by nothing. A model trained without them is
not a faster version of this — it is a different claim.

## Why the weights matter more here than anywhere else

Triple-barrier labels overlap: an event at bar 100 with a 480-bar horizon shares
477 bars with an event at bar 103. Handed to LightGBM unweighted, that cluster
counts as many independent observations, and the tree splits on whatever the
cluster happened to do. The effective sample size is a fraction of the row count,
the model is confident in proportion to how overlapped the data is, and every
statistic computed afterwards inherits the inflation.

`average_uniqueness` gives each label the mean of `1 / concurrency` across the
bars it spans — the fraction of its own window it did not share. Passing that as
`weight` is the correction. `WEIGHTS_REQUIRED` makes it non-optional: a caller can
pass explicit uniform weights and say so, but it cannot silently omit them.

## The benchmark is the majority class, and it is not the naive baseline module

`models.naive_baseline` tests a **level forecast** against a random walk with
squared loss. A classifier emits class probabilities, and squared loss against
"the last class" is a category error — the random walk's whole content is that
prices persist, and a class label does not.

The right benchmark for a classifier on imbalanced financial labels is the
majority class, which is exactly the trap here: labels are often 55/45, a model
that learned nothing predicts the majority every time and scores 55%, and 55%
reads as skill to anyone who has not asked what the base rate was. So `base_rate`
rides every result, and `beats_majority_class` is a bootstrap test rather than a
comparison of two accuracies.

The two modules share `superior_predictive_ability` underneath, which is
deliberate: one significance machinery, one place for it to be wrong.

## Determinism is forced, because the registry records numbers

`deterministic=True`, `force_row_wise=True`, a fixed `seed`, and `num_threads=1`.
LightGBM's multithreaded histogram construction is not bit-reproducible, so the
same trial re-run gives a slightly different score — and an append-only ledger of
scores that cannot be reproduced is a ledger of anecdotes. Single-threaded is
slower and that is the trade being made knowingly; the alternative is a registry
whose rows nobody can check.

## No Sharpe is reported, and that is not an omission

A classifier produces predictions, not positions. The Sharpe belongs to whatever
converts one into the other — a sizing rule, a cost model, an execution
assumption — and inventing one here would put three decisions this module does
not make inside a number the Deflated Sharpe gate would then treat as measured.
`result["sharpe"]` is `None`, which `TrialRegistry.trial_sharpes` already
excludes from the DSR's variance while still counting the trial in N. That is the
correct accounting: the fit consumed a look at the data and produced no Sharpe.

## What is deliberately not here

No hyperparameter search. `train_gbt` fits one configuration and counts it; a
search is a loop over this function, written by the caller, where the number of
configurations is visible in the caller's code and lands in the registry one row
at a time. A search hidden inside a training function is the mechanism by which N
and the true trial count diverge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from features.sample_uniqueness import average_uniqueness_by_group, LabelSpan, average_uniqueness
from models.openmp_runtime import ensure_openmp
from validation.purged_cross_validation import CpcvFold, combinatorial_purged_folds
from validation.superior_predictive_ability import (
    NotEnoughPaths, superior_predictive_ability,
)
from validation.trial_registry import TrialRegistry, TrialSpec

ensure_openmp()
import lightgbm as lgb  # noqa: E402 - must follow the OpenMP preload

# Forced on every fit. See the module docstring: LightGBM's threaded histogram
# construction is not bit-reproducible, and a registry of scores nobody can
# reproduce is a registry of anecdotes.
DETERMINISM_PARAMS = {
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}

# Defaults deliberately small. A 31-leaf tree on a few thousand overlapping
# financial labels memorises; these are the starting point a caller overrides,
# and every override lands in the registry as its own trial.
#
# The consequence, measured 2026-08-16 and worth knowing before it is diagnosed
# the hard way: `min_data_in_leaf = 50` means a fold with only ~120 training rows
# admits no split LightGBM will take, so the model predicts the MAJORITY CLASS on
# every row and its accuracy equals the base rate exactly. That is not a silent
# failure here - `reproduced_majority_class` reports it - but it does mean these
# defaults need a dataset of some hundreds of rows per fold to say anything, and
# a smaller one wants a smaller `min_data_in_leaf` chosen out loud.
DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
}

DEFAULT_BOOST_ROUNDS = 200

# Below this, out-of-fold predictions cannot support a bootstrap comparison and
# the fit is refused rather than scored. Mirrors the SPA machinery's own floor.
MIN_OUT_OF_FOLD_PREDICTIONS = 20


class LabelsNotBinary(ValueError):
    """Labels are not exactly two classes.

    Triple-barrier labels are -1, +1 or unresolved. Unresolved must be dropped by
    the caller - dropping it here would hide how much of the dataset went - and a
    third class means the caller has kept something this objective cannot fit.
    """


class WeightsRequired(ValueError):
    """No sample weights were supplied and none were explicitly waived.

    Refused rather than defaulted to 1.0. Overlapping triple-barrier labels
    violate IID, and unweighted training counts an overlapping cluster as many
    independent observations - which inflates confidence in proportion to how
    overlapped the data is, silently, in the flattering direction.
    """


class NotEnoughOutOfFoldPredictions(ValueError):
    """The folds produced too few scored rows to test anything.

    A gate that cannot test must not report a pass - the same posture
    `models.naive_baseline` takes when it is handed too short a series.
    """


@dataclass(frozen=True)
class GbtResult:
    """One counted fit, scored out of fold against the majority class.

    `base_rate` rides the result because an accuracy without it is unreadable:
    labels are routinely 55/45, and a model that learned nothing scores 55%.
    """
    accuracy: float
    base_rate: float
    beats_majority_class: bool
    # The classifier's lag-one trap: it predicted the majority class on every
    # row. On imbalanced labels that scores well and learned nothing, so it is
    # reported as its own finding rather than folded into "did not beat".
    reproduced_majority_class: bool
    p_value: float
    n_out_of_fold: int
    n_folds: int
    # How many CPCV folds tested the average row. Reported because the naive
    # implementation concatenates folds and reports `n_out_of_fold` multiplied by
    # this number, which reads as a large sample and is a repeated small one.
    paths_per_row: float
    mean_train_fraction_retained: float
    rows_purged: int
    effective_sample_fraction: float

    def as_registry_result(self) -> dict:
        """The dict the trial registry stores. `sharpe` is None on purpose.

        A classifier produces predictions, not positions; the Sharpe belongs to
        whatever converts one into the other. `trial_sharpes` excludes a None
        while still counting the trial in N, which is the correct accounting for
        a fit that consumed a look at the data and produced no Sharpe.
        """
        return {
            "sharpe": None,
            "accuracy": self.accuracy,
            "base_rate": self.base_rate,
            "beats_majority_class": self.beats_majority_class,
            "reproduced_majority_class": self.reproduced_majority_class,
            "p_value": self.p_value,
            "n_out_of_fold": self.n_out_of_fold,
            "n_folds": self.n_folds,
            "paths_per_row": self.paths_per_row,
            "mean_train_fraction_retained": self.mean_train_fraction_retained,
            "rows_purged": self.rows_purged,
            "effective_sample_fraction": self.effective_sample_fraction,
        }

    def describe(self) -> str:
        if self.reproduced_majority_class:
            verdict = ("REPRODUCED THE MAJORITY CLASS - it predicted the same "
                       "label on every row, which scores")
        else:
            verdict = "beats" if self.beats_majority_class else "does NOT beat"
        return (f"accuracy {self.accuracy:.4f} against a {self.base_rate:.4f} "
                f"base rate over {self.n_out_of_fold} row(s) scored once each, "
                f"{self.n_folds} fold(s), {self.paths_per_row:.1f} path(s) per "
                f"row; {verdict} the majority class "
                f"(p={self.p_value:.4f}); {self.rows_purged} training row(s) "
                f"purged, effective sample "
                f"{self.effective_sample_fraction:.3f} of nominal")


def uniqueness_weights(spans: Sequence[LabelSpan], n_bars: int) -> list[float]:
    """Sample weights from label uniqueness — FE-009's first real consumer.

    Each weight is the mean of `1 / concurrency` over the bars the label spans:
    the fraction of its own window it did not share with another label. A cluster
    of overlapping labels therefore contributes about as much as one independent
    observation, which is what it is.
    """
    return list(average_uniqueness(spans, n_bars))


def _rows_in_blocks(blocks: Sequence[tuple[int, int]]) -> np.ndarray:
    if not blocks:
        return np.empty(0, dtype=int)
    return np.concatenate([np.arange(start, end) for start, end in blocks])


def _fit_one_fold(features: np.ndarray, targets: np.ndarray,
                  weights: np.ndarray, fold: CpcvFold, params: dict,
                  boost_rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train on the fold's purged training blocks, predict its test blocks."""
    train_rows = _rows_in_blocks(fold.train_blocks)
    test_rows = _rows_in_blocks(fold.test_blocks)
    if len(train_rows) == 0 or len(test_rows) == 0:
        return np.empty(0, dtype=int), np.empty(0)

    dataset = lgb.Dataset(features[train_rows], label=targets[train_rows],
                          weight=weights[train_rows], free_raw_data=False)
    booster = lgb.train({**params, **DETERMINISM_PARAMS, "seed": seed},
                        dataset, num_boost_round=boost_rounds)
    return test_rows, booster.predict(features[test_rows])


def train_gbt(features, labels, spans: Sequence[LabelSpan] | None, *,
              family: str, registry: TrialRegistry, trial_name: str,
              params: dict | None = None,
              boost_rounds: int = DEFAULT_BOOST_ROUNDS,
              seed: int = 0, n_groups: int = 6, k_test: int = 2,
              uniform_weights: bool = False,
              span_groups: Sequence | None = None) -> GbtResult:
    """Fit and score one LightGBM configuration, counted and purged.

    `spans` are the triple-barrier label spans for each row, in the same order —
    the weights come from them. Passing `uniform_weights=True` waives that
    explicitly, which is allowed and is recorded in the trial's params; passing
    neither raises, because a silent 1.0 is the flattering default.

    The fit runs inside `registry.evaluate`, so the trial is on disk before the
    first tree is grown. An exception during the fit still leaves a counted,
    abandoned trial: the count is not negotiable.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"{len(features)} feature row(s) against {len(labels)} label(s). "
            f"Refused rather than trimmed - a misaligned fit is a different fit, "
            f"not a smaller one")

    classes = sorted(set(labels.tolist()))
    if len(classes) != 2:
        raise LabelsNotBinary(
            f"expected exactly two label classes, got {classes}. Unresolved "
            f"triple-barrier labels must be dropped by the caller, so that how "
            f"much of the dataset went is visible where the decision was made")
    # Map to {0, 1} in sorted order, so -1/+1 becomes 0/1 and the positive class
    # is the larger label. Stated rather than assumed: a flipped mapping trains a
    # model that is exactly wrong and scores exactly as well.
    targets = (labels == classes[1]).astype(int)

    if spans is None and not uniform_weights:
        raise WeightsRequired(
            "no label spans supplied and uniform_weights was not set. "
            "Overlapping triple-barrier labels violate IID, and training "
            "unweighted counts an overlapping cluster as many independent "
            "observations - which inflates confidence in proportion to how "
            "overlapped the data is")
    if uniform_weights:
        weights = np.ones(len(features))
        effective_fraction = 1.0
    elif span_groups is not None:
        # **Per series, because a span index is local to its own series.**
        # Pooling a cross-section onto one timeline makes symbol A's bar 500 and
        # symbol B's bar 500 the same bar, and uniqueness then divides by the
        # symbol count rather than by genuine label overlap. Measured 2026-08-19:
        # perp reported 0.00223 over 573 symbols, an implied per-symbol figure of
        # 1.28 - above the 1.0 ceiling uniqueness has, which is the proof.
        weights = np.asarray(
            average_uniqueness_by_group(spans, span_groups), dtype=float)
    else:
        weights = np.asarray(uniqueness_weights(spans, len(features)), dtype=float)
        if len(weights) != len(features):
            raise ValueError(
                f"{len(weights)} weight(s) against {len(features)} row(s)")
        effective_fraction = float(weights.mean())

    folds = combinatorial_purged_folds(len(features), family,
                                       n_groups=n_groups, k_test=k_test)

    def evaluate(_spec: TrialSpec) -> dict:
        return _score_folds(features, targets, weights, folds,
                            {**DEFAULT_PARAMS, **(params or {})},
                            boost_rounds, seed, effective_fraction
                            ).as_registry_result()

    spec = TrialSpec(
        name=trial_name, family=family,
        params={**DEFAULT_PARAMS, **(params or {}),
                "boost_rounds": boost_rounds, "seed": seed,
                "n_groups": n_groups, "k_test": k_test,
                # Recorded so a later reader can see whether the IID correction
                # was applied, without re-deriving it from the code of the day.
                "uniform_weights": uniform_weights})
    stored = registry.evaluate(spec, evaluate)
    return GbtResult(
        accuracy=stored["accuracy"], base_rate=stored["base_rate"],
        beats_majority_class=stored["beats_majority_class"],
        reproduced_majority_class=stored["reproduced_majority_class"],
        p_value=stored["p_value"], n_out_of_fold=stored["n_out_of_fold"],
        n_folds=stored["n_folds"],
        paths_per_row=stored["paths_per_row"],
        mean_train_fraction_retained=stored["mean_train_fraction_retained"],
        rows_purged=stored["rows_purged"],
        effective_sample_fraction=stored["effective_sample_fraction"])


def _score_folds(features: np.ndarray, targets: np.ndarray, weights: np.ndarray,
                 folds: Sequence[CpcvFold], params: dict, boost_rounds: int,
                 seed: int, effective_fraction: float) -> GbtResult:
    """Out-of-fold predictions, deduplicated by row, then the majority-class test.

    **Each row is scored once, whatever CPCV does.** This is the correction the
    obvious implementation misses. With `n_groups=6, k_test=2` there are 15 folds
    and every group sits in the test set of 5 of them, so naively concatenating
    every fold's predictions produces five copies of every row. The bootstrap
    then sees 3,000 "observations" from 600, treats the copies as distinct, and
    reports a p-value computed on an effective sample five times the real one -
    inflated, in the flattering direction, and invisible in the output because
    3,000 out-of-fold predictions reads as a large sample rather than a repeated
    small one.

    So the probabilities for a row are AVERAGED across the folds that tested it
    and scored once. Two consequences, both stated rather than absorbed:

      * The averaged prediction is a small ensemble of `paths_per_row` boosters,
        which is a marginally different - and usually marginally better - model
        than any single fold's. That is the honest cost of deduplicating, and it
        is smaller than the cost of a five-fold-inflated p-value.
      * `paths_per_row` is reported, so a reader can see how much averaging is
        behind each prediction rather than inferring it from the fold geometry.
    """
    probability_sums = np.zeros(len(targets))
    probability_counts = np.zeros(len(targets), dtype=int)
    for fold in folds:
        test_rows, probabilities = _fit_one_fold(
            features, targets, weights, fold, params, boost_rounds, seed)
        if len(test_rows) == 0:
            continue
        probability_sums[test_rows] += probabilities
        probability_counts[test_rows] += 1

    scored_rows = np.flatnonzero(probability_counts > 0)
    mean_probability = (probability_sums[scored_rows]
                        / probability_counts[scored_rows])
    predicted = (mean_probability >= 0.5).astype(int).tolist()
    actual = targets[scored_rows].tolist()
    paths_per_row = (float(probability_counts[scored_rows].mean())
                     if len(scored_rows) else 0.0)

    if len(actual) < MIN_OUT_OF_FOLD_PREDICTIONS:
        raise NotEnoughOutOfFoldPredictions(
            f"{len(actual)} out-of-fold prediction(s), need >= "
            f"{MIN_OUT_OF_FOLD_PREDICTIONS}. Refused rather than scored: a gate "
            f"that cannot test must not report a pass")

    correct = [1.0 if p == a else 0.0 for p, a in zip(predicted, actual)]
    majority = 1 if sum(actual) * 2 >= len(actual) else 0
    # The benchmark predicts the majority class on every observation - the model
    # that learned nothing, and the one a raw accuracy is silently compared to.
    benchmark = [1.0 if majority == a else 0.0 for a in actual]

    accuracy = sum(correct) / len(correct)
    base_rate = sum(benchmark) / len(benchmark)

    # The classifier's version of `naive_baseline.reproduced_lag_one_trap`, and
    # the most common outcome on imbalanced financial labels: the model learned
    # to predict the majority class on every row, so its per-observation
    # performance series is IDENTICAL to the benchmark's. Detected by comparing
    # the series rather than by catching the exception SPA raises on zero
    # bootstrap variance - an exact check does not depend on another module's
    # error text, and it lets the finding be reported as its own field instead of
    # as a failure. It is diagnosed apart from "beaten but not by enough" for the
    # reason `naive_baseline` splits the same two: they call for different work.
    reproduced = predicted == [majority] * len(predicted)
    if reproduced:
        beats, p_value = False, 1.0
    else:
        try:
            spa = superior_predictive_ability(
                incumbent=benchmark, challengers={"gbt": correct}, seed=seed)
            beats, p_value = "gbt" in spa.superior, spa.p_value
        except NotEnoughPaths as error:            # pragma: no cover - guarded above
            raise NotEnoughOutOfFoldPredictions(str(error)) from error

    return GbtResult(
        accuracy=accuracy, base_rate=base_rate, beats_majority_class=beats,
        reproduced_majority_class=reproduced,
        p_value=p_value, n_out_of_fold=len(actual), n_folds=len(folds),
        paths_per_row=paths_per_row,
        mean_train_fraction_retained=float(
            np.mean([f.train_fraction_retained for f in folds])),
        rows_purged=int(sum(f.rows_purged for f in folds)),
        effective_sample_fraction=effective_fraction)
