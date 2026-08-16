"""Refit, step forward, refit again — and report the shape of the decay.

`FEATURES.md` §3 (P1): *"Rolling walk-forward retrain — preferred over
continual-learning ML"*. Ledger MD-011.

## What this is for, and what it is not

CPCV (`validation.purged_cross_validation`) answers *"is there an edge in this
dataset"* by testing on blocks scattered through it. Walk-forward answers a
different question — *"was there an edge at each point in time, and is there
still one"* — and the two are not substitutes. A model that scores well under
CPCV and degrades monotonically under walk-forward has an edge that has already
decayed, and the CPCV number cannot see that: it averages the good early blocks
with the bad late ones and reports the mean.

So the primary output here is **the per-window series**, not the pooled score. A
single number is what a backtest reports, and reporting one is how a decayed
strategy survives a review.

## Decay is measured, not left to the reader

`decay` compares the first half of the windows against the second half. It is a
difference in mean accuracy with a bootstrap p-value from
`validation.superior_predictive_ability` — the same machinery
`models.gradient_boosted_trees` and `models.naive_baseline` use, so the project
has one significance implementation and one place for it to be wrong.

The comparison is deliberately blunt. Fitting a trend line to six window scores
and reporting its slope's p-value would be a regression on six points, which is
an impressive-looking statistic about almost nothing. Halves is the coarsest
split that can answer *"did it stop working"*, and that is the question.

## The train/test boundary is purged, exactly as CPCV's is

The obvious walk-forward is `train[0:t], test[t:t+w]`, and it leaks. A training
row at `t-1` whose label matures `label_horizon` rows later is scored on an
outcome inside the test window. `validation.purged_cross_validation.
purge_and_embargo` already implements the correct exclusion zone, including the
left edge that the donor implementation this project inherited got wrong (one
sided, leaking a full label horizon into every test block), so it is called here
rather than reimplemented.

There is no embargo on the right in a forward split, and the reason is worth
stating: the embargo exists to stop information flowing backwards from a test
block into training rows that follow it, and in a walk-forward there are no
training rows after the test window — the next window's training set starts
before it and is purged on its own terms.

## Expanding or rolling, declared per run and recorded

Two schemes, and the choice is an assumption about the market rather than a
tuning knob:

* `expanding` keeps all history. Right if the relationship is stable; wrong if
  the market regime-shifted, because ancient data outvotes the recent regime.
* `rolling` keeps a fixed lookback. Right if it shifted; wrong if the edge is
  weak and needs every observation.

Neither is the default in the sense of being the safe choice — `WindowScheme` has
no default at all, so a caller has to pick, and the pick lands in the trial
registry alongside the window sizes. The lookback for a rolling scheme is a
hyperparameter and is counted as one.

## Every run is one trial, not one per window

A walk-forward with eight windows fits eight models and answers **one** question,
so it registers **one** trial. Registering eight would inflate N eightfold and
deflate every downstream gate by a search nobody performed. This is the mirror of
the rule in `gradient_boosted_trees`, where a hyperparameter sweep registers one
trial per configuration because each configuration IS a separate question.

## The fitter is injected, so this is not a LightGBM module

`fit_predict(train_rows, test_rows) -> probabilities` is the whole contract. That
keeps this usable for the naive baseline, for a GBT, and for whatever comes after
it, and it keeps the leak-prevention in one place rather than once per learner.
It also means the tests here run in milliseconds against a trivial fitter, so
what they check is the SPLITTING - which is where the defects are.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np

from validation.purged_cross_validation import (
    FAMILY_LABEL_HORIZONS, UnknownStrategyFamily, purge_and_embargo,
)
from validation.superior_predictive_ability import (
    NotEnoughPaths, superior_predictive_ability,
)
from validation.trial_registry import TrialRegistry, TrialSpec


class WindowScheme(str, Enum):
    """How much history each refit sees. No default - the caller must choose.

    The choice is an assumption about whether the market is stationary, not a
    tuning knob, and a default would make it silently for everyone.
    """
    EXPANDING = "expanding"
    ROLLING = "rolling"


# A window scored on fewer rows than this is noise with a decimal point, and a
# decay series built from such windows would report a trend in sampling error.
MIN_TEST_ROWS = 20

# Fewer than this and there are no halves to compare, so decay cannot be measured
# at all - reported as None rather than as "no decay found".
MIN_WINDOWS_FOR_DECAY = 4


class NotEnoughHistory(ValueError):
    """The series cannot support even one purged train/test split.

    Refused rather than silently producing a single window, which would turn a
    walk-forward into an ordinary holdout while still being called one.
    """


class RollingNeedsLookback(ValueError):
    """A rolling scheme was chosen with no lookback length.

    Refused rather than defaulted: the lookback IS the scheme's content, and a
    default would be an unsearched hyperparameter chosen by whoever wrote this
    file rather than by whoever is making the claim.
    """


@dataclass(frozen=True)
class WalkForwardWindow:
    """One refit and the block it was scored on."""
    index: int
    train_blocks: tuple[tuple[int, int], ...]
    test_block: tuple[int, int]
    train_rows: int
    rows_purged: int
    accuracy: float
    base_rate: float


@dataclass(frozen=True)
class DecayVerdict:
    """Did the later windows do worse than the earlier ones?

    `p_value` is None when there were too few windows to split - stated as an
    absence rather than as a non-significant result, because "we could not test"
    and "we tested and found nothing" are different findings and only one of them
    is reassuring.
    """
    early_accuracy: float
    late_accuracy: float
    difference: float
    p_value: float | None
    is_significant: bool


@dataclass(frozen=True)
class WalkForwardResult:
    """The per-window series first, the pooled number second.

    Ordered that way deliberately. A single pooled score is what a backtest
    reports, and a decayed strategy survives review by being reported as one.
    """
    windows: list[WalkForwardWindow]
    pooled_accuracy: float
    pooled_base_rate: float
    decay: DecayVerdict | None
    scheme: WindowScheme

    def as_registry_result(self) -> dict:
        """What the trial registry stores. `sharpe` is None for the reason
        `gradient_boosted_trees` gives: a classifier produces predictions, not
        positions."""
        return {
            "sharpe": None,
            "scheme": self.scheme.value,
            "n_windows": len(self.windows),
            "pooled_accuracy": self.pooled_accuracy,
            "pooled_base_rate": self.pooled_base_rate,
            "window_accuracies": [w.accuracy for w in self.windows],
            "early_accuracy": self.decay.early_accuracy if self.decay else None,
            "late_accuracy": self.decay.late_accuracy if self.decay else None,
            "decay_p_value": self.decay.p_value if self.decay else None,
            "decay_is_significant": (self.decay.is_significant
                                     if self.decay else None),
        }

    def describe(self) -> str:
        if self.decay is None:
            decay_note = (f"too few windows to test for decay "
                          f"(need >= {MIN_WINDOWS_FOR_DECAY})")
        elif self.decay.is_significant:
            decay_note = (f"DECAYED: {self.decay.early_accuracy:.4f} early "
                          f"against {self.decay.late_accuracy:.4f} late "
                          f"(p={self.decay.p_value:.4f})")
        else:
            decay_note = (f"no measured decay: {self.decay.early_accuracy:.4f} "
                          f"early against {self.decay.late_accuracy:.4f} late")
        return (f"{len(self.windows)} {self.scheme.value} window(s), pooled "
                f"accuracy {self.pooled_accuracy:.4f} against a "
                f"{self.pooled_base_rate:.4f} base rate; {decay_note}")


def plan_windows(n_rows: int, *, scheme: WindowScheme, test_rows: int,
                 initial_train_rows: int, family: str,
                 lookback_rows: int | None = None,
                 label_horizon: int | None = None,
                 ) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    """The (train blocks, test block) pairs, purged, in time order.

    Separated from the fitting so the splitting can be tested on its own - it is
    where the leaks live, and a test that has to train a model to check a
    boundary is a test nobody writes enough of.
    """
    if scheme is WindowScheme.ROLLING and lookback_rows is None:
        raise RollingNeedsLookback(
            "a rolling scheme needs an explicit lookback_rows. Refused rather "
            "than defaulted - the lookback is the scheme's whole content, and a "
            "default would be an unsearched hyperparameter chosen by this file "
            "rather than by whoever is making the claim")
    if test_rows < MIN_TEST_ROWS:
        raise ValueError(
            f"test_rows must be >= {MIN_TEST_ROWS}, got {test_rows}; a window "
            f"scored on fewer rows is noise with a decimal point, and a decay "
            f"series built from such windows reports a trend in sampling error")

    if label_horizon is None:
        if family not in FAMILY_LABEL_HORIZONS:
            raise UnknownStrategyFamily(
                f"no label horizon declared for family {family!r}, so the purge "
                f"cannot be correct. Known: {sorted(FAMILY_LABEL_HORIZONS)}")
        label_horizon = FAMILY_LABEL_HORIZONS[family]

    windows = []
    test_start = initial_train_rows
    while test_start + test_rows <= n_rows:
        test_block = (test_start, test_start + test_rows)
        train_start = (0 if scheme is WindowScheme.EXPANDING
                       else max(0, test_start - lookback_rows))
        # No embargo: it exists to stop information flowing backwards from a test
        # block into training rows AFTER it, and in a forward split there are
        # none - the next window's training set starts before this test block and
        # is purged on its own terms.
        blocks = purge_and_embargo((train_start, test_start), [test_block],
                                   label_horizon, embargo=0)
        if blocks:
            windows.append((tuple(blocks), test_block))
        test_start += test_rows

    if not windows:
        raise NotEnoughHistory(
            f"{n_rows} row(s) support no purged window at "
            f"initial_train_rows={initial_train_rows}, test_rows={test_rows}, "
            f"label_horizon={label_horizon}. Refused rather than producing a "
            f"single window, which would be an ordinary holdout called a "
            f"walk-forward")
    return windows


def _measure_decay(accuracies: Sequence[float], correctness: Sequence[list[float]],
                   seed: int) -> DecayVerdict | None:
    """Early half against late half, with a bootstrap p-value.

    Deliberately blunt. A trend line through six window scores and a p-value on
    its slope is a regression on six points - an impressive-looking statistic
    about almost nothing. Halves is the coarsest split that answers "did it stop
    working", which is the question.
    """
    if len(accuracies) < MIN_WINDOWS_FOR_DECAY:
        return None
    midpoint = len(accuracies) // 2
    early = [x for window in correctness[:midpoint] for x in window]
    late = [x for window in correctness[midpoint:] for x in window]
    early_accuracy = sum(early) / len(early)
    late_accuracy = sum(late) / len(late)

    p_value: float | None = None
    significant = False
    # Truncated to the shorter half so the two series align for SPA. The halves
    # are equal-length windows, so this only bites when the count is odd.
    n = min(len(early), len(late))
    try:
        # "Did the early half do better?" - early as challenger against late as
        # incumbent, so a significant result means performance FELL.
        spa = superior_predictive_ability(
            incumbent=late[:n], challengers={"early": early[:n]}, seed=seed)
        p_value = spa.p_value
        significant = "early" in spa.superior
    except (NotEnoughPaths, ValueError):
        # Too few rows, or two halves that are identical. Reported as an absence
        # rather than as "no decay found": the second is reassuring and would be
        # unearned.
        p_value = None
    return DecayVerdict(early_accuracy=early_accuracy, late_accuracy=late_accuracy,
                        difference=early_accuracy - late_accuracy,
                        p_value=p_value, is_significant=significant)


def walk_forward(labels, fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 *, scheme: WindowScheme, test_rows: int,
                 initial_train_rows: int, family: str,
                 registry: TrialRegistry, trial_name: str,
                 lookback_rows: int | None = None,
                 label_horizon: int | None = None,
                 seed: int = 0) -> WalkForwardResult:
    """Refit forward through the series and report every window separately.

    `fit_predict(train_rows, test_rows)` is handed row INDICES and returns one
    probability per test row. Injecting the fitter keeps this usable for any
    learner and keeps the leak-prevention in one place instead of once per model.

    The whole run is **one** trial: eight windows fit eight models and answer one
    question. Registering one per window would inflate N eightfold and deflate
    every downstream gate by a search nobody performed.
    """
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    if len(classes) != 2:
        raise ValueError(
            f"expected exactly two label classes, got {classes}")
    targets = (labels == classes[1]).astype(int)

    planned = plan_windows(len(targets), scheme=scheme, test_rows=test_rows,
                           initial_train_rows=initial_train_rows, family=family,
                           lookback_rows=lookback_rows,
                           label_horizon=label_horizon)

    def evaluate(_spec: TrialSpec) -> dict:
        return _run_windows(targets, fit_predict, planned, scheme, seed
                            ).as_registry_result()

    spec = TrialSpec(
        name=trial_name, family=family,
        params={"scheme": scheme.value, "test_rows": test_rows,
                "initial_train_rows": initial_train_rows,
                "lookback_rows": lookback_rows, "seed": seed})
    registry.evaluate(spec, evaluate)
    # Re-run rather than reconstructing the dataclasses from the stored dict: the
    # fit is deterministic, and rebuilding a rich result from a JSON round-trip is
    # a second representation to keep in agreement with the first.
    return _run_windows(targets, fit_predict, planned, scheme, seed)


def _run_windows(targets: np.ndarray, fit_predict, planned, scheme: WindowScheme,
                 seed: int) -> WalkForwardResult:
    windows: list[WalkForwardWindow] = []
    correctness: list[list[float]] = []
    for index, (train_blocks, test_block) in enumerate(planned):
        train_rows = np.concatenate(
            [np.arange(start, end) for start, end in train_blocks])
        test_rows = np.arange(*test_block)
        probabilities = np.asarray(fit_predict(train_rows, test_rows))
        predicted = (probabilities >= 0.5).astype(int)
        actual = targets[test_rows]

        correct = [1.0 if p == a else 0.0 for p, a in zip(predicted, actual)]
        majority = 1 if actual.sum() * 2 >= len(actual) else 0
        base_rate = float((actual == majority).mean())
        nominal = test_block[0] - (train_blocks[0][0] if train_blocks else 0)

        correctness.append(correct)
        windows.append(WalkForwardWindow(
            index=index, train_blocks=tuple(train_blocks), test_block=test_block,
            train_rows=len(train_rows), rows_purged=nominal - len(train_rows),
            accuracy=sum(correct) / len(correct), base_rate=base_rate))

    pooled = [x for window in correctness for x in window]
    all_actual = np.concatenate([targets[np.arange(*block)]
                                 for _blocks, block in planned])
    pooled_majority = 1 if all_actual.sum() * 2 >= len(all_actual) else 0
    return WalkForwardResult(
        windows=windows,
        pooled_accuracy=sum(pooled) / len(pooled),
        pooled_base_rate=float((all_actual == pooled_majority).mean()),
        decay=_measure_decay([w.accuracy for w in windows], correctness, seed),
        scheme=scheme)
