"""Stacking's one famous defect is closed by the shape of the API, not by a flag.

Training the meta-learner on the base models' IN-SAMPLE predictions makes a stack
look brilliant: on the training set a boosted tree is nearly perfect, so the
stacker sees a column that is nearly the answer and learns to trust whichever
base model overfit hardest. The reported score is outstanding and unreachable in
production.

`stack()` takes FITTERS, not predictions, and calls them itself on folds it
built - so there is no argument through which in-sample base predictions can
arrive. `test_no_base_learner_ever_sees_a_row_it_is_asked_to_predict` is the one
that checks the shape actually holds.
"""
import numpy as np
import pytest

from models.stacked_ensemble import (
    META_RIDGE,
    MIN_BASE_LEARNERS,
    BaseLearner,
    NotAnEnsemble,
    _sigmoid,
    fit_logistic_ridge,
    stack,
)
from validation.trial_registry import TrialRegistry

_FAMILY = "microstructure"          # label horizon 5
_HORIZON = 5
_FOLDS = {"n_groups": 4, "k_test": 2}
_N = 600


def _registry(tmp_path):
    return TrialRegistry(tmp_path / "trials")


# --- the meta-learner -----------------------------------------------------

def test_the_logistic_fit_recovers_a_known_direction():
    """Checked against a coefficient ratio the fixture sets, not against a
    reimplementation of the same solver."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(4000, 2))
    logit = 2.0 * features[:, 0] - 1.0 * features[:, 1] + 0.5
    labels = (rng.random(4000) < _sigmoid(logit)).astype(float)
    design = np.column_stack([np.ones(len(features)), features])

    weights, converged, _steps = fit_logistic_ridge(design, labels, ridge=1e-6)

    assert converged
    assert weights[1] / -weights[2] == pytest.approx(2.0, rel=0.15)
    assert weights[0] == pytest.approx(0.5, abs=0.15)


def test_the_ridge_shrinks_the_slopes_and_leaves_the_intercept_alone():
    """Penalising the intercept would pull every prediction toward 0.5 in
    proportion to how imbalanced the labels are - a bias introduced by the
    regulariser rather than by the data."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(500, 2))
    # Heavily imbalanced, so a shrunk intercept would be obvious.
    labels = (rng.random(500) < _sigmoid(3.0 * features[:, 0] - 2.5)).astype(float)
    design = np.column_stack([np.ones(len(features)), features])

    light, _, _ = fit_logistic_ridge(design, labels, ridge=1e-6)
    heavy, _, _ = fit_logistic_ridge(design, labels, ridge=500.0)

    assert abs(heavy[1]) < abs(light[1]), "the slope is shrunk"
    assert heavy[0] < -1.0, "and the intercept still carries the base rate"


def test_the_sigmoid_does_not_overflow_at_either_extreme():
    """exp of a large positive number is inf, and inf/inf is nan - which would
    silently poison a whole column rather than raising."""
    extreme = _sigmoid(np.array([-800.0, -50.0, 0.0, 50.0, 800.0]))

    assert np.all(np.isfinite(extreme))
    assert extreme[0] == pytest.approx(0.0)
    assert extreme[-1] == pytest.approx(1.0)
    assert extreme[2] == pytest.approx(0.5)


def test_a_capped_solve_reports_that_it_did_not_converge():
    """Coefficients from an unfinished solve are wherever the last step left
    them, and two stacks are not comparable when one of them did not finish."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 2))
    labels = (features[:, 0] > 0).astype(float)      # perfectly separable
    design = np.column_stack([np.ones(len(features)), features])

    _weights, converged, steps = fit_logistic_ridge(
        design, labels, ridge=0.0, max_iterations=3)

    assert not converged
    assert steps == 3


# --- the fixture ----------------------------------------------------------

def _dataset(n=_N, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 2))
    labels = np.where(features[:, 0] + features[:, 1] > 0, 1, -1)
    return features, labels


def _column_learner(features, column, name, calls=None):
    """A base model that only looks at one column - so two of them are
    complementary and the stack has something real to combine."""
    def fit_predict(train_rows, test_rows):
        if calls is not None:
            calls.append((np.asarray(train_rows), np.asarray(test_rows)))
        return _sigmoid(3.0 * features[test_rows, column])
    return BaseLearner(name, fit_predict)


# --- the trap, closed by construction -------------------------------------

def test_no_base_learner_ever_sees_a_row_it_is_asked_to_predict(tmp_path):
    """The whole reason this module takes fitters. Every call it makes is
    recorded, and the training rows must not intersect the test rows nor come
    within the label horizon of them."""
    features, labels = _dataset()
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    learners = [_column_learner(features, 0, "a", calls),
                _column_learner(features, 1, "b", calls)]

    stack(labels, learners, family=_FAMILY, registry=_registry(tmp_path),
          trial_name="leak-check", **_FOLDS)

    assert calls, "the module must have called the fitters itself"
    for train_rows, test_rows in calls:
        assert not set(train_rows.tolist()) & set(test_rows.tolist())
        # And purged: no training row within the label horizon before a test row.
        for start in np.unique(test_rows):
            forbidden = set(range(start - _HORIZON, start))
            assert not forbidden & set(train_rows.tolist()) or start < _HORIZON


def test_every_row_gets_one_out_of_fold_prediction_per_learner(tmp_path):
    """Rows tested by several folds are averaged, not repeated - repetition
    would let every downstream statistic treat it as sample size."""
    features, labels = _dataset()
    learners = [_column_learner(features, 0, "a"),
                _column_learner(features, 1, "b")]

    result = stack(labels, learners, family=_FAMILY,
                   registry=_registry(tmp_path), trial_name="dedup", **_FOLDS)

    assert result.n_scored == _N


# --- what the stack has to beat -------------------------------------------

def test_complementary_base_models_are_worth_combining(tmp_path):
    """The only test here that can report success. Each base sees half the
    truth; together they are the whole of it."""
    features, labels = _dataset()
    learners = [_column_learner(features, 0, "a"),
                _column_learner(features, 1, "b")]

    result = stack(labels, learners, family=_FAMILY,
                   registry=_registry(tmp_path), trial_name="complementary",
                   **_FOLDS)

    assert result.beats_best_base
    assert result.stack_accuracy > result.base_accuracies[result.best_base]
    assert "beats" in result.describe()


def test_redundant_base_models_are_not_worth_combining(tmp_path):
    """Three correlated models combine to approximately one of them, and the
    honest answer on this technique is often no. A module that always reports an
    improvement is reporting the fit, not the finding."""
    features, labels = _dataset()
    # Both learners read the same column, with a trivial monotone difference.
    learners = [_column_learner(features, 0, "a"),
                BaseLearner("b", lambda _t, test_rows:
                            _sigmoid(2.9 * features[test_rows, 0]))]

    result = stack(labels, learners, family=_FAMILY,
                   registry=_registry(tmp_path), trial_name="redundant",
                   **_FOLDS)

    assert not result.beats_best_base


def test_the_base_accuracies_ride_the_result(tmp_path):
    """A stack reported without the numbers it beat is a claim nobody can
    check."""
    features, labels = _dataset()
    learners = [_column_learner(features, 0, "a"),
                _column_learner(features, 1, "b")]

    result = stack(labels, learners, family=_FAMILY,
                   registry=_registry(tmp_path), trial_name="audit", **_FOLDS)

    assert set(result.base_accuracies) == {"a", "b"}
    assert result.best_base in result.base_accuracies


def test_the_stack_is_scored_against_the_best_base_not_the_base_rate(tmp_path):
    """A stack that beats the base rate has proved nothing - its base models
    already do."""
    features, labels = _dataset()
    learners = [_column_learner(features, 0, "a"),
                _column_learner(features, 1, "b")]

    result = stack(labels, learners, family=_FAMILY,
                   registry=_registry(tmp_path), trial_name="benchmark",
                   **_FOLDS)

    assert result.base_accuracies[result.best_base] > 0.6, (
        "the benchmark here is a real model, not a coin flip")


# --- refusals and counting ------------------------------------------------

def test_one_base_learner_is_not_an_ensemble(tmp_path):
    """The meta-learner would fit a monotone transform of the single column,
    which cannot beat it and cannot fail to match it - so every number reported
    would be a tautology."""
    features, labels = _dataset()
    with pytest.raises(NotAnEnsemble):
        stack(labels, [_column_learner(features, 0, "a")], family=_FAMILY,
              registry=_registry(tmp_path), trial_name="solo", **_FOLDS)


def test_duplicate_learner_names_are_refused(tmp_path):
    """A duplicate name would silently drop one learner's column."""
    features, labels = _dataset()
    with pytest.raises(ValueError, match="unique"):
        stack(labels, [_column_learner(features, 0, "a"),
                       _column_learner(features, 1, "a")],
              family=_FAMILY, registry=_registry(tmp_path),
              trial_name="clash", **_FOLDS)


def test_a_whole_stack_is_one_trial(tmp_path):
    """It answers one question - does combining these beat the best of them -
    and one trial per base model would inflate N by a search nobody performed."""
    features, labels = _dataset()
    registry = _registry(tmp_path)
    stack(labels, [_column_learner(features, 0, "a"),
                   _column_learner(features, 1, "b")],
          family=_FAMILY, registry=registry, trial_name="counted", **_FOLDS)

    assert registry.cumulative_count() == 1
    assert registry.trials()[0]["params"]["base_learners"] == ["a", "b"]


def test_the_meta_coefficients_land_in_the_trial(tmp_path):
    """A stack whose combination nobody recorded cannot be compared with the
    next one."""
    features, labels = _dataset()
    registry = _registry(tmp_path)
    stack(labels, [_column_learner(features, 0, "a"),
                   _column_learner(features, 1, "b")],
          family=_FAMILY, registry=registry, trial_name="coefficients", **_FOLDS)

    stored = registry.trials()[0]["result"]
    assert set(stored["meta_coefficients"]) == {"a", "b"}
    assert stored["meta_converged"] is True
    assert stored["sharpe"] is None


def test_the_ridge_default_is_not_zero():
    """Base predictions are highly correlated with each other, and an
    unregularised logistic fit on collinear columns produces enormous opposing
    coefficients exquisitely tuned to this sample."""
    assert META_RIDGE > 0
    assert MIN_BASE_LEARNERS >= 2
