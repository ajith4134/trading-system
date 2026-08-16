"""The fit is four lines; everything defended here is what makes it trustworthy.

Three of the failures below produce a model that scores well and is worthless:
overlapping labels counted as independent observations, CPCV folds concatenated
so every row appears five times in the significance test, and an accuracy read
without the base rate it should be compared against. The fourth - a fit that
never reached the trial registry - produces a model that may be fine and a
promotion gate that can no longer defend anything, because every gate downstream
divides its protection by N.
"""
import numpy as np
import pytest

from features.sample_uniqueness import LabelSpan
from models.gradient_boosted_trees import (
    DEFAULT_PARAMS,
    DETERMINISM_PARAMS,
    LabelsNotBinary,
    WeightsRequired,
    train_gbt,
    uniqueness_weights,
)
from validation.purged_cross_validation import UnknownStrategyFamily
from validation.trial_registry import TrialRegistry

# Small and fast: four groups, two in test, six folds, three paths per row.
_FOLD_SHAPE = {"n_groups": 4, "k_test": 2}
_ROUNDS = 20
_N = 320


def _registry(tmp_path):
    return TrialRegistry(tmp_path / "trials")


def _spans(n, width=4):
    """Overlapping label spans, which is what triple-barrier labels always are."""
    return [LabelSpan(i, min(i + width, n - 1)) for i in range(n)]


def _separable(n=_N, seed=0, noise=0.3):
    """A dataset with a real, learnable relationship in the first column."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 4))
    labels = np.where(features[:, 0] + noise * rng.normal(size=n) > 0, 1, -1)
    return features, labels


def _pure_noise(n=_N, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 4)), rng.integers(0, 2, size=n) * 2 - 1


def _train(tmp_path, features, labels, **kwargs):
    registry = kwargs.pop("registry", None) or _registry(tmp_path)
    return train_gbt(features, labels, _spans(len(features)),
                     family="microstructure", registry=registry,
                     trial_name=kwargs.pop("trial_name", "test"),
                     boost_rounds=_ROUNDS, **_FOLD_SHAPE, **kwargs), registry


# --- the pipeline can find signal, and cannot find it in noise ------------

def test_a_real_relationship_beats_the_majority_class(tmp_path):
    """The only test here that can report success. Without it, every refusal
    below is satisfied by a module that always fails."""
    result, _ = _train(tmp_path, *_separable())

    assert result.accuracy > result.base_rate
    assert result.beats_majority_class
    assert result.p_value < 0.05


def test_pure_noise_does_not_beat_the_majority_class(tmp_path):
    """The test a broken purge fails. If a test block's outcome leaks into the
    fit, a model trained on random labels scores above the base rate - which is
    the defect that makes every backtest in this literature look good."""
    result, _ = _train(tmp_path, *_pure_noise())

    assert not result.beats_majority_class
    assert result.p_value > 0.05


def test_an_imbalanced_dataset_is_scored_against_its_own_base_rate(tmp_path):
    """90/10 labels: predicting the majority every time scores 0.90, and 0.90
    reads as skill to anyone who did not ask what the base rate was."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(_N, 4))
    labels = np.where(rng.random(_N) < 0.9, 1, -1)     # no relationship at all
    result, _ = _train(tmp_path, features, labels)

    assert result.base_rate > 0.85
    assert result.accuracy >= 0.85, "it does score high - that is the trap"
    assert not result.beats_majority_class, (
        "and it must not be reported as skill")
    # The classifier's lag-one trap, diagnosed apart from "beaten but not by
    # enough" because the two call for different work: this one says the model
    # learned nothing at all, and its 0.90 accuracy is the base rate wearing a
    # score's clothes.
    assert result.reproduced_majority_class
    assert "REPRODUCED THE MAJORITY CLASS" in result.describe()


def test_a_model_that_learned_something_is_not_called_a_copy(tmp_path):
    """The other half of the diagnosis. A verdict that fires on everything is a
    verdict nobody can act on."""
    result, _ = _train(tmp_path, *_separable())

    assert not result.reproduced_majority_class


# --- each row is scored once, whatever CPCV does -------------------------

def test_every_row_is_scored_once_not_once_per_path(tmp_path):
    """With 4 groups and 2 in test there are 6 folds and every group is tested
    in 3 of them. Concatenating fold predictions would put three copies of every
    row into the bootstrap, which would then compute a p-value on an effective
    sample three times the real one - inflated, invisible, and in the flattering
    direction.
    """
    result, _ = _train(tmp_path, *_separable())

    assert result.n_out_of_fold == _N, (
        f"{result.n_out_of_fold} scored rows from {_N} - the folds are being "
        f"concatenated rather than averaged")
    assert result.paths_per_row == pytest.approx(3.0), (
        "and the averaging behind each prediction is reported, not implied")


# --- every fit is counted -------------------------------------------------

def test_the_fit_lands_in_the_trial_registry(tmp_path):
    registry = _registry(tmp_path)
    _train(tmp_path, *_separable(), registry=registry)

    assert registry.cumulative_count() == 1
    trial = registry.trials()[0]
    assert trial["family"] == "microstructure"
    assert trial["outcome"] == "complete"


def test_a_sweep_raises_its_own_bar(tmp_path):
    """Three configurations is N=3, not N=1. Every gate downstream divides its
    protection by N, so a search that does not register is a search whose cost is
    borne by nothing."""
    registry = _registry(tmp_path)
    features, labels = _separable()
    for leaves in (4, 8, 16):
        _train(tmp_path, features, labels, registry=registry,
               trial_name=f"leaves-{leaves}", params={"num_leaves": leaves})

    assert registry.cumulative_count() == 3


def test_a_failed_fit_is_still_counted(tmp_path):
    """It consumed a look at the data. A trial that vanishes because it raised is
    exactly how N drifts below the true count."""
    registry = _registry(tmp_path)
    features, labels = _separable()
    with pytest.raises(Exception):
        train_gbt(features, labels, _spans(len(features)),
                  family="microstructure", registry=registry,
                  trial_name="doomed", boost_rounds=_ROUNDS,
                  params={"objective": "not-an-objective"}, **_FOLD_SHAPE)

    assert registry.cumulative_count() == 1
    assert registry.trials()[0]["outcome"] == "abandoned"


def test_no_sharpe_is_recorded_and_the_trial_still_counts(tmp_path):
    """A classifier produces predictions, not positions. Inventing a Sharpe would
    put three decisions this module does not make inside a number the Deflated
    Sharpe gate would treat as measured."""
    registry = _registry(tmp_path)
    _train(tmp_path, *_separable(), registry=registry)

    assert registry.trials()[0]["result"]["sharpe"] is None
    assert registry.trial_sharpes() == []
    assert registry.cumulative_count() == 1


# --- the IID correction is not optional ----------------------------------

def test_omitting_the_weights_is_refused_rather_than_defaulted(tmp_path):
    """A silent 1.0 counts an overlapping cluster as many independent
    observations, and inflates confidence in proportion to the overlap."""
    features, labels = _separable()
    with pytest.raises(WeightsRequired):
        train_gbt(features, labels, None, family="microstructure",
                  registry=_registry(tmp_path), trial_name="unweighted",
                  boost_rounds=_ROUNDS, **_FOLD_SHAPE)


def test_uniform_weights_can_be_waived_but_only_out_loud(tmp_path):
    """Allowed, and recorded in the trial's own params so a later reader can see
    whether the correction was applied without re-deriving it from the code."""
    registry = _registry(tmp_path)
    features, labels = _separable()
    train_gbt(features, labels, None, family="microstructure",
              registry=registry, trial_name="waived", boost_rounds=_ROUNDS,
              uniform_weights=True, **_FOLD_SHAPE)

    assert registry.trials()[0]["params"]["uniform_weights"] is True


def test_overlapping_labels_get_weights_well_below_one(tmp_path):
    """The measurement behind the correction: labels sharing most of their window
    are worth a fraction of an independent observation each."""
    weights = uniqueness_weights(_spans(100, width=9), n_bars=100)

    assert max(weights) <= 1.0
    assert sum(weights) / len(weights) < 0.4, (
        "ten-bar labels one bar apart share nine tenths of their windows")


def test_the_effective_sample_fraction_is_reported(tmp_path):
    """A row count without it overstates the dataset by however much the labels
    overlap, which for a 480-bar carry label is most of it."""
    result, _ = _train(tmp_path, *_separable())

    assert 0.0 < result.effective_sample_fraction < 1.0


# --- refusals -------------------------------------------------------------

def test_a_third_label_class_is_refused(tmp_path):
    """Unresolved triple-barrier labels must be dropped by the CALLER, so that
    how much of the dataset went is visible where the decision was made."""
    features, labels = _separable()
    labels[:10] = 0
    with pytest.raises(LabelsNotBinary):
        _train(tmp_path, features, labels)


def test_misaligned_features_and_labels_are_refused(tmp_path):
    features, labels = _separable()
    with pytest.raises(ValueError, match="different fit"):
        _train(tmp_path, features, labels[:-5])


def test_an_unknown_family_is_refused_rather_than_purged_by_guess(tmp_path):
    """The purge horizon differs by orders of magnitude between families, and
    defaulting to zero disables it for the family least likely to have been
    thought about."""
    features, labels = _separable()
    with pytest.raises(UnknownStrategyFamily):
        train_gbt(features, labels, _spans(len(features)),
                  family="a-family-nobody-declared",
                  registry=_registry(tmp_path), trial_name="unknown",
                  boost_rounds=_ROUNDS, **_FOLD_SHAPE)


def test_the_purge_actually_removes_training_rows(tmp_path):
    """A purge that removed nothing would report the same accuracy and would be
    doing none of the work it is here for."""
    result, _ = _train(tmp_path, *_separable())

    assert result.rows_purged > 0
    assert result.mean_train_fraction_retained < 1.0


# --- determinism ----------------------------------------------------------

def test_the_same_trial_twice_gives_the_same_number(tmp_path):
    """An append-only ledger of scores nobody can reproduce is a ledger of
    anecdotes. LightGBM's threaded histogram build is not bit-reproducible, which
    is why `num_threads` is pinned to 1 and not left to the machine."""
    features, labels = _separable()
    first, registry = _train(tmp_path, features, labels, trial_name="a")
    second, _ = _train(tmp_path, features, labels, registry=registry,
                       trial_name="b")

    assert first.accuracy == second.accuracy
    assert first.p_value == second.p_value


def test_determinism_cannot_be_overridden_by_a_caller_s_params(tmp_path):
    """A caller passing `num_threads: 8` for speed would silently make every
    number in the registry unreproducible."""
    registry = _registry(tmp_path)
    features, labels = _separable()
    fast, _ = _train(tmp_path, features, labels, registry=registry,
                     trial_name="threaded", params={"num_threads": 8})
    pinned, _ = _train(tmp_path, features, labels, registry=registry,
                       trial_name="pinned")

    assert fast.accuracy == pinned.accuracy
    assert DETERMINISM_PARAMS["num_threads"] == 1


def test_the_defaults_are_conservative_enough_to_state(tmp_path):
    """A 31-leaf tree on a few thousand overlapping financial labels memorises.
    Pinned so a widening is a decision rather than a drift."""
    assert DEFAULT_PARAMS["num_leaves"] <= 8
    assert DEFAULT_PARAMS["min_data_in_leaf"] >= 50
