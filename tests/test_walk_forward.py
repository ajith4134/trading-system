"""A pooled score is what a backtest reports, and a decayed edge survives as one.

The dataset in `test_a_dead_edge_is_visible_per_window_and_not_in_the_pool` is
the argument for this module: signal in the first half, noise in the second,
pooled accuracy 0.70 against a 0.52 base rate. Read as one number it is a
working strategy. Read per window it stopped working halfway through and never
recovered.

The other tests defend the split, because the split is where the leaks are - and
the fitter is injected precisely so they can run in milliseconds against a
trivial one rather than needing a model to check a boundary.
"""
import numpy as np
import pytest

from models.walk_forward import (
    MIN_TEST_ROWS,
    MIN_WINDOWS_FOR_DECAY,
    NotEnoughHistory,
    RollingNeedsLookback,
    WindowScheme,
    plan_windows,
    walk_forward,
)
from validation.purged_cross_validation import UnknownStrategyFamily
from validation.trial_registry import TrialRegistry

_FAMILY = "microstructure"          # label horizon 5
_HORIZON = 5


def _registry(tmp_path):
    return TrialRegistry(tmp_path / "trials")


# --- the split ------------------------------------------------------------

def test_windows_step_forward_and_never_overlap():
    planned = plan_windows(1000, scheme=WindowScheme.EXPANDING, test_rows=100,
                           initial_train_rows=200, family=_FAMILY)

    test_blocks = [block for _train, block in planned]
    assert test_blocks[0] == (200, 300)
    for earlier, later in zip(test_blocks, test_blocks[1:]):
        assert earlier[1] == later[0], "windows abut, never overlap"


def test_the_training_set_is_purged_back_from_the_test_block():
    """The obvious walk-forward is train[0:t], test[t:t+w] and it leaks: a
    training row at t-1 whose label matures `label_horizon` rows later is scored
    on an outcome inside the test window."""
    planned = plan_windows(1000, scheme=WindowScheme.EXPANDING, test_rows=100,
                           initial_train_rows=200, family=_FAMILY)

    train_blocks, test_block = planned[0]
    last_training_row = max(end for _start, end in train_blocks)
    assert last_training_row <= test_block[0] - _HORIZON, (
        f"training runs to {last_training_row} against a test block starting at "
        f"{test_block[0]} with a {_HORIZON}-row label horizon")


def test_an_expanding_scheme_keeps_all_history():
    planned = plan_windows(1000, scheme=WindowScheme.EXPANDING, test_rows=100,
                           initial_train_rows=200, family=_FAMILY)

    starts = [min(start for start, _end in blocks) for blocks, _ in planned]
    assert set(starts) == {0}


def test_a_rolling_scheme_forgets_at_a_declared_rate():
    planned = plan_windows(1000, scheme=WindowScheme.ROLLING, test_rows=100,
                           initial_train_rows=300, family=_FAMILY,
                           lookback_rows=150)

    for blocks, test_block in planned:
        earliest = min(start for start, _end in blocks)
        assert earliest >= test_block[0] - 150


def test_a_rolling_scheme_without_a_lookback_is_refused():
    """The lookback IS the scheme's content. A default would be an unsearched
    hyperparameter chosen by the module rather than by whoever makes the claim."""
    with pytest.raises(RollingNeedsLookback):
        plan_windows(1000, scheme=WindowScheme.ROLLING, test_rows=100,
                     initial_train_rows=300, family=_FAMILY)


def test_there_is_no_default_scheme():
    """The choice is an assumption about whether the market is stationary, not a
    tuning knob, and a default would make it silently for everyone."""
    with pytest.raises(TypeError):
        plan_windows(1000, test_rows=100, initial_train_rows=200,   # noqa: F821
                     family=_FAMILY)


def test_too_little_history_is_refused_rather_than_reduced_to_one_window():
    """One window is an ordinary holdout, and calling it a walk-forward is the
    claim this refuses to let anyone make by accident."""
    with pytest.raises(NotEnoughHistory):
        plan_windows(220, scheme=WindowScheme.EXPANDING, test_rows=100,
                     initial_train_rows=200, family=_FAMILY)


def test_a_tiny_test_window_is_refused():
    """A decay series built from windows of eight rows reports a trend in
    sampling error."""
    with pytest.raises(ValueError, match="noise with a decimal point"):
        plan_windows(1000, scheme=WindowScheme.EXPANDING,
                     test_rows=MIN_TEST_ROWS - 1, initial_train_rows=200,
                     family=_FAMILY)


def test_an_unknown_family_is_refused_rather_than_purged_by_guess():
    with pytest.raises(UnknownStrategyFamily):
        plan_windows(1000, scheme=WindowScheme.EXPANDING, test_rows=100,
                     initial_train_rows=200, family="nobody-declared-this")


# --- what the windows reveal that the pool hides -------------------------

def _decaying_dataset(n=1200, seed=0):
    """Signal in the first half, pure noise in the second."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 2))
    labels = np.where(np.arange(n) < n // 2,
                      np.sign(features[:, 0]),
                      rng.integers(0, 2, n) * 2 - 1).astype(int)
    return features, labels


def _sign_fitter(features):
    """A fitter that always predicts the sign of column 0 - right on the first
    half of the decaying dataset and coin-flip on the second."""
    def fit_predict(_train_rows, test_rows):
        return (features[test_rows, 0] > 0).astype(float)
    return fit_predict


def _run(tmp_path, features, labels, **kwargs):
    registry = kwargs.pop("registry", None) or _registry(tmp_path)
    return walk_forward(labels, _sign_fitter(features),
                        scheme=kwargs.pop("scheme", WindowScheme.EXPANDING),
                        test_rows=100, initial_train_rows=200, family=_FAMILY,
                        registry=registry,
                        trial_name=kwargs.pop("trial_name", "test"),
                        **kwargs), registry


def test_a_dead_edge_is_visible_per_window_and_not_in_the_pool(tmp_path):
    """The module's whole argument. Pooled it reads as a working strategy."""
    result, _ = _run(tmp_path, *_decaying_dataset())

    assert result.pooled_accuracy > result.pooled_base_rate + 0.1, (
        "pooled, this looks like an edge - which is the point")
    early = [w.accuracy for w in result.windows[:4]]
    late = [w.accuracy for w in result.windows[-4:]]
    assert min(early) > 0.9 and max(late) < 0.7, (
        "and per window it is alive then dead")
    assert result.decay is not None
    assert result.decay.is_significant
    assert "DECAYED" in result.describe()


def test_a_stable_edge_is_not_reported_as_decayed(tmp_path):
    """A decay verdict that fires on everything is a verdict nobody can act on."""
    rng = np.random.default_rng(3)
    n = 1200
    features = rng.normal(size=(n, 2))
    labels = np.sign(features[:, 0]).astype(int)
    result, _ = _run(tmp_path, features, labels)

    assert not result.decay.is_significant
    assert "no measured decay" in result.describe()


def test_too_few_windows_reports_no_decay_test_rather_than_no_decay(tmp_path):
    """"We could not test" and "we tested and found nothing" are different
    findings, and only one of them is reassuring."""
    features, labels = _decaying_dataset(n=500)
    result, _ = _run(tmp_path, features, labels)

    assert len(result.windows) < MIN_WINDOWS_FOR_DECAY
    assert result.decay is None
    assert "too few windows" in result.describe()


def test_every_window_is_scored_against_its_own_base_rate(tmp_path):
    """A window's accuracy is unreadable without it - the class balance drifts
    across a series, and a 0.62 in a 0.60 window is not a 0.62 in a 0.50 one."""
    result, _ = _run(tmp_path, *_decaying_dataset())

    assert all(0.0 <= w.base_rate <= 1.0 for w in result.windows)
    assert len({round(w.base_rate, 3) for w in result.windows}) > 1


def test_the_per_window_series_is_carried_into_the_registry(tmp_path):
    """A stored pooled number alone would let a reader of the ledger repeat
    exactly the mistake this module exists to prevent."""
    registry = _registry(tmp_path)
    _run(tmp_path, *_decaying_dataset(), registry=registry)

    stored = registry.trials()[0]["result"]
    assert len(stored["window_accuracies"]) == 10
    assert stored["decay_is_significant"] is True


# --- counting -------------------------------------------------------------

def test_a_whole_run_is_one_trial_not_one_per_window(tmp_path):
    """Ten windows fit ten models and answer ONE question. Registering ten would
    inflate N tenfold and deflate every downstream gate by a search nobody
    performed."""
    registry = _registry(tmp_path)
    _run(tmp_path, *_decaying_dataset(), registry=registry)

    assert registry.cumulative_count() == 1
    assert registry.trials()[0]["result"]["n_windows"] == 10


def test_the_scheme_and_its_lookback_land_in_the_trial(tmp_path):
    """Both are hyperparameters. A run whose scheme is not recorded cannot be
    compared with one that chose the other."""
    registry = _registry(tmp_path)
    features, labels = _decaying_dataset()
    walk_forward(labels, _sign_fitter(features), scheme=WindowScheme.ROLLING,
                 test_rows=100, initial_train_rows=300, family=_FAMILY,
                 lookback_rows=150, registry=registry, trial_name="rolling")

    params = registry.trials()[0]["params"]
    assert params["scheme"] == "rolling"
    assert params["lookback_rows"] == 150


def test_no_sharpe_is_recorded(tmp_path):
    """A classifier produces predictions, not positions - the same accounting
    `models.gradient_boosted_trees` uses."""
    registry = _registry(tmp_path)
    _run(tmp_path, *_decaying_dataset(), registry=registry)

    assert registry.trials()[0]["result"]["sharpe"] is None
    assert registry.trial_sharpes() == []
