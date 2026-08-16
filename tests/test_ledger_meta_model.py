"""A meta-model over thirty trials is a lookup table with an unearned p-value.

Every other model in this package is fitted on market data, of which there is a
lot. This one is fitted on the experiment ledger, of which there are a few dozen
rows. So the module separates the descriptive half - honest at any N - from the
fit, and refuses the fit below a declared floor rather than producing
coefficients that describe the sample.

The refusal is a returned VALUE, not an exception, because "not yet" is the
expected state for months and a board should render it without a try block.
"""
import pytest

from models.ledger_meta_model import (
    MIN_TEST_TRIALS,
    MIN_TRIALS_TO_FIT,
    FitRefused,
    MetaModelFit,
    TrialOutcome,
    fit,
    summarise,
)


def _outcome(index, family, regime, succeeded):
    return TrialOutcome(trial_id=index, family=family, regime=regime,
                        succeeded=succeeded, ran_at_ns=1_000 + index)


def _ledger(n, *, families=("carry", "trend"), regimes=("calm", "violent"),
            rule=None):
    """`n` trials cycling through the families and regimes.

    `rule(family, regime)` decides success, so a test can plant a relationship
    or refuse to.
    """
    rule = rule or (lambda family, regime: family == "carry")
    out = []
    for i in range(n):
        family = families[i % len(families)]
        regime = regimes[(i // len(families)) % len(regimes)]
        out.append(_outcome(i, family, regime, rule(family, regime)))
    return out


# --- the descriptive half -------------------------------------------------

def test_every_cell_carries_its_count(tmp_path):
    """A cell that succeeded once out of one reports 100%, and the count beside
    it is the only thing that stops that being read as a finding."""
    summary = summarise([_outcome(0, "carry", "calm", True)])

    assert summary.cells[0].success_rate == 1.0
    assert summary.cells[0].trials == 1


def test_the_rates_are_conditional_on_family_and_regime():
    outcomes = _ledger(40, rule=lambda family, regime: regime == "violent")
    summary = summarise(outcomes)

    by_key = {(c.family, c.regime): c for c in summary.cells}
    assert by_key[("carry", "violent")].success_rate == 1.0
    assert by_key[("carry", "calm")].success_rate == 0.0


def test_the_largest_family_share_is_reported():
    """A meta-model over a ledger where 80% of trials are one family learns
    where attention went. The number does not improve with N - only with a
    broader search."""
    outcomes = ([_outcome(i, "carry", "calm", True) for i in range(80)]
                + [_outcome(80 + i, "trend", "calm", False) for i in range(20)])
    summary = summarise(outcomes)

    assert summary.largest_family_share == pytest.approx(0.8)
    assert "80%" in summary.describe()


def test_abandoned_trials_are_reported_beside_every_rate():
    """A trial that raised has no outcome, so conditioning on completed trials
    conditions on completion. It cannot be closed structurally here - an
    abandoned trial genuinely has no outcome - so it is made visible."""
    summary = summarise(_ledger(20), abandoned=7)

    assert summary.abandoned == 7
    assert "7 abandoned" in summary.describe()


def test_an_empty_ledger_summarises_to_nothing_rather_than_raising():
    summary = summarise([])

    assert summary.cells == []
    assert summary.total_trials == 0
    assert summary.largest_family_share == 0.0


# --- the refusal ----------------------------------------------------------

def test_a_small_ledger_refuses_the_fit_and_says_by_how_much():
    """The refusal carries the count it has and the count it needs, so a board
    can render progress toward being able to answer."""
    refused = fit(_ledger(30))

    assert isinstance(refused, FitRefused)
    assert refused.trials == 30
    assert refused.required == MIN_TRIALS_TO_FIT
    assert "lookup table" in refused.reason


def test_the_refusal_is_a_value_not_an_exception():
    """"Not yet" is the expected state for months."""
    result = fit(_ledger(5))

    assert isinstance(result, FitRefused)


def test_a_ledger_with_too_few_later_trials_is_refused_too():
    """A 200-trial ledger with 5 distinct later trials is still 5 - the total
    passing the floor does not mean the forward split can score anything."""
    n = MIN_TRIALS_TO_FIT + 5
    result = fit(_ledger(n))
    later = n - int(n * (2 / 3))

    if later < MIN_TEST_TRIALS:
        assert isinstance(result, FitRefused)
        assert "after the forward split" in result.reason
    else:
        assert isinstance(result, MetaModelFit)


# --- the fit, once it is allowed ------------------------------------------

def test_a_real_relationship_is_found_once_the_ledger_is_big_enough():
    """The only test here that can report a fit. Without it, every refusal above
    is satisfied by a module that never fits anything."""
    outcomes = _ledger(600, rule=lambda family, regime: family == "carry")

    fitted = fit(outcomes)

    assert isinstance(fitted, MetaModelFit)
    assert fitted.converged
    assert fitted.test_accuracy > 0.9


def test_a_ledger_with_no_relationship_does_not_beat_its_base_rate():
    """Coefficients are not evidence that anything was learned.

    The outcome is drawn INDEPENDENTLY of family and regime. An earlier version
    of this fixture cycled families on `i % 2` and set success on `i % 2` as
    well, which encoded a perfect relationship while claiming to encode none -
    and the module found it, correctly.
    """
    import random

    rng = random.Random(11)
    outcomes = [TrialOutcome(o.trial_id, o.family, o.regime,
                             rng.random() < 0.5, o.ran_at_ns)
                for o in _ledger(600)]

    fitted = fit(outcomes)

    assert isinstance(fitted, MetaModelFit)
    assert not fitted.beats_base_rate


def test_the_split_is_chronological():
    """Random K-fold over the ledger would train on trials run AFTER the ones it
    tests, and the whole question is whether past experiments predict future
    ones. Checked by planting the relationship only in the later trials: a
    chronological fit cannot have learned it."""
    n = 600
    early = [_outcome(i, "carry", "calm", False) for i in range(n // 2)]
    late = [_outcome(n // 2 + i, "carry", "calm", True) for i in range(n // 2)]

    fitted = fit(early + late)

    assert isinstance(fitted, MetaModelFit)
    # Trained on all-False trials, tested on all-True ones: it must score 0.
    assert fitted.test_accuracy == pytest.approx(0.0)
    assert not fitted.beats_base_rate


def test_a_family_that_appears_only_after_the_split_creates_no_column():
    """Taking the one-hot levels from the whole ledger would let a family the
    fit never saw create a column - a lookahead through the schema rather than
    through the values."""
    n = 600
    early = [_outcome(i, "carry", "calm", i % 2 == 0) for i in range(n // 2)]
    late = [_outcome(n // 2 + i, "brand-new-family", "calm", True)
            for i in range(n // 2)]

    fitted = fit(early + late)

    assert isinstance(fitted, MetaModelFit)
    assert not any("brand-new-family" in name for name in fitted.coefficients)


def test_the_design_drops_a_level_of_each_factor():
    """Keeping every level alongside an intercept makes the design rank
    deficient, and a ridge solve on a rank-deficient design returns a valid
    answer to an ill-posed question - ordinary-looking and not comparable
    between fits."""
    fitted = fit(_ledger(600))

    assert isinstance(fitted, MetaModelFit)
    families = [n for n in fitted.coefficients if n.startswith("family=")]
    regimes = [n for n in fitted.coefficients if n.startswith("regime=")]
    assert len(families) == 1, "two families, one column"
    assert len(regimes) == 1, "two regimes, one column"


def test_equal_to_the_base_rate_is_not_reported_as_beating_it():
    """On a ledger this sparse, reproducing the majority class is the likeliest
    outcome and is not a result worth calling a pass."""
    outcomes = _ledger(600, rule=lambda family, regime: True)

    fitted = fit(outcomes)

    assert fitted.test_accuracy == pytest.approx(fitted.test_base_rate)
    assert not fitted.beats_base_rate
