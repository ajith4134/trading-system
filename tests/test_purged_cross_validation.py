"""CPCV with purge and embargo, configured per family — VX-004 and VX-007.

Combinatorial Purged Cross-Validation (López de Prado, *Advances in Financial
Machine Learning*, ch. 7 & 12) replaces one walk-forward Sharpe with a
*distribution* of Sharpes over C(n_groups, k_test) recombinations. That
distribution is what the Deflated Sharpe and PBO gates need; a single number
cannot tell selection bias from skill.

Two donor implementations were read raw before writing this, both flagged in the
ledger as portable prior art. Neither was:

**`nse-crypto-bot-final/trading/strategy/cpcv.py`** has the right combinatorics
and real purge machinery, but its exclusion zone is `[test_start, test_end +
embargo)`. Its own docstring states the contract correctly - *"drop training rows
whose label window overlaps a test block"* - and the code implements something
narrower: drop training rows at or after the test start. **The two differ by
exactly the label horizon on the left edge.** A training row that begins one bar
before the test block, with a label that matures inside it, is trained on an
outcome the test block is about to be scored on. That row leaks, and it is kept.

**`nse-botonly/.../combinatorial_purged_cross_validation.py`** purges nothing at
all - defensibly, since it operates on a label-free realized-returns series - but
keeps `embargo_group_count: int = 1` as a config knob wired to nothing.

So the fix, and the reason VX-004's per-family configuration is not a nicety: the
purge zone is `[test_start − label_horizon, test_end + embargo)`, and **without a
label horizon there is no correct left edge to purge to.** `FEATURES.md` §8 notes
horizons differ by orders of magnitude between strategy families - a funding-carry
label matures in eight hours, a trend label in weeks - so the horizon has to come
from the family, and a family nobody declared is a refusal rather than a zero.
"""
import pytest

from validation.purged_cross_validation import (
    FAMILY_LABEL_HORIZONS,
    UnknownStrategyFamily,
    combinatorial_purged_folds,
    n_cpcv_paths,
    purge_and_embargo,
)


# --- the combinatorics ------------------------------------------------------

def test_the_number_of_paths_is_the_binomial_coefficient():
    """C(6,2) = 15 paths, versus 4 or 5 from rolling walk-forward. The richer
    distribution is the entire reason to pay for the combinatorics."""
    assert n_cpcv_paths(n_groups=6, k_test=2) == 15
    assert n_cpcv_paths(n_groups=10, k_test=2) == 45


def test_every_combination_of_test_groups_appears_exactly_once():
    folds = combinatorial_purged_folds(n_rows=600, family="trend",
                                       n_groups=6, k_test=2)
    groups = [f.test_groups for f in folds]
    assert len(groups) == 15
    assert len(set(groups)) == 15


def test_a_meaningful_cpcv_needs_at_least_three_groups():
    with pytest.raises(ValueError):
        combinatorial_purged_folds(n_rows=600, family="trend", n_groups=2, k_test=1)


def test_too_few_rows_for_the_requested_groups_is_refused():
    """Refused rather than silently producing empty groups, which would yield a
    distribution of Sharpes computed over nothing."""
    with pytest.raises(ValueError):
        combinatorial_purged_folds(n_rows=5, family="trend", n_groups=6, k_test=2)


def test_the_folds_are_deterministic():
    a = combinatorial_purged_folds(600, family="trend", n_groups=5, k_test=2)
    b = combinatorial_purged_folds(600, family="trend", n_groups=5, k_test=2)
    assert [(f.test_groups, f.train_blocks) for f in a] == \
           [(f.test_groups, f.train_blocks) for f in b]


# --- the property that matters: no train index is ever a test index ---------

def test_no_training_index_is_also_a_testing_index():
    for fold in combinatorial_purged_folds(600, family="trend",
                                           n_groups=6, k_test=2):
        train = {i for s, e in fold.train_blocks for i in range(s, e)}
        test = {i for s, e in fold.test_blocks for i in range(s, e)}
        assert not (train & test), f"train/test overlap in fold {fold.test_groups}"


# --- the donor's bug, pinned ------------------------------------------------

def test_purge_removes_training_rows_whose_label_matures_inside_the_test_block():
    """**The defect in the donor implementation.** A training row at index 95 with
    a 10-bar label horizon has a label that matures at 105 - inside a test block
    starting at 100. It was trained on an outcome the test block will be scored
    on. `cpcv.py`'s zone starts at the test start, so index 95 survives and the
    leak is silent."""
    kept = purge_and_embargo(train=(0, 100), test_blocks=[(100, 200)],
                             label_horizon=10, embargo=0)
    kept_indices = {i for s, e in kept for i in range(s, e)}

    assert 95 not in kept_indices, (
        "a training row whose label matures inside the test block survived the "
        "purge - this is the one-sided-purge bug from the donor cpcv.py")
    assert 89 in kept_indices, "the purge reached further back than the horizon"


def test_the_purge_zone_extends_exactly_one_horizon_before_the_test_block():
    kept = purge_and_embargo(train=(0, 100), test_blocks=[(100, 200)],
                             label_horizon=10, embargo=0)
    kept_indices = {i for s, e in kept for i in range(s, e)}
    assert max(kept_indices) == 89, f"left edge is wrong: kept up to {max(kept_indices)}"


def test_a_zero_label_horizon_still_purges_the_test_block_itself():
    """A signal scored on the same bar has no label window, so nothing before the
    test block leaks - but the test block's own rows must still not be trained on."""
    kept = purge_and_embargo(train=(0, 300), test_blocks=[(100, 200)],
                             label_horizon=0, embargo=0)
    kept_indices = {i for s, e in kept for i in range(s, e)}
    assert 99 in kept_indices
    assert not any(100 <= i < 200 for i in kept_indices)


def test_embargo_removes_training_rows_immediately_after_the_test_block():
    """Serial correlation runs forward too: a training row just after the test
    block carries information about it even with no label overlap."""
    kept = purge_and_embargo(train=(0, 300), test_blocks=[(100, 200)],
                             label_horizon=0, embargo=25)
    kept_indices = {i for s, e in kept for i in range(s, e)}
    assert 224 not in kept_indices
    assert 225 in kept_indices


def test_purge_and_embargo_apply_around_every_test_block_not_just_the_first():
    """k_test > 1 means several disjoint test blocks per path. A purge that only
    handles the first leaks around all the others."""
    kept = purge_and_embargo(train=(0, 500), test_blocks=[(100, 150), (300, 350)],
                             label_horizon=10, embargo=10)
    kept_indices = {i for s, e in kept for i in range(s, e)}
    for leaked in (95, 145, 155, 295, 345, 355):
        assert leaked not in kept_indices, f"index {leaked} survived around a test block"
    assert 50 in kept_indices and 250 in kept_indices and 450 in kept_indices


def test_a_train_range_entirely_inside_the_purge_zone_disappears():
    """Not an error - a legitimately empty result. Silently returning the range
    unpurged would be the dangerous alternative."""
    assert purge_and_embargo(train=(100, 150), test_blocks=[(100, 200)],
                             label_horizon=0, embargo=0) == []


# --- per-family horizons ----------------------------------------------------

def test_families_have_different_declared_label_horizons():
    """`FEATURES.md` §8: horizons differ by orders of magnitude between families.
    A funding-carry label matures at the next settlement; a trend label in weeks."""
    assert FAMILY_LABEL_HORIZONS["carry"] != FAMILY_LABEL_HORIZONS["trend"]


def test_an_undeclared_family_is_refused_rather_than_defaulted():
    """Fail closed. A default horizon of zero is the same as no purge, and it
    would be applied precisely to the family nobody thought about - which is the
    family most likely to have a long label window."""
    with pytest.raises(UnknownStrategyFamily):
        combinatorial_purged_folds(600, family="some-new-idea",
                                   n_groups=6, k_test=2)


def test_a_longer_horizon_purges_more_training_data():
    short = combinatorial_purged_folds(6000, family="carry", n_groups=6, k_test=2)
    long = combinatorial_purged_folds(6000, family="trend", n_groups=6, k_test=2)

    def kept(folds):
        return sum(e - s for f in folds for s, e in f.train_blocks)

    assert kept(long) < kept(short), (
        "the longer-horizon family did not lose more training data to purging")


def test_an_explicit_horizon_overrides_the_family_default():
    """The family gives the default; a strategy that knows its own label window
    states it. The override exists so a correct horizon is always expressible."""
    folds = combinatorial_purged_folds(6000, family="carry", n_groups=6, k_test=2,
                                       label_horizon=2000)
    kept = sum(e - s for f in folds for s, e in f.train_blocks)
    default = sum(e - s for f in combinatorial_purged_folds(
        6000, family="carry", n_groups=6, k_test=2) for s, e in f.train_blocks)
    assert kept < default


# --- the fold reports what it cost ------------------------------------------

def test_each_fold_reports_how_much_training_data_purging_cost():
    """Rule 8 on a statistical harness: a fold that quietly lost 80% of its
    training data still returns a number, and that number means much less. The
    cost has to be visible where the result is."""
    fold = combinatorial_purged_folds(6000, family="trend",
                                      n_groups=6, k_test=2)[0]
    assert fold.rows_purged > 0
    assert 0.0 < fold.train_fraction_retained < 1.0
