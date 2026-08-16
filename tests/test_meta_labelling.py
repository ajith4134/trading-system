"""One line decides whether the secondary model helps or inverts the system.

`meta_label = 1 if barrier_label == side else 0` is the module. Get the sign
wrong and the dataset is still well-formed, the base rate is still plausible, it
still trains without complaint, and the only symptom is a live system that loses
money in proportion to its confidence. That is what most of this file defends.

The rest defends the default that would be worse: labelling an unresolved event
`0`. Unresolved events cluster at the NEWEST end of every dataset - the stretch
closest to live - so zeroing them teaches a secondary model to veto recent
signals as a class.
"""
import pytest

from features.meta_labelling import (
    MAX_WORKABLE_BASE_RATE,
    MIN_MINORITY_EVENTS,
    MIN_WORKABLE_BASE_RATE,
    InvalidSide,
    MetaEvent,
    MetaLabelSet,
    UnalignedSides,
    make_meta_labels,
)
from features.triple_barrier import BarrierTouch


def _touch(index, label, *, ambiguous=False):
    return BarrierTouch(
        event_index=index, label=label,
        reason="upper" if label == 1 else "lower" if label == -1 else "unresolved",
        touched_at_index=None if label is None else index + 1,
        entry_price=100.0, upper_barrier=102.0, lower_barrier=98.0,
        is_ambiguous=ambiguous)


# --- the sign -------------------------------------------------------------

def test_a_long_that_was_right_is_labelled_one():
    result = make_meta_labels([_touch(0, 1)], [1])

    assert [event.meta_label for event in result.events] == [1]


def test_a_short_that_was_right_is_labelled_one():
    """The test the whole module rests on.

    A barrier label of -1 means price reached the LOWER barrier first. Under a
    SHORT call that is a WIN. An implementation reading the barrier label as the
    outcome regardless of side gets 0 here, produces a perfectly well-formed
    dataset, and teaches the secondary model to veto exactly the calls it should
    approve.
    """
    result = make_meta_labels([_touch(0, -1)], [-1])

    assert [event.meta_label for event in result.events] == [1]


def test_a_long_into_a_lower_barrier_is_labelled_zero():
    result = make_meta_labels([_touch(0, -1)], [1])

    assert [event.meta_label for event in result.events] == [0]


def test_a_short_into_an_upper_barrier_is_labelled_zero():
    result = make_meta_labels([_touch(0, 1)], [-1])

    assert [event.meta_label for event in result.events] == [0]


def test_the_grade_follows_the_side_not_the_price():
    """Same paths, opposite calls: every grade must flip."""
    touches = [_touch(i, 1 if i % 2 else -1) for i in range(10)]
    longs = make_meta_labels(touches, [1] * 10)
    shorts = make_meta_labels(touches, [-1] * 10)

    for long_event, short_event in zip(longs.events, shorts.events):
        assert long_event.meta_label != short_event.meta_label


# --- what is excluded, and what is never invented -------------------------

def test_an_unresolved_event_is_excluded_not_labelled_zero():
    """The most damaging default available here. Unresolved events sit at the
    newest end of every dataset."""
    result = make_meta_labels([_touch(0, 1), _touch(1, None), _touch(2, -1)],
                              [1, 1, -1])

    assert result.graded == 2
    assert result.unresolved == 1
    assert all(event.barrier_label is not None for event in result.events)


def test_an_ambiguous_touch_is_excluded_and_counted_apart():
    """A bar that touched both barriers has an unknown intra-bar path, so the
    outcome is a convention. Counted separately from unresolved because the two
    call for different work - one wants more data, the other finer bars."""
    result = make_meta_labels([_touch(0, 1), _touch(1, 1, ambiguous=True)],
                              [1, 1])

    assert result.graded == 1
    assert result.ambiguous == 1
    assert result.unresolved == 0


def test_meta_labelling_can_never_produce_more_positives_than_signals():
    """Structural: the secondary model only sees calls the primary already made,
    so it can veto and never create. An implementation reporting otherwise has a
    pairing bug, and the object refuses to hold the state at all."""
    with pytest.raises(ValueError, match="unreachable by construction"):
        MetaLabelSet(events=[MetaEvent(0, 1, 1, 1), MetaEvent(1, 1, 1, 1)],
                     primary_signals=1, unresolved=0, ambiguous=0)


def test_every_primary_signal_is_accounted_for():
    """graded + unresolved + ambiguous == primary_signals, always. A dataset
    that quietly shrank reads exactly like a primary that fired less often."""
    touches = ([_touch(i, 1) for i in range(5)]
               + [_touch(5 + i, None) for i in range(3)]
               + [_touch(8 + i, -1, ambiguous=True) for i in range(2)])
    result = make_meta_labels(touches, [1] * 10)

    assert result.primary_signals == 10
    assert result.graded + result.unresolved + result.ambiguous == 10


# --- refusals -------------------------------------------------------------

def test_mismatched_lengths_are_refused_rather_than_zipped():
    """After the first mismatch every side is attached to the wrong event, and
    the resulting dataset is well-formed and wrong."""
    with pytest.raises(UnalignedSides):
        make_meta_labels([_touch(0, 1), _touch(1, 1)], [1])


def test_a_zero_side_is_refused_by_name():
    """'No position' is not a call a secondary model can grade, and admitting it
    would add a third class to a binary problem."""
    with pytest.raises(InvalidSide, match="third class"):
        make_meta_labels([_touch(0, 1)], [0])


# --- the base rate, which says whether any of this is worth doing ---------

def test_an_empty_set_has_no_base_rate_rather_than_a_zero_one():
    """'The primary was never right' and 'the primary was never graded' are
    different findings, and 0.0 says the first while meaning the second."""
    result = make_meta_labels([_touch(0, None)], [1])

    assert result.base_rate is None
    assert result.is_degenerate


def test_a_primary_that_is_almost_always_right_is_degenerate():
    """Nothing left to filter, and a class balance that will make the secondary
    model predict 1 always."""
    n = 200
    touches = [_touch(i, 1) for i in range(n)]
    sides = [1] * n
    # Two per cent wrong: base rate 0.98, above the workable ceiling.
    for i in range(0, n, 50):
        sides[i] = -1
    result = make_meta_labels(touches, sides)

    assert result.base_rate > MAX_WORKABLE_BASE_RATE
    assert result.is_degenerate


def test_a_primary_that_is_almost_always_wrong_is_degenerate():
    """Not a primary model - a sign the side rule is inverted."""
    n = 200
    result = make_meta_labels([_touch(i, 1) for i in range(n)], [-1] * n)

    assert result.base_rate < MIN_WORKABLE_BASE_RATE
    assert result.is_degenerate


def test_a_workable_base_rate_with_too_small_a_minority_is_still_degenerate():
    """The rate can sit inside the band while the minority class cannot be
    learned - the check is on both, because either alone passes something the
    other rejects."""
    n = 40
    sides = [1] * n
    wrong = MIN_MINORITY_EVENTS - 5              # inside the band, too few
    result = make_meta_labels([_touch(i, 1) for i in range(n)],
                              [-1] * wrong + sides[wrong:])

    assert MIN_WORKABLE_BASE_RATE <= result.base_rate <= MAX_WORKABLE_BASE_RATE
    assert result.is_degenerate


def test_a_balanced_set_with_enough_of_both_classes_is_workable():
    """The positive case, so 'degenerate' is a judgement rather than a constant."""
    n = 200
    sides = [1 if i % 2 else -1 for i in range(n)]
    result = make_meta_labels([_touch(i, 1) for i in range(n)], sides)

    assert result.base_rate == pytest.approx(0.5)
    assert not result.is_degenerate
    assert "workable" in result.describe()


def test_the_description_states_the_exclusions_beside_the_number():
    """A base rate without what was dropped to produce it is a number nobody
    can weight."""
    touches = [_touch(i, 1) for i in range(60)] + [_touch(60, None)]
    result = make_meta_labels(touches, [1] * 61)

    described = result.describe()
    assert "unresolved" in described and "61 primary signal(s)" in described
