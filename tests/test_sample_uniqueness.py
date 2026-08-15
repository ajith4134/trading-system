"""FE-009 / VX-102 — overlapping labels are not independent samples.

`FEATURES.md` §2: *"Sample uniqueness + sequential bootstrap — overlapping labels
violate IID."* Ledger VX-102 adds the honest caveat: it is *"the least contested
part of the AFML toolkit, but sequential bootstrap is inherently
non-parallelisable — a real engineering expense."*

**Why it matters here specifically.** FE-008 labels an event by the first barrier
its path touches, so a label's information window runs from its event to its
touch. Two events three bars apart with ten-bar holding periods share seven bars
of the same price path. They are not two observations. A model trained on them as
though they were is being told the same story twice and counting it as
corroboration, and every statistic computed from that count — the trial N, the
deflated Sharpe, the standard error on any backtest — inherits the inflation.

The correction is not cosmetic. On a dataset where labels overlap ten-deep, the
effective sample size is roughly a tenth of the row count, and a Sharpe
significance test using the row count is wrong by a factor of about three.

Three things are defended here:

**Concurrency is counted over the label's actual span**, event to touch, not over
the maximum holding period. FE-008 exists partly to supply that touch, and
assuming the maximum would over-count every fast trade's overlap.

**An unresolved label has no span.** FE-008 returns `None` for an event whose
window ran past the data. It cannot contribute concurrency, and silently treating
it as spanning to the end of the series would inflate the overlap of everything
near the tail.

**Sequential bootstrap has to actually work.** The test is empirical, not
structural: draws made sequentially must achieve measurably higher average
uniqueness than uniform draws on the same overlapping spans. A "sequential
bootstrap" that does not beat uniform sampling is an expensive loop.
"""
import pytest

from features.sample_uniqueness import (
    LabelSpan,
    UnresolvedLabel,
    average_uniqueness,
    label_concurrency,
    sequential_bootstrap,
    spans_from_touches,
)


def _spans(pairs):
    return [LabelSpan(event_index=a, touched_at_index=b) for a, b in pairs]


# --- concurrency -------------------------------------------------------------

def test_a_single_label_is_concurrent_with_itself_across_its_span():
    counts = label_concurrency(_spans([(0, 2)]), n_bars=4)
    assert counts == [1, 1, 1, 0]


def test_two_disjoint_labels_never_overlap():
    counts = label_concurrency(_spans([(0, 1), (2, 3)]), n_bars=4)
    assert counts == [1, 1, 1, 1]


def test_overlapping_labels_stack_on_the_bars_they_share():
    counts = label_concurrency(_spans([(0, 3), (2, 5)]), n_bars=6)
    #        bar: 0  1  2  3  4  5
    assert counts == [1, 1, 2, 2, 1, 1]


def test_a_bar_no_label_covers_has_zero_concurrency():
    counts = label_concurrency(_spans([(0, 1)]), n_bars=4)
    assert counts[3] == 0


def test_concurrency_uses_the_touch_not_the_maximum_holding_period():
    """FE-008 exists partly to supply the real touch. Assuming the maximum would
    over-count the overlap of every trade that resolved early, and over-counting
    overlap under-counts the effective sample size in the safe-looking
    direction - it makes the data seem MORE dependent than it is, which is not
    the flattering error, but it is still wrong and it discards real samples."""
    early = label_concurrency(_spans([(0, 1), (2, 3)]), n_bars=4)
    assert max(early) == 1


# --- average uniqueness ------------------------------------------------------

def test_a_label_with_no_overlap_is_perfectly_unique():
    (uniqueness,) = average_uniqueness(_spans([(0, 3)]), n_bars=4)
    assert uniqueness == 1.0


def test_two_labels_sharing_every_bar_are_each_half_unique():
    values = average_uniqueness(_spans([(0, 3), (0, 3)]), n_bars=4)
    assert values == [0.5, 0.5]


def test_partial_overlap_gives_a_uniqueness_between_the_two():
    # spans 0-3 and 2-5 over 6 bars: concurrency [1,1,2,2,1,1].
    # first label covers bars 0..3 -> mean(1, 1, .5, .5) = 0.75
    first, second = average_uniqueness(_spans([(0, 3), (2, 5)]), n_bars=6)
    assert first == pytest.approx(0.75)
    assert second == pytest.approx(0.75)


def test_uniqueness_falls_as_more_labels_pile_onto_the_same_bars():
    alone = average_uniqueness(_spans([(0, 3)]), n_bars=4)[0]
    crowded = average_uniqueness(_spans([(0, 3)] * 4), n_bars=4)[0]
    assert crowded < alone
    assert crowded == pytest.approx(0.25)


# --- unresolved labels have no span ------------------------------------------

def test_an_unresolved_touch_is_refused_rather_than_spanning_to_the_end():
    """FE-008 returns None for an event whose window ran past the data.
    Treating that as spanning to the end of the series would inflate the overlap
    of everything near the tail - the newest data, and the part a model is most
    likely to be evaluated on."""
    from features.triple_barrier import BarrierTouch
    unresolved = BarrierTouch(event_index=1, label=None, reason="unresolved",
                              touched_at_index=None, entry_price=100.0,
                              upper_barrier=102.0, lower_barrier=98.0,
                              is_ambiguous=False)
    with pytest.raises(UnresolvedLabel):
        spans_from_touches([unresolved])


def test_resolved_touches_convert_to_spans():
    from features.triple_barrier import BarrierTouch
    resolved = BarrierTouch(event_index=1, label=1, reason="profit_take",
                            touched_at_index=4, entry_price=100.0,
                            upper_barrier=102.0, lower_barrier=98.0,
                            is_ambiguous=False)
    (span,) = spans_from_touches([resolved])
    assert span.event_index == 1 and span.touched_at_index == 4


def test_unresolved_touches_can_be_dropped_deliberately():
    """Excluding them must be an explicit act, never a silent default."""
    from features.triple_barrier import BarrierTouch
    touches = [
        BarrierTouch(event_index=0, label=1, reason="profit_take",
                     touched_at_index=2, entry_price=100.0, upper_barrier=102.0,
                     lower_barrier=98.0, is_ambiguous=False),
        BarrierTouch(event_index=3, label=None, reason="unresolved",
                     touched_at_index=None, entry_price=100.0,
                     upper_barrier=102.0, lower_barrier=98.0,
                     is_ambiguous=False),
    ]
    spans = spans_from_touches(touches, drop_unresolved=True)
    assert len(spans) == 1


# --- sequential bootstrap ----------------------------------------------------

def test_the_bootstrap_draws_the_number_of_samples_asked_for():
    draws = sequential_bootstrap(_spans([(0, 3), (2, 5), (4, 7)]), n_bars=8,
                                 size=5, seed=1)
    assert len(draws) == 5


def test_every_draw_is_a_valid_label_index():
    spans = _spans([(0, 3), (2, 5), (4, 7)])
    draws = sequential_bootstrap(spans, n_bars=8, size=20, seed=1)
    assert all(0 <= d < len(spans) for d in draws)


def test_the_bootstrap_is_reproducible_under_a_seed():
    spans = _spans([(0, 3), (2, 5), (4, 7)])
    a = sequential_bootstrap(spans, n_bars=8, size=10, seed=7)
    b = sequential_bootstrap(spans, n_bars=8, size=10, seed=7)
    assert a == b


def test_different_seeds_give_different_draws():
    spans = _spans([(0, 5), (1, 6), (2, 7), (3, 8)])
    a = sequential_bootstrap(spans, n_bars=9, size=20, seed=1)
    b = sequential_bootstrap(spans, n_bars=9, size=20, seed=2)
    assert a != b


def test_sequential_draws_are_more_unique_than_uniform_ones():
    """The empirical claim, and the only one that matters. A sequential
    bootstrap that does not beat uniform sampling on overlapping data is an
    expensive loop with a good name.

    **The first version of this test could not have failed.** It drew 20 labels
    of equal length from spans covering the whole series, and both methods scored
    0.13636... to every digit. For equal-length spans there is an identity:

        mean average-uniqueness of a drawn set
            = (distinct bars the set covers) / (n_draws x span length)

    because summing 1/concurrency over every label's covered bars just counts the
    bars with any coverage at all. Once a draw saturates the series, distinct
    coverage is pinned at the series length and the metric cannot move, whatever
    the sampler does. The regime here leaves room for coverage to differ, and the
    comparison is averaged over seeds so it is a claim about the method rather
    than about one lucky draw.
    """
    import random
    from statistics import mean as _mean
    # Forty labels of five bars over two hundred, drawing eight: coverage can
    # range from 5 bars (all draws identical) to 40 (all disjoint).
    spans = _spans([(i * 5, i * 5 + 4) for i in range(40)])
    n_bars = 200
    size = 8

    def mean_uniqueness(draw):
        drawn = [spans[i] for i in draw]
        return sum(average_uniqueness(drawn, n_bars=n_bars)) / len(drawn)

    sequential_scores, uniform_scores = [], []
    for seed in range(30):
        sequential_scores.append(mean_uniqueness(
            sequential_bootstrap(spans, n_bars=n_bars, size=size, seed=seed)))
        rng = random.Random(seed)
        uniform_scores.append(mean_uniqueness(
            [rng.randrange(len(spans)) for _ in range(size)]))

    assert _mean(sequential_scores) > _mean(uniform_scores)


def test_the_uniqueness_identity_that_broke_the_first_version_of_that_test():
    """Pinned, because it is the reason a plausible test proved nothing. For
    equal-length spans, mean uniqueness is distinct-coverage over total slots -
    so any comparison made where coverage is saturated is measuring a constant."""
    spans = _spans([(0, 4), (10, 14)])       # two disjoint 5-bar spans
    values = average_uniqueness(spans, n_bars=20)
    distinct_covered = 10
    assert sum(values) / len(values) == pytest.approx(
        distinct_covered / (len(spans) * 5))


def test_the_bootstrap_prefers_the_label_that_overlaps_least():
    """One isolated label among a tight cluster. Uniform sampling picks it 1 in
    4; sequential sampling should reach for it far more often."""
    spans = _spans([(0, 2), (0, 2), (0, 2), (10, 12)])
    draws = sequential_bootstrap(spans, n_bars=13, size=200, seed=5)
    assert draws.count(3) > size_quarter(200)


def size_quarter(n):
    return n // 4


# --- refusals ----------------------------------------------------------------

def test_a_span_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError):
        LabelSpan(event_index=5, touched_at_index=2)


def test_a_span_reaching_past_the_series_is_refused():
    with pytest.raises(IndexError):
        label_concurrency(_spans([(0, 9)]), n_bars=4)


def test_an_empty_label_set_is_refused_rather_than_returning_nothing():
    """Silently returning an empty weight vector would let a training run
    proceed with no samples and no complaint."""
    with pytest.raises(ValueError):
        average_uniqueness([], n_bars=4)


def test_a_bootstrap_of_size_zero_is_refused():
    with pytest.raises(ValueError):
        sequential_bootstrap(_spans([(0, 1)]), n_bars=2, size=0, seed=1)
