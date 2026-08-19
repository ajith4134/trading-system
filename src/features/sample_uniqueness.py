"""Overlapping labels are not independent samples, and counting them as such lies.

`FEATURES.md` §2, FE-008's companion: *"Sample uniqueness + sequential bootstrap
— overlapping labels violate IID."* Ledger VX-102 carries the honest caveat that
it is *"the least contested part of the AFML toolkit, but sequential bootstrap is
inherently non-parallelisable — a real engineering expense."*

**Why this matters to this project specifically.** `features.triple_barrier`
labels an event by the first barrier its path touches, so a label's information
window runs from its event to its touch. Two events three bars apart, each with a
ten-bar holding period, share seven bars of the same price path. They are not two
observations of the market. A model trained on them as though they were is being
told one story twice and counting it as corroboration.

Everything downstream inherits that inflation: the standard error on any
backtest, the effective sample size the MinBTL gate compares against, the
significance of every result. On labels that overlap ten-deep, the effective
sample is roughly a tenth of the row count and a Sharpe test using the row count
is wrong by about a factor of three — in the direction that promotes things.

Three properties this module holds to:

**Concurrency is counted over the label's ACTUAL span**, event to touch.
`triple_barrier` exists partly to supply that touch; assuming the maximum holding
period instead would over-count the overlap of every trade that resolved early.

**An unresolved label has no span at all.** `BarrierTouch.touched_at_index` is
`None` when the data ran out before the window closed. Treating that as spanning
to the end of the series would inflate the measured overlap of everything near
the tail — the newest data, which is the part a model is most likely to be judged
on. Dropping them is allowed; doing so silently is not, so it takes a flag.

**The sequential bootstrap has to earn its cost.** Drawing proportional to
*remaining* uniqueness — recomputed after every draw, which is exactly why it
does not parallelise — must measurably beat uniform sampling on overlapping data.
A version that does not is an expensive loop with a good name, and there is a
test that would catch one.

**What it is actually worth, measured 2026-08-15** rather than assumed, because
VX-102 calls the non-parallelisable cost "a real engineering expense" and an
expense wants a benefit beside it. Forty labels of twenty bars stepping by two
over two hundred bars — each sharing ~90% of its window with its neighbours —
averaged over 100 seeds:

    draws    sequential   uniform    lift
        4       0.7570     0.7133    +6.1%
        8       0.5186     0.4816    +7.7%
       16       0.2931     0.2797    +4.8%

On lightly overlapping data (forty five-bar labels, eight draws) it is +3.5%,
and it produces a fully disjoint sample in 130 runs of 200 against uniform
sampling's 96. So: a real improvement, consistently in the right direction, and
NOT a transformation. Anyone budgeting engineering time against this should read
"5-8% more effective samples", not "fixes the IID problem".

**An identity worth knowing, because it invalidated the first test written for
this module.** For spans of equal length,

    mean average-uniqueness of a drawn set
        = (distinct bars the set covers) / (n_draws x span length)

since summing 1/concurrency over each label's covered bars just counts the bars
with any coverage at all. The first version of the empirical test drew enough
labels to saturate the series, where distinct coverage is pinned at the series
length — so both samplers scored 0.13636... to every digit and the test could not
have failed. It is pinned by its own test now.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


class UnresolvedLabel(ValueError):
    """A label whose barrier window never closed was handed in as a sample.

    It has no information window, so it has no span, so it cannot contribute
    concurrency. Refused rather than defaulted to the end of the series: that
    default would inflate the overlap of the newest labels, which are the ones
    closest to live.
    """


@dataclass(frozen=True)
class LabelSpan:
    """The bars over which one label's information window is open.

    Inclusive at both ends: a label resolved on the bar after its event used two
    bars of the path, and half-open arithmetic here would report it as one and
    quietly halve every concurrency count in the dataset.
    """

    event_index: int
    touched_at_index: int

    def __post_init__(self) -> None:
        if self.touched_at_index < self.event_index:
            raise ValueError(
                f"a label cannot resolve at bar {self.touched_at_index} before "
                f"its event at {self.event_index}")

    @property
    def length(self) -> int:
        return self.touched_at_index - self.event_index + 1


def spans_from_touches(touches, drop_unresolved: bool = False) -> list[LabelSpan]:
    """`BarrierTouch` records to spans, refusing the ones that never resolved.

    `drop_unresolved` exists so that excluding them is a decision somebody made
    and can be seen making, rather than a default that quietly shrinks a dataset.
    """
    spans: list[LabelSpan] = []
    for touch in touches:
        if touch.touched_at_index is None:
            if drop_unresolved:
                continue
            raise UnresolvedLabel(
                f"the label for the event at bar {touch.event_index} never "
                f"resolved ({touch.reason}), so it has no information window and "
                f"cannot contribute concurrency. Pass drop_unresolved=True to "
                f"exclude these deliberately - defaulting them to the end of the "
                f"series would inflate the overlap of every label near the tail")
        spans.append(LabelSpan(event_index=touch.event_index,
                               touched_at_index=touch.touched_at_index))
    return spans


def label_concurrency(spans: Sequence[LabelSpan], n_bars: int) -> list[int]:
    """How many labels' windows are open on each bar."""
    counts = [0] * n_bars
    for span in spans:
        if span.touched_at_index >= n_bars:
            raise IndexError(
                f"a label spans bar {span.touched_at_index} in a series of "
                f"{n_bars}. Refused rather than clipped - a clipped span reports "
                f"less overlap than really exists, which inflates the effective "
                f"sample size in the flattering direction")
        for index in range(span.event_index, span.touched_at_index + 1):
            counts[index] += 1
    return counts


def average_uniqueness(spans: Sequence[LabelSpan], n_bars: int) -> list[float]:
    """Each label's mean 1/concurrency across the bars it covers.

    1.0 means nothing else was open while this label was; 0.25 means it shared
    every one of its bars with three others and is worth a quarter of a sample.
    """
    if not spans:
        raise ValueError(
            "no labels supplied. Returning an empty weight vector would let a "
            "training run proceed with no samples and no complaint")
    counts = label_concurrency(spans, n_bars)
    out: list[float] = []
    for span in spans:
        covered = range(span.event_index, span.touched_at_index + 1)
        out.append(sum(1.0 / counts[i] for i in covered) / span.length)
    return out


def _uniqueness_given_drawn(spans: Sequence[LabelSpan], n_bars: int,
                            drawn_counts: list[int]) -> list[float]:
    """Each candidate's average uniqueness *if it were added next*.

    This is the whole method and the whole expense. The concurrency a candidate
    would face depends on what has already been drawn, so it cannot be computed
    once up front - which is precisely why VX-102 records the sequential
    bootstrap as inherently non-parallelisable.
    """
    out: list[float] = []
    for span in spans:
        covered = range(span.event_index, span.touched_at_index + 1)
        # +1 for the candidate itself: without it a bar nothing has drawn yet
        # divides by zero, and with it the figure is the concurrency this draw
        # would actually experience.
        out.append(sum(1.0 / (drawn_counts[i] + 1) for i in covered) / span.length)
    return out


def sequential_bootstrap(spans: Sequence[LabelSpan], n_bars: int, size: int,
                         seed: int = 0) -> list[int]:
    """Draw label indices with replacement, favouring the least-overlapped.

    Each draw is proportional to the uniqueness a candidate would have GIVEN what
    is already in the sample, so a label sharing its bars with three already-drawn
    labels becomes a quarter as likely as an isolated one. The result is a sample
    that behaves far more like an IID draw than uniform sampling does on the same
    data.

    Reproducible under `seed`: a resampling scheme whose answer moves between runs
    is one people re-run until it agrees with them.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if not spans:
        raise ValueError("no labels to draw from")
    label_concurrency(spans, n_bars)          # bounds check, before any drawing

    rng = random.Random(seed)
    drawn_counts = [0] * n_bars
    picks: list[int] = []

    for _ in range(size):
        weights = _uniqueness_given_drawn(spans, n_bars, drawn_counts)
        total = sum(weights)
        if total <= 0:
            raise ValueError(
                "every candidate has zero uniqueness, which cannot happen for "
                "non-empty spans - refusing rather than falling back to a "
                "uniform draw that would silently undo the whole correction")
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = len(spans) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen = index
                break
        picks.append(chosen)
        span = spans[chosen]
        for index in range(span.event_index, span.touched_at_index + 1):
            drawn_counts[index] += 1

    return picks


def average_uniqueness_by_group(spans: Sequence[LabelSpan],
                                groups: Sequence) -> list[float]:
    """Uniqueness computed WITHIN each series, then reassembled in input order.

    **The defect this exists to fix, measured 2026-08-19.** `learn.training_set`
    builds every span with an index local to its own symbol - `event_index=i`
    where `i` restarts at 0 for each symbol - and then appends them all into one
    flat list. Handing that to `average_uniqueness` counts concurrency on ONE
    SHARED TIMELINE, so symbol A's bar 500 and symbol B's bar 500 are treated as
    the same bar. Uniqueness then divides by the number of symbols:

        1 symbol, genuine 10-deep overlap   mean uniqueness 0.10000
        10 symbols, identical labels                        0.01000
        100 symbols, identical labels                       0.00100
        800 symbols, identical labels                       0.00013

    Identical labels; only the symbol count changes. The live perp model reported
    0.00223 over 573 symbols and spot 0.00186 over 1,318 - and dividing those back
    out gives implied per-symbol uniqueness of 1.28 and 2.45, both ABOVE the 1.0
    ceiling uniqueness has, which is the arithmetic proof that the pooling did it.

    **Why it is wrong in principle, not just in scale.** Uniqueness measures how
    much a label's information window overlaps OTHER LABELS ON THE SAME PRICE
    PATH: two events three bars apart with ten-bar horizons share seven bars of
    one story. Two symbols labelled at the same instant share no bars at all -
    they are two observations. Pooling them conflates "we watch many symbols"
    with "we were told one story many times", so the metric ends up measuring the
    size of the universe.

    Cross-sectional correlation IS real and severe in crypto - 800 correlated
    streams are not 800 independent bets - but that is a CORRELATION haircut and
    belongs where the promotion gate applies its effective-breadth multiplier. It
    is not the same quantity as label overlap, and collapsing the two into one
    number leaves neither measurable.
    """
    if len(spans) != len(groups):
        raise ValueError(
            f"{len(spans)} span(s) against {len(groups)} group label(s). Refused "
            f"rather than zipped short - a misaligned grouping silently computes "
            f"uniqueness across the wrong series")
    if not spans:
        raise ValueError(
            "no labels supplied. Returning an empty weight vector would let a "
            "training run proceed with no samples and no complaint")

    by_group: dict = {}
    for position, (span, group) in enumerate(zip(spans, groups)):
        by_group.setdefault(group, []).append((position, span))

    out: list[float] = [0.0] * len(spans)
    for members in by_group.values():
        member_spans = [span for _position, span in members]
        # Each series gets its OWN length, not the pooled maximum. Using the
        # pooled length would leave every short symbol padded with bars nothing
        # covers, which does not change its concurrency but does invite the same
        # class of confusion this function exists to remove.
        n_bars = max(span.touched_at_index for span in member_spans) + 1
        for (position, _span), value in zip(members,
                                            average_uniqueness(member_spans, n_bars)):
            out[position] = value
    return out
