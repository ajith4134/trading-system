"""Meta-labels: was the primary model's call, at that moment, the right one?

`FEATURES.md` §2 (P2): *"Meta-labelling — Bet sizing / precision. **Not an
overfitting cure**"*. Ledger FE-016. The construction is López de Prado's: a
**primary** model decides the SIDE, and a **secondary** model decides whether to
act on it and how large. This module builds the dataset the secondary model is
trained on, and nothing else — it fits no model, and it makes no claim about how
much precision a secondary model would recover.

That last restraint is the catalogue's own wording and it is load-bearing.
`~/research/finml-feature-engineering.md` records meta-labelling as *"a sound
methodological fix to a real labeling flaw (basically definitional)"* while
naming the **magnitude** of improvement as *"evidence from an interested party,
not an independently replicated result"*. So: the construction, honestly, with
the base rate attached — and no number about what it buys.

## The meta-label is about the SIDE, not about the price

A triple-barrier label of `+1` means price reached the upper barrier first. If
the primary model said SHORT, that is a **loss**, and its meta-label is `0`.

    meta_label = 1 if side == barrier_label else 0

Getting this sign wrong produces a dataset that is perfectly well-formed,
trains without complaint, and teaches the secondary model to approve exactly the
calls it should veto. Nothing downstream can detect it: the label distribution
looks normal, the base rate looks plausible, and the only symptom is a live
system that loses money in proportion to how confident it is.
`test_a_short_that_was_right_is_labelled_one` is the whole defence.

## Meta-labelling can raise precision and can never raise recall

Structural, not empirical. The secondary model only ever sees events the primary
already proposed, so it can turn a proposed trade into a skipped one and can
never create a trade the primary did not offer. Any implementation reporting more
positives after meta-labelling than before has a bug, and `MetaLabelSet` carries
both counts so the relation is visible rather than assumed —
`meta_positives <= primary_signals` is checked in the constructor, because a
class that can hold an impossible state will eventually be handed one.

The consequence for a consumer is the one worth stating: if the primary model
misses a move, no amount of meta-labelling recovers it. Meta-labelling is a
filter, and a filter's ceiling is its input.

## An unresolved barrier is EXCLUDED, never labelled zero

An event whose vertical barrier ran past the end of the data has no outcome.
Labelling it `0` — "the primary was wrong" — is the single most damaging default
available here, because unresolved events are concentrated at the **newest** end
of every dataset: the stretch closest to live, and the stretch a secondary model
most needs to be right about. A model trained that way learns to veto recent
signals as a class.

`features.triple_barrier` already refuses to zero them (`label is None`), and
this module carries that refusal forward rather than re-deciding it. Excluded
events are counted in `unresolved` so a shrinking dataset is visible.

## Ambiguous touches are excluded too, and counted apart

`BarrierTouch.is_ambiguous` marks a bar that touched both barriers — the path
inside the bar is unknown, so which came first is unknown. `triple_barrier`
still assigns a label there under its stated convention, but a META dataset is
the wrong place to accept one: the secondary model's whole job is to grade the
primary's judgement, and an event where the outcome itself is a convention is
noise wearing a label's clothes. They are counted separately from `unresolved`,
because the two call for different work — one wants more data, the other wants
finer bars.

## The base rate is the number that says whether this is worth doing

`base_rate` is the fraction of the primary's resolved calls that were right. It
is reported because meta-labelling has a floor and a ceiling that follow from it:
a primary right 95% of the time leaves a secondary model almost nothing to filter
and a class balance that will make it predict `1` always; a primary right 20% of
the time is not a primary model, it is a sign the side rule is inverted.

`is_degenerate` flags a set whose base rate sits outside `[MIN_WORKABLE_BASE_RATE,
MAX_WORKABLE_BASE_RATE]` or whose minority class has fewer than
`MIN_MINORITY_EVENTS` members. Flagged, not raised — the caller may still want
the set, and a dataset builder that refuses to build is harder to diagnose than
one that builds and says what it built.

## No model is fitted here, and no split is chosen here

The secondary model must not be trained on the data the primary was fitted on —
the primary's in-sample overconfidence would become the secondary's strongest
feature. That is a real constraint and it belongs to whoever fits the two models,
where the split is visible; encoding a split in this module would put a decision
about training data inside a labelling utility, which is where such decisions go
to be forgotten.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from features.triple_barrier import BarrierTouch

# Outside this band there is nothing for a secondary model to learn, in one
# direction or the other. Not thresholds on a gate - `is_degenerate` is a flag
# on a dataset, and the caller decides what to do about it.
MIN_WORKABLE_BASE_RATE = 0.20
MAX_WORKABLE_BASE_RATE = 0.90

# A class with fewer members than this cannot be learned however good the base
# rate looks: 200 events at a 0.95 base rate leaves ten negatives.
MIN_MINORITY_EVENTS = 20

_SIDES = (-1, 1)


class UnalignedSides(ValueError):
    """One side per event, and the two series do not line up.

    Refused rather than zipped to the shorter, matching
    `models.naive_baseline.ForecastLengthMismatch`: a truncated pairing is not a
    smaller dataset, it is a differently-labelled one, and after the first
    mismatch every side is attached to the wrong event.
    """


class InvalidSide(ValueError):
    """A side that is neither long nor short.

    Zero is refused explicitly and is the case worth naming: "no position" is
    not a call the secondary model can grade, and admitting it would silently
    add a third class to a binary problem.
    """


@dataclass(frozen=True)
class MetaEvent:
    """One graded call: what the primary said, what happened, and the grade."""
    event_index: int
    side: int
    barrier_label: int
    meta_label: int


@dataclass(frozen=True)
class MetaLabelSet:
    """The secondary model's dataset, with what was dropped and why.

    `primary_signals` counts every call the primary made, including the ones
    dropped here. Carried so that `meta_positives <= primary_signals` is
    readable on the object rather than being a fact about the code - see the
    module docstring on why that relation is structural.
    """
    events: list[MetaEvent]
    primary_signals: int
    unresolved: int
    ambiguous: int

    def __post_init__(self) -> None:
        if self.meta_positives > self.primary_signals:
            raise ValueError(
                f"{self.meta_positives} meta-positive(s) from "
                f"{self.primary_signals} primary signal(s). Meta-labelling can "
                f"only veto a call the primary already made, so this state is "
                f"unreachable by construction and means the pairing is wrong")

    @property
    def graded(self) -> int:
        return len(self.events)

    @property
    def meta_positives(self) -> int:
        return sum(event.meta_label for event in self.events)

    @property
    def base_rate(self) -> float | None:
        """The fraction of resolved calls the primary got right.

        None on an empty set rather than 0.0: "the primary was never right" and
        "the primary was never graded" are different findings, and 0.0 says the
        first while meaning the second.
        """
        if not self.events:
            return None
        return self.meta_positives / len(self.events)

    @property
    def is_degenerate(self) -> bool:
        """Is there anything here for a secondary model to learn?

        True for an empty set, for a base rate outside the workable band, and
        for a minority class too small to be learned however good the rate looks.
        """
        rate = self.base_rate
        if rate is None:
            return True
        if not MIN_WORKABLE_BASE_RATE <= rate <= MAX_WORKABLE_BASE_RATE:
            return True
        minority = min(self.meta_positives, len(self.events) - self.meta_positives)
        return minority < MIN_MINORITY_EVENTS

    def describe(self) -> str:
        """One line a board or a log can carry, stating the limits with the number."""
        rate = self.base_rate
        if rate is None:
            return (f"0 graded from {self.primary_signals} primary signal(s) "
                    f"({self.unresolved} unresolved, {self.ambiguous} ambiguous) "
                    f"- nothing to learn from")
        minority = min(self.meta_positives, len(self.events) - self.meta_positives)
        verdict = "DEGENERATE" if self.is_degenerate else "workable"
        return (f"{self.graded} graded from {self.primary_signals} primary "
                f"signal(s); base rate {rate:.3f}, minority class {minority}; "
                f"{self.unresolved} unresolved and {self.ambiguous} ambiguous "
                f"excluded - {verdict}")


def make_meta_labels(touches: Sequence[BarrierTouch],
                     sides: Sequence[int]) -> MetaLabelSet:
    """Grade each of the primary model's calls against how the path resolved.

    `sides[i]` is the side the primary took on `touches[i]` — `+1` long, `-1`
    short. The meta-label is 1 exactly when the side agrees with the barrier the
    path touched first.

    Unresolved and ambiguous events are excluded and counted, never labelled 0.
    See the module docstring: unresolved events cluster at the newest end of
    every dataset, so zeroing them teaches a secondary model to veto recent
    signals as a class.
    """
    if len(touches) != len(sides):
        raise UnalignedSides(
            f"{len(touches)} barrier touch(es) against {len(sides)} side(s). "
            f"Refused rather than zipped to the shorter - after the first "
            f"mismatch every side is attached to the wrong event")

    events: list[MetaEvent] = []
    unresolved = ambiguous = 0
    for touch, side in zip(touches, sides):
        if side not in _SIDES:
            raise InvalidSide(
                f"side must be one of {_SIDES}, got {side!r} at event "
                f"{touch.event_index}. Zero is refused explicitly: 'no position' "
                f"is not a call a secondary model can grade, and admitting it "
                f"would add a third class to a binary problem")
        if not touch.is_resolved:
            unresolved += 1
            continue
        if touch.is_ambiguous:
            ambiguous += 1
            continue
        events.append(MetaEvent(
            event_index=touch.event_index,
            side=side,
            barrier_label=int(touch.label),
            # The whole module in one line, and the one line that is silent when
            # it is wrong: a +1 barrier under a SHORT call is a loss.
            meta_label=1 if int(touch.label) == side else 0))

    return MetaLabelSet(events=events, primary_signals=len(touches),
                        unresolved=unresolved, ambiguous=ambiguous)
