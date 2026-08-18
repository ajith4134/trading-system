"""LB-03: a claim with its provenance, its epistemic class and its half-life.

## Why this is the widest point of the design and the narrowest interface

**RL-027, in the user's words:** *"also lots of futer featues and intelliences learnin
will be connectin to tis"*.

So the thing future capabilities attach to must be chosen now, and it must be chosen so
that attaching one does not modify the engine, the arbiter, or any segment bot. This is
that thing.

A capability that PRODUCES a claim attaches as a belief **source**. One that CHALLENGES
a claim attaches as a belief **critic**. Nothing else is an extension point, and a
decision path that reads a raw model output instead of a belief has bypassed it.

`§1a.6` orders ten capabilities first, and the striking thing about that list is that
almost every one is belief-shaped: provenance and half-life · read/verified/observed
classes where only observed may size · verification before ingestion · abstention as a
real action with its P&L measured · calibration scoring · own-footprint attribution.
They are not ten features needing ten integrations. They are one carrier and ten things
that attach to it.

## The failure this is designed against

`§1a.0`, and it is worth quoting because it is the whole reason for the half-life:

> The failure mode this standard is designed against is not stupidity. It is confident
> staleness — a system that learned something true, never noticed it stopped being
> true, and keeps betting on it.

A model output has no expiry. A belief does. `is_expired()` is the difference between a
system that knows its edge was measured on Tuesday and one that keeps trading Tuesday's
edge on Friday.

## Epistemic class, and the rule that gives it teeth

Three classes, and **only OBSERVED may size a position** — `§1a.6`, and it is enforced
here in `may_size()` rather than left as documentation:

| class | means | may size |
|---|---|---|
| `READ` | taken from a document, a paper, a feed's own claim about itself | no |
| `VERIFIED` | checked against an independent source, but not seen happening | no |
| `OBSERVED` | measured from data this system captured itself | **yes** |

The distinction is not pedantry. A funding rate an exchange's REST endpoint *asserts*
and a funding rate this system *watched settle* are different epistemic objects, and a
position sized on the first is sized on a claim nobody checked.

## What cannot be constructed

A belief with no provenance. Not "is discouraged" — cannot be built, because
`__post_init__` refuses it. The reason is `§1a` L1: every number in the decision path
traces to a fit with a logged loss and a dataset reference, or to an explicitly
disclosed design choice. A belief whose provenance is empty is a number whose origin
nobody can name, and those are exactly what this standard exists to catch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

READ = "READ"
VERIFIED = "VERIFIED"
OBSERVED = "OBSERVED"
EPISTEMIC_CLASSES = (READ, VERIFIED, OBSERVED)

# Only this one may size a position (§1a.6). Kept as a set rather than an `if`
# scattered through callers, so adding a class later is one edit in one place.
MAY_SIZE = frozenset({OBSERVED})


class ProvenanceRequired(ValueError):
    """A belief was constructed without naming what produced it."""


class UnknownEpistemicClass(ValueError):
    """A class outside the closed set."""


class ExpiredBeliefUsed(RuntimeError):
    """Something acted on a belief past its half-life.

    Raised rather than warned. A stale belief that merely logs a warning is a
    stale belief that keeps sizing positions, which is `§1a.0`'s confident
    staleness with a paper trail.
    """


@dataclass(frozen=True)
class Provenance:
    """Where a claim came from, in enough detail to audit it later.

    `fit_reference` is what `§1a` L1 asks for: the trial and the model version that
    produced the number, and the loss it earned. A belief from a trained model with no
    fit reference is indistinguishable from a belief from somebody's guess, which is
    the exact confusion L1 exists to prevent.
    """

    source: str
    # The dataset, journal or feed the claim was computed from.
    evidence: dict = field(default_factory=dict)
    # Trial id, model version and out-of-fold score, when a fit produced this.
    fit_reference: dict | None = None
    # Set when this belief is a design choice rather than a learned quantity. L1
    # allows that explicitly - what it refuses is an undisclosed one.
    disclosed_design_choice: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ProvenanceRequired(
                "a belief must name what produced it; an unnamed source is a "
                "number whose origin nobody can audit (§1a L1)")

    @property
    def is_learned(self) -> bool:
        return bool(self.fit_reference)

    def describe(self) -> dict:
        return {"source": self.source, "evidence": self.evidence,
                "fit_reference": self.fit_reference,
                "disclosed_design_choice": self.disclosed_design_choice,
                "is_learned": self.is_learned}


@dataclass(frozen=True)
class Belief:
    """One claim, with everything needed to decide whether to act on it.

    Deliberately not a subclass of anything and deliberately not tied to trading:
    a belief about a funding rate, about this system's own competence, and about
    whether a data feed is trustworthy are the same shape, which is what lets a
    future capability attach without a new integration.
    """

    claim: str
    # The claim's value. A probability, a rate, a boolean - whatever the claim is
    # about. Untyped on purpose: constraining it would be the first thing a future
    # capability had to work around.
    value: object
    epistemic_class: str
    provenance: Provenance
    held_at_ns: int
    # After this many nanoseconds the belief is stale and may not be acted on.
    # There is no default and there deliberately cannot be one: how long a claim
    # stays true is a property of the claim, and a shared default would be the
    # single assumption most likely to be wrong everywhere at once.
    half_life_ns: int
    # 0..1 where the source can express one. None means the source does not claim
    # to know its own reliability, which is different from claiming zero.
    confidence: Decimal | None = None
    # Set by a critic that challenged this belief and did not kill it. Carried so
    # a belief that survived scrutiny is distinguishable from one nobody examined.
    critiques: tuple = ()

    def __post_init__(self) -> None:
        if self.epistemic_class not in EPISTEMIC_CLASSES:
            raise UnknownEpistemicClass(
                f"{self.epistemic_class!r} is not one of {EPISTEMIC_CLASSES}")
        if self.half_life_ns <= 0:
            raise ValueError(
                f"belief {self.claim!r} has a half-life of {self.half_life_ns}; a "
                f"claim that never expires is the confident-staleness failure §1a.0 "
                f"names, written as a constant")

    def age_ns(self, now_ns: int) -> int:
        return now_ns - self.held_at_ns

    def is_expired(self, now_ns: int) -> bool:
        return self.age_ns(now_ns) > self.half_life_ns

    def may_size(self, now_ns: int) -> bool:
        """Whether a position may be sized on this belief.

        Two conditions, both required: the class is OBSERVED, and it has not
        expired. `§1a.6` gives the first; `§1a.0` gives the second.
        """
        return self.epistemic_class in MAY_SIZE and not self.is_expired(now_ns)

    def require_fresh(self, now_ns: int) -> Belief:
        """Return self, or raise. For callers that must not silently degrade."""
        if self.is_expired(now_ns):
            raise ExpiredBeliefUsed(
                f"belief {self.claim!r} was held at {self.held_at_ns} with a "
                f"half-life of {self.half_life_ns}ns and is {self.age_ns(now_ns)}ns "
                f"old; acting on it now is betting on something that has expired")
        return self

    def critiqued(self, critique: dict) -> Belief:
        """A new belief carrying one more critique. Beliefs are never mutated."""
        return Belief(
            claim=self.claim, value=self.value, epistemic_class=self.epistemic_class,
            provenance=self.provenance, held_at_ns=self.held_at_ns,
            half_life_ns=self.half_life_ns, confidence=self.confidence,
            critiques=(*self.critiques, critique))

    def describe(self, now_ns: int | None = None) -> dict:
        """What the journal and the board record. Every field an auditor needs."""
        payload = {
            "claim": self.claim,
            "value": self.value,
            "epistemic_class": self.epistemic_class,
            "half_life_ns": self.half_life_ns,
            "held_at_ns": self.held_at_ns,
            "confidence": None if self.confidence is None else str(self.confidence),
            "provenance": self.provenance.describe(),
            "critiques": list(self.critiques),
        }
        if now_ns is not None:
            payload["age_ns"] = self.age_ns(now_ns)
            payload["expired"] = self.is_expired(now_ns)
            payload["may_size"] = self.may_size(now_ns)
        return payload


def observed(claim: str, value, *, source: str, held_at_ns: int, half_life_ns: int,
             evidence: dict | None = None, fit_reference: dict | None = None,
             confidence: Decimal | None = None) -> Belief:
    """A belief measured from data this system captured itself. May size."""
    return Belief(
        claim=claim, value=value, epistemic_class=OBSERVED,
        provenance=Provenance(source=source, evidence=evidence or {},
                              fit_reference=fit_reference),
        held_at_ns=held_at_ns, half_life_ns=half_life_ns, confidence=confidence)


def read(claim: str, value, *, source: str, held_at_ns: int, half_life_ns: int,
         evidence: dict | None = None) -> Belief:
    """A belief taken from something that asserted it. May NOT size."""
    return Belief(
        claim=claim, value=value, epistemic_class=READ,
        provenance=Provenance(source=source, evidence=evidence or {}),
        held_at_ns=held_at_ns, half_life_ns=half_life_ns)


class BeliefSet:
    """The beliefs held about one subject at one moment.

    A plain container on purpose. Future capabilities add beliefs to it and future
    critics remove or annotate them; anything cleverer here would be a decision made
    on behalf of capabilities that do not exist yet.
    """

    def __init__(self, beliefs=()) -> None:
        self._beliefs = list(beliefs)

    def add(self, belief: Belief) -> BeliefSet:
        self._beliefs.append(belief)
        return self

    def __iter__(self):
        return iter(self._beliefs)

    def __len__(self) -> int:
        return len(self._beliefs)

    def by_claim(self, claim: str):
        return [b for b in self._beliefs if b.claim == claim]

    def live(self, now_ns: int):
        """Only the beliefs that have not expired."""
        return [b for b in self._beliefs if not b.is_expired(now_ns)]

    def sizeable(self, now_ns: int):
        """Only the beliefs a position may be sized on."""
        return [b for b in self._beliefs if b.may_size(now_ns)]

    def describe(self, now_ns: int) -> list:
        return [b.describe(now_ns) for b in self._beliefs]
