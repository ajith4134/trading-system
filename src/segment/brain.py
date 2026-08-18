"""BF-03: what a BULL or BEAR brain is, so a rule brain and a trained model are one interface.

## The shape, and the ruling behind it

**RL-023, 2026-08-18:** the three bots are BULL, BEAR and PROFIT-TAIL. The arbiter is
the selection step that consumes all three, not a brain. This module owns the two
directional ones; `segment.profit_tail` owns the third.

A brain has a **stance** and cannot argue against it. A BULL proposes long entries or
declines; it has no vocabulary for a short. That is not a style choice - it is what
makes two brains an independent pair rather than one model and its negation, which
`~/research/dual-agent-spec.md` was superseded for. `WrongSideProposed` is raised
rather than corrected, because a BULL emitting a short is a bug in that brain and
silently flipping it would hide the bug behind plausible trades.

## Declining is an outcome, not an absence

`Decline` is a value, with a reason and the evidence behind it. A brain that returns
nothing for a symbol and a brain that examined the symbol and said no are different
events, and only one of them is information. FLAT is frequently correct - both spec
documents say so - and a design where "no opinion" is expressed by silence cannot
tell a working brain from a crashed one.

## Evidence is mandatory

Every `Proposal` and every `Decline` carries the feature values that produced it.
`accepts` on BF-03 requires it, RL-013 requires it of anything claiming intelligence,
and it is what makes a rule brain replaceable: when a trained model takes this
interface, the journal keeps the same shape and the two are comparable on the same
record rather than on two incompatible ones.

## The no-edge-claim label travels with the decision

**RL-025:** today's brains decide by explicit rules and the board says so. Every
proposal carries `makes_edge_claim`, defaulting False, and it is the brain that sets
it - not the engine, not the board. When a trained brain arrives, it sets True and
every downstream reader changes with it. A label the display owns is a label that can
be forgotten at the display; a label the decision owns cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

BULL = "BULL"
BEAR = "BEAR"
STANCES = (BULL, BEAR)

LONG = "LONG"
SHORT = "SHORT"

# The side each stance is permitted to propose. A BULL that proposes SHORT is not
# expressing a bearish view - it is a brain that has lost track of what it is.
_PERMITTED = {BULL: LONG, BEAR: SHORT}


class WrongSideProposed(RuntimeError):
    """A brain proposed the side its stance forbids."""


class UnknownStance(ValueError):
    """A brain declared a stance outside the closed set."""


@dataclass(frozen=True)
class Proposal:
    """One brain's positive opinion on one symbol, with what produced it."""

    brain: str
    stance: str
    venue: str
    symbol: str
    side: str
    # 0..1. Not a probability until a brain is calibrated, and an uncalibrated brain
    # says so rather than letting a sizing rule read it as one. `dual-agent-spec.md`
    # makes calibration mandatory before sizing, and this flag is how that survives
    # into the journal.
    confidence: Decimal
    calibrated: bool
    evidence: dict
    at_ns: int
    makes_edge_claim: bool = False

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise UnknownStance(f"{self.brain}: stance {self.stance!r} is not one of {STANCES}")
        permitted = _PERMITTED[self.stance]
        if self.side != permitted:
            raise WrongSideProposed(
                f"{self.brain}: a {self.stance} may only propose {permitted}, "
                f"not {self.side!r}")
        if not self.evidence:
            raise ValueError(
                f"{self.brain}: a proposal with no evidence cannot be journalled or "
                f"compared against a model's; BF-03 requires the features that "
                f"produced it")


@dataclass(frozen=True)
class Decline:
    """A brain looked at a symbol and said no. A first-class outcome."""

    brain: str
    stance: str
    venue: str
    symbol: str
    reason: str
    evidence: dict
    at_ns: int

    @property
    def side(self) -> None:
        return None


@runtime_checkable
class Brain(Protocol):
    """What every BULL and BEAR implements - rule brain and trained model alike."""

    name: str
    stance: str

    def __call__(self, frame) -> Proposal | Decline:
        ...


@dataclass(frozen=True)
class BrainOutputs:
    """One symbol's pair of directional opinions, as the arbiter receives them."""

    bull: Proposal | Decline
    bear: Proposal | Decline

    @property
    def bull_proposal(self) -> Proposal | None:
        return self.bull if isinstance(self.bull, Proposal) else None

    @property
    def bear_proposal(self) -> Proposal | None:
        return self.bear if isinstance(self.bear, Proposal) else None


def decline(brain, *, venue: str, symbol: str, reason: str,
            evidence: dict, at_ns: int) -> Decline:
    """Build a decline, keeping the brain's identity and stance attached to it."""
    return Decline(brain=brain.name, stance=brain.stance, venue=venue, symbol=symbol,
                   reason=reason, evidence=evidence or {"examined": True}, at_ns=at_ns)


def propose(brain, *, venue: str, symbol: str, confidence: Decimal,
            evidence: dict, at_ns: int, calibrated: bool = False,
            makes_edge_claim: bool = False) -> Proposal:
    """Build a proposal on the brain's own permitted side.

    The side is derived from the stance rather than passed in. A caller that could
    name the side could name the wrong one, and this is the one place where that
    mistake is cheap to make and expensive to find.
    """
    if brain.stance not in STANCES:
        raise UnknownStance(f"{brain.name}: stance {brain.stance!r} is not one of {STANCES}")
    return Proposal(
        brain=brain.name, stance=brain.stance, venue=venue, symbol=symbol,
        side=_PERMITTED[brain.stance], confidence=Decimal(confidence),
        calibrated=calibrated, evidence=evidence, at_ns=at_ns,
        makes_edge_claim=makes_edge_claim)
