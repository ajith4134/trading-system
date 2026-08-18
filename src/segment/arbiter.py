"""BF-05: selection — one side or an abstention, from BULL, BEAR and PROFIT-TAIL's numbers.

## What this is, and what it is not

**RL-023:** the arbiter is the SELECTION STEP, not a third brain. It holds no view of
its own about direction. It reads two directional proposals and one advisory
assessment and decides which, if any, becomes a trade.

The line the ruling draws, and the one this module exists to keep:

* **Before selection**, PROFIT-TAIL's expectancy and loss tail are *features* — they
  raise or lower the bar a proposal has to clear, exactly as any other input would.
* **After selection**, PROFIT-TAIL has no vote at all. It owns timing and the
  position; it cannot reject the trade.

Both halves are enforced here. `select()` consumes `TailAssessment` values, and it is
the only function in the system that may. `segment.profit_tail.ProfitTail.reject`
raises for anything that tries the other direction.

## Abstention is a decision and is journalled

FLAT is frequently correct. An abstention carries the reason and both brains'
evidence, so a symbol nobody wanted and a symbol the arbiter refused are separable in
the record. Without that, a bot that stops trading looks identical to a market with no
opportunities.

## Never both sides

Two proposals on the same symbol resolve to one side or to nothing. The margin rule
does that: the winner has to beat the loser by `min_margin`, and a near-tie abstains
rather than picking the larger of two indistinguishable numbers. A design that took
the higher confidence with no margin would trade every disagreement, and disagreement
between two independent brains is the case where the evidence is weakest.

## Uncalibrated confidences are not compared as probabilities

`dual-agent-spec.md` makes calibration mandatory before sizing. Until both brains are
calibrated the comparison is ordinal only — which is larger — and the selection is
recorded as uncalibrated so nothing downstream reads the margin as a probability
difference. Sizing off an uncalibrated confidence is the specific error the spec names.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from segment.brain import BrainOutputs, Proposal

LONG = "LONG"
SHORT = "SHORT"
ABSTAIN = "ABSTAIN"

# Why a symbol was not traded. Journalled, so each is a distinct diagnosis rather
# than a single undifferentiated "no trade".
NO_PROPOSAL = "NO_PROPOSAL"
MARGIN_TOO_THIN = "MARGIN_TOO_THIN"
BELOW_CONFIDENCE_FLOOR = "BELOW_CONFIDENCE_FLOOR"
NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
TAIL_TOO_WIDE = "TAIL_TOO_WIDE"


@dataclass(frozen=True)
class Selection:
    """The arbiter's decision for one symbol, with everything that produced it."""

    venue: str
    symbol: str
    side: str          # LONG, SHORT or ABSTAIN
    reason: str
    confidence: Decimal
    calibrated: bool
    evidence: dict
    at_ns: int
    makes_edge_claim: bool = False

    @property
    def is_trade(self) -> bool:
        return self.side in (LONG, SHORT)


def _confidence(proposal: Proposal | None) -> Decimal:
    return Decimal(0) if proposal is None else Decimal(proposal.confidence)


def select(*, outputs: BrainOutputs, tail, at_ns: int,
           min_confidence: Decimal = Decimal("0.55"),
           min_margin: Decimal = Decimal("0.10"),
           max_loss_tail: Decimal | None = None) -> Selection:
    """Choose the side, or abstain. PROFIT-TAIL's numbers are inputs, never a veto.

    `tail` is a `TailAssessment`. Its `net_expectancy` and `loss_tail` move the bar;
    they do not carry a decision. Note what is deliberately absent: there is no
    `if tail.refuses:` branch, and no boolean on the assessment for one to read.
    """
    bull = outputs.bull_proposal
    bear = outputs.bear_proposal
    venue = outputs.bull.venue
    symbol = outputs.bull.symbol

    evidence = {
        "bull": _describe(outputs.bull),
        "bear": _describe(outputs.bear),
        "tail": {"net_expectancy": str(tail.net_expectancy),
                 "loss_tail": str(tail.loss_tail),
                 "confidence": str(tail.confidence),
                 "authority": "advisory-input-only"},
        "thresholds": {"min_confidence": str(min_confidence),
                       "min_margin": str(min_margin)},
    }

    if bull is None and bear is None:
        return _abstain(venue, symbol, NO_PROPOSAL, evidence, at_ns)

    bull_confidence = _confidence(bull)
    bear_confidence = _confidence(bear)

    # Ordinal comparison only until both are calibrated. Recorded so no downstream
    # sizing rule reads the margin as a probability difference.
    calibrated = bool(bull and bull.calibrated) and bool(bear and bear.calibrated)

    if bull_confidence == bear_confidence:
        return _abstain(venue, symbol, MARGIN_TOO_THIN, evidence, at_ns)

    if bull_confidence > bear_confidence:
        winner, side, loser_confidence = bull, LONG, bear_confidence
    else:
        winner, side, loser_confidence = bear, SHORT, bull_confidence

    winner_confidence = Decimal(winner.confidence)
    margin = winner_confidence - loser_confidence
    evidence["margin"] = str(margin)

    if winner_confidence < min_confidence:
        return _abstain(venue, symbol, BELOW_CONFIDENCE_FLOOR, evidence, at_ns)

    if margin < min_margin:
        # Two independent brains disagreeing to within a hair is the weakest
        # evidence there is, not the strongest.
        return _abstain(venue, symbol, MARGIN_TOO_THIN, evidence, at_ns)

    # PROFIT-TAIL enters here, as arithmetic on the bar rather than as a vote.
    if tail.net_expectancy <= 0:
        return _abstain(venue, symbol, NEGATIVE_EXPECTANCY, evidence, at_ns)

    if max_loss_tail is not None and tail.loss_tail > max_loss_tail:
        evidence["max_loss_tail"] = str(max_loss_tail)
        return _abstain(venue, symbol, TAIL_TOO_WIDE, evidence, at_ns)

    return Selection(
        venue=venue, symbol=symbol, side=side, reason="SELECTED",
        confidence=winner_confidence, calibrated=calibrated,
        evidence=evidence, at_ns=at_ns,
        makes_edge_claim=bool(winner.makes_edge_claim))


def _describe(output) -> dict:
    if isinstance(output, Proposal):
        return {"brain": output.brain, "outcome": "PROPOSAL", "side": output.side,
                "confidence": str(output.confidence),
                "calibrated": output.calibrated, "evidence": output.evidence}
    return {"brain": output.brain, "outcome": "DECLINE", "reason": output.reason,
            "evidence": output.evidence}


def _abstain(venue: str, symbol: str, reason: str, evidence: dict,
             at_ns: int) -> Selection:
    return Selection(venue=venue, symbol=symbol, side=ABSTAIN, reason=reason,
                     confidence=Decimal(0), calibrated=False,
                     evidence=evidence, at_ns=at_ns)
