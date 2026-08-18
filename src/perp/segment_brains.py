"""PB-15 / PB-03 / PB-04: the perp segment's BULL, BEAR and PROFIT-TAIL.

## What makes these the PERP brains and not a shared default

RL-019: each segment is its own bot with its own data and features. These three read
what only a perpetual has:

* **order flow imbalance** — the aggressor side of live prints, which needs the trade
  tape the perp feed carries at sub-second cadence,
* **funding** — the second observable the perp slice was chosen first for, and a cost
  of carry that no spot symbol has,
* **momentum over the live mid**, on the seconds-to-minutes horizon RL-022 named as
  this segment's edge.

The spot brains read cross-venue divergence, the dated brains term structure, the
options brains the quoted chain. None of the four can run on another's frame, which
is the point.

## Rule brains, labelled as such

**RL-025:** these decide by explicit rules and every decision carries the feature
values that produced it. `makes_edge_claim` is False on every proposal, and the board
tile says RULE BRAIN. They exist to run the live loop end to end, to be the
deterministic baseline the trained brains must beat, and — per `FEATURES.md` line 99 —
to generate the record those brains are later trained on.

They are scaffolding with an honest label. They are to be replaced, not extended.

## The bull and the bear are not each other's negation

They read the same frame and apply different tests, and both can decline the same
symbol. A single scoring function with a sign flip would make the pair one brain
wearing two names, which is what `dual-agent-spec.md` was superseded for.

The asymmetry is deliberate and follows the spec: short-side limits are stricter than
long-side. The bear needs a wider confirmation than the bull, because a crowded short
into a funding squeeze is the perp-specific way to be right about direction and still
be liquidated.
"""
from __future__ import annotations

from decimal import Decimal

from segment.brain import BEAR, BULL, decline, propose
from segment.profit_tail import ProfitTail

# Thresholds are stated, named and journalled rather than tuned. Each is a RULE with
# a reason; none was fitted to a backtest, and PB-06's acceptance requires exactly
# that of anything this segment runs.
#
# A spread wider than this eats a scalp before it starts: the round trip costs twice
# the spread, and this segment's take-profit is 40 bp.
MAX_RELATIVE_SPREAD = Decimal("0.0015")
# Below this the tape is too thin for the aggressor side to mean anything.
MIN_TRADES = 3
# How one-sided the live flow must be before it counts as pressure.
BULL_FLOW = 0.15
BEAR_FLOW = -0.25          # stricter, per the short-side asymmetry above
# The momentum the mid must have shown over the window, in the proposed direction.
BULL_MOMENTUM = 0.0004
BEAR_MOMENTUM = -0.0006    # stricter for the same reason


def _shared_evidence(frame) -> dict:
    return {
        "order_flow_imbalance": frame.get("order_flow_imbalance"),
        "momentum": frame.get("momentum"),
        "relative_spread": _as_float(frame.get("relative_spread")),
        "trade_count": frame.get("trade_count"),
        "realized_volatility": _as_float(frame.get("realized_volatility")),
        "funding_rate": frame.get("venue_funding_rate"),
        "samples": frame.get("samples"),
        "rule_brain": True,
    }


def _as_float(value):
    return None if value is None else float(value)


def _tradeable_or_reason(frame) -> str | None:
    """The gates both brains share. A reason string, or None if the symbol is usable."""
    spread = frame.get("relative_spread")
    if spread is None:
        return "NO_TWO_SIDED_QUOTE"
    if Decimal(str(spread)) > MAX_RELATIVE_SPREAD:
        return "SPREAD_WIDER_THAN_EDGE"
    if (frame.get("trade_count") or 0) < MIN_TRADES:
        return "TAPE_TOO_THIN"
    if frame.get("order_flow_imbalance") is None:
        return "NO_AGGRESSOR_SIDE"
    return None


class PerpBullBrain:
    """Proposes longs on live buy pressure confirmed by upward mid momentum."""

    name = "perp-bull-rule"
    stance = BULL
    makes_edge_claim = False

    def __call__(self, frame):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        evidence = _shared_evidence(frame)

        blocked = _tradeable_or_reason(frame)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)

        flow = frame["order_flow_imbalance"]
        momentum = frame["momentum"]

        if flow < BULL_FLOW:
            return decline(self, venue=venue, symbol=symbol,
                           reason="FLOW_NOT_BUY_SIDE", evidence=evidence, at_ns=at_ns)
        if momentum < BULL_MOMENTUM:
            return decline(self, venue=venue, symbol=symbol,
                           reason="MOMENTUM_NOT_UP", evidence=evidence, at_ns=at_ns)

        # Two independent confirmations, each scaled to how far past its own
        # threshold it is, averaged and floored at the confidence gate. Uncalibrated
        # and declared so: `calibrated=False` keeps every sizing rule from reading
        # this as a probability.
        flow_strength = min(1.0, (flow - BULL_FLOW) / (1.0 - BULL_FLOW))
        momentum_strength = min(1.0, momentum / (BULL_MOMENTUM * 8))
        confidence = Decimal(str(round(0.55 + 0.4 * (flow_strength + momentum_strength) / 2, 4)))

        return propose(self, venue=venue, symbol=symbol, confidence=confidence,
                       evidence={**evidence, "flow_strength": round(flow_strength, 4),
                                 "momentum_strength": round(momentum_strength, 4),
                                 "rule": "buy pressure AND upward mid momentum"},
                       at_ns=at_ns, calibrated=False,
                       makes_edge_claim=self.makes_edge_claim)


class PerpBearBrain:
    """Proposes shorts on live sell pressure confirmed by downward mid momentum.

    Not the bull inverted: its thresholds are wider on both inputs, and it refuses
    outright when funding is deeply negative. Shorting a perpetual whose funding is
    paying shorts to hold means the crowd is already there, and that is where a
    squeeze takes the position out while the directional view was correct.
    """

    name = "perp-bear-rule"
    stance = BEAR
    makes_edge_claim = False

    # Funding below this means shorts are being paid heavily - a crowded short.
    CROWDED_SHORT_FUNDING = Decimal("-0.0005")

    def __call__(self, frame):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        evidence = _shared_evidence(frame)

        blocked = _tradeable_or_reason(frame)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)

        funding = frame.get("venue_funding_rate")
        if funding is not None:
            try:
                if Decimal(str(funding)) < self.CROWDED_SHORT_FUNDING:
                    return decline(self, venue=venue, symbol=symbol,
                                   reason="CROWDED_SHORT_FUNDING",
                                   evidence=evidence, at_ns=at_ns)
            except (TypeError, ArithmeticError):
                pass

        flow = frame["order_flow_imbalance"]
        momentum = frame["momentum"]

        if flow > BEAR_FLOW:
            return decline(self, venue=venue, symbol=symbol,
                           reason="FLOW_NOT_SELL_SIDE", evidence=evidence, at_ns=at_ns)
        if momentum > BEAR_MOMENTUM:
            return decline(self, venue=venue, symbol=symbol,
                           reason="MOMENTUM_NOT_DOWN", evidence=evidence, at_ns=at_ns)

        flow_strength = min(1.0, (BEAR_FLOW - flow) / (1.0 + BEAR_FLOW))
        momentum_strength = min(1.0, momentum / (BEAR_MOMENTUM * 8))
        confidence = Decimal(str(round(0.55 + 0.4 * (flow_strength + momentum_strength) / 2, 4)))

        return propose(self, venue=venue, symbol=symbol, confidence=confidence,
                       evidence={**evidence, "flow_strength": round(flow_strength, 4),
                                 "momentum_strength": round(momentum_strength, 4),
                                 "rule": "sell pressure AND downward mid momentum, "
                                         "refused when funding shows a crowded short"},
                       at_ns=at_ns, calibrated=False,
                       makes_edge_claim=self.makes_edge_claim)


def perp_profit_tail(band: str) -> ProfitTail:
    """The perp PROFIT-TAIL, one instance per exit band.

    **PB-06:** the fast and the slow band run side by side on the same entry signal
    and are journalled apart, and measured winrate and net profit decide which
    survives. They are two configurations of one policy, not two policies — so the
    comparison is of the horizon and nothing else.
    """
    if band == "fast":
        return ProfitTail(
            segment="perp", name="perp-profit-tail-fast",
            take_profit=Decimal("0.0025"), ratchet_trigger=Decimal("0.0015"),
            ratchet_give_back=Decimal("0.0006"),
            signal_expiry_ns=20_000_000_000,
            max_hold_ns=300_000_000_000)          # 5 minutes
    if band == "slow":
        return ProfitTail(
            segment="perp", name="perp-profit-tail-slow",
            take_profit=Decimal("0.0060"), ratchet_trigger=Decimal("0.0035"),
            ratchet_give_back=Decimal("0.0015"),
            signal_expiry_ns=60_000_000_000,
            max_hold_ns=3_600_000_000_000)        # 60 minutes
    raise ValueError(f"perp has two bands, fast and slow; got {band!r}")
