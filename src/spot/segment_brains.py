"""SB-02: the spot segment's BULL, BEAR and PROFIT-TAIL.

## What spot has that perp does not, and what it lacks

**No funding.** The perp bear refuses on a crowded-short funding rate; spot has no
such observable, and inheriting that check would mean a brain reading a field that is
always absent and silently never firing. RL-019 is why these are separate files.

**No leverage, so no liquidation distance** — the perp risk gate's third limit has
nothing to bind here.

**Spot is the reference leg.** `features.consolidated_price` and
`features.price_divergence` exist because spot is where a cross-venue reference price
is formed. That is this segment's own edge and these brains are written toward it.

## Today's honest limitation, published rather than smoothed over

Cross-venue divergence needs two venues quoting the same symbol at once. The spot
feed subscribes one venue, so **the divergence input is not available and these
brains do not pretend otherwise**: they read trade-intensity-confirmed momentum, and
every proposal's evidence carries `divergence_available: False`.

The alternative — computing divergence against a stale second venue from the store —
is exactly the RL-024 violation this whole build exists to remove. A one-venue
divergence is not a divergence, and naming the gap keeps it a known gap rather than a
silently wrong feature.
"""
from __future__ import annotations

from decimal import Decimal

from segment.brain import BEAR, BULL, decline, propose
from segment.profit_tail import ProfitTail

# Spot spreads are tighter than perp on the majors and the segment's take-profit is
# smaller, so the tolerable spread is tighter too.
MAX_RELATIVE_SPREAD = Decimal("0.0010")
MIN_TRADES = 4
# Spot momentum thresholds sit above perp's: without leverage the same move is a
# smaller return on capital, so a spot entry has to clear a higher bar to be worth
# the round trip.
BULL_MOMENTUM = 0.0006
BEAR_MOMENTUM = -0.0009
BULL_FLOW = 0.20
BEAR_FLOW = -0.30


def _evidence(frame) -> dict:
    return {
        "momentum": frame.get("momentum"),
        "order_flow_imbalance": frame.get("order_flow_imbalance"),
        "relative_spread": _as_float(frame.get("relative_spread")),
        "trade_count": frame.get("trade_count"),
        "realized_volatility": _as_float(frame.get("realized_volatility")),
        "samples": frame.get("samples"),
        # Named on every decision so the gap is visible in the record rather than
        # discovered later by someone wondering why divergence never fires.
        "divergence_available": False,
        "rule_brain": True,
    }


def _as_float(value):
    return None if value is None else float(value)


def _blocked_reason(frame) -> str | None:
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


class SpotBullBrain:
    name = "spot-bull-rule"
    stance = BULL
    makes_edge_claim = False

    def __call__(self, frame):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        evidence = _evidence(frame)
        blocked = _blocked_reason(frame)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)
        flow, momentum = frame["order_flow_imbalance"], frame["momentum"]
        if momentum < BULL_MOMENTUM:
            return decline(self, venue=venue, symbol=symbol,
                           reason="MOMENTUM_NOT_UP", evidence=evidence, at_ns=at_ns)
        if flow < BULL_FLOW:
            return decline(self, venue=venue, symbol=symbol,
                           reason="FLOW_NOT_BUY_SIDE", evidence=evidence, at_ns=at_ns)
        strength = min(1.0, momentum / (BULL_MOMENTUM * 6))
        flow_strength = min(1.0, (flow - BULL_FLOW) / (1.0 - BULL_FLOW))
        confidence = Decimal(str(round(0.55 + 0.4 * (strength + flow_strength) / 2, 4)))
        return propose(self, venue=venue, symbol=symbol, confidence=confidence,
                       evidence={**evidence, "momentum_strength": round(strength, 4),
                                 "flow_strength": round(flow_strength, 4),
                                 "rule": "upward mid momentum confirmed by buy-side "
                                         "aggressor flow, no leverage assumed"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


class SpotBearBrain:
    """Spot shorts are structurally harder and the brain says so.

    Selling spot requires holding the asset or borrowing it. `dual-agent-spec.md`
    makes short-side limits stricter than long-side everywhere, and here there is a
    further reason: a paper spot short that ignores borrow is modelling a trade the
    live system could not place. The evidence records `borrow_modelled: False` so
    nothing later reads this segment's short P&L as achievable without checking.
    """

    name = "spot-bear-rule"
    stance = BEAR
    makes_edge_claim = False

    def __call__(self, frame):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        evidence = {**_evidence(frame), "borrow_modelled": False}
        blocked = _blocked_reason(frame)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)
        flow, momentum = frame["order_flow_imbalance"], frame["momentum"]
        if momentum > BEAR_MOMENTUM:
            return decline(self, venue=venue, symbol=symbol,
                           reason="MOMENTUM_NOT_DOWN", evidence=evidence, at_ns=at_ns)
        if flow > BEAR_FLOW:
            return decline(self, venue=venue, symbol=symbol,
                           reason="FLOW_NOT_SELL_SIDE", evidence=evidence, at_ns=at_ns)
        strength = min(1.0, momentum / (BEAR_MOMENTUM * 6))
        flow_strength = min(1.0, (BEAR_FLOW - flow) / (1.0 + BEAR_FLOW))
        confidence = Decimal(str(round(0.55 + 0.4 * (strength + flow_strength) / 2, 4)))
        return propose(self, venue=venue, symbol=symbol, confidence=confidence,
                       evidence={**evidence, "momentum_strength": round(strength, 4),
                                 "flow_strength": round(flow_strength, 4),
                                 "rule": "downward mid momentum confirmed by sell-side "
                                         "aggressor flow; borrow cost not modelled"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


def spot_profit_tail(band: str = "slow") -> ProfitTail:
    """Spot holds longer than perp: no funding to pay and no liquidation to avoid."""
    if band == "fast":
        return ProfitTail(
            segment="spot", name="spot-profit-tail-fast",
            take_profit=Decimal("0.0030"), ratchet_trigger=Decimal("0.0018"),
            ratchet_give_back=Decimal("0.0008"),
            signal_expiry_ns=30_000_000_000,
            max_hold_ns=600_000_000_000)          # 10 minutes
    return ProfitTail(
        segment="spot", name="spot-profit-tail-slow",
        take_profit=Decimal("0.0080"), ratchet_trigger=Decimal("0.0045"),
        ratchet_give_back=Decimal("0.0020"),
        signal_expiry_ns=90_000_000_000,
        max_hold_ns=5_400_000_000_000)            # 90 minutes, still intraday (RL-018)
