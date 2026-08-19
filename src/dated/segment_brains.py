"""DB-02: the dated futures segment's BULL, BEAR and PROFIT-TAIL — on basis and time to expiry.

## Why these brains cannot be the perp brains

A dated future has a **death date**. Everything about it is conditioned on that:

* its **basis to the index converges to zero at delivery**, which is a force no
  perpetual has,
* its **liquidity thins as expiry approaches**, so a contract that was tradable this
  morning may not be this afternoon,
* **time to expiry is an input to every decision**, which DB-02's acceptance requires
  in those words.

The perp brains read funding — the perpetual's substitute for exactly this
convergence. Running them here would mean reading a field that means something
different, which is worse than reading one that is missing.

## Basis is the signal, and its sign is the direction

`annualised_basis = (mark - index) / index * (year / time_to_expiry)`.

* **Positive basis** (contango): the future trades above spot and must fall toward it
  by delivery. That is the BEAR's case.
* **Negative basis** (backwardation): the future trades below spot and must rise.
  That is the BULL's case.

Note this inverts the usual momentum reading, and deliberately: the edge here is
convergence, not trend. A momentum brain on a dated contract fights the one force
that is guaranteed to act.

`features.term_structure` already reads bybit's dated contracts and is the intended
research-side companion; this module is the live-decision side of the same idea.

## The near-expiry refusal is a hard one

Inside the final settlement window both brains decline. Basis convergence is nearly
complete, the remaining move is small, liquidity is worst, and settlement mechanics
begin to dominate price. DB-01 excludes such contracts from the tradable universe for
the same reason; the check is repeated here because a brain that depends on its
caller having filtered correctly is a brain with an unstated precondition.
"""
from __future__ import annotations

from decimal import Decimal

from segment.brain import BEAR, BULL, decline, propose
from segment.profit_tail import ProfitTail

NS_PER_YEAR = 365 * 24 * 3_600_000_000_000
# Inside this window before delivery, neither brain will act. Two days: enough that
# the convergence trade has time to work and enough distance from settlement
# mechanics that price is still driven by the market.
FINAL_SETTLEMENT_WINDOW_NS = 2 * 24 * 3_600_000_000_000
# Annualised basis that has to be on the table before the trade is worth its costs.
MIN_ANNUALISED_BASIS = Decimal("0.04")
MAX_RELATIVE_SPREAD = Decimal("0.0025")


def _time_to_expiry_ns(frame) -> int | None:
    delivery = frame.get("venue_delivery_ns")
    at_ns = frame.get("at_ns")
    if not delivery or not at_ns:
        return None
    remaining = int(delivery) - int(at_ns)
    return remaining if remaining > 0 else 0


def _annualised_basis(frame) -> Decimal | None:
    """Basis to the venue's own index, annualised over the remaining life."""
    mark = frame.get("venue_mark_price")
    index = frame.get("venue_index_price")
    remaining = _time_to_expiry_ns(frame)
    if mark is None or index is None or not remaining:
        return None
    try:
        mark_d, index_d = Decimal(str(mark)), Decimal(str(index))
    except (TypeError, ArithmeticError):
        return None
    if index_d <= 0:
        return None
    raw = (mark_d - index_d) / index_d
    return raw * Decimal(NS_PER_YEAR) / Decimal(remaining)


def _evidence(frame) -> dict:
    remaining = _time_to_expiry_ns(frame)
    basis = _annualised_basis(frame)
    return {
        "time_to_expiry_ns": remaining,
        "time_to_expiry_days": None if remaining is None else round(remaining / 86_400_000_000_000, 3),
        "annualised_basis": None if basis is None else str(basis),
        "mark_price": frame.get("venue_mark_price"),
        "index_price": frame.get("venue_index_price"),
        "relative_spread": None if frame.get("relative_spread") is None else float(frame["relative_spread"]),
        # **The window this frame was computed from (BF-02).** Its acceptance is
        # that every feature row NAMES the window behind it, and this brain was
        # the only evidence on the board that did not - so a reader could not
        # tell a decision made on two observations from one made on two hundred.
        "samples": frame.get("samples"),
        "window_ns": frame.get("window_ns"),
        "rule_brain": True,
    }


def _blocked_reason(frame) -> str | None:
    remaining = _time_to_expiry_ns(frame)
    if remaining is None:
        return "NO_EXPIRY_KNOWN"
    if remaining <= FINAL_SETTLEMENT_WINDOW_NS:
        return "INSIDE_FINAL_SETTLEMENT_WINDOW"
    spread = frame.get("relative_spread")
    if spread is None:
        return "NO_TWO_SIDED_QUOTE"
    if Decimal(str(spread)) > MAX_RELATIVE_SPREAD:
        return "SPREAD_WIDER_THAN_EDGE"
    if _annualised_basis(frame) is None:
        return "NO_INDEX_TO_MEASURE_BASIS_AGAINST"
    return None


def _confidence_from_basis(basis: Decimal) -> Decimal:
    """Confidence rises with how far the basis is past the threshold, capped."""
    excess = (abs(basis) - MIN_ANNUALISED_BASIS) / MIN_ANNUALISED_BASIS
    scaled = min(Decimal("1"), max(Decimal("0"), excess))
    return Decimal(str(round(float(Decimal("0.55") + Decimal("0.4") * scaled), 4)))


class DatedBullBrain:
    """Long the contract when it trades BELOW its index — backwardation converges up."""

    name = "dated-bull-rule"
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
        basis = _annualised_basis(frame)
        if basis >= 0:
            return decline(self, venue=venue, symbol=symbol, reason="IN_CONTANGO",
                           evidence=evidence, at_ns=at_ns)
        if abs(basis) < MIN_ANNUALISED_BASIS:
            return decline(self, venue=venue, symbol=symbol,
                           reason="BASIS_TOO_SMALL_FOR_COSTS",
                           evidence=evidence, at_ns=at_ns)
        return propose(self, venue=venue, symbol=symbol,
                       confidence=_confidence_from_basis(basis),
                       evidence={**evidence,
                                 "rule": "backwardation: the contract trades below its "
                                         "index and must converge up by delivery"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


class DatedBearBrain:
    """Short the contract when it trades ABOVE its index — contango converges down."""

    name = "dated-bear-rule"
    stance = BEAR
    makes_edge_claim = False

    def __call__(self, frame):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        evidence = _evidence(frame)
        blocked = _blocked_reason(frame)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)
        basis = _annualised_basis(frame)
        if basis <= 0:
            return decline(self, venue=venue, symbol=symbol, reason="IN_BACKWARDATION",
                           evidence=evidence, at_ns=at_ns)
        if basis < MIN_ANNUALISED_BASIS:
            return decline(self, venue=venue, symbol=symbol,
                           reason="BASIS_TOO_SMALL_FOR_COSTS",
                           evidence=evidence, at_ns=at_ns)
        return propose(self, venue=venue, symbol=symbol,
                       confidence=_confidence_from_basis(basis),
                       evidence={**evidence,
                                 "rule": "contango: the contract trades above its index "
                                         "and must converge down by delivery"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


def dated_profit_tail(band: str = "slow") -> ProfitTail:
    """Convergence is slow, so this segment's horizon is the longest of the four.

    Still bounded by RL-018's intraday rule — the position closes on the clock even
    though the force it trades acts over weeks. What that costs is real and is the
    honest tension between RL-018 and a convergence edge; it is recorded here rather
    than resolved by quietly holding overnight.
    """
    return ProfitTail(
        segment="dated", name=f"dated-profit-tail-{band}",
        take_profit=Decimal("0.0050"), ratchet_trigger=Decimal("0.0030"),
        ratchet_give_back=Decimal("0.0012"),
        signal_expiry_ns=120_000_000_000,
        max_hold_ns=7_200_000_000_000)            # 2 hours
