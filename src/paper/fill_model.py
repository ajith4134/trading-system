"""When a resting order fills on paper, and what that fill is worth.

Pure: no store, no clock, no filesystem. (resting order, market events,
participation) in, fills out. That is deliberate — this is the one component whose
central claim has to be provable by property test, and a pure function is provable
without standing up an archive.

The rule a fill has to clear: **a trade must have printed strictly through the
resting price.** A print *at* the price proves nothing, because it does not say
whether the queue ahead of us absorbed it. Awarding a fill there is the assumption
that manufactures edge.
"""
from dataclasses import dataclass
from decimal import Decimal

_SIDES = ("BUY", "SELL")


@dataclass(frozen=True)
class RestingOrder:
    """A limit order sitting in the book, as far as paper can tell."""

    side: str
    limit_price: Decimal
    remaining: Decimal

    def __post_init__(self) -> None:
        if self.side.upper() not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {self.side!r}")


@dataclass(frozen=True)
class MarketEvent:
    """One trade print, with the touch as it stood when the print happened."""

    trade_price: Decimal
    trade_quantity: Decimal
    best_bid: Decimal
    best_ask: Decimal


@dataclass(frozen=True)
class Participation:
    """The share of printed volume the queue would plausibly have given us.

    `calibrated` is carried rather than assumed: a fraction measured from the depth
    archive and a fraction defaulted because no depth existed are different claims,
    and a result that cannot tell them apart lets an assumed number be read as a
    measured one.
    """

    fraction: Decimal
    calibrated: bool


@dataclass(frozen=True)
class PaperFill:
    quantity: Decimal
    price: Decimal
    liquidity: str              # "maker" | "taker"


@dataclass(frozen=True)
class FillOutcome:
    """Both accountings of the same events, never one blended number."""

    optimistic: tuple[PaperFill, ...]
    pessimistic: tuple[PaperFill, ...]
    participation: Decimal
    uncalibrated: bool


def signed_cash(fills: tuple[PaperFill, ...], *, side: str,
                maker_bps: Decimal, taker_bps: Decimal) -> Decimal:
    """Cash in (positive) or out (negative) for these fills, fees included.

    Signed rather than absolute so both sides compare with one inequality: worse
    is always smaller, whether the order bought or sold. That is what lets the
    pessimistic-never-beats-optimistic invariant be stated once instead of twice.

    Fees are charged at the rate for the liquidity each fill actually took. A
    single blended rate is how a taker fill gets quietly priced as a maker.
    """
    total = Decimal("0")
    for fill in fills:
        rate = maker_bps if fill.liquidity == "maker" else taker_bps
        notional = fill.quantity * fill.price
        fee = notional * rate / Decimal("10000")
        total += -(notional + fee) if side.upper() == "BUY" else notional - fee
    return total


def _printed_through(order: RestingOrder, event: MarketEvent) -> bool:
    """Did a trade go strictly beyond the resting price?

    Strictly: a print *at* our price is exactly the case where queue position
    decides, and queue position is what the data cannot show.
    """
    if order.side.upper() == "BUY":
        return event.trade_price < order.limit_price
    return event.trade_price > order.limit_price


def _crossing_price(order: RestingOrder, event: MarketEvent) -> Decimal:
    """What taking liquidity would have cost, never better than resting did.

    A print through us usually leaves the touch *inside* our limit, so the raw
    crossing price can be the better of the two. Reporting that would let the
    pessimistic accounting beat the optimistic one by construction — a free lunch
    produced by the accounting, not by the strategy. If crossing were genuinely
    better, the order should have crossed rather than rested.
    """
    if order.side.upper() == "BUY":
        return max(order.limit_price, event.best_ask)
    return min(order.limit_price, event.best_bid)


def simulate_fills(order: RestingOrder, events: tuple[MarketEvent, ...], *,
                   participation: Participation) -> FillOutcome:
    """Fills under both accountings, from the same events."""
    optimistic: list[PaperFill] = []
    pessimistic: list[PaperFill] = []
    remaining = order.remaining

    for event in events:
        if remaining <= 0:
            break
        if not _printed_through(order, event):
            continue
        available = event.trade_quantity * participation.fraction
        quantity = min(remaining, available)
        remaining -= quantity
        optimistic.append(PaperFill(quantity=quantity,
                                    price=order.limit_price, liquidity="maker"))
        pessimistic.append(PaperFill(quantity=quantity,
                                     price=_crossing_price(order, event),
                                     liquidity="taker"))

    return FillOutcome(
        optimistic=tuple(optimistic),
        pessimistic=tuple(pessimistic),
        participation=participation.fraction,
        uncalibrated=not participation.calibrated,
    )
