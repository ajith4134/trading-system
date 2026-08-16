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
    # None means a MARKET order: no queue, no participation, crosses on the next
    # print. See `simulate_market_fills`.
    limit_price: Decimal | None
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
        # A market order has no limit to be better than: it pays the touch.
        if order.limit_price is None:
            return event.best_ask
        return max(order.limit_price, event.best_ask)
    if order.limit_price is None:
        return event.best_bid
    return min(order.limit_price, event.best_bid)


def simulate_market_fills(order: RestingOrder,
                          events: tuple[MarketEvent, ...]) -> FillOutcome:
    """A market order crossing, under both accountings.

    Added 2026-08-16 at the user's instruction that orders be market orders
    wherever that is possible. It is possible, and for this system it is the
    MORE honest of the two paths - which is worth stating, because refusing
    market orders had a reason and the reason is answered rather than overruled.

    **No participation fraction appears here at all.** That parameter exists
    because a resting limit order sits in a queue and only some of the printed
    volume reaches it, and nobody has measured what share - which is why every
    fill this system has produced carries `uncalibrated=true`. A market order
    does not queue. It takes what is printed, and the only cap is the print
    itself.

    **Both accountings are TAKERS**, and neither may claim a maker rebate. The
    limit path books the optimistic side as a maker because a resting order that
    is printed through genuinely earned the queue position; a market order never
    did, and awarding it maker pricing would be the cheapest possible fiction
    here - it is a fee difference on every single fill.

    * optimistic - the print's own trade price. Generous, and defensible: an
      order arriving in the same instant as that trade could have got it.
    * pessimistic - the far touch. A BUY lifts the ask, a SELL hits the bid.
      That is what crossing means, and it is the model of slippage the limit
      path could not express: a protective stop resting as a limit AT its level
      books the level, and a stop that crosses books the touch.

    The original refusal - "filling it at the last touch this broker happened to
    see prices it at a moment already known to have gone the right way" - was
    about LOOKAHEAD, and it is answered by ordering rather than by pricing: the
    engine applies prints to resting orders BEFORE it submits anything from that
    same print, so an order submitted on bar N is first offered bar N+1. It never
    fills on the print that caused it.
    """
    optimistic: list[PaperFill] = []
    pessimistic: list[PaperFill] = []
    remaining = order.remaining

    for event in events:
        if remaining <= 0:
            break
        # Capped by the printed volume. Without a depth snapshot there is no
        # basis for claiming an order walked further into the book than the
        # trade that printed, and assuming it did is the flattering direction.
        quantity = min(remaining, event.trade_quantity)
        if quantity <= 0:
            continue
        remaining -= quantity
        optimistic.append(PaperFill(quantity=quantity,
                                    price=event.trade_price, liquidity="taker"))
        pessimistic.append(PaperFill(quantity=quantity,
                                     price=_crossing_price(order, event),
                                     liquidity="taker"))

    return FillOutcome(
        optimistic=tuple(optimistic), pessimistic=tuple(pessimistic),
        # A market order has no queue, so no participation rate was assumed and
        # none is reported. `uncalibrated` is False because nothing here rests
        # on an uncalibrated number - which is the point of preferring them.
        participation=Decimal(1), uncalibrated=False)


def simulate_fills(order: RestingOrder, events: tuple[MarketEvent, ...], *,
                   participation: Participation) -> FillOutcome:
    """Fills under both accountings, from the same events.

    Dispatches on `limit_price`: None is a market order and goes to
    `simulate_market_fills`, where no participation rate is used. One entry point
    so a caller cannot pick the wrong path by accident.
    """
    if order.limit_price is None:
        return simulate_market_fills(order, events)
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
