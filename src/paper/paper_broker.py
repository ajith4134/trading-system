"""The paper venue, as a transport for the write-ahead log that already exists.

`2026-08-08-paper-execution-engine-design.md`, the load-bearing idea:
`OrderIntentWal.submit(intent, transport, now_ns=None)` takes
`transport: Callable[[OrderIntent, str], dict]`, which is exactly the shape of a
paper broker. So paper is not a simulator standing beside the execution path — it
is plugged into it, and inherits for free:

  * write-before-send durability, fsynced
  * deterministic client order ids (`derive_client_order_id`)
  * signal expiry, refused before the log write
  * the unknown-outcome path, the only state that prompts a query over a resend

That is `ARCHITECTURE.md`'s *"same code path across backtest/paper/live"* made
literal rather than aspirational: paper exercises the code a live transport would
run, so the parts of it that are wrong are wrong now, cheaply.

**A refusal is returned, never raised.** `submit` appends `outcome=unknown`
*before* calling the transport, and a transport that raises leaves it there.
`unknown` is a specific claim — *the venue may be holding this order* — and it is
what `unresolved()` puts on the startup work list. For a local refusal that claim
is false, there is no venue to query, and the entry would block reconciliation
forever. So every refusal here comes back as a rejection response, which the WAL
records as a known outcome.

**Two orders, one per accounting.** `simulate_fills` returns identical quantities
under both accountings and different prices, so a single `Order` would have to
carry one accounting's prices under a name claiming neither, and
`average_fill_price` would silently mean "optimistic". Two orders keep each one
internally truthful. They must stay in lockstep, and `AccountingDiverged` says so
out loud rather than reconciling them — a divergence is a bug in this file, and a
bug that repairs itself in the flattering direction is the failure mode the whole
reality-filter layer exists to catch.

**No network, no credentials, nothing that could reach a venue.** The existence
of verified fee data does not make this able to trade, and nothing here is a step
toward placing an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cost.fee_schedule import FeeRate
from execution.order_intent_wal import OrderIntent
from execution.order_lifecycle import Order
from ops.venue_halt import VenueHaltRegistry
from paper.fill_model import MarketEvent, Participation, RestingOrder, simulate_fills
from paper.position_book import Accounting, PositionBook


class AccountingDiverged(RuntimeError):
    """The optimistic and pessimistic orders stopped agreeing on quantity.

    They fill at different prices and identical sizes by construction, so a size
    disagreement is a defect here, not a market condition. Raised rather than
    repaired: silently re-syncing them would leave the promotion gate reading a
    pessimistic number that no longer describes the same trades.
    """


@dataclass(frozen=True)
class BrokerFill:
    """One fill, priced under both accountings, never blended into one number."""

    client_order_id: str
    symbol: str
    venue: str
    side: str
    quantity: Decimal
    optimistic_price: Decimal
    optimistic_liquidity: str
    pessimistic_price: Decimal
    pessimistic_liquidity: str
    participation: Decimal
    uncalibrated: bool


@dataclass
class RestingPair:
    """One submitted order, tracked once per accounting."""

    client_order_id: str
    symbol: str
    venue: str
    side: str
    # None is a MARKET order - it crosses on the next print rather than
    # resting at a price. See `paper.fill_model.simulate_market_fills`.
    limit_price: Decimal | None
    optimistic_order: Order
    pessimistic_order: Order

    @property
    def is_open(self) -> bool:
        return self.optimistic_order.is_open


def _rejected(reason: str) -> dict:
    return {"status": "rejected", "reason": reason}


class PaperBroker:
    """A local venue that rests orders and fills them off the tape."""

    def __init__(self, participation: Participation,
                 fees_by_venue: dict[str, FeeRate],
                 halts: VenueHaltRegistry | None = None) -> None:
        self._participation = participation
        self._fees_by_venue = dict(fees_by_venue)
        self._halts = halts
        self._resting: dict[str, RestingPair] = {}
        self.optimistic = PositionBook(accounting=Accounting.OPTIMISTIC,
                                       fees_by_venue=fees_by_venue)
        self.pessimistic = PositionBook(accounting=Accounting.PESSIMISTIC,
                                        fees_by_venue=fees_by_venue)

    # --- reads ---------------------------------------------------------------

    @property
    def open_order_count(self) -> int:
        return sum(1 for pair in self._resting.values() if pair.is_open)

    def resting(self, client_order_id: str) -> RestingPair:
        return self._resting[client_order_id]

    # --- the transport contract ----------------------------------------------

    def __call__(self, intent: OrderIntent, client_order_id: str) -> dict:
        """Accept or reject an intent. The signature `OrderIntentWal` calls."""
        if client_order_id in self._resting:
            raise_reason = (
                f"{client_order_id} is already resting — accepting it twice "
                f"would be the duplicate order the deterministic id exists to "
                f"make impossible")
            return _rejected(raise_reason)
        if intent.reduce_only:
            # Implemented 2026-08-16, having been refused until then. An exit is
            # a reduce-only order and refusing the flag meant every exit was
            # rejected. Checked against the position rather than trusted: an
            # order that silently does not reduce is worse than one that was
            # never accepted, and so is one that reduces past flat and opens the
            # other way.
            held = self.optimistic.position(intent.symbol, intent.venue).quantity
            reducing = ((intent.side.upper() == "SELL" and held > 0)
                        or (intent.side.upper() == "BUY" and held < 0))
            if not reducing:
                return _rejected(
                    f"reduce_only order on {intent.venue}:{intent.symbol} with "
                    f"a position of {held} — there is nothing to reduce, so "
                    f"accepting it would open a position under a flag that says "
                    f"it cannot")
            if intent.quantity > abs(held):
                return _rejected(
                    f"reduce_only order for {intent.quantity} against a "
                    f"position of {abs(held)} — refused rather than trimmed, "
                    f"because a caller that asked to close more than it holds "
                    f"has a different view of the position than this book does, "
                    f"and silently trimming hides the disagreement")
        if intent.price is None:
            # A MARKET order. Accepted since 2026-08-16 at the user's
            # instruction, and the original objection is answered rather than
            # overruled: it was that filling at "the last touch this broker
            # happened to see prices it at a moment already known to have gone
            # the right way" - lookahead. The engine applies prints to resting
            # orders BEFORE submitting anything from that same print, so a market
            # order is first offered the NEXT bar and never fills on the print
            # that caused it. See `paper.fill_model.simulate_market_fills`.
            pass
        if intent.venue not in self._fees_by_venue:
            return _rejected(
                f"no fee schedule declared for venue {intent.venue!r} — refused "
                f"at submit rather than at the first fill, so the refusal names "
                f"the order that caused it")
        if self._halts is not None and not self._halts.is_tradeable(intent.venue):
            reason = self._halts.halt_reason(intent.venue) or "never observed"
            return _rejected(
                f"venue {intent.venue!r} is not tradeable: {reason}")

        common = dict(client_order_id=client_order_id, symbol=intent.symbol,
                      venue=intent.venue, side=intent.side,
                      quantity=intent.quantity)
        self._resting[client_order_id] = RestingPair(
            client_order_id=client_order_id, symbol=intent.symbol,
            venue=intent.venue, side=intent.side.upper(),
            limit_price=intent.price,
            optimistic_order=Order(**common), pessimistic_order=Order(**common))
        return {"status": "resting", "client_order_id": client_order_id,
                "limit_price": (None if intent.price is None
                                else str(intent.price)),
                "order_type": "market" if intent.price is None else "limit"}

    # --- the tape drives the fills -------------------------------------------

    def on_market_event(self, symbol: str, venue: str,
                        event: MarketEvent) -> tuple[BrokerFill, ...]:
        """Apply one print to every order resting on that instrument.

        A halted venue produces no fills and leaves open orders exactly where
        they were — frozen, per the design's error table. Cancelling them here
        would be a decision about the position, and this object does not make
        those.
        """
        if self._halts is not None and not self._halts.is_tradeable(venue):
            return ()

        produced: list[BrokerFill] = []
        for pair in self._resting.values():
            if not pair.is_open or pair.symbol != symbol or pair.venue != venue:
                continue
            if pair.optimistic_order.remaining != pair.pessimistic_order.remaining:
                raise AccountingDiverged(
                    f"{pair.client_order_id}: optimistic has "
                    f"{pair.optimistic_order.remaining} remaining and pessimistic "
                    f"has {pair.pessimistic_order.remaining}. They fill at "
                    f"identical sizes by construction, so this is a defect here, "
                    f"and repairing it silently would leave the promotion gate "
                    f"reading a pessimistic number describing different trades")

            outcome = simulate_fills(
                RestingOrder(side=pair.side, limit_price=pair.limit_price,
                             remaining=pair.optimistic_order.remaining),
                (event,), participation=self._participation)

            for optimistic, pessimistic in zip(outcome.optimistic,
                                               outcome.pessimistic):
                pair.optimistic_order.fill(optimistic.quantity, optimistic.price)
                pair.pessimistic_order.fill(pessimistic.quantity, pessimistic.price)
                self.optimistic.apply_fill(
                    symbol=symbol, venue=venue, side=pair.side,
                    quantity=optimistic.quantity, price=optimistic.price,
                    liquidity=optimistic.liquidity)
                self.pessimistic.apply_fill(
                    symbol=symbol, venue=venue, side=pair.side,
                    quantity=pessimistic.quantity, price=pessimistic.price,
                    liquidity=pessimistic.liquidity)
                produced.append(BrokerFill(
                    client_order_id=pair.client_order_id, symbol=symbol,
                    venue=venue, side=pair.side, quantity=optimistic.quantity,
                    optimistic_price=optimistic.price,
                    optimistic_liquidity=optimistic.liquidity,
                    pessimistic_price=pessimistic.price,
                    pessimistic_liquidity=pessimistic.liquidity,
                    participation=outcome.participation,
                    uncalibrated=outcome.uncalibrated))

        return tuple(produced)
