"""What we believe we hold, and what holding it has cost so far.

This is source 2 of the three `execution.state_recovery` reconciles at startup —
the WAL holds what we *intended*, the venue holds what is *real*, and this holds
what we *believe*. Its value comes from being a separate believer that can be
caught out; a position model derived from venue truth would agree with it by
construction and prove nothing.

No I/O, no clock, no store. The same boundary `order_lifecycle` draws: durability
belongs to `order_intent_wal`, and this stays arithmetic that can be checked by
hand.

**One book per accounting.** The paper engine scores optimistic and pessimistic
fills separately and gates promotion on the pessimistic number, so the broker
holds two of these rather than one holding two sets of figures. A single book
with both invites someone to average them, and the average of an optimistic and a
pessimistic fill is precisely the manufactured edge `ARCHITECTURE.md:92` warns
about: a number that describes no accounting anybody can defend. Each book also
carries the name of its own accounting, so a journal row can never be read as the
other one.

**A flat position is not a position.** `recover_and_reconcile` reads a local
position with no venue counterpart as an unmanaged live position and refuses the
start. A book reporting every symbol it had ever touched at quantity zero would
raise one of those per closed round trip, and the reconciler would be
unstartable within a day of trading. So `local_positions()` reports open
positions only, and a flat entry carries `average_entry_price = None` rather than
zero — zero is a price, and a price meaning "no position" ends up in P&L
arithmetic eventually.

**Entry fees are realized when they are paid, never amortised.** Charging a
fraction of the entry fee as each part of the position closes would leave an open
position quietly carrying a cost that has already left the account. The fee is
cash gone at the moment of the fill; the P&L says so at that moment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from cost.fee_schedule import FeeRate
from execution.state_recovery import VenuePosition

_SIDES = ("BUY", "SELL")
_LIQUIDITY = ("maker", "taker")
_BPS = Decimal("10000")


class Accounting(str, Enum):
    """Which fill accounting this book is keeping.

    Carried rather than inferred from context: the two books hold the same
    symbols at the same times and differ only in the prices they were fed, so a
    row that loses this label is unattributable and cannot be got back.
    """

    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"


class UndeclaredFeeSchedule(LookupError):
    """A fill arrived on a venue whose fee rates were never declared.

    Refused rather than defaulted, per `CLAUDE.md`: *no quote may be produced
    from a default*. A guessed rate is indistinguishable from a measured one once
    it is inside a P&L, and it is wrong in the flattering direction exactly when
    the venue is expensive.
    """


@dataclass
class InstrumentPosition:
    """One instrument on one venue. Quantity is signed: long positive, short
    negative, so a flip is arithmetic rather than a special case with a flag."""

    symbol: str
    venue: str
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal | None = None

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass
class PositionBook:
    """The local position model for one accounting."""

    accounting: Accounting
    fees_by_venue: dict[str, FeeRate]
    realized_pnl_after_fees: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    _positions: dict[tuple[str, str], InstrumentPosition] = field(
        default_factory=dict)

    # --- reads ---------------------------------------------------------------

    def position(self, symbol: str, venue: str) -> InstrumentPosition:
        """The position on this instrument, flat if we have never traded it.

        A flat position is a truthful answer to "what do we hold" — it is only
        `local_positions()` that must omit it, because that feeds a reconciler
        which reads presence as a claim.
        """
        return self._positions.get(
            (venue, symbol), InstrumentPosition(symbol=symbol, venue=venue))

    def local_positions(self) -> list[VenuePosition]:
        """Open positions, in the shape `recover_and_reconcile` takes.

        Sorted so a mismatch list is stable across restarts: an operator diffing
        two reconciliation reports should see what changed, not a reordering.
        """
        return [
            VenuePosition(symbol=p.symbol, venue=p.venue, quantity=p.quantity,
                          average_price=p.average_entry_price)
            for _, p in sorted(self._positions.items())
            if not p.is_flat
        ]

    # --- the one mutation ----------------------------------------------------

    def apply_fill(self, *, symbol: str, venue: str, side: str,
                   quantity: Decimal, price: Decimal, liquidity: str) -> None:
        """Apply one fill: move the position, charge the fee, realize what closed.

        Validated before anything is mutated, so a refused fill leaves the book
        exactly as it was. A partially-applied fill would be a position model
        that is wrong in a way nothing recorded.
        """
        side = side.upper()
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
        if liquidity not in _LIQUIDITY:
            raise ValueError(
                f"liquidity must be one of {_LIQUIDITY}, got {liquidity!r} — "
                f"refused rather than priced as maker, which is the cheaper of "
                f"the two and therefore the wrong way to guess")
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        if quantity <= 0:
            raise ValueError(f"fill quantity must be > 0, got {quantity}")
        rate = self.fees_by_venue.get(venue)
        if rate is None:
            raise UndeclaredFeeSchedule(
                f"no fee schedule declared for venue {venue!r} — refusing the "
                f"fill rather than pricing it at a default, which would put an "
                f"invented cost into a P&L that gates promotion")

        fee = quantity * price * (
            rate.maker_bps if liquidity == "maker" else rate.taker_bps) / _BPS
        signed = quantity if side == "BUY" else -quantity

        key = (venue, symbol)
        position = self._positions.setdefault(
            key, InstrumentPosition(symbol=symbol, venue=venue))
        held = position.quantity

        if held == 0:
            position.quantity = signed
            position.average_entry_price = price
        elif (held > 0) == (signed > 0):
            # Adding to the same side: size-weighted average of the open leg.
            entry = position.average_entry_price
            position.average_entry_price = (
                (abs(held) * entry + quantity * price) / (abs(held) + quantity))
            position.quantity = held + signed
        else:
            # Reducing, closing or flipping. Only the overlap realizes.
            entry = position.average_entry_price
            closed = min(quantity, abs(held))
            direction = Decimal("1") if held > 0 else Decimal("-1")
            self.realized_pnl_after_fees += closed * (price - entry) * direction
            position.quantity = held + signed
            if position.quantity == 0:
                position.average_entry_price = None
            elif quantity > abs(held):
                # Flipped: the surviving leg was opened by this fill, at this
                # price. Carrying any of the old entry across would misstate
                # every P&L the new leg ever reports.
                position.average_entry_price = price

        self.realized_pnl_after_fees -= fee
        self.fees_paid += fee
