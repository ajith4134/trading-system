"""An order's state, tracked by remaining quantity rather than a filled flag.

`FEATURES.md` §5 requires partial-fill tracking by remaining quantity, **never a
binary filled flag**, and the ledger names the bug class: `if status == "FILLED"`
mishandles `PARTIALLY_FILLED` and double-counts. The failure is not subtle - a
partially filled order treated as done abandons the remainder while the system
believes the whole position is on.

    PENDING ──fill(part)──► PARTIAL ──fill(rest)──► FILLED    (terminal)
       │                       │
       ├────────cancel─────────┴──────────────────► CANCELLED (terminal)
       └────────reject────────────────────────────► REJECTED  (terminal)

**Ported, with credit.** `nse-crypto-bot-final/trading/execution/order_state.py`
(ledger EX-049) was read raw like every other donor in this audit, and it is the
first one that holds up: real transition guards, over-fill caught, size-weighted
average price, `remaining` as a property. The structure here is theirs.

Two deliberate departures:

**`Decimal`, not `float` with `EPS = 1e-9`.** Exchange step sizes are decimal
quantities, and in Decimal they sum exactly: 0.1 + 0.2 + 0.7 closes a unit order
where binary float leaves the residue the donor's epsilon existed to absorb. No
epsilon means no epsilon quietly tolerating a slight over-fill either.

Decimal is exact only for *representable* values, so this is not a claim of
exactness in general - `Decimal(1)/Decimal(3)` carries 28 digits and three of them
leave 1E-28 outstanding. The resolution for that is `quantity_step`, i.e. instrument
precision, which RX-005 already requires for reconciliation. With no step declared,
sub-precision residue keeps the order PARTIAL: visible, and wrong in the safe
direction rather than rounded away.

**A rejection is not a fill.** The donor does `fills.append({"rejected": reason})`,
which puts a record with no `qty` into the list everything else sums over. Anything
totalling filled quantity from that list either raises or silently skips a row.

No I/O here, deliberately - the same boundary the donor drew. `order_intent_wal`
owns durability, and this stays a pure state machine that can be reasoned about and
tested without a filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

_SIDES = ("BUY", "SELL")


class OrderStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                        OrderStatus.REJECTED)

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.PENDING, OrderStatus.PARTIAL)


class InvalidTransition(RuntimeError):
    """An order was driven through a state change the lifecycle forbids.

    Raised rather than tolerated. Every one of these represents a real race - a
    fill arriving after a cancel, a duplicate fill, an over-fill - and applying it
    would create or destroy a position that nothing else believes in.
    """


@dataclass(frozen=True)
class Fill:
    quantity: Decimal
    price: Decimal


@dataclass
class Order:
    """One order's lifecycle. Mutated by fills; never by anything else."""

    client_order_id: str
    symbol: str
    venue: str
    side: str
    quantity: Decimal
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    fills: tuple[Fill, ...] = field(default_factory=tuple)
    rejection_reason: str | None = None
    # The exchange's quantity step for this instrument, when known. RX-005 requires
    # reconciliation "to instrument precision", and the same number answers "is this
    # order complete": below one step there is no order any exchange would accept.
    # None means no step is declared, and residue then stays visible as PARTIAL
    # rather than being rounded away - wrong in the safe direction.
    quantity_step: Decimal | None = None

    def __post_init__(self) -> None:
        self.side = self.side.upper()
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {self.side!r}")
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")

    @property
    def remaining(self) -> Decimal:
        """What still needs filling. The number every decision should read.

        Quantized to `quantity_step` when one is declared. Decimal is exact for
        representable values, but a fill of `Decimal(1)/Decimal(3)` is not one, and
        three of those leave 1E-28 outstanding - enough to keep an order PARTIAL
        forever. Instrument precision is the principled resolution; an epsilon is
        the one this replaces.
        """
        outstanding = max(Decimal("0"), self.quantity - self.filled_quantity)
        if self.quantity_step is not None and outstanding < self.quantity_step:
            return Decimal("0")
        return outstanding

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def average_fill_price(self) -> Decimal | None:
        """Size-weighted average, or `None` when nothing has filled.

        `None` rather than zero: zero is a price, and reporting it puts a number
        meaning "no data" into P&L arithmetic.
        """
        if self.filled_quantity == 0:
            return None
        return (sum((f.quantity * f.price for f in self.fills), Decimal("0"))
                / self.filled_quantity)

    # --- transitions ---------------------------------------------------------

    def fill(self, quantity: Decimal, price: Decimal) -> "Order":
        """Apply a fill, moving PENDING -> PARTIAL -> FILLED as the total closes."""
        if not self.status.is_open:
            raise InvalidTransition(
                f"cannot fill order {self.client_order_id} in terminal state "
                f"{self.status.value} - a fill arriving after a cancel or a "
                f"duplicate fill would create a position nothing believes it holds")
        quantity = Decimal(str(quantity))
        if quantity <= 0:
            raise ValueError(f"fill quantity must be > 0, got {quantity}")
        if quantity > self.remaining:
            raise InvalidTransition(
                f"over-fill: order {self.client_order_id} has {self.remaining} "
                f"remaining, got a fill of {quantity}")

        self.fills = self.fills + (Fill(quantity, Decimal(str(price))),)
        self.filled_quantity += quantity
        self.status = (OrderStatus.FILLED if self.remaining == 0
                       else OrderStatus.PARTIAL)
        return self

    def cancel(self) -> "Order":
        """Cancel an open order, keeping whatever already filled - it is real."""
        if not self.status.is_open:
            raise InvalidTransition(
                f"cannot cancel order {self.client_order_id} in terminal state "
                f"{self.status.value}")
        self.status = OrderStatus.CANCELLED
        return self

    def reject(self, reason: str) -> "Order":
        """Mark a still-pending order rejected."""
        if self.status is not OrderStatus.PENDING:
            raise InvalidTransition(
                f"only a PENDING order can be rejected; {self.client_order_id} is "
                f"{self.status.value}")
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        return self

    # --- serialisation, for the WAL ------------------------------------------

    def to_dict(self) -> dict:
        """JSON-ready form. Decimals as strings, because floats lose precision and
        this record is what decides whether an order still needs sending."""
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "venue": self.venue,
            "side": self.side,
            "quantity": str(self.quantity),
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "fills": [{"quantity": str(f.quantity), "price": str(f.price)}
                      for f in self.fills],
            "rejection_reason": self.rejection_reason,
            "quantity_step": (str(self.quantity_step)
                              if self.quantity_step is not None else None),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Order":
        """Reconstruct exactly, fills included.

        A round trip that loses the fills rebuilds an order looking PENDING, and the
        recovery path then sends it again. That is the duplicate-order failure the
        WAL exists to prevent, so it must not be reintroduced here.
        """
        return cls(
            client_order_id=payload["client_order_id"],
            symbol=payload["symbol"],
            venue=payload["venue"],
            side=payload["side"],
            quantity=Decimal(payload["quantity"]),
            status=OrderStatus(payload["status"]),
            filled_quantity=Decimal(payload["filled_quantity"]),
            fills=tuple(Fill(Decimal(f["quantity"]), Decimal(f["price"]))
                        for f in payload.get("fills", ())),
            rejection_reason=payload.get("rejection_reason"),
            quantity_step=(Decimal(payload["quantity_step"])
                           if payload.get("quantity_step") else None),
        )
