"""Open positions and closed round trips, from the fills the engine journalled.

Added 2026-08-16 because the user asked when they would see paper trades — open
and closed — and the answer was that the fills exist and nothing renders them.

## What was already there, and what was not

The forward engine has been journalling fills to
`~/capture/paper/forward/fills-<date>.ndjson` since 2026-08-15. On 2026-08-16 it
held **12,252 fills**. So the trades exist.

What did not exist is a **view**: nothing turned that journal into "here is what
is open" and "here is what closed, and for how much". This is that, and it is
deliberately a reader — it computes nothing the journal does not already contain
and it writes nothing back.

## The finding this module produced on its first run

**Every one of those 12,252 fills is a BUY.** `plumbing-momentum` rests a bid at
the previous close and never sells, so there are **no closed round trips at all**
and there cannot be until a strategy exits a position.

That is not a defect in the blotter and it is not a broken engine — it is what
the running strategy does, and `makes_edge_claim` is `false` on every row for
exactly that reason. But an empty closed-trades table looks identical to a quiet
day, so `BlotterView.closed_trades_possible` says which of the two it is: a
journal with no sells cannot produce a closed trade, and the view reports that
rather than an empty list.

## Both accountings, always, and the gap between them is the number

Every fill carries `optimistic_price` and `pessimistic_price` — the same fill
priced as though it rested and as though it crossed. A blotter reporting one
figure would be choosing which, and the choice that gets chosen is the flattering
one.

So every position and every closed trade carries both, and `spread_bps` is
carried too. That gap is the honest measure of how much of a paper result is an
execution assumption rather than a market outcome, and on a strategy whose
participation is `uncalibrated` it is the first number worth reading.

## Unrealised P&L requires a mark, and a stale mark is the classic overstatement

An open position has no P&L until it is marked, and marking it against a price
from an hour ago reports a number about a market that has moved. `mark_prices` is
supplied by the caller, and a position with no mark reports
`unrealised_pnl=None` rather than zero — the same distinction
`features.staleness` insists on, because zero reads as "flat" and the truth is
"unknown".

## FIFO, declared

Round trips are paired first-in-first-out per (venue, symbol). FIFO is a
convention rather than a fact, and the convention is stated because it changes
the reported P&L of every partially-closed position: LIFO on a rising market
books different trades. It matches how a venue reports its own fills, which is
the only tie-breaker that matters when this is later reconciled against exchange
truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence

FORWARD_DIR = "paper/forward"
FILLS_GLOB = "fills-*.ndjson"

BUY = "BUY"
SELL = "SELL"

OPTIMISTIC = "optimistic"
PESSIMISTIC = "pessimistic"
ACCOUNTINGS = (OPTIMISTIC, PESSIMISTIC)

_BPS = Decimal(10_000)


@dataclass(frozen=True)
class Fill:
    """One journalled fill, priced both ways.

    `makes_edge_claim` rides every fill and therefore every position derived
    from it: a blotter showing P&L for a strategy that claims no edge, without
    saying so, invites reading it as performance.
    """
    at_ns: int
    strategy: str
    makes_edge_claim: bool
    venue: str
    symbol: str
    side: str
    quantity: Decimal
    optimistic_price: Decimal
    pessimistic_price: Decimal
    uncalibrated: bool

    def price(self, accounting: str) -> Decimal:
        return (self.optimistic_price if accounting == OPTIMISTIC
                else self.pessimistic_price)


@dataclass(frozen=True)
class ClosedTrade:
    """One round trip, priced under both accountings.

    `spread_bps` is the gap between them - how much of this result is an
    execution assumption rather than a market outcome.
    """
    venue: str
    symbol: str
    strategy: str
    quantity: Decimal
    opened_at_ns: int
    closed_at_ns: int
    entry_optimistic: Decimal
    exit_optimistic: Decimal
    entry_pessimistic: Decimal
    exit_pessimistic: Decimal
    side: str

    def pnl(self, accounting: str) -> Decimal:
        entry = (self.entry_optimistic if accounting == OPTIMISTIC
                 else self.entry_pessimistic)
        exit_price = (self.exit_optimistic if accounting == OPTIMISTIC
                      else self.exit_pessimistic)
        direction = Decimal(1) if self.side == BUY else Decimal(-1)
        return (exit_price - entry) * self.quantity * direction

    @property
    def spread_bps(self) -> Decimal | None:
        """How far apart the two accountings put this trade, in bps of entry."""
        if self.entry_optimistic <= 0:
            return None
        gap = self.pnl(OPTIMISTIC) - self.pnl(PESSIMISTIC)
        notional = self.entry_optimistic * self.quantity
        if notional <= 0:
            return None
        return gap / notional * _BPS

    @property
    def held_ns(self) -> int:
        return self.closed_at_ns - self.opened_at_ns


@dataclass
class OpenPosition:
    """What is still on, and what it cost to get there.

    `unrealised_pnl` is None without a mark rather than zero: zero reads as flat
    and the truth is unknown, which is the distinction `features.staleness`
    insists on for the same reason.
    """
    venue: str
    symbol: str
    strategy: str
    side: str
    quantity: Decimal
    average_optimistic: Decimal
    average_pessimistic: Decimal
    opened_at_ns: int
    fills: int
    mark_price: Decimal | None = None

    def unrealised_pnl(self, accounting: str) -> Decimal | None:
        if self.mark_price is None:
            return None
        entry = (self.average_optimistic if accounting == OPTIMISTIC
                 else self.average_pessimistic)
        direction = Decimal(1) if self.side == BUY else Decimal(-1)
        return (self.mark_price - entry) * self.quantity * direction


@dataclass(frozen=True)
class BlotterView:
    """Everything the journal can say about open and closed paper trades."""
    open_positions: list[OpenPosition]
    closed_trades: list[ClosedTrade]
    fills_read: int
    strategies: tuple[str, ...]
    makes_edge_claim: bool
    uncalibrated_fills: int
    sides_seen: tuple[str, ...]
    unmarked_positions: int

    @property
    def closed_trades_possible(self) -> bool:
        """Could a round trip exist at all, given what is in the journal?

        An empty closed-trades table looks identical to a quiet day. A journal
        holding only BUYs cannot produce a closed trade whatever the market did,
        and that is a fact about the strategy rather than about the session.
        """
        return len(set(self.sides_seen)) > 1

    def realised_pnl(self, accounting: str) -> Decimal:
        return sum((trade.pnl(accounting) for trade in self.closed_trades),
                   start=Decimal(0))

    def describe(self) -> str:
        if not self.fills_read:
            return "no fills journalled - the paper engine has not traded"
        claim = ("" if self.makes_edge_claim
                 else " The running strategy makes NO EDGE CLAIM, so none of this"
                      " is performance.")
        if not self.closed_trades_possible:
            only = self.sides_seen[0] if self.sides_seen else "no"
            return (f"{self.fills_read} fill(s), {len(self.open_positions)} open "
                    f"position(s), and ZERO closed trades - every fill is a "
                    f"{only}, so a round trip is impossible rather than absent."
                    f"{claim}")
        return (f"{self.fills_read} fill(s), {len(self.open_positions)} open "
                f"position(s), {len(self.closed_trades)} closed trade(s); "
                f"realised {self.realised_pnl(OPTIMISTIC):.6f} optimistic against "
                f"{self.realised_pnl(PESSIMISTIC):.6f} pessimistic.{claim}")


def _decimal(value) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)


def read_fills(forward_root: Path) -> list[Fill]:
    """Every journalled fill, oldest first, across every day file.

    A torn final line is skipped rather than fatal - the engine appends while
    this reads, so a partial last row is the ordinary case rather than
    corruption.
    """
    forward_root = Path(forward_root)
    if not forward_root.is_dir():
        return []
    fills: list[Fill] = []
    for path in sorted(forward_root.glob(FILLS_GLOB)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fills.append(Fill(
                at_ns=int(row["at_ns"]), strategy=str(row.get("strategy", "")),
                makes_edge_claim=bool(row.get("makes_edge_claim", False)),
                venue=str(row.get("venue", "")), symbol=str(row.get("symbol", "")),
                side=str(row.get("side", "")),
                quantity=_decimal(row.get("quantity")),
                optimistic_price=_decimal(row.get("optimistic_price")),
                pessimistic_price=_decimal(row.get("pessimistic_price")),
                uncalibrated=bool(row.get("uncalibrated", False))))
    fills.sort(key=lambda fill: fill.at_ns)
    return fills


@dataclass
class _Lot:
    quantity: Decimal
    optimistic: Decimal
    pessimistic: Decimal
    at_ns: int


def build_blotter(fills: Sequence[Fill],
                  mark_prices: dict[tuple[str, str], Decimal] | None = None,
                  ) -> BlotterView:
    """Pair fills into round trips FIFO and report what is still open.

    FIFO is declared rather than assumed: it changes the reported P&L of every
    partially-closed position, and it matches how a venue reports its own fills -
    the only tie-breaker that matters once this is reconciled against exchange
    truth.
    """
    mark_prices = mark_prices or {}
    lots: dict[tuple[str, str], list[_Lot]] = {}
    open_side: dict[tuple[str, str], str] = {}
    closed: list[ClosedTrade] = []
    strategies: set[str] = set()
    sides: list[str] = []
    edge_claim = False
    uncalibrated = 0

    for fill in fills:
        strategies.add(fill.strategy)
        if fill.side not in sides:
            sides.append(fill.side)
        edge_claim = edge_claim or fill.makes_edge_claim
        uncalibrated += int(fill.uncalibrated)

        key = (fill.venue, fill.symbol)
        queue = lots.setdefault(key, [])
        side = open_side.get(key)

        if side is None or side == fill.side or not queue:
            # Opening, or adding to an existing position on the same side.
            open_side[key] = fill.side
            queue.append(_Lot(fill.quantity, fill.optimistic_price,
                              fill.pessimistic_price, fill.at_ns))
            continue

        # Opposite side: close lots FIFO until this fill is used up.
        remaining = fill.quantity
        while remaining > 0 and queue:
            lot = queue[0]
            matched = min(lot.quantity, remaining)
            closed.append(ClosedTrade(
                venue=fill.venue, symbol=fill.symbol, strategy=fill.strategy,
                quantity=matched, opened_at_ns=lot.at_ns,
                closed_at_ns=fill.at_ns,
                entry_optimistic=lot.optimistic,
                exit_optimistic=fill.optimistic_price,
                entry_pessimistic=lot.pessimistic,
                exit_pessimistic=fill.pessimistic_price, side=side))
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= 0:
                queue.pop(0)
        if remaining > 0:
            # The exit was larger than the position: the remainder opens the
            # other way. A reversal is a new position, never a flipped one.
            open_side[key] = fill.side
            queue.append(_Lot(remaining, fill.optimistic_price,
                              fill.pessimistic_price, fill.at_ns))
        elif not queue:
            open_side.pop(key, None)

    positions: list[OpenPosition] = []
    unmarked = 0
    for (venue, symbol), queue in sorted(lots.items()):
        if not queue:
            continue
        quantity = sum((lot.quantity for lot in queue), start=Decimal(0))
        if quantity <= 0:
            continue
        weighted_optimistic = sum((lot.optimistic * lot.quantity for lot in queue),
                                  start=Decimal(0)) / quantity
        weighted_pessimistic = sum((lot.pessimistic * lot.quantity for lot in queue),
                                   start=Decimal(0)) / quantity
        mark = mark_prices.get((venue, symbol))
        if mark is None:
            unmarked += 1
        positions.append(OpenPosition(
            venue=venue, symbol=symbol,
            strategy=sorted(strategies)[0] if strategies else "",
            side=open_side.get((venue, symbol), BUY), quantity=quantity,
            average_optimistic=weighted_optimistic,
            average_pessimistic=weighted_pessimistic,
            opened_at_ns=min(lot.at_ns for lot in queue), fills=len(queue),
            mark_price=mark))

    return BlotterView(
        open_positions=positions, closed_trades=closed, fills_read=len(fills),
        strategies=tuple(sorted(strategies)), makes_edge_claim=edge_claim,
        uncalibrated_fills=uncalibrated, sides_seen=tuple(sides),
        unmarked_positions=unmarked)


def read_blotter(capture_root: Path,
                 mark_prices: dict[tuple[str, str], Decimal] | None = None,
                 ) -> BlotterView:
    """The blotter for whatever the forward engine has journalled so far."""
    return build_blotter(read_fills(Path(capture_root) / FORWARD_DIR),
                         mark_prices)
