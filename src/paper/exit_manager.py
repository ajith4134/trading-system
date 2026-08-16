"""Positions the engine opened, closed by the deterministic policy.

Wired 2026-08-16 at the user's instruction — *"wire it into the forward engine so
trades close"*. Until then the forward engine only ever bought: 12,983 fills, 363
open positions and zero closed round trips, because `plumbing-momentum` bought
and had no exit. This is the exit.

## Why the exit is not the strategy's job

`PlumbingMomentum` is deliberately not a strategy — it exists so orders flow
through the fill path — and giving it an exit rule would make it one. More
importantly `bull-bear-profit-agents-spec.md` §3b puts the hard stop with the
risk gate and the exit policy in its own layer: *"the hard stop overrides
PROFIT-TAIL absolutely… never moved by the bot."* An exit that lives inside
whatever produced the entry is an exit the entry logic can negotiate with.

So this sits beside the strategy, sees the same tape, and applies the same policy
to any position however it was opened.

## One implementation of the policy, not two

The obvious build tracks the trailing lock incrementally — a running extreme, a
running lock, a rail check per bar. That is a *second* implementation of
`strategy.deterministic_exit`, in a different shape, and the two would disagree
the first time either changed. `what_makes_code_deep.md` names near-identical
shallow duplicates as the failure generated code falls into.

So this keeps the bar path since entry and calls `run_exit_policy` on it, whole,
each time a new bar arrives for that instrument. The policy stays in one place
and this module owns only the bookkeeping: what is open, what it cost, and
whether an exit is already resting.

## The entry price used for the rails is the PESSIMISTIC one

Every fill is priced twice. The rails are computed from the pessimistic entry,
which for a long is the higher price — so the hard stop sits higher, the position
stops out sooner, and the policy is judged against the worse of the two fills it
might have got.

Using the optimistic entry would place every rail where the flattering fill put
them, which is the same error as reporting one accounting: it does not change the
market, it changes what the record says happened.

## The exit is a MARKET order, which is what a stop actually is

Changed 2026-08-16 at the user's instruction. The first version rested a limit at
the rail price, and that was wrong in a way worth recording: **a real stop becomes
a market order when touched and fills *through* the level**, worse, and worst of
all exactly when the move is fast. Resting a limit AT the stop books the stop
price — the one outcome a stop reliably does not get.

As a market order the pessimistic accounting fills at the far touch, which is the
model of crossing the limit path could not express. What is still not modelled is
a **gap beyond the bar** — a print that jumps past the level entirely — because
the bar's own high and low bound what this can see. `slippage_beyond_bar_modelled`
is False and rides every record, so a stop-exit P&L cannot be read as complete.

## One resting exit per instrument, sized when the intent is written

A second exit while one rests would sell the position twice. `pending_exit`
prevents it, and the quantity is fixed at intent time — the position can only
grow between intent and fill (entries add, and the only thing that shrinks it is
this exit), so the exit can never be larger than what is there.

`reduce_only` is set on the intent and the broker **enforces** it as of
2026-08-16 — it refuses an order with nothing to reduce, and refuses one larger
than the position rather than trimming it, because a caller asking to close more
than it holds has a different view of the position than the book does and
trimming hides the disagreement. Until that day the broker rejected the flag
outright, which would have rejected every exit here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path

from execution.order_intent_wal import OrderIntent
from strategy.deterministic_exit import (
    ATR_PERIOD, LONG, SHORT, STILL_OPEN, Bar, CorruptBar, ExitPolicy,
    ExitRecord, NoTrueRange, average_true_range, run_exit_policy,
)

BUY = "BUY"
SELL = "SELL"

# How long an exit intent stays valid. Short, and shorter than the entry's, for
# a reason: an exit priced off a rail computed at one bar is a decision about
# that bar, and firing it minutes later prices a level the market has left.
EXIT_VALID_NS = 60_000_000_000

# Bars kept per instrument before a position exists, so an ATR can be measured
# the moment one opens. One more than the ATR period, because a true range needs
# the previous close.
_WARMUP_BARS = ATR_PERIOD + 1


@dataclass
class _Position:
    """What is open on one instrument, and the path since it opened."""
    side: str
    quantity: Decimal = Decimal(0)
    # Volume-weighted, under BOTH accountings. The rails use the pessimistic one.
    cost_optimistic: Decimal = Decimal(0)
    cost_pessimistic: Decimal = Decimal(0)
    atr_at_entry: Decimal | None = None
    path: list[Bar] = field(default_factory=list)
    pending_exit: bool = False
    # How many bars the position ran before its rails could be measured. Zero is
    # the normal case; anything else says this position was unprotected for that
    # long, and it is carried rather than smoothed over.
    bars_before_rails: int = 0
    # The event time of the last bar appended, so a hole in the series can be
    # detected. A gapped path still produces real exits - it is just not a claim
    # about which rail was touched FIRST.
    last_bar_ns: int | None = None
    path_has_gap: bool = False

    @property
    def entry_optimistic(self) -> Decimal:
        return self.cost_optimistic / self.quantity if self.quantity else Decimal(0)

    @property
    def entry_pessimistic(self) -> Decimal:
        return self.cost_pessimistic / self.quantity if self.quantity else Decimal(0)


@dataclass(frozen=True)
class ExitProposal:
    """An exit the policy asks for, with the record that justifies it.

    `slippage_beyond_bar_modelled` is always False and rides every proposal. The
    exit is a market order, so crossing the spread IS modelled - the pessimistic
    side fills at the far touch. What is not is a print that gaps past the level
    entirely, because the bar's own high and low bound what this can see.
    """
    intent: OrderIntent
    record: ExitRecord
    slippage_beyond_bar_modelled: bool = False


class ExitManager:
    """Bookkeeping for open positions, and the policy applied to them.

    Owns no policy of its own: every exit decision comes from
    `strategy.deterministic_exit.run_exit_policy`, called on the path since entry.
    """

    def __init__(self, strategy_name: str, policy: ExitPolicy | None = None,
                 *, valid_for_ns: int = EXIT_VALID_NS) -> None:
        self._strategy = strategy_name
        self._policy = policy or ExitPolicy()
        self._valid_for_ns = int(valid_for_ns)
        self._positions: dict[tuple[str, str], _Position] = {}
        self._warmup: dict[tuple[str, str], list[Bar]] = {}
        self.exits_proposed = 0
        self.exits_by_reason: dict[str, int] = {}
        self.atr_unavailable = 0
        self.rails_set_late = 0
        self.gapped_paths = 0
        self.corrupt_bars = 0

    # --- state ------------------------------------------------------------

    @property
    def open_positions(self) -> int:
        return sum(1 for p in self._positions.values() if p.quantity > 0)

    def position(self, venue: str, symbol: str) -> _Position | None:
        return self._positions.get((venue, symbol))

    def observe(self, venue: str, symbol: str, bar: Bar,
                at_ns: int | None = None,
                bar_interval_ns: int = 60_000_000_000) -> None:
        """Record a bar, for the ATR before a position and the path after one.

        Called for every event whether or not anything is open, because the ATR
        has to be measurable at the instant a position opens - measuring it after
        the fact from bars that include the entry's own move would let the entry
        widen its own stop.
        """
        key = (venue, symbol)
        warmup = self._warmup.setdefault(key, [])
        warmup.append(bar)
        if len(warmup) > _WARMUP_BARS:
            del warmup[0]

        position = self._positions.get(key)
        if position is not None and position.quantity > 0:
            # A position that opened before enough history existed has NO rails,
            # and without this it would never get any - it would run unprotected
            # for its whole life, which is strictly worse than rails set late.
            # Found by running the engine end to end: `plumbing-momentum` opens
            # on the third bar and the ATR needs fifteen, so every position in
            # that run had no rails and the engine proposed zero exits.
            #
            # The objection to measuring an ATR after entry was that bars
            # including the entry's own move let a position widen its own stop.
            # It does not apply at this size - the orders are 0.001 and move
            # nothing - and the cost of the alternative is total. `
            # bars_before_rails` records how late they arrived so a reader can
            # see which positions ran naked and for how long.
            # A hole in the series. The store builds bars an hour at a time and
            # skips the hour a live writer holds, and the archive carries a
            # 141-hour outage - so this is the ordinary case, not an alarm. It is
            # recorded because a rail touched across a hole is the first one we
            # SAW, not necessarily the first one that happened.
            if (at_ns is not None and position.last_bar_ns is not None
                    and at_ns - position.last_bar_ns != bar_interval_ns):
                position.path_has_gap = True
                self.gapped_paths += 1
            if at_ns is not None:
                position.last_bar_ns = int(at_ns)
            if position.atr_at_entry is None:
                position.bars_before_rails += 1
                atr = self._atr(key)
                if atr is not None:
                    position.atr_at_entry = atr
                    self.rails_set_late += 1
            position.path.append(bar)
            # Bounded by the vertical barrier: a path longer than the policy can
            # hold is a path the policy will never read.
            if len(position.path) > self._policy.max_holding_bars:
                del position.path[0]

    def on_fill(self, *, venue: str, symbol: str, side: str, quantity: Decimal,
                optimistic_price: Decimal, pessimistic_price: Decimal) -> None:
        """Fold a fill into the position. Entries add, exits reduce."""
        key = (venue, symbol)
        position = self._positions.get(key)
        opening_side = LONG if side == BUY else SHORT

        if position is None or position.quantity <= 0:
            atr = self._atr(key)
            self._positions[key] = _Position(
                side=opening_side, quantity=quantity,
                cost_optimistic=optimistic_price * quantity,
                cost_pessimistic=pessimistic_price * quantity,
                atr_at_entry=atr, path=[], pending_exit=False)
            if atr is None:
                self.atr_unavailable += 1
            return

        if (side == BUY) == (position.side == LONG):
            # Adding. The ATR stays at the FIRST entry's: re-measuring it on
            # every add would let a position that grew during a calm stretch
            # tighten its own stop, and one that grew during a spike widen it.
            position.quantity += quantity
            position.cost_optimistic += optimistic_price * quantity
            position.cost_pessimistic += pessimistic_price * quantity
            return

        # Reducing.
        closed = min(quantity, position.quantity)
        share = closed / position.quantity if position.quantity else Decimal(0)
        position.cost_optimistic -= position.cost_optimistic * share
        position.cost_pessimistic -= position.cost_pessimistic * share
        position.quantity -= closed
        position.pending_exit = False
        if position.quantity <= 0:
            self._positions.pop(key, None)

    # --- the decision -----------------------------------------------------

    def exit_proposal(self, venue: str, symbol: str,
                      now_ns: int) -> ExitProposal | None:
        """Run the policy over this instrument's path and ask for an exit if a
        rail was touched. Returns None while the position should stay on."""
        key = (venue, symbol)
        position = self._positions.get(key)
        if position is None or position.quantity <= 0 or position.pending_exit:
            return None
        if position.atr_at_entry is None or not position.path:
            return None
        # The path is only judged from the bar the rails existed. Running the
        # policy over bars the position had no rails for would exit it on a move
        # nothing was watching, and date the exit before the protection.
        path = position.path[position.bars_before_rails:] or position.path[-1:]

        record = run_exit_policy(
            side=position.side, entry_price=position.entry_pessimistic,
            path=path, atr_at_entry=position.atr_at_entry,
            policy=self._policy)
        record = replace(record, path_has_gap=position.path_has_gap)
        if record.exit_reason == STILL_OPEN:
            return None

        position.pending_exit = True
        self.exits_proposed += 1
        self.exits_by_reason[record.exit_reason] = (
            self.exits_by_reason.get(record.exit_reason, 0) + 1)
        return ExitProposal(
            intent=OrderIntent(
                strategy=self._strategy, symbol=symbol, venue=venue,
                side=SELL if position.side == LONG else BUY,
                quantity=position.quantity, created_at_ns=now_ns,
                valid_for_ns=self._valid_for_ns,
                # MARKET. A stop that rests as a limit at its own level books
                # the level, which is the one price a real stop does not get.
                price=None,
                reduce_only=True),
            record=record)

    def _atr(self, key: tuple[str, str]) -> Decimal | None:
        """The ATR from the bars seen before this position opened, or None.

        None rather than a default: `strategy.deterministic_exit` refuses a zero
        or absent ATR because it puts every rail on the entry price, and a
        position with no measurable volatility gets no rails rather than
        arbitrary ones. Counted in `atr_unavailable` so it is visible.
        """
        bars = self._warmup.get(key, [])
        try:
            atr = average_true_range(bars)
        except NoTrueRange:
            return None
        return atr if atr > 0 else None

    def describe(self) -> str:
        if not self.exits_proposed:
            return (f"{self.open_positions} open position(s), no exit proposed "
                    f"yet; {self.atr_unavailable} opened with no measurable ATR "
                    f"and {self.rails_set_late} had rails set late")
        reasons = ", ".join(f"{reason} {count}" for reason, count
                            in sorted(self.exits_by_reason.items()))
        return (f"{self.open_positions} open, {self.exits_proposed} exit(s) "
                f"proposed ({reasons}); {self.gapped_paths} path(s) have a hole "
                f"so their exit is the first rail SEEN rather than the first "
                f"touched; exits are MARKET orders so crossing is modelled, but "
                f"a gap beyond the bar is not")
