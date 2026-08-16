"""The exit policy every learned exit has to beat, and the dataset it generates.

`FEATURES.md` §3b (P1): *"Deterministic exit policy as P1 fallback and permanent
rollback target — immediate entry + triple barrier + ATR trail + ratchet.
**Generates the dataset PROFIT-TAIL is later trained on** — the ordering is
forced."* Ledger MD-019, with MD-017's ratchet inside it.

`bull-bear-profit-agents-spec.md` §4 states the same thing from the other end:
*"Mandatory baseline it must beat: immediate entry, fixed triple-barrier, ATR
trailing stop. Beat that out-of-sample or it does not get capital."*

## Why a deterministic policy comes first

The ordering is not a convenience. PROFIT-TAIL is a model of when to exit, and a
model needs a dataset of exits with their outcomes. That dataset does not exist
until something has been exiting positions and recording why. So the plain policy
runs first, generates the record, and then becomes the benchmark the learned one
has to beat — and stays as the rollback target forever, because a system whose
only exit logic is a model has no state to fall back to when the model is wrong.

## Four rules, and their precedence is total and stated

Two of these can fire on the same bar. Which one wins changes the reported P&L of
every trade, so the order is declared rather than emergent:

1. **Hard stop.** Set at fill, never moved by anything here. §3b: *"Hard stop
   overrides PROFIT-TAIL absolutely… its mandate to close in profit is an
   objective, not a veto."* It is first because it is the only rule that exists
   to bound a loss rather than to shape a gain.
2. **Profit lock** — the monotone ratchet, plus the ATR trail, whichever is
   tighter. Both are trailing exits and the tighter one is the one that would
   have triggered.
3. **Profit target** — the triple barrier's upper rail.
4. **Vertical barrier** — time. Last because it is the only rule that fires on
   nothing having happened.

## Ambiguity inside a bar is resolved AGAINST the position, always

A bar whose high touched the target and whose low touched the stop contains both
outcomes, and the order within it is unknown — a one-minute bar is a summary, not
a path. Assuming the favourable one is the flattering error, and it is worth
between a few basis points and the whole trade on exactly the bars that matter
most.

So the adverse rail wins, `is_ambiguous` is set on the record, and the count is
reported. `features.triple_barrier` takes the same posture on the same question,
and this is deliberately the same rule rather than a second opinion about it.

## The ratchet is monotone by construction, not by assertion

`ratchet_profit_lock` takes the previous lock and the new candidate and returns
`max` on the favourable side. There is no branch through which it can widen,
which is stronger than a check that it did not — §3b: *"It never widens, under
any condition, for any model output. A lock that can loosen is not a lock."*

Its distance is **volatility-scaled**, never a fixed percent, for the reason the
spec gives: a fixed percentage is either strangling the tail or protecting
nothing, depending on the regime.

## What this deliberately does not do

**No venue-side mirror.** MD-018 requires the lock to rest at the exchange as a
`reduce-only` stop, because a lock held only in memory protects nothing during a
crash, deploy or partition — exactly the events it exists for. That is a separate
row and it needs an order path this system does not have. Recorded as absent
rather than implied: everything here is a decision about a position, and nothing
here places an order.

**No sizing.** The policy says when to leave, never how much was there.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

LONG = "LONG"
SHORT = "SHORT"
_SIDES = (LONG, SHORT)

# Exit reasons, in the precedence they are evaluated. The order is the policy.
HARD_STOP = "hard_stop"
PROFIT_LOCK = "profit_lock"
PROFIT_TARGET = "profit_target"
VERTICAL_BARRIER = "vertical_barrier"
STILL_OPEN = "still_open"

PRECEDENCE = (HARD_STOP, PROFIT_LOCK, PROFIT_TARGET, VERTICAL_BARRIER)

# How many bars of true range make the ATR. Declared, not searched: this is the
# baseline a learned policy has to beat, and a baseline whose parameters were
# tuned is not a baseline - it is a competitor that got to look at the data first.
ATR_PERIOD = 14


class NoTrueRange(ValueError):
    """Not enough bars to measure an ATR.

    Refused rather than defaulted to a fixed percentage. A trailing distance that
    silently stops being volatility-scaled is the exact failure the spec names -
    a fixed percent either strangles the tail or protects nothing - and it would
    be invisible in the output.
    """


@dataclass(frozen=True)
class ExitPolicy:
    """The four rails, as multiples of ATR and of price.

    Every distance is a MULTIPLE of measured volatility rather than a price or a
    percentage, so one policy applies identically to BTC and to a token priced in
    cents. That is the same rule `strategy.funding_carry` follows for its setup
    definition, and for the same reason: a per-instrument parameter turns one
    hypothesis into N.
    """
    hard_stop_atr: Decimal = Decimal("3")
    profit_target_atr: Decimal = Decimal("4")
    trail_atr: Decimal = Decimal("2")
    # How far behind the favourable extreme the ratchet sits, once it engages.
    # Tighter than the raw trail on purpose: the ratchet's job is to keep a gain
    # that already exists, and the trail's is to stay with a move that is still
    # running.
    ratchet_atr: Decimal = Decimal("1.5")
    # The ratchet does not engage until the position is this far in profit.
    # Without it the lock would sit inside the noise from the first bar and close
    # every trade at random.
    ratchet_arm_atr: Decimal = Decimal("2")
    max_holding_bars: int = 480          # the carry family's own label horizon


@dataclass(frozen=True)
class Bar:
    """One bar of the path after entry."""
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True)
class ExitRecord:
    """One completed exit, and the path that produced it.

    This is the row PROFIT-TAIL is trained on, which is why it carries the path
    statistics and not just the outcome. §3b lists "realised path data - the
    shape of the move after entry, not just its endpoint" as data the model needs
    and the directional bots do not.

    `max_favourable_excursion` and `max_adverse_excursion` are that shape in two
    numbers: how much the trade was ever up, and how much it was ever down. A
    record with only the exit price cannot distinguish a trade that went straight
    to target from one that nearly stopped out first, and those are different
    trades for anything learning when to leave.
    """
    side: str
    entry_price: Decimal
    exit_price: Decimal
    exit_reason: str
    exit_bar: int
    holding_bars: int
    atr_at_entry: Decimal
    hard_stop: Decimal
    final_lock: Decimal | None
    max_favourable_excursion: Decimal
    max_adverse_excursion: Decimal
    is_ambiguous: bool

    @property
    def pnl_per_unit(self) -> Decimal:
        direction = Decimal(1) if self.side == LONG else Decimal(-1)
        return (self.exit_price - self.entry_price) * direction

    def describe(self) -> str:
        note = " [AMBIGUOUS BAR - resolved against the position]" if self.is_ambiguous else ""
        return (f"{self.side} {self.entry_price} -> {self.exit_price} on "
                f"{self.exit_reason} after {self.holding_bars} bar(s); "
                f"MFE {self.max_favourable_excursion}, MAE "
                f"{self.max_adverse_excursion}{note}")


def true_range(bar: Bar, previous_close: Decimal) -> Decimal:
    """The classic true range: the largest of the three gaps that matter.

    Includes the gap from the previous close, which is the whole reason it is not
    just `high - low`: an instrument that gapped overnight moved, and a range that
    ignores the gap reports a quiet bar on the most violent night of the year.
    """
    return max(bar.high - bar.low,
               abs(bar.high - previous_close),
               abs(bar.low - previous_close))


def average_true_range(bars: Sequence[Bar], period: int = ATR_PERIOD) -> Decimal:
    """Simple mean of the last `period` true ranges.

    A simple mean rather than Wilder's smoothing, and the choice is declared: the
    smoothed version has a decay constant, a decay constant is a parameter, and a
    baseline whose parameters were chosen is not a baseline. The two differ by a
    few percent on any real series, which is far inside the multiples above.
    """
    if len(bars) < period + 1:
        raise NoTrueRange(
            f"need >= {period + 1} bars for a {period}-period ATR, got "
            f"{len(bars)}. Refused rather than defaulted to a fixed percentage: "
            f"a trailing distance that silently stops being volatility-scaled "
            f"either strangles the tail or protects nothing, and neither shows "
            f"in the output")
    ranges = [true_range(bars[i], bars[i - 1].close)
              for i in range(len(bars) - period, len(bars))]
    return sum(ranges, start=Decimal(0)) / Decimal(period)


def ratchet_profit_lock(side: str, previous_lock: Decimal | None,
                        candidate: Decimal) -> Decimal:
    """The monotone lock. MD-017.

    Monotone BY CONSTRUCTION rather than by assertion: for a long it is `max`, for
    a short it is `min`, and there is no branch through which the lock can move
    away from the position. §3b: *"It never widens, under any condition, for any
    model output. A lock that can loosen is not a lock."*
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if previous_lock is None:
        return candidate
    return (max(previous_lock, candidate) if side == LONG
            else min(previous_lock, candidate))


def _favourable(side: str, price: Decimal, entry: Decimal) -> Decimal:
    """How far `price` is in the position's favour, signed."""
    return (price - entry) if side == LONG else (entry - price)


def run_exit_policy(side: str, entry_price: Decimal, path: Sequence[Bar],
                    atr_at_entry: Decimal,
                    policy: ExitPolicy | None = None) -> ExitRecord:
    """Walk the path bar by bar and report where the policy would have left.

    Entry is IMMEDIATE, at `entry_price`, on the bar before `path[0]` - the
    baseline's whole point is that it makes no timing decision, so that a learned
    timing policy has something to beat that is not itself a timing policy.

    `atr_at_entry` is measured once, at entry, and does not follow the position.
    A trailing distance that re-measured every bar would tighten into a
    volatility collapse and loosen into a spike, which is a second undeclared
    policy riding inside the first.
    """
    policy = policy or ExitPolicy()
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if atr_at_entry <= 0:
        raise NoTrueRange(
            f"ATR at entry must be > 0, got {atr_at_entry}; a zero ATR puts "
            f"every rail on the entry price, where the first bar touches all of "
            f"them and the exit reason becomes whichever was checked first")
    if not path:
        raise ValueError("no path after entry - there is nothing to exit through")

    direction = Decimal(1) if side == LONG else Decimal(-1)
    hard_stop = entry_price - direction * policy.hard_stop_atr * atr_at_entry
    target = entry_price + direction * policy.profit_target_atr * atr_at_entry
    arm_distance = policy.ratchet_arm_atr * atr_at_entry

    lock: Decimal | None = None
    best = entry_price
    max_favourable = Decimal(0)
    max_adverse = Decimal(0)

    for index, bar in enumerate(path):
        adverse_extreme = bar.low if side == LONG else bar.high
        favourable_extreme = bar.high if side == LONG else bar.low
        max_favourable = max(max_favourable,
                             _favourable(side, favourable_extreme, entry_price))
        max_adverse = min(max_adverse,
                          _favourable(side, adverse_extreme, entry_price))

        # 1. Hard stop. First, always, and never moved.
        hit_hard_stop = (adverse_extreme <= hard_stop if side == LONG
                         else adverse_extreme >= hard_stop)
        hit_target = (favourable_extreme >= target if side == LONG
                      else favourable_extreme <= target)
        # Both rails inside one bar: the order is unknown and a bar is a summary,
        # not a path. The adverse one wins - see the module docstring.
        ambiguous = hit_hard_stop and hit_target
        if hit_hard_stop:
            return _record(side, entry_price, hard_stop, HARD_STOP, index,
                           atr_at_entry, hard_stop, lock, max_favourable,
                           max_adverse, ambiguous)

        # 2. The profit lock, if one is armed. Checked before the target because
        # a lock that is already above the entry is protecting a gain that
        # exists, and the target is a gain that might not arrive.
        if lock is not None:
            hit_lock = (adverse_extreme <= lock if side == LONG
                        else adverse_extreme >= lock)
            if hit_lock:
                return _record(side, entry_price, lock, PROFIT_LOCK, index,
                               atr_at_entry, hard_stop, lock, max_favourable,
                               max_adverse, hit_lock and hit_target)

        # 3. The profit target.
        if hit_target:
            return _record(side, entry_price, target, PROFIT_TARGET, index,
                           atr_at_entry, hard_stop, lock, max_favourable,
                           max_adverse, False)

        # Update the trail from this bar's favourable extreme, for the NEXT bar.
        # Applied after the exit checks, because a lock cannot be hit by the same
        # bar that created it - that would be reading the bar's own high as
        # though it had happened before its low.
        best = (max(best, favourable_extreme) if side == LONG
                else min(best, favourable_extreme))
        if abs(_favourable(side, best, entry_price)) >= arm_distance:
            trail = best - direction * policy.trail_atr * atr_at_entry
            ratchet = best - direction * policy.ratchet_atr * atr_at_entry
            # Whichever is TIGHTER - nearer the price - is the one that would
            # have triggered, so it is the one that governs.
            candidate = (max(trail, ratchet) if side == LONG
                         else min(trail, ratchet))
            lock = ratchet_profit_lock(side, lock, candidate)

        if index + 1 >= policy.max_holding_bars:
            return _record(side, entry_price, bar.close, VERTICAL_BARRIER, index,
                           atr_at_entry, hard_stop, lock, max_favourable,
                           max_adverse, False)

    # The path ran out before any rail was touched. That is not an exit, and
    # reporting the last close as one would invent a trade that never closed -
    # the same refusal `features.triple_barrier` makes for an unresolved label.
    return _record(side, entry_price, path[-1].close, STILL_OPEN,
                   len(path) - 1, atr_at_entry, hard_stop, lock,
                   max_favourable, max_adverse, False)


def _record(side, entry, exit_price, reason, index, atr, hard_stop, lock,
            max_favourable, max_adverse, ambiguous) -> ExitRecord:
    return ExitRecord(
        side=side, entry_price=entry, exit_price=exit_price, exit_reason=reason,
        exit_bar=index, holding_bars=index + 1, atr_at_entry=atr,
        hard_stop=hard_stop, final_lock=lock,
        max_favourable_excursion=max_favourable,
        max_adverse_excursion=max_adverse, is_ambiguous=ambiguous)


@dataclass(frozen=True)
class ExitSummary:
    """What a batch of exits did, by reason.

    Reported by REASON rather than as one win rate, because the reasons are the
    diagnosis: a policy exiting mostly on the vertical barrier is not trading,
    one exiting mostly on the hard stop has its rails in the wrong order, and one
    exiting mostly on the lock is doing what it was built to do. A single number
    hides all three.
    """
    records: list[ExitRecord]

    @property
    def by_reason(self) -> dict[str, int]:
        counts = {reason: 0 for reason in (*PRECEDENCE, STILL_OPEN)}
        for record in self.records:
            counts[record.exit_reason] += 1
        return counts

    @property
    def ambiguous(self) -> int:
        return sum(1 for record in self.records if record.is_ambiguous)

    def total_pnl_per_unit(self) -> Decimal:
        return sum((record.pnl_per_unit for record in self.records),
                   start=Decimal(0))

    def describe(self) -> str:
        if not self.records:
            return "no exits recorded"
        counts = ", ".join(f"{reason} {count}"
                           for reason, count in self.by_reason.items() if count)
        ambiguous = (f"; {self.ambiguous} resolved against the position on an "
                     f"ambiguous bar" if self.ambiguous else "")
        return (f"{len(self.records)} exit(s): {counts}; total "
                f"{self.total_pnl_per_unit()} per unit{ambiguous}")
