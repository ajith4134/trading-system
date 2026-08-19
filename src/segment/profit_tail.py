"""BF-04: PROFIT-TAIL — entry timing and the whole position after fill, and what it may never do.

## The ruling, and the gap it closed

**RL-023, 2026-08-18, in the user's words:** *"This is not as in the plans
bull/bear/arbiter it is bull bear and profit tailgating"*.

The perp slice had PB-03 (bull), PB-04 (bear) and PB-05 (an arbiter over the two).
That is a two-brain design with a chooser — precisely the shape
`~/research/dual-agent-spec.md` was superseded for on 2026-08-03. The third bot was
missing from the plan for fifteen days.

## Its authority, stated as what it CANNOT do

`~/research/bull-bear-profit-agents-spec.md` §2, in the user's words: PROFIT-TAIL
*"will not reject trades after the trades are selected."* Three prohibitions follow,
and each has a test that trips on violating it:

1. **It cannot reject a selected trade.** Its expectancy and tail estimates are
   features the ARBITER consumes before selection. After selection it has no vote.
   A PROFIT-TAIL that could refuse would be a second risk gate, and every blow-up in
   the research set traces to risk authority being split.
2. **It cannot refuse to close a loser.** Its objective is to close in profit; that
   is an objective, never a veto. `manage()` cannot return HOLD once the hard stop is
   breached — the code path does not exist, rather than existing and being discouraged.
3. **It has no line to execution.** Every action it raises — add, scale-out, close —
   re-enters the risk gate like a fresh order.

## Timing is bounded, and the bound is the point

It may wait for a better entry. It may not wait forever: a signal has an expiry, and
past it the trade is abandoned and journalled as a **missed entry attributed to
PROFIT-TAIL**.

Without that attribution the failure is invisible in the flattering direction — a
timing bot that misses the biggest movers shows *excellent average fill prices*,
because the trades it skipped were the ones that ran away. `FEATURES.md` names
missed-entry rate as a monitored metric for exactly this reason. The trades it did not
take are the evidence, so they are recorded rather than dropped.

## Why the rule version is the deterministic policy, and why that ordering is forced

`FEATURES.md` line 99: the deterministic exit policy — immediate entry, triple
barrier, ATR trail, ratchet — is the P1 fallback and the permanent rollback target,
and it **generates the dataset PROFIT-TAIL is later trained on**. So a rule
PROFIT-TAIL is not a placeholder standing in for the real one; it is the thing that
has to run first for the real one to have a training set at all.

`strategy.deterministic_exit` already implements that policy and is reused here
rather than reimplemented.

## A win-rate objective would invert the design

Spec §2 warns of it and it is worth keeping in front of whoever trains the successor:
optimise hit rate and the bot learns to take a tick of profit off every winner and
hold every loser, which maximises the metric and loses money. The objective is the
distribution of forward P&L, not the fraction of trades that end green.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, field
from decimal import Decimal

# Position actions. Strings because they are journalled, rendered on a tile and read
# back by the performance report; a name that survives serialisation unchanged is one
# fewer thing to keep in step.
HOLD = "HOLD"
RATCHET_LOCK = "RATCHET_LOCK"
SCALE_OUT = "SCALE_OUT"
CLOSE = "CLOSE"
REQUEST_ADD = "REQUEST_ADD"

# Entry timing outcomes.
ENTER_NOW = "ENTER_NOW"
WAIT = "WAIT"
MISSED_ENTRY = "MISSED_ENTRY"

# Why a close was forced. A close that the hard stop caused and a close PROFIT-TAIL
# chose are different events and the journal must not blur them: one is the risk
# system working, the other is the bot's own judgement, and their rates say different
# things about the system.
STOP_BREACHED = "STOP_BREACHED"
TARGET_REACHED = "TARGET_REACHED"
SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
TAIL_EXHAUSTED = "TAIL_EXHAUSTED"


class ProfitTailOverreach(RuntimeError):
    """PROFIT-TAIL attempted something its authority forbids.

    Raised rather than ignored. A silent no-op would leave a bot whose exit logic
    quietly does nothing, and the position would sit open looking managed.
    """


@dataclass(frozen=True)
class TailAssessment:
    """PROFIT-TAIL's ADVISORY numbers, handed to the arbiter before selection.

    Advisory is a load-bearing word. The arbiter consumes these as features; this
    object carries no side, no veto and no authority, and `segment.arbiter` never
    reads a boolean off it.
    """

    venue: str
    symbol: str
    # Expected net P&L per unit as a fraction of price, fees and modelled slippage
    # already taken out. Signed: negative is a trade expected to lose.
    net_expectancy: Decimal
    # The bad tail, as a positive fraction. How much of the move goes against the
    # position before it resolves, at the quantile the segment cares about.
    loss_tail: Decimal
    confidence: Decimal
    evidence: dict
    at_ns: int

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(
                f"{self.symbol}: a tail assessment with no evidence cannot be "
                f"journalled; BF-04 requires what produced it")


@dataclass(frozen=True)
class EntryDecision:
    """When to enter, or the record that the chance was lost."""

    outcome: str
    venue: str
    symbol: str
    side: str
    reason: str
    evidence: dict
    at_ns: int
    limit_price: Decimal | None = None

    @property
    def is_missed(self) -> bool:
        return self.outcome == MISSED_ENTRY


@dataclass(frozen=True)
class PositionDirective:
    """What PROFIT-TAIL wants done with an open position.

    Every one of these re-enters the risk gate. `reduce_only` is set on anything that
    closes or trims, so a close can never accidentally open the reverse position —
    spec §2 rule 4.
    """

    action: str
    venue: str
    symbol: str
    reason: str
    evidence: dict
    at_ns: int
    quantity: Decimal | None = None
    reduce_only: bool = True
    # Where the ratchet has locked profit to. Only moves in the favourable direction;
    # a ratchet that can retreat is a trailing stop with extra steps.
    locked_stop: Decimal | None = None


@dataclass(frozen=True)
class OpenPosition:
    """What `manage()` is told about a live position. Read-only by construction."""

    venue: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    entry_ns: int
    # Set by the RISK GATE at fill and never moved by PROFIT-TAIL. Spec §4: the hard
    # stop is outside all of it and overrides everything.
    hard_stop: Decimal
    band: str
    locked_stop: Decimal | None = None
    # **The best and the worst this position ever got while it was open (RL-042).**
    #
    # `peak_favourable` existed here before and was assigned NOWHERE in the tree, so
    # it was `None` for every position that has ever existed - a field that read as
    # built and was not. Both are now written by `sample_excursions` on every poll.
    #
    # They are SAMPLED, not true extremes. The engine polls every six seconds and
    # cannot see between polls, so each is a LOWER BOUND on the real excursion, and
    # `excursion_samples` is carried beside them so a two-sample trade is never read
    # as a measured one.
    peak_favourable: Decimal | None = None
    peak_adverse: Decimal | None = None
    excursion_samples: int = 0
    # True when this position was rebuilt from the journal after a restart. Its
    # peaks then cover only the time SINCE the restart, and the move it made
    # before is gone - unrecoverable, because the marks that would show it were
    # never journalled. Carried so the close can say so rather than publish a
    # partial peak as if it were the whole trade's.
    excursion_from_restart: bool = False
    # **What this position was sized at (CL-05, CL-06).** Carried on the position
    # rather than looked up at close, because the declaration can be edited while a
    # trade is open and a reload applies to NEW ENTRIES ONLY - charging a close
    # against a multiple the trade was never opened at would bill it for a decision
    # nobody made.
    margin_usdt: Decimal | None = None
    notional_usdt: Decimal | None = None
    leverage: Decimal | None = None


def _is_stop_breached(position: OpenPosition, mark: Decimal) -> bool:
    if position.side == "LONG":
        return mark <= position.hard_stop
    return mark >= position.hard_stop


def _favourable_excursion(position: OpenPosition, mark: Decimal) -> Decimal:
    """Signed move in the position's favour, as a fraction of entry."""
    if position.entry_price == 0:
        return Decimal(0)
    move = (mark - position.entry_price) / position.entry_price
    return move if position.side == "LONG" else -move


def sample_excursions(position: OpenPosition, mark: Decimal) -> OpenPosition:
    """Fold one observed mark into this position's best and worst, and count it.

    Returns a NEW position rather than mutating, because `OpenPosition` is frozen
    by construction so that a brain cannot quietly rewrite the trade it is judging.

    **Called before `manage`, never inside it.** `manage` returns early on a
    breached hard stop and on max hold, so an excursion updated inside it would
    miss the very poll that closes the trade - which is the one poll that decides
    whether the journalled peak brackets the realised result at all.

    Both figures are non-negative magnitudes: `peak_favourable` is how far it ever
    went the right way, `peak_adverse` how far the wrong way. A trade that never
    went green has `peak_favourable` of zero, not a negative number, so the two
    columns never have to be read against each other's sign.
    """
    excursion = _favourable_excursion(position, mark)
    favourable = max(excursion, Decimal(0))
    adverse = max(-excursion, Decimal(0))
    return replace(
        position,
        peak_favourable=(favourable if position.peak_favourable is None
                         else max(favourable, position.peak_favourable)),
        peak_adverse=(adverse if position.peak_adverse is None
                      else max(adverse, position.peak_adverse)),
        excursion_samples=position.excursion_samples + 1)


@dataclass
class ProfitTail:
    """The rule PROFIT-TAIL: deterministic policy, honest label, real authority limits.

    `segment` and `name` travel into the journal so the four segments' timing records
    stay separable — RL-019 makes each segment its own bot, and a shared journal that
    could not be split by segment would make that unmeasurable.
    """

    segment: str
    name: str
    # Take profit and stop distances as fractions of entry price. Stated as rules per
    # segment in `segment.bot_registry`, never tuned here — PB-06's acceptance
    # requires each band's exit be a rule rather than a fitted constant.
    take_profit: Decimal
    ratchet_trigger: Decimal
    ratchet_give_back: Decimal
    # How long a selected-but-unentered signal stays valid. Past it the trade is
    # abandoned and attributed as a miss.
    signal_expiry_ns: int
    # How long a position may live regardless of price. RL-018 makes every segment
    # intraday, so a position with no time bound is a design error, not a stylistic one.
    max_hold_ns: int
    makes_edge_claim: bool = False

    # ------------------------------------------------------------------ advisory

    def assess(self, *, venue: str, symbol: str, frame, at_ns: int) -> TailAssessment:
        """The advisory numbers the arbiter consumes. Never a side, never a veto."""
        spread = frame.get("relative_spread")
        # The WINDOW volatility, never the per-observation one. See the note in
        # `segment.live_features`: comparing a per-tick sigma to a take-profit made
        # every expectancy negative and every selected trade untaken.
        volatility = frame.get("window_volatility")
        window_ns = frame.get("window_ns") or 0
        evidence = {"relative_spread": spread, "window_volatility": volatility,
                    "window_ns": window_ns, "max_hold_ns": self.max_hold_ns,
                    "policy": "deterministic", "segment": self.segment}
        if spread is None or volatility is None:
            # A refusal naming what was missing, never a default. The expectancy is
            # zero and the tail is the whole take-profit distance, which makes the
            # trade unattractive to the arbiter WITHOUT PROFIT-TAIL refusing it -
            # the distinction the ruling turns on.
            return TailAssessment(
                venue=venue, symbol=symbol, net_expectancy=Decimal(0),
                loss_tail=self.take_profit, confidence=Decimal(0),
                evidence={**evidence, "missing": [
                    k for k, v in (("relative_spread", spread),
                                   ("window_volatility", volatility)) if v is None]},
                at_ns=at_ns)
        spread = Decimal(str(spread))
        volatility = Decimal(str(volatility))
        # Scale the window's movement to THIS band's holding horizon before comparing
        # it to anything. A 60-second volatility and a 5-minute hold are different
        # durations, and square-root-of-time is the standard, stated, un-fitted way to
        # put them on one scale. Without it the fast and the slow band would read the
        # same market as offering the same opportunity, which is precisely the
        # comparison PB-06 exists to make.
        horizon_scale = Decimal(1)
        if window_ns > 0 and self.max_hold_ns > 0:
            horizon_scale = Decimal(str(math.sqrt(self.max_hold_ns / window_ns)))
        reachable = volatility * horizon_scale
        evidence["horizon_scale"] = str(horizon_scale)
        evidence["reachable_move"] = str(reachable)

        # What is collectable is bounded by what moves, and what it costs is the
        # spread crossed twice. Deliberately crude and deliberately stated: this is
        # the deterministic baseline a trained PROFIT-TAIL must beat, and a baseline
        # nobody can read is a baseline nobody can beat honestly.
        collectable = min(self.take_profit, reachable)
        net = collectable - (spread * 2)
        return TailAssessment(
            venue=venue, symbol=symbol, net_expectancy=net,
            loss_tail=max(reachable, self.take_profit), confidence=Decimal("0.5"),
            evidence={**evidence, "collectable": str(collectable),
                      "round_trip_spread": str(spread * 2)},
            at_ns=at_ns)

    # -------------------------------------------------------------------- timing

    def time_entry(self, *, venue: str, symbol: str, side: str, selected_at_ns: int,
                   now_ns: int, frame) -> EntryDecision:
        """Decide the moment. May wait; may never refuse.

        There is no branch that returns "do not take this trade". The only outcome
        that does not end in an entry is MISSED_ENTRY, which is time expiring, is
        attributed to PROFIT-TAIL, and is journalled as its failure rather than as a
        decision it was entitled to make.
        """
        age = now_ns - selected_at_ns
        quote_age = frame.get("quote_age_ns")
        evidence = {"signal_age_ns": age, "quote_age_ns": quote_age,
                    "policy": "deterministic", "segment": self.segment}
        if age > self.signal_expiry_ns:
            return EntryDecision(
                outcome=MISSED_ENTRY, venue=venue, symbol=symbol, side=side,
                reason=SIGNAL_EXPIRED,
                evidence={**evidence, "expiry_ns": self.signal_expiry_ns},
                at_ns=now_ns)
        if not frame.get("has_two_sided_quote"):
            # Waiting for a quote is timing, not refusal: the signal is still alive
            # and this returns WAIT, so the next poll reconsiders it.
            return EntryDecision(
                outcome=WAIT, venue=venue, symbol=symbol, side=side,
                reason="NO_TWO_SIDED_QUOTE", evidence=evidence, at_ns=now_ns)
        # The deterministic policy enters immediately. `FEATURES.md` line 99 makes
        # this the P1 fallback whose record trains the learned successor, so the
        # baseline has to be the un-clever one - an aggressive rule here would put a
        # tuned constant into the dataset the trained bot inherits.
        return EntryDecision(
            outcome=ENTER_NOW, venue=venue, symbol=symbol, side=side,
            reason="IMMEDIATE_ENTRY_BASELINE", evidence=evidence, at_ns=now_ns)

    # ---------------------------------------------------------------- management

    def manage(self, *, position: OpenPosition, mark: Decimal,
               now_ns: int, frame=None) -> PositionDirective:
        """What to do with an open position. Cannot refuse to close a loser.

        The hard stop is checked FIRST and returns CLOSE unconditionally. No later
        branch can reach past it, so the prohibition is enforced by control flow
        rather than by a rule somebody has to remember.
        """
        evidence = {"mark": str(mark), "entry": str(position.entry_price),
                    "hard_stop": str(position.hard_stop), "band": position.band,
                    "policy": "deterministic", "segment": self.segment}

        if _is_stop_breached(position, mark):
            return PositionDirective(
                action=CLOSE, venue=position.venue, symbol=position.symbol,
                reason=STOP_BREACHED, evidence=evidence, at_ns=now_ns,
                quantity=position.quantity, reduce_only=True)

        held = now_ns - position.entry_ns
        if held >= self.max_hold_ns:
            # RL-018: intraday on every segment. Time is an exit reason in its own
            # right, and one that is never negotiable.
            return PositionDirective(
                action=CLOSE, venue=position.venue, symbol=position.symbol,
                reason="MAX_HOLD_REACHED",
                evidence={**evidence, "held_ns": held}, at_ns=now_ns,
                quantity=position.quantity, reduce_only=True)

        excursion = _favourable_excursion(position, mark)
        evidence["favourable_excursion"] = str(excursion)

        if excursion >= self.take_profit:
            return PositionDirective(
                action=CLOSE, venue=position.venue, symbol=position.symbol,
                reason=TARGET_REACHED, evidence=evidence, at_ns=now_ns,
                quantity=position.quantity, reduce_only=True)

        if excursion >= self.ratchet_trigger:
            # Lock profit behind the move. The locked level only ever advances -
            # `max`/`min` against the existing one - because a ratchet that can
            # retreat gives back exactly what it was built to protect.
            give_back = self.ratchet_give_back
            if position.side == "LONG":
                candidate = position.entry_price * (1 + excursion - give_back)
                locked = (candidate if position.locked_stop is None
                          else max(candidate, position.locked_stop))
            else:
                candidate = position.entry_price * (1 - excursion + give_back)
                locked = (candidate if position.locked_stop is None
                          else min(candidate, position.locked_stop))
            return PositionDirective(
                action=RATCHET_LOCK, venue=position.venue, symbol=position.symbol,
                reason="PROFIT_RATCHETED",
                evidence={**evidence, "locked_stop": str(locked)}, at_ns=now_ns,
                locked_stop=locked, reduce_only=True)

        if position.locked_stop is not None and _crossed_lock(position, mark):
            return PositionDirective(
                action=CLOSE, venue=position.venue, symbol=position.symbol,
                reason=TAIL_EXHAUSTED,
                evidence={**evidence, "locked_stop": str(position.locked_stop)},
                at_ns=now_ns, quantity=position.quantity, reduce_only=True)

        return PositionDirective(
            action=HOLD, venue=position.venue, symbol=position.symbol,
            reason="RUNNING", evidence=evidence, at_ns=now_ns, reduce_only=True)

    # ------------------------------------------------------------- authority guard

    def reject(self, *args, **kwargs):
        """Exists only to fail. PROFIT-TAIL has no rejection authority.

        A method that raises is better than no method at all: a caller reaching for
        one finds this and the ruling behind it, instead of finding nothing and
        concluding the check belongs somewhere else.
        """
        raise ProfitTailOverreach(
            "PROFIT-TAIL cannot reject a selected trade (RL-023). Its expectancy and "
            "tail estimates are inputs the arbiter consumes BEFORE selection; after "
            "selection it owns timing and the position, and nothing else.")


def _crossed_lock(position: OpenPosition, mark: Decimal) -> bool:
    locked = position.locked_stop
    if locked is None:
        return False
    return mark <= locked if position.side == "LONG" else mark >= locked
