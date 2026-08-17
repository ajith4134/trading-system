"""The forward paper engine: the supervised process Phase J is waiting on.

    python -m paper.forward_engine --strategy plumbing-momentum --participation 0.1

This is the root that makes the paper path reachable. Until it existed,
`position_book`, `paper_broker` and `market_replay` were three modules importing
each other with nothing running any of them — a dead subsystem vouching for
itself, which is the shape `integrity.unsupported_claims` was written to catch and
which `CLAUDE.md` names as the trap this codebase keeps falling into.

**`--strategy` has no default, deliberately.** The only signal that exists today
is a plumbing signal with no edge claim, and an engine that ran it by default
would quietly accumulate a P&L series that later reads as a result. Naming it is
the point: `makes_edge_claim` is false for it, and that flag rides on every fill
row and every heartbeat rather than sitting in a header that gets separated from
its data.

**It refuses to start while a kill is in force.** `ops.watchdog.is_killed` is the
one question a trading path must ask, and it is asked before the journal is
touched — a run that was killed should leave no record implying it traded.

**Every poll writes a heartbeat**, whether or not anything happened. An engine
that is running and finding nothing produces exactly the same fills as an engine
that died three days ago, and this box has already demonstrated the failure: it
was off from 2026-08-10 to 2026-08-15 while the board went on looking healthy.

No network, no credentials, no venue. `PaperBroker` is a local object and the
existence of verified fee data does not make this able to trade.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import json

from cost.fee_schedule import FeeRate
from execution.order_intent_wal import IntentExpired, OrderIntent, OrderIntentWal
from ops.watchdog import is_killed, kill_reason
from paper.equity_curve import EquityCurve
from paper.exit_manager import ExitManager
from paper.fill_model import Participation
from paper.forward_journal import ForwardJournal
from paper.market_replay import MarketReplay, TapeEvent
from paper.paper_broker import PaperBroker
from risk.pre_trade_gate import (
    LimitsNotSet, PreTradeGate, read_gate_limits, seed_gate_limits,
)
from risk.tail_cap import CeilingNotSet, read_ceiling
from store.clock_gated_reader import ClockGatedReader
from strategy.deterministic_exit import Bar, CorruptBar
from validation.holdout_custodian import HoldoutCustodian

BARS_DATASET = "bars_60000000000ns"
# One bar. An intent that outlives the bar it was decided on would fire against a
# market it never saw, which is what valid_for_ns exists to prevent.
INTENT_VALID_NS = 60_000_000_000

# Verified 2026-08-09 from signed endpoints at this account's own tier, recorded
# in ~/capture/fee-verification/latest.json. Not a documentation page.
VERIFIED_FEES = {
    "binance": FeeRate(maker_bps=Decimal("2.0"), taker_bps=Decimal("5.0")),
    "hyperliquid": FeeRate(maker_bps=Decimal("1.5"), taker_bps=Decimal("4.5")),
}


class KillSwitchEngaged(RuntimeError):
    """A kill is in force, so the engine must not start."""


class PlumbingMomentum:
    """Buy at market on every bar after the first.

    Changed 2026-08-16 at the user's instruction that orders be market orders
    wherever possible. It rested a BUY at the previous close until then, and the
    market version is both simpler and more honest here: a resting order needs a
    participation rate nobody has measured, which is why every fill this system
    has produced carries `uncalibrated=true`. A market order does not queue, so
    that assumption disappears from these fills entirely.

    **This is not a strategy and its P&L is not evidence of edge.** It is a coin
    flip with extra steps, kept because the engine needs *something* to submit
    before Phase C produces a model, and because a signal whose worthlessness is
    stated is safer than an engine with no orders at all — the fill path only
    proves itself when orders actually flow through it.

    `makes_edge_claim` is False and that value is written onto every fill row.
    """

    name = "plumbing-momentum"
    makes_edge_claim = False

    def __init__(self, quantity: Decimal) -> None:
        self._quantity = quantity
        self._previous_close: dict[tuple[str, str], Decimal] = {}

    def __call__(self, event: TapeEvent, now_ns: int) -> OrderIntent | None:
        """`now_ns` is the DECISION clock, not the bar's.

        `created_at_ns` is when this signal was computed, which is now. Stamping
        it with the bar's event time conflates two different things and gets one
        of them wrong: EX-003 asks "was this decision made too long ago to act
        on", and the age of the DATA behind the decision is a separate question
        that `features.staleness` (FE-001) already carries per value.

        Measured 2026-08-15, which is how this was found: stamped with the bar
        time, a run over the live archive fed 3,494 events and submitted zero -
        every intent expired against a 60-second window before it could be sent,
        and the engine looked idle rather than blocked.
        """
        key = (event.venue, event.symbol)
        previous = self._previous_close.get(key)
        self._previous_close[key] = event.market_event.trade_price
        if previous is None:
            return None
        return OrderIntent(
            strategy=self.name, symbol=event.symbol, venue=event.venue,
            side="BUY", quantity=self._quantity,
            created_at_ns=now_ns, valid_for_ns=INTENT_VALID_NS,
            # No price: a MARKET order. It crosses on the next print, never on
            # the one that produced this decision - the engine applies prints to
            # resting orders before it submits from them.
            price=None)


STRATEGIES = {PlumbingMomentum.name: PlumbingMomentum}


@dataclass
class EngineCounts:
    """What the run did, counted by outcome rather than summarised into one word."""

    polls: int = 0
    events_fed: int = 0
    orders_submitted: int = 0
    # Exits counted apart from entries. Lumped together, a run that stopped
    # exiting entirely reads as a run that submitted slightly fewer orders.
    exits_submitted: int = 0
    exits_proposed: int = 0
    # Blocked BEFORE the WAL, so they were never written as something that was
    # sent. Counted apart from broker rejections: the gate refusing an order is
    # a risk decision, and the broker refusing one is a market or config
    # condition, and a board that lumps them cannot tell a cap from an outage.
    orders_gated: int = 0
    orders_rejected: int = 0
    orders_expired: int = 0
    # Counted apart from orders_rejected because they mean different things: the
    # broker refusing an order is a market/config condition, and the WAL refusing
    # one is its duplicate guard working. Lumped together, a cold-start burst of
    # duplicates reads as a broker rejecting everything.
    orders_duplicate: int = 0
    primed_events: int = 0
    fills: int = 0
    last_event_time_ns: int | None = None


class ForwardEngine:
    """One poll of the tape, applied to the broker, journalled. Repeat."""

    def __init__(self, *, replay: MarketReplay, broker: PaperBroker,
                 wal: OrderIntentWal, journal: ForwardJournal, strategy,
                 kill_root: Path, exits: ExitManager | None = None,
                 gate: PreTradeGate | None = None,
                 nav: Decimal | None = None,
                 equity: EquityCurve | None = None) -> None:
        self._replay = replay
        self._broker = broker
        self._wal = wal
        self._journal = journal
        self._strategy = strategy
        self._kill_root = Path(kill_root)
        # Optional so a caller can run the engine with no exit policy at all -
        # which is what it did until 2026-08-16, and which produced 12,983 fills
        # and zero closed round trips. A default of None rather than a default
        # policy: an exit policy applied because nobody passed one is a policy
        # nobody chose.
        self._exits = exits
        # `FEATURES.md` §6 (P0): "blocks the order BEFORE it is sent". Optional
        # so the engine can be run ungated deliberately - which is what it was
        # until 2026-08-16, and which produced 363 unbounded positions.
        self._gate = gate
        self._nav = nav
        # The paper book's equity through time, and the adaptive cap measured
        # against it. Optional so the engine runs without a live ceiling to
        # bound against - which is a real state, not a failure - and it says so
        # rather than inventing one.
        self._equity = equity
        self.counts = EngineCounts()

    def _open_positions(self) -> dict:
        """Signed quantity per (venue, symbol), from the OPTIMISTIC book.

        One accounting has to be chosen and it is stated: the two hold identical
        sizes by construction - `paper_broker.AccountingDiverged` is raised if
        they ever do not - so the choice affects nothing but the name in this
        line, and picking one silently would leave a reader guessing which.
        """
        return {(p.venue, p.symbol): p.quantity
                for p in self._broker.optimistic.local_positions()}

    def _assert_not_killed(self) -> None:
        if is_killed(self._kill_root):
            reason = kill_reason(self._kill_root) or "no reason recorded"
            raise KillSwitchEngaged(
                f"a kill is in force ({reason}) - refusing to run, before "
                f"anything is journalled")

    def resume_or_prime(self, now_ns: int, state_dir: Path) -> int:
        """Skip the full prime when a watermark from a previous run survives.

        Measured 2026-08-17: a cold start re-fed 2,316,171 archived events before
        the first poll, ~11 minutes with nothing trading, and the process was
        OOM-killed at 5.5 GB (exit 137, fourth restart that day). None of that
        work was new - the previous process had already done it.

        The bounded mark-seen still runs, and it is not optional: the watermark
        bound is inclusive, so rows sitting exactly at it would otherwise be read
        with an empty emitted set and TRADED. It is cheap, because it reads only
        from the watermark forward.

        A marker for another strategy, or an unreadable one, falls back to the
        full prime. A slow start is an acceptable cost; a wrong one is not - the
        2026-08-15 defect turned 3,494 archived bars into 1,074 fills at prices
        days old, and that direction of failure must stay unreachable.
        """
        marker = read_prime_marker(state_dir, strategy=self._strategy.name)
        if marker is None:
            return self.prime(now_ns)

        self._assert_not_killed()
        self._replay.resume_at(marker)
        marked = self._replay.mark_seen(now_ns, not_before_ns=marker)
        self._journal.record_heartbeat(
            now_ns=now_ns, strategy=self._strategy.name,
            makes_edge_claim=self._strategy.makes_edge_claim,
            events_fed=0, orders_submitted=0, orders_rejected=0, fills=0,
            open_orders=0, last_event_time_ns=None,
            detail=f"resumed at availability watermark {marker}; {marked} row(s) "
                   f"at or after it marked as seen without trading")
        return marked

    def prime(self, now_ns: int) -> int:
        """Mark everything already in the store as fed, WITHOUT trading it.

        Forward operation has to begin at the current clock. Without this, the
        first poll of a fresh journal replays the whole archive as though it were
        live: measured 2026-08-15 on BTCUSDT alone, 3,494 archived bars fed in one
        poll produced 1,074 fills against prices days old, and every one of them
        would have entered the forward journal as a forward result.

        It is not a backtest either - a backtest advances a simulated clock and
        this discards. Replay is `poll()` with a simulated clock, which is the
        whole reason there is one method.
        """
        self._assert_not_killed()
        # `mark_seen`, not `len(poll(...))`. The old form built a TapeEvent and a
        # MarketEvent for every archived row purely to count them - 2,316,171 of
        # them on 2026-08-17, at 5.5 GB, on a box with no swap.
        fed = self._replay.mark_seen(now_ns)
        self.counts.primed_events += fed
        self._journal.record_heartbeat(
            now_ns=now_ns, strategy=self._strategy.name,
            makes_edge_claim=self._strategy.makes_edge_claim,
            events_fed=0, orders_submitted=0, orders_rejected=0, fills=0,
            open_orders=0, last_event_time_ns=None,
            detail=f"primed: {fed} archived event(s) marked as seen without "
                   f"trading, so forward operation starts at this clock")
        return fed

    def _submit(self, intent, now_ns: int, *, is_exit: bool,
                reference_price=None) -> None:
        """Send one intent and count its outcome. Shared by entries and exits.

        The pre-trade gate runs FIRST, before the WAL, because §6's own words are
        "blocks the order before it is sent" - an order written to the intent log
        and then blocked is a different claim from one that never existed.

        Exits are counted apart from entries throughout: lumped together, a run
        that stopped exiting entirely would read as one that submitted slightly
        fewer orders.
        """
        if self._gate is not None:
            decision = self._gate.evaluate(
                venue=intent.venue, symbol=intent.symbol, side=intent.side,
                quantity=intent.quantity, reference_price=reference_price,
                limit_price=intent.price, nav=self._nav,
                open_positions=self._open_positions(), now_ns=now_ns)
            if not decision.approved:
                self.counts.orders_gated += 1
                return
        try:
            response = self._wal.submit(intent, self._broker, now_ns=now_ns)
        except IntentExpired:
            self.counts.orders_expired += 1
            return
        except RuntimeError:
            # Already submitted: two decisions in the same nanosecond, on the
            # same instrument, size and limit, hash to the same client order id.
            # That is the deterministic id doing its job - blind-retrying an
            # order whose response was lost is a second order - and it is not a
            # broker rejection, so it is counted apart.
            self.counts.orders_duplicate += 1
            return
        if response.get("status") == "rejected":
            self.counts.orders_rejected += 1
        elif is_exit:
            self.counts.exits_submitted += 1
        else:
            self.counts.orders_submitted += 1

    def poll_once(self, now_ns: int) -> tuple:
        """Feed everything newly available, submit, fill, journal. Returns fills."""
        self._assert_not_killed()
        self.counts.polls += 1

        produced = []
        for event in self._replay.poll(now_ns):
            self.counts.events_fed += 1
            self.counts.last_event_time_ns = event.event_time_ns

            # Fills first, on the orders already resting. An order submitted from
            # this same print must not fill on it: it was not in the book when the
            # trade happened, and awarding it a fill is the cheapest way to invent
            # a strategy that front-runs its own data.
            for fill in self._broker.on_market_event(
                    event.symbol, event.venue, event.market_event):
                self.counts.fills += 1
                produced.append(fill)
                self._journal.record_fill(
                    fill, now_ns=now_ns, strategy=self._strategy.name,
                    makes_edge_claim=self._strategy.makes_edge_claim)
                if self._exits is not None:
                    self._exits.on_fill(
                        venue=fill.venue, symbol=fill.symbol, side=fill.side,
                        quantity=fill.quantity,
                        optimistic_price=fill.optimistic_price,
                        pessimistic_price=fill.pessimistic_price)

            # The bar is shown to the exit manager AFTER fills on it and BEFORE
            # any new entry, so the path a position is judged on starts at the
            # bar after its own entry - a position cannot be exited by the print
            # that opened it.
            if self._exits is not None:
                try:
                    self._exits.observe(
                        event.venue, event.symbol,
                        Bar(high=event.market_event.best_ask,
                            low=event.market_event.best_bid,
                            close=event.market_event.trade_price),
                        at_ns=event.event_time_ns)
                except CorruptBar:
                    # A bar that cannot be traded through - the store's
                    # documented placeholder-price defect. Counted and skipped
                    # rather than walked: a policy that walks one exits at
                    # nothing, and the replay that found this booked a loss of
                    # 187,831 per unit off exactly that.
                    self._exits.corrupt_bars += 1
                proposal = self._exits.exit_proposal(
                    event.venue, event.symbol, now_ns)
                if proposal is not None:
                    self.counts.exits_proposed += 1
                    self._submit(proposal.intent, now_ns, is_exit=True,
                                 reference_price=event.market_event.trade_price)

            intent = self._strategy(event, now_ns)
            if intent is None:
                continue
            self._submit(intent, now_ns, is_exit=False,
                         reference_price=event.market_event.trade_price)

        # Recorded AFTER the poll's fills, so the sample reflects what this poll
        # did rather than what the last one left behind. Assessed in PAPER mode:
        # a breach is recorded and nothing is stopped, because enforcing it
        # would destroy the evidence by preventing the breach it observes.
        if self._equity is not None:
            self._equity.record(
                at_ns=now_ns,
                realised_optimistic=self._broker.optimistic.realized_pnl_after_fees,
                realised_pessimistic=self._broker.pessimistic.realized_pnl_after_fees,
                unmarked_positions=len(self._open_positions()))
            self._equity.assess_latest()

        self._journal.record_heartbeat(
            now_ns=now_ns, strategy=self._strategy.name,
            makes_edge_claim=self._strategy.makes_edge_claim,
            events_fed=self.counts.events_fed,
            # Entries plus exits: the heartbeat's "submitted" is how many
            # orders went out, and an exit is an order.
            orders_submitted=(self.counts.orders_submitted
                              + self.counts.exits_submitted),
            orders_rejected=(self.counts.orders_rejected
                             + self.counts.orders_expired
                             + self.counts.orders_duplicate
                             + self.counts.orders_gated),
            fills=self.counts.fills,
            open_orders=self._broker.open_order_count,
            last_event_time_ns=self.counts.last_event_time_ns,
            detail=self._replay.watermark_note)
        return tuple(produced)


def build_engine(*, store_root: Path, capture_root: Path, strategy_name: str,
                 participation: Participation, quantity: Decimal,
                 symbols=None, holdout_start_ns: int,
                 holdout_end_ns: int) -> ForwardEngine:
    """Wire the engine. The custodian is not optional on this path."""
    if strategy_name not in STRATEGIES:
        raise SystemExit(
            f"unknown strategy {strategy_name!r}; available: "
            f"{sorted(STRATEGIES)}. There is no default - the only signal that "
            f"exists makes no edge claim, and running it unnamed would let its "
            f"P&L be read later as a result")
    paper_root = Path(capture_root) / "paper" / "forward"
    custodian = HoldoutCustodian(Path(capture_root) / "holdout",
                                 holdout_start_ns=holdout_start_ns,
                                 holdout_end_ns=holdout_end_ns)
    reader = ClockGatedReader(Path(store_root), BARS_DATASET, custodian=custodian)
    # FEATURES.md §6 (P0). Seeded on first run so the gate exists rather than
    # being skipped by absence, and then never overwritten - the numbers are the
    # user's to edit. If the file is unreadable the engine runs UNGATED and says
    # so, rather than refusing to trade at all: a paper engine that will not
    # start is a paper engine nobody watches.
    risk_root = Path(capture_root) / "risk"
    seed_gate_limits(risk_root)
    try:
        limits, nav = read_gate_limits(risk_root)
        gate = PreTradeGate(limits)
    except LimitsNotSet as error:
        print(f"NO PRE-TRADE GATE: {error}", file=sys.stderr)
        gate, nav = None, None

    # The equity curve, and the adaptive paper cap measured against it. Needs
    # the live ceiling as the bound the paper cap may not exceed; without one
    # there is nothing to bound against, so the engine runs without the curve
    # and says so rather than inventing a ceiling §6 reserves to the user.
    equity = None
    if nav is not None:
        try:
            equity = EquityCurve(nav=nav, live_ceiling=read_ceiling(risk_root))
        except CeilingNotSet as error:
            print(f"NO EQUITY CAP: {error}", file=sys.stderr)

    return ForwardEngine(
        replay=MarketReplay(reader=reader, symbols=symbols),
        broker=PaperBroker(participation=participation,
                           fees_by_venue=VERIFIED_FEES),
        wal=OrderIntentWal(paper_root / "wal"),
        journal=ForwardJournal(paper_root),
        strategy=STRATEGIES[strategy_name](quantity=quantity),
        kill_root=Path(capture_root) / "ops",
        gate=gate, nav=nav, equity=equity,
        # Wired 2026-08-16. Until then the engine had no exit at all: 12,983
        # fills, 363 open positions and zero closed round trips, because
        # `plumbing-momentum` rests a bid and never sells. The policy is
        # `strategy.deterministic_exit`, the P1 fallback every learned exit has
        # to beat.
        exits=ExitManager(strategy_name))


_MARKER_NAME = "prime-marker.json"


def write_prime_marker(state_dir: Path, last_availability_ns: int,
                       strategy: str = "") -> Path:
    """Record how far the archive was primed, so a restart resumes.

    An AVAILABILITY time, not an event time. The engine's contract is that it
    never trades a row it has already seen, and availability is the clock the
    store is gated on; stamping the event time would let a late-arriving
    correction to an old bar look like something new.

    The strategy is part of the marker because a different strategy has NOT seen
    this archive, and resuming from its neighbour's watermark would silently skip
    every row it should have primed on - producing an engine that looks healthy
    and is trading a tape it never read the start of.
    """
    marker = Path(state_dir) / _MARKER_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "last_availability_ns": int(last_availability_ns),
        "strategy": strategy,
    }))
    return marker


def read_prime_marker(state_dir: Path, strategy: str = "") -> int | None:
    """The watermark to resume from, or None meaning prime the whole archive.

    Every failure path returns None. A corrupt, foreign or absent marker must
    cost a slow start, never a wrong one: the 2026-08-15 defect turned 3,494
    archived bars into 1,074 forward fills at prices days old, and that is the
    direction this must never fail in.
    """
    marker = Path(state_dir) / _MARKER_NAME
    if not marker.is_file():
        return None
    try:
        held = json.loads(marker.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(held, dict):
        return None
    if strategy and held.get("strategy", "") != strategy:
        return None
    value = held.get("last_availability_ns")
    return int(value) if isinstance(value, int) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="paper.forward_engine",
        description="The forward paper engine. Journals fills under both "
                    "accountings; claims no edge it was not told to claim.")
    parser.add_argument("--strategy", required=True,
                        help=f"one of {sorted(STRATEGIES)}. No default, on "
                             f"purpose - see the module docstring")
    parser.add_argument("--participation", required=True,
                        help="share of printed volume a resting order receives. "
                             "No default: an invented rate is what manufactures "
                             "edge. Open question 1 of the design")
    parser.add_argument("--participation-calibrated", action="store_true",
                        help="assert the rate came from the depth archive. Off "
                             "by default, and every fill then carries "
                             "uncalibrated=true")
    parser.add_argument("--quantity", default="0.001")
    parser.add_argument("--symbols", default=None,
                        help="comma-separated; omit for every symbol in the store")
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--polls", type=int, default=0,
                        help="stop after this many polls; 0 runs until stopped")
    parser.add_argument("--replay-archive-on-start", action="store_true",
                        help="trade everything already in the store on the first "
                             "poll. OFF by default: forward operation starts at "
                             "the current clock, or a fresh journal opens with "
                             "days-old fills recorded as forward results")
    parser.add_argument("--holdout-start-ns", type=int, default=2_000_000_000_000_000_000)
    parser.add_argument("--holdout-end-ns", type=int, default=2_100_000_000_000_000_000)
    args = parser.parse_args(argv)

    # The same directory `build_engine` puts the journal in - the marker is
    # state about this journal's progress and belongs beside it, so a journal
    # moved or archived takes its watermark with it rather than leaving one
    # behind for the next run to resume from.
    state_dir = Path(args.capture_root) / "paper" / "forward"

    engine = build_engine(
        store_root=Path(args.store_root), capture_root=Path(args.capture_root),
        strategy_name=args.strategy,
        participation=Participation(fraction=Decimal(args.participation),
                                    calibrated=args.participation_calibrated),
        quantity=Decimal(args.quantity),
        symbols=(args.symbols.split(",") if args.symbols else None),
        holdout_start_ns=args.holdout_start_ns,
        holdout_end_ns=args.holdout_end_ns)

    print(f"forward paper engine: strategy={args.strategy} "
          f"edge_claim={STRATEGIES[args.strategy].makes_edge_claim} "
          f"participation={args.participation} "
          f"calibrated={args.participation_calibrated}", flush=True)

    if not args.replay_archive_on_start:
        primed = engine.resume_or_prime(time.time_ns(), state_dir)
        print(f"primed: {primed} archived event(s) marked as seen without "
              f"trading; forward operation starts now", flush=True)

    polls = 0
    while True:
        engine.poll_once(time.time_ns())
        polls += 1
        # Written AFTER the poll, never before: a marker ahead of the work it
        # claims would let the next start skip rows nothing has fed, and those
        # rows would then be invisible forever rather than merely re-read.
        write_prime_marker(state_dir, engine._replay.availability_watermark,
                           strategy=args.strategy)
        counts = engine.counts
        # orders_expired is printed because it was INVISIBLE on the first live
        # run: 3,494 events fed, 0 submitted, and nothing said that every single
        # intent had expired. A silent zero and a silent 3,494 looked identical.
        print(f"poll {polls}: {engine._replay.watermark_note}; "
              f"{counts.orders_submitted} submitted, "
              f"{counts.orders_rejected} rejected, "
              f"{counts.orders_gated} blocked by the pre-trade gate, "
              f"{counts.orders_duplicate} refused as duplicates, "
              f"{counts.orders_expired} expired before submission, "
              # Exits reported on the operator line, not only in the counts.
              # The engine ran for hours submitting nothing but entries and the
              # log said "N submitted" either way - a run that has stopped
              # exiting and one that never started look identical without this.
              f"{counts.exits_proposed} exit(s) proposed of which "
              f"{counts.exits_submitted} sent, "
              f"{counts.fills} filled", flush=True)
        if args.polls and polls >= args.polls:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
