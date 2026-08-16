"""Until this existed the engine had bought 12,983 times and closed nothing.

The test that matters is `test_a_round_trip_actually_closes_end_to_end`: an
entry, a move, an exit, and a position back at flat, driven through the real
engine rather than through the manager alone. Everything else defends the
bookkeeping around it - one exit at a time, sized to the position, priced off the
worse of the two entry fills.
"""
from decimal import Decimal

import pandas as pd
import pytest

from cost.fee_schedule import FeeRate
from execution.order_intent_wal import OrderIntentWal
from paper.exit_manager import BUY, SELL, ExitManager
from paper.fill_model import Participation
from paper.forward_engine import ForwardEngine, PlumbingMomentum
from paper.forward_journal import ForwardJournal
from paper.market_replay import MarketReplay
from paper.paper_broker import PaperBroker
from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from validation.holdout_custodian import HoldoutCustodian
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)
from strategy.deterministic_exit import (
    ATR_PERIOD, HARD_STOP, LONG, PROFIT_TARGET, Bar, ExitPolicy,
)

DATASET = "bars_60000000000ns"
BINANCE = FeeRate(maker_bps=Decimal("2.0"), taker_bps=Decimal("5.0"))
_VENUE, _SYMBOL = "binance", "BTCUSDT"


def _bar(high, low, close):
    return Bar(high=Decimal(str(high)), low=Decimal(str(low)),
               close=Decimal(str(close)))


def _warm(manager, n=ATR_PERIOD + 2, high=101, low=99, close=100):
    """Enough bars for an ATR of 2 before anything opens."""
    for _ in range(n):
        manager.observe(_VENUE, _SYMBOL, _bar(high, low, close))


def _fill(manager, side=BUY, quantity="1", optimistic="100", pessimistic="101"):
    manager.on_fill(venue=_VENUE, symbol=_SYMBOL, side=side,
                    quantity=Decimal(quantity),
                    optimistic_price=Decimal(optimistic),
                    pessimistic_price=Decimal(pessimistic))


# --- the rails reach the position ----------------------------------------

def test_a_position_that_runs_to_target_is_exited():
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    # ATR is 2, entry (pessimistic) 101, target = 101 + 4*2 = 109.
    manager.observe(_VENUE, _SYMBOL, _bar(110, 105, 109))

    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    assert proposal is not None
    assert proposal.record.exit_reason == PROFIT_TARGET
    assert proposal.intent.side == SELL


def test_a_position_that_stops_out_is_exited():
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    # Hard stop = 101 - 3*2 = 95.
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))

    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    assert proposal.record.exit_reason == HARD_STOP


def test_a_position_inside_its_rails_is_left_alone():
    """A manager that exits everything is not a policy."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(102, 100, 101))

    assert manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000) is None


# --- the rails are computed off the WORSE entry --------------------------

def test_the_rails_use_the_pessimistic_entry_price():
    """Using the optimistic entry would place every rail where the flattering
    fill put them - the same error as reporting one accounting."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager, optimistic="100", pessimistic="110")

    position = manager.position(_VENUE, _SYMBOL)

    assert position.entry_optimistic == Decimal(100)
    assert position.entry_pessimistic == Decimal(110)
    # Stop is 110 - 3*2 = 104, not 100 - 6 = 94. A bar to 100 stops this out.
    manager.observe(_VENUE, _SYMBOL, _bar(111, 100, 104))
    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)
    assert proposal.record.exit_reason == HARD_STOP


# --- exits are market orders ---------------------------------------------

def test_the_exit_is_a_market_order():
    """A stop that rests as a limit at its own level books the level, which is
    the one price a real stop does not get."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))

    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    assert proposal.intent.price is None
    assert proposal.intent.reduce_only


def test_a_gap_beyond_the_bar_is_declared_unmodelled():
    """Crossing IS modelled now the exit is a market order. A print that jumps
    past the level entirely is not, because the bar's high and low bound what
    this can see."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))

    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    assert proposal.slippage_beyond_bar_modelled is False


# --- one exit at a time, sized to the position ---------------------------

def test_only_one_exit_rests_at_a_time():
    """A second exit while one is pending would sell the position twice."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))

    first = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)
    manager.observe(_VENUE, _SYMBOL, _bar(101, 93, 94))
    second = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=2_000)

    assert first is not None
    assert second is None


def test_the_exit_is_sized_to_the_whole_position():
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager, quantity="2")
    _fill(manager, quantity="3")
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))

    proposal = manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    assert proposal.intent.quantity == Decimal(5)


def test_a_filled_exit_clears_the_position_and_the_pending_flag():
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(101, 94, 95))
    manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000)

    _fill(manager, side=SELL, optimistic="95", pessimistic="94")

    assert manager.position(_VENUE, _SYMBOL) is None
    assert manager.open_positions == 0


def test_adding_to_a_position_keeps_the_first_entry_s_atr():
    """Re-measuring on every add would let a position that grew during a calm
    stretch tighten its own stop, and one that grew during a spike widen it."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)
    first_atr = manager.position(_VENUE, _SYMBOL).atr_at_entry

    for _ in range(ATR_PERIOD + 2):
        manager.observe(_VENUE, _SYMBOL, _bar(140, 60, 100))    # violent
    _fill(manager)

    assert manager.position(_VENUE, _SYMBOL).atr_at_entry == first_atr


# --- no rails is not arbitrary rails -------------------------------------

def test_a_position_opened_with_no_measurable_atr_gets_no_rails():
    """A zero or absent ATR puts every rail on the entry price, where the first
    bar touches all of them. No rails, counted, rather than arbitrary ones."""
    manager = ExitManager("test")
    manager.observe(_VENUE, _SYMBOL, _bar(101, 99, 100))        # too few bars
    _fill(manager)
    manager.observe(_VENUE, _SYMBOL, _bar(200, 10, 20))

    assert manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000) is None
    assert manager.atr_unavailable == 1
    assert "no measurable ATR" in manager.describe()


def test_a_position_cannot_be_exited_by_the_bar_that_opened_it():
    """The path starts at the bar AFTER entry. A position exited by its own
    entry print is a strategy trading on information it did not have."""
    manager = ExitManager("test")
    _warm(manager)
    _fill(manager)

    assert manager.exit_proposal(_VENUE, _SYMBOL, now_ns=1_000) is None


# --- end to end, through the real engine ---------------------------------

def _row(event, available, close, high, low):
    return {SYMBOL: _SYMBOL, VENUE: _VENUE, EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available,
            "open": close, "high": high, "low": low, "close": close,
            "volume": 1000.0, "trades": 50}


def test_a_round_trip_actually_closes_end_to_end(tmp_path):
    """The whole point. Entry, move, exit, flat - through the real engine.

    Before this wiring the engine had produced 12,983 fills and zero closed round
    trips, because nothing ever sold.
    """
    rows = []
    clock = 100
    # Quiet bars to build an ATR, then a collapse that takes out the stop.
    for i in range(ATR_PERIOD + 4):
        rows.append(_row(clock, clock, 100.0, 101.0, 99.0))
        clock += 100
    for close in (95.0, 88.0, 80.0):
        rows.append(_row(clock, clock, close, close + 1, close - 1))
        clock += 100

    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64",
         AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", DATASET, frame, "snap1")

    paper_root = tmp_path / "paper"
    engine = ForwardEngine(
        replay=MarketReplay(reader=ClockGatedReader(
            tmp_path / "store", DATASET,
            # The engine refuses an unguarded reader: a holdout that has been
            # seen cannot be un-seen.
            custodian=HoldoutCustodian(tmp_path / "holdout",
                                       holdout_start_ns=10**18,
                                       holdout_end_ns=2 * 10**18))),
        broker=PaperBroker(
            participation=Participation(fraction=Decimal("0.5"),
                                        calibrated=False),
            fees_by_venue={_VENUE: BINANCE}),
        wal=OrderIntentWal(paper_root / "wal"),
        journal=ForwardJournal(paper_root),
        strategy=PlumbingMomentum(quantity=Decimal("1")),
        kill_root=tmp_path / "ops",
        exits=ExitManager("plumbing-momentum"))

    for poll in range(1, 40):
        engine.poll_once(now_ns=poll * 100)

    assert engine.counts.fills > 0, "the engine has to trade before it can close"
    assert engine.counts.exits_proposed > 0, (
        "the collapse takes out the hard stop - an exit must have been proposed")
    assert engine.counts.exits_submitted > 0, (
        "and it must have reached the broker; a proposal nobody submits closes "
        "nothing")

    sells = [f for f in _journalled_sides(paper_root)]
    assert SELL in sells, (
        f"no SELL was ever journalled, so nothing closed - sides seen: "
        f"{sorted(set(sells))}")


def _journalled_sides(paper_root):
    import json
    sides = []
    for path in sorted(paper_root.glob("fills-*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sides.append(json.loads(line)["side"])
    return sides


def test_the_engine_counts_exits_apart_from_entries():
    """Lumped together, a run that stopped exiting entirely would read as one
    that submitted slightly fewer orders."""
    from paper.forward_engine import EngineCounts

    counts = EngineCounts()

    assert hasattr(counts, "exits_submitted")
    assert hasattr(counts, "exits_proposed")
    assert counts.exits_submitted == 0
