"""The forward engine and its journal — Phase J's supervised process.

Two silences a status board has to be able to tell apart:

    an engine running, finding nothing to trade
    an engine that stopped three days ago

Both produce no fills, so fills cannot distinguish them. The heartbeat is written
on **every** poll and carries the clock it was written at, and the wall grades on
that age. This box has already demonstrated the failure being defended against: it
was off from 2026-08-10 to 2026-08-15 while the board went on looking healthy.

The other thing under test is that the engine claims nothing it was not told to
claim. The only signal that exists makes no edge claim, and that flag has to
reach every fill row — not a header, which is the first thing separated from its
data.
"""
from decimal import Decimal

import pandas as pd
import pytest

from execution.order_intent_wal import OrderIntentWal
from paper.fill_model import Participation
from paper.forward_engine import (
    ForwardEngine, KillSwitchEngaged, PlumbingMomentum, build_engine,
)
from paper.forward_journal import ForwardJournal, count_fills, read_heartbeat
from paper.market_replay import MarketReplay
from paper.paper_broker import PaperBroker
from paper.position_book import Accounting
from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)
from validation.holdout_custodian import HoldoutCustodian
from cost.fee_schedule import FeeRate

DATASET = "bars_60000000000ns"
BINANCE = FeeRate(maker_bps=Decimal("2.0"), taker_bps=Decimal("5.0"))


def _bar(event, available, close, *, high=None, low=None, volume=1000.0):
    return {SYMBOL: "BTCUSDT", VENUE: "binance", EVENT_TIME: event,
            INGESTION_TIME: available, AVAILABILITY_TIME: available,
            "open": close, "high": high if high is not None else close + 1,
            "low": low if low is not None else close - 1,
            "close": close, "volume": volume, "trades": 50}


def _write(root, rows):
    frame = pd.DataFrame(rows).astype(
        {EVENT_TIME: "int64", INGESTION_TIME: "int64",
         AVAILABILITY_TIME: "int64"})
    append_partition(root, DATASET, frame, "snap1")


def an_engine(tmp_path, *, quantity="1", participation="0.5"):
    custodian = HoldoutCustodian(tmp_path / "holdout",
                                 holdout_start_ns=10**18,
                                 holdout_end_ns=2 * 10**18)
    reader = ClockGatedReader(tmp_path / "store", DATASET, custodian=custodian)
    paper_root = tmp_path / "paper"
    return ForwardEngine(
        replay=MarketReplay(reader=reader),
        broker=PaperBroker(participation=Participation(
            fraction=Decimal(participation), calibrated=False),
            fees_by_venue={"binance": BINANCE}),
        wal=OrderIntentWal(paper_root / "wal"),
        journal=ForwardJournal(paper_root),
        strategy=PlumbingMomentum(quantity=Decimal(quantity)),
        kill_root=tmp_path / "ops")


# --- the heartbeat is the whole point ---------------------------------------

def test_a_journal_that_never_ran_reads_as_absent_not_as_quiet(tmp_path):
    # None, not a zeroed Heartbeat: a zeroed one reads as "ran, did nothing",
    # and the wall must be able to say NOT BUILT instead of OK-but-quiet.
    assert read_heartbeat(tmp_path) is None


def test_a_poll_that_found_nothing_still_writes_a_heartbeat(tmp_path):
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=5_000)
    beat = read_heartbeat(tmp_path / "paper")
    assert beat is not None and beat.events_fed == 0


def test_the_heartbeat_carries_the_clock_it_was_written_at(tmp_path):
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=7_777)
    beat = read_heartbeat(tmp_path / "paper")
    assert beat.written_at_ns == 7_777
    assert beat.age_ns(now_ns=9_777) == 2_000


def test_an_unparseable_heartbeat_reads_as_absent_rather_than_as_defaults(
        tmp_path):
    (tmp_path / "heartbeat.json").write_text("{not json", encoding="utf-8")
    assert read_heartbeat(tmp_path) is None


def test_a_heartbeat_missing_a_field_is_not_filled_in_with_a_green_default(
        tmp_path):
    (tmp_path / "heartbeat.json").write_text('{"written_at_ns": 1}',
                                             encoding="utf-8")
    assert read_heartbeat(tmp_path) is None


# --- the kill switch is asked before anything is journalled -----------------

def test_a_kill_in_force_refuses_the_poll(tmp_path):
    (tmp_path / "ops").mkdir(parents=True)
    from ops.watchdog import KILL_FILE
    (tmp_path / "ops" / KILL_FILE).write_text('{"reason": "test"}',
                                              encoding="utf-8")
    with pytest.raises(KillSwitchEngaged, match="test"):
        an_engine(tmp_path).poll_once(now_ns=5_000)


def test_a_killed_run_leaves_no_record_implying_it_traded(tmp_path):
    (tmp_path / "ops").mkdir(parents=True)
    from ops.watchdog import KILL_FILE
    (tmp_path / "ops" / KILL_FILE).write_text('{"reason": "test"}',
                                              encoding="utf-8")
    with pytest.raises(KillSwitchEngaged):
        an_engine(tmp_path).poll_once(now_ns=5_000)
    assert read_heartbeat(tmp_path / "paper") is None


# --- orders flow, and fills reach the journal -------------------------------

def test_the_first_bar_produces_no_order_because_there_is_no_previous_close(
        tmp_path):
    _write(tmp_path / "store", [_bar(100, 200, 100.0)])
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=200)
    assert engine.counts.orders_submitted == 0
    assert engine.counts.events_fed == 1


def test_a_second_bar_rests_an_order_at_the_previous_close(tmp_path):
    _write(tmp_path / "store", [_bar(100, 200, 100.0), _bar(160, 260, 99.0)])
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=200)
    engine.poll_once(now_ns=260)
    assert engine.counts.orders_submitted == 1


def test_an_order_cannot_fill_on_the_very_print_that_created_it(tmp_path):
    """It was not in the book when that trade happened. Awarding it a fill is
    the cheapest possible way to invent a strategy that front-runs its own data."""
    _write(tmp_path / "store", [_bar(100, 200, 100.0), _bar(160, 260, 90.0)])
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=200)
    engine.poll_once(now_ns=260)
    assert engine.counts.fills == 0


def test_a_later_print_through_the_resting_order_fills_it(tmp_path):
    _write(tmp_path / "store", [_bar(100, 200, 100.0),
                                _bar(160, 260, 99.0),
                                _bar(220, 320, 90.0)])
    engine = an_engine(tmp_path)
    for clock in (200, 260, 320):
        engine.poll_once(now_ns=clock)
    assert engine.counts.fills >= 1


def test_every_fill_reaches_the_journal_on_disk(tmp_path):
    _write(tmp_path / "store", [_bar(100, 200, 100.0),
                                _bar(160, 260, 99.0),
                                _bar(220, 320, 90.0)])
    engine = an_engine(tmp_path)
    for clock in (200, 260, 320):
        engine.poll_once(now_ns=clock)
    assert count_fills(tmp_path / "paper") == engine.counts.fills


def test_a_journalled_fill_carries_both_accountings_never_one(tmp_path):
    import json
    _write(tmp_path / "store", [_bar(100, 200, 100.0),
                                _bar(160, 260, 99.0),
                                _bar(220, 320, 90.0)])
    engine = an_engine(tmp_path)
    for clock in (200, 260, 320):
        engine.poll_once(now_ns=clock)
    (path,) = list((tmp_path / "paper").glob("fills-*.ndjson"))
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "optimistic_price" in row and "pessimistic_price" in row
    assert row["optimistic_liquidity"] == "maker"
    assert row["pessimistic_liquidity"] == "taker"


def test_no_edge_claim_rides_on_the_fill_row_not_on_a_header(tmp_path):
    import json
    _write(tmp_path / "store", [_bar(100, 200, 100.0),
                                _bar(160, 260, 99.0),
                                _bar(220, 320, 90.0)])
    engine = an_engine(tmp_path)
    for clock in (200, 260, 320):
        engine.poll_once(now_ns=clock)
    (path,) = list((tmp_path / "paper").glob("fills-*.ndjson"))
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["makes_edge_claim"] is False


def test_the_plumbing_signal_says_out_loud_that_it_claims_no_edge():
    assert PlumbingMomentum.makes_edge_claim is False


# --- forward operation starts now, not at the beginning of the archive ------

def test_priming_marks_the_archive_as_seen_without_trading_any_of_it(tmp_path):
    """Found by running it, 2026-08-15: on BTCUSDT alone a fresh journal's first
    poll fed 3,494 archived bars and produced 1,074 fills against prices days
    old — every one of which would have entered the forward journal as a forward
    result."""
    _write(tmp_path / "store", [_bar(100, 200, 100.0), _bar(160, 260, 99.0),
                                _bar(220, 320, 90.0)])
    engine = an_engine(tmp_path)
    assert engine.prime(now_ns=400) == 3
    assert engine.counts.fills == 0
    assert engine.counts.orders_submitted == 0


def test_a_primed_engine_trades_only_what_arrives_afterwards(tmp_path):
    _write(tmp_path / "store", [_bar(100, 200, 100.0), _bar(160, 260, 99.0)])
    engine = an_engine(tmp_path)
    engine.prime(now_ns=300)
    engine.poll_once(now_ns=400)
    assert engine.counts.events_fed == 0


def test_priming_still_says_so_in_the_heartbeat(tmp_path):
    engine = an_engine(tmp_path)
    engine.prime(now_ns=400)
    beat = read_heartbeat(tmp_path / "paper")
    assert beat is not None and "primed" in beat.detail


def test_priming_refuses_under_a_kill_like_every_other_path(tmp_path):
    (tmp_path / "ops").mkdir(parents=True)
    from ops.watchdog import KILL_FILE
    (tmp_path / "ops" / KILL_FILE).write_text('{"reason": "test"}',
                                              encoding="utf-8")
    with pytest.raises(KillSwitchEngaged):
        an_engine(tmp_path).prime(now_ns=400)


def test_a_duplicate_intent_is_counted_apart_from_a_broker_rejection(tmp_path):
    """Two decisions in the same nanosecond on the same instrument, size and
    limit hash to one client order id — the deterministic id working, not the
    broker refusing. Lumped together, a cold-start burst reads as a broker that
    rejected everything."""
    _write(tmp_path / "store", [_bar(100, 200, 100.0), _bar(160, 200, 100.0),
                                _bar(220, 200, 100.0)])
    engine = an_engine(tmp_path)
    engine.poll_once(now_ns=200)
    assert engine.counts.orders_duplicate >= 1
    assert engine.counts.orders_rejected == 0


# --- the two books stay separate all the way to the journal -----------------

def test_the_broker_keeps_a_book_per_accounting(tmp_path):
    engine = an_engine(tmp_path)
    assert engine._broker.optimistic.accounting is Accounting.OPTIMISTIC
    assert engine._broker.pessimistic.accounting is Accounting.PESSIMISTIC


# --- wiring -----------------------------------------------------------------

def test_an_unknown_strategy_is_refused_and_the_message_names_what_exists(
        tmp_path):
    with pytest.raises(SystemExit, match="plumbing-momentum"):
        build_engine(store_root=tmp_path / "store", capture_root=tmp_path,
                     strategy_name="the-good-one",
                     participation=Participation(fraction=Decimal("0.1"),
                                                 calibrated=False),
                     quantity=Decimal("1"), holdout_start_ns=10**18,
                     holdout_end_ns=2 * 10**18)


def test_the_wired_engine_journals_under_capture_paper_forward(tmp_path):
    engine = build_engine(
        store_root=tmp_path / "store", capture_root=tmp_path,
        strategy_name="plumbing-momentum",
        participation=Participation(fraction=Decimal("0.1"), calibrated=False),
        quantity=Decimal("1"), holdout_start_ns=10**18,
        holdout_end_ns=2 * 10**18)
    engine.poll_once(now_ns=1_000)
    # The exact path the build-progress board probes for Phase J.
    assert (tmp_path / "paper" / "forward" / "heartbeat.json").is_file()
