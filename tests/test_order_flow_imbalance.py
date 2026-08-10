"""Level-1 OBI is spoofable for the price of one cancelled order; a depth-
weighted measure is not. Every test here either proves the weighting changes
the answer, or proves a bad row is refused rather than pricing an imbalance
that was never really there.
"""
import json
from decimal import Decimal

import pandas as pd

from features.order_flow_imbalance import compute_order_flow_imbalance
from store.parquet_partition import append_partition

_SEC = 1_000_000_000


def _book_row(venue, symbol, bid_levels, ask_levels, event_ns, avail_ns=None):
    """`bid_levels`/`ask_levels` are (price, size) pairs, best level first -
    matching what `store.book_snapshots.build_book_frame` actually writes.
    """
    return {
        "venue": venue, "symbol": symbol,
        "bids": json.dumps([[str(p), str(s)] for p, s in bid_levels]),
        "asks": json.dumps([[str(p), str(s)] for p, s in ask_levels]),
        "last_update_id": event_ns,
        "event_time_ns": event_ns,
        "ingestion_time_ns": event_ns,
        "availability_time_ns": avail_ns if avail_ns is not None else event_ns,
    }


def _write(tmp_path, rows, snapshot="ofi-test"):
    append_partition(tmp_path, "book", pd.DataFrame(rows), snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["event_time_ns"] for r in rows) + 1


# --- the hand-computed number, and depth changing the answer --------------

def test_depth_weighting_changes_the_answer_from_level1(tmp_path):
    """Level 1 alone says one thing; the deeper, larger levels say another.
    A depth-weighted measure that agreed with level 1 here would just be
    level 1 wearing a longer docstring.

    Book (bid, ask) sizes by level:
      level 1: bid=120, ask=80   -> level-1 imbalance = 40/200   = 0.20
      level 2: bid=60,  ask=20   (weight 1/2, harmonic by rank)

    Hand-computed depth-weighted imbalance (harmonic weights 1, 1/2):
      numerator   = 1*(120-80) + 0.5*(60-20) = 40 + 20 = 60
      denominator = 1*(120+80) + 0.5*(60+20) = 200 + 40 = 240
      imbalance   = 60/240 = 0.25
    """
    row = _book_row(
        "binance", "BTCUSDT",
        bid_levels=[("100.00", "120"), ("99.99", "60")],
        ask_levels=[("100.01", "80"), ("100.02", "20")],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert len(table.rows) == 1
    got = table.rows.iloc[0]
    assert got.level1_imbalance_spoofable == Decimal("0.2")
    assert got.depth_weighted_imbalance == Decimal("0.25")
    assert got.depth_weighted_imbalance != got.level1_imbalance_spoofable, (
        "the whole point of the module: depth weighting must actually move "
        "the number away from the spoofable level-1 reading")
    assert got.levels_present == 2
    assert sum(table.refused.values()) == 0


def test_a_level1_spoof_is_pulled_down_by_genuine_deeper_size(tmp_path):
    """The attack this module exists for: a spoofer fakes a large imbalance at
    the touch (cheap - one order, cancellable in milliseconds) while the
    resting book below it is calm and balanced. Level 1 alone reports a
    lopsided market; harmonic weighting must pull the depth-weighted figure
    well below it once the genuinely balanced deeper levels are big enough to
    outweigh the touch (weight 1) - here levels 2-5 are 2x the spoofed size,
    so their combined weight (1/2+1/3+1/4+1/5 = 0.7833, applied to twice the
    notional) outweighs level 1's weight of 1 applied to its notional.

    Hand-computed (weights 1, 1/2, 1/3, 1/4, 1/5):
      numerator   = 1*(10000-10) + 0+0+0+0                         = 9990
      denominator = 1*(10000+10) + (1/2+1/3+1/4+1/5)*(10000+10000)
                  = 10010 + 0.783333...*20000 = 10010 + 15666.67 = 25676.67
      imbalance   = 9990 / 25676.67 ~= 0.389 - well below level 1's 0.998,
      and this module does not claim to erase a spoof, only to make faking
      it this deep and this large - real capital at real levels - the price
      of moving the number this much.
    """
    row = _book_row(
        "binance", "ETHUSDT",
        # level 1: a spoofed wall of bids vs a thin ask -> screams "buy"
        # levels 2-5: genuinely balanced, and larger in aggregate than the spoof
        bid_levels=[("100.00", "10000"), ("99.99", "10000"), ("99.98", "10000"),
                    ("99.97", "10000"), ("99.96", "10000")],
        ask_levels=[("100.01", "10"), ("100.02", "10000"), ("100.03", "10000"),
                    ("100.04", "10000"), ("100.05", "10000")],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    got = table.rows.iloc[0]
    level1 = (Decimal("10000") - Decimal("10")) / (Decimal("10000") + Decimal("10"))
    assert got.level1_imbalance_spoofable == level1
    assert level1 > Decimal("0.99"), "level 1 alone reads as almost pure buy pressure"
    assert got.depth_weighted_imbalance < Decimal("0.45"), (
        "the genuinely balanced deeper levels must pull the depth-weighted "
        f"figure well away from the spoofed level-1 reading, got {got.depth_weighted_imbalance}")
    assert got.depth_weighted_imbalance > 0, (
        "this module does not claim to erase a spoof entirely - only to raise "
        "the cost of moving the number this far")


# --- levels actually present, per row --------------------------------------

def test_levels_present_reflects_what_the_row_actually_carried(tmp_path):
    """A 2-level book and a 5-level book are not the same measurement, and
    the row has to say which one it is rather than let them share a column
    unlabelled.
    """
    thin = _book_row(
        "bybit", "XRPUSDT",
        bid_levels=[("0.50", "1000"), ("0.499", "1000")],
        ask_levels=[("0.501", "1000"), ("0.502", "1000")],
        event_ns=1_000 * _SEC,
    )
    deep = _book_row(
        "binance", "BTCUSDT",
        bid_levels=[(str(100 - i * 0.01), "100") for i in range(5)],
        ask_levels=[(str(100.01 + i * 0.01), "100") for i in range(5)],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [thin, deep])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([thin, deep]))

    by_symbol = {row.symbol: row for row in table.rows.itertuples()}
    assert by_symbol["XRPUSDT"].levels_present == 2
    assert by_symbol["BTCUSDT"].levels_present == 5


def test_levels_present_is_the_shorter_side_when_sides_are_uneven(tmp_path):
    """A book with 5 bid levels and 2 ask levels can only pair 2 levels - the
    unpaired bid levels have no counterpart to weigh against, so counting
    them as 'present' would overstate what was actually measured.
    """
    row = _book_row(
        "binance", "SOLUSDT",
        bid_levels=[("100.00", "10"), ("99.99", "10"), ("99.98", "10"),
                    ("99.97", "10"), ("99.96", "10")],
        ask_levels=[("100.01", "10"), ("100.02", "10")],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert table.rows.iloc[0].levels_present == 2


# --- refusals: missing side, zero depth, crossed book ----------------------

def test_a_one_sided_book_is_refused_not_priced_off_the_side_that_exists(tmp_path):
    row = _book_row(
        "binance", "DOGEUSDT",
        bid_levels=[("0.10", "1000")],
        ask_levels=[],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert len(table.rows) == 0
    assert table.refused["one_sided_book"] == 1


def test_zero_total_depth_is_refused_not_divided_by(tmp_path):
    row = _book_row(
        "binance", "SHIBUSDT",
        bid_levels=[("0.00001", "0"), ("0.000009", "0")],
        ask_levels=[("0.000011", "0"), ("0.000012", "0")],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert len(table.rows) == 0
    assert table.refused["zero_total_depth"] == 1


def test_a_crossed_book_is_refused_not_reported_as_an_imbalance(tmp_path):
    """Best bid at or above best ask is a broken snapshot, not a market -
    reporting an imbalance for it dresses up a data defect as a signal."""
    row = _book_row(
        "binance", "ADAUSDT",
        bid_levels=[("100.05", "100")],
        ask_levels=[("100.00", "100")],
        event_ns=1_000 * _SEC,
    )
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert len(table.rows) == 0
    assert table.refused["crossed_book"] == 1


def test_an_unparseable_book_is_counted_not_guessed(tmp_path):
    row = _book_row(
        "binance", "LTCUSDT",
        bid_levels=[("100.00", "10")], ask_levels=[("100.01", "10")],
        event_ns=1_000 * _SEC,
    )
    row["bids"] = "not json"
    store = _write(tmp_path, [row])
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of([row]))

    assert len(table.rows) == 0
    assert table.refused["unparseable"] == 1


def test_an_empty_store_returns_nothing_and_claims_nothing(tmp_path):
    table = compute_order_flow_imbalance(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0


# --- FE-001: staleness rides every row --------------------------------------

def test_every_row_carries_its_staleness_stamp(tmp_path):
    rows = [
        _book_row("binance", "BTCUSDT",
                  bid_levels=[("100.00", "10")], ask_levels=[("100.01", "10")],
                  event_ns=1_000 * _SEC + i * _SEC)
        for i in range(5)
    ]
    store = _write(tmp_path, rows)
    table = compute_order_flow_imbalance(store, as_of_ns=_as_of(rows))

    got = table.rows.iloc[0]
    for column in ("as_of_ns", "input_event_ns", "age_ns", "routine_gap_ns", "freshness"):
        assert column in table.rows.columns, f"missing FE-001 column {column}"
    assert got.freshness == "FRESH"
    assert got.age_ns == 1


def test_a_frozen_book_is_stamped_stale_rather_than_silently_priced(tmp_path):
    """The book cadence here is one snapshot per second; asking long after the
    last one must produce a STALE stamp rather than a value that looks as
    fresh as any other."""
    rows = [
        _book_row("binance", "BTCUSDT",
                  bid_levels=[("100.00", "10")], ask_levels=[("100.01", "10")],
                  event_ns=1_000 * _SEC + i * _SEC)
        for i in range(20)
    ]
    store = _write(tmp_path, rows)
    as_of = max(r["event_time_ns"] for r in rows) + 3600 * _SEC
    table = compute_order_flow_imbalance(store, as_of_ns=as_of)

    assert table.rows.iloc[0].freshness == "STALE"


# --- the clock gate, because it is the only door to this dataset ----------

def test_the_clock_gates_the_imbalance(tmp_path):
    early = _book_row("binance", "BTCUSDT",
                      bid_levels=[("100.00", "100")], ask_levels=[("100.01", "10")],
                      event_ns=1_000 * _SEC)
    late = _book_row("binance", "BTCUSDT",
                     bid_levels=[("100.00", "10")], ask_levels=[("100.01", "100")],
                     event_ns=1_010 * _SEC)
    store = _write(tmp_path, [early, late])
    table = compute_order_flow_imbalance(store, as_of_ns=1_005 * _SEC)

    assert len(table.rows) == 1
    assert table.rows.iloc[0].depth_weighted_imbalance > 0, (
        "the late, ask-heavy book is not knowable yet at this clock")
