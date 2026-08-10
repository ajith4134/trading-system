"""The microprice is the depth-weighted price between the touch prices.

Weighting each side's TOUCH price by the OPPOSITE side's size is the load-
bearing arithmetic: get the weighting backwards and the result still looks
like a price (it sits between bid and ask) while leaning the wrong way, which
is a defect a spot check would never catch. Every test here either pins the
direction with a hand-computed value or attacks a book shape that must never
produce a substituted number.
"""
import json
from decimal import Decimal

import pandas as pd

from features.microprice import compute_microprice
from store.parquet_partition import append_partition

_SEC = 1_000_000_000


def _levels_json(levels):
    return json.dumps([[str(price), str(size)] for price, size in levels])


def _book_row(venue, symbol, bids, asks, event_ns, avail_ns=None):
    return {
        "venue": venue, "symbol": symbol,
        "bids": _levels_json(bids),
        "asks": _levels_json(asks),
        "last_update_id": event_ns,
        "event_time_ns": event_ns,
        "ingestion_time_ns": event_ns,
        "availability_time_ns": avail_ns if avail_ns is not None else event_ns,
    }


def _series(venue, symbol, bids, asks, n=10, start_ns=1_000 * _SEC, step=_SEC):
    return [_book_row(venue, symbol, bids, asks, start_ns + i * step)
            for i in range(n)]


def _write(tmp_path, rows, snapshot="microprice-test"):
    append_partition(tmp_path, "book", pd.DataFrame(rows), snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["event_time_ns"] for r in rows) + 1


# --- the direction the weighting must lean ---------------------------------

def test_bid_size_far_greater_than_ask_size_pulls_microprice_toward_the_ask(tmp_path):
    """bid=100 size=99, ask=101 size=1. Weighting each side by the OPPOSITE
    size gives (100*1 + 101*99) / 100 = 100.99 - a cent off the ask, not
    halfway to it. Getting the weighting backwards (each side weighted by its
    OWN size) would instead produce 100.01, a hair off the BID - a number
    that still looks like a plausible price and leans exactly the wrong way.
    """
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(100), Decimal(99))],
                   asks=[(Decimal(101), Decimal(1))])
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))

    assert len(table.rows) == 1
    got = table.rows.iloc[0].microprice
    assert got == Decimal("100.99"), f"hand-computed microprice is 100.99, got {got}"
    assert Decimal(100) < got < Decimal(101)
    assert (Decimal(101) - got) < (got - Decimal(100)), (
        "heavier bid size must land the microprice nearer the ASK, not the BID")


# --- depth-weighted vs level-1-only -----------------------------------------

def test_depth_weighted_microprice_differs_from_the_level1_spoofable_variant(tmp_path):
    """Two levels a side. Level 1 alone is symmetric (size 1 vs 1, so the
    level-1-only number is the plain mid, 101). The book beyond level 1 is
    not symmetric - the ask side carries far more resting size (1 + 29 = 30)
    than the bid side (1 + 9 = 10) - and only the multi-level, depth-weighted
    number sees that.

    Hand-computed:
      level-1-only: (100*1 + 102*1) / (1+1)          = 202/2   = 101
      depth-weighted (2 levels): (100*30 + 102*10)/40 = 4020/40 = 100.5

    A book that dumps size two levels deep on the ask side is more sell
    pressure than a level-1-only snapshot can see, and the depth-weighted
    number is pulled toward the bid to reflect it - the level-1 number is
    blind to it entirely.
    """
    bids = [(Decimal(100), Decimal(1)), (Decimal(98), Decimal(9))]
    asks = [(Decimal(102), Decimal(1)), (Decimal(104), Decimal(29))]
    rows = _series("binance", "BTCUSDT", bids=bids, asks=asks)
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))

    got = table.rows.iloc[0]
    assert got.microprice == Decimal("100.5")
    assert got.microprice_level1_spoofable == Decimal("101")
    assert got.microprice != got.microprice_level1_spoofable


def test_a_thin_deep_book_uses_no_more_than_the_disclosed_level_count(tmp_path):
    """The archive holds 20 levels a side; a level six deep and beyond must
    not move the number at all, because the module discloses that it uses a
    bounded number of levels rather than the whole visible book. Level 3 here
    carries an enormous size that would swing the price hard if it were being
    read - the assertion that it is NOT swung is the test.
    """
    from features.microprice import LEVELS_USED
    assert LEVELS_USED < 20, "must be fewer than the archive's 20 levels"

    within_window = [(Decimal(100) - i, Decimal(1)) for i in range(LEVELS_USED)]
    beyond_window = [(Decimal(100) - LEVELS_USED, Decimal(1_000_000))]
    bids = within_window + beyond_window
    asks = [(Decimal(101) + i, Decimal(1)) for i in range(LEVELS_USED)]

    rows_windowed = _series("binance", "BTCUSDT", bids=within_window, asks=asks)
    rows_deep = _series("binance", "BTCUSDT", bids=bids, asks=asks)

    price_windowed = compute_microprice(
        _write(tmp_path / "a", rows_windowed), _as_of(rows_windowed)).rows.iloc[0].microprice
    price_deep = compute_microprice(
        _write(tmp_path / "b", rows_deep), _as_of(rows_deep)).rows.iloc[0].microprice

    assert price_windowed == price_deep, (
        "a level beyond the disclosed window must not change the microprice")


# --- refusals, never a substituted number -----------------------------------

def test_a_crossed_book_is_refused_by_its_own_named_reason(tmp_path):
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(101), Decimal(10))],
                   asks=[(Decimal(100), Decimal(10))])
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["crossed_or_locked_book"] == 1


def test_a_locked_book_bid_equals_ask_is_refused_the_same_way(tmp_path):
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(100), Decimal(10))],
                   asks=[(Decimal(100), Decimal(10))])
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["crossed_or_locked_book"] == 1


def test_an_empty_book_is_refused_not_defaulted_to_a_stale_mid(tmp_path):
    rows = _series("binance", "BTCUSDT", bids=[], asks=[(Decimal(101), Decimal(1))])
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["no_bid"] == 1

    rows2 = _series("bybit", "BTCUSDT", bids=[(Decimal(100), Decimal(1))], asks=[])
    store2 = _write(tmp_path / "two", rows2)
    table2 = compute_microprice(store2, _as_of(rows2))
    assert len(table2.rows) == 0
    assert table2.refused["no_ask"] == 1


def test_zero_size_at_the_touch_is_refused_not_divided_by(tmp_path):
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(100), Decimal(0))],
                   asks=[(Decimal(101), Decimal(10))])
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["zero_size"] == 1


def test_an_unparseable_book_is_counted_not_guessed(tmp_path):
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(100), Decimal(10))],
                   asks=[(Decimal(101), Decimal(10))])
    for row in rows:
        row["bids"] = "not json"
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["unparseable_book"] == 1


def test_an_empty_store_returns_nothing_and_claims_nothing(tmp_path):
    table = compute_microprice(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0


# --- FE-001: staleness is mandatory, on every row ---------------------------

def test_every_row_carries_a_staleness_stamp(tmp_path):
    rows = _series("binance", "BTCUSDT",
                   bids=[(Decimal(100), Decimal(10))],
                   asks=[(Decimal(101), Decimal(10))], n=25)
    store = _write(tmp_path, rows)
    table = compute_microprice(store, _as_of(rows))

    got = table.rows.iloc[0]
    for column in ("as_of_ns", "input_event_ns", "age_ns", "routine_gap_ns",
                   "freshness"):
        assert column in table.rows.columns, f"missing FE-001 column {column}"
    assert got.freshness in ("FRESH", "STALE", "UNKNOWN_CADENCE")
    assert got.input_event_ns == max(r["event_time_ns"] for r in rows)


def test_the_clock_gates_the_microprice(tmp_path):
    early = _series("binance", "BTCUSDT",
                    bids=[(Decimal(100), Decimal(10))],
                    asks=[(Decimal(101), Decimal(10))], n=5, start_ns=1_000 * _SEC)
    late = _series("binance", "BTCUSDT",
                   bids=[(Decimal(500), Decimal(10))],
                   asks=[(Decimal(501), Decimal(10))], n=5, start_ns=1_010 * _SEC)
    store = _write(tmp_path, early + late)
    table = compute_microprice(store, 1_005 * _SEC)
    assert len(table.rows) == 1
    assert table.rows.iloc[0].microprice == Decimal("100.5"), (
        "the 500-level book is not knowable at this clock")
