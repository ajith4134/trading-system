"""A consolidated price that a thin or frozen venue can move is the Mango bug.

Every test here is an attack or a degradation, because the honest cases are
not what this module exists for.
"""
import json

import pandas as pd

from features.consolidated_price import consolidate_prices
from store.parquet_partition import append_partition

_SEC = 1_000_000_000


def _book_row(venue, symbol, bid, ask, size, event_ns, avail_ns=None):
    return {
        "venue": venue, "symbol": symbol,
        "bids": json.dumps([[f"{bid}", f"{size}"]]),
        "asks": json.dumps([[f"{ask}", f"{size}"]]),
        "last_update_id": event_ns,
        "event_time_ns": event_ns,
        "ingestion_time_ns": event_ns,
        "availability_time_ns": avail_ns if avail_ns is not None else event_ns,
    }


def _series(venue, symbol, bid, ask, size, n=10, start_ns=1_000 * _SEC, step=_SEC):
    return [_book_row(venue, symbol, bid, ask, size, start_ns + i * step)
            for i in range(n)]


def _write(tmp_path, rows, snapshot="consolidated-test"):
    append_partition(tmp_path, "book", pd.DataFrame(rows), snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["event_time_ns"] for r in rows) + 1


# --- the attack this module exists for ------------------------------------

def test_a_thin_venue_quoting_a_moved_price_barely_moves_the_result(tmp_path):
    """The Mango shape: a small venue is pushed far off and an equal-weighted
    mean follows it. Deep venue at 100 with 1,000,000 notional; thin venue at
    200 with 100. A mean says 150."""
    deep = _series("binance", "BTCUSDT", 99.99, 100.01, 10_000.0)
    thin = _series("binance-spot", "BTCUSDT", 199.99, 200.01, 0.5)
    store = _write(tmp_path, deep + thin)
    table = consolidate_prices(store, _as_of(deep + thin))
    price = table.rows.iloc[0].price
    assert price < 101.0, f"an equal-weighted mean would be ~150, got {price}"
    assert table.rows.iloc[0].venues_used == 2, "the thin venue is outvoted, not dropped"


def test_the_disagreement_is_reported_not_smoothed_away(tmp_path):
    deep = _series("binance", "BTCUSDT", 99.99, 100.01, 10_000.0)
    thin = _series("binance-spot", "BTCUSDT", 199.99, 200.01, 0.5)
    store = _write(tmp_path, deep + thin)
    table = consolidate_prices(store, _as_of(deep + thin))
    assert table.rows.iloc[0].disagreement_bps > 5_000, (
        "a 100% gap between inputs must survive into the output")


# --- staleness -------------------------------------------------------------

def test_a_frozen_venue_is_excluded_even_though_its_depth_is_intact(tmp_path):
    """A frozen book keeps its depth and therefore its weight - which makes
    staleness more dangerous than thinness, not less."""
    # Both venues tick once a second and both are deep. The only difference
    # is that one stopped six seconds ago - past three times its own cadence -
    # while the other is a second old.
    live = _series("binance", "BTCUSDT", 99.99, 100.01, 100.0, n=16)
    frozen = _series("binance-spot", "BTCUSDT", 89.99, 90.01, 100_000.0, n=10)
    store = _write(tmp_path, live + frozen)
    as_of = max(r["event_time_ns"] for r in live) + _SEC
    table = consolidate_prices(store, as_of)
    got = table.rows.iloc[0]
    assert got.venues_used == 1
    assert got.venues == "binance"
    assert table.excluded["stale"] == 1
    assert abs(got.price - 100.0) < 0.1, (
        "the frozen venue's 90.0 must not reach the price at all")


def test_a_venue_with_one_snapshot_is_not_called_stale_on_no_evidence(tmp_path):
    single = [_book_row("bybit", "BTCUSDT", 99.99, 100.01, 100.0, 1_000 * _SEC)]
    store = _write(tmp_path, single)
    table = consolidate_prices(store, 1_000 * _SEC + 3600 * _SEC)
    assert table.excluded["stale"] == 0
    assert len(table.rows) == 1


# --- honesty about what backed the number ---------------------------------

def test_a_single_venue_price_says_it_is_one_venue(tmp_path):
    rows = _series("binance", "BTCUSDT", 99.99, 100.01, 100.0)
    store = _write(tmp_path, rows)
    table = consolidate_prices(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.venues_used == 1
    assert got.disagreement_bps == 0.0


def test_an_unparseable_book_is_counted_not_guessed(tmp_path):
    good = _series("binance", "BTCUSDT", 99.99, 100.01, 100.0)
    broken = _series("binance-spot", "BTCUSDT", 99.99, 100.01, 100.0)
    for row in broken:
        row["bids"] = "not json"
    store = _write(tmp_path, good + broken)
    table = consolidate_prices(store, _as_of(good + broken))
    assert table.excluded["unparseable"] == 1
    assert table.rows.iloc[0].venues_used == 1


def test_the_clock_gates_the_price(tmp_path):
    early = _series("binance", "BTCUSDT", 99.99, 100.01, 100.0, n=5,
                    start_ns=1_000 * _SEC)
    late = _series("binance", "BTCUSDT", 499.99, 500.01, 100.0, n=5,
                   start_ns=1_010 * _SEC)
    store = _write(tmp_path, early + late)
    table = consolidate_prices(store, 1_005 * _SEC)
    assert abs(table.rows.iloc[0].price - 100.0) < 0.1, (
        "the 500 book is not knowable at this clock")


def test_an_empty_store_returns_nothing_and_claims_nothing(tmp_path):
    table = consolidate_prices(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.excluded.values()) == 0
