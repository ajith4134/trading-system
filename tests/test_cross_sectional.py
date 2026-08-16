"""A rank is only meaningful against a stated universe, at a stated instant.

The failures defended here all produce a number that looks like a percentile and
is not one: a rank among six symbols read as a rank among six hundred, a delisted
instrument ranked on three-day-old prices, a symbol contributing one print and
landing in the middle of the universe with a zero return, and ties broken by
whatever the sort was stable on.
"""
import math

import pandas as pd
import pytest

from features.cross_sectional import (
    HORIZONS_NS,
    MIN_UNIVERSE_SIZE,
    compute_cross_sectional,
    midranks,
)
from features.realized_volatility import BAR_INTERVAL_NS
from store.parquet_partition import append_partition


# --- the rank itself ------------------------------------------------------

def test_the_rank_spans_zero_to_one():
    assert midranks([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]


def test_ties_share_one_position_rather_than_an_invented_order():
    """Twenty symbols with identical volume are not ranked 1st through 20th.

    Without the midrank the order comes from whatever the sort was stable on,
    which is the input order - so a symbol's rank would depend on where it
    happened to sit in the frame.
    """
    ranks = midranks([5.0, 5.0, 5.0, 9.0])

    assert ranks[:3] == [pytest.approx(1 / 3)] * 3
    assert ranks[3] == pytest.approx(1.0)


def test_the_rank_is_returned_in_the_input_s_order():
    """The function sorts internally; a caller zips it against its own list."""
    values = [7.0, 1.0, 4.0, 9.0, 2.0]
    ranks = midranks(values)

    by_value = sorted(zip(values, ranks))
    assert [rank for _value, rank in by_value] == sorted(ranks)
    assert ranks[3] == 1.0, "9.0 is the largest and keeps its position in the list"


# --- fixtures -------------------------------------------------------------

def _bars(symbol, venue, closes, start_ns=0, volume=1.0):
    return [{
        "venue": venue, "symbol": symbol,
        "open": close, "high": close, "low": close, "close": close,
        "volume": volume, "trades": 10,
        "event_time_ns": start_ns + i * BAR_INTERVAL_NS,
        "ingestion_time_ns": start_ns + i * BAR_INTERVAL_NS,
        "availability_time_ns": start_ns + i * BAR_INTERVAL_NS,
    } for i, close in enumerate(closes)]


def _universe(venue, n, *, drift, start_ns=0, bars=30, volume=1.0):
    """`n` symbols, each drifting by a different amount over `bars` minutes."""
    rows = []
    for i in range(n):
        step = drift(i)
        closes = [100.0 * math.exp(step * b) for b in range(bars)]
        rows.extend(_bars(f"SYM{i:03d}", venue, closes, start_ns=start_ns,
                          volume=volume))
    return rows


def _write(tmp_path, rows, snapshot="xsec-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


# --- end to end -----------------------------------------------------------

def test_the_strongest_mover_tops_its_venue_s_universe(tmp_path):
    n = MIN_UNIVERSE_SIZE + 5
    rows = _universe("binance", n, drift=lambda i: (i + 1) * 1e-4)
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    assert len(hourly) == n, table.refused
    top = hourly.loc[hourly["return_rank"].idxmax()]
    assert top["symbol"] == f"SYM{n - 1:03d}"
    assert top["universe_size"] == n
    assert top["age_ns"] is not None                 # FE-001


def test_the_universe_size_rides_every_row(tmp_path):
    """A rank among four and a rank among four hundred print identically."""
    n = MIN_UNIVERSE_SIZE + 3
    rows = _universe("binance", n, drift=lambda i: i * 1e-4)
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    assert set(table.rows["universe_size"]) == {n}


def test_a_universe_below_the_floor_is_refused_whole(tmp_path):
    """Six symbols is a small sample with a percentile's name on it, and the
    top of six reads exactly like the top percentile of six hundred."""
    rows = _universe("binance", MIN_UNIVERSE_SIZE - 4, drift=lambda i: i * 1e-4)
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    # Counted per symbol lost, not once per venue - the tally has to reflect
    # what was dropped.
    assert table.refused["universe_too_small"] == (
        (MIN_UNIVERSE_SIZE - 4) * len(HORIZONS_NS))


def test_venues_are_ranked_as_separate_universes(tmp_path):
    """`BTCUSDT` on Binance and `BTC` on Hyperliquid are the same asset; pooling
    them makes the cross-section partly a measurement of the venues."""
    binance = _universe("binance", MIN_UNIVERSE_SIZE, drift=lambda i: i * 1e-4)
    # Every hyperliquid symbol moves more than every binance symbol. Pooled,
    # the binance top would not be a top; separated, both universes have one.
    hyperliquid = _universe("hyperliquid", MIN_UNIVERSE_SIZE,
                            drift=lambda i: 1e-2 + i * 1e-4)
    rows = binance + hyperliquid
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    for venue in ("binance", "hyperliquid"):
        venue_rows = hourly[hourly["venue"] == venue]
        assert venue_rows["universe_size"].max() == MIN_UNIVERSE_SIZE
        assert venue_rows["return_rank"].max() == pytest.approx(1.0)
        assert venue_rows["return_rank"].min() == pytest.approx(0.0)


def test_a_symbol_that_stopped_printing_leaves_the_cross_section(tmp_path):
    """The survivorship half the clock gate does NOT close.

    A delisted symbol's rows stay visible forever, so nothing stops it entering
    today's cross-section on stale prices. What stops it is needing a close at
    both ends of the window - which is a consequence of the return's definition
    rather than a filter bolted on beside the reader.
    """
    live = _universe("binance", MIN_UNIVERSE_SIZE, drift=lambda i: i * 1e-4,
                     start_ns=10 * HORIZONS_NS["24h"])
    dead = _bars("DEADCOIN", "binance", [100.0, 101.0], start_ns=0)
    store = _write(tmp_path, live + dead)

    table = compute_cross_sectional(store, _as_of(live))
    hourly = table.rows[table.rows["horizon"] == "1h"]

    assert "DEADCOIN" not in set(hourly["symbol"])
    assert hourly["universe_size"].iloc[0] == MIN_UNIVERSE_SIZE


def test_a_symbol_with_one_print_in_the_window_is_dropped_not_zeroed(tmp_path):
    """A single close is not a return. Contributing zero would rank it in the
    middle of the universe, which is the flattering direction for anything
    selecting on the extremes."""
    rows = _universe("binance", MIN_UNIVERSE_SIZE, drift=lambda i: (i + 1) * 1e-4)
    as_of = _as_of(rows)
    # One bar, landing inside the 1h window and nowhere near a second.
    rows += _bars("LATECOIN", "binance", [100.0],
                  start_ns=as_of - 2 * BAR_INTERVAL_NS)
    table = compute_cross_sectional(_write(tmp_path, rows), as_of + 1)

    hourly = table.rows[table.rows["horizon"] == "1h"]
    assert "LATECOIN" not in set(hourly["symbol"])
    assert table.refused["incomplete_window"] >= 1


def test_a_zero_close_drops_the_symbol_not_the_universe(tmp_path):
    """The placeholder-price defect. Unlike the single-instrument features, one
    bad symbol must not refuse the whole venue - the other 24 are fine and a
    cross-section is exactly the thing that survives losing one member."""
    rows = _universe("binance", MIN_UNIVERSE_SIZE + 1, drift=lambda i: i * 1e-4)
    for row in rows:
        if row["symbol"] == "SYM000":
            row["close"] = 0.0
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    assert "SYM000" not in set(hourly["symbol"])
    assert len(hourly) == MIN_UNIVERSE_SIZE
    assert table.refused["non_positive_close"] >= 1


def test_volume_is_ranked_over_the_same_window(tmp_path):
    n = MIN_UNIVERSE_SIZE + 2
    rows = []
    for i in range(n):
        closes = [100.0 * math.exp(1e-4 * b) for b in range(30)]
        rows.extend(_bars(f"SYM{i:03d}", "binance", closes,
                          volume=float(i + 1)))
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    loudest = hourly.loc[hourly["volume_rank"].idxmax()]
    assert loudest["symbol"] == f"SYM{n - 1:03d}"
    assert loudest["window_volume"] == pytest.approx(30.0 * n)


def test_both_horizons_are_reported_for_the_same_universe(tmp_path):
    rows = _universe("binance", MIN_UNIVERSE_SIZE, drift=lambda i: i * 1e-4)
    table = compute_cross_sectional(_write(tmp_path, rows), _as_of(rows))

    assert set(table.rows["horizon"]) == set(HORIZONS_NS)


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    table = compute_cross_sectional(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}
