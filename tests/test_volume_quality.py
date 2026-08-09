"""Reported volume is corroborated against price action, or discounted.

The module claims an upper bound on uncorroborated volume, never a wash-trading
verdict, and these tests hold it to exactly that claim.
"""
import numpy as np
import pandas as pd

from features.volume_quality import measure_volume_quality
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000


def _bar(symbol, o, h, l, c, volume, trades, i, venue="binance-spot"):
    return {
        "venue": venue, "symbol": symbol,
        "open": o, "high": h, "low": l, "close": c,
        "volume": volume, "trades": trades,
        "event_time_ns": 10**12 + i * _BAR_NS,
        "ingestion_time_ns": 10**12 + i * _BAR_NS,
        "availability_time_ns": 10**12 + i * _BAR_NS,
    }


def _write(tmp_path, rows, snapshot="volume-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def _moving(symbol, n=200, volume=10.0, trades=50, venue="binance-spot", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = 100.0 + rng.normal(0, 0.5)
        rows.append(_bar(symbol, base, base + 0.2, base - 0.2, base,
                         volume, trades, i, venue))
    return rows


def _frozen(symbol, n=200, volume=10.0, trades=50, venue="binance-spot"):
    return [_bar(symbol, 100.0, 100.0, 100.0, 100.0, volume, trades, i, venue)
            for i in range(n)]


# --- the measurement ------------------------------------------------------

def test_volume_that_moved_price_is_not_discounted(tmp_path):
    rows = _moving("BTCUSDT")
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.no_impact_fraction == 0.0
    assert got.discounted_volume == got.reported_volume


def test_volume_in_minutes_that_never_moved_is_discounted_away(tmp_path):
    rows = _frozen("FAKEUSDT")
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.no_impact_fraction == 1.0
    assert got.discounted_volume == 0.0
    assert got.reported_volume == 2000.0, "the reported figure is still reported"


def test_the_discount_is_the_share_of_volume_not_the_share_of_bars(tmp_path):
    """A hundred tiny still bars and one huge moving bar is not 99% suspect -
    counting bars instead of volume would say it was."""
    still = [_bar("MIXUSDT", 100.0, 100.0, 100.0, 100.0, 1.0, 5, i)
             for i in range(99)]
    moved = [_bar("MIXUSDT", 100.0, 101.0, 99.0, 100.5, 901.0, 500, 99)]
    rows = still + moved
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert abs(got.no_impact_fraction - 0.099) < 0.001


def test_a_bar_that_returned_to_its_open_still_formed_price(tmp_path):
    """Open equals close but the price moved between them. Comparing open to
    close instead of high to low would inflate the bound."""
    rows = [_bar("ROUNDUSDT", 100.0, 101.0, 99.0, 100.0, 10.0, 50, i)
            for i in range(100)]
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    assert table.rows.iloc[0].no_impact_fraction == 0.0


# --- size uniformity ------------------------------------------------------

def test_identical_clip_sizes_read_as_uniform(tmp_path):
    rows = _frozen("BOTUSDT", volume=10.0, trades=50)
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    assert table.rows.iloc[0].size_uniformity == 0.0


def test_ragged_flow_reads_as_ragged(tmp_path):
    rng = np.random.default_rng(7)
    rows = [_bar("REALUSDT", 100.0, 100.5, 99.5, 100.0,
                 float(rng.uniform(1, 100)), int(rng.integers(1, 50)), i)
            for i in range(200)]
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    assert table.rows.iloc[0].size_uniformity > 0.3


# --- refusals -------------------------------------------------------------

def test_venues_are_measured_separately(tmp_path):
    """The same asset can be honest on one venue and not on another."""
    real = _moving("XUSDT", venue="binance", seed=1)
    fake = _frozen("XUSDT", venue="binance-spot")
    store = _write(tmp_path, real + fake)
    table = measure_volume_quality(store, _as_of(real + fake))
    by_venue = {r.venue: r.no_impact_fraction for r in table.rows.itertuples()}
    assert by_venue["binance"] == 0.0
    assert by_venue["binance-spot"] == 1.0


def test_too_short_a_history_is_skipped_and_counted(tmp_path):
    rows = _moving("NEWUSDT", n=10)
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.skipped["too_few_bars"] == 1


def test_a_symbol_with_no_volume_is_skipped_not_reported_as_clean(tmp_path):
    rows = [_bar("DEADUSDT", 100.0, 100.0, 100.0, 100.0, 0.0, 0, i)
            for i in range(100)]
    store = _write(tmp_path, rows)
    table = measure_volume_quality(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.skipped["no_volume"] == 1


def test_the_clock_gates_the_measurement(tmp_path):
    early = _moving("BTCUSDT", n=100)
    late = [_bar("BTCUSDT", 100.0, 100.0, 100.0, 100.0, 10_000.0, 5, i)
            for i in range(100, 200)]
    store = _write(tmp_path, early + late)
    table = measure_volume_quality(store, early[-1]["availability_time_ns"] + 1)
    assert table.rows.iloc[0].no_impact_fraction == 0.0, (
        "the frozen bars land later and are not knowable at this clock")


def test_an_empty_store_measures_nothing(tmp_path):
    table = measure_volume_quality(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.skipped.values()) == 0
