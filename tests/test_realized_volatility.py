"""Realized volatility is one formula; the module exists to refuse it correctly.

Every refusal path here corresponds to a number that would otherwise look
perfectly reasonable: a two-return window computed as confidently as a
thousand-return one, a zero close producing -inf instead of a refusal, a
252-trading-day annualisation convention borrowed from equities and silently
wrong for a market that never closes.
"""
import math

import pandas as pd
import pytest

from features.realized_volatility import (
    BAR_INTERVAL_NS,
    HORIZONS_NS,
    MIN_REQUIRED_RETURNS,
    compute_realized_volatility,
)
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000


def _bars(symbol, closes, venue="binance-spot", start_ns=10**12):
    """One bar per close, exactly one bar interval apart - a clean tape."""
    return [{
        "venue": venue, "symbol": symbol,
        "open": c, "high": c, "low": c, "close": c,
        "volume": 1.0, "trades": 10,
        "event_time_ns": start_ns + i * _BAR_NS,
        "ingestion_time_ns": start_ns + i * _BAR_NS,
        "availability_time_ns": start_ns + i * _BAR_NS,
    } for i, c in enumerate(closes)]


def _write(tmp_path, rows, snapshot="rv-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


# --- the arithmetic, checked against an independent computation -----------

def test_realized_vol_matches_a_hand_computed_value(tmp_path):
    """Three bars, two equal-ratio steps: closes 100 -> 110 -> 121.

    Both log returns equal ln(1.1) exactly, so realized_vol and the
    annualised figure are computed here from first principles with
    `math.log` - a path that shares no code with the module under test - and
    compared against the module's Decimal-based output.
    """
    closes = [100.0, 110.0, 121.0]
    rows = _bars("BTCUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))

    r = math.log(1.1)
    expected_realized_vol = math.sqrt(r * r + r * r)
    periods_per_year = (365 * 24 * 3600 * 10**9) // BAR_INTERVAL_NS
    expected_annualised = math.sqrt((r * r) * periods_per_year)

    row = table.rows[(table.rows.symbol == "BTCUSDT")
                     & (table.rows.horizon == "5m")].iloc[0]
    assert row.observations == 2
    assert float(row.realized_vol) == pytest.approx(expected_realized_vol, rel=1e-9)
    assert float(row.annualised_vol) == pytest.approx(expected_annualised, rel=1e-9)


def test_annualisation_factor_is_derived_from_the_bar_interval():
    """525,600 one-minute bars in a 365-day year - never sqrt(252).

    Crypto trades 24/7, so a trading-day convention understates the periods
    per year by a factor that grows as the bar interval shrinks. The module's
    own constant is checked against an independently computed one.
    """
    year_ns = 365 * 24 * 3600 * 10**9
    assert year_ns % BAR_INTERVAL_NS == 0, "the year must divide evenly for the factor to be exact"
    assert year_ns // BAR_INTERVAL_NS == 525_600


def test_minimum_required_returns_scales_with_horizon_length():
    """A 5-minute horizon is not held to a 24-hour horizon's bar count -
    the failure `peg_monitor` and `volume_quality` already document for a
    single fixed window, generalised across five different window lengths."""
    assert MIN_REQUIRED_RETURNS["5m"] == 2, "the floor: below two returns there is nothing to average"
    assert MIN_REQUIRED_RETURNS["5m"] < MIN_REQUIRED_RETURNS["1h"] < MIN_REQUIRED_RETURNS["24h"]


# --- too few observations ---------------------------------------------------

def test_too_few_observations_is_refused_for_every_horizon(tmp_path):
    """One bar gives zero returns - refused for 5m, and every longer horizon
    too, since none of them has more data than the 5m window does here."""
    rows = _bars("ONEUSDT", [100.0])
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["too_few_observations"] == len(HORIZONS_NS)


def test_exactly_at_the_floor_succeeds_one_below_it_refuses(tmp_path):
    """5m's minimum is 2 returns (3 bars). Two bars (1 return) must refuse;
    three bars (2 returns) must produce a value - the boundary the floor
    actually draws, not an approximation of it."""
    two_bars = _bars("EDGEUSDT", [100.0, 101.0])
    store = _write(tmp_path, two_bars)
    table = compute_realized_volatility(store, _as_of(two_bars))
    assert len(table.rows[table.rows.horizon == "5m"]) == 0
    assert table.refused["too_few_observations"] >= 1

    three_bars = _bars("EDGE2USDT", [100.0, 101.0, 102.0])
    store2 = _write(tmp_path, three_bars, snapshot="rv-test-2")
    table2 = compute_realized_volatility(store2, _as_of(three_bars))
    assert len(table2.rows[table2.rows.horizon == "5m"]) == 1


def test_refused_counts_reasons_never_returns_a_default(tmp_path):
    """A window that cannot be priced never appears in `rows` with a
    filled-in number - it is absent from `rows` and present in `refused`."""
    rows = _bars("SHORTUSDT", [100.0, 101.0])  # one return: below every horizon's floor
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows[table.rows.symbol == "SHORTUSDT"]) == 0
    assert sum(table.refused.values()) == len(HORIZONS_NS)


# --- non-positive and unparseable closes ------------------------------------

def test_a_zero_close_is_refused_not_divided_by(tmp_path):
    """A documented past defect: placeholder frames carry price '0'. A log
    return over it must never reach -inf, and the window is refused whole
    rather than silently recomputed around the bad bar."""
    closes = [100.0, 0.0, 101.0, 100.5, 100.2, 100.8]
    rows = _bars("ZEROUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows[table.rows.symbol == "ZEROUSDT"]) == 0
    assert table.refused["non_positive_close"] >= 1


def test_a_negative_close_is_refused(tmp_path):
    closes = [100.0, -5.0, 101.0]
    rows = _bars("NEGUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows[table.rows.symbol == "NEGUSDT"]) == 0
    assert table.refused["non_positive_close"] >= 1


def test_a_nan_close_is_refused_as_unparseable(tmp_path):
    closes = [100.0, float("nan"), 101.0]
    rows = _bars("NANUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows[table.rows.symbol == "NANUSDT"]) == 0
    assert table.refused["unparseable_close"] >= 1


# --- staleness (FE-001) -----------------------------------------------------

def test_every_output_row_carries_a_staleness_stamp(tmp_path):
    rows = _bars("STAMPUSDT", [100.0 + i * 0.01 for i in range(30)])
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    assert len(table.rows) > 0
    for column in ("as_of_ns", "input_event_ns", "age_ns", "routine_gap_ns", "freshness"):
        assert column in table.rows.columns
    assert table.rows["freshness"].notna().all()


def test_staleness_reflects_the_venues_own_feed_not_a_shared_constant(tmp_path):
    """A feed queried the instant it last posted reads FRESH; the same feed
    queried twenty bar-widths after it stopped reads STALE - measured against
    its own cadence, not a constant shared with a faster or slower feed."""
    n = 1500
    fresh_store = tmp_path / "fresh"
    stale_store = tmp_path / "stale"
    fresh_rows = _bars("FRESHUSDT", [100.0 + i * 0.01 for i in range(n)])
    stale_rows = _bars("STALEUSDT", [100.0 + i * 0.01 for i in range(n)])
    _write(fresh_store, fresh_rows)
    _write(stale_store, stale_rows)

    fresh_table = compute_realized_volatility(fresh_store, _as_of(fresh_rows))
    stale_as_of = stale_rows[-1]["availability_time_ns"] + 20 * _BAR_NS
    stale_table = compute_realized_volatility(stale_store, stale_as_of)

    fresh_row = fresh_table.rows[(fresh_table.rows.symbol == "FRESHUSDT")
                                 & (fresh_table.rows.horizon == "1h")].iloc[0]
    stale_row = stale_table.rows[(stale_table.rows.symbol == "STALEUSDT")
                                 & (stale_table.rows.horizon == "1h")].iloc[0]
    assert fresh_row.freshness == "FRESH"
    assert stale_row.freshness == "STALE"


# --- the clock gate ----------------------------------------------------------

def test_the_clock_gates_the_window(tmp_path):
    """Bars available after the as-of clock do not exist for this read."""
    early = _bars("GATEDUSDT", [100.0, 101.0, 102.0])
    late = _bars("GATEDUSDT", [500.0, 900.0, 50.0],
                 start_ns=early[-1]["event_time_ns"] + _BAR_NS)
    store = _write(tmp_path, early + late)
    table = compute_realized_volatility(store, _as_of(early))
    row = table.rows[(table.rows.symbol == "GATEDUSDT")
                     & (table.rows.horizon == "5m")].iloc[0]
    assert row.observations == 2, "the wild late bars must be invisible at this clock"


# --- multi-horizon is not decorative ----------------------------------------

def test_a_recent_shock_moves_the_short_horizon_without_moving_the_long_one(tmp_path):
    """The reason for more than one horizon: a single long window buries a
    recent shock inside a full day's average. 23 quiet hours, then five
    violent minutes - compared on the ANNUALISED figure, because the raw
    `realized_vol` is a cumulative sum and only grows with more data; the
    per-bar, horizon-normalised `annualised_vol` is the one a caller can
    compare across horizons."""
    quiet = [100.0 + math.sin(i) * 0.001 for i in range(1380)]
    shock = [100.0, 130.0, 90.0, 105.0, 100.0, 100.0]
    closes = quiet + shock
    rows = _bars("SHOCKUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_realized_volatility(store, _as_of(rows))
    by_horizon = {r.horizon: r for r in table.rows[table.rows.symbol == "SHOCKUSDT"].itertuples()}
    assert float(by_horizon["5m"].annualised_vol) > float(by_horizon["24h"].annualised_vol) * 3


# --- refusal accounting per venue and the empty store -----------------------

def test_venues_are_measured_separately(tmp_path):
    """The same symbol can be volatile on one venue and dead flat on another."""
    calm = _bars("XUSDT", [100.0] * 10, venue="binance")
    wild = _bars("XUSDT", [100.0, 150.0, 80.0, 120.0, 90.0, 110.0,
                           95.0, 130.0, 85.0, 105.0], venue="binance-spot")
    store = _write(tmp_path, calm + wild)
    table = compute_realized_volatility(store, _as_of(calm + wild))
    by_venue = {r.venue: r for r in table.rows[
        (table.rows.symbol == "XUSDT") & (table.rows.horizon == "5m")].itertuples()}
    assert float(by_venue["binance"].realized_vol) == 0.0
    assert float(by_venue["binance-spot"].realized_vol) > 0.0


def test_an_empty_store_returns_an_empty_table_with_zero_refusals(tmp_path):
    table = compute_realized_volatility(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0
