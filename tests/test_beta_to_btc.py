"""A beta between two clocks is a number in [-1, 1] and it is noise.

The defect this file is mostly about: two symbols' bar series are different
lengths, so zipping them by position pairs BTC's 09:04 return with an altcoin's
11:17 return. The result looks entirely normal. Every other test here defends a
tautology or a division that would report a number where there is none.
"""
import math

import pandas as pd
import pytest

from features.beta_to_btc import (
    BENCHMARK_BY_VENUE,
    HORIZONS_NS,
    MIN_OVERLAPPING_RETURNS,
    adjacent_log_returns,
    beta_and_correlation,
    compute_beta_to_btc,
)
from features.realized_volatility import BAR_INTERVAL_NS
from store.parquet_partition import append_partition

_BARS = MIN_OVERLAPPING_RETURNS + 20


# --- the statistics, checked against a hand-computed case -----------------

def test_a_symbol_moving_twice_the_benchmark_has_beta_two():
    benchmark = [0.01, -0.01, 0.02, -0.02, 0.005, -0.005] * 10
    symbol = [2 * r for r in benchmark]

    beta, correlation = beta_and_correlation(symbol, benchmark)

    assert beta == pytest.approx(2.0)
    assert correlation == pytest.approx(1.0)


def test_an_inverse_symbol_has_negative_beta_and_correlation():
    benchmark = [0.01, -0.01, 0.02, -0.02] * 10
    symbol = [-1.5 * r for r in benchmark]

    beta, correlation = beta_and_correlation(symbol, benchmark)

    assert beta == pytest.approx(-1.5)
    assert correlation == pytest.approx(-1.0)


def test_a_flat_benchmark_refuses_rather_than_dividing():
    """An hour in which BTC did not move gives no beta, not a large one."""
    assert beta_and_correlation([0.01, -0.01] * 20, [0.0] * 40) == (
        "benchmark_did_not_move")


def test_a_flat_symbol_refuses_rather_than_reporting_zero_correlation():
    """0.0 would say 'moves independently of BTC' about an instrument that did
    not move at all - a claim about the relationship where there was no
    observation of one."""
    assert beta_and_correlation([0.0] * 40, [0.01, -0.01] * 20) == (
        "symbol_did_not_move")


def test_beta_and_correlation_are_not_the_same_number():
    """Beta 1.8 with correlation 0.9 is a leveraged BTC position; beta 1.8 with
    correlation 0.2 is an instrument that happened to move. Sizing them alike is
    how a book ends up accidentally concentrated, so both are reported."""
    benchmark = [0.01, -0.01, 0.01, -0.01] * 10
    # Same beta, different reliability: noise added in the null space of the
    # benchmark so the covariance is untouched and only the symbol's variance
    # grows.
    clean = [1.8 * r for r in benchmark]
    noisy = [c + (0.05 if i % 2 else -0.05) * (1 if i % 4 < 2 else -1)
             for i, c in enumerate(clean)]

    clean_beta, clean_correlation = beta_and_correlation(clean, benchmark)
    noisy_beta, noisy_correlation = beta_and_correlation(noisy, benchmark)

    assert clean_beta == pytest.approx(noisy_beta, rel=1e-9)
    assert abs(noisy_correlation) < abs(clean_correlation)


# --- returns are keyed by the clock ---------------------------------------

def test_only_adjacent_bars_produce_a_return():
    """A ratio across a five-day hole would enter the covariance as one enormous
    observation and dominate every statistic computed from it."""
    from decimal import Decimal

    times = [0, BAR_INTERVAL_NS, 100 * BAR_INTERVAL_NS,
             101 * BAR_INTERVAL_NS]
    closes = [Decimal("100"), Decimal("101"), Decimal("300"), Decimal("303")]

    returns = adjacent_log_returns(times, closes)

    assert set(returns) == {BAR_INTERVAL_NS, 101 * BAR_INTERVAL_NS}
    assert 100 * BAR_INTERVAL_NS not in returns, "the gap makes no return"


# --- fixtures -------------------------------------------------------------

def _bars(symbol, venue, closes, start_ns=0):
    return [{
        "venue": venue, "symbol": symbol,
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1.0, "trades": 10,
        "event_time_ns": start_ns + i * BAR_INTERVAL_NS,
        "ingestion_time_ns": start_ns + i * BAR_INTERVAL_NS,
        "availability_time_ns": start_ns + i * BAR_INTERVAL_NS,
    } for i, close in enumerate(closes)]


def _walk(steps, start=100.0):
    price, closes = start, [start]
    for step in steps:
        price *= math.exp(step)
        closes.append(price)
    return closes


def _steps(n, scale=1.0, seed=0):
    """Deterministic pseudo-returns; no RNG so the expected beta is exact."""
    return [scale * (0.01 if (i * 7 + seed) % 3 else -0.008) for i in range(n)]


def _write(tmp_path, rows, snapshot="beta-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


# --- end to end -----------------------------------------------------------

def test_a_symbol_tracking_btc_at_twice_the_size_reports_beta_two(tmp_path):
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    assert len(hourly) == 1, table.refused
    row = hourly.iloc[0]
    assert row["symbol"] == "ALTUSDT"
    assert row["benchmark"] == "BTCUSDT"
    assert row["beta"] == pytest.approx(2.0, rel=1e-6)
    assert row["correlation"] == pytest.approx(1.0, rel=1e-6)
    assert row["age_ns"] is not None                # FE-001


def test_misaligned_series_contribute_only_their_shared_minutes(tmp_path):
    """The whole module. The altcoin starts an hour late, so a positional zip
    would pair its first return with BTC's first - two unrelated moments - and
    report a correlation between them.
    """
    steps = _steps(_BARS + 60)
    btc = _bars("BTCUSDT", "binance", _walk(steps))
    # Same underlying path, offset by 60 bars in the STORE but not in the walk:
    # the altcoin prints the LATER stretch of the same series.
    late_start = 60 * BAR_INTERVAL_NS
    alt = _bars("ALTUSDT", "binance",
                _walk([2 * s for s in steps[60:]]), start_ns=late_start)
    table = compute_beta_to_btc(_write(tmp_path, btc + alt), _as_of(btc))

    row = table.rows[table.rows["horizon"] == "24h"].iloc[0]
    # Aligned on the clock, the altcoin is exactly twice BTC over the shared
    # minutes. Zipped by position it would be twice a DIFFERENT hour of BTC.
    assert row["beta"] == pytest.approx(2.0, rel=1e-6)
    assert row["overlapping_returns"] < len(steps), "only the shared minutes"


def test_the_overlap_count_rides_every_row(tmp_path):
    """A beta from 40 shared minutes and one from 4,000 print the same way."""
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert set(table.rows["overlapping_returns"]) == {_BARS}


def test_btc_s_beta_to_itself_is_not_reported(tmp_path):
    """1.0 by construction, and it would sit in the table looking like the
    best-measured row there."""
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert "BTCUSDT" not in set(table.rows["symbol"])


def test_a_thin_overlap_is_refused_rather_than_measured(tmp_path):
    """A beta from a handful of shared minutes is noise with two decimals."""
    steps = _steps(MIN_OVERLAPPING_RETURNS - 5)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["too_few_overlapping_returns"] >= 1


def test_a_venue_with_no_benchmark_on_record_is_refused(tmp_path):
    """Picking the wrong ticker gives a beta to something that is not the market
    factor, and nothing downstream would notice."""
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "okx", _walk(steps))
            + _bars("ALTUSDT", "okx", _walk(steps)))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["unknown_benchmark_symbol"] == 1


def test_a_venue_whose_btc_feed_is_absent_is_refused_apart(tmp_path):
    """Counted separately from a thin instrument: one is a feed outage and the
    other is an illiquid symbol, and they call for different work."""
    rows = _bars("ALTUSDT", "binance", _walk(_steps(_BARS)))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["benchmark_absent"] == 1
    assert table.refused["too_few_overlapping_returns"] == 0


def test_each_venue_is_measured_against_its_own_btc(tmp_path):
    """Another venue's BTC would import that venue's basis, outages and
    microstructure - largest exactly when venues diverge, which is when a beta
    matters most."""
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps]))
            + _bars("BTC", "hyperliquid", _walk(steps))
            + _bars("ETH", "hyperliquid", _walk([3 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    hourly = table.rows[table.rows["horizon"] == "1h"]
    by_venue = {row.venue: row for row in hourly.itertuples()}
    assert by_venue["binance"].benchmark == "BTCUSDT"
    assert by_venue["hyperliquid"].benchmark == "BTC"
    assert by_venue["hyperliquid"].beta == pytest.approx(3.0, rel=1e-6)


def test_the_benchmark_map_covers_every_venue_being_captured():
    """A venue whose bars land in the store and whose BTC ticker is not on the
    list produces no beta at all, silently, until someone reads the refusals."""
    assert {"binance", "binance-spot", "hyperliquid", "bybit",
            "coinbase"} <= set(BENCHMARK_BY_VENUE)


def test_a_zero_close_refuses_the_symbol(tmp_path):
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    for row in rows:
        if row["symbol"] == "ALTUSDT" and row["event_time_ns"] == 5 * BAR_INTERVAL_NS:
            row["close"] = 0.0
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["non_positive_close"] >= 1


def test_both_horizons_are_reported(tmp_path):
    steps = _steps(_BARS)
    rows = (_bars("BTCUSDT", "binance", _walk(steps))
            + _bars("ALTUSDT", "binance", _walk([2 * s for s in steps])))
    table = compute_beta_to_btc(_write(tmp_path, rows), _as_of(rows))

    assert set(table.rows["horizon"]) == set(HORIZONS_NS)


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    table = compute_beta_to_btc(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}
