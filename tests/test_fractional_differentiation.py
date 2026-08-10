"""d is searched, never typed - a hardcoded order is a claim about one series
wearing the clothes of a universal constant, the same defect a hardcoded peg
level or staleness bound would be.

The weight recursion is checked two ways: against its own exact boundary
cases (d=0 is the identity transform, d=1 is a plain first difference - both
fall out of the SAME recursion with no special-casing), and against weights
computed by hand for d=0.5 at a threshold chosen to make the truncated
window short enough to check on paper.
"""
import numpy as np
import pandas as pd
import pytest

from features.fractional_differentiation import (
    _MIN_BARS,
    compute_fractional_differentiation,
    _ffd_weights,
    fdiff,
)
from features.staleness import COLUMNS as STALENESS_COLUMNS
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000
_DATASET = "bars_60000000000ns"


def _bars(symbol, closes, venue="binance-spot", start_ns=1_000_000_000_000):
    return [{
        "venue": venue, "symbol": symbol,
        "open": c, "high": c, "low": c, "close": c,
        "volume": 1.0, "trades": 10,
        "event_time_ns": start_ns + i * _BAR_NS,
        "ingestion_time_ns": start_ns + i * _BAR_NS,
        "availability_time_ns": start_ns + i * _BAR_NS,
    } for i, c in enumerate(closes)]


def _write(tmp_path, rows, snapshot="fracdiff-test"):
    append_partition(tmp_path, _DATASET, pd.DataFrame(rows), snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def _random_walk(n, level=100.0, sigma=1.0, seed=0):
    """A positive I(1) series: cumulative sum of iid steps. Non-stationary at
    d=0 by construction - each level carries the whole path before it."""
    rng = np.random.default_rng(seed)
    return list(level + np.cumsum(rng.normal(0, sigma, n)))


def _stationary(n, level=100.0, sigma=1.0, seed=0):
    """Already stationary: iid noise around a fixed level, no walk at all."""
    rng = np.random.default_rng(seed)
    return list(level + rng.normal(0, sigma, n))


# --- the weight recursion itself, independent of the store -------------------

def test_ffd_weights_at_d_zero_is_the_identity_weight():
    """w_0=1 and the recursion's own next term is exactly zero at d=0 - (d-1+1)
    = d = 0 - so the window truncates to length 1 with no special-casing."""
    weights = _ffd_weights(0.0, threshold=1e-4, max_length=100)
    assert weights.tolist() == [1.0]


def test_ffd_weights_at_d_one_is_a_plain_difference_pair():
    """w_1=-1 and w_2's own factor (d-2+1)=0 at d=1, so the recursion
    truncates to exactly [1, -1] - the first-difference kernel - again with
    no special-casing for the integer case."""
    weights = _ffd_weights(1.0, threshold=1e-4, max_length=100)
    assert weights.tolist() == [1.0, -1.0]


def test_ffd_weights_hand_checked_at_d_half():
    """w_k = -w_{k-1} * (d-k+1)/k, computed by hand for d=0.5:
    w0=1
    w1 = -1        * (0.5-1+1)/1 = -0.5
    w2 = -(-0.5)   * (0.5-2+1)/2 = -0.125
    w3 = -(-0.125) * (0.5-3+1)/3 = -0.0625
    w4 = -(-0.0625)* (0.5-4+1)/4 = -0.0390625
    w5 = -(-0.0390625) * (0.5-5+1)/5 ~= -0.02734, magnitude below the 0.03
    threshold chosen here so the window stops at w4."""
    weights = _ffd_weights(0.5, threshold=0.03, max_length=100)
    assert weights == pytest.approx(
        [1.0, -0.5, -0.125, -0.0625, -0.0390625], abs=1e-9)


def test_fdiff_d_zero_returns_the_original_series_unchanged():
    series = pd.Series([10.0, 20.5, 7.3, 42.0, 5.5])
    out = fdiff(series, 0.0)
    assert out.tolist() == series.tolist()
    assert len(out) == len(series)


def test_fdiff_d_one_matches_a_plain_first_difference():
    series = pd.Series([10.0, 20.5, 7.3, 42.0, 5.5, 61.0, 8.25])
    out = fdiff(series, 1.0)
    expected = series.diff().dropna()
    assert out.tolist() == pytest.approx(expected.tolist())


# --- the search, through the public store-backed entry point -----------------

def test_a_random_walk_is_made_stationary_while_a_stationary_series_barely_moves(tmp_path):
    """The standard method: ascending d, first grid point whose FFD-differenced
    series rejects the ADF unit-root null. A random walk needs real
    differencing; a series that was already stationary needs almost none."""
    walk_rows = _bars("WALKUSDT", _random_walk(_MIN_BARS + 50, seed=1))
    flat_rows = _bars("FLATUSDT", _stationary(_MIN_BARS + 50, seed=2))
    store = _write(tmp_path, walk_rows + flat_rows)
    table = compute_fractional_differentiation(store, _as_of(walk_rows + flat_rows))

    by_symbol = {row.symbol: row for row in table.rows.itertuples()}
    assert by_symbol["WALKUSDT"].d > by_symbol["FLATUSDT"].d
    # every reported d is backed by an ADF rejection at the declared level
    assert by_symbol["WALKUSDT"].adf_pvalue < 0.05
    assert by_symbol["FLATUSDT"].adf_pvalue < 0.05
    assert sum(table.refused.values()) == 0


def test_fractional_order_keeps_more_memory_than_a_plain_first_difference(tmp_path):
    """The catalogue's claim, checked per row: the chosen d's differenced
    series must correlate with the original level more than a d=1 difference
    would, on a series where the two are known to differ (a random walk)."""
    rows = _bars("WALKUSDT", _random_walk(_MIN_BARS + 50, seed=1))
    store = _write(tmp_path, rows)
    table = compute_fractional_differentiation(store, _as_of(rows))
    row = table.rows.iloc[0]

    assert row.d < 1.0
    level = pd.Series(_random_walk(_MIN_BARS + 50, seed=1))
    plain_diff = fdiff(level, 1.0)
    chosen_diff = fdiff(level, row.d)
    aligned_level = level.iloc[len(level) - len(plain_diff):]
    plain_corr = np.corrcoef(plain_diff, aligned_level)[0, 1]
    assert row.memory_retained_corr > abs(plain_corr)


# --- refusals ------------------------------------------------------------

def test_too_few_bars_refuses_rather_than_searching_a_short_window(tmp_path):
    rows = _bars("THINUSDT", _stationary(_MIN_BARS - 10, seed=3))
    store = _write(tmp_path, rows)
    table = compute_fractional_differentiation(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["too_few_observations"] == 1


def test_non_positive_price_is_refused_never_differenced(tmp_path):
    closes = _stationary(_MIN_BARS + 10, seed=4)
    closes[5] = 0.0
    rows = _bars("BROKENUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_fractional_differentiation(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["non_positive_price"] == 1


def test_a_series_that_never_stabilises_is_refused_not_forced_to_d_one(tmp_path):
    """An explosive series (compounding growth) stays non-stationary at every
    grid point up to and including d=1 - refused rather than silently
    reported at the top of the grid as though that were an answer."""
    n = _MIN_BARS + 50
    rng = np.random.default_rng(5)
    closes = list(10.0 * (1.003 ** np.arange(n)) + rng.normal(0, 0.01, n))
    rows = _bars("EXPLOSIVEUSDT", closes)
    store = _write(tmp_path, rows)
    table = compute_fractional_differentiation(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["no_stationary_d_found"] == 1


# --- staleness (FE-001) ---------------------------------------------------

def test_every_row_is_staleness_stamped(tmp_path):
    rows = _bars("WALKUSDT", _random_walk(_MIN_BARS + 50, seed=1))
    store = _write(tmp_path, rows)
    table = compute_fractional_differentiation(store, _as_of(rows))
    for column in STALENESS_COLUMNS:
        assert column in table.rows.columns
    # the newest bar in the fixture is one bar-width before as_of, well inside
    # any reasonable multiple of the series' own 60s cadence
    assert table.rows.iloc[0].freshness == "FRESH"


def test_empty_store_returns_empty_rows_not_an_error(tmp_path):
    table = compute_fractional_differentiation(tmp_path, as_of_ns=1_000)
    assert len(table.rows) == 0
    assert set(table.refused.values()) == {0}
