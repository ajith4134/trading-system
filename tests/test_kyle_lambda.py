"""Kyle's lambda is a fitted slope; these tests hold the fit to the house rule
that a regression must never hand back a number it cannot support.

The synthetic-slope test builds a series where the true price-impact
coefficient is known by construction (zero noise), so recovering it is a
closed check rather than a plausibility read. The refusal tests each isolate
one degenerate case - too few points, a regressor with no variance, a price
that never moved, a bar that cannot be priced at all - because a regression
that silently returns a slope on a degenerate window is exactly the failure
the catalogue note ("descriptor for sizing, not a standalone signal") warns a
caller against trusting blind.
"""
import math

import pandas as pd

from features.kyle_lambda import MIN_OBSERVATIONS, compute_kyle_lambda
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000


def _bar(symbol, close, volume, i, venue="binance-spot", trades=10):
    return {
        "venue": venue, "symbol": symbol,
        "open": close, "high": close, "low": close, "close": close,
        "volume": volume, "trades": trades,
        "event_time_ns": 10**12 + i * _BAR_NS,
        "ingestion_time_ns": 10**12 + i * _BAR_NS,
        "availability_time_ns": 10**12 + i * _BAR_NS,
    }


def _write(tmp_path, rows, snapshot="kyle-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


# A hand-built net-flow sequence: nonzero, varied in magnitude and sign, long
# enough to clear MIN_OBSERVATIONS. Reused by several tests as the "healthy"
# series so each test only has to explain what it changes.
_FLOWS = [10, -5, 8, -12, 6, -3, 15, -7, 9, -11, 4, -6, 13, -9, 5, -8, 10, -4,
          7, -10, 12, -6, 8, -5, 9, -7, 11, -3, 6]
assert len(_FLOWS) >= MIN_OBSERVATIONS


def _known_slope_series(symbol, lam, flows=_FLOWS, base=100.0, venue="binance-spot"):
    """Bars whose close-to-close price change is EXACTLY `lam * flow` per step,
    and whose volume is exactly `|flow|` - so the tick rule's sign(delta) *
    volume reproduces `flow` on the nose and the fit has zero residual to find.
    """
    closes = [base]
    for f in flows:
        closes.append(closes[-1] + lam * f)
    rows = [_bar(symbol, closes[0], 1.0, 0, venue=venue)]
    for i, f in enumerate(flows, start=1):
        rows.append(_bar(symbol, closes[i], abs(f), i, venue=venue))
    return rows


# --- the fit ----------------------------------------------------------------

def test_recovers_a_known_slope_from_a_zero_residual_series(tmp_path):
    lam = 0.05
    rows = _known_slope_series("BTCUSDT", lam)
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert sum(table.refused.values()) == 0
    got = table.rows.iloc[0]
    assert abs(got.lambda_price_impact - lam) < 1e-9
    assert got.r_squared > 0.999999, "zero-noise construction must fit almost perfectly"
    assert got.observations == len(_FLOWS)
    assert got.window_bars == len(rows)
    assert got.window_start_ns == rows[0]["event_time_ns"]
    assert got.window_end_ns == rows[-1]["event_time_ns"]


def test_fit_quality_and_window_ride_the_row_as_data(tmp_path):
    """The catalogue calls this a sizing input, not a signal - a slope with no
    goodness-of-fit or window beside it is unjudgeable, so both must be columns
    on the row itself rather than only asserted in the docstring."""
    rows = _known_slope_series("ETHUSDT", 0.02)
    store = _write(tmp_path, rows)
    got = compute_kyle_lambda(store, _as_of(rows)).rows.iloc[0]
    for column in ("lambda_price_impact", "r_squared", "residual_std",
                   "observations", "window_bars", "window_start_ns",
                   "window_end_ns", "signed_flow_source"):
        assert column in got.index, f"missing {column}"
    assert got.signed_flow_source == "tick_rule_signed_volume_proxy", (
        "the flow is not real trade-side data and the row must say so, not just the docstring")


def test_exactly_the_minimum_observations_is_accepted(tmp_path):
    flows = _FLOWS[:MIN_OBSERVATIONS]
    rows = _known_slope_series("MINUSDT", 0.03, flows=flows)
    store = _write(tmp_path, rows)
    table = compute_kyle_lambda(store, _as_of(rows))
    assert sum(table.refused.values()) == 0
    assert table.rows.iloc[0].observations == MIN_OBSERVATIONS


# --- refusals -----------------------------------------------------------

def test_too_few_observations_is_refused_and_counted_not_defaulted(tmp_path):
    """A regression on a handful of points must not silently return a slope."""
    flows = _FLOWS[:MIN_OBSERVATIONS - 1]
    rows = _known_slope_series("THINUSDT", 0.03, flows=flows)
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert len(table.rows) == 0
    assert table.refused["too_few_observations"] == 1


def test_zero_variance_regressor_is_refused_not_reported_as_zero_impact(tmp_path):
    """Every bar trades the same direction with the same size, so the tick-rule
    flow is a constant - the OLS slope is 0/0, and reporting 0 would read as
    "this instrument has no price impact" rather than "unmeasurable here"."""
    increments = [1, 3, 2, 5, 4, 6, 2, 3, 5, 1, 4, 2, 6, 3, 5, 2, 4, 1, 3, 5]
    assert len(increments) >= MIN_OBSERVATIONS
    closes = [100.0]
    for step in increments:
        closes.append(closes[-1] + step)
    volume = 7.0  # constant clip size on every bar, including the first
    rows = [_bar("FLATFLOWUSDT", closes[0], volume, 0)]
    for i, _ in enumerate(increments, start=1):
        rows.append(_bar("FLATFLOWUSDT", closes[i], volume, i))
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert len(table.rows) == 0
    assert table.refused["zero_variance_regressor"] == 1


def test_zero_price_variance_is_refused_not_reported_with_a_fake_r_squared(tmp_path):
    """The close moves by the identical amount every bar - delta_price has no
    variance, R^2 is 0/0, and a slope with no fit quality beside it is exactly
    the unjudgeable number the catalogue note warns a sizing caller against."""
    volumes = [5, 10, 3, 8, 12, 4, 9, 6, 11, 2, 7, 13, 5, 9, 3, 8, 10, 4, 6, 12]
    assert len(volumes) >= MIN_OBSERVATIONS
    delta = 2.0
    rows = [_bar("CONSTDELTAUSDT", 100.0, volumes[0], 0)]
    for i, vol in enumerate(volumes, start=1):
        rows.append(_bar("CONSTDELTAUSDT", 100.0 + i * delta, vol, i))
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert len(table.rows) == 0
    assert table.refused["zero_price_variance"] == 1


def test_a_non_positive_close_in_the_window_is_refused_and_counted(tmp_path):
    rows = _known_slope_series("BADPRICEUSDT", 0.02)
    rows[10]["close"] = -1.0
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert len(table.rows) == 0
    assert table.refused["non_positive_price"] == 1


def test_an_unparseable_close_is_refused_and_counted(tmp_path):
    rows = _known_slope_series("NANPRICEUSDT", 0.02)
    rows[10]["close"] = math.nan
    store = _write(tmp_path, rows)

    table = compute_kyle_lambda(store, _as_of(rows))

    assert len(table.rows) == 0
    assert table.refused["unparseable_bar"] == 1


# --- staleness (FE-001) ------------------------------------------------------

def test_staleness_is_stamped_per_venue_symbol(tmp_path):
    rows = _known_slope_series("FRESHUSDT", 0.02)
    store = _write(tmp_path, rows)
    as_of = _as_of(rows)

    got = compute_kyle_lambda(store, as_of).rows.iloc[0]

    assert got.freshness == "FRESH"
    assert got.routine_gap_ns == _BAR_NS
    assert got.input_event_ns == rows[-1]["event_time_ns"]
    assert got.age_ns == as_of - rows[-1]["event_time_ns"]


# --- clock gating -------------------------------------------------------

def test_the_clock_gates_the_fit(tmp_path):
    """Bars available only after `as_of_ns` must not enter the window."""
    lam = 0.04
    early = _known_slope_series("GATEDUSDT", lam)
    future_start = len(early)
    future = [_bar("GATEDUSDT", -1.0, 5.0, future_start + i)
              for i in range(40)]  # would poison every refusal check if visible
    store = _write(tmp_path, early + future)

    table = compute_kyle_lambda(store, _as_of(early))

    assert sum(table.refused.values()) == 0
    got = table.rows.iloc[0]
    assert abs(got.lambda_price_impact - lam) < 1e-9
    assert got.window_end_ns == early[-1]["event_time_ns"]


# --- empty store -------------------------------------------------------

def test_an_empty_store_returns_an_empty_table_with_zero_refusals(tmp_path):
    table = compute_kyle_lambda(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0
