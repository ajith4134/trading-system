"""The decile is easy; refusing to let it become a gate is the row.

FE-011 exists because FE-032 - a standalone regime detector with veto power -
was DECLINED. The constraint that survived is "feature only, never a gate", and
a constraint that lives only in a docstring is one this corpus has already
watched die five times. So the first test here is the one that pins the absence
of anything a gate could be built from without the caller writing the threshold
in its own code.
"""
import math

import pandas as pd
import pytest

from features import volatility_regime
from features.har_rv import SHORT_BARS
from features.realized_volatility import BAR_INTERVAL_NS
from features.volatility_regime import (
    DECILES,
    MIN_HISTORY_PERIODS,
    compute_volatility_regime,
    decile_of,
)
from store.parquet_partition import append_partition


# --- the constraint, made mechanical -------------------------------------

def test_no_boolean_or_threshold_is_emitted():
    """A gate needs a boolean. This table has none, so building one means
    writing the comparison in the caller - where the threshold is visible,
    reviewable and attributable, instead of hiding inside a feature module
    where it would read as a property of the market."""
    forbidden = {"is_high_volatility", "is_low_volatility", "regime",
                 "should_trade", "halt", "veto", "threshold", "elevated"}
    assert forbidden.isdisjoint(volatility_regime._COLUMNS)


def test_the_module_declares_no_volatility_cutoff_at_all():
    """The other shape this dies in: a module-level constant nobody notices,
    named for a market state rather than for a sample-size floor."""
    declared = {name for name in vars(volatility_regime)
                if name.isupper() and not name.startswith("_")}
    assert declared == {"MIN_HISTORY_PERIODS", "DECILES"}, declared


# --- the decile ----------------------------------------------------------

def test_the_extremes_land_in_the_first_and_last_tenth():
    history = [float(i) for i in range(100)]

    assert decile_of(history, -1.0)[0] == 1
    assert decile_of(history, 1_000.0)[0] == DECILES


def test_a_value_below_everything_is_in_the_first_decile_not_a_zeroth():
    """An unclamped int(percentile * 10) + 1 is fine at the bottom and breaks at
    the top; the clamp is checked at both ends because only one of them is
    obvious."""
    decile, percentile = decile_of([1.0] * 50, 0.5)

    assert percentile == 0.0
    assert decile == 1


def test_a_flat_history_sits_in_the_middle_not_at_the_bottom():
    """A tape whose variance has been identical for a hundred hours is in the
    MIDDLE of its own distribution. A strictly-below definition puts it in
    decile 1 and a consumer reads 'unusually calm'."""
    decile, percentile = decile_of([1.0] * 100, 1.0)

    assert percentile == pytest.approx(0.5)
    assert decile == 6, "the midrank of a plateau sits at the 50th percentile"


# --- end to end ----------------------------------------------------------

def _bars(symbol, venue, period_variances, start_ns=0):
    """One bar a minute; each hour's realized variance is exactly as given.

    Returns alternate sign at a constant magnitude so the sum of squares over
    the hour is the target variance and the price does not wander.
    """
    rows, price, bar = [], 100.0, 0
    for slot, variance in enumerate(period_variances):
        n_returns = SHORT_BARS - 1 if slot == 0 else SHORT_BARS
        step = math.sqrt(variance / n_returns)
        for i in range(SHORT_BARS):
            if not (slot == 0 and i == 0):
                price *= math.exp(step if i % 2 else -step)
            t = start_ns + bar * BAR_INTERVAL_NS
            rows.append({
                "venue": venue, "symbol": symbol,
                "open": price, "high": price, "low": price, "close": price,
                "volume": 1.0, "trades": 10,
                "event_time_ns": t, "ingestion_time_ns": t,
                "availability_time_ns": t,
            })
            bar += 1
    return rows


def _write(tmp_path, rows, snapshot="regime-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def test_a_calm_tape_ending_in_a_storm_reports_the_top_decile(tmp_path):
    variances = [1e-6] * MIN_HISTORY_PERIODS + [1e-2]
    rows = _bars("BTCUSDT", "binance", variances)
    table = compute_volatility_regime(_write(tmp_path, rows), _as_of(rows))

    assert len(table.rows) == 1, table.refused
    row = table.rows.iloc[0]
    assert row["decile"] == DECILES
    assert row["percentile"] > 0.99
    assert row["observations"] == MIN_HISTORY_PERIODS + 1
    assert row["age_ns"] is not None            # FE-001


def test_an_ordinary_hour_is_not_reported_as_extreme(tmp_path):
    """The decile has to be able to come back unremarkable, or it is not
    measuring anything - a feature that only ever reports 10 is a constant."""
    variances = [1e-6 * (1 + i % 20) for i in range(MIN_HISTORY_PERIODS)]
    variances.append(1e-6 * 10)                 # squarely mid-range
    rows = _bars("ETHUSDT", "binance", variances)
    table = compute_volatility_regime(_write(tmp_path, rows), _as_of(rows))

    decile = int(table.rows.iloc[0]["decile"])
    assert 3 <= decile <= 8, decile


def test_the_decile_cannot_see_a_bar_the_clock_gate_withheld(tmp_path):
    """A decile ranked against a future the model had not lived through is the
    same silent leak `features.funding_basis` guards its percentile against."""
    variances = [1e-6] * MIN_HISTORY_PERIODS + [1e-4]
    rows = _bars("SOLUSDT", "binance", variances)
    # A later, far more violent stretch that has not become available yet.
    future = _bars("SOLUSDT", "binance", [1e-1] * 30,
                   start_ns=(MIN_HISTORY_PERIODS + 5) * SHORT_BARS * BAR_INTERVAL_NS)
    for row in future:
        row["availability_time_ns"] = 10**18
    store = _write(tmp_path, rows + future)

    gated = compute_volatility_regime(store, _as_of(rows)).rows.iloc[0]

    assert gated["decile"] == DECILES, (
        "top of the history it could see - the violent future must be invisible")
    assert gated["observations"] == MIN_HISTORY_PERIODS + 1


def test_too_little_history_is_refused_rather_than_binned(tmp_path):
    """A decile from a dozen points looks exactly like one from twelve thousand."""
    rows = _bars("BTCUSDT", "binance", [1e-6] * (MIN_HISTORY_PERIODS - 5))
    table = compute_volatility_regime(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["too_few_periods"] == 1


def test_a_zero_close_refuses_the_symbol(tmp_path):
    """The placeholder-price defect, which this store has actually shipped."""
    rows = _bars("BTCUSDT", "binance", [1e-6] * (MIN_HISTORY_PERIODS + 5))
    rows[len(rows) // 2]["close"] = 0.0
    table = compute_volatility_regime(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["non_positive_close"] == 1


def test_a_gap_leaves_a_missing_hour_rather_than_a_longer_one(tmp_path):
    """Inherited from `har_rv.base_period_variances`, and checked here because
    this module's history length is what the floor is applied to - an hour
    silently absorbed into its neighbour would inflate every count."""
    early = _bars("SOLUSDT", "binance", [1e-6] * 60)
    late = _bars("SOLUSDT", "binance", [1e-6] * 60,
                 start_ns=200 * SHORT_BARS * BAR_INTERVAL_NS)
    table = compute_volatility_regime(_write(tmp_path, early + late),
                                      _as_of(late))

    assert table.refused["insufficient_period_coverage"] > 0, (
        "the empty hours between the two stretches must be counted, not spanned")


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    table = compute_volatility_regime(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}
