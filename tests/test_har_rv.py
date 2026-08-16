"""HAR-RV is four coefficients; the module exists to stop them lying.

Every test here defends a way this model produces an impressive number without
having learned anything: a fit that saw its own target, a cascade averaged over
a window with holes in it, sixty times as many observations as there is
information, a negative variance clamped into plausibility, and an in-sample
t-statistic on overlapping regressors. Each of those failures is in the
flattering direction, which is why none of them is caught by looking at the
output.
"""
import math

import numpy as np
import pandas as pd
import pytest

from features import har_rv
from features.har_rv import (
    LONG_PERIODS,
    MEDIUM_PERIODS,
    MIN_EVALUATION_OBSERVATIONS,
    MIN_FIT_OBSERVATIONS,
    MIN_RETURNS_PER_PERIOD,
    SHORT_BARS,
    BasePeriod,
    base_period_variances,
    cascade_observations,
    compute_har_rv,
    walk_forward_forecasts,
)
from features.realized_volatility import BAR_INTERVAL_NS
from models import naive_baseline
from store.parquet_partition import append_partition

_PERIOD_NS = SHORT_BARS * BAR_INTERVAL_NS


def _period(index, variance):
    """One clean base period at slot `index`."""
    return BasePeriod(start_ns=index * _PERIOD_NS,
                      end_ns=(index + 1) * _PERIOD_NS,
                      variance=variance, n_returns=SHORT_BARS)


def _cascade_series(n_periods, *, b_short, b_medium, b_long, seed=0):
    """A variance series that genuinely obeys a HAR cascade, plus noise.

    Built forward from the model itself so the relation the module is meant to
    find is actually present. Without this, "HAR did not beat the random walk"
    would be the only result the suite could ever observe, and a module that
    always fails is indistinguishable from a module that never works.
    """
    rng = np.random.default_rng(seed)
    variances = list(rng.uniform(0.8, 1.2, LONG_PERIODS) * 1e-4)
    for _ in range(n_periods - LONG_PERIODS):
        window = variances[-LONG_PERIODS:]
        short = window[-1]
        medium = sum(window[-MEDIUM_PERIODS:]) / MEDIUM_PERIODS
        long = sum(window) / LONG_PERIODS
        nxt = b_short * short + b_medium * medium + b_long * long
        nxt *= float(rng.normal(1.0, 0.05))
        variances.append(max(nxt, 1e-9))
    return variances


# --- periods are cut on the clock, never on bar count ---------------------

def test_a_gap_produces_a_missing_period_not_a_longer_one():
    """Bars for hour 0 and hour 2, nothing for hour 1.

    Cutting on bar count would fold hour 2's bars into slot 1 and report an
    unbroken series; cutting on the clock leaves slot 1 empty and says so. The
    difference is invisible in the output and total in what it means.
    """
    from decimal import Decimal

    times, closes = [], []
    for slot in (0, 2):
        for i in range(SHORT_BARS):
            times.append(slot * _PERIOD_NS + i * BAR_INTERVAL_NS)
            closes.append(Decimal("100") + Decimal(i % 2))

    periods, refused = base_period_variances(times, closes)

    assert len(periods) == 3, "three slots span hour 0 to hour 2"
    assert periods[0] is not None
    assert periods[1] is None, "the missing hour must be missing, not absorbed"
    assert periods[2] is not None
    assert refused["insufficient_period_coverage"] == 1


def test_a_thin_period_is_refused_rather_than_computed():
    """A handful of returns is not a variance, and the floor is half an hour."""
    from decimal import Decimal

    n = MIN_RETURNS_PER_PERIOD          # one short of the floor once the
    times = [i * BAR_INTERVAL_NS for i in range(n)]   # first bar is consumed
    closes = [Decimal("100") + Decimal(i % 2) for i in range(n)]

    periods, refused = base_period_variances(times, closes)

    assert periods == [None]
    assert refused["insufficient_period_coverage"] == 1


def test_consecutive_periods_share_no_bar():
    """The targets are non-overlapping, which is the whole IID claim."""
    from decimal import Decimal

    times = [i * BAR_INTERVAL_NS for i in range(SHORT_BARS * 3)]
    closes = [Decimal("100") + Decimal(i % 2) for i in range(SHORT_BARS * 3)]

    periods, _ = base_period_variances(times, closes)
    clean = [p for p in periods if p is not None]

    assert len(clean) == 3
    for earlier, later in zip(clean, clean[1:]):
        assert earlier.end_ns == later.start_ns, "periods abut, never overlap"


# --- the cascade refuses windows with holes -------------------------------

def test_a_hole_anywhere_in_the_window_drops_the_observation():
    """An average over a 24-hour window missing an hour is not a 24-hour average.

    Computing it anyway is how a tape's holes enter the regressors with nothing
    marking them - the flattering direction, because a shorter window is
    smoother and a smoother regressor fits better.
    """
    periods = [_period(i, 1e-4) for i in range(LONG_PERIODS + 2)]
    with_hole = list(periods)
    with_hole[3] = None

    full_design, _, _, full_skipped = cascade_observations(periods)
    holed_design, _, _, holed_skipped = cascade_observations(with_hole)

    assert full_skipped == 0
    assert len(full_design) == 2
    assert holed_skipped == 2, "both windows containing the hole are dropped"
    assert holed_design == []


def test_the_naive_benchmark_is_the_short_term_not_the_previous_row():
    """The random walk's claim is about time, not about list position.

    With a hole between two surviving observations, the previous ROW is not the
    previous HOUR, and deriving the benchmark from the target list would compare
    HAR against a random walk that skipped a gap it never had to forecast across.
    """
    periods = [_period(i, 1e-4 * (i + 1)) for i in range(LONG_PERIODS + 6)]
    periods[LONG_PERIODS + 1] = None

    design, targets, naive, _ = cascade_observations(periods)

    assert len(design) == len(targets) == len(naive)
    for row, benchmark in zip(design, naive):
        assert row[0] == benchmark, "the benchmark is this period's own variance"


# --- the fit never sees its own target ------------------------------------

def test_each_forecast_comes_from_a_fit_that_excluded_it():
    """Reproduce one forecast from scratch and compare.

    The check that matters most in the module: a walk-forward that leaked would
    still produce forecasts, still produce coefficients, and would beat the
    random walk every time - which reads as success.
    """
    variances = _cascade_series(80, b_short=0.2, b_medium=0.3, b_long=0.5)
    periods = [_period(i, v) for i, v in enumerate(variances)]
    design, targets, _naive, _ = cascade_observations(periods)

    forecasts, indices, _negative, _coefficients = walk_forward_forecasts(
        design, targets)

    assert indices, "the series is long enough to produce forecasts"
    i = indices[0]
    past = np.column_stack([np.ones(i), np.asarray(design[:i], dtype=float)])
    independent, *_ = np.linalg.lstsq(past, np.asarray(targets[:i], dtype=float),
                                      rcond=None)
    expected = float(independent[0] + independent[1:] @ np.asarray(design[i]))

    assert forecasts[0] == pytest.approx(max(expected, 0.0), rel=1e-9)


def test_a_future_target_cannot_move_an_earlier_forecast():
    """Corrupt the last target beyond recognition; every earlier forecast holds.

    A leak through the design matrix would not change the shape of the output,
    only its values, and only for the better.
    """
    variances = _cascade_series(80, b_short=0.2, b_medium=0.3, b_long=0.5)
    periods = [_period(i, v) for i, v in enumerate(variances)]
    design, targets, _naive, _ = cascade_observations(periods)

    clean, indices, _n, _c = walk_forward_forecasts(design, targets)
    poisoned_targets = list(targets)
    poisoned_targets[-1] *= 1e6
    poisoned, poisoned_indices, _n, _c = walk_forward_forecasts(
        design, poisoned_targets)

    assert indices == poisoned_indices
    assert clean[:-1] == pytest.approx(poisoned[:-1], rel=1e-12)


def test_a_rank_deficient_design_is_refused_not_least_normed():
    """Three identical regressors. `lstsq` answers anyway; the module must not.

    A least-norm solution to an underdetermined system is a confident-looking
    set of coefficients for a cascade that has collapsed to one number.
    """
    design = [(v, v, v) for v in np.linspace(1e-4, 2e-4, MIN_FIT_OBSERVATIONS + 5)]
    targets = [v * 1.1 for v, _, _ in design]

    forecasts, indices, _negative, coefficients = walk_forward_forecasts(
        design, targets)

    assert forecasts == [] and indices == []
    assert coefficients is None


def test_negative_forecasts_are_clamped_and_counted():
    """OLS is unconstrained; a negative variance is impossible, not merely small.

    Clamping without counting would hide a fit extrapolating outside its data.
    """
    # Targets that FALL as the cascade rises, over regressors ordered low to
    # high, so the points being forecast sit at the top of the range and the
    # fitted line runs below zero there. The three regressors carry independent
    # noise so the design stays full rank - a collinear one would be refused by
    # `_fit_coefficients` first and this test would pass for the wrong reason.
    rng = np.random.default_rng(3)
    n = MIN_FIT_OBSERVATIONS + 25
    shorts = np.linspace(1e-5, 6e-4, n)
    mediums = shorts * 0.8 + rng.normal(0.0, 5e-5, n)
    longs = shorts * 0.6 + rng.normal(0.0, 5e-5, n)
    design = list(zip(shorts, mediums, longs))
    targets = list(1.5e-4 - 0.5 * shorts - 0.3 * mediums
                   + rng.normal(0.0, 1e-6, n))

    forecasts, _indices, negative, _coefficients = walk_forward_forecasts(
        design, targets)

    assert negative > 0, "this design does extrapolate below zero"
    assert min(forecasts) >= 0.0, "no negative variance survives into the output"


# --- constants are pinned to the machinery they mirror --------------------

def test_the_evaluation_floor_matches_the_test_that_enforces_it():
    """Two floors for one requirement drift, and the drift is silent."""
    assert MIN_EVALUATION_OBSERVATIONS == naive_baseline.MIN_OBSERVATIONS


def test_the_cascade_ratios_are_declared_not_searched():
    """1 : 6 : 24 - an hour, a shift, a day. Pinned so a change is a decision."""
    assert SHORT_BARS * BAR_INTERVAL_NS == 3_600_000_000_000
    assert (MEDIUM_PERIODS, LONG_PERIODS) == (6, 24)


# --- end to end, through the store ----------------------------------------

def _bars_from_variances(variances, symbol="BTCUSDT", venue="binance",
                         start_ns=10**15):
    """Bars whose per-hour realized variance is exactly the series given.

    Each period's returns alternate sign at a constant magnitude, so the sum of
    squares is the target variance to floating-point and the price does not
    wander off across ninety hours. Building the bars from the variance rather
    than measuring the variance of arbitrary bars is what makes the expected
    result knowable independently of the module.
    """
    rows = []
    price = 100.0
    bar = 0
    for slot, variance in enumerate(variances):
        # The first slot loses one return to the series' opening bar; every
        # other slot gets a full complement.
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


def _write(tmp_path, rows, snapshot="har-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def test_a_real_cascade_beats_the_random_walk_and_says_so(tmp_path):
    """A mean-reverting cascade is exactly what the random walk is bad at.

    This is the only test in the file that can report success, and it exists so
    that the refusal tests are not the whole story: a module that refuses
    everything passes every negative test ever written.
    """
    variances = _cascade_series(120, b_short=0.15, b_medium=0.25, b_long=0.60,
                                seed=11)
    rows = _bars_from_variances(variances)
    table = compute_har_rv(_write(tmp_path, rows), _as_of(rows))

    assert len(table.rows) == 1, table.refused
    row = table.rows.iloc[0]
    assert bool(row["beats_naive"]), row.to_dict()
    assert row["mean_loss_har"] < row["mean_loss_naive"]
    assert row["n_evaluated"] >= MIN_EVALUATION_OBSERVATIONS
    assert row["negative_forecasts"] == 0
    # FE-001: every feature value carries how old its inputs were.
    assert row["age_ns"] is not None and row["freshness"] is not None


def test_a_random_walk_in_variance_is_reported_as_not_beaten(tmp_path):
    """When there is no cascade, the honest answer is that there is no cascade.

    Reported as a row with `beats_naive` False rather than suppressed: the
    literature this module cites records no universal winner, so "HAR bought
    nothing here" is a result, and a table that only ever contains wins is a
    table nobody can use to decide anything.
    """
    rng = np.random.default_rng(5)
    variances, level = [], 1e-4
    for _ in range(120):
        level = max(level * float(rng.lognormal(0.0, 0.30)), 1e-9)
        variances.append(level)
    rows = _bars_from_variances(variances, symbol="ETHUSDT")
    table = compute_har_rv(_write(tmp_path, rows), _as_of(rows))

    assert len(table.rows) == 1, table.refused
    row = table.rows.iloc[0]
    assert not bool(row["beats_naive"])
    assert row["p_value"] > 0.05


def test_too_little_history_is_refused_rather_than_fitted(tmp_path):
    """A cascade needs a day behind it before its first observation exists."""
    rows = _bars_from_variances([1e-4] * (LONG_PERIODS - 2), symbol="SOLUSDT")
    table = compute_har_rv(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["too_few_bars"] == 1


def test_a_zero_close_refuses_the_symbol_rather_than_dropping_the_bar(tmp_path):
    """The placeholder-price defect, which this store has actually shipped.

    Dropping the bad bar would shrink a period around the defect and still
    report a variance for it, carrying no sign of what was missing.
    """
    rows = _bars_from_variances([1e-4] * (LONG_PERIODS + 40), symbol="BTCUSDT")
    rows[len(rows) // 2]["close"] = 0.0
    table = compute_har_rv(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["non_positive_close"] == 1


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    """No input is not a refusal - there was nothing to refuse."""
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    table = compute_har_rv(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}


def test_no_in_sample_inference_is_reported(tmp_path):
    """The omission is deliberate and worth defending against a future addition.

    HAR's regressors overlap and are strongly autocorrelated; an in-sample
    t-statistic or R² over them is inflated, and it is the easiest impressive
    number in this literature to produce.
    """
    forbidden = {"r_squared", "t_statistic", "in_sample_p_value", "adjusted_r2"}
    assert forbidden.isdisjoint(har_rv._COLUMNS)
