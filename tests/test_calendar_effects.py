"""A clock reading is trivial; the claim that the clock matters is not.

The failures defended here are the ones that make a calendar feature look
informative when it is not: an integer hour that puts 23 and 00 a day apart, a
funding-hour effect measured on a venue where every hour is a funding hour, a
significance test that shuffles clustered returns and therefore finds everything
significant, and a p-value reported to more precision than the eight distinct
alignments of a 3-per-day schedule can carry.
"""
import math

import pandas as pd
import pytest

from features.calendar_effects import (
    MIN_BUCKET_OBSERVATIONS,
    NoSettlementSchedule,
    alignment_permutation_p_value,
    compute_calendar_effects,
    cyclical,
    day_of_week,
    distinct_alignments,
    hour_of_day_utc,
    hours_to_next_settlement,
    min_achievable_p_value,
    variance_ratio,
)
from features.realized_volatility import BAR_INTERVAL_NS
from store.parquet_partition import append_partition

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR

# Binance's real settlement schedule, and the one whose 8-hour periodicity makes
# the alignment count 8 rather than 24.
_BINANCE_HOURS = (0, 8, 16)


# --- the clock arithmetic, checked against an independent path ------------

def test_the_hour_matches_python_s_own_calendar():
    """Integer arithmetic over int64 nanoseconds, cross-checked with datetime -
    a path that shares no code with the module under test."""
    import datetime as dt

    for offset_hours in (0, 1, 7, 8, 15, 16, 23, 24, 25, 8_760):
        ns = offset_hours * _NS_PER_HOUR
        expected = dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc)
        assert hour_of_day_utc(ns) == expected.hour
        assert day_of_week(ns) == expected.weekday()


def test_the_epoch_was_a_thursday():
    """The one constant in the module that a wrong value would shift silently."""
    assert day_of_week(0) == 3


def test_the_last_hour_of_the_day_is_adjacent_to_the_first():
    """The whole reason the encoding is a pair. As integers 23 and 0 are 23
    apart; on the circle they are one step apart, and every distance-based model
    handed the integer learns a midnight discontinuity that does not exist."""
    def distance(a, b):
        return math.dist(cyclical(a, 24), cyclical(b, 24))

    assert distance(23, 0) == pytest.approx(distance(11, 12))
    assert distance(23, 0) < distance(23, 12)


# --- the settlement clock -------------------------------------------------

def test_time_to_the_next_settlement_wraps_past_midnight():
    """From 20:00 UTC on Binance the next settlement is 00:00, four hours out -
    not a negative number and not tomorrow's 08:00."""
    assert hours_to_next_settlement("binance", 20 * _NS_PER_HOUR) == pytest.approx(4.0)
    assert hours_to_next_settlement("binance", 1 * _NS_PER_HOUR) == pytest.approx(7.0)
    assert hours_to_next_settlement("binance", 8 * _NS_PER_HOUR) == pytest.approx(8.0)


def test_a_venue_without_a_schedule_refuses_rather_than_defaulting():
    """Assuming 8-hourly for an unknown venue is the mistake the schedule
    exists to prevent, and it would be silent."""
    with pytest.raises(NoSettlementSchedule):
        hours_to_next_settlement("okx", 0)
    with pytest.raises(NoSettlementSchedule):
        hours_to_next_settlement("binance-spot", 0)


# --- the effect, and the test of it ---------------------------------------

def _buckets(inside_hours, inside_value, outside_value, n=200):
    """Squared returns bucketed by hour: one value inside the window, another
    outside, so the ratio is known exactly."""
    return {hour: [inside_value if hour in inside_hours else outside_value] * n
            for hour in range(24)}


def test_the_ratio_is_inside_over_outside():
    buckets = _buckets({0, 8, 16}, 4.0, 1.0)
    ratio, inside, outside = variance_ratio(buckets, {0, 8, 16})

    assert ratio == pytest.approx(4.0)
    assert inside == 3 * 200
    assert outside == 21 * 200


def test_a_thin_bucket_refuses_rather_than_reporting_a_ratio():
    """A settlement window with 800 observations against 12 outside it is as
    unusable as the reverse, and the ratio between them looks the same."""
    buckets = {0: [1.0] * (MIN_BUCKET_OBSERVATIONS - 1)}
    buckets.update({h: [1.0] * 200 for h in range(1, 24)})

    assert variance_ratio(buckets, {0}) is None


def test_a_genuine_alignment_beats_every_shifted_one():
    """00/08/16 really are the loud hours here, so no other alignment matches."""
    buckets = _buckets({0, 8, 16}, 9.0, 1.0)
    ratio, _, _ = variance_ratio(buckets, {0, 8, 16})

    p_value = alignment_permutation_p_value(buckets, (0, 8, 16), ratio)

    assert p_value == pytest.approx(min_achievable_p_value(_BINANCE_HOURS))


def test_a_flat_tape_gives_the_least_significant_answer_available():
    """Every hour identical: every alignment is exactly as unusual as the real
    one, so the test must return 1.0 rather than a small number."""
    buckets = {hour: [1.0] * 200 for hour in range(24)}
    ratio, _, _ = variance_ratio(buckets, {0, 8, 16})

    assert ratio == pytest.approx(1.0)
    assert alignment_permutation_p_value(buckets, (0, 8, 16), ratio) == 1.0


def test_a_suspiciously_calm_settlement_hour_is_detected_too():
    """The comparison is on absolute deviation from 1.0. A one-sided test would
    report nothing here, and 'settlement hours are unusually quiet' is a finding
    about predictable flow just as much as its opposite."""
    buckets = _buckets({0, 8, 16}, 0.1, 1.0)
    ratio, _, _ = variance_ratio(buckets, {0, 8, 16})

    assert ratio < 1.0
    assert alignment_permutation_p_value(buckets, (0, 8, 16), ratio) == pytest.approx(
        min_achievable_p_value(_BINANCE_HOURS))


def test_a_periodic_schedule_has_fewer_alignments_than_offsets():
    """00/08/16 has period 8, so shifting by 8 or 16 reproduces it exactly.

    Enumerating all 23 non-zero offsets would compare the observation against a
    distribution holding three copies of the observation, dragging every p-value
    toward the floor - significance manufactured by double-counting.
    """
    alignments = distinct_alignments(_BINANCE_HOURS)

    assert len(alignments) == 7, "eight distinct alignments, one of them the real one"
    assert frozenset(_BINANCE_HOURS) not in alignments


def test_the_resolution_floor_is_a_property_of_the_schedule():
    """Reported on every row rather than assumed. It is not improved by more
    data: more bars sharpen the ratio and leave the alignment count at eight."""
    assert min_achievable_p_value(_BINANCE_HOURS) == pytest.approx(1 / 8)
    # An hourly schedule has no distinct alignment at all - which is why that
    # venue is refused before this test is ever reached.
    assert min_achievable_p_value(tuple(range(24))) == pytest.approx(1.0)


# --- end to end -----------------------------------------------------------

def _bars(symbol, venue, n_hours, loud_hours, *, loud=8e-4, quiet=1e-4,
          start_ns=0):
    """One bar a minute for `n_hours`, moving more inside `loud_hours`.

    Alternating sign at a fixed magnitude so the per-hour mean squared return is
    exactly the magnitude squared and the expected ratio is knowable without
    running the module.
    """
    rows, price = [], 100.0
    for bar in range(n_hours * 60):
        t = start_ns + bar * BAR_INTERVAL_NS
        step = loud if hour_of_day_utc(t) in loud_hours else quiet
        price *= math.exp(step if bar % 2 else -step)
        rows.append({
            "venue": venue, "symbol": symbol,
            "open": price, "high": price, "low": price, "close": price,
            "volume": 1.0, "trades": 10,
            "event_time_ns": t, "ingestion_time_ns": t,
            "availability_time_ns": t,
        })
    return rows


def _write(tmp_path, rows, snapshot="calendar-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def test_a_real_funding_hour_effect_is_found_and_scored(tmp_path):
    """The only test here that can report an effect, so the refusals are not
    the whole story - a module that finds nothing passes every negative test."""
    rows = _bars("BTCUSDT", "binance", n_hours=96, loud_hours={0, 8, 16})
    table = compute_calendar_effects(_write(tmp_path, rows), _as_of(rows))

    assert len(table.rows) == 1, table.refused
    row = table.rows.iloc[0]
    # 8e-4 against 1e-4, squared: a ratio of 64 before boundary bars soften it.
    assert row["funding_hour_variance_ratio"] > 10
    assert row["p_value"] == pytest.approx(min_achievable_p_value(_BINANCE_HOURS))
    assert row["settlement_hours_per_day"] == 3
    assert row["observations_in_window"] >= MIN_BUCKET_OBSERVATIONS
    assert row["age_ns"] is not None            # FE-001


def test_a_tape_with_no_hour_effect_is_reported_as_having_none(tmp_path):
    """Not suppressed. 'This instrument has no funding-hour effect' is the
    answer for most instruments and a consumer needs to be able to read it."""
    rows = _bars("ETHUSDT", "binance", n_hours=96, loud_hours=set())
    table = compute_calendar_effects(_write(tmp_path, rows), _as_of(rows))

    row = table.rows.iloc[0]
    assert row["funding_hour_variance_ratio"] == pytest.approx(1.0, rel=0.05)
    assert row["p_value"] > 0.5


def test_a_venue_settling_every_hour_is_refused_not_scored(tmp_path):
    """Hyperliquid funds hourly: every hour is a funding hour, the ratio is 1.0
    by construction, and a p-value over it is arithmetic on a tautology. The
    naive implementation returns a tidy 1.00 and nothing about it looks empty.
    """
    rows = _bars("BTC", "hyperliquid", n_hours=96, loud_hours={0, 8, 16})
    table = compute_calendar_effects(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["no_contrast_in_schedule"] == 1


def test_spot_is_refused_for_the_same_reason(tmp_path):
    """No settlements at all is the other end of the same absence."""
    rows = _bars("BTCUSDT", "binance-spot", n_hours=96, loud_hours={0, 8, 16})
    table = compute_calendar_effects(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["no_contrast_in_schedule"] == 1


def test_an_unknown_venue_is_refused_rather_than_given_a_schedule(tmp_path):
    rows = _bars("BTCUSDT", "okx", n_hours=96, loud_hours={0, 8, 16})
    table = compute_calendar_effects(_write(tmp_path, rows), _as_of(rows))

    assert table.rows.empty
    assert table.refused["unknown_settlement_schedule"] == 1


def test_a_gap_in_the_tape_produces_no_return_across_it(tmp_path):
    """A return computed across a five-day hole is not a return, and it lands in
    whatever hour the tape happened to resume in - inventing an effect at an
    arbitrary hour, which is exactly the shape of finding this module claims to
    detect.
    """
    early = _bars("SOLUSDT", "binance", n_hours=48, loud_hours=set(),
                  start_ns=0)
    # Resume five days later at 03:00 UTC, at a wildly different price.
    late = _bars("SOLUSDT", "binance", n_hours=48, loud_hours=set(),
                 start_ns=5 * _NS_PER_DAY + 3 * _NS_PER_HOUR)
    for row in late:
        for column in ("open", "high", "low", "close"):
            row[column] *= 3.0

    table = compute_calendar_effects(_write(tmp_path, early + late),
                                     _as_of(late))

    row = table.rows.iloc[0]
    # The bridging return would have been ln(3) ~ 1.1, squared ~ 1.2 - roughly
    # 100 million times the 1e-8 of an ordinary bar here, and enough on its own
    # to make hour 3 the loudest hour of the day.
    assert row["funding_hour_variance_ratio"] == pytest.approx(1.0, rel=0.05)


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    table = compute_calendar_effects(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}
