"""Every feature value says how old its inputs were, judged against their cadence.

The goal spec names the failure this defends against in one line: *"the failure
mode this standard is designed against is not stupidity. It is confident
staleness - a system that learned something true, never noticed it stopped being
true, and keeps betting on it."*

The tests that matter most here are the two about the THIRD state. A series with
no measurable cadence is not fresh, and folding it into `False` is how "nobody
measured this" becomes "this is fine".
"""
import pandas as pd
import pytest

from features.staleness import (
    CADENCE_SAMPLE, COLUMNS, FRESH, STALE, STALE_MULTIPLE, UNKNOWN_CADENCE,
    measure_staleness, stamp,
)

SECOND = 1_000_000_000
NOW = 1_786_344_000_000_000_000


def _every(gap_ns, count, end_ns=NOW):
    """A series arriving on a fixed cadence, newest last."""
    return [end_ns - i * gap_ns for i in range(count - 1, -1, -1)]


# --- the measurement ------------------------------------------------------

def test_a_series_arriving_on_time_is_fresh():
    got = measure_staleness(_every(60 * SECOND, 30), NOW)

    assert got.verdict == FRESH
    assert got.age_ns == 0
    assert got.routine_gap_ns == 60 * SECOND


def test_a_series_older_than_its_own_cadence_allows_is_stale():
    """Three times its routine gap - the same multiple `venue_recorder` uses to
    decide a subscribed stream has stopped speaking. One question, one number."""
    times = _every(60 * SECOND, 30, end_ns=NOW - 200 * SECOND)

    got = measure_staleness(times, NOW)

    assert got.verdict == STALE
    assert got.age_ns == 200 * SECOND
    assert STALE_MULTIPLE == 3.0


def test_the_bound_is_the_series_own_cadence_not_a_constant():
    """A five-minute feed and a one-minute feed are both healthy at four
    minutes old, and one shared limit calls exactly one of them wrong."""
    four_minutes = 240 * SECOND
    slow = measure_staleness(_every(300 * SECOND, 30, end_ns=NOW - four_minutes), NOW)
    fast = measure_staleness(_every(60 * SECOND, 30, end_ns=NOW - four_minutes), NOW)

    assert slow.verdict == FRESH
    assert fast.verdict == STALE


# --- the third state ------------------------------------------------------

def test_one_observation_has_shown_no_cadence_and_is_not_called_fresh():
    """The state that would otherwise be a lie. A series with one point has
    demonstrated nothing, and reporting FRESH asserts something nobody
    measured."""
    got = measure_staleness([NOW - 10 * SECOND], NOW)

    assert got.verdict == UNKNOWN_CADENCE
    assert got.routine_gap_ns is None
    assert got.is_stale is False       # not stale, and not fresh either


def test_observations_sharing_one_timestamp_have_no_cadence_either():
    """A venue batching its updates under one clock. Dividing by that gap would
    call everything fresh forever."""
    got = measure_staleness([NOW] * 10, NOW)

    assert got.verdict == UNKNOWN_CADENCE
    assert got.routine_gap_ns is None


def test_no_observations_at_all_raises_rather_than_stamping_zero():
    """A feature value with no input is not fresh, it is absent - and a zero
    age on nothing is the most flattering answer available."""
    with pytest.raises(ValueError, match="no observations"):
        measure_staleness([], NOW)


# --- what it reads --------------------------------------------------------

def test_the_newest_observation_is_taken_by_value_not_by_position():
    """A caller handing over an unsorted frame must not be told the wrong age."""
    shuffled = [NOW - 5 * SECOND, NOW, NOW - 60 * SECOND, NOW - 30 * SECOND]

    assert measure_staleness(shuffled, NOW).age_ns == 0


def test_only_the_recent_window_defines_routine():
    """A cadence change an hour ago is what should be measured, not
    yesterday's. Older gaps outside the sample do not drag the median."""
    old = [NOW - 3600 * SECOND - i * 600 * SECOND for i in range(40, 0, -1)]
    recent = _every(60 * SECOND, CADENCE_SAMPLE)

    got = measure_staleness(old + recent, NOW)

    assert got.routine_gap_ns == 60 * SECOND


# --- the stamp on a frame -------------------------------------------------

def test_a_frame_carries_the_whole_contract_in_named_columns():
    rows = pd.DataFrame({"venue": ["binance"], "symbol": ["BTCUSDT"],
                         "value": [1.0]})
    ages = {("binance", "BTCUSDT"): measure_staleness(_every(60 * SECOND, 30), NOW)}

    stamped = stamp(rows, ages, ["venue", "symbol"])

    assert set(COLUMNS) <= set(stamped.columns)
    assert stamped.iloc[0]["freshness"] == FRESH
    assert stamped.iloc[0]["as_of_ns"] == NOW


def test_a_row_nobody_measured_keeps_its_value_and_says_nothing_about_its_age():
    """Dropping it would be the same class of loss this module exists to
    prevent - a value disappearing because its age was not measured."""
    rows = pd.DataFrame({"venue": ["binance", "kraken"],
                         "symbol": ["BTCUSDT", "BTCUSD"], "value": [1.0, 2.0]})
    ages = {("binance", "BTCUSDT"): measure_staleness(_every(60 * SECOND, 30), NOW)}

    stamped = stamp(rows, ages, ["venue", "symbol"])

    assert len(stamped) == 2
    unmeasured = stamped[stamped["venue"] == "kraken"].iloc[0]
    assert unmeasured["value"] == 2.0
    # Absent, however pandas spells absent in a mixed column - the point is
    # that nothing was asserted about this row's age, not which null it is.
    assert pd.isna(unmeasured["freshness"])
    assert pd.isna(unmeasured["age_ns"])


def test_stamping_does_not_mutate_the_frame_it_was_given():
    rows = pd.DataFrame({"venue": ["binance"], "symbol": ["BTCUSDT"], "value": [1.0]})

    stamp(rows, {}, ["venue", "symbol"])

    assert list(rows.columns) == ["venue", "symbol", "value"]


def test_an_empty_frame_still_carries_the_columns():
    """So a consumer reading `freshness` does not fail differently on a quiet
    market than on a busy one."""
    empty = pd.DataFrame({"venue": [], "symbol": [], "value": []})

    stamped = stamp(empty, {}, ["venue", "symbol"])

    assert set(COLUMNS) <= set(stamped.columns)
    assert stamped.empty
