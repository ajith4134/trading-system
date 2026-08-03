"""The schema exists to make one class of bug impossible, so these tests try to commit it."""
from __future__ import annotations

import pandas as pd
import pytest

from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, REQUIRED_COLUMNS, SYMBOL, VENUE,
    TemporalInvariantError, validate_temporal_frame,
)


def _frame(**overrides) -> pd.DataFrame:
    base = {
        SYMBOL: ["BTCUSDT"],
        VENUE: ["binance"],
        EVENT_TIME: [1_000],
        INGESTION_TIME: [1_100],
        AVAILABILITY_TIME: [1_100],
    }
    base.update(overrides)
    return pd.DataFrame(base).astype({c: "int64" for c in base if c.endswith("_ns")})


def test_a_well_formed_frame_passes():
    validate_temporal_frame(_frame())


def test_missing_availability_time_is_refused():
    frame = _frame().drop(columns=[AVAILABILITY_TIME])
    with pytest.raises(TemporalInvariantError, match=AVAILABILITY_TIME):
        validate_temporal_frame(frame)


def test_null_availability_time_is_refused():
    """A row with no availability time cannot be clock-gated, so it cannot be stored."""
    frame = _frame()
    frame[AVAILABILITY_TIME] = pd.Series([pd.NA], dtype="Int64")
    with pytest.raises(TemporalInvariantError, match="null"):
        validate_temporal_frame(frame)


def test_availability_earlier_than_ingestion_is_refused():
    """Claiming data was usable before it arrived is the leak this whole layer prevents."""
    with pytest.raises(TemporalInvariantError, match="before"):
        validate_temporal_frame(_frame(**{AVAILABILITY_TIME: [900], INGESTION_TIME: [1_100]}))


def test_event_time_after_ingestion_is_allowed():
    """Venue clocks drift ahead. That is recorded, not corrected.

    Enforcing event <= ingestion would silently rewrite the venue's own account of
    when a trade happened, which is data loss disguised as validation.
    """
    validate_temporal_frame(_frame(**{EVENT_TIME: [2_000], INGESTION_TIME: [1_100]}))


def test_null_event_time_is_allowed():
    """Some frames carry no venue timestamp; absence is recorded, not invented."""
    frame = _frame()
    frame[EVENT_TIME] = pd.Series([pd.NA], dtype="Int64")
    validate_temporal_frame(frame)


def test_float_timestamps_are_refused():
    """Float nanoseconds lose precision above 2^53 and silently reorder events."""
    frame = _frame()
    frame[AVAILABILITY_TIME] = frame[AVAILABILITY_TIME].astype("float64")
    with pytest.raises(TemporalInvariantError, match="int64"):
        validate_temporal_frame(frame)


def test_required_columns_are_the_documented_five():
    assert set(REQUIRED_COLUMNS) == {SYMBOL, VENUE, EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME}
