"""The three timestamps every stored row carries, and what must be true of them.

`event_time_ns`  - when the venue says it happened.
`ingestion_time_ns` - when this machine received it.
`availability_time_ns` - the earliest moment a decision could have used it.

The third is the only one any read path may filter on, and it is the whole point
of the layer. Event time is what a naive backtest joins on, and joining on it
means a bar built from a frame that arrived 32 seconds late appears to have been
available 32 seconds before it existed. That produces an excellent backtest and a
broken system, and nothing in the result looks wrong.

Only `availability >= ingestion` is enforced. Event time is deliberately
unconstrained: venue clocks drift ahead of ours, and rejecting or clamping that
would rewrite the venue's own account of when a trade happened.
"""
from __future__ import annotations

import pandas as pd

SYMBOL = "symbol"
VENUE = "venue"
EVENT_TIME = "event_time_ns"
INGESTION_TIME = "ingestion_time_ns"
AVAILABILITY_TIME = "availability_time_ns"

REQUIRED_COLUMNS = (SYMBOL, VENUE, EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME)

# event_time may be absent when a venue sends no timestamp, so it is the one
# nullable member of the trio.
_NON_NULLABLE = (INGESTION_TIME, AVAILABILITY_TIME)
_INTEGER_DTYPES = {"int64", "Int64"}


class TemporalInvariantError(ValueError):
    """A frame violates the contract that makes clock-gating trustworthy."""


def validate_temporal_frame(frame: pd.DataFrame) -> None:
    """Raise unless every row can be safely clock-gated."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise TemporalInvariantError(f"missing required column(s): {missing}")

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME):
        if str(frame[column].dtype) not in _INTEGER_DTYPES:
            raise TemporalInvariantError(
                f"{column} must be int64 nanoseconds, got {frame[column].dtype}. "
                f"Float nanoseconds lose precision above 2^53 and silently reorder events")

    for column in _NON_NULLABLE:
        if frame[column].isna().any():
            raise TemporalInvariantError(f"{column} contains null values")

    if frame.empty:
        return

    too_early = frame[AVAILABILITY_TIME] < frame[INGESTION_TIME]
    if bool(too_early.any()):
        first = frame.loc[too_early].iloc[0]
        raise TemporalInvariantError(
            f"{AVAILABILITY_TIME} is before {INGESTION_TIME} on {int(too_early.sum())} row(s) "
            f"(first: available {first[AVAILABILITY_TIME]}, ingested {first[INGESTION_TIME]}) - "
            f"a row cannot have been usable before it arrived")
