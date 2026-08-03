"""The only path to stored data, shared by backtest and live.

One reader for both, because two access paths is how an off-by-one in windowing
produces a great backtest and a broken system: the backtest reads one way, live
reads another, and the discrepancy is invisible until capital is behind it. Live
passes the wall clock as `sim_clock_ns`; a backtest passes its simulated clock.
Neither has any other door.

Filtering is on `availability_time_ns` and never on `event_time_ns`. Joining on
event time is the classic leak - it attaches a row to a moment before that row
existed - so `join_as_of` keys on availability time and there is no parameter to
change that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from store.parquet_partition import read_dataset
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, SYMBOL, VENUE


class ClockGatedReader:
    """Serves rows whose availability time has arrived, newest version per event."""

    def __init__(self, store_root: Path, dataset: str) -> None:
        self._store_root = Path(store_root)
        self._dataset = dataset

    def read_as_of(self, sim_clock_ns: int,
                   symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Everything knowable at `sim_clock_ns`, and nothing else.

        Inclusive at the boundary: a row available exactly at T is usable at T.
        Off by one in this comparison silently drops the newest bar on every read.
        """
        frame = read_dataset(self._store_root, self._dataset)
        if frame.empty:
            return frame

        visible = frame[frame[AVAILABILITY_TIME] <= int(sim_clock_ns)]
        if symbols is not None:
            visible = visible[visible[SYMBOL].isin(list(symbols))]
        if visible.empty:
            return visible.reset_index(drop=True)

        # A correction is a second row for the same (symbol, venue, event) with a
        # later availability time. Keeping both would double-count the bar;
        # keeping the first would ignore the correction. Sort ascending and take
        # the last visible version per key.
        #
        # The key includes VENUE, not just SYMBOL and EVENT_TIME: build_bars keys
        # bars by (symbol, venue, bar_open_ns), so two venues can legitimately
        # produce distinct bars carrying the same symbol and the same event time.
        # Grouping without venue would silently collapse them into one and drop a
        # venue's bar - data loss, not correction resolution. A correction
        # replaces the same venue's bar, never another venue's.
        #
        # drop_duplicates(keep="last"), not groupby(...).last(): groupby's .last()
        # defaults to skipna=True and picks the last non-null value per COLUMN
        # independently, not the last physical ROW. A correction that legitimately
        # carries a null in a non-key column would then be spliced together with a
        # stale value from an older row - a composite that was never stored.
        # drop_duplicates keeps whole rows, so the result is always exactly one row
        # that was actually written.
        ordered = visible.sort_values(AVAILABILITY_TIME, kind="mergesort")
        latest = ordered.drop_duplicates(subset=[SYMBOL, VENUE, EVENT_TIME], keep="last")
        return latest.sort_values([SYMBOL, EVENT_TIME]).reset_index(drop=True)


def join_as_of(left: pd.DataFrame, right: pd.DataFrame, suffix: str,
               tolerance_ns: int | None = None) -> pd.DataFrame:
    """Attach the most recent right-hand row that was already available.

    `direction="backward"` is what makes this safe: it can only reach into the
    past. A forward or nearest join reaches into the future by construction.

    `tolerance_ns` bounds how stale a match may be. Without it, a funding rate
    from an hour ago attaches to a signal now and reads as current context when
    it describes a different market.
    """
    if left.empty:
        return left.copy()

    left_sorted = left.sort_values(AVAILABILITY_TIME, kind="mergesort")
    if right.empty:
        return left_sorted.reset_index(drop=True)
    right_sorted = right.sort_values(AVAILABILITY_TIME, kind="mergesort")

    return pd.merge_asof(
        left_sorted,
        right_sorted,
        on=AVAILABILITY_TIME,
        by=SYMBOL,
        direction="backward",
        tolerance=tolerance_ns,
        suffixes=("", suffix),
    ).reset_index(drop=True)
