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
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from store.parquet_partition import read_dataset
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, SYMBOL, VENUE

if TYPE_CHECKING:                    # import-time cycle: validation reads the store
    from validation.holdout_custodian import HoldoutCustodian


class ClockGatedReader:
    """Serves rows whose availability time has arrived, newest version per event."""

    def __init__(self, store_root: Path, dataset: str,
                 custodian: "HoldoutCustodian | None" = None) -> None:
        self._store_root = Path(store_root)
        self._dataset = dataset
        # The holdout custodian, when supplied, gets to refuse the read before any
        # data is touched. It hangs off the reader rather than the caller because
        # this is the one door every market-data read goes through, and a guard on
        # any other door is a guard with a way around it.
        #
        # Optional, and that is a real gap rather than a covered case: a reader
        # built without a custodian is unguarded. See holdout_custodian's module
        # docstring - do not read the existence of this parameter as proof the
        # holdout cannot be read.
        self._custodian = custodian

    @property
    def is_guarded(self) -> bool:
        """Whether a holdout custodian can refuse reads through this reader.

        Public because callers for whom a custodian is *not* optional need to be
        able to say so at construction rather than discovering it when a sealed
        range is quietly read. `paper.market_replay` refuses an unguarded reader
        on exactly this.
        """
        return self._custodian is not None

    def read_as_of(self, sim_clock_ns: int,
                   symbols: Sequence[str] | None = None,
                   not_before_ns: int | None = None,
                   columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Everything knowable at `sim_clock_ns`, and nothing else.

        Inclusive at the boundary: a row available exactly at T is usable at T.
        Off by one in this comparison silently drops the newest bar on every read.

        `columns` narrows the read SIDEWAYS, for a caller that needs a count or a
        freshness stamp rather than the rows themselves. The four temporal keys
        are always read whether or not they were asked for, so the clock bound
        and correction resolution are unchanged - a narrower read is never a
        looser one. Measured 2026-08-19: the bars dataset holds 168,639 parquet
        fragments, and three status probes were each materialising every column
        of all of them to compute a row count.

        `not_before_ns` narrows the read from BELOW, for a caller that already
        holds everything older and does not want to materialise it again. It is a
        bound on availability time, so a correction to an old bar - which carries
        a later availability time by definition - still arrives. Default None
        reads exactly what this method read before the parameter existed.

        The clock still wins: a lower bound can never reveal a row that is not
        yet available, because the upper bound is applied regardless.

        Raises `HoldoutSealed` when a custodian is attached and the clock is inside
        its sealed range - checked before reading, so a refused query never loads
        the rows it was refused.
        """
        if self._custodian is not None:
            self._custodian.assert_readable(int(sim_clock_ns))
        # **The four temporal keys ride along whatever the caller asked for.**
        # Without them the clock bound cannot be applied and a correction cannot
        # be resolved, so a projection that omitted one would return rows this
        # reader exists to filter out. Asking for fewer columns must never mean
        # asking for weaker as-of semantics.
        wanted = None
        if columns is not None:
            wanted = list(dict.fromkeys(
                [*columns, AVAILABILITY_TIME, EVENT_TIME, SYMBOL, VENUE]))
        frame = read_dataset(self._store_root, self._dataset,
                             not_before_ns=not_before_ns, columns=wanted)
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
        # VENUE is in the sort key for the same reason it is in the dedupe key:
        # without it, two venues tying on (symbol, event_time) come back in
        # parquet discovery order, which depends on the filesystem and on the
        # snapshot ids a build happened to produce. A backtest whose row order
        # varies from machine to machine is not reproducible, which is the whole
        # claim of having one reader.
        return latest.sort_values([SYMBOL, VENUE, EVENT_TIME]).reset_index(drop=True)


def _join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str] | str:
    """Symbol and venue when both frames name a venue; symbol alone when they do not.

    Symbol alone is not an identity - `read_as_of` dedupes on (symbol, venue,
    event_time) for exactly this reason, and the same reasoning has to reach the
    join fifteen lines below it. Keyed on symbol alone, a binance signal matched
    a hyperliquid reference price whenever both venues carried a row at the same
    availability time. That is not a row from the future; it is a price from a
    market the strategy is not trading, and it reads as a perfectly plausible
    number.

    The fallback is deliberate and it is a real weakening, stated here rather
    than left to be discovered: a frame with no venue column carries no way to
    tell venues apart, so the join can only match on symbol and a caller who
    later adds the column will get stricter matching than they had. `merge_asof`
    would raise on a `by` column present in only one frame, so both frames must
    name it for the strict form to apply.
    """
    if VENUE in left.columns and VENUE in right.columns:
        return [SYMBOL, VENUE]
    return SYMBOL


def join_as_of(left: pd.DataFrame, right: pd.DataFrame, suffix: str,
               tolerance_ns: int | None = None) -> pd.DataFrame:
    """Attach the most recent right-hand row that was already available.

    `direction="backward"` is what makes this safe: it can only reach into the
    past. A forward or nearest join reaches into the future by construction.

    Matching is keyed on symbol AND venue whenever both frames carry a venue
    column, falling back to symbol alone when they do not - see `_join_keys` for
    what the fallback costs.

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
        by=_join_keys(left_sorted, right_sorted),
        direction="backward",
        tolerance=tolerance_ns,
        suffixes=("", suffix),
    ).reset_index(drop=True)
