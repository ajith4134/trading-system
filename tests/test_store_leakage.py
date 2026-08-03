# tests/test_store_leakage.py
"""Adversarial tests: try to read the future out of the store, and fail.

Each test here encodes a real way look-ahead enters a backtest. They are written
as attacks rather than as behaviour checks because a leak is not a wrong answer -
it is a plausible one that is better than reality, and it will not look like a bug.
"""
from __future__ import annotations

import pandas as pd
import pytest

from store.clock_gated_reader import ClockGatedReader, join_as_of
from store.parquet_partition import append_partition
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE
from store.trade_bars import Trade, build_bars

MINUTE_NS = 60_000_000_000


def _store(tmp_path, frame: pd.DataFrame, snapshot: str):
    append_partition(tmp_path, "bars_1m", frame, snapshot)
    return ClockGatedReader(tmp_path, "bars_1m")


def test_a_bar_built_from_late_data_is_invisible_at_its_close(tmp_path):
    """The headline case, end to end.

    A hyperliquid reconnect delivered a trade 32 seconds after its bar closed.
    Anyone reading at the close must not see that bar - it did not exist yet.
    """
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    arrived_ns = close_ns + 32 * 1_000_000_000
    bars = build_bars(
        [Trade("BTC", "hyperliquid", 63123.0, 0.0002, open_ns + 21, arrived_ns)],
        MINUTE_NS)
    reader = _store(tmp_path, bars, "snap1")

    assert reader.read_as_of(close_ns).empty, "the bar was readable before it arrived"
    assert len(reader.read_as_of(arrived_ns)) == 1


def test_a_promptly_built_bar_is_visible_at_its_close(tmp_path):
    """The guard against over-correcting: gating must not hide ordinary data."""
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    bars = build_bars(
        [Trade("BTCUSDT", "binance", 63113.2, 0.001, open_ns + 5, open_ns + 70_000_000)],
        MINUTE_NS)
    assert len(_store(tmp_path, bars, "snap1").read_as_of(close_ns)) == 1


def test_no_read_can_return_a_row_from_its_own_future(tmp_path):
    """Swept over many clocks rather than one: an off-by-one hides at a single point.

    The assertion is deliberately NOT `availability_time <= clock`. That is a
    verbatim restatement of the filter predicate, so the reader satisfies it by
    construction and the test stays green even with `build_bars` reverted to
    stamping availability at the bar's close - the original leak, which makes a
    bar built from a trade that arrived two minutes late appear to have been
    usable two minutes before it existed. A self-consistent store can be
    consistently wrong.

    So the bound is recomputed from the trades themselves: a bar could not be
    known before its own close, nor before the last trade in it arrived. That
    quantity lives outside the store and does not move when the stored column
    does.
    """
    open_ns = 100 * MINUTE_NS
    trades = [
        Trade("BTCUSDT", "binance", 100.0, 1.0, open_ns + 5, open_ns + 70_000_000),
        Trade("BTCUSDT", "binance", 101.0, 1.0, open_ns + MINUTE_NS + 5,
              open_ns + MINUTE_NS + 80_000_000),
        # The third bar holds a prompt trade and one that took two minutes of
        # reconnect backfill to land. The bar closes at open+3m; it was not
        # knowable until open+4m. Two trades, not one, so that a build reading
        # the WRONG arrival out of the bar - the earliest rather than the latest -
        # produces a row that still satisfies availability >= ingestion and
        # therefore passes every schema check on the way in.
        Trade("BTCUSDT", "binance", 102.0, 1.0, open_ns + 2 * MINUTE_NS + 5,
              open_ns + 2 * MINUTE_NS + 70_000_000),
        Trade("BTCUSDT", "binance", 103.0, 1.0, open_ns + 2 * MINUTE_NS + 10,
              open_ns + 4 * MINUTE_NS),
    ]
    knowable_at: dict[int, int] = {}
    for trade in trades:
        bar_open = (trade.event_time_ns // MINUTE_NS) * MINUTE_NS
        knowable_at[bar_open] = max(knowable_at.get(bar_open, bar_open + MINUTE_NS),
                                    trade.ingestion_time_ns)

    reader = _store(tmp_path, build_bars(trades, MINUTE_NS), "snap1")
    for step in range(0, 5 * 60, 7):
        clock = open_ns + step * 1_000_000_000
        visible = reader.read_as_of(clock)
        for bar_open in visible[EVENT_TIME]:
            assert knowable_at[int(bar_open)] <= clock, (
                f"the bar opening at {int(bar_open)} was readable at clock {clock}, "
                f"but its own trades had not all arrived until "
                f"{knowable_at[int(bar_open)]}")


def test_a_correction_cannot_be_seen_before_it_was_made(tmp_path):
    """Late data revises a closed bar. The revision is not knowable in advance."""
    open_ns = 100 * MINUTE_NS
    close_ns = open_ns + MINUTE_NS
    prompt = build_bars(
        [Trade("BTC", "hyperliquid", 100.0, 1.0, open_ns + 5, open_ns + 300_000_000)],
        MINUTE_NS)
    revised = build_bars(
        [Trade("BTC", "hyperliquid", 100.0, 1.0, open_ns + 5, open_ns + 300_000_000),
         Trade("BTC", "hyperliquid", 999.0, 5.0, open_ns + 10, close_ns + 30_000_000_000)],
        MINUTE_NS)

    append_partition(tmp_path, "bars_1m", prompt, "snap1")
    append_partition(tmp_path, "bars_1m", revised, "snap2")
    reader = ClockGatedReader(tmp_path, "bars_1m")

    at_close = reader.read_as_of(close_ns)
    assert len(at_close) == 1
    assert at_close.iloc[0]["high"] == 100.0, "the revision leaked backwards"

    after = reader.read_as_of(close_ns + 60 * 1_000_000_000)
    assert len(after) == 1, "the correction duplicated the bar instead of replacing it"
    assert after.iloc[0]["high"] == 999.0


def test_joining_on_event_time_would_leak_and_the_api_will_not_do_it(tmp_path):
    """A funding row describing an old event that only arrived later.

    Keyed on event time it attaches to a signal an hour before it was published.
    Keyed on availability time it correctly attaches to nothing.
    """
    signal = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [1_000],
                           AVAILABILITY_TIME: [1_000], "signal": [1.0]})
    funding = pd.DataFrame({SYMBOL: ["BTCUSDT"], EVENT_TIME: [900],
                            AVAILABILITY_TIME: [50_000], "funding": [0.01]})
    joined = join_as_of(signal, funding, suffix="_f")
    assert pd.isna(joined.loc[0, "funding"])


def test_the_same_reader_serves_backtest_and_live_identically(tmp_path):
    """Two paths would diverge; this asserts there is only one.

    'Live' is genuinely driven by the wall clock through a callable, over data
    anchored to real time so that every bar is already in the past. 'Backtest' is
    the same reader handed an explicit simulated clock partway through that
    history, and it must see exactly the prefix that was available then.

    Two readers of one class compared against the same literal integer would be
    `assert f(x) == f(x)` and would hold against any implementation at all,
    including one that ignored the clock entirely. The failure this defends is
    the reader consulting the wall clock instead of the argument it was handed -
    a live path and a backtest path diverging inside one function - which shows
    up here as the backtest seeing bars its simulated clock says do not exist.
    """
    import time

    def wall_clock_ns() -> int:
        return time.time_ns()

    open_ns = ((wall_clock_ns() - 10 * MINUTE_NS) // MINUTE_NS) * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0 + i, 1.0,
                    open_ns + i * MINUTE_NS + 5,
                    open_ns + i * MINUTE_NS + 70_000_000) for i in range(5)]
    reader = _store(tmp_path, build_bars(trades, MINUTE_NS), "snap1")

    live = reader.read_as_of(wall_clock_ns())
    assert len(live) == 5, "the wall clock is past every bar, so live sees all of them"

    # Bars 0, 1 and 2 close at or before this instant; 3 and 4 have not happened.
    simulated_clock_ns = open_ns + 3 * MINUTE_NS
    backtest = reader.read_as_of(simulated_clock_ns)
    pd.testing.assert_frame_equal(backtest, live.iloc[:3].reset_index(drop=True))


def test_history_never_shrinks_as_the_clock_advances(tmp_path):
    """Monotonicity across a sweep. A row that was visible must stay visible."""
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0 + i, 1.0,
                    open_ns + i * MINUTE_NS + 5,
                    open_ns + i * MINUTE_NS + 70_000_000) for i in range(5)]
    reader = _store(tmp_path, build_bars(trades, MINUTE_NS), "snap1")

    seen: set[int] = set()
    for step in range(0, 8 * 60, 11):
        clock = open_ns + step * 1_000_000_000
        events = set(reader.read_as_of(clock)[EVENT_TIME]) if not reader.read_as_of(clock).empty else set()
        assert seen <= events, f"history shrank at clock {clock}"
        seen = events
