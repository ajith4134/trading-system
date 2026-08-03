"""Bars are where look-ahead enters, so these tests attack the availability time."""
from __future__ import annotations

import pandas as pd
import pytest

from capture.frame_codec import IndexEntry
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL
from store.trade_bars import Trade, UnknownVenueFormat, build_bars, extract_trades

MINUTE_NS = 60_000_000_000

# Real frames from the archive, 2026-08-02.
BINANCE_FRAME = (
    '{"stream":"btcusdt@trade","data":{"e":"trade","E":1785685177439,'
    '"T":1785685177439,"s":"BTCUSDT","t":7947325360,"p":"63113.20","q":"0.001",'
    '"X":"MARKET","m":false,"st":1}}'
)
HYPERLIQUID_FRAME = (
    '{"channel":"trades","data":['
    '{"coin":"BTC","side":"A","px":"63123.0","sz":"0.0002","time":1785685210021,"tid":1},'
    '{"coin":"BTC","side":"B","px":"63124.0","sz":"0.00343","time":1785685212972,"tid":2}]}'
)


def _entry(t_recv_ns: int, t_exch_ms: int | None = None) -> IndexEntry:
    return IndexEntry(n=0, t_recv_ns=t_recv_ns, t_exch_ms=t_exch_ms,
                      seq=None, kind="data", esc=False)


def test_binance_frame_yields_one_trade():
    entry = _entry(1785685177508349176, 1785685177439)
    trades = extract_trades(BINANCE_FRAME, entry, "binance", "BTCUSDT")
    assert len(trades) == 1
    assert trades[0].price == pytest.approx(63113.20)
    assert trades[0].size == pytest.approx(0.001)
    assert trades[0].event_time_ns == 1785685177439 * 1_000_000
    assert trades[0].ingestion_time_ns == 1785685177508349176


def test_hyperliquid_frame_yields_every_trade_in_the_array():
    """One frame carries many trades; dropping to the first loses real volume."""
    entry = _entry(1785685241650215450, 1785685210021)
    trades = extract_trades(HYPERLIQUID_FRAME, entry, "hyperliquid", "BTC")
    assert len(trades) == 2
    assert [t.price for t in trades] == pytest.approx([63123.0, 63124.0])


def test_every_trade_in_a_batch_shares_the_frames_ingestion_time():
    """They arrived together, whatever their venue timestamps say.

    Hyperliquid's reconnect backfill puts trades up to 32 seconds apart in one
    frame. All of them became knowable at the instant the frame landed, and
    assigning each its own arrival would fabricate an arrival that never happened.
    """
    entry = _entry(1785685241650215450, 1785685210021)
    trades = extract_trades(HYPERLIQUID_FRAME, entry, "hyperliquid", "BTC")
    assert {t.ingestion_time_ns for t in trades} == {1785685241650215450}
    assert len({t.event_time_ns for t in trades}) == 2


def test_an_unknown_venue_is_refused_rather_than_guessed():
    with pytest.raises(UnknownVenueFormat, match="kraken"):
        extract_trades("{}", _entry(1), "kraken", "BTCUSD")


def test_bar_availability_is_its_close_when_data_arrived_promptly():
    """The ordinary case: a bar becomes usable when it closes, not before."""
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0, 1.0,
                    event_time_ns=open_ns + 1_000, ingestion_time_ns=open_ns + 71_000_000)]
    bars = build_bars(trades, MINUTE_NS)
    assert bars.loc[0, AVAILABILITY_TIME] == open_ns + MINUTE_NS


def test_a_late_trade_pushes_availability_past_the_bar_close():
    """The leak this layer exists to prevent, in its exact form.

    A trade whose venue time falls inside a bar but which arrived 32 seconds after
    that bar closed did not exist at close. Marking the bar available at close
    lets a backtest read a price that had not arrived, and nothing in the output
    looks wrong.
    """
    open_ns = 100 * MINUTE_NS
    arrived_ns = open_ns + MINUTE_NS + 32 * 1_000_000_000
    trades = [Trade("BTC", "hyperliquid", 100.0, 1.0,
                    event_time_ns=open_ns + 1_000, ingestion_time_ns=arrived_ns)]
    bars = build_bars(trades, MINUTE_NS)
    assert bars.loc[0, AVAILABILITY_TIME] == arrived_ns
    assert bars.loc[0, AVAILABILITY_TIME] > open_ns + MINUTE_NS


def test_bar_event_time_is_the_bar_open():
    open_ns = 100 * MINUTE_NS
    trades = [Trade("BTCUSDT", "binance", 100.0, 1.0, open_ns + 5, open_ns + 10)]
    assert build_bars(trades, MINUTE_NS).loc[0, EVENT_TIME] == open_ns


def test_ohlcv_is_computed_in_event_time_order():
    """Frames can arrive out of order; open and close must follow the venue's clock."""
    open_ns = 100 * MINUTE_NS
    trades = [
        Trade("BTCUSDT", "binance", 102.0, 1.0, open_ns + 30, open_ns + 40),
        Trade("BTCUSDT", "binance", 100.0, 2.0, open_ns + 10, open_ns + 50),
        Trade("BTCUSDT", "binance", 105.0, 3.0, open_ns + 20, open_ns + 60),
    ]
    bar = build_bars(trades, MINUTE_NS).iloc[0]
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (100.0, 105.0, 100.0, 102.0)
    assert bar["volume"] == pytest.approx(6.0)
    assert bar["trades"] == 3


def test_bars_from_no_trades_are_empty_not_zero_filled():
    """A bar with no trades did not happen; inventing a flat one invents liquidity."""
    assert build_bars([], MINUTE_NS).empty
