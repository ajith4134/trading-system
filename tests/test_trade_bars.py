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


# A real captured spot frame, byte for byte from
# capture/raw/binance-spot/2026-08-08/trade_BTCUSDT_2026-08-08T10.ndjson.zst.
# Copied rather than adapted from the perp fixture: the whole question this
# answers is whether the two shapes are actually the same, and a fixture written
# by editing the perp one would assume the answer.
BINANCE_SPOT_FRAME = (
    '{"stream":"btcusdt@trade","data":{"e":"trade","E":1786183200055,'
    '"s":"BTCUSDT","t":6564147733,"p":"64994.31000000","q":"0.00330000",'
    '"T":1786183200055,"m":true,"M":true}}'
)


def test_a_real_spot_frame_yields_its_trade():
    """Spot carries `M` and no `X`/`st`; the fields that matter are identical.

    1,363 captured symbols were unbuildable purely because the venue was not
    registered - not because its frames needed anything new.
    """
    entry = _entry(1786183200100000000, 1786183200055)
    trades = extract_trades(BINANCE_SPOT_FRAME, entry, "binance-spot", "BTCUSDT")

    assert len(trades) == 1
    assert trades[0].venue == "binance-spot"
    assert trades[0].price == pytest.approx(64994.31)
    assert trades[0].size == pytest.approx(0.0033)
    assert trades[0].event_time_ns == 1786183200055 * 1_000_000


def test_the_same_symbol_on_spot_and_perp_stays_two_instruments():
    """BTCUSDT exists on both venues and they are not the same thing.

    This is the actual risk in adding spot: the symbol strings collide. If venue
    were not part of the bar key, spot prints would be folded into the perp's
    OHLCV and the resulting bar would describe a market that does not exist.
    """
    entry = _entry(1786183200100000000, 1786183200055)
    trades = (extract_trades(BINANCE_SPOT_FRAME, entry, "binance-spot", "BTCUSDT")
              + extract_trades(BINANCE_FRAME, entry, "binance", "BTCUSDT"))

    bars = build_bars(trades, 60_000_000_000)

    assert len(bars) == 2
    assert sorted(bars["venue"]) == ["binance", "binance-spot"]


def test_a_venue_frame_with_a_zero_price_is_not_a_trade():
    """Binance really sends these, and one of them poisons a whole bar's low.

    Captured verbatim from the live tape, 2026-08-03T18 on binance BTCUSDT, where 60
    of 17,227 trade frames looked like this: price "0", quantity "0", `X` "NA". Layer
    0 is right to store them - it never modifies what a venue said - but they are not
    trades, and `float(data["p"])` turned each into a Trade at price 0.0.

    The damage was not subtle once looked for. In the real store on 2026-08-08, 746
    of 1,671 bars carried `low <= 0`: BTCUSDT 237 of 278, ETHUSDT 268 of 278, SOLUSDT
    241 of 278, plus 25 zero opens and 19 zero closes. Hyperliquid's 837 bars were
    untouched, because only this feed emits them. `min()` needs one zero to ruin a
    bar, and every one of those bars otherwise looked perfectly normal.
    """
    from capture.frame_codec import IndexEntry
    from store.trade_bars import extract_trades

    entry = IndexEntry(n=0, t_recv_ns=1785780046200_000_000, t_exch_ms=1785780046072,
                       seq=None, kind="data", esc=False)
    placeholder = ('{"stream":"btcusdt@trade","data":{"e":"trade","E":1785780046072,'
                   '"T":1785780046071,"s":"BTCUSDT","t":7950493454,"p":"0","q":"0",'
                   '"X":"NA","m":true,"st":1}}')
    real = ('{"stream":"btcusdt@trade","data":{"e":"trade","E":1785780046072,'
            '"T":1785780046071,"s":"BTCUSDT","t":7950493455,"p":"63884.20",'
            '"q":"0.01","X":"MARKET","m":true}}')

    assert extract_trades(placeholder, entry, "binance", "BTCUSDT") == []
    kept = extract_trades(real, entry, "binance", "BTCUSDT")
    assert [t.price for t in kept] == [63884.20]


def test_a_zero_price_frame_cannot_reach_the_low_of_a_bar():
    """The end-to-end shape of the defect: one placeholder, one real trade, one bar.

    Asserted through `build_bars` rather than only at the extractor, because the
    aggregation is where the harm appeared - `low=("price", "min")` has no way to
    tell a zero apart from a cheap fill.
    """
    from capture.frame_codec import IndexEntry
    from store.trade_bars import build_bars, extract_trades

    entry = IndexEntry(n=0, t_recv_ns=1785780046200_000_000, t_exch_ms=1785780046072,
                       seq=None, kind="data", esc=False)
    frames = [
        ('{"data":{"e":"trade","T":1785780046071,"s":"BTCUSDT","p":"0","q":"0",'
         '"X":"NA","st":1}}'),
        '{"data":{"e":"trade","T":1785780046072,"s":"BTCUSDT","p":"63884.20","q":"0.01"}}',
        '{"data":{"e":"trade","T":1785780046073,"s":"BTCUSDT","p":"63880.10","q":"0.02"}}',
    ]
    trades = [t for payload in frames
              for t in extract_trades(payload, entry, "binance", "BTCUSDT")]

    bars = build_bars(trades, 60_000_000_000)
    assert len(bars) == 1
    assert float(bars.iloc[0]["low"]) == 63880.10
    assert float(bars.iloc[0]["open"]) == 63884.20
    assert int(bars.iloc[0]["trades"]) == 2, "the placeholder must not be counted as a trade"


# --- streaming aggregation, added with BarAccumulator 2026-08-10 ----------

def test_open_and_close_follow_the_venue_clock_not_the_arrival_order():
    """The semantics the frame version got from a stable sort, and the reason
    this is not pedantry.

    Venue frames arrive out of event order - hyperliquid reconnect backfill was
    measured carrying trades spanning 32.4 seconds of venue time in one frame -
    so "the last trade added" and "the last trade of the minute" are different
    rows. Taking the wrong one puts a stale price in the close that every return
    is computed from.
    """
    from store.trade_bars import BarAccumulator

    minute = 60_000_000_000
    accumulator = BarAccumulator(minute)
    # Added middle, then last, then FIRST - the shape a reconnect produces.
    for price, event_ns in ((101.0, 30 * 10**9), (102.0, 50 * 10**9),
                            (100.0, 1 * 10**9)):
        accumulator.add(Trade(symbol="BTCUSDT", venue="binance", price=price,
                              size=1.0, event_time_ns=event_ns,
                              ingestion_time_ns=event_ns))

    bar = accumulator.to_frame().iloc[0]

    assert bar["open"] == 100.0, "open must be the earliest event, not the first seen"
    assert bar["close"] == 102.0, "close must be the latest event, not the last seen"
    assert bar["high"] == 102.0 and bar["low"] == 100.0


def test_trades_sharing_an_event_time_keep_the_order_they_arrived_in():
    """What a STABLE sort did: among ties, open takes the first row seen and
    close takes the last. Reproduced here by a strict `<` for open and a `>=`
    for close."""
    from store.trade_bars import BarAccumulator

    accumulator = BarAccumulator(60_000_000_000)
    for price in (10.0, 20.0, 30.0):
        accumulator.add(Trade(symbol="BTCUSDT", venue="binance", price=price,
                              size=1.0, event_time_ns=5 * 10**9,
                              ingestion_time_ns=5 * 10**9))

    bar = accumulator.to_frame().iloc[0]

    assert bar["open"] == 10.0
    assert bar["close"] == 30.0


def test_the_accumulator_holds_one_bucket_per_bar_not_one_per_trade():
    """The whole point: peak follows the number of BARS. Ten thousand trades in
    two minutes is two buckets."""
    from store.trade_bars import BarAccumulator

    minute = 60_000_000_000
    accumulator = BarAccumulator(minute)
    for i in range(10_000):
        accumulator.add(Trade(symbol="BTCUSDT", venue="binance", price=100.0 + i,
                              size=1.0, event_time_ns=i * 12_000_000,
                              ingestion_time_ns=i * 12_000_000))

    assert len(accumulator) == 2
    assert accumulator.to_frame()["trades"].sum() == 10_000


def test_the_streaming_and_one_shot_forms_agree():
    """`build_bars` is the same accumulator with the list fed in, so a caller
    that already holds its trades cannot get a different answer from one that
    streams them."""
    from store.trade_bars import BarAccumulator

    minute = 60_000_000_000
    trades = [
        Trade(symbol="BTCUSDT", venue="binance", price=100.0 + (i % 7),
              size=0.5, event_time_ns=i * 7_000_000_000,
              ingestion_time_ns=i * 7_000_000_000 + 1_000_000)
        for i in range(50)
    ]

    streamed = BarAccumulator(minute)
    for trade in trades:
        streamed.add(trade)

    pd.testing.assert_frame_equal(streamed.to_frame(), build_bars(trades, minute))
