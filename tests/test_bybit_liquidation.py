"""The liquidation venue files what the wire delivers, and refuses what it cannot see.

The frame shapes in these tests are the ones the 2026-08-09 probe recorded off
the live socket, not invented ones — a decoder tested against imagined frames
is tested against nothing.
"""
import pytest

from capture.venues import UrlBudgetTooSmall, shard_by_subscribe_budget, shard_for_connection
from capture.venues.bybit_liquidation import BybitLiquidationVenue, UniverseTruncated
from capture.venues.hyperliquid import HyperliquidVenue

# Verbatim from the probe, 2026-08-09.
_LIVE_FRAME = {
    "topic": "allLiquidation.PUMPFUNUSDT", "type": "snapshot", "ts": 1786290003229,
    "data": [{"T": 1786290003196, "s": "PUMPFUNUSDT", "S": "Sell",
              "v": "172900", "p": "0.0026913"}],
}

_LISTING = {
    "retCode": 0,
    "result": {
        "category": "linear",
        "nextPageCursor": "",
        "list": [
            {"symbol": "BTCUSDT", "status": "Trading", "quoteCoin": "USDT"},
            {"symbol": "SOLPERP", "status": "Trading", "quoteCoin": "USDC"},
            {"symbol": "OLDUSDT", "status": "Closed", "quoteCoin": "USDT"},
        ],
    },
}


def _venue():
    return BybitLiquidationVenue()


# --- extract: routing measured frames ------------------------------------

def test_live_probe_frame_is_filed_under_its_symbol():
    meta = _venue().extract(_LIVE_FRAME)
    assert meta.kind == "data"
    assert meta.stream == "allLiquidation"
    assert meta.symbol == "PUMPFUNUSDT"
    assert meta.t_exch_ms == 1786290003229


def test_subscribe_ack_is_control_not_data():
    ack = {"success": True, "ret_msg": "", "op": "subscribe", "conn_id": "x"}
    meta = _venue().extract(ack)
    assert meta.kind == "control"
    assert meta.stream == "subscribe"


def test_malformed_row_falls_back_to_the_topic_symbol():
    frame = {"topic": "allLiquidation.ETHUSDT", "ts": 1786290007872, "data": "gone"}
    meta = _venue().extract(frame)
    assert meta.kind == "data", "the topic still names a real stream"
    assert meta.symbol == "ETHUSDT"


def test_non_integer_venue_clock_is_dropped_not_guessed():
    frame = {**_LIVE_FRAME, "ts": "1786290003229"}
    assert _venue().extract(frame).t_exch_ms is None


# --- the universe ---------------------------------------------------------

def test_only_trading_instruments_are_subscribed():
    assert _venue().parse_instruments(_LISTING) == ["BTCUSDT", "SOLPERP"]


def test_quote_currency_is_read_never_inferred_from_the_suffix():
    quotes = _venue().parse_quote_assets(_LISTING)
    assert quotes == {"BTCUSDT": "USDT", "SOLPERP": "USDC"}, (
        "SOLPERP has no parseable suffix - the venue's own quoteCoin is the "
        "only honest source")


def test_a_second_listing_page_refuses_rather_than_truncating():
    paged = {"retCode": 0, "result": {**_LISTING["result"], "nextPageCursor": "abc"}}
    with pytest.raises(UniverseTruncated):
        _venue().parse_instruments(paged)


def test_error_body_with_http_200_yields_no_instruments():
    assert _venue().parse_instruments({"retCode": 10001, "result": {}}) == []


# --- subscribing ----------------------------------------------------------

def test_subscribe_messages_batch_and_cover_every_topic():
    specs = _venue().core_specs([f"SYM{n:03d}USDT" for n in range(700)])
    messages = _venue().subscribe_messages(specs)
    assert all(m["op"] == "subscribe" for m in messages)
    assert all(len(m["args"]) <= 300 for m in messages)
    sent = [topic for m in messages for topic in m["args"]]
    assert sent == [s.channel for s in specs], "every topic exactly once, in order"


# --- sharding by subscribe budget ----------------------------------------

def _chars(spec):
    return len(spec.channel) + 3


def test_each_shard_fits_the_documented_budget_and_none_is_lost():
    venue = _venue()
    specs = venue.core_specs([f"SYM{n:04d}USDT" for n in range(805)])
    shards = shard_by_subscribe_budget(venue, specs)
    assert len(shards) > 1, "805 symbols measured over 21,000 chars on 2026-08-09"
    for shard in shards:
        assert sum(_chars(s) for s in shard) <= venue.max_subscribe_chars
    rejoined = [spec for shard in shards for spec in shard]
    assert rejoined == specs, "a dropped spec is a symbol never captured"


def test_one_topic_over_budget_refuses_with_its_name():
    venue = _venue()
    venue_small = type(venue)()
    venue_small.max_subscribe_chars = 10
    specs = venue.core_specs(["BTCUSDT"])
    with pytest.raises(UrlBudgetTooSmall, match="allLiquidation.BTCUSDT"):
        shard_by_subscribe_budget(venue_small, specs)


def test_shard_for_connection_picks_the_budget_the_venue_declares():
    liq = _venue()
    liq_shards = shard_for_connection(liq, liq.core_specs(
        [f"SYM{n:04d}USDT" for n in range(805)]))
    assert len(liq_shards) > 1

    hyper = HyperliquidVenue()
    hyper_shards = shard_for_connection(hyper, hyper.tail_specs(["BTC", "ETH"]))
    assert len(hyper_shards) == 1, (
        "a venue declaring no budget keeps its single connection")


# --- market-wide silence semantics ----------------------------------------

def test_venue_declares_liquidations_market_wide():
    assert "allLiquidation" in _venue().market_wide_streams, (
        "per-symbol silence on an event stream recorded 800 of 805 symbols "
        "silent in one 75-second run - the whole feed is the unit that dies")
