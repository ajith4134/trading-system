import json

from capture.venues.binance import BinanceVenue
from capture.venues.hyperliquid import HyperliquidVenue

# Captured verbatim from wss://fstream.binance.com/stream?streams=btcusdt@trade
# on 2026-08-02T11:51Z. Not invented: the shape of a real frame is the thing
# under test, and `st` for one is undocumented.
REAL_BINANCE_TRADE_FRAME = json.loads(
    '{"stream":"btcusdt@trade","data":{"e":"trade","E":1785671500407,'
    '"T":1785671500407,"s":"BTCUSDT","t":7947131392,"p":"63105.80",'
    '"q":"0.030","X":"MARKET","m":true,"st":1}}')


def test_binance_builds_combined_stream_url():
    v = BinanceVenue()
    specs = v.core_specs(["BTCUSDT", "ETHUSDT"])
    url = v.ws_url(specs)
    assert url.startswith("wss://fstream.binance.com/stream?streams=")
    assert "btcusdt@depth@100ms" in url
    assert "btcusdt@trade" in url


def test_binance_captures_individual_trades_not_aggregated_ones():
    """aggTrade delivers nothing over the websocket (measured 2026-08-02) and
    aggregated trades are less raw than individual ones either way."""
    v = BinanceVenue()
    core = {spec.channel for spec in v.core_specs(["BTCUSDT"])}
    tail = {spec.channel for spec in v.tail_specs(["BTCUSDT"])}
    assert "btcusdt@trade" in core and "btcusdt@trade" in tail
    assert not any("aggTrade" in channel for channel in core | tail)


def test_binance_trade_frame_routes_to_the_stream_its_spec_named():
    """`StreamSpec.stream` and `ExtractedMeta.stream` have to agree. If they
    drift, one logical stream is written under two filenames and neither holds
    the whole record."""
    v = BinanceVenue()
    spec = next(s for s in v.core_specs(["BTCUSDT"]) if s.channel == "btcusdt@trade")
    meta = v.extract(REAL_BINANCE_TRADE_FRAME)

    assert meta.stream == spec.stream == "trade"
    assert meta.symbol == spec.symbol == "BTCUSDT"
    assert meta.kind == "data"
    assert meta.t_exch_ms == 1785671500407
    # A trade carries a trade id (`t`) but no U/u/pu chain to reconcile against,
    # so there is no sequence to record. The id is in the payload verbatim for
    # anyone who later wants to check trade-id continuity.
    assert meta.seq is None


def test_binance_extract_reads_both_timestamps_and_chain():
    v = BinanceVenue()
    parsed = {"e": "depthUpdate", "E": 1785650606302, "T": 1785650606300,
              "s": "BTCUSDT", "U": 11192579046493, "u": 11192579053768,
              "pu": 11192579046406}
    meta = v.extract(parsed)
    assert meta.t_exch_ms == 1785650606302
    assert meta.seq == {"U": 11192579046493, "u": 11192579053768,
                        "pu": 11192579046406, "T": 1785650606300}
    assert meta.kind == "data"
    assert meta.symbol == "BTCUSDT"


def test_binance_instruments_request_is_a_plain_get():
    v = BinanceVenue()
    assert v.instruments_request() == (
        "GET", "https://fapi.binance.com/fapi/v1/exchangeInfo", None,
    )


def test_binance_parses_perp_instruments_only():
    v = BinanceVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING"},
        {"symbol": "ETHUSDT_240329", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
        {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "status": "BREAK"},
    ]}
    assert v.parse_instruments(payload) == ["BTCUSDT"]


def test_hyperliquid_instruments_request_is_a_post_with_meta_body():
    v = HyperliquidVenue()
    assert v.instruments_request() == (
        "POST", "https://api.hyperliquid.xyz/info", {"type": "meta"},
    )


def test_hyperliquid_subscribe_messages_cover_each_spec():
    v = HyperliquidVenue()
    specs = v.core_specs(["BTC"])
    msgs = v.subscribe_messages(specs)
    assert {"method": "subscribe",
            "subscription": {"type": "l2Book", "coin": "BTC"}} in msgs


def test_hyperliquid_extract_flags_control_frames():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "subscriptionResponse", "data": {}})
    assert meta.kind == "control"
    assert meta.seq is None


def test_hyperliquid_extract_reads_snapshot_time():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "l2Book",
                      "data": {"coin": "BTC", "time": 1785650605471, "levels": []}})
    assert meta.kind == "data"
    assert meta.t_exch_ms == 1785650605471
    assert meta.symbol == "BTC"
    assert meta.seq is None            # no sequence numbers exist


# --- Malformed input, per the wire, never trusted ---


def test_binance_extract_does_not_raise_on_missing_event_field():
    v = BinanceVenue()
    meta = v.extract({"s": "BTCUSDT", "U": 1, "u": 2})
    assert meta.kind == "control"
    assert meta.seq is None
    assert meta.t_exch_ms is None


def test_binance_extract_does_not_raise_on_non_dict_frame():
    v = BinanceVenue()
    for bad in (None, [], "not json", 5, {"data": "not a dict"}):
        meta = v.extract(bad)
        assert meta.kind == "control"


def test_binance_extract_reads_top_level_symbol_when_present():
    v = BinanceVenue()
    meta = v.extract({"e": "aggTrade", "E": 1785650606302, "s": "ETHUSDT"})
    assert meta.symbol == "ETHUSDT"


def test_binance_extract_falls_back_to_nested_order_symbol_for_force_order():
    v = BinanceVenue()
    parsed = {"e": "forceOrder", "E": 1785650606302,
              "o": {"s": "BTCUSDT", "S": "SELL", "q": "1.000"}}
    meta = v.extract(parsed)
    assert meta.symbol == "BTCUSDT"
    assert meta.kind == "data"
    assert meta.stream == "forceOrder"


def test_binance_extract_depth_update_missing_chain_fields_has_no_seq():
    v = BinanceVenue()
    meta = v.extract({"e": "depthUpdate", "E": 1, "s": "BTCUSDT"})
    assert meta.seq is None


def test_binance_parse_instruments_skips_malformed_entries():
    v = BinanceVenue()
    payload = {"symbols": [
        {"contractType": "PERPETUAL", "status": "TRADING"},          # missing symbol
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING"},
        "not-a-dict",
        None,
    ]}
    assert v.parse_instruments(payload) == ["BTCUSDT"]


def test_binance_parse_instruments_handles_missing_or_wrong_shaped_symbols():
    v = BinanceVenue()
    assert v.parse_instruments({}) == []
    assert v.parse_instruments({"symbols": "not-a-list"}) == []
    assert v.parse_instruments("not-a-dict") == []


def test_hyperliquid_extract_handles_missing_data():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "l2Book"})
    assert meta.kind == "data"
    assert meta.t_exch_ms is None
    assert meta.symbol == "unknown"


def test_hyperliquid_extract_handles_list_shaped_data():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "trades",
                      "data": [{"coin": "BTC", "time": 1785650605471}]})
    assert meta.kind == "data"
    assert meta.symbol == "BTC"
    assert meta.t_exch_ms == 1785650605471


def test_hyperliquid_extract_handles_empty_list_data():
    v = HyperliquidVenue()
    meta = v.extract({"channel": "trades", "data": []})
    assert meta.kind == "data"
    assert meta.symbol == "unknown"
    assert meta.t_exch_ms is None


def test_hyperliquid_extract_does_not_raise_on_non_dict_frame():
    v = HyperliquidVenue()
    for bad in (None, [], "not json", 5):
        meta = v.extract(bad)
        assert meta.kind == "control"


def test_hyperliquid_parse_instruments_skips_delisted_and_malformed():
    v = HyperliquidVenue()
    payload = {"universe": [
        {"name": "BTC", "isDelisted": False},
        {"name": "OLD", "isDelisted": True},
        {"isDelisted": False},                 # missing name
        "not-a-dict",
    ]}
    assert v.parse_instruments(payload) == ["BTC"]
