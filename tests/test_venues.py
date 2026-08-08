import json

from capture.venues import UrlBudgetTooSmall, shard_by_url_budget
from capture.venues.binance import BinanceVenue
from capture.venues.binance_spot import BinanceSpotVenue
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


# Captured verbatim from https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT
# on 2026-08-03T17:36Z. This is the replacement source for the mark price feed
# the websocket withholds - see the module docstring of test_rest_poller.py.
REAL_BINANCE_PREMIUM_INDEX_BODY = json.loads(
    '{"symbol":"BTCUSDT","markPrice":"63856.20000000",'
    '"indexPrice":"63883.54000000","estimatedSettlePrice":"63846.38997717",'
    '"lastFundingRate":"0.00000707","interestRate":"0.00010000",'
    '"nextFundingTime":1785801600000,"time":1785778577000}')


def test_binance_poll_specs_cover_every_symbol():
    v = BinanceVenue()
    specs = v.poll_specs(["BTCUSDT", "ETHUSDT"])
    assert {spec.symbol for spec in specs} == {"BTCUSDT", "ETHUSDT"}
    assert all(spec.stream == "premiumIndex" for spec in specs)
    assert all(spec.url.startswith("https://fapi.binance.com/fapi/v1/premiumIndex")
               for spec in specs)
    # Per-symbol, not the whole-market form: the all-symbols call returns every
    # perp on the venue at request weight 10, and writing 500 instruments to
    # disk to read 3 of them is not a raw archive of what was asked for.
    assert all(f"symbol={spec.symbol}" in spec.url for spec in specs)


def test_binance_premium_index_response_routes_to_its_own_stream():
    """It is not filed as `markPrice`. The payload shape differs from a
    `markPriceUpdate` frame, and one filename holding two shapes makes the
    archive undecodable without knowing which day it was written."""
    v = BinanceVenue()
    spec = next(s for s in v.poll_specs(["BTCUSDT"]))
    meta = v.extract(REAL_BINANCE_PREMIUM_INDEX_BODY)

    assert meta.stream == spec.stream == "premiumIndex"
    assert meta.symbol == spec.symbol == "BTCUSDT"
    assert meta.kind == "data"
    assert meta.t_exch_ms == 1785778577000
    assert meta.seq is None


def test_binance_does_not_subscribe_to_the_mark_price_stream_the_venue_withholds():
    """Measured 2026-08-03 across three edge IPs: zero frames on `markPrice@1s`
    and `!markPrice@arr@1s` while `trade` flowed on the same sockets. Keeping
    the subscription would report the stream silent forever and bury the real
    source, which is now the REST poll."""
    v = BinanceVenue()
    channels = {s.channel for s in v.core_specs(["BTCUSDT"])} | \
               {s.channel for s in v.tail_specs(["BTCUSDT"])}
    assert not any("markPrice" in channel for channel in channels)


def test_binance_still_subscribes_to_liquidations_despite_the_silence():
    """Deliberate, and the opposite of the mark price decision: `forceOrder` has
    no REST replacement (allForceOrders was withdrawn from the public API), so
    dropping it would remove the only way to notice the venue starting to
    deliver it. It costs one idle subscription and leaves the tile red, which is
    what an unavailable feed should look like."""
    v = BinanceVenue()
    channels = {s.channel for s in v.core_specs(["BTCUSDT"])}
    assert "btcusdt@forceOrder" in channels


# --- Sharding the broad tail across connections -----------------------------
# Measured against the live venue 2026-08-08, recorded in
# ~/research/binance-fstream-connection-limits.md: fstream's binding constraint
# is the length of the request line, not the documented 1024-stream cap. 928
# streams (16,338 chars) connect; 960 (16,886) return HTTP 414. The full tail is
# 569 perpetuals x 2 channels = 1,138 streams, so it cannot be one socket.

def test_shards_never_exceed_the_venues_url_budget():
    v = BinanceVenue()
    specs = v.tail_specs([f"SYMBOL{n}USDT" for n in range(400)])
    shards = shard_by_url_budget(v, specs)
    assert len(shards) > 1, "400 symbols should not fit one shard"
    for shard in shards:
        assert len(v.ws_url(shard)) <= v.max_url_bytes


def test_sharding_preserves_every_spec_exactly_once_and_in_order():
    """A dropped spec is a symbol that is silently never captured, and the
    archive cannot be backfilled. Losing one must be impossible, not unlikely."""
    v = BinanceVenue()
    specs = v.tail_specs([f"SYMBOL{n}USDT" for n in range(400)])
    rejoined = [spec for shard in shard_by_url_budget(v, specs) for spec in shard]
    assert rejoined == specs


def test_sharding_a_single_spec_that_cannot_fit_is_refused_not_truncated():
    """An over-budget shard connects to nothing and returns HTTP 414, which
    looks exactly like a quiet market. Refusing names the problem instead."""
    v = BinanceVenue()
    specs = v.tail_specs(["BTCUSDT"])
    try:
        shard_by_url_budget(v, specs, max_url_bytes=10)
    except UrlBudgetTooSmall as exc:
        assert "BTCUSDT".lower() in str(exc).lower()
    else:
        raise AssertionError("expected UrlBudgetTooSmall")


def test_binance_url_budget_sits_below_the_measured_414_threshold():
    """928 streams at 16,338 bytes connected; 960 at 16,886 did not. The budget
    must leave room for the universe changing under us - a batch of long-named
    tokens listing must not be what discovers the ceiling."""
    v = BinanceVenue()
    assert v.max_url_bytes <= 16_338
    assert v.max_url_bytes < 16_338 * 0.9, "no headroom for a changing universe"


def test_the_whole_binance_tail_shards_into_a_handful_of_connections():
    """569 perpetuals x 2 channels. Not one, and not so many that the shard
    count starts competing with the 300-connections-per-5-minutes budget."""
    v = BinanceVenue()
    specs = v.tail_specs([f"SYM{n:03d}USDT" for n in range(569)])
    shards = shard_by_url_budget(v, specs)
    assert 2 <= len(shards) <= 8


def test_hyperliquid_subscribes_over_the_socket_so_its_tail_is_one_shard():
    """Its URL carries no channels, so the fstream constraint does not apply and
    splitting would spend connections for nothing."""
    v = HyperliquidVenue()
    specs = v.tail_specs([f"COIN{n}" for n in range(300)])
    assert len(shard_by_url_budget(v, specs)) == 1


# --- Binance spot -----------------------------------------------------------
# Goal doc 10.2: spot-perp basis is a P1 "start here" family and the prime
# directive rests on carry, but only the perp leg was ever captured. Measured
# 2026-08-08: 1,377 spot symbols TRADING, 489 of them USDT-quoted.

def test_spot_is_a_separate_venue_from_futures():
    """The venue name becomes a directory, and (stream, symbol) becomes a
    filename inside it. Sharing a name would file spot BTCUSDT and perp BTCUSDT
    as one instrument - two different markets merged, silently."""
    assert BinanceSpotVenue().name != BinanceVenue().name


def test_spot_connects_to_the_spot_endpoint_not_the_futures_one():
    v = BinanceSpotVenue()
    url = v.ws_url(v.core_specs(["BTCUSDT"]))
    assert url.startswith("wss://stream.binance.com")
    assert "fstream" not in url


def test_spot_captures_depth_and_trades():
    v = BinanceSpotVenue()
    channels = {s.channel for s in v.core_specs(["BTCUSDT"])}
    assert "btcusdt@depth@100ms" in channels
    assert "btcusdt@trade" in channels


def test_spot_does_not_subscribe_to_feeds_that_do_not_exist_on_spot():
    """There are no liquidations and no funding on spot. Subscribing anyway
    would report both silent forever, which is how a real outage gets lost in
    the noise of two feeds that were never coming."""
    v = BinanceSpotVenue()
    channels = {s.channel for s in v.core_specs(["BTCUSDT"])} | \
               {s.channel for s in v.tail_specs(["BTCUSDT"])}
    assert not any("forceOrder" in c for c in channels)
    assert v.poll_specs(["BTCUSDT"]) == []


def test_spot_parses_only_tradeable_symbols():
    v = BinanceSpotVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"},
        {"symbol": "DEADUSDT", "status": "BREAK", "quoteAsset": "USDT"},
        {"symbol": "ETHBTC", "status": "TRADING", "quoteAsset": "BTC"},
    ]}
    assert v.parse_instruments(payload) == ["BTCUSDT", "ETHBTC"]


def test_the_whole_spot_universe_shards_into_a_handful_of_connections():
    """1,377 symbols on the cheap channel. Measured: 1,024 streams at 14,846
    bytes connect on the spot endpoint."""
    v = BinanceSpotVenue()
    shards = shard_by_url_budget(v, v.tail_specs([f"SYM{n:04d}USDT" for n in range(1377)]))
    assert 2 <= len(shards) <= 8
    for shard in shards:
        assert len(v.ws_url(shard)) <= v.max_url_bytes
