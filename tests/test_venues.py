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


def test_binance_polls_funding_for_the_whole_market_in_one_request():
    """Reversed 2026-08-09, and the reason is worth stating.

    This asserted the per-symbol form, on the reasoning that writing 500
    instruments to disk to read 3 of them is not a raw archive of what was asked
    for. Right while the universe was three symbols; it became the binding
    constraint on Phase 4, which needs funding across the universe and cannot
    backfill it. Per symbol, 857 perps cost 857 weight a tick against a
    2,400/minute budget. The all-market form is one request at weight 10.
    """
    v = BinanceVenue()
    specs = v.poll_specs(["BTCUSDT", "ETHUSDT"])

    funding = [s for s in specs if s.stream == "premiumIndex"]
    assert len(funding) == 1, "one request covers the market, not one per symbol"
    assert funding[0].fan_out is True
    assert funding[0].url == "https://fapi.binance.com/fapi/v1/premiumIndex"
    assert "symbol=" not in funding[0].url, "still asking for one instrument"

    # The depth snapshot stays per symbol: only the core carries depth diffs to
    # replay a snapshot onto, and there is no all-market form of it.
    snapshots = [s for s in specs if s.stream == "depthSnapshot"]
    assert {s.symbol for s in snapshots} == {"BTCUSDT", "ETHUSDT"}
    assert all(f"symbol={s.symbol}" in s.url for s in snapshots)


def test_the_all_market_funding_response_splits_per_instrument():
    """Each element is written under its own symbol, byte-compatible with what
    the per-symbol poll used to write - which is what lets the funding history
    already captured continue without a seam."""
    v = BinanceVenue()
    spec = next(s for s in v.poll_specs(["BTCUSDT"]) if s.fan_out)

    pairs = v.fan_out_poll(spec, [
        {"symbol": "BTCUSDT", "lastFundingRate": "0.0001"},
        {"symbol": "ETHUSDT", "lastFundingRate": "0.0002"},
    ])

    assert [symbol for symbol, _ in pairs] == ["BTCUSDT", "ETHUSDT"]
    assert pairs[0][1]["lastFundingRate"] == "0.0001"


def test_an_element_naming_no_symbol_is_dropped_rather_than_guessed():
    """Every file here is keyed on symbol and a wrong one poisons a carry cost
    that reads it back. Filing under the request symbol would put the whole
    market in one instrument's file."""
    v = BinanceVenue()
    spec = next(s for s in v.poll_specs(["BTCUSDT"]) if s.fan_out)

    pairs = v.fan_out_poll(spec, [{"lastFundingRate": "0.0001"},
                                  {"symbol": "BTCUSDT"}])

    assert [symbol for symbol, _ in pairs] == ["BTCUSDT"]


def test_a_response_that_is_not_a_list_splits_into_nothing():
    """A shape change at the venue, which the recorder records as observation
    loss rather than treating as an absent market."""
    v = BinanceVenue()
    spec = next(s for s in v.poll_specs(["BTCUSDT"]) if s.fan_out)
    assert v.fan_out_poll(spec, {"symbol": "BTCUSDT"}) == []


def test_binance_premium_index_response_routes_to_its_own_stream():
    """It is not filed as `markPrice`. The payload shape differs from a
    `markPriceUpdate` frame, and one filename holding two shapes makes the
    archive undecodable without knowing which day it was written."""
    v = BinanceVenue()
    spec = next(s for s in v.poll_specs(["BTCUSDT"]) if s.stream == "premiumIndex")
    meta = v.extract(REAL_BINANCE_PREMIUM_INDEX_BODY)

    assert meta.stream == spec.stream == "premiumIndex"
    # The spec now names the REQUEST, and the symbol comes off the element the
    # fan-out split out - which is what `extract` reads here.
    assert meta.symbol == "BTCUSDT"
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
    # Spot polls a depth snapshot, but must never poll funding - there is none.
    assert not any(s.stream == "premiumIndex" for s in v.poll_specs(["BTCUSDT"]))


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


# --- depth snapshots, the missing half of a reconstructable book -------------
# Captured depth frames are depthUpdate diffs (U/u/pu chained), not books.
# Replaying them into a book needs an initial snapshot to apply them onto, and
# none was ever captured - so no book dataset could be built at all.

def test_binance_polls_a_depth_snapshot_for_every_core_symbol():
    specs = {s.stream: s for s in BinanceVenue().poll_specs(["BTCUSDT"])}
    assert "depthSnapshot" in specs
    assert "/fapi/v1/depth" in specs["depthSnapshot"].url
    assert "BTCUSDT" in specs["depthSnapshot"].url


def test_spot_polls_its_own_depth_endpoint():
    specs = {s.stream: s for s in BinanceSpotVenue().poll_specs(["BTCUSDT"])}
    assert "/api/v3/depth" in specs["depthSnapshot"].url


def test_the_snapshot_polls_far_slower_than_the_funding_poll():
    """Measured 2026-08-08 off the x-mbx-used-weight header: a limit=1000 spot
    snapshot costs 50 request-weight, against 1 for premiumIndex. At the
    funding cadence it would spend the whole budget on books."""
    specs = {s.stream: s for s in BinanceVenue().poll_specs(["BTCUSDT"])}
    assert specs["depthSnapshot"].interval_seconds is not None
    assert specs["depthSnapshot"].interval_seconds >= 30
    assert specs["premiumIndex"].interval_seconds is None      # keeps the run cadence


def test_spot_still_polls_no_funding():
    """Spot has no funding. Adding a snapshot poll must not smuggle one in."""
    streams = {s.stream for s in BinanceSpotVenue().poll_specs(["BTCUSDT"])}
    assert streams == {"depthSnapshot"}


def test_binance_spot_parse_quote_assets_reads_the_venues_own_field():
    """Real listings from the live endpoint, 2026-08-08, that break suffix parsing.

    `BTCU` is BTC quoted in `U`, `XRPRLUSD` is quoted in `RLUSD`, `EUREURI` in
    `EURI`. Only the venue's `quoteAsset` gets all three right.
    """
    v = BinanceSpotVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"},
        {"symbol": "BTCU", "quoteAsset": "U", "status": "TRADING"},
        {"symbol": "XRPRLUSD", "quoteAsset": "RLUSD", "status": "TRADING"},
        {"symbol": "EUREURI", "quoteAsset": "EURI", "status": "TRADING"},
    ]}
    assert v.parse_quote_assets(payload) == {
        "BTCUSDT": "USDT", "BTCU": "U", "XRPRLUSD": "RLUSD", "EUREURI": "EURI"}


def test_binance_spot_parse_quote_assets_covers_exactly_the_parsed_universe():
    """A map naming symbols the universe excludes lets a caller request them.

    Same TRADING filter as `parse_instruments`, so the two cannot disagree about
    which symbols exist.
    """
    v = BinanceSpotVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"},
        {"symbol": "DEADUSDT", "quoteAsset": "USDT", "status": "BREAK"},
    ]}
    assert v.parse_instruments(payload) == ["BTCUSDT"]
    assert v.parse_quote_assets(payload) == {"BTCUSDT": "USDT"}


def test_a_symbol_whose_quote_is_missing_is_left_out_not_guessed():
    """Absent from the map it is classified `unknown` and counted; guessed, it is
    silently on whichever side the guess picked."""
    v = BinanceSpotVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT", "status": "TRADING"},
        {"symbol": "NOQUOTE", "status": "TRADING"},
        {"symbol": "BADQUOTE", "quoteAsset": 7, "status": "TRADING"},
    ]}
    assert v.parse_quote_assets(payload) == {"BTCUSDT": "USDT"}


def test_binance_spot_parse_quote_assets_handles_wrong_shaped_payloads():
    v = BinanceSpotVenue()
    assert v.parse_quote_assets({}) == {}
    assert v.parse_quote_assets({"symbols": "not-a-list"}) == {}
    assert v.parse_quote_assets("not-a-dict") == {}


def test_binance_futures_parse_quote_assets_is_not_all_dollars():
    """526 USDT, 38 USDC, 2 USD1, 2 in `U`, 1 in BTC over the 569 live perpetuals.

    Nearly-all-dollars is what makes the exceptions dangerous: they read as a
    rounding error right up until one of them is in a P&L.
    """
    v = BinanceVenue()
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "quoteAsset": "USDT",
         "contractType": "PERPETUAL", "status": "TRADING"},
        {"symbol": "ETHU", "quoteAsset": "U",
         "contractType": "PERPETUAL", "status": "TRADING"},
        {"symbol": "BTCUSDT_260327", "quoteAsset": "USDT",
         "contractType": "CURRENT_QUARTER", "status": "TRADING"},
    ]}
    assert v.parse_quote_assets(payload) == {"BTCUSDT": "USDT", "ETHU": "U"}


def test_hyperliquid_quotes_every_perp_in_usd():
    """The venue publishes no per-symbol quote field, so this is a constant.

    Measured 2026-08-08: a `meta` universe entry carries isDelisted, marginMode,
    marginTableId, maxLeverage, name, onlyIsolated and szDecimals - no quote - and
    the payload's `collateralToken` is the integer 0, a token index rather than a
    name. A `quoteAsset` lookup here would find nothing and leave all 232 symbols
    unclassified.
    """
    v = HyperliquidVenue()
    payload = {"universe": [{"name": "BTC", "isDelisted": False},
                            {"name": "ETH", "isDelisted": False},
                            {"name": "OLD", "isDelisted": True}],
               "collateralToken": 0}
    assert v.parse_quote_assets(payload) == {"BTC": "USD", "ETH": "USD"}


# --------------------------------------------------------------------------
# hyperliquid funding — positionally paired, which is the whole danger
# --------------------------------------------------------------------------

def test_hyperliquid_polls_funding_because_it_pushes_none():
    """`poll_specs` returned [] until 2026-08-09, on the note that this venue
    pushes everything it is asked for. True of trades and books, and it left
    funding uncaptured entirely - on a venue whose funding is hourly on an
    oracle price and capped at 4%/hour, materially different economics from
    Binance's 8-hourly mark."""
    from capture.venues.hyperliquid import HyperliquidVenue

    specs = HyperliquidVenue().poll_specs(["BTC"])

    assert len(specs) == 1
    assert specs[0].fan_out is True
    assert specs[0].method == "POST", "/info answers no GET"
    assert specs[0].body == '{"type":"metaAndAssetCtxs"}'


def test_hyperliquid_pairs_the_universe_with_its_contexts_by_position():
    from capture.venues.hyperliquid import HyperliquidVenue

    v = HyperliquidVenue()
    spec = v.poll_specs(["BTC"])[0]

    pairs = v.fan_out_poll(spec, [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [{"funding": "0.001", "markPx": "64000"}, {"funding": "0.002", "markPx": "3000"}],
    ])

    assert [name for name, _ in pairs] == ["BTC", "ETH"]
    assert pairs[0][1]["funding"] == "0.001"
    assert pairs[1][1]["funding"] == "0.002"


def test_the_coin_is_merged_into_the_stored_record():
    """A ctx names no coin. A record that cannot be decoded without the half of
    the response that was not stored beside it is not an archive."""
    from capture.venues.hyperliquid import HyperliquidVenue

    v = HyperliquidVenue()
    pairs = v.fan_out_poll(v.poll_specs(["BTC"])[0], [
        {"universe": [{"name": "BTC"}]}, [{"funding": "0.001"}]])

    assert pairs[0][1]["coin"] == "BTC"


def test_a_length_mismatch_drops_the_whole_response_rather_than_zipping():
    """`zip` truncates to the shorter side without a word, and the halves are
    joined by INDEX - so an off-by-one silently files BTC's funding under ETH
    and every carry number downstream is wrong while every file looks
    well-formed. The venue changing shape must be loud, not lossy."""
    from capture.venues.hyperliquid import HyperliquidVenue

    v = HyperliquidVenue()
    spec = v.poll_specs(["BTC"])[0]

    assert v.fan_out_poll(spec, [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [{"funding": "0.001"}],
    ]) == [], "zipped a mismatched response instead of refusing it"


def test_a_response_of_the_wrong_shape_splits_into_nothing():
    from capture.venues.hyperliquid import HyperliquidVenue

    v = HyperliquidVenue()
    spec = v.poll_specs(["BTC"])[0]

    assert v.fan_out_poll(spec, {"universe": []}) == []
    assert v.fan_out_poll(spec, [{"universe": [{"name": "BTC"}]}]) == []
    assert v.fan_out_poll(spec, [{"nope": []}, []]) == []
