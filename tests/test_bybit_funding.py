"""Bybit funding — captured, not traded, and not the same shape as either sibling.

`ARCHITECTURE.md` §3b defers Bybit as an EXECUTION venue on fee tiers, key
scopes, integration cost and its Feb 2025 record. That stands: no key exists for
it and the endpoint here is public. What it adds is a third independent funding
curve, on the argument that widened the other two — funding cannot be backfilled
into a bitemporal archive after the fact.
"""
import json
from dataclasses import dataclass
from decimal import Decimal

import pytest

from capture.venues.bybit import BybitPushesNothingHere, BybitVenue
from store.build_polled import source_for
from store.funding_rates import (
    FUNDS_ON_MARK, build_funding_frame, extract_bybit_funding,
)

RECEIVED_NS = 1786276800_000_000_000
VENUE_TIME_MS = 1786264072337


@dataclass(frozen=True)
class _Entry:
    t_recv_ns: int


def _ticker(**over):
    base = {"symbol": "BTCUSDT", "fundingRate": "-0.00002216", "markPrice": "64861.50",
            "indexPrice": "64891.20", "nextFundingTime": "1786291200000",
            "fundingIntervalHour": "8", "deliveryTime": "0", "time": VENUE_TIME_MS}
    base.update(over)
    return json.dumps(base)


def _one(payload=None):
    return extract_bybit_funding(payload or _ticker(), _Entry(RECEIVED_NS),
                                 "bybit", "BTCUSDT")[0]


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def test_nothing_subscribes_a_stream_here():
    """Poll-only. `run_capture` filters empty shards, so no websocket source is
    built at all - and asking for one must fail loudly rather than return a URL
    that would quietly connect to a feed nobody meant to capture."""
    v = BybitVenue()
    assert v.core_specs(["BTCUSDT"]) == []
    assert v.tail_specs(["BTCUSDT"]) == []
    with pytest.raises(BybitPushesNothingHere):
        v.ws_url([])
    with pytest.raises(BybitPushesNothingHere):
        v.subscribe_messages([])


def test_one_request_covers_the_linear_market():
    spec = BybitVenue().poll_specs(["BTCUSDT"])[0]
    assert spec.fan_out is True
    assert spec.interval_seconds >= 30, (
        "a fan-out poll must never inherit a one-second cadence - the cost is "
        "the ~765 files the response fans out into, not the request")


def test_the_venue_clock_is_merged_onto_every_ticker():
    """`time` is a top-level field and no ticker carries one. A record that
    cannot be dated without the envelope it arrived in is not an archive."""
    v = BybitVenue()
    pairs = v.fan_out_poll(v.poll_specs([])[0], {
        "retCode": 0, "time": VENUE_TIME_MS,
        "result": {"list": [{"symbol": "BTCUSDT", "fundingRate": "0.0001"}]}})

    assert pairs[0][0] == "BTCUSDT"
    assert pairs[0][1]["time"] == VENUE_TIME_MS


def test_an_error_body_returned_with_http_200_splits_into_nothing():
    """Bybit answers 200 with a non-zero retCode. Splitting that would file an
    error object under every instrument in it."""
    v = BybitVenue()
    spec = v.poll_specs([])[0]
    assert v.fan_out_poll(spec, {"retCode": 10001, "retMsg": "params error",
                                 "result": {"list": []}}) == []
    assert v.fan_out_poll(spec, {"retCode": 0, "result": None}) == []
    assert v.fan_out_poll(spec, ["not", "a", "dict"]) == []


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

def test_a_perpetual_reads_as_funding_on_mark():
    o = _one()
    assert o.symbol == "BTCUSDT"
    assert o.funding_rate == Decimal("-0.00002216")
    assert o.mark_price == Decimal("64861.50")
    assert o.index_price == Decimal("64891.20")
    assert o.funds_on == FUNDS_ON_MARK
    assert o.oracle_price is None


def test_the_venue_stamp_is_the_event_time_not_our_receipt():
    """Unlike Hyperliquid, this venue publishes a clock - so the flag that says
    "this is a receipt" must be false here."""
    o = _one()
    assert o.event_time_ns == VENUE_TIME_MS * 1_000_000
    assert o.ingestion_time_ns == RECEIVED_NS
    assert o.event_time_is_receipt is False


def test_the_per_symbol_funding_interval_is_carried_through():
    """The column Bybit forced. Measured 2026-08-09: 408 of its perps settle
    4-hourly, 356 8-hourly, one hourly. Annualising a 4-hourly rate as an
    8-hourly one is wrong by a factor of two, in the flattering direction."""
    assert _one(_ticker(fundingIntervalHour="4")).funding_interval_hours == 4
    assert _one().funding_interval_hours == 8


def test_a_dated_future_is_not_a_funding_row():
    """`category=linear` carries 40 dated futures beside 765 perps. A dated
    contract has no funding and says so - `fundingRate` is the empty string.
    Skipped on the venue's own say-so, not by matching a delivery suffix, which
    is a naming convention that would break the day it changed."""
    dated = _ticker(symbol="BTCUSDT-14AUG26", fundingRate="",
                    fundingIntervalHour="", nextFundingTime="0",
                    deliveryTime="1786694400000")
    assert extract_bybit_funding(dated, _Entry(RECEIVED_NS), "bybit", "X") == []


def test_a_frame_that_is_not_a_ticker_is_ignored():
    assert extract_bybit_funding("not json", _Entry(1), "bybit", "X") == []
    assert extract_bybit_funding("[]", _Entry(1), "bybit", "X") == []
    assert extract_bybit_funding(json.dumps({"symbol": "X", "px": "1"}),
                                 _Entry(1), "bybit", "X") == []


def test_the_dataset_routes_bybit_to_its_own_stream_and_reader():
    stream, reader = source_for("funding", "bybit")
    assert stream == "linearTickers"
    assert reader is extract_bybit_funding


def test_three_venues_write_a_schema_that_unifies(tmp_path):
    """The failure this guards was real and total: one venue's all-null column
    made every venue's funding unreadable. A third venue is a third chance."""
    from store.parquet_partition import append_partition, read_dataset
    from store.funding_rates import extract_funding, extract_hyperliquid_funding

    binance = json.dumps({"symbol": "BTCUSDT", "markPrice": "64965.9",
                          "indexPrice": "65007.98", "lastFundingRate": "0.00004",
                          "nextFundingTime": 1786291200000, "time": 1786276800000})
    hyper = json.dumps({"coin": "BTC", "funding": "0.0000036", "markPx": "64795.0",
                        "oraclePx": "64826.1"})

    for name, rows in (
        ("b", extract_funding(binance, _Entry(RECEIVED_NS), "binance", "BTCUSDT")),
        ("h", extract_hyperliquid_funding(hyper, _Entry(RECEIVED_NS), "hyperliquid", "BTC")),
        ("y", extract_bybit_funding(_ticker(), _Entry(RECEIVED_NS), "bybit", "BTCUSDT")),
    ):
        append_partition(tmp_path, "funding", build_funding_frame(rows), name)

    both = read_dataset(tmp_path, "funding")
    assert sorted(both["venue"].unique()) == ["binance", "bybit", "hyperliquid"]
    # And the interval column survives a partition where nobody published one.
    assert both.loc[both.venue == "bybit", "funding_interval_hours"].iloc[0] == 8
    assert both.loc[both.venue == "binance", "funding_interval_hours"].isna().all()
