"""Deribit option-chain capture — the third segment, which had no data at all.

The frames below are the real shape, measured against the live venue on
2026-08-16, nulls included. The nulls are the point: `bid_price` is null on 89 of
818 BTC instruments and `high`/`low` on 588, and the prior-art client in
`nse-crypto-bot-final` coerced exactly those with `float(x or 0.0)` - turning a
missing bid into a bid of zero, which makes a worthless option look free.
"""
import pytest

from capture.cli import _VENUES, archive_name_for
from capture.venues import PollSpec
from capture.venues.deribit import (
    OPTION_CURRENCIES,
    DeribitPushesNothingHere,
    DeribitVenue,
)

# One deep out-of-the-money strike with no bid and no traded range, and one
# liquid strike. Both verbatim from the live response.
_NO_BID = {
    "high": None, "low": None, "last": None, "price_change": None,
    "instrument_name": "BTC-26DEC26-400000-C", "bid_price": None,
    "ask_price": 0.0005, "mid_price": None, "open_interest": 0.0,
    "interest_rate": 0.0, "mark_price": 0.00021, "creation_timestamp": 1786898049283,
    "estimated_delivery_price": 63305.95, "volume": 0.0, "mark_iv": 78.11,
    "underlying_price": 64312.37, "underlying_index": "BTC-26DEC26",
    "base_currency": "BTC", "quote_currency": "BTC", "volume_usd": 0.0,
}
_LIQUID = {
    "high": 0.61, "low": 0.55, "last": 0.579, "price_change": 1.2,
    "instrument_name": "BTC-25DEC26-104000-P", "bid_price": 0.564,
    "ask_price": 0.6865, "mid_price": 0.62525, "open_interest": 0.1,
    "interest_rate": 0.0, "mark_price": 0.62020722,
    "creation_timestamp": 1786898049291, "estimated_delivery_price": 63305.95,
    "volume": 0.0, "mark_iv": 41.42, "underlying_price": 64312.37,
    "underlying_index": "BTC-25DEC26", "base_currency": "BTC",
    "quote_currency": "BTC", "volume_usd": 0.0,
}


def _chain(rows) -> dict:
    return {"jsonrpc": "2.0", "usIn": 1786898050001644, "usOut": 1786898050002779,
            "usDiff": 1135, "testnet": False, "result": rows}


def _spec() -> PollSpec:
    return DeribitVenue().poll_specs([])[0]


# --- the venue is reachable through the same door as every other -----------

def test_deribit_is_registered_and_writes_to_its_own_archive_directory():
    """A venue class that exists but is not in `_VENUES` is unreachable - the
    trap this codebase names: built, tested, called by nothing."""
    assert _VENUES["deribit"] is DeribitVenue
    assert archive_name_for("deribit") == "deribit"


# --- nulls survive capture, because they are real market structure ---------

def test_a_null_bid_is_carried_through_untouched(tmp_path):
    """An option with no bid is one nobody will buy at any price - ordinary for a
    far out-of-the-money strike, and NOT a bid of zero.

    The prior-art client did `float(info.get("bid_price") or 0.0)` on this exact
    field. On the measured chain that invents a bid for 89 of 818 instruments,
    every one of them in the direction that flatters a trade.
    """
    pairs = DeribitVenue().fan_out_poll(_spec(), _chain([_NO_BID]))

    assert len(pairs) == 1
    symbol, row = pairs[0]
    assert symbol == "BTC-26DEC26-400000-C"
    assert row["bid_price"] is None, "a missing bid must stay missing"
    assert row["mid_price"] is None
    assert row["high"] is None and row["low"] is None


def test_rows_are_passed_through_unchanged(tmp_path):
    """No envelope clock is merged in, unlike Bybit and Hyperliquid.

    Each row already carries `creation_timestamp` - the summary's generation
    time, measured as 9 distinct values spanning 8ms across 818 rows. Merging the
    envelope's `usOut` beside it would put a second, differently-derived
    timestamp on every record and leave the reader to choose.
    """
    pairs = DeribitVenue().fan_out_poll(_spec(), _chain([_LIQUID]))

    assert pairs[0][1] == _LIQUID
    assert "usOut" not in pairs[0][1] and "usIn" not in pairs[0][1]


def test_every_instrument_gets_its_own_record():
    pairs = DeribitVenue().fan_out_poll(_spec(), _chain([_NO_BID, _LIQUID]))

    assert [symbol for symbol, _ in pairs] == [
        "BTC-26DEC26-400000-C", "BTC-25DEC26-104000-P"]


# --- an error body must not be filed as a chain ---------------------------

def test_a_json_rpc_error_yields_nothing_rather_than_being_split():
    """Deribit answers a bad request with HTTP 200 and an `error` member. Split
    blindly, that files an error object under every instrument in the last good
    chain - a failure wearing a successful response, which is the shape this
    archive keeps being bitten by."""
    error_body = {"jsonrpc": "2.0", "error": {"code": 10009,
                                              "message": "not_enough_funds"},
                  "usIn": 1, "usOut": 2, "testnet": False}

    assert DeribitVenue().fan_out_poll(_spec(), error_body) == []


def test_a_result_that_is_not_a_list_yields_nothing():
    assert DeribitVenue().fan_out_poll(_spec(), _chain({"unexpected": 1})) == []


def test_a_row_with_no_instrument_name_is_skipped_not_named():
    """Generating a name would put a record in the archive under a symbol that
    does not exist on the venue."""
    pairs = DeribitVenue().fan_out_poll(
        _spec(), _chain([{**_LIQUID, "instrument_name": None}, _LIQUID]))

    assert [symbol for symbol, _ in pairs] == ["BTC-25DEC26-104000-P"]


# --- polled, never subscribed ---------------------------------------------

def test_nothing_subscribes_a_websocket():
    """Deribit has one. Using it would mean 1,502 channels to fetch what a single
    request already returns, so asking for a URL is a bug and says so."""
    venue = DeribitVenue()

    assert venue.core_specs(["BTC"]) == []
    assert venue.tail_specs(["BTC"]) == []
    with pytest.raises(DeribitPushesNothingHere):
        venue.ws_url([])
    with pytest.raises(DeribitPushesNothingHere):
        venue.subscribe_messages([])


def test_one_fan_out_poll_per_currency_that_actually_lists_options():
    """BTC 818 instruments and ETH 684, measured; the other 49 currencies Deribit
    lists have zero options. Polling one of those would report a captured venue
    forever while returning nothing."""
    specs = DeribitVenue().poll_specs(["ignored"])

    assert len(specs) == len(OPTION_CURRENCIES) == 2
    for spec, currency in zip(specs, OPTION_CURRENCIES):
        assert spec.fan_out is True
        assert f"currency={currency}" in spec.url
        assert "kind=option" in spec.url
        # Names the request. No file is ever written under it.
        assert spec.symbol == "ALL"


def test_the_poll_cadence_is_not_faster_than_the_edge_cache():
    """`cache-control: public, max-age=1`, so a faster poll cannot return
    anything newer - and the real cost is the 1,502 files one response fans out
    into, which killed capture outright once already at a one-second cadence."""
    for spec in DeribitVenue().poll_specs([]):
        assert spec.interval_seconds >= 60.0


def test_a_frame_the_venue_cannot_explain_is_control_not_a_symbol():
    meta = DeribitVenue().extract({"anything": 1})

    assert meta.kind == "control"
    assert meta.symbol == "unknown"
