"""Deribit chain polls into a clock-gated option dataset.

The frames are the real shape, measured against the live venue on 2026-08-16.

The tests that matter most are the ones about ABSENCE. `bid_price` is null on 89
of 818 BTC instruments while `mark_price` is genuinely 0.0 on others, and the
prior-art client collapsed both into zero with `float(x or 0.0)` - which makes a
worthless option look free and is one more prior-art defect failing in the
flattering direction.
"""
import datetime as dt

import pandas as pd
import pytest

from store.option_chain import (
    DATASET,
    UnparsableInstrumentName,
    build_option_frame,
    extract_option_quotes,
    parse_instrument_name,
)
from store.temporal_schema import validate_temporal_frame


class _Entry:
    """The one field the extractor reads off an index entry."""
    def __init__(self, t_recv_ns: int) -> None:
        self.t_recv_ns = t_recv_ns


_CREATED_MS = 1786898049283
_RECV_NS = 1786898049500_000_000

_LIQUID = {
    "high": 0.61, "low": 0.55, "last": 0.579, "price_change": 1.2,
    "instrument_name": "BTC-25DEC26-104000-P", "bid_price": 0.564,
    "ask_price": 0.6865, "mid_price": 0.62525, "open_interest": 0.1,
    "interest_rate": 0.0, "mark_price": 0.62020722,
    "creation_timestamp": _CREATED_MS, "estimated_delivery_price": 63305.95,
    "volume": 0.0, "mark_iv": 41.42, "underlying_price": 64312.37,
    "underlying_index": "BTC-25DEC26", "base_currency": "BTC",
    "quote_currency": "BTC", "volume_usd": 0.0,
}
# A far out-of-the-money strike: no bid, no traded range, and a mark of exactly
# zero. Both kinds of "nothing" on one row, which is the case that must not blur.
_NO_BID_ZERO_MARK = {
    **_LIQUID,
    "instrument_name": "BTC-17AUG26-56000-P",
    "bid_price": None, "mid_price": None, "high": None, "low": None,
    "last": None, "price_change": None, "mark_price": 0.0, "mark_iv": 54.34,
}


def _quotes(*bodies):
    import json
    out = []
    for body in bodies:
        out.extend(extract_option_quotes(json.dumps(body), _Entry(_RECV_NS),
                                         "deribit", "ALL"))
    return out


# --- absence is not zero, and both occur ---------------------------------

def test_a_missing_bid_stays_missing_and_a_real_zero_mark_stays_zero():
    """The single most important behaviour in this module.

    `float(info.get("bid_price") or 0.0)` - the prior art - turns 89 of 818
    missing bids into a bid of zero, a valid-looking price that prices any spread
    built on it at 100% of its value. Meanwhile a mark of exactly 0.0 is REAL: a
    deep out-of-the-money strike days from expiry marks at nothing.
    """
    quote = _quotes(_NO_BID_ZERO_MARK)[0]

    assert quote.bid_price is None, "an absent bid must not become a price"
    assert quote.mid_price is None
    assert quote.mark_price == 0.0, "a real zero must not become an absence"


def test_absence_and_zero_stay_distinguishable_in_the_frame():
    """NaN in a float column, not a sentinel. An integer or nullable-int dtype
    would force a choice between the two and lose whichever it did not pick."""
    frame = build_option_frame(_quotes(_NO_BID_ZERO_MARK, _LIQUID))

    absent = frame[frame.symbol == "BTC-17AUG26-56000-P"].iloc[0]
    assert pd.isna(absent.bid_price)
    assert absent.mark_price == 0.0
    assert not pd.isna(absent.mark_price)


def test_a_row_missing_a_field_the_venue_always_populates_is_refused():
    """`mark_price`, `mark_iv`, `underlying_price`, `open_interest` and `volume`
    were never null across 1,502 measured instruments, so a null means the
    venue's shape changed. Defaulting one would price every position built on
    it."""
    for field in ("mark_price", "mark_iv", "underlying_price", "open_interest",
                  "volume"):
        assert _quotes({**_LIQUID, field: None}) == [], f"{field} was defaulted"


# --- units, stated in the name -------------------------------------------

def test_implied_vol_is_converted_from_percent_to_a_fraction_once():
    """The venue reports 41.42 meaning 41.42%. A column called `mark_iv` holding
    41.42 next to a model expecting 0.4142 is wrong by a factor of 100 in a way
    no test of the column's presence would catch, so the unit is in the name."""
    quote = _quotes(_LIQUID)[0]

    assert quote.mark_iv_fraction == pytest.approx(0.4142)


def test_the_quote_currency_is_carried_because_these_are_not_dollar_prices():
    """A BTC option's mark price is a fraction of a COIN: 0.62 means 0.62 BTC,
    not $0.62. `underlying_price` rides along so the conversion is always
    possible, and nothing can apply the dollar default by accident."""
    quote = _quotes(_LIQUID)[0]

    assert quote.quote_currency == "BTC"
    assert quote.underlying_price == pytest.approx(64312.37)


# --- instrument names are parsed strictly, never guessed ------------------

def test_an_instrument_name_yields_underlying_type_strike_and_expiry():
    underlying, option_type, strike, expiry_ns = parse_instrument_name(
        "BTC-25DEC26-104000-P")

    assert (underlying, option_type, strike) == ("BTC", "P", 104000.0)
    assert dt.datetime.fromtimestamp(expiry_ns / 1e9, dt.timezone.utc) == \
        dt.datetime(2026, 12, 25, 8, tzinfo=dt.timezone.utc)


def test_expiry_is_the_settlement_instant_not_midnight():
    """Deribit settles at 08:00 UTC. Midnight would overstate time-to-expiry by
    twelve hours, which is most of the remaining life of a daily option."""
    _, _, _, expiry_ns = parse_instrument_name("BTC-17AUG26-56000-C")

    assert dt.datetime.fromtimestamp(expiry_ns / 1e9, dt.timezone.utc).hour == 8


def test_a_single_digit_expiry_day_parses():
    """`ETH-4SEP26-2400-P` is a real captured name; a two-digit-only pattern
    would refuse
    every option expiring before the tenth of a month."""
    _, _, _, expiry_ns = parse_instrument_name("ETH-4SEP26-2400-P")

    assert dt.datetime.fromtimestamp(expiry_ns / 1e9, dt.timezone.utc).day == 4


@pytest.mark.parametrize("name", [
    "BTC-25DEC26-104000",            # no option type
    "BTC-25DEC26-104000-X",          # not a call or a put
    "BTC-PERPETUAL",                 # a future, not an option
    "BTC-25XXX26-104000-P",          # not a month
    "BTC-25DEC26-104000-P-EXTRA",    # trailing content the pattern cannot explain
    "",
])
def test_a_name_this_module_cannot_fully_explain_is_refused(name):
    """Anchored at both ends. A strike read out of a name the venue meant
    differently misprices every surface built on it."""
    with pytest.raises(UnparsableInstrumentName):
        parse_instrument_name(name)


def test_an_unparsable_instrument_costs_only_its_own_row():
    """One bad instrument must not cost the poll."""
    assert _quotes({**_LIQUID, "instrument_name": "BTC-PERPETUAL"}) == []
    assert len(_quotes({**_LIQUID, "instrument_name": "BTC-PERPETUAL"},
                       _LIQUID)) == 1


# --- routing: a frame from another stream is not read as a chain ---------

def test_a_frame_without_the_fields_this_endpoint_is_polled_for_is_ignored():
    """The archive holds several streams side by side. `mark_iv` is what
    distinguishes an option summary from the futures summary the same endpoint
    shape serves."""
    assert _quotes({"symbol": "BTCUSDT", "lastFundingRate": "0.0001",
                    "markPrice": "64000"}) == []
    assert _quotes({**_LIQUID, "mark_iv": None}) == []


def test_a_record_with_no_venue_clock_is_refused_not_stamped_from_receipt():
    """Stamping receipt as the event time invents a venue clock and silently
    reorders this tape against the other venues'."""
    assert _quotes({**_LIQUID, "creation_timestamp": None}) == []
    assert _quotes({**_LIQUID, "creation_timestamp": "1786898049283"}) == []


def test_malformed_json_is_ignored_rather_than_raised():
    assert extract_option_quotes("{not json", _Entry(_RECV_NS),
                                 "deribit", "ALL") == []


# --- the temporal contract ------------------------------------------------

def test_a_polled_snapshot_is_knowable_the_moment_it_lands():
    """Availability is ingestion EXACTLY, not a max over anything. There is no
    bar to close and nothing to wait for - the same words the funding dataset
    uses, because it is the same fact."""
    frame = build_option_frame(_quotes(_LIQUID))

    assert frame.availability_time_ns.iloc[0] == _RECV_NS
    assert frame.ingestion_time_ns.iloc[0] == _RECV_NS
    # The venue stamped it earlier than we received it, which is the round trip.
    assert frame.event_time_ns.iloc[0] == _CREATED_MS * 1_000_000
    assert frame.event_time_ns.iloc[0] < frame.availability_time_ns.iloc[0]


def test_the_frame_satisfies_the_clock_gate():
    validate_temporal_frame(build_option_frame(_quotes(_LIQUID,
                                                       _NO_BID_ZERO_MARK)))


def test_an_empty_build_still_carries_the_columns():
    """A dataset whose empty partition has no columns makes every reader of it
    branch on emptiness."""
    frame = build_option_frame([])

    assert frame.empty
    for column in ("symbol", "venue", "strike", "mark_iv_fraction",
                   "bid_price", "availability_time_ns"):
        assert column in frame.columns


def test_the_builder_registry_names_this_dataset():
    """A dataset with an extractor the builder cannot reach is exactly what
    happened to coinbase for a day."""
    from store.build_polled import DATASETS

    stream, extract, build = DATASETS[DATASET]
    assert stream == "optionChain"
    assert extract is extract_option_quotes
    assert build is build_option_frame
