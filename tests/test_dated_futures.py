"""The expiring half of the bybit ticker poll, which nothing used to read.

Both frames below are real, taken from this box's own archive on 2026-08-10.
That matters for the central test here: the perpetual and the dated contract
arrive in the SAME response, on the same stream, under the same venue, and the
only thing separating them is what the venue says about them. A reader that
told them apart by the `-14AUG26` suffix would work until bybit renamed
something, and the failure would be silent - dated contracts would simply stop
appearing on the curve.
"""
import json
from decimal import Decimal

import pandas as pd

from store.dated_futures import (
    DatedFutureObservation,
    build_dated_futures_frame,
    extract_bybit_dated_future,
)
from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    SYMBOL,
    VENUE,
    validate_temporal_frame,
)


class FakeEntry:
    """Stands in for the archive index entry, which carries receipt time."""

    def __init__(self, t_recv_ns: int) -> None:
        self.t_recv_ns = t_recv_ns


RECV_NS = 1_786_336_806_435_346_403

# A real dated contract, archived 2026-08-10T04.
REAL_DATED = json.dumps({
    "symbol": "BTCUSDT-14AUG26", "lastPrice": "64976.8",
    "indexPrice": "64999.5", "markPrice": "65011.9", "openInterest": "186.288",
    "fundingRate": "", "nextFundingTime": "0", "predictedDeliveryPrice": "0.0",
    "basisRate": "0.00011907", "deliveryFeeRate": "0",
    "deliveryTime": "1786694400000", "basis": "-22.71",
    "fundingIntervalHour": "", "basisRateYear": "0.01723021",
    "time": 1786336806394,
})

# The perpetual from the same response. Note `deliveryTime` is the string "0"
# and `basisRateYear` is empty - the venue is explicit about which is which.
REAL_PERPETUAL = json.dumps({
    "symbol": "BTCUSDT", "lastPrice": "64965.30", "indexPrice": "64999.51",
    "markPrice": "64970.71", "openInterest": "59657.919",
    "fundingRate": "0.00005399", "nextFundingTime": "1786348800000",
    "predictedDeliveryPrice": "", "basisRate": "", "deliveryTime": "0",
    "basis": "", "fundingIntervalHour": "8", "basisRateYear": "",
    "time": 1786336806394,
})


def _extract(payload, recv_ns=RECV_NS):
    return extract_bybit_dated_future(payload, FakeEntry(recv_ns),
                                      venue="bybit", symbol="BTCUSDT-14AUG26")


# --- telling the two instruments apart ------------------------------------

def test_a_dated_contract_becomes_one_observation():
    got = _extract(REAL_DATED)
    assert len(got) == 1
    assert got[0].symbol == "BTCUSDT-14AUG26"
    assert got[0].mark_price == Decimal("65011.9")
    assert got[0].index_price == Decimal("64999.5")
    # Milliseconds on the wire, nanoseconds in the store, everywhere.
    assert got[0].delivery_time_ns == 1_786_694_400_000 * 1_000_000


def test_a_perpetual_from_the_same_response_is_not_a_curve_point():
    """`funding` owns it. Filing a perpetual here would put a contract with no
    expiry onto a curve keyed on time-to-expiry."""
    assert extract_bybit_dated_future(
        REAL_PERPETUAL, FakeEntry(RECV_NS), venue="bybit", symbol="BTCUSDT") == []


def test_the_split_is_the_venues_own_evidence_not_the_symbol_suffix():
    """Same payload, a symbol carrying no date, still a dated contract. The
    reader must key on the blank funding rate and the delivery time."""
    renamed = json.loads(REAL_DATED)
    renamed["symbol"] = "BTCUSDT_Q3"
    got = extract_bybit_dated_future(json.dumps(renamed), FakeEntry(RECV_NS),
                                     venue="bybit", symbol="BTCUSDT_Q3")
    assert len(got) == 1
    assert got[0].delivery_time_ns > 0


def test_neither_a_rate_nor_a_delivery_date_is_refused():
    """A frame with no funding AND no expiry is not a perpetual and not a dated
    contract. Filing it as either invents an instrument."""
    neither = json.loads(REAL_DATED)
    neither["deliveryTime"] = "0"
    assert extract_bybit_dated_future(json.dumps(neither), FakeEntry(RECV_NS),
                                      venue="bybit", symbol="X") == []


def test_a_frame_without_a_mark_price_is_refused():
    no_mark = json.loads(REAL_DATED)
    del no_mark["markPrice"]
    assert _extract(json.dumps(no_mark)) == []


def test_an_unparseable_payload_is_refused_rather_than_raising():
    """One malformed line must not take the venue-day down with it - the whole
    build would be unreported, which is indistinguishable from healthy."""
    assert _extract("not json at all") == []
    assert _extract(json.dumps([1, 2, 3])) == []


# --- what is kept, and at what precision ----------------------------------

def test_prices_keep_full_decimal_precision():
    """A float mark on a five-figure contract loses the cents the basis is
    measured in - the basis here is 12.4 on 65,011.9."""
    got = _extract(REAL_DATED)[0]
    assert got.mark_price - got.index_price == Decimal("12.4")


def test_the_venues_own_basis_arithmetic_is_stored_for_comparison():
    """Stored, never used as the answer - and this frame is why.

    Bybit's `basis` of -22.71 is *nearly* lastPrice minus indexPrice
    (64976.8 - 64999.5 = -22.70) and off by a cent, while `basisRateYear`
    annualises something nearer the mark than the last price. So the venue's
    three basis fields do not reconcile with each other or with the prices
    published in the same frame. Substituting any of them for our own
    computation would import that inconsistency and hide it.
    """
    got = _extract(REAL_DATED)[0]
    assert got.venue_basis == Decimal("-22.71")
    assert got.venue_basis_rate_year == Decimal("0.01723021")
    residual = (got.last_price - got.index_price) - got.venue_basis
    assert residual == Decimal("0.01")


def test_an_empty_optional_price_is_absent_not_zero():
    """A zero price on a curve point is not a conservative default: it prices
    the contract at a 100% discount to index, the most attractive carry the
    arithmetic can produce."""
    blank = json.loads(REAL_DATED)
    blank["indexPrice"] = ""
    blank["openInterest"] = ""
    got = _extract(json.dumps(blank))[0]
    assert got.index_price is None
    assert got.open_interest is None


# --- the stored frame -----------------------------------------------------

def _observation(**overrides) -> DatedFutureObservation:
    base = dict(symbol="BTCUSDT-14AUG26", venue="bybit",
                mark_price=Decimal("65011.9"), index_price=Decimal("64999.5"),
                last_price=Decimal("64976.8"),
                delivery_time_ns=1_786_694_400_000_000_000,
                event_time_ns=1_786_336_806_394_000_000,
                ingestion_time_ns=RECV_NS)
    base.update(overrides)
    return DatedFutureObservation(**base)


def test_the_frame_satisfies_the_temporal_contract():
    frame = build_dated_futures_frame([_observation()])
    validate_temporal_frame(frame)
    assert frame.iloc[0][VENUE] == "bybit"
    assert frame.iloc[0][SYMBOL] == "BTCUSDT-14AUG26"


def test_a_poll_is_available_the_moment_it_lands():
    """Availability is our receipt, never the venue's stamp. The venue's clock
    runs ahead of ours by the round trip, and keying availability on it would
    let a backtest read a price before it arrived."""
    frame = build_dated_futures_frame([_observation()])
    row = frame.iloc[0]
    assert row[AVAILABILITY_TIME] == row[INGESTION_TIME] == RECV_NS
    assert row[EVENT_TIME] < row[INGESTION_TIME]


def test_no_observations_is_an_empty_frame():
    assert build_dated_futures_frame([]).empty


def test_an_all_null_price_column_still_has_a_string_type():
    """The defect that broke the live funding dataset: pyarrow infers an
    all-null column as type `null`, and the next partition carrying a value
    cannot unify with it - `Unsupported cast from large_string to null` made
    one venue's absent field break every venue's read."""
    frame = build_dated_futures_frame([
        _observation(index_price=None, last_price=None, open_interest=None,
                     venue_basis=None, venue_basis_rate=None,
                     venue_basis_rate_year=None)])
    for column in ("index_price", "last_price", "open_interest", "venue_basis",
                   "venue_basis_rate", "venue_basis_rate_year"):
        assert str(frame[column].dtype) == "string", column
        assert pd.isna(frame.iloc[0][column])
