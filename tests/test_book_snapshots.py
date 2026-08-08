"""Archived depth snapshots, turned into a clock-gated book dataset.

Built from snapshots rather than by replaying diffs onto them. A snapshot is a
true book as the venue reported it, once a minute; diff replay would give
sub-second resolution and is a much larger, much riskier piece of work - one
mis-sequenced update and the book silently diverges from reality with nothing
to compare it against. A book that is right once a minute is worth more than a
book that is plausible continuously.

The leakage rule is the same as funding: a book is knowable when we received
it, never when the venue stamped it.
"""
import json
from decimal import Decimal

import pytest

from store.book_snapshots import (
    BookSnapshot,
    build_book_frame,
    extract_book_snapshot,
    levels_from_row,
)
from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    validate_temporal_frame,
)


class FakeEntry:
    def __init__(self, t_recv_ns: int) -> None:
        self.t_recv_ns = t_recv_ns


# Shaped exactly like the live response measured 2026-08-08. Futures carries E
# and T; spot carries neither, which is why event time falls back to receipt.
FUTURES_SNAPSHOT = json.dumps({
    "lastUpdateId": 11241165546628, "E": 1786184000123, "T": 1786184000100,
    "bids": [["64971.20", "1.5"], ["64971.10", "3.0"]],
    "asks": [["64971.40", "2.0"], ["64971.50", "4.0"]],
})

SPOT_SNAPSHOT = json.dumps({
    "lastUpdateId": 98359081841,
    "bids": [["64971.23", "0.17"]], "asks": [["65019.99", "0.28"]],
})


def test_a_snapshot_becomes_one_book_row():
    got = extract_book_snapshot(FUTURES_SNAPSHOT, FakeEntry(1_786_184_000_500_000_000),
                                venue="binance", symbol="BTCUSDT")
    assert len(got) == 1
    assert got[0].symbol == "BTCUSDT"
    assert got[0].last_update_id == 11241165546628


def test_a_frame_that_is_not_a_snapshot_yields_nothing():
    """The archive holds several streams. A depth *diff* carries `e` and `U`;
    misreading one as a book would price against a changeset."""
    diff = json.dumps({"data": {"e": "depthUpdate", "U": 1, "u": 2,
                                "b": [["1", "1"]], "a": [["2", "1"]]}})
    assert extract_book_snapshot(diff, FakeEntry(1), venue="binance",
                                 symbol="BTCUSDT") == []


def test_a_book_is_knowable_when_we_received_it():
    """The venue stamped this 377ms before it reached us. Availability keyed on
    the venue clock lets a backtest price against a book that had not arrived."""
    received = 1_786_184_000_500_000_000
    frame = build_book_frame(extract_book_snapshot(
        FUTURES_SNAPSHOT, FakeEntry(received), venue="binance", symbol="BTCUSDT"))
    row = frame.iloc[0]
    assert row[EVENT_TIME] == 1_786_184_000_123_000_000
    assert row[AVAILABILITY_TIME] == received


def test_a_venue_that_stamps_nothing_falls_back_to_receipt():
    """Spot's snapshot carries no E or T at all. Inventing an event time would
    be worse than admitting we only know when it landed."""
    received = 1_786_184_000_500_000_000
    got = extract_book_snapshot(SPOT_SNAPSHOT, FakeEntry(received),
                                venue="binance-spot", symbol="BTCUSDT")
    assert got[0].event_time_ns == received


def test_the_frame_satisfies_the_bitemporal_contract():
    frame = build_book_frame(extract_book_snapshot(
        FUTURES_SNAPSHOT, FakeEntry(1_786_184_000_500_000_000),
        venue="binance", symbol="BTCUSDT"))
    validate_temporal_frame(frame)
    assert INGESTION_TIME in frame.columns


def test_levels_survive_the_round_trip_as_decimals():
    """The whole point of the dataset. A float here puts a representation error
    into the spread that gates every strategy."""
    frame = build_book_frame(extract_book_snapshot(
        FUTURES_SNAPSHOT, FakeEntry(1), venue="binance", symbol="BTCUSDT"))
    bids, asks = levels_from_row(frame.iloc[0])
    assert bids[0] == (Decimal("64971.20"), Decimal("1.5"))
    assert asks[0] == (Decimal("64971.40"), Decimal("2.0"))


def test_levels_come_back_best_first():
    """Impact walks the book from the touch outward. Reversed levels would
    price a large clip against the far side of the book and report it cheap."""
    frame = build_book_frame(extract_book_snapshot(
        FUTURES_SNAPSHOT, FakeEntry(1), venue="binance", symbol="BTCUSDT"))
    bids, asks = levels_from_row(frame.iloc[0])
    assert bids[0][0] > bids[1][0], "bids must descend from the touch"
    assert asks[0][0] < asks[1][0], "asks must ascend from the touch"


def test_no_snapshots_is_an_empty_frame():
    assert build_book_frame([]).empty


def test_an_empty_side_is_refused_not_stored():
    """A book with no bid has no mid, and storing it would put a row in the
    dataset that every consumer has to defend against."""
    one_sided = json.dumps({"lastUpdateId": 1, "bids": [], "asks": [["1", "1"]]})
    assert extract_book_snapshot(one_sided, FakeEntry(1), venue="binance",
                                 symbol="BTCUSDT") == []
