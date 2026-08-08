"""Calibrating the queue share from measurement, not from a round number.

`fill_model` takes a participation fraction and caps every fill by it. Where that
number comes from decides whether paper is a measurement or a wish: an invented
rate is exactly what `ARCHITECTURE.md:92` means by paper mode manufacturing edge.

So it is measured off the depth archive, and where the archive is silent the
calibration **refuses** rather than defaulting. A refusal propagates; a default
would be indistinguishable downstream from a measurement.

Design: `docs/superpowers/specs/2026-08-08-paper-execution-engine-design.md`.
"""
import json
from decimal import Decimal

import pandas as pd
import pytest

from paper.participation_calibration import (
    Calibration,
    CalibrationRefused,
    calibrate_participation,
    record_calibration_receipt,
)
from store.clock_gated_reader import ClockGatedReader
from validation.holdout_custodian import HoldoutCustodian, HoldoutSealed
from store.parquet_partition import append_partition

MINUTE_NS = 60_000_000_000
T0 = 1_786_000_000_000_000_000


def a_bar_row(symbol="BTCUSDT", venue="binance", at_ns=T0, volume=100.0, trades=10):
    """One 60s bar. `trades` matters: it is what makes the print size knowable."""
    return {"symbol": symbol, "venue": venue, "event_time_ns": at_ns,
            "ingestion_time_ns": at_ns, "availability_time_ns": at_ns,
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "volume": volume, "trades": trades}


def a_book_row(symbol="BTCUSDT", venue="binance", at_ns=T0,
               bid_size="5", ask_size="5"):
    """Levels as JSON strings, which is what the store actually holds.

    Not a stylistic choice. Writing nested lists here produced a fixture the code
    passed against and the real dataset did not: `book_snapshots` serialises the
    levels to JSON, so a test built from lists exercises a shape that never
    reaches production.
    """
    return {"symbol": symbol, "venue": venue, "event_time_ns": at_ns,
            "ingestion_time_ns": at_ns, "availability_time_ns": at_ns,
            "bids": json.dumps([["99.99", bid_size]]),
            "asks": json.dumps([["100.01", ask_size]]),
            "last_update_id": 1}


def a_store(tmp_path, bars=(), books=()):
    if bars:
        append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(list(bars)),
                         snapshot_id="bars-1")
    if books:
        append_partition(tmp_path, "book", pd.DataFrame(list(books)),
                         snapshot_id="book-1")
    return tmp_path


def readers(store_root):
    """Both datasets through the one door every market-data read goes through."""
    return (ClockGatedReader(store_root, "book"),
            ClockGatedReader(store_root, "bars_60000000000ns"))


# --- silence is a refusal, never a default -----------------------------------

def test_calibration_refuses_when_the_depth_archive_holds_no_book(tmp_path):
    """Tier 1 has 2,123 symbols and depth for three of them.

    Defaulting the other 2,120 to some plausible-looking fraction would put an
    invented number into every fill they produce, indistinguishable from the three
    that were actually measured.
    """
    store = a_store(tmp_path, bars=[a_bar_row()])
    book_reader, bar_reader = readers(store)

    result = calibrate_participation(
        book_reader, bar_reader, venue="binance", symbol="BTCUSDT",
        as_of_ns=T0 + MINUTE_NS, bar_ns=MINUTE_NS)

    assert isinstance(result, CalibrationRefused)
    assert result.missing == "book"


# --- what the number actually means ------------------------------------------

def test_the_fraction_is_the_volume_left_after_the_queue_ahead_is_consumed(tmp_path):
    """Price-time priority is the whole mechanism.

    An order joining the touch sits behind the size already resting there, and
    everything that arrives afterwards queues behind *it*. So the volume it can
    expect over a bar is what is left once the queue ahead has been consumed:
    100 traded against 5 resting leaves 95, a 0.95 share.

    The queue ahead is taken from the deeper of the two sides, not the nearer: a
    strategy trades both, and the flattering choice is the one that would make
    paper fills look easy.
    """
    store = a_store(tmp_path,
                    bars=[a_bar_row(at_ns=T0, volume=100.0, trades=10)],
                    books=[a_book_row(at_ns=T0 + 1, bid_size="5", ask_size="5")])
    book_reader, bar_reader = readers(store)

    result = calibrate_participation(
        book_reader, bar_reader, venue="binance", symbol="BTCUSDT",
        as_of_ns=T0 + 2 * MINUTE_NS, bar_ns=MINUTE_NS)

    assert result.fraction == Decimal("0.95")
    assert result.n_observations == 1


def test_calibration_refuses_when_no_bar_overlaps_a_depth_snapshot(tmp_path):
    """Depth present and volume present is not the same as depth *during* volume.

    Snapshots are polled at roughly 30s and bars are 60s, so a stretch of trading
    with no snapshot inside it is ordinary rather than exceptional. Calibrating off
    the bars that happen to line up and reporting the rest as covered would be a
    number describing a different window than the one it claims.
    """
    store = a_store(tmp_path,
                    bars=[a_bar_row(at_ns=T0, volume=100.0, trades=10)],
                    books=[a_book_row(at_ns=T0 + 10 * MINUTE_NS)])
    book_reader, bar_reader = readers(store)

    result = calibrate_participation(
        book_reader, bar_reader, venue="binance", symbol="BTCUSDT",
        as_of_ns=T0 + 20 * MINUTE_NS, bar_ns=MINUTE_NS)

    assert isinstance(result, CalibrationRefused)
    assert result.missing == "overlapping bars"


def calibrate(store, as_of_ns):
    book_reader, bar_reader = readers(store)
    return calibrate_participation(
        book_reader, bar_reader, venue="binance", symbol="BTCUSDT",
        as_of_ns=as_of_ns, bar_ns=MINUTE_NS)


def test_a_queue_deeper_than_the_bar_volume_gives_a_zero_share(tmp_path):
    """Nothing reached us, which is a share of zero and never a negative one.

    A negative share would flow straight into `fill_model` as a negative fill
    quantity — a paper order that sells by buying.
    """
    store = a_store(tmp_path,
                    bars=[a_bar_row(at_ns=T0, volume=4.0, trades=2)],
                    books=[a_book_row(at_ns=T0 + 1, bid_size="5", ask_size="5")])

    assert calibrate(store, T0 + 2 * MINUTE_NS).fraction == Decimal("0")


def test_the_deeper_side_of_the_touch_sets_the_queue_ahead(tmp_path):
    """Five resting on the bid and forty on the offer is a forty-deep queue.

    Taking the thin side would report the easier half of the book as though it
    were the whole of it, and every fill downstream would inherit that.
    """
    store = a_store(tmp_path,
                    bars=[a_bar_row(at_ns=T0, volume=100.0, trades=10)],
                    books=[a_book_row(at_ns=T0 + 1, bid_size="5", ask_size="40")])

    assert calibrate(store, T0 + 2 * MINUTE_NS).fraction == Decimal("0.6")


def test_the_fraction_is_the_median_across_bars_not_the_mean(tmp_path):
    """One quiet minute against a deep book should not drag the estimate.

    Bar volume is heavy-tailed, so a mean is set by whichever few minutes happened
    to be busy. The median is the number that describes a typical minute, which is
    what a resting order actually faces.
    """
    bars = [a_bar_row(at_ns=T0 + i * MINUTE_NS, volume=v, trades=10)
            for i, v in enumerate((100.0, 100.0, 100.0))]
    books = [a_book_row(at_ns=T0 + i * MINUTE_NS + 1, bid_size=q, ask_size=q)
             for i, q in enumerate(("5", "40", "1000"))]
    store = a_store(tmp_path, bars=bars, books=books)

    result = calibrate(store, T0 + 10 * MINUTE_NS)

    assert result.n_observations == 3
    assert result.fraction == Decimal("0.6")      # mean would be 0.5166...


# --- the sealed range is sealed for this reader too --------------------------

@pytest.mark.parametrize("sealed_dataset", ["book", "bars_60000000000ns"])
def test_calibrating_inside_the_sealed_holdout_is_refused(tmp_path, sealed_dataset):
    """Calibration is a look at the data like any other.

    A participation rate tuned on the holdout would carry the holdout's shape into
    every fill of every strategy scored against it — a leak that never shows up as
    a backtest reading the holdout, because on paper none ever did.

    Each dataset is sealed on its own: sealing both together would still pass if
    one of the two reads had been quietly rewritten into a direct Parquet read,
    because the other would raise and hide it.
    """
    store = a_store(tmp_path,
                    bars=[a_bar_row(at_ns=T0, volume=100.0, trades=10)],
                    books=[a_book_row(at_ns=T0 + 1)])
    custodian = HoldoutCustodian(tmp_path / "custody", holdout_start_ns=T0,
                                 holdout_end_ns=T0 + 100 * MINUTE_NS)
    guarded = {sealed_dataset: custodian}

    def reader(dataset):
        return ClockGatedReader(store, dataset, custodian=guarded.get(dataset))

    with pytest.raises(HoldoutSealed):
        calibrate_participation(reader("book"), reader("bars_60000000000ns"),
                                venue="binance", symbol="BTCUSDT",
                                as_of_ns=T0 + 2 * MINUTE_NS, bar_ns=MINUTE_NS)


# --- the receipt, which is what the wall is allowed to read ------------------

def a_calibration(fraction="0.95", symbol="BTCUSDT"):
    return Calibration(venue="binance", symbol=symbol, fraction=Decimal(fraction),
                       event_unit_ns=MINUTE_NS, n_observations=3, as_of_ns=T0,
                       detail="median over 3 bars")


def test_the_receipt_carries_the_fraction_and_the_unit_it_was_measured_on(tmp_path):
    """Rule 8: the number and its proof travel together or neither is usable.

    Serialised as a string, not a float: the fraction multiplies every fill
    quantity, and a receipt that round-trips 0.95 into 0.9499999 puts a
    representation error into the size of every paper trade.
    """
    path = record_calibration_receipt(tmp_path, [a_calibration()],
                                      measured_at_ns=T0 + MINUTE_NS)

    receipt = json.loads(path.read_text(encoding="utf-8"))
    entry = receipt["symbols"]["binance:BTCUSDT"]
    assert receipt["measured_at_ns"] == T0 + MINUTE_NS
    assert entry["measured"] is True
    assert entry["fraction"] == "0.95"
    assert entry["event_unit_ns"] == MINUTE_NS
    assert entry["n_observations"] == 3


def test_a_refused_symbol_is_recorded_with_no_fraction_at_all(tmp_path):
    """Rule 8: absence of evidence renders as its own state, never as a number.

    A refusal that carried some placeholder fraction would let the wall draw a
    tile, and a drawn tile reads as a measurement. The symbol has to be visibly
    unmeasured, with the reason attached.
    """
    refusal = CalibrationRefused(venue="binance", symbol="DOGEUSDT",
                                 missing="book", reason="no depth snapshots")

    path = record_calibration_receipt(tmp_path, [a_calibration(), refusal],
                                      measured_at_ns=T0)

    entry = json.loads(path.read_text(encoding="utf-8"))["symbols"]["binance:DOGEUSDT"]
    assert entry["measured"] is False
    assert "fraction" not in entry
    assert entry["missing"] == "book"
