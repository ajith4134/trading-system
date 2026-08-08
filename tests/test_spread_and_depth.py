"""What a book can actually fill, and what it costs to cross it.

Every number here is priced off an explicit book. There is no default spread
and no extrapolation past the last level: a book that cannot fill the notional
produces a refusal, because "the book could not fill this" is a real answer and
a guessed one is how paper mode manufactures edge that evaporates live.
"""
from decimal import Decimal

import pytest

from cost.spread_and_depth import (
    BookTooThin,
    NoBookAvailable,
    half_spread_bps,
    impact_bps,
    load_book_as_of,
)

# Prices as strings, parsed to Decimal, for the reason fee_schedule gives: a
# representation error in the number that gates every strategy is not an
# acceptable class of bug.
def book(bids, asks):
    return ([(Decimal(p), Decimal(q)) for p, q in bids],
            [(Decimal(p), Decimal(q)) for p, q in asks])


def test_half_spread_is_measured_from_the_touch():
    """Mid 100, touch 99.95/100.05 - a 10 bps spread, so 5 bps to cross half."""
    bids, asks = book([("99.95", "10")], [("100.05", "10")])
    assert half_spread_bps(bids, asks) == pytest.approx(Decimal("5"), abs=Decimal("0.001"))


def test_a_tighter_book_costs_less_to_cross():
    wide = book([("99.90", "10")], [("100.10", "10")])
    tight = book([("99.99", "10")], [("100.01", "10")])
    assert half_spread_bps(*tight) < half_spread_bps(*wide)


def test_half_spread_refuses_a_one_sided_book():
    """A book with no bid has no mid. Returning the ask alone would price a
    round trip against a price nobody is offering."""
    with pytest.raises(NoBookAvailable):
        half_spread_bps([], [(Decimal("100.05"), Decimal("10"))])


def test_a_clip_inside_the_touch_pays_only_the_half_spread():
    """Impact is what the walk costs *beyond* the touch. A clip that never
    leaves the first level has not moved the book."""
    bids, asks = book([("99.95", "1000")], [("100.05", "1000")])
    assert impact_bps(bids, asks, Decimal("100"), side="buy") == Decimal(0)


def test_a_clip_that_eats_two_levels_pays_the_weighted_average():
    """Buy 150 units: 100 at 100.00 and 50 at 101.00, against a mid of 99.995.
    The volume-weighted price is 100.3333, so impact is the distance from the
    touch, not from the mid - the half-spread already charged that."""
    bids, asks = book([("99.99", "1000")], [("100.00", "100"), ("101.00", "500")])
    impact = impact_bps(bids, asks, Decimal("15050"), side="buy")
    assert impact > Decimal(0)
    assert impact == pytest.approx(Decimal("33.2"), abs=Decimal("1.5"))


def test_a_notional_larger_than_the_book_refuses_rather_than_extrapolating():
    """The single most important behaviour in this module. Extrapolating past
    the last level invents liquidity, and a strategy sized on invented
    liquidity is sized on nothing."""
    bids, asks = book([("99.95", "1")], [("100.05", "1")])
    with pytest.raises(BookTooThin) as excinfo:
        impact_bps(bids, asks, Decimal("1000000"), side="buy")
    assert "1000000" in str(excinfo.value) or "depth" in str(excinfo.value).lower()


def test_selling_walks_the_bids_not_the_asks():
    """A book can be deep one side and thin the other, and pricing a sell off
    the ask would report the wrong cost in exactly the situation that matters."""
    bids, asks = book([("99.99", "1")], [("100.01", "100000")])
    with pytest.raises(BookTooThin):
        impact_bps(bids, asks, Decimal("100000"), side="sell")
    assert impact_bps(bids, asks, Decimal("100"), side="buy") == Decimal(0)


def test_impact_refuses_an_unknown_side():
    bids, asks = book([("99.95", "10")], [("100.05", "10")])
    with pytest.raises(ValueError):
        impact_bps(bids, asks, Decimal("100"), side="sideways")


def test_loading_a_book_refuses_when_no_book_dataset_exists(tmp_path):
    """Layer 1 currently holds trade bars and nothing else. Until a book
    dataset is built, the honest answer is a refusal naming what is missing -
    not a spread borrowed from somewhere else."""
    with pytest.raises(NoBookAvailable) as excinfo:
        load_book_as_of(tmp_path, "binance", "BTCUSDT", at_ns=1_785_648_600_000_000_000)
    assert "book" in str(excinfo.value).lower()


# --- reading the dataset once it exists --------------------------------------

def _write_book_dataset(store_root, at_ns, bid="99.95", ask="100.05"):
    import json as _json
    from store.book_snapshots import build_book_frame, extract_book_snapshot
    from store.parquet_partition import append_partition

    class E:
        t_recv_ns = at_ns

    payload = _json.dumps({"lastUpdateId": 1, "E": at_ns // 1_000_000,
                           "bids": [[bid, "10"]], "asks": [[ask, "10"]]})
    frame = build_book_frame(extract_book_snapshot(payload, E(), venue="binance",
                                                   symbol="BTCUSDT"))
    append_partition(store_root, "book", frame, snapshot_id="test")


def test_a_book_is_served_once_the_dataset_exists(tmp_path):
    """The refusal was about a missing dataset, not a permanent state."""
    at = 1_786_184_000_500_000_000
    _write_book_dataset(tmp_path, at)

    bids, asks = load_book_as_of(tmp_path, "binance", "BTCUSDT", at_ns=at)
    assert bids[0] == (Decimal("99.95"), Decimal("10"))
    assert half_spread_bps(bids, asks) == pytest.approx(Decimal("5"),
                                                        abs=Decimal("0.01"))


def test_a_book_published_after_the_moment_asked_about_is_invisible(tmp_path):
    """The leakage case through the clock gate. A backtest pricing at T must not
    see the book that arrived at T+1."""
    at = 1_786_184_000_500_000_000
    _write_book_dataset(tmp_path, at + 60 * 10**9)      # a minute in the future

    with pytest.raises(NoBookAvailable):
        load_book_as_of(tmp_path, "binance", "BTCUSDT", at_ns=at)
