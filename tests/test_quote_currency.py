"""The dollar-quote filter, and the string-parsing shortcut it refuses.

Open question 4 of the paper-execution design, settled 2026-08-08: filter to
dollar quotes rather than convert. These tests defend the two things that make
the filter trustworthy - that it reads the venue's own field, and that a quote
currency nobody has classified stays visible instead of picking a side.
"""
from __future__ import annotations

import time

import pytest

from capture.universe_tracker import UniverseTracker
from store.quote_currency import (
    DOLLAR_QUOTE_ASSETS, NON_DOLLAR_QUOTE_ASSETS, QuoteAssetsNotRecorded,
    dollar_quoted_symbols, partition_by_quote,
)


def test_the_quote_decides_not_the_symbol_string():
    """Every one of these is a real listing a suffix rule gets wrong.

    Measured live on Binance 2026-08-08. `BTCU` is BTC quoted in `U` - 46 spot
    pairs and 2 perpetuals use that quote. `XRPRLUSD` is quoted in `RLUSD`, which
    a longest-suffix table holding `USD` reads as `XRPRL/USD`, a pair that does
    not exist. `EUREURI` has base and quote that are each other's substring. And
    `USDCUSDT` is a dollar pair whose base is also a dollar quote asset.
    """
    partition = partition_by_quote({
        "BTCU": "U",                # not a dollar, despite ending in a dollar letter
        "XRPRLUSD": "RLUSD",        # IS a dollar, despite RL sitting before USD
        "EUREURI": "EURI",          # not a dollar; base and quote overlap
        "USDCUSDT": "USDT",         # a dollar pair whose base is a stablecoin
        "0GTRY": "TRY",             # the first symbol the real spot build printed
    })

    assert partition.dollar == ("USDCUSDT", "XRPRLUSD")
    assert partition.non_dollar == {"BTCU": "U", "EUREURI": "EURI", "0GTRY": "TRY"}
    assert partition.unknown == {}


def test_an_unclassified_quote_asset_is_its_own_state_not_a_side():
    """A new stablecoin must be reported, not silently dropped or silently traded.

    Folded into non-dollar it shrinks the tradeable universe on the day it lists,
    with no symptom but a count nobody watches. Folded into dollar it puts an
    unvetted denomination into a P&L. So it is neither, and `describe()` says so
    out loud - Rule 8: absence of evidence renders as its own state.
    """
    partition = partition_by_quote({
        "BTCUSDT": "USDT", "BTCNEWCOIN": "NEWCOIN", "BTCTRY": "TRY"})

    assert partition.dollar == ("BTCUSDT",)
    assert partition.unknown == {"BTCNEWCOIN": "NEWCOIN"}
    assert "1 unclassified quote asset(s)" in partition.describe()


def test_a_zero_unknown_count_is_still_printed():
    """Silence reads equally well as 'none' and as 'this does not report that'."""
    assert "0 unclassified" in partition_by_quote({"BTCUSDT": "USDT"}).describe()


def test_an_empty_quote_map_is_refused_rather_than_filtered():
    """Both defaults are wrong and both look like a working system.

    Read as "nothing is dollar-quoted" the universe empties and looks like a
    market with no candidates. Read as "everything is" it admits 312 lira-quoted
    pairs into a dollar P&L.
    """
    with pytest.raises(QuoteAssetsNotRecorded, match="empty quote map"):
        partition_by_quote({})


def test_the_two_classified_sets_do_not_overlap():
    """A quote asset in both sets would be classified by dict iteration order."""
    assert not DOLLAR_QUOTE_ASSETS & NON_DOLLAR_QUOTE_ASSETS


def test_u_is_a_real_quote_asset_and_is_classified_non_dollar():
    """The single fact that rules out parsing the quote from the symbol string."""
    assert "U" in NON_DOLLAR_QUOTE_ASSETS
    assert "U" not in DOLLAR_QUOTE_ASSETS


def test_the_partition_is_read_point_in_time_from_the_universe_snapshot(tmp_path):
    """A snapshot recorded after the moment asked about is not knowledge then.

    Same rule as `UniverseTracker.load_last`: classifying a symbol by a listing
    that had not happened yet is lookahead, and it is the reason the quote map is
    recorded with the universe rather than fetched when needed.
    """
    tracker = UniverseTracker(tmp_path, "binance-spot")
    recorded_at = time.time_ns()
    tracker.record_snapshot(
        ["BTCUSDT", "BTCTRY"], recorded_at,
        quote_assets={"BTCUSDT": "USDT", "BTCTRY": "TRY"})

    partition = dollar_quoted_symbols(tmp_path, "binance-spot", recorded_at)
    assert partition.dollar == ("BTCUSDT",)
    assert partition.non_dollar == {"BTCTRY": "TRY"}

    with pytest.raises(QuoteAssetsNotRecorded, match="record_universe_snapshot"):
        dollar_quoted_symbols(tmp_path, "binance-spot", recorded_at - 1)


def test_a_snapshot_predating_quote_capture_refuses_instead_of_guessing(tmp_path):
    """Every snapshot before 2026-08-08 has symbols and no quote map.

    Three capture processes were already running when quote assets began being
    recorded, so this is the state of the real archive, not a hypothetical.
    """
    tracker = UniverseTracker(tmp_path, "binance-spot")
    recorded_at = time.time_ns()
    tracker.record_snapshot(["BTCUSDT", "BTCTRY"], recorded_at)

    with pytest.raises(QuoteAssetsNotRecorded, match="no universe snapshot with quote assets"):
        dollar_quoted_symbols(tmp_path, "binance-spot", recorded_at)
