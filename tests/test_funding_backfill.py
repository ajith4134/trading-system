"""Reconstructed funding must be unusable as a backtest input, mechanically.

Binance serves 166 days of settled funding, which is what the §5a.4 breadth gate
needs and the live archive will not hold for months. The numbers are
recoverable; the fact that we observed them is not. They arrive with no capture
provenance and no availability time we witnessed, into a store whose whole claim
is that every row records when we could have known it.

The safeguard is not a warning. It is that `availability_time` is the FETCH, so
every simulated clock in the past sees an empty frame and the clock gate refuses
the dataset for free.
"""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from store.clock_gated_reader import ClockGatedReader
from store.funding_backfill import (
    DATASET, backfill, build_reconstructed_frame, parse_binance_history,
)
from store.parquet_partition import append_partition

SETTLED_MS = 1771891200007          # 2026-02-24, months before the archive existed
FETCHED_NS = 1786264800_000_000_000  # 2026-08-09

# The real shape, from a live page on 2026-08-09.
PAGE = json.dumps([
    {"symbol": "BTCUSDT", "fundingTime": SETTLED_MS, "fundingRate": "0.00005706",
     "markPrice": "64630.15072464", "rateType": "Regular"},
    {"symbol": "BTCUSDT", "fundingTime": SETTLED_MS + 28_800_000,
     "fundingRate": "-0.00001200", "markPrice": "64700.00", "rateType": "Regular"},
])


def _frame():
    return build_reconstructed_frame(parse_binance_history(PAGE, "binance", FETCHED_NS))


# --------------------------------------------------------------------------
# the safeguard
# --------------------------------------------------------------------------

def test_a_backtest_in_the_past_sees_nothing(tmp_path):
    """The whole design. At every simulated instant before the fetch this
    dataset is empty, so a backtest cannot consume it by accident - the clock
    gate refuses it without knowing anything about where it came from."""
    append_partition(tmp_path, DATASET, _frame(), "run1")
    reader = ClockGatedReader(tmp_path, DATASET)

    # A backtest run over the period the rows actually describe.
    assert reader.read_as_of(SETTLED_MS * 1_000_000 + 1).empty
    # And right up to the instant before we fetched them.
    assert reader.read_as_of(FETCHED_NS - 1).empty


def test_research_asking_now_sees_all_of_it(tmp_path):
    """Unusable as a backtest input, fully usable for the question it exists to
    answer - whether triggers cluster and return streams correlate."""
    append_partition(tmp_path, DATASET, _frame(), "run1")

    served = ClockGatedReader(tmp_path, DATASET).read_as_of(FETCHED_NS)
    assert len(served) == 2


def test_availability_is_the_fetch_and_never_the_settlement():
    """Stamping availability at settlement would make these rows
    indistinguishable from captured ones to every consumer. That is the one
    thing this must never do."""
    frame = _frame()
    assert set(frame["availability_time_ns"]) == {FETCHED_NS}
    assert set(frame["ingestion_time_ns"]) == {FETCHED_NS}
    # The settlement instant survives as the EVENT time, which is honest and
    # cannot leak - availability governs what a reader may see.
    assert frame["event_time_ns"].min() == SETTLED_MS * 1_000_000


def test_every_row_says_it_is_reconstructed():
    """Carried per row so a frame that is filtered, joined or copied elsewhere
    still says what it is, long after it left this dataset."""
    assert _frame()["is_reconstructed"].all()


def test_it_is_not_written_into_the_observed_funding_dataset():
    assert DATASET != "funding"
    assert "reconstructed" in DATASET


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_the_rate_survives_as_a_decimal_not_a_float():
    rows = parse_binance_history(PAGE, "binance", FETCHED_NS)
    assert rows[0].funding_rate == Decimal("0.00005706")
    assert rows[0].mark_price == Decimal("64630.15072464")
    assert rows[0].settled_at_ns == SETTLED_MS * 1_000_000


def test_a_row_without_a_usable_rate_is_skipped_not_zeroed():
    """Charging zero funding to a carry strategy is the optimistic default, and
    it flatters exactly the family this exists to measure."""
    payload = json.dumps([
        {"symbol": "X", "fundingTime": SETTLED_MS, "fundingRate": ""},
        {"symbol": "Y", "fundingTime": SETTLED_MS},
        {"symbol": "Z", "fundingRate": "0.001"},
        {"symbol": "OK", "fundingTime": SETTLED_MS, "fundingRate": "0.001"},
    ])
    rows = parse_binance_history(payload, "binance", FETCHED_NS)
    assert [r.symbol for r in rows] == ["OK"]


def test_junk_parses_to_nothing_rather_than_raising():
    assert parse_binance_history("not json", "binance", FETCHED_NS) == []
    assert parse_binance_history('{"code":-1121}', "binance", FETCHED_NS) == []


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def test_a_symbol_whose_fetch_fails_does_not_abandon_the_rest(tmp_path):
    """One symbol's history is not worth losing the other 856 for."""
    def flaky(url):
        if "BAD" in url:
            raise OSError("connection reset")
        return PAGE

    result = backfill(tmp_path, ["BAD", "BTCUSDT"], fetch=flaky,
                      now_ns=lambda: FETCHED_NS)

    assert result["rows"] == 2
    assert result["empty_symbols"] == ["BAD"]


def test_the_whole_run_is_one_partition_at_one_instant(tmp_path):
    """One act of reconstruction at one moment. Splitting it per symbol would
    suggest the parts were observed at different times, which is the impression
    this module exists to avoid."""
    result = backfill(tmp_path, ["BTCUSDT", "ETHUSDT"], fetch=lambda url: PAGE,
                      now_ns=lambda: FETCHED_NS)

    assert result["appended"] is True
    served = ClockGatedReader(tmp_path, DATASET).read_as_of(FETCHED_NS)
    assert set(served["availability_time_ns"]) == {FETCHED_NS}


def test_a_budget_that_refuses_stops_the_fetch(tmp_path):
    """Per-IP and exchange-wide. A backfill sweeping 857 symbols must spend from
    the same bucket the live capture does, or it earns the ban for both."""
    class Broke:
        def try_spend(self, weight):
            return False

    called = []
    result = backfill(tmp_path, ["BTCUSDT"], budget=Broke(),
                      fetch=lambda url: called.append(url) or PAGE,
                      now_ns=lambda: FETCHED_NS)

    assert called == [], "spent weight the budget had refused"
    assert result["rows"] == 0


def test_the_venue_is_asked_for_its_own_symbol_not_our_filename(tmp_path):
    """The archive names a symbol path-safely and the venue does not know that
    name. Asking Binance for `_b32_4W4IDZNORHSLVOXHSSPVK` returns an empty list,
    which is how three symbols came back with "no history" and read like
    delistings rather than like us sending our own filename.
    """
    from capture.venue_recorder import _safe_path_token
    from store.funding_backfill import fetch_binance_history

    encoded = _safe_path_token("币安人生USDT")
    asked = []

    def record(url):
        asked.append(url)
        return "[]"

    fetch_binance_history(encoded, fetch=record, now_ns=lambda: FETCHED_NS)

    assert asked, "no request was made"
    assert encoded not in asked[0], "sent our own filename to the venue"
    # URL-encoded, because the venue's name is not ASCII.
    assert "%E5%B8%81" in asked[0], asked[0]
