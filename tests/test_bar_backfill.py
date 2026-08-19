"""Bars fetched for minutes nobody watched, and the label that keeps them apart.

The property every test here circles is the one in `DECISIONS.md` §13: a
backfilled row must never be blendable with an observed one. It is defended
mechanically rather than by rule - availability time is the FETCH, so
`read_as_of` at any past clock returns nothing at all, and
`test_a_backtest_clock_cannot_see_a_reconstructed_bar` runs that through the
real reader rather than asserting on a column.

The kline below is real, pulled from binance 2026-08-10: BTCUSDT 1m at
1786341660000, 1,553 trades, 48.823 base volume.
"""
import json
import time

import pandas as pd
import pytest

from store.bar_backfill import (
    DEFAULT_INTERVAL_NS, INTERVAL_NAME_BY_NS, ReconstructedBar, backfill_bars,
    build_reconstructed_bars_frame, compare_reconstructed_to_observed,
    dataset_name, fetch_binance_bars, parse_binance_klines,
)
from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
    validate_temporal_frame,
)

MINUTE_NS = DEFAULT_INTERVAL_NS
BAR_OPEN_MS = 1786341660000
BAR_OPEN_NS = BAR_OPEN_MS * 1_000_000
# Well after that minute closed, so the fixture bar is a finished one.
FETCHED_NS = BAR_OPEN_NS + 3 * MINUTE_NS

REAL_KLINE = [BAR_OPEN_MS, "65138.80", "65141.80", "65104.10", "65141.70",
              "48.823", BAR_OPEN_MS + 59999, "3179347.50620", 1553,
              "30.400", "1979615.63050", "0"]


def _page(*klines):
    return json.dumps(list(klines))


def _parse(payload, fetched_at_ns=FETCHED_NS):
    return parse_binance_klines(payload, "BTCUSDT", "binance", fetched_at_ns)


# --- reading a page -------------------------------------------------------

def test_a_kline_becomes_one_bar():
    got = _parse(_page(REAL_KLINE))
    assert len(got) == 1
    bar = got[0]
    assert bar.bar_open_ns == BAR_OPEN_NS
    assert (bar.open, bar.high, bar.low, bar.close) == (
        65138.80, 65141.80, 65104.10, 65141.70)
    assert bar.volume == 48.823
    assert bar.trades == 1553


def test_the_minute_still_running_is_dropped():
    """Binance returns the current minute as a kline like any other, with
    partial volume. Stored, it would quietly disagree with the finished version
    of itself an hour later - and disagree low, which flatters anything
    measuring realised volume."""
    open_now = [BAR_OPEN_MS, "65138.80", "65141.80", "65104.10", "65141.70",
                "1.0", BAR_OPEN_MS + 59999, "0", 12, "0", "0", "0"]
    # Fetched thirty seconds into that minute.
    got = _parse(_page(open_now), fetched_at_ns=BAR_OPEN_NS + 30_000_000_000)
    assert got == []


def test_a_zero_price_is_refused():
    """The stored-bar validity gate removed these on the observed side; a
    backfill that let them in would put them straight back."""
    placeholder = list(REAL_KLINE)
    placeholder[3] = "0"          # low
    assert _parse(_page(placeholder)) == []


def test_a_malformed_row_is_skipped_not_zero_filled():
    """No trades is not a flat bar - the rule `build_bars` already holds to."""
    assert _parse(_page(["nonsense"], REAL_KLINE)) != []
    assert len(_parse(_page(["nonsense"], REAL_KLINE))) == 1
    assert _parse("not json") == []
    assert _parse(json.dumps({"error": "rate limited"})) == []


def test_a_negative_trade_count_is_refused():
    broken = list(REAL_KLINE)
    broken[8] = -3
    assert _parse(_page(broken)) == []


# --- the label and the clock ----------------------------------------------

def _bar(**over):
    base = dict(symbol="BTCUSDT", venue="binance", open=65138.8, high=65141.8,
                low=65104.1, close=65141.7, volume=48.823, trades=1553,
                bar_open_ns=BAR_OPEN_NS, fetched_at_ns=FETCHED_NS)
    base.update(over)
    return ReconstructedBar(**base)


def test_availability_is_the_fetch_not_the_bar_close():
    """The single line the whole design rests on. Stamping availability at bar
    close would make a reconstructed bar indistinguishable from an observed one
    to every consumer."""
    frame = build_reconstructed_bars_frame([_bar()])
    row = frame.iloc[0]
    assert row[AVAILABILITY_TIME] == row[INGESTION_TIME] == FETCHED_NS
    assert row[EVENT_TIME] == BAR_OPEN_NS
    assert row[AVAILABILITY_TIME] > row[EVENT_TIME] + MINUTE_NS


def test_every_row_says_it_is_reconstructed():
    """Carried per row so a frame that is filtered, joined or copied elsewhere
    still says what it is."""
    frame = build_reconstructed_bars_frame([_bar(), _bar(bar_open_ns=BAR_OPEN_NS + MINUTE_NS)])
    assert frame["is_reconstructed"].all()
    assert str(frame["is_reconstructed"].dtype) == "bool"


def test_the_frame_satisfies_the_temporal_contract():
    validate_temporal_frame(build_reconstructed_bars_frame([_bar()]))


def test_no_bars_is_an_empty_frame():
    assert build_reconstructed_bars_frame([]).empty


def test_a_backtest_clock_cannot_see_a_reconstructed_bar(tmp_path):
    """Run through the real reader, not asserted on a column. At every
    simulated instant before the fetch this dataset is empty, so a backtest
    cannot consume it by accident - whether or not the person writing that
    backtest ever heard of this module."""
    append_partition(tmp_path, dataset_name(),
                     build_reconstructed_bars_frame([_bar()]),
                     snapshot_id="backfill-test")
    reader = ClockGatedReader(tmp_path, dataset_name())

    assert reader.read_as_of(BAR_OPEN_NS + MINUTE_NS).empty
    assert reader.read_as_of(FETCHED_NS - 1).empty
    assert len(reader.read_as_of(FETCHED_NS)) == 1


def test_reconstructed_bars_live_beside_the_observed_dataset_never_inside_it():
    """A consumer that wants both has to ask for both, in a line that says so."""
    assert dataset_name() == f"bars_reconstructed_{DEFAULT_INTERVAL_NS}ns"
    assert dataset_name() != f"bars_{DEFAULT_INTERVAL_NS}ns"


# --- walking the window ---------------------------------------------------

class _Pages:
    """A fake venue serving one page per call, recording what was asked."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.urls: list[str] = []

    def __call__(self, url, timeout=20.0):
        self.urls.append(url)
        return self.payloads.pop(0) if self.payloads else _page()


def _kline_at(open_ms, close_price="65141.70"):
    row = list(REAL_KLINE)
    row[0] = open_ms
    row[4] = close_price
    row[6] = open_ms + 59999
    return row


def test_the_walk_goes_forward_from_the_older_end():
    """A bar gap has two known ends, so paging from the older one makes
    progress provable: each request starts one millisecond past the newest bar
    already held."""
    first = _page(_kline_at(BAR_OPEN_MS), _kline_at(BAR_OPEN_MS + 60_000))
    fetch = _Pages(first, _page())
    got = fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + 10 * MINUTE_NS,
                             fetch=fetch, now_ns=lambda: FETCHED_NS)

    assert [b.bar_open_ns for b in got] == [BAR_OPEN_NS, BAR_OPEN_NS + MINUTE_NS]
    assert f"startTime={BAR_OPEN_MS}" in fetch.urls[0]
    assert f"startTime={BAR_OPEN_MS + 60_000 + 1}" in fetch.urls[1]


def test_a_page_that_does_not_advance_stops_the_walk():
    """A venue ignoring `startTime` would otherwise return the same page until
    the page budget ran out."""
    same = _page(_kline_at(BAR_OPEN_MS))
    fetch = _Pages(same, same, same, same)
    got = fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + 10 * MINUTE_NS,
                             fetch=fetch, now_ns=lambda: FETCHED_NS)

    assert len(got) == 1
    assert len(fetch.urls) == 2


def test_the_walk_stops_when_the_rate_budget_refuses():
    """This shares a weight allowance with the pollers that keep live capture
    alive - a backfill must not be able to starve them."""
    class _Spent:
        def try_spend(self, weight):
            return False

    fetch = _Pages(_page(_kline_at(BAR_OPEN_MS)))
    got = fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + 10 * MINUTE_NS,
                             budget=_Spent(), fetch=fetch,
                             now_ns=lambda: FETCHED_NS)

    assert got == []
    assert fetch.urls == []


def test_an_interval_binance_has_no_kline_for_raises():
    with pytest.raises(ValueError, match="no binance kline interval"):
        fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + MINUTE_NS,
                           interval_ns=7_000_000_000, fetch=_Pages(),
                           now_ns=lambda: FETCHED_NS)


# --- writing --------------------------------------------------------------

def test_a_window_the_venue_has_nothing_for_writes_no_partition(tmp_path):
    """An empty backfill is a fact about the window. A zero-row partition would
    look like a completed one."""
    result = backfill_bars(tmp_path, ["BTCUSDT"], BAR_OPEN_NS,
                           BAR_OPEN_NS + MINUTE_NS, fetch=_Pages(),
                           now_ns=lambda: FETCHED_NS)

    assert result["appended"] is False
    assert result["bars"] == 0
    assert not (tmp_path / dataset_name()).exists()


def test_a_backfill_lands_as_one_labelled_partition(tmp_path):
    fetch = _Pages(_page(_kline_at(BAR_OPEN_MS), _kline_at(BAR_OPEN_MS + 60_000)),
                   _page())
    result = backfill_bars(tmp_path, ["BTCUSDT"], BAR_OPEN_NS,
                           BAR_OPEN_NS + 10 * MINUTE_NS, fetch=fetch,
                           now_ns=lambda: FETCHED_NS)

    assert result == {"dataset": dataset_name(), "venue": "binance",
                      "symbols": 1, "bars": 2, "appended": True,
                      "first_bar_ns": BAR_OPEN_NS,
                      "last_bar_ns": BAR_OPEN_NS + MINUTE_NS}
    stored = ClockGatedReader(tmp_path, dataset_name()).read_as_of(FETCHED_NS)
    assert stored["is_reconstructed"].all()


# --- what a reconstructed bar is worth ------------------------------------

def _observed_row(bar_open_ns, close, volume, symbol="BTCUSDT"):
    return {VENUE: "binance", SYMBOL: symbol, "open": close, "high": close,
            "low": close, "close": close, "volume": volume, "trades": 10,
            EVENT_TIME: bar_open_ns, INGESTION_TIME: bar_open_ns + MINUTE_NS,
            AVAILABILITY_TIME: bar_open_ns + MINUTE_NS}


def test_no_overlap_is_reported_as_zero_not_as_agreement(tmp_path):
    """`overlapping: 0` is a real answer - it says the two have never been
    compared on this store, which is not the same as them agreeing."""
    append_partition(tmp_path, f"bars_{DEFAULT_INTERVAL_NS}ns",
                     pd.DataFrame([_observed_row(BAR_OPEN_NS, 100.0, 5.0)]),
                     snapshot_id="observed")
    append_partition(tmp_path, dataset_name(),
                     build_reconstructed_bars_frame(
                         [_bar(bar_open_ns=BAR_OPEN_NS + 99 * MINUTE_NS)]),
                     snapshot_id="reconstructed")

    summary = compare_reconstructed_to_observed(tmp_path, FETCHED_NS + MINUTE_NS)

    assert summary["overlapping"] == 0
    assert "close_agreement_bps_median" not in summary


def test_the_distance_from_the_observed_bar_is_measured_where_both_exist(tmp_path):
    append_partition(tmp_path, f"bars_{DEFAULT_INTERVAL_NS}ns",
                     pd.DataFrame([_observed_row(BAR_OPEN_NS, 100.0, 5.0)]),
                     snapshot_id="observed")
    append_partition(tmp_path, dataset_name(),
                     build_reconstructed_bars_frame(
                         [_bar(bar_open_ns=BAR_OPEN_NS, close=101.0, volume=10.0)]),
                     snapshot_id="reconstructed")

    summary = compare_reconstructed_to_observed(tmp_path, FETCHED_NS + MINUTE_NS)

    assert summary["overlapping"] == 1
    assert summary["close_agreement_bps_median"] == pytest.approx(100.0)
    assert summary["volume_ratio_median"] == pytest.approx(2.0)


def test_a_reconstructed_row_inside_the_observed_dataset_is_counted(tmp_path):
    """The one thing this design is supposed to make impossible, counted rather
    than trusted. If it is ever non-zero the two have been blended, which is the
    defect DM-015 was written about."""
    leaked = _observed_row(BAR_OPEN_NS, 100.0, 5.0)
    leaked["is_reconstructed"] = True
    append_partition(tmp_path, f"bars_{DEFAULT_INTERVAL_NS}ns",
                     pd.DataFrame([leaked]), snapshot_id="observed")
    append_partition(tmp_path, dataset_name(),
                     build_reconstructed_bars_frame([_bar()]),
                     snapshot_id="reconstructed")

    summary = compare_reconstructed_to_observed(tmp_path, FETCHED_NS + MINUTE_NS)

    assert summary["reconstructed_rows_in_observed"] == 1


def test_an_untouched_observed_dataset_reports_zero_blended_rows(tmp_path):
    append_partition(tmp_path, f"bars_{DEFAULT_INTERVAL_NS}ns",
                     pd.DataFrame([_observed_row(BAR_OPEN_NS, 100.0, 5.0)]),
                     snapshot_id="observed")
    append_partition(tmp_path, dataset_name(),
                     build_reconstructed_bars_frame([_bar()]),
                     snapshot_id="reconstructed")

    summary = compare_reconstructed_to_observed(tmp_path, FETCHED_NS + MINUTE_NS)

    assert summary["reconstructed_rows_in_observed"] == 0


def test_an_empty_store_compares_to_nothing_rather_than_raising(tmp_path):
    summary = compare_reconstructed_to_observed(tmp_path, int(time.time_ns()))
    assert summary == {"observed_bars": 0, "reconstructed_bars": 0,
                       "reconstructed_rows_in_observed": 0, "overlapping": 0}


# --- RL-043's four intraday timeframes -------------------------------------


def test_every_timeframe_the_intraday_ruling_names_can_be_fetched():
    """RL-043 names 1m, 5m, 15m and 30m as the bots' intraday timeframes.

    A ruling that names a timeframe the backfill cannot ask for is a plan with a
    hole in it, and the hole is invisible until something tries to train on the
    missing bars.
    """
    assert {"1m", "5m", "15m", "30m"} <= set(INTERVAL_NAME_BY_NS.values())


def test_every_declared_interval_is_a_real_binance_kline_name():
    # A name the venue does not publish comes back as an error page that parses
    # to zero bars, which reads on a board as a market with no trades.
    binance_klines = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h",
                      "8h", "12h", "1d", "3d", "1w", "1M"}

    assert set(INTERVAL_NAME_BY_NS.values()) <= binance_klines


def test_each_interval_key_is_its_own_length_in_nanoseconds():
    # The key IS the duration - `parse_binance_klines` drops the unclosed bar by
    # comparing `bar_open_ns + interval_ns` against the fetch time, so a key that
    # disagreed with its name would silently keep a bar that is still forming.
    seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
    for interval_ns, name in INTERVAL_NAME_BY_NS.items():
        assert interval_ns == seconds[name] * 1_000_000_000, name


def test_a_fifteen_minute_page_asks_the_venue_for_fifteen_minute_klines():
    fetch = _Pages()
    fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + 900_000_000_000,
                       interval_ns=900_000_000_000, fetch=fetch,
                       now_ns=lambda: FETCHED_NS)

    assert fetch.urls, "no request was made"
    assert "interval=15m" in fetch.urls[0]


def test_a_thirty_minute_page_asks_the_venue_for_thirty_minute_klines():
    fetch = _Pages()
    fetch_binance_bars("BTCUSDT", BAR_OPEN_NS, BAR_OPEN_NS + 1_800_000_000_000,
                       interval_ns=1_800_000_000_000, fetch=fetch,
                       now_ns=lambda: FETCHED_NS)

    assert fetch.urls, "no request was made"
    assert "interval=30m" in fetch.urls[0]


def test_each_interval_writes_its_own_dataset():
    # Two timeframes sharing a dataset name would interleave 15-minute and
    # 30-minute bars into one series that is neither.
    names = {dataset_name(ns) for ns in INTERVAL_NAME_BY_NS}

    assert len(names) == len(INTERVAL_NAME_BY_NS)
