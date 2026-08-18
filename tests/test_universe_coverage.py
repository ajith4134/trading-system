"""A watch list that drops what it stopped seeing cannot answer its own question.

`features.cross_sectional` drops a symbol the moment it stops printing, and is
right to - ranking an instrument on three-day-old prices is worse than not
ranking it. This module keeps it, and the quiet row is the interesting one. The
two postures must not be merged, and the first test here is the one that pins
the difference.

The rest defend the classifier (evidence, never a name) and the distinction
between "the venue stopped speaking" and "our build is behind", which the first
live run made unavoidable: 4,486 symbols, every one of them stale, and the feeds
were fine.
"""
from decimal import Decimal

import pandas as pd
import pytest

from features.staleness import FRESH, STALE, UNKNOWN_CADENCE
from features.universe_coverage import (
    DATED_FUTURE,
    PERPETUAL,
    SEGMENTS,
    SPOT,
    compute_universe_coverage,
)
from store.parquet_partition import append_partition

_MINUTE = 60_000_000_000
_START = 1_700_000_000_000_000_000


def _bars(symbol, venue, n, start_ns=_START, step_ns=_MINUTE):
    return [{
        "venue": venue, "symbol": symbol,
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": 1.0, "trades": 10,
        "event_time_ns": start_ns + i * step_ns,
        "ingestion_time_ns": start_ns + i * step_ns,
        "availability_time_ns": start_ns + i * step_ns,
    } for i in range(n)]


def _funding(symbol, venue, n, start_ns=_START, step_ns=_MINUTE):
    return [{
        "venue": venue, "symbol": symbol, "funds_on": "mark",
        "funding_rate": Decimal("0.0001"),
        "mark_price": Decimal("100.000"), "index_price": Decimal("100.000"),
        "oracle_price": None,
        "event_time_is_receipt": False, "next_funding_time_unknown": False,
        "next_funding_time_ns": start_ns + 1, "funding_interval_hours": 8,
        "event_time_ns": start_ns + i * step_ns,
        "ingestion_time_ns": start_ns + i * step_ns,
        "availability_time_ns": start_ns + i * step_ns,
    } for i in range(n)]


def _dated(symbol, venue, n, start_ns=_START, step_ns=_MINUTE):
    return [{
        "venue": venue, "symbol": symbol,
        "underlying": "BTCUSDT", "expiry_ns": start_ns + 10**15,
        "mark_price": Decimal("100.000"), "index_price": Decimal("100.000"),
        "event_time_ns": start_ns + i * step_ns,
        "ingestion_time_ns": start_ns + i * step_ns,
        "availability_time_ns": start_ns + i * step_ns,
    } for i in range(n)]


def _write(tmp_path, dataset, rows, snapshot="coverage-test"):
    if rows:
        append_partition(tmp_path, dataset, pd.DataFrame(rows),
                         snapshot_id=snapshot)
    return tmp_path


def _rows_for(coverage, symbol):
    frame = coverage.rows
    return frame[frame["symbol"] == symbol]


# --- the posture that makes this not cross_sectional ---------------------

def test_a_symbol_that_went_quiet_is_kept_and_marked(tmp_path):
    """The whole reason this module exists. A watch list that drops what it
    stopped seeing cannot answer 'is that instrument gone, or did our feed
    stop?'."""
    live = _bars("LIVEUSDT", "binance-spot", 60)
    dead = _bars("DEADUSDT", "binance-spot", 60, start_ns=_START - 500 * _MINUTE)
    store = _write(tmp_path, "bars_60000000000ns", live + dead)

    coverage = compute_universe_coverage(store, _START + 60 * _MINUTE)

    assert set(coverage.rows["symbol"]) == {"LIVEUSDT", "DEADUSDT"}
    assert _rows_for(coverage, "DEADUSDT")["freshness"].iloc[0] == STALE


# --- segments come from evidence, never from a name ----------------------

def test_a_key_with_funding_is_a_perpetual(tmp_path):
    _write(tmp_path, "bars_60000000000ns", _bars("BTCUSDT", "binance", 60))
    _write(tmp_path, "funding", _funding("BTCUSDT", "binance", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert _rows_for(coverage, "BTCUSDT")["segment"].iloc[0] == PERPETUAL


def test_a_key_in_the_dated_dataset_is_a_dated_future(tmp_path):
    _write(tmp_path, "dated_futures", _dated("BTCUSDT-14AUG26", "bybit", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert _rows_for(coverage, "BTCUSDT-14AUG26")["segment"].iloc[0] == DATED_FUTURE


def test_bars_without_funding_are_spot(tmp_path):
    _write(tmp_path, "bars_60000000000ns", _bars("ETHUSDT", "binance-spot", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert _rows_for(coverage, "ETHUSDT")["segment"].iloc[0] == SPOT


def test_the_venue_name_does_not_decide_the_segment(tmp_path):
    """`binance-spot` obviously means spot to a human. A classifier that reads
    names is wrong silently on the first venue that spells things differently -
    the same reasoning `features.term_structure` gives for refusing to establish
    an underlying by splitting a symbol.
    """
    # A venue NAMED spot, carrying funding. Evidence wins.
    _write(tmp_path, "bars_60000000000ns", _bars("ODDUSDT", "binance-spot", 60))
    _write(tmp_path, "funding", _funding("ODDUSDT", "binance-spot", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert _rows_for(coverage, "ODDUSDT")["segment"].iloc[0] == PERPETUAL


def test_a_symbol_appears_in_exactly_one_segment(tmp_path):
    """A key carried by two datasets must not be double-counted - the totals are
    what a reader trusts, and a symbol in two segments inflates them silently."""
    _write(tmp_path, "bars_60000000000ns", _bars("BTCUSDT", "binance", 60))
    _write(tmp_path, "funding", _funding("BTCUSDT", "binance", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert len(_rows_for(coverage, "BTCUSDT")) == 1
    assert coverage.total_symbols == len(coverage.rows)


# --- build lag against feed death ----------------------------------------

def test_a_uniformly_lagging_build_does_not_read_as_dead_symbols(tmp_path):
    """The first live run: 4,486 symbols, every one stale, feeds fine. Bars are
    a one-minute series, so an hour-old build makes the whole universe stale at
    once and a board of 4,486 STALE rows informs nobody."""
    rows = []
    for i in range(5):
        rows += _bars(f"SYM{i}", "binance-spot", 60)
    store = _write(tmp_path, "bars_60000000000ns", rows)

    # An hour past the newest bar: everything is stale, and everything is
    # exactly as current as the pipeline allows.
    coverage = compute_universe_coverage(store, _START + 120 * _MINUTE)
    spot = [s for s in coverage.segments if s.segment == SPOT][0]

    assert spot.stale == 5
    assert spot.build_lag_ns > 0
    assert spot.behind_build == 0, (
        "nothing is late relative to the build - they are all the build")


def test_a_symbol_quiet_beyond_the_build_is_counted(tmp_path):
    """The number that does not move when the build catches up."""
    live = []
    for i in range(5):
        live += _bars(f"SYM{i}", "binance-spot", 60)
    dead = _bars("GONEUSDT", "binance-spot", 60,
                 start_ns=_START - 5000 * _MINUTE)
    store = _write(tmp_path, "bars_60000000000ns", live + dead)

    coverage = compute_universe_coverage(store, _START + 120 * _MINUTE)
    spot = [s for s in coverage.segments if s.segment == SPOT][0]

    assert spot.behind_build == 1
    assert _rows_for(coverage, "GONEUSDT")["age_beyond_build_ns"].iloc[0] > 0


def test_lateness_is_judged_against_each_symbol_s_own_cadence(tmp_path):
    """A tail symbol that prints once an hour has an hour-old bar and is
    behaving normally; a core symbol three minutes behind has stopped. One fixed
    threshold cannot be right about both - which is what the first version of
    this used, and it was wrong in the way this module's docstring forbids.
    """
    fast = []
    for i in range(3):
        fast += _bars(f"FAST{i}", "binance-spot", 60)
    # Prints once an hour, and its newest print is one hour behind the build.
    slow = _bars("SLOWUSDT", "binance-spot", 20, start_ns=_START - 1200 * _MINUTE,
                 step_ns=60 * _MINUTE)
    store = _write(tmp_path, "bars_60000000000ns", fast + slow)

    coverage = compute_universe_coverage(store, _START + 120 * _MINUTE)
    slow_row = _rows_for(coverage, "SLOWUSDT").iloc[0]

    assert slow_row["routine_gap_ns"] == 60 * _MINUTE
    # It is behind the build, but by less than three of its own gaps, so it is
    # not counted as having gone quiet.
    spot = [s for s in coverage.segments if s.segment == SPOT][0]
    assert spot.behind_build == 0


def test_a_symbol_with_no_measurable_cadence_is_not_counted_as_late(tmp_path):
    """It has shown nothing to be late against, and counting it would put "we
    have never seen this trade twice" in the same bucket as "this stopped"."""
    rows = []
    for i in range(3):
        rows += _bars(f"SYM{i}", "binance-spot", 60)
    rows += _bars("ONCEUSDT", "binance-spot", 1, start_ns=_START - 9000 * _MINUTE)
    store = _write(tmp_path, "bars_60000000000ns", rows)

    coverage = compute_universe_coverage(store, _START + 120 * _MINUTE)
    once = _rows_for(coverage, "ONCEUSDT").iloc[0]
    spot = [s for s in coverage.segments if s.segment == SPOT][0]

    assert once["freshness"] == UNKNOWN_CADENCE
    assert spot.unknown_cadence == 1
    assert spot.behind_build == 0


# --- counts and their denominators ---------------------------------------

def test_every_count_comes_with_its_denominator(tmp_path):
    """"92% fresh" over 1,800 symbols and over 12 print identically."""
    rows = []
    for i in range(7):
        rows += _bars(f"SYM{i}", "binance-spot", 60)
    coverage = compute_universe_coverage(
        _write(tmp_path, "bars_60000000000ns", rows), _START + 60 * _MINUTE)
    spot = [s for s in coverage.segments if s.segment == SPOT][0]

    assert spot.symbols == 7
    assert spot.fresh + spot.stale + spot.unknown_cadence == spot.symbols
    assert "7 symbol(s)" in spot.describe()


def test_segments_are_reported_in_a_declared_order(tmp_path):
    _write(tmp_path, "bars_60000000000ns",
           _bars("SPOTUSDT", "binance-spot", 60) + _bars("BTCUSDT", "binance", 60))
    _write(tmp_path, "funding", _funding("BTCUSDT", "binance", 60))
    _write(tmp_path, "dated_futures", _dated("BTCUSDT-14AUG26", "bybit", 60))

    coverage = compute_universe_coverage(tmp_path, _START + 60 * _MINUTE)

    assert [s.segment for s in coverage.segments] == list(SEGMENTS)
    assert coverage.total_symbols == 3


def test_a_missing_dataset_does_not_cost_the_other_segments(tmp_path):
    """`dated_futures` exists only because bybit publishes expiring contracts,
    and a store built before that capture landed has no such directory."""
    coverage = compute_universe_coverage(
        _write(tmp_path, "bars_60000000000ns", _bars("BTCUSDT", "binance", 60)),
        _START + 60 * _MINUTE)

    assert coverage.total_symbols == 1
    assert [s.segment for s in coverage.segments] == [SPOT]


def test_an_empty_store_watches_nothing_and_says_so(tmp_path):
    (tmp_path / "bars_60000000000ns").mkdir(parents=True)
    coverage = compute_universe_coverage(tmp_path, _START)

    assert coverage.rows.empty
    assert coverage.segments == []
    assert "no symbols visible" in coverage.describe()


def test_bars_already_read_are_used_instead_of_read_again(tmp_path, monkeypatch):
    """The bars read is the expensive one - minutes against the live store - and
    `perp.tradable_universe` needs the same frame for its liquidity refusal.
    Handing it over must produce the same roll-call as letting the module read
    it, or the caller is quietly working from a different universe.
    """
    from features import universe_coverage as module

    _write(tmp_path, "bars_60000000000ns", _bars("BTCUSDT", "binance", 60))
    _write(tmp_path, "funding", _funding("BTCUSDT", "binance", 60))
    store, as_of = tmp_path, _START + 60 * _MINUTE
    expected = compute_universe_coverage(store, as_of)

    handed = module._read(store, module.BARS_DATASET, as_of, None)
    reads: list[str] = []
    original = module._read

    def _counted(store_root, dataset, clock, custodian):
        reads.append(dataset)
        return original(store_root, dataset, clock, custodian)

    monkeypatch.setattr(module, "_read", _counted)
    got = compute_universe_coverage(store, as_of, bars=handed)

    assert module.BARS_DATASET not in reads, "the bars dataset was read again"
    assert got.total_symbols == expected.total_symbols
    assert [s.describe() for s in got.segments] == \
           [s.describe() for s in expected.segments]
