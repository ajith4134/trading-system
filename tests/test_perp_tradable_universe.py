"""PB-01: which perpetuals the bot may trade, and why every other one may not.

The failure this row exists to prevent is the one the plan already names: a bot
that picks a handful of symbols, trades them, and never says what it skipped or
why. RL-009 and RL-014 both refuse that. So the interesting assertions here are
not "the good symbol was admitted" - they are that every excluded symbol carries
a reason, that the counts add up to the population, and that a symbol we cannot
run the strategy on is refused rather than quietly admitted.

The tier tests matter as much as the admission tests. Measured 2026-08-17, bars
cover 2,235 symbols and the `book` dataset covers six. A universe that reported
"2,235 tradable" would be true about bars and false about the strategy, and the
first thing anyone would do with that number is build against it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from features.staleness import CADENCE_SAMPLE
from perp.tradable_universe import (
    BARS_ONLY, DEEP, ILLIQUID, NOT_PERPETUAL, TOO_LITTLE_HISTORY, WENT_QUIET,
    select_tradable_perp_universe,
)
from store.parquet_partition import append_partition, clear_schema_cache
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

MINUTE_NS = 60_000_000_000
MIDNIGHT_NS = 1_786_924_800_000_000_000
# Late enough that every bar written below is available, and close enough that a
# symbol printing every minute is fresh rather than stale.
AS_OF_NS = MIDNIGHT_NS + 400 * MINUTE_NS


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


def _bar_rows(symbol: str, venue: str, count: int, *, first_ns: int = MIDNIGHT_NS,
              step_ns: int = MINUTE_NS, volume: float = 100.0,
              trades: int = 50) -> pd.DataFrame:
    rows = []
    for index in range(count):
        stamp = first_ns + index * step_ns
        rows.append({SYMBOL: symbol, VENUE: venue, EVENT_TIME: stamp,
                     INGESTION_TIME: stamp, AVAILABILITY_TIME: stamp,
                     "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                     "volume": volume, "trades": trades})
    return pd.DataFrame(rows).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                                      AVAILABILITY_TIME: "int64", "trades": "int64"})


def _funding_rows(symbol: str, venue: str, count: int = 3) -> pd.DataFrame:
    rows = []
    for index in range(count):
        stamp = MIDNIGHT_NS + index * 8 * 60 * MINUTE_NS
        rows.append({SYMBOL: symbol, VENUE: venue, EVENT_TIME: stamp,
                     INGESTION_TIME: stamp, AVAILABILITY_TIME: stamp,
                     "funding_rate": 0.0001, "funding_interval_hours": 8.0})
    return pd.DataFrame(rows).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                                      AVAILABILITY_TIME: "int64"})


def _book_rows(symbol: str, venue: str, count: int = 5) -> pd.DataFrame:
    rows = []
    for index in range(count):
        stamp = MIDNIGHT_NS + index * MINUTE_NS
        rows.append({SYMBOL: symbol, VENUE: venue, EVENT_TIME: stamp,
                     INGESTION_TIME: stamp, AVAILABILITY_TIME: stamp,
                     "bid_price": 1.0, "ask_price": 1.01,
                     "bid_size": 10.0, "ask_size": 10.0})
    return pd.DataFrame(rows).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                                      AVAILABILITY_TIME: "int64"})


def _stocked(store, *, bars, funding=(), book=()):
    for index, frame in enumerate(bars):
        append_partition(store, "bars_60000000000ns", frame, f"bars{index}")
    for index, frame in enumerate(funding):
        append_partition(store, "funding", frame, f"funding{index}")
    for index, frame in enumerate(book):
        append_partition(store, "book", frame, f"book{index}")
    return store


# --- admission ---------------------------------------------------------------

def test_a_live_perpetual_with_depth_is_admitted_as_deep(tmp_path):
    _stocked(tmp_path,
             bars=[_bar_rows("BTCUSDT", "binance", 400)],
             funding=[_funding_rows("BTCUSDT", "binance")],
             book=[_book_rows("BTCUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.admitted_symbols == [("binance", "BTCUSDT")]
    assert universe.tier_of("binance", "BTCUSDT") == DEEP
    assert universe.excluded_count == 0


def test_a_perpetual_without_depth_is_admitted_but_only_bars_deep(tmp_path):
    """The measured shape of the live store: 2,235 symbols with bars, six with a
    book. Refusing every symbol without depth would throw away the universe the
    user asked for; admitting them as DEEP would let a book-based feature be
    built against symbols that have no book. The tier is how both stay true."""
    _stocked(tmp_path,
             bars=[_bar_rows("ARBUSDT", "binance", 400)],
             funding=[_funding_rows("ARBUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.admitted_symbols == [("binance", "ARBUSDT")]
    assert universe.tier_of("binance", "ARBUSDT") == BARS_ONLY
    assert universe.deep_count == 0


# --- exclusion, and every exclusion says why ---------------------------------

def test_a_spot_symbol_is_excluded_as_not_perpetual(tmp_path):
    """Segment comes from which dataset carries the key - no funding, not a
    perpetual - never from reading `binance-spot` out of the venue name."""
    _stocked(tmp_path, bars=[_bar_rows("BTCUSDT", "binance-spot", 400)])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.admitted_symbols == []
    assert universe.reason_for("binance-spot", "BTCUSDT") == NOT_PERPETUAL


def test_a_perpetual_that_went_quiet_is_excluded(tmp_path):
    """Quiet by its OWN cadence, not a shared constant. This symbol printed every
    minute and then stopped hours ago, while the rest of the build is current."""
    _stocked(tmp_path,
             bars=[_bar_rows("LIVEUSDT", "binance", 400),
                   _bar_rows("DEADUSDT", "binance", 40)],
             funding=[_funding_rows("LIVEUSDT", "binance"),
                      _funding_rows("DEADUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert ("binance", "LIVEUSDT") in universe.admitted_symbols
    assert ("binance", "DEADUSDT") not in universe.admitted_symbols
    assert universe.reason_for("binance", "DEADUSDT") == WENT_QUIET


def test_a_perpetual_with_too_few_bars_to_measure_a_cadence_is_excluded(tmp_path):
    """Fewer observations than `CADENCE_SAMPLE` means no cadence has been shown,
    so nothing can be said about whether it is fresh - and a scalping feature has
    nothing to roll over. Admitting it would put "we have never seen this trade
    twice" in the same bucket as a liquid perpetual."""
    _stocked(tmp_path,
             bars=[_bar_rows("NEWUSDT", "binance", CADENCE_SAMPLE - 1,
                             first_ns=AS_OF_NS - CADENCE_SAMPLE * MINUTE_NS)],
             funding=[_funding_rows("NEWUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.admitted_symbols == []
    assert universe.reason_for("binance", "NEWUSDT") == TOO_LITTLE_HISTORY


def test_a_perpetual_that_prints_bars_but_never_trades_is_excluded(tmp_path):
    """A bar with zero volume and zero trades is a bar the venue emitted, not a
    market. Scalping it would model fills against an order book nobody is
    hitting, and the fill model would happily oblige."""
    _stocked(tmp_path,
             bars=[_bar_rows("GHOSTUSDT", "binance", 400, volume=0.0, trades=0)],
             funding=[_funding_rows("GHOSTUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.admitted_symbols == []
    assert universe.reason_for("binance", "GHOSTUSDT") == ILLIQUID


# --- the accounting, which is the row's actual acceptance ---------------------

def test_every_symbol_resolves_to_admitted_or_a_named_reason(tmp_path):
    """The row's acceptance sentence, asserted directly: nothing falls between."""
    _stocked(tmp_path,
             bars=[_bar_rows("BTCUSDT", "binance", 400),
                   _bar_rows("ARBUSDT", "binance", 400),
                   _bar_rows("DEADUSDT", "binance", 40),
                   _bar_rows("GHOSTUSDT", "binance", 400, volume=0.0, trades=0),
                   _bar_rows("BTCUSDT", "binance-spot", 400)],
             funding=[_funding_rows("BTCUSDT", "binance"),
                      _funding_rows("ARBUSDT", "binance"),
                      _funding_rows("DEADUSDT", "binance"),
                      _funding_rows("GHOSTUSDT", "binance")],
             book=[_book_rows("BTCUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.considered == 5
    assert universe.admitted_count + universe.excluded_count == universe.considered
    for _, row in universe.rows.iterrows():
        if row["admitted"]:
            assert row["reason"] == "", "an admitted symbol must carry no reason"
        else:
            assert row["reason"], f"{row['symbol']} was excluded with no reason"


def test_the_excluded_count_is_published_broken_down_by_reason(tmp_path):
    """"Published rather than hidden" is the acceptance. A total with no
    breakdown is the number that gets quoted and never acted on."""
    _stocked(tmp_path,
             bars=[_bar_rows("ARBUSDT", "binance", 400),
                   _bar_rows("DEADUSDT", "binance", 40),
                   _bar_rows("BTCUSDT", "binance-spot", 400)],
             funding=[_funding_rows("ARBUSDT", "binance"),
                      _funding_rows("DEADUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.excluded_by_reason == {WENT_QUIET: 1, NOT_PERPETUAL: 1}
    assert sum(universe.excluded_by_reason.values()) == universe.excluded_count


def test_an_empty_store_reports_nothing_considered_rather_than_nothing_wrong(tmp_path):
    """Absence renders as its own state (Rule 8). Zero admitted out of zero
    considered must not read the same as zero admitted out of two thousand."""
    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS)

    assert universe.considered == 0
    assert universe.admitted_symbols == []
    assert universe.excluded_by_reason == {}
    assert "0 considered" in universe.describe()


def test_the_description_names_counts_and_never_a_bare_percentage(tmp_path):
    """A percentage without its denominator is how "92% tradable" gets read the
    same over 12 symbols and over 1,800."""
    _stocked(tmp_path,
             bars=[_bar_rows("BTCUSDT", "binance", 400),
                   _bar_rows("ARBUSDT", "binance", 400)],
             funding=[_funding_rows("BTCUSDT", "binance"),
                      _funding_rows("ARBUSDT", "binance")],
             book=[_book_rows("BTCUSDT", "binance")])

    described = select_tradable_perp_universe(tmp_path, AS_OF_NS).describe()

    assert "2 considered" in described
    assert "2 admitted" in described
    assert "1 DEEP" in described
    assert "%" not in described


# --- the bars read is bounded, and the window is part of the answer ----------

def test_a_perpetual_whose_bars_predate_the_window_is_quiet_not_new(tmp_path):
    """A perpetual is established by its FUNDING prints, so one that has stopped
    printing bars still arrives carrying a few funding observations. Reporting
    that as TOO_LITTLE_HISTORY would send someone looking for a young listing
    when the series actually stopped."""
    _stocked(tmp_path,
             bars=[_bar_rows("OLDUSDT", "binance", 400,
                             first_ns=MIDNIGHT_NS - 10_000 * MINUTE_NS)],
             funding=[_funding_rows("OLDUSDT", "binance")])

    universe = select_tradable_perp_universe(tmp_path, AS_OF_NS,
                                             lookback_ns=6 * 3_600_000_000_000)

    assert universe.reason_for("binance", "OLDUSDT") == WENT_QUIET
    assert universe.reason_for("binance", "OLDUSDT") != TOO_LITTLE_HISTORY


def test_the_bars_read_is_bounded_so_hour_pruning_can_help_it(tmp_path):
    """Measured 2026-08-17: an unbounded read of the live bars dataset had not
    returned after 26 minutes, and hour pruning cannot help it - a read with no
    lower bound needs every hour by definition. The bound is what lets SL-15's
    layout change reach this row at all."""
    from store import clock_gated_reader as module

    _stocked(tmp_path,
             bars=[_bar_rows("BTCUSDT", "binance", 400)],
             funding=[_funding_rows("BTCUSDT", "binance")])

    bounds: list[int | None] = []
    original = module.ClockGatedReader.read_as_of

    def _recorded(self, sim_clock_ns, symbols=None, not_before_ns=None):
        if self._dataset == "bars_60000000000ns":
            bounds.append(not_before_ns)
        return original(self, sim_clock_ns, symbols, not_before_ns)

    module.ClockGatedReader.read_as_of = _recorded
    try:
        select_tradable_perp_universe(tmp_path, AS_OF_NS,
                                      lookback_ns=6 * 3_600_000_000_000)
    finally:
        module.ClockGatedReader.read_as_of = original

    assert bounds, "the bars dataset was never read through the clock gate"
    assert all(bound == AS_OF_NS - 6 * 3_600_000_000_000 for bound in bounds), \
        f"the bars read was not bounded from below: {bounds}"


def test_the_window_rides_the_summary(tmp_path):
    """"1,900 admitted" over an hour and over a week are different statements."""
    _stocked(tmp_path,
             bars=[_bar_rows("BTCUSDT", "binance", 400)],
             funding=[_funding_rows("BTCUSDT", "binance")])

    described = select_tradable_perp_universe(
        tmp_path, AS_OF_NS, lookback_ns=6 * 3_600_000_000_000).describe()

    assert "last 6h" in described
