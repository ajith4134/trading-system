"""A peg is found by behaviour and judged against its own history.

The two adversarial cases here are real symbols from this box's own store, and
either one breaks the obvious implementation: `TUSDT` is the T token at $0.0037
and a prefix rule reads it as TrueUSD; `EURIUSDT` is a genuine peg at $1.16 and
a hardcoded 1.0 mis-prices it as permanently 16% broken.
"""
import numpy as np
import pandas as pd
import pytest

from features.peg_monitor import monitor_pegs
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000


def _bars(symbol, closes, venue="binance-spot", start_ns=1_000_000_000_000):
    return [{
        "venue": venue, "symbol": symbol,
        "open": c, "high": c, "low": c, "close": c,
        "volume": 1.0, "trades": 10,
        "event_time_ns": start_ns + i * _BAR_NS,
        "ingestion_time_ns": start_ns + i * _BAR_NS,
        "availability_time_ns": start_ns + i * _BAR_NS,
    } for i, c in enumerate(closes)]


def _write(tmp_path, rows, snapshot="peg-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _steady(level, n=400, wobble=0.00002, seed=0):
    rng = np.random.default_rng(seed)
    return list(level + rng.normal(0, level * wobble, n))


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


# --- what counts as a peg -------------------------------------------------

def test_a_dollar_peg_is_found_and_held(tmp_path):
    rows = _bars("USDCUSDT", _steady(1.0))
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 1
    assert table.rows.iloc[0].verdict == "HELD"


def test_a_euro_peg_is_judged_at_its_own_level_not_at_one(tmp_path):
    """EURIUSDT trades near 1.156. A hardcoded 1.0 would report a permanent
    16% depeg on an asset that never moved."""
    rows = _bars("EURIUSDT", _steady(1.156))
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 1
    got = table.rows.iloc[0]
    assert got.verdict == "HELD"
    assert 1.15 < got.peg_level < 1.16
    assert got.deviation < 0.001


def test_the_t_token_is_not_mistaken_for_trueusd(tmp_path):
    """TUSDT is the T token near $0.0037 with ordinary token volatility. Any
    rule keying on the symbol's spelling classifies it as a stablecoin."""
    rng = np.random.default_rng(1)
    closes = list(0.0037 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    rows = _bars("TUSDT", closes)
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.skipped["not_pegged"] == 1


def test_an_asset_between_the_bands_is_undecided_not_forced(tmp_path):
    rng = np.random.default_rng(2)
    closes = list(1.0 + rng.normal(0, 0.0008, 400))
    rows = _bars("DRIFTUSDT", closes)
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.skipped["undecided"] == 1


# --- the breach -----------------------------------------------------------

def test_a_depeg_past_the_assets_own_history_is_breached(tmp_path):
    """The USDe case: a peg that held all window and then broke."""
    closes = _steady(1.0, n=399) + [0.65]
    rows = _bars("USDEUSDT", closes)
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 1
    got = table.rows.iloc[0]
    assert got.verdict == "BREACHED"
    assert got.deviation > 0.3
    assert got.breach_threshold < 0.01, (
        "the threshold is this asset's own ordinary wobble, not a typed number")


def test_the_threshold_is_fitted_per_asset_not_shared(tmp_path):
    """A tight peg and a loose one must not inherit one number - that is the
    single threshold this module exists to avoid."""
    tight = _bars("TIGHTUSDT", _steady(1.0, wobble=0.00001, seed=3))
    loose = _bars("LOOSEUSDT", _steady(1.0, wobble=0.0002, seed=4))
    store = _write(tmp_path, tight + loose)
    table = monitor_pegs(store, _as_of(tight + loose))
    by_symbol = {r.symbol: r for r in table.rows.itertuples()}
    assert by_symbol["TIGHTUSDT"].breach_threshold < \
        by_symbol["LOOSEUSDT"].breach_threshold


def test_venues_are_judged_separately(tmp_path):
    """USDe hit $0.65 on Binance alone in Oct 2025. Averaging venues hides
    exactly the event this monitor exists for."""
    held = _bars("USDEUSDT", _steady(1.0, seed=5), venue="bybit")
    broke = _bars("USDEUSDT", _steady(1.0, n=399, seed=6) + [0.65],
                  venue="binance-spot")
    store = _write(tmp_path, held + broke)
    table = monitor_pegs(store, _as_of(held + broke))
    verdicts = {r.venue: r.verdict for r in table.rows.itertuples()}
    assert verdicts == {"bybit": "HELD", "binance-spot": "BREACHED"}


# --- refusals -------------------------------------------------------------

def test_too_short_a_history_is_skipped_and_counted(tmp_path):
    rows = _bars("NEWUSDT", _steady(1.0, n=50))
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.skipped["too_few_bars"] == 1


def test_the_clock_gates_the_monitor(tmp_path):
    """A depeg that lands after the as-of clock has not happened yet."""
    rows = _bars("USDEUSDT", _steady(1.0, n=399) + [0.65])
    store = _write(tmp_path, rows)
    before_break = rows[-1]["availability_time_ns"] - 1
    table = monitor_pegs(store, before_break)
    assert table.rows.iloc[0].verdict == "HELD"


def test_an_empty_store_judges_nothing_and_claims_nothing(tmp_path):
    table = monitor_pegs(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.skipped.values()) == 0


def test_a_peg_that_walked_to_a_new_level_is_not_reported_as_held(tmp_path):
    """Confident staleness: a depeg that persists becomes the new median, and
    every individual bar then looks calm at the broken level. The window's own
    first quarter against its last is what notices the level moved."""
    walked = _steady(1.0, n=200, seed=7) + _steady(0.97, n=200, seed=8)
    rows = _bars("SLIPUSDT", walked)
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert len(table.rows) == 1
    got = table.rows.iloc[0]
    assert got.level_moved, "the peg settled 3% lower and that must be visible"
    assert got.drift > 0.02
    assert got.peg_level > 0.99, "the level reported is the one it left, not the new one"


def test_a_tick_sized_wobble_does_not_breach_when_cost_can_floor_it(tmp_path):
    """A peg quiet enough to have almost no observed wobble gets a threshold
    near zero, and then every price tick is an alarm. Measured on the live
    store: six assets read BREACHED on two to four basis points, one against a
    threshold of 1e-6. The floor is the round-trip cost - a move smaller than
    the cost of trading it is not something anyone can act on."""
    closes = _steady(1.0, n=399, wobble=0.0000001, seed=9) + [1.0002]
    rows = _bars("USDCUSDT", closes, venue="binance")
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.threshold_floored, "binance perp fees are verified; this must floor"
    assert got.verdict == "HELD", "2 bps is inside a 10 bps round trip"
    assert got.breach_threshold >= 0.0005


def test_an_unfloored_threshold_says_so(tmp_path):
    """binance-spot has no verified fee schedule, so its thresholds rest on
    measured wobble alone - a weaker claim, and it must not read alike."""
    rows = _bars("USDCUSDT", _steady(1.0), venue="binance-spot")
    store = _write(tmp_path, rows)
    table = monitor_pegs(store, _as_of(rows))
    assert not table.rows.iloc[0].threshold_floored
