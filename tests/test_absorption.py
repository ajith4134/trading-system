"""Delta-vs-price-hold: strong flow alone is not absorption, and neither is a
bar that merely failed to advance. Absorption is the CONJUNCTION - and every
test here that claims absorption also claims one of the two conditions was
missing, to prove the conjunction is actually enforced rather than either half
alone driving the verdict.

There is no signed/taker-side volume column in `bars_60000000000ns` (only
open, high, low, close, volume, trades), so `proxy_delta` is a disclosed proxy
- volume times the bar's own intrabar excursion away from open, toward
whichever of high/low it travelled further. These fixtures use the real
dataset schema and the real store writer, so a schema drift breaks these tests
rather than silently breaking the feature.
"""
from decimal import Decimal

import pandas as pd

from features import staleness
from features.absorption import compute_absorption
from store.parquet_partition import append_partition

_BAR_NS = 60_000_000_000


def _bar(symbol, o, h, l, c, volume, i, venue="binance-spot", trades=5):
    return {
        "venue": venue, "symbol": symbol,
        "open": o, "high": h, "low": l, "close": c,
        "volume": volume, "trades": trades,
        "event_time_ns": 10**12 + i * _BAR_NS,
        "ingestion_time_ns": 10**12 + i * _BAR_NS,
        "availability_time_ns": 10**12 + i * _BAR_NS,
    }


def _write(tmp_path, rows, snapshot="absorption-test"):
    append_partition(tmp_path, "bars_60000000000ns", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _as_of(rows):
    return max(r["availability_time_ns"] for r in rows) + 1


def _quiet_baseline(symbol, n=20, venue="binance-spot", start=0):
    """20 identical quiet bars: open 100, high 100.5, low 99.5, close 100,
    volume 10. Hand-computable: push_up == push_down == 0.5 (a tie, which
    resolves to `up` by this module's disclosed tie-break), so
    proxy_delta = 10 * 0.5 = 5 on every baseline bar, median(baseline) = 5,
    and the strong-flow threshold this fixture always produces is
    2 * 5 = 10.
    """
    return [_bar(symbol, 100.0, 100.5, 99.5, 100.0, 10.0, start + i, venue)
            for i in range(n)]


# --- the conjunction, hand-computed ---------------------------------------

def test_strong_flow_that_holds_is_not_absorption(tmp_path):
    """push_up = 10, volume = 50 -> proxy_delta = 500, far past the
    threshold of 10 -> strong flow. retention = (109.5-100)/10 = 0.95 ->
    held. Strong delta, price advanced and kept it: real strength, not
    absorption."""
    rows = _quiet_baseline("BTCUSDT")
    rows.append(_bar("BTCUSDT", 100.0, 110.0, 99.5, 109.5, 50.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.direction == "up"
    assert got.proxy_delta == Decimal("500")
    assert got.proxy_delta_threshold == Decimal("10")
    assert bool(got.proxy_flow_strong) is True
    assert got.retention == Decimal("0.95")
    assert bool(got.price_held) is True
    assert bool(got.absorption) is False
    assert got.absorption_signal is None
    assert sum(table.refused.values()) == 0


def test_strong_flow_that_fails_to_hold_is_absorption_and_bearish(tmp_path):
    """Same push and volume as the case above (proxy_delta = 500, strong),
    but the close gives almost all of it back: retention =
    (100.5-100)/10 = 0.05 -> not held. Strong delta at a high that could not
    hold is absorption, and the catalogue calls it bearish because the
    buying got absorbed."""
    rows = _quiet_baseline("ETHUSDT")
    rows.append(_bar("ETHUSDT", 100.0, 110.0, 99.5, 100.5, 50.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.direction == "up"
    assert got.proxy_delta == Decimal("500")
    assert bool(got.proxy_flow_strong) is True
    assert got.retention == Decimal("0.05")
    assert bool(got.price_held) is False
    assert bool(got.absorption) is True
    assert got.absorption_signal == "bearish"


def test_strong_down_flow_that_fails_to_hold_is_absorption_and_bullish(tmp_path):
    """The symmetric case: push_down = 10, volume = 50 -> proxy_delta =
    -500 (strong). retention = (100-99.5)/10 = 0.05 -> not held. Selling
    that could not hold the low is bullish - the selling got absorbed."""
    rows = _quiet_baseline("SOLUSDT")
    rows.append(_bar("SOLUSDT", 100.0, 100.5, 90.0, 99.5, 50.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.direction == "down"
    assert got.proxy_delta == Decimal("-500")
    assert bool(got.proxy_flow_strong) is True
    assert got.retention == Decimal("0.05")
    assert bool(got.price_held) is False
    assert bool(got.absorption) is True
    assert got.absorption_signal == "bullish"


def test_weak_flow_that_fails_to_hold_is_still_not_absorption(tmp_path):
    """This is the conjunction test that actually bites: retention here is
    0.0 - literally worse than the absorption case above - but proxy_delta
    is only 6, under the threshold of 10. Absorption requires BOTH strong
    flow AND a failure to hold; a merely-quiet bar that drifted back to its
    open is not absorption just because it also failed to hold, because
    there was no real push behind it to begin with. Scoring on
    "failed to hold" alone is the naive half of the catalogue's complaint."""
    rows = _quiet_baseline("DOGEUSDT")
    rows.append(_bar("DOGEUSDT", 100.0, 100.6, 99.6, 100.0, 10.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    got = table.rows.iloc[0]
    assert got.proxy_delta == Decimal("6.0")
    assert bool(got.proxy_flow_strong) is False
    assert bool(got.price_held) is False, "the price genuinely did not hold"
    assert bool(got.absorption) is False, (
        "no absorption without strong flow, even though price failed to hold")


# --- refusals: a bar that cannot be judged is refused, never scored calm --

def test_a_zero_range_bar_is_refused_never_scored_as_no_absorption(tmp_path):
    """The master case: high == low means there is no push to have held or
    not held. Retention is undefined, not zero - so this must be a refusal,
    and the symbol must be entirely ABSENT from `rows`, not present with
    absorption=False. A silent 'no absorption' here is indistinguishable
    from a calm market, which is the exact failure this house is written
    against."""
    rows = _quiet_baseline("FLATUSDT")
    rows.append(_bar("FLATUSDT", 100.0, 100.0, 100.0, 100.0, 10.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["zero_range_bar"] == 1
    assert "FLATUSDT" not in set(table.rows.get("symbol", []))


def test_too_few_bars_is_refused_and_counted(tmp_path):
    rows = _quiet_baseline("NEWUSDT", n=10)
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["too_few_bars"] == 1


def test_a_window_with_no_volume_is_refused_not_scored_calm(tmp_path):
    rows = [_bar("QUIETUSDT", 100.0, 100.5, 99.5, 100.0, 0.0, i)
            for i in range(21)]
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["no_volume"] == 1


def test_a_non_positive_price_bar_is_refused_not_dropped_silently(tmp_path):
    rows = _quiet_baseline("BADUSDT", n=21)
    rows[5]["low"] = 0.0
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["non_positive_price"] == 1


def test_a_nan_price_is_refused_not_treated_as_zero(tmp_path):
    rows = _quiet_baseline("NANUSDT", n=21)
    rows[3]["close"] = float("nan")
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["unparseable_price"] == 1


def test_an_internally_inconsistent_bar_is_refused(tmp_path):
    """low above high is not a real bar; treating it as data would let a
    corrupt row silently drive a verdict."""
    rows = _quiet_baseline("BROKENUSDT", n=21)
    rows[10]["low"], rows[10]["high"] = 101.0, 100.0
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    assert len(table.rows) == 0
    assert table.refused["invalid_bar_shape"] == 1


def test_an_empty_store_measures_nothing(tmp_path):
    table = compute_absorption(tmp_path, as_of_ns=10**18)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0


# --- clock gating and staleness -------------------------------------------

def test_the_clock_gates_the_baseline_and_the_judged_bar(tmp_path):
    """A late, huge-volume run of bars must not leak into the baseline
    median or replace the judged bar for a clock set before they existed."""
    early = _quiet_baseline("GATEDUSDT", n=20)
    early.append(_bar("GATEDUSDT", 100.0, 110.0, 99.5, 109.5, 50.0, 20))
    late = [_bar("GATEDUSDT", 100.0, 1000.0, 1.0, 1000.0, 999_999.0, i)
            for i in range(21, 60)]
    store = _write(tmp_path, early + late)
    table = compute_absorption(store, _as_of(early))
    got = table.rows.iloc[0]
    assert got.proxy_delta == Decimal("500"), (
        "the late run must not have become the judged bar")
    assert got.proxy_delta_threshold == Decimal("10"), (
        "the late run's huge volume must not have entered the baseline median")


def test_staleness_is_stamped_on_every_row(tmp_path):
    rows = _quiet_baseline("FRESHUSDT")
    rows.append(_bar("FRESHUSDT", 100.0, 110.0, 99.5, 109.5, 50.0, 20))
    store = _write(tmp_path, rows)
    table = compute_absorption(store, _as_of(rows))
    got = table.rows.iloc[0]
    for column in staleness.COLUMNS:
        assert column in table.rows.columns
    assert got.freshness == staleness.FRESH
    assert got.routine_gap_ns == _BAR_NS


def test_venues_are_judged_separately(tmp_path):
    """The same symbol can be absorbing on one venue and clean on another -
    grouping by symbol alone would blend the two into one wrong verdict."""
    held = _quiet_baseline("XUSDT", venue="binance")
    held.append(_bar("XUSDT", 100.0, 110.0, 99.5, 109.5, 50.0, 20, venue="binance"))
    absorbed = _quiet_baseline("XUSDT", venue="binance-spot")
    absorbed.append(_bar("XUSDT", 100.0, 110.0, 99.5, 100.5, 50.0, 20, venue="binance-spot"))
    store = _write(tmp_path, held + absorbed)
    table = compute_absorption(store, _as_of(held + absorbed))
    by_venue = {r.venue: r.absorption for r in table.rows.itertuples()}
    assert bool(by_venue["binance"]) is False
    assert bool(by_venue["binance-spot"]) is True
