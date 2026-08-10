"""Mark against its venue's reference, judged rather than reported.

The claim this file defends is the one that separates DM-021 from the basis
feature next door: **the same number of basis points is fine on one instrument
and broken on another**, so the threshold has to come from each symbol's own
history. `test_the_same_gap_is_normal_on_one_symbol_and_extreme_on_another` is
that claim stated as a test - any fixed band fails it in one direction or the
other.
"""
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from features.price_divergence import (
    EXTREME, EXTREME_Z, MIN_OBSERVATIONS, NORMAL, STALE_MARK,
    UNJUDGED_NO_DISPERSION, UNJUDGED_TOO_FEW, WIDE, WIDE_Z, reconcile_prices,
)
from store.parquet_partition import append_partition

NOW = 1_786_336_806_435_346_403
_MINUTE_NS = 60_000_000_000


def _rows(symbol, marks, references, *, venue="binance", funds_on="mark",
          reference_column="index_price", first_ns=None, step_ns=_MINUTE_NS):
    """One funding row per (mark, reference) pair, a minute apart, ending now."""
    count = len(marks)
    start = first_ns if first_ns is not None else NOW - count * step_ns
    out = []
    for i, (mark, reference) in enumerate(zip(marks, references)):
        available = start + i * step_ns
        row = {
            "venue": venue, "symbol": symbol, "funding_rate": "0.0001",
            "mark_price": str(mark), "index_price": None, "oracle_price": None,
            "funds_on": funds_on, "funding_interval_hours": 8,
            "event_time_is_receipt": False, "next_funding_time_unknown": False,
            "next_funding_time_ns": 0,
            "event_time_ns": available, "ingestion_time_ns": available,
            "availability_time_ns": available,
        }
        row[reference_column] = None if reference is None else str(reference)
        out.append(row)
    return out


def _store(tmp_path, *row_groups, snapshot="divergence-test"):
    rows = [r for group in row_groups for r in group]
    append_partition(tmp_path, "funding", pd.DataFrame(rows), snapshot_id=snapshot)
    return tmp_path


def _steady(reference, gap_bps, n, *, noise_bps=0.0, seed=0):
    """Marks sitting `gap_bps` from a flat reference, with normal noise.

    Noise rather than an alternating wobble on purpose: a gap that takes
    exactly two values has a median absolute deviation of zero, because half
    the deviations are zero and the median of the rest is taken over a set that
    contains them. Such a fixture is judged UNJUDGED_NO_DISPERSION - correctly,
    and it is its own test below rather than an accident in every other one.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(0.0, noise_bps, n) if noise_bps else np.zeros(n)
    marks = [reference * (1 + (gap_bps + float(d)) / 10_000) for d in draws]
    return marks, [reference] * n


def _verdict(table, symbol):
    row = table.rows[table.rows["symbol"] == symbol]
    assert len(row) == 1, f"{symbol} appeared {len(row)} times"
    return row.iloc[0]


# --- the central claim ----------------------------------------------------

def test_the_same_gap_is_normal_on_one_symbol_and_extreme_on_another(tmp_path):
    """CALMUSDT never moves more than a fraction of a basis point from its
    index; WOBBLYUSDT swings 20 either way all day. A 40 bps gap is a broken
    feed on the first and an ordinary Tuesday on the second, and no fixed band
    can say both.
    """
    calm_marks, calm_refs = _steady(100.0, gap_bps=0.0, n=40, noise_bps=0.5)
    calm_marks[-1] = 100.0 * (1 + 40 / 10_000)
    wobbly_marks, wobbly_refs = _steady(100.0, gap_bps=0.0, n=40, noise_bps=20.0, seed=1)
    wobbly_marks[-1] = 100.0 * (1 + 40 / 10_000)

    store = _store(tmp_path,
                   _rows("CALMUSDT", calm_marks, calm_refs),
                   _rows("WOBBLYUSDT", wobbly_marks, wobbly_refs))
    table = reconcile_prices(store, NOW)

    assert _verdict(table, "CALMUSDT").verdict == EXTREME
    assert _verdict(table, "WOBBLYUSDT").verdict == NORMAL


def test_a_gap_inside_its_own_history_is_normal(tmp_path):
    marks, references = _steady(100.0, gap_bps=8.0, n=40, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    assert _verdict(reconcile_prices(store, NOW), "BTCUSDT").verdict == NORMAL


def test_a_moderate_excursion_is_wide_before_it_is_extreme(tmp_path):
    """The two cutoffs are a judgement and the middle band exists so that
    'unusual' and 'not the same distribution' are not the same alarm."""
    marks, references = _steady(100.0, gap_bps=0.0, n=40, noise_bps=1.0)
    marks[-1] = 100.0 * (1 + 4.0 / 10_000)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    row = _verdict(reconcile_prices(store, NOW), "BTCUSDT")

    assert row.verdict == WIDE
    assert WIDE_Z <= row.robust_z < EXTREME_Z


def test_the_current_observation_is_not_part_of_its_own_scale(tmp_path):
    """A single extreme point included in its own window drags the median and
    the dispersion toward itself and partly hides the excursion it is."""
    marks, references = _steady(100.0, gap_bps=0.0, n=40, noise_bps=1.0)
    marks[-1] = 100.0 * (1 + 40 / 10_000)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    row = _verdict(reconcile_prices(store, NOW), "BTCUSDT")

    assert row.observations == 39
    assert abs(row.median_gap_bps) < 2


# --- the states that are not verdicts about the market --------------------

def test_a_mark_frozen_while_the_reference_moves_is_stale_not_a_trend(tmp_path):
    """A dead feed produces a smoothly drifting basis, which reads as a trend
    to anything that only looks at the number."""
    references = [100.0 + i * 0.05 for i in range(40)]
    marks = [100.0] * 40
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    assert _verdict(reconcile_prices(store, NOW), "BTCUSDT").verdict == STALE_MARK


def test_too_little_history_is_stated_rather_than_called_normal(tmp_path):
    marks, references = _steady(100.0, gap_bps=3.0, n=10, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    row = _verdict(reconcile_prices(store, NOW), "BTCUSDT")

    assert row.verdict == UNJUDGED_TOO_FEW
    assert row.observations < MIN_OBSERVATIONS
    # Still a row. Dropping it would leave a display that reads as a market
    # where everything is fine.
    assert row.gap_bps > 0


def test_a_gap_that_never_varied_has_no_scale_to_judge_against(tmp_path):
    """Measured on the live store 2026-08-10: 212 of 1,854 instruments sit at
    a gap of exactly zero for every observation - the venue publishing its
    index back as the mark. Substituting a fallback band here would put back
    the fixed threshold this module exists to avoid."""
    marks, references = _steady(100.0, gap_bps=0.0, n=40, noise_bps=0.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    assert (_verdict(reconcile_prices(store, NOW), "BTCUSDT").verdict
            == UNJUDGED_NO_DISPERSION)


# --- which price is the reference -----------------------------------------

def test_hyperliquid_is_judged_against_its_oracle_not_an_index(tmp_path):
    """It funds on the oracle price; judging it against an index it does not
    publish would refuse every hyperliquid instrument."""
    marks, references = _steady(100.0, gap_bps=5.0, n=40, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTC", marks, references, venue="hyperliquid",
                                   funds_on="oracle",
                                   reference_column="oracle_price"))

    row = _verdict(reconcile_prices(store, NOW), "BTC")

    assert row.reference == "oracle_price"
    assert row.verdict == NORMAL


def test_a_venue_declaring_something_unrecognised_is_refused_not_guessed(tmp_path):
    marks, references = _steady(100.0, gap_bps=5.0, n=40, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references,
                                   funds_on="vibes"))

    table = reconcile_prices(store, NOW)

    assert table.rows.empty
    assert table.refused["unrecognised_funds_on"] == 1


def test_a_missing_reference_price_is_refused_and_counted(tmp_path):
    marks, references = _steady(100.0, gap_bps=5.0, n=40, noise_bps=1.0)
    references[-1] = None
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    table = reconcile_prices(store, NOW)

    assert table.rows.empty
    assert table.refused["no_reference_price"] == 1


def test_a_zero_reference_price_is_refused(tmp_path):
    marks, references = _steady(100.0, gap_bps=5.0, n=40, noise_bps=1.0)
    references[-1] = 0
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    table = reconcile_prices(store, NOW)

    assert table.rows.empty
    assert table.refused["unparseable_price"] == 1


# --- the clock and the window ---------------------------------------------

def test_observations_that_had_not_arrived_are_not_in_the_scale(tmp_path):
    """A scale fitted on data from after the decision point makes every past
    divergence look ordinary - the flattering direction, and invisible."""
    marks, references = _steady(100.0, gap_bps=0.0, n=40, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    early = NOW - 35 * _MINUTE_NS
    table = reconcile_prices(store, early)

    assert _verdict(table, "BTCUSDT").observations < 10


def test_history_older_than_the_window_is_not_used(tmp_path):
    marks, references = _steady(100.0, gap_bps=0.0, n=40, noise_bps=1.0)
    store = _store(tmp_path, _rows("BTCUSDT", marks, references))

    table = reconcile_prices(store, NOW, window_ns=5 * _MINUTE_NS)

    row = _verdict(table, "BTCUSDT")
    assert row.observations < 6
    assert row.verdict == UNJUDGED_TOO_FEW
    assert table.window_ns == 5 * _MINUTE_NS


def test_an_empty_store_is_an_empty_table_not_an_error(tmp_path):
    table = reconcile_prices(tmp_path, NOW)
    assert table.rows.empty
    assert table.by_verdict() == {}


def test_the_verdict_counts_are_what_the_tile_reads(tmp_path):
    calm_marks, calm_refs = _steady(100.0, gap_bps=0.0, n=40, noise_bps=0.5)
    calm_marks[-1] = 100.0 * (1 + 40 / 10_000)
    ok_marks, ok_refs = _steady(100.0, gap_bps=8.0, n=40, noise_bps=1.0)
    store = _store(tmp_path,
                   _rows("BROKENUSDT", calm_marks, calm_refs),
                   _rows("FINEUSDT", ok_marks, ok_refs))

    counts = reconcile_prices(store, NOW).by_verdict()

    assert counts == {EXTREME: 1, NORMAL: 1}
