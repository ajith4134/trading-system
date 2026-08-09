"""The §5a.4 breadth gate: are these 850 bets, or one bet wearing 850 hats?

    If triggers cluster AND returns correlate, the architecture is one macro bet
    wearing 1,290 hats and must be redesigned.

The conjunction is load-bearing and the first draft of `verdict` read it as a
disjunction, which would have failed the gate on clustering alone - on real data
that flips the answer from PASS to REDESIGN, and would have sent Phase 4 back to
the drawing board over a capital-timing property §5a.5 already prescribes a rule
for.
"""
import numpy as np
import pandas as pd
import pytest

from validation.breadth import (
    COLLAPSE_BAND, INCONCLUSIVE, PASS, PASS_BURSTY, REDESIGN,
    clustering_ratio, effective_breadth, funding_matrix, measure,
    percentile_triggers, recent_window, run_gate, verdict,
)

DAY_NS = 86_400_000_000_000
SETTLE_NS = DAY_NS // 3          # 8-hourly


def _frame(rows):
    """rows: (symbol, event_time_ns, rate)"""
    return pd.DataFrame([{"symbol": s, "event_time_ns": t, "funding_rate": str(r)}
                         for s, t, r in rows])


# --------------------------------------------------------------------------
# the daily grid, which the real data forced
# --------------------------------------------------------------------------

def test_settlements_are_summed_into_days():
    """A 4-hourly symbol and an 8-hourly one are not comparable per settlement
    and are per day. Measured on the real history: modal gaps of 4h for 255,513
    settlements, 8h for 115,108, 1h for 6,018."""
    base = 1786_000_000_000_000_000 // DAY_NS * DAY_NS
    matrix = funding_matrix(_frame([
        ("FOURLY", base, 0.001), ("FOURLY", base + DAY_NS // 6, 0.001),
        ("FOURLY", base + 2 * DAY_NS // 6, 0.001),
        ("EIGHTLY", base, 0.003),
    ]))

    assert matrix.shape[0] == 1, "one day"
    assert matrix.loc[matrix.index[0], "FOURLY"] == pytest.approx(0.003)
    assert matrix.loc[matrix.index[0], "EIGHTLY"] == pytest.approx(0.003)


def test_millisecond_jitter_does_not_fragment_the_grid():
    """Real settlements land at `...200007`. Pivoting on the raw instant made a
    grid of 6,363 columns of time where every symbol occupied its own, and the
    coverage filter then correctly threw away all 850 of them."""
    base = 1786_000_000_000_000_000 // DAY_NS * DAY_NS
    matrix = funding_matrix(_frame([("A", base + 7_000_000, 0.001),
                                    ("B", base + 11_000_000, 0.002)]))

    assert matrix.shape == (1, 2)


def test_a_missing_rate_is_not_a_trigger():
    """A symbol that had not listed yet stood aside. Left as NA it propagates
    into the arithmetic and turns a boolean matrix into an object array."""
    matrix = pd.DataFrame({"A": [0.001] * 5 + [np.nan, 0.9]})
    fired = percentile_triggers(matrix, lookback=3, percentile=0.5)

    assert fired.dtypes.iloc[0] == bool
    assert not fired["A"].iloc[5]


def test_the_window_is_taken_before_coverage_is_judged():
    """The grid is ragged - symbols list and delist. Judging coverage against the
    union of 1,083 days would drop every symbol and report an empty universe as
    a finding."""
    matrix = pd.DataFrame({"A": range(100)})
    assert len(recent_window(matrix, 30)) == 30


# --------------------------------------------------------------------------
# the two measurements
# --------------------------------------------------------------------------

def test_independent_triggers_disperse_at_about_one():
    """The index of dispersion is 1.0 when symbols fire independently. If this
    drifts, every clustering verdict is measured against the wrong zero."""
    rng = np.random.default_rng(20260809)
    triggers = pd.DataFrame(rng.random((4000, 60)) < 0.1)

    ratio, _ = clustering_ratio(triggers)
    assert 0.85 < ratio < 1.15, ratio


def test_triggers_that_all_fire_on_the_same_days_score_far_above_one():
    rows, cols = 400, 50
    triggers = pd.DataFrame(False, index=range(rows), columns=range(cols))
    triggers.iloc[::10, :] = True            # every symbol, every tenth day

    ratio, busiest = clustering_ratio(triggers)
    assert ratio > 5, ratio
    assert busiest > 0.4


def test_independent_streams_keep_their_breadth():
    rng = np.random.default_rng(20260809)
    returns = pd.DataFrame(rng.normal(size=(1500, 40)))

    breadth, correlation = effective_breadth(returns)
    assert breadth > 30, breadth
    assert abs(correlation) < 0.05


def test_one_factor_collapses_breadth_to_almost_one():
    """The failure §5a.4 fears: 1,290 symbols that are one BTC factor plus noise."""
    rng = np.random.default_rng(20260809)
    factor = rng.normal(size=(1500, 1))
    returns = pd.DataFrame(factor + 0.01 * rng.normal(size=(1500, 40)))

    breadth, correlation = effective_breadth(returns)
    assert breadth < 2, breadth
    assert correlation > 0.9


def test_a_symbol_that_never_triggered_does_not_pad_the_count():
    """A constant-zero stream has undefined correlation with everything, and
    counting it would inflate breadth with a bet nobody placed."""
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(size=(500, 3)))
    returns[3] = 0.0

    breadth, _ = effective_breadth(returns)
    assert breadth <= 3


# --------------------------------------------------------------------------
# the verdict, and the conjunction that decides it
# --------------------------------------------------------------------------

def _measurement(breadth, ratio, symbols=400):
    from validation.breadth import BreadthMeasurement
    return BreadthMeasurement(
        setup="t", lookback=90, percentile=0.9, window_days=365, symbols=symbols,
        settlements=365, triggers=1000, trigger_rate=0.01,
        clustering_ratio=ratio, busiest_5pct_share=0.2,
        mean_pairwise_correlation=0.05, effective_breadth=breadth,
        effective_breadth_share=breadth / symbols)


def test_clustering_alone_is_not_a_redesign():
    """The bug this file exists for. §5a.4 says cluster AND correlate; read as
    OR, the real measurement - breadth 58, clustering 28x - reports REDESIGN and
    sends Phase 4 back over a problem §5a.5 already answers."""
    answer, why = verdict(_measurement(breadth=58.0, ratio=28.0))

    assert answer == PASS_BURSTY
    assert answer != REDESIGN
    assert "§5a.5" in why


def test_collapsed_breadth_is_a_redesign_however_the_triggers_arrive():
    """Correlated returns gut the proposition, and no arrival pattern saves it."""
    assert verdict(_measurement(breadth=8.0, ratio=28.0))[0] == REDESIGN
    assert verdict(_measurement(breadth=8.0, ratio=1.0))[0] == REDESIGN


def test_a_clean_pass_needs_both():
    assert verdict(_measurement(breadth=200.0, ratio=1.2))[0] == PASS


def test_the_collapse_band_is_the_corpus_number_not_one_invented_here():
    """§5a.4: "collapsing effective breadth to ~5-15". A gate that picked its own
    threshold would be grading itself against a number chosen to pass."""
    assert COLLAPSE_BAND == (5.0, 15.0)


def test_too_little_data_is_inconclusive_never_a_pass():
    answer, why = verdict(_measurement(breadth=float("nan"), ratio=float("nan")))
    assert answer == INCONCLUSIVE
    assert answer not in (PASS, PASS_BURSTY)


# --------------------------------------------------------------------------
# the scan is counted
# --------------------------------------------------------------------------

def test_every_scan_lands_in_the_trial_registry(tmp_path):
    """§5a.5: *every scan counts in the Trial Registry*. The count is what keeps
    a later deflated Sharpe honest about how many things were tried."""
    from validation.trial_registry import TrialRegistry

    rng = np.random.default_rng(7)
    base = 1786_000_000_000_000_000 // DAY_NS * DAY_NS
    rows = [(f"S{s}", base + d * DAY_NS, float(rng.normal(scale=0.001)))
            for s in range(6) for d in range(200)]
    registry = TrialRegistry(tmp_path)

    run_gate(_frame(rows), registry, lookback=30, percentile=0.9, window_days=150)
    run_gate(_frame(rows), registry, lookback=30, percentile=0.95, window_days=150)

    assert registry.cumulative_count() == 2
    assert all(t["family"] == "carry" for t in registry.trials())
    assert registry.trials()[0]["params"]["gate"].startswith("breadth")


def test_a_scan_that_raises_is_still_counted(tmp_path):
    """Counted first, then run - a search that abandons a candidate must not
    also abandon its N."""
    from validation.trial_registry import TrialRegistry
    registry = TrialRegistry(tmp_path)

    with pytest.raises(Exception):
        run_gate("not a frame", registry)

    assert registry.cumulative_count() == 1
