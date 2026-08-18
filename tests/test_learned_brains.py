"""The guarantees the learned brain stack makes, and the ones it refuses to make.

Named for behaviour rather than for functions, following this repo's convention: a
test that fails should say what stopped being true, not which symbol moved.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from learn.axis_probes import FAIL, NOT_MEASURED, PASS, probe_out_of_regime, probe_realised_coverage
from learn.belief import (
    OBSERVED, READ, VERIFIED, Belief, ExpiredBeliefUsed, Provenance,
    ProvenanceRequired, UnknownEpistemicClass, observed, read,
)
from learn.online_calibration import MIN_CONFORMAL_SCORES, OnlineCalibration
from learn.training_set import FEATURE_NAMES, MarketBar, compute_features
from segment.live_features import LIVE_BAR_NS, _SealedBar

NOW = 1_787_000_000_000_000_000
MINUTE = 60_000_000_000


# --------------------------------------------------------------- beliefs (LB-03)

def test_a_belief_cannot_be_built_without_provenance():
    """§1a L1: a number whose origin nobody can name is what the standard catches."""
    with pytest.raises(ProvenanceRequired):
        Provenance(source="")


def test_a_belief_cannot_be_built_without_an_expiry():
    """§1a.0's confident staleness, written as a constant, is refused."""
    with pytest.raises(ValueError, match="half-life"):
        Belief(claim="x", value=1, epistemic_class=OBSERVED,
               provenance=Provenance(source="test"), held_at_ns=NOW, half_life_ns=0)


def test_only_an_observed_belief_may_size_a_position():
    """§1a.6 gives OBSERVED that privilege and gives the other two classes none."""
    fresh = dict(held_at_ns=NOW, half_life_ns=MINUTE)
    assert observed("c", 1, source="s", **fresh).may_size(NOW) is True
    assert read("c", 1, source="s", **fresh).may_size(NOW) is False
    verified = Belief(claim="c", value=1, epistemic_class=VERIFIED,
                      provenance=Provenance(source="s"), **fresh)
    assert verified.may_size(NOW) is False


def test_an_expired_belief_may_not_size_however_it_was_classed():
    """A belief that was true stops being allowed to act once it is stale."""
    belief = observed("c", 1, source="s", held_at_ns=NOW, half_life_ns=MINUTE)
    assert belief.may_size(NOW + MINUTE // 2) is True
    assert belief.may_size(NOW + MINUTE * 2) is False
    with pytest.raises(ExpiredBeliefUsed):
        belief.require_fresh(NOW + MINUTE * 2)


def test_an_unknown_epistemic_class_is_refused():
    with pytest.raises(UnknownEpistemicClass):
        Belief(claim="c", value=1, epistemic_class="GUESSED",
               provenance=Provenance(source="s"), held_at_ns=NOW,
               half_life_ns=MINUTE)


def test_a_belief_is_never_mutated_by_a_critique():
    """Critics annotate; they do not edit. The original stays auditable."""
    belief = observed("c", 1, source="s", held_at_ns=NOW, half_life_ns=MINUTE)
    critiqued = belief.critiqued({"critic": "t", "verdict": "survived"})
    assert belief.critiques == ()
    assert len(critiqued.critiques) == 1
    assert critiqued is not belief


def test_a_belief_describes_everything_an_auditor_needs():
    belief = observed("c", 0.7, source="model", held_at_ns=NOW, half_life_ns=MINUTE,
                      fit_reference={"trial_id": 4, "model_version": "abc"},
                      confidence=Decimal("0.7"))
    described = belief.describe(NOW)
    assert described["provenance"]["fit_reference"]["trial_id"] == 4
    assert described["provenance"]["is_learned"] is True
    assert described["may_size"] is True
    # It must serialise: a belief that cannot be journalled cannot be audited.
    json.dumps(described, default=str)


# -------------------------------------------------- live calibration (LB-05, R4)

def test_an_uncalibrated_bin_passes_the_raw_score_through_and_says_so():
    """Sizing off an uncalibrated score is the error dual-agent-spec names."""
    calibration = OnlineCalibration(segment="perp")
    probability, is_calibrated = calibration.calibrate(0.73)
    assert probability == 0.73
    assert is_calibrated is False


def test_calibration_learns_a_bin_from_realised_outcomes():
    """The live loop's own parameter update (§1a L2)."""
    calibration = OnlineCalibration(segment="perp")
    # The model says 0.75 here; it is actually right 20% of the time.
    for i in range(100):
        calibration.observe(score=0.75, outcome=1 if i % 5 == 0 else 0, acted=False)
    probability, is_calibrated = calibration.calibrate(0.75)
    assert is_calibrated is True
    assert probability == pytest.approx(0.2, abs=0.05), (
        "the bin must report what happened, not what the model claimed")


def test_a_brain_with_no_track_record_abstains_on_everything():
    """R4: no coverage guarantee means no trade, not a hedged one."""
    calibration = OnlineCalibration(segment="perp")
    abstain, evidence = calibration.should_abstain(0.99)
    assert abstain is True
    assert evidence["reason"] == "NO_CONFORMAL_TRACK_RECORD"
    assert evidence["conformal_quantile"] is None


def test_the_conformal_quantile_appears_once_there_is_a_record():
    calibration = OnlineCalibration(segment="perp")
    for i in range(MIN_CONFORMAL_SCORES + 10):
        calibration.observe(score=0.9, outcome=1, acted=True)
    assert calibration.quantile() is not None
    abstain, evidence = calibration.should_abstain(0.9)
    assert abstain is False
    assert evidence["reason"] == "WITHIN_COVERAGE"


def test_abstention_rises_as_the_model_becomes_wrong():
    """§1a.5's master test: does it change behaviour when the system is WRONG?

    This is the whole justification for the module. A model that starts being
    wrong pushes its own non-conformity scores up, which raises the quantile, which
    makes the brain abstain more - with nobody intervening.
    """
    calibration = OnlineCalibration(segment="perp")
    for _ in range(MIN_CONFORMAL_SCORES + 20):
        calibration.observe(score=0.95, outcome=1, acted=True)
    confident_quantile = calibration.quantile()

    for _ in range(200):
        calibration.observe(score=0.95, outcome=0, acted=True)   # confidently wrong
    degraded_quantile = calibration.quantile()

    assert degraded_quantile > confident_quantile, (
        "being wrong must widen the non-conformity quantile; if it does not, "
        "abstention cannot respond to the model degrading")


def test_realised_coverage_is_audited_against_the_promise():
    """R4's real requirement - a guarantee nobody checks is a hedge with arithmetic."""
    calibration = OnlineCalibration(segment="perp", alpha=0.35)
    for i in range(100):
        calibration.observe(score=0.8, outcome=1 if i % 10 else 0, acted=True)
    coverage = calibration.realised_coverage()
    assert coverage["promised_coverage"] == 0.65
    assert coverage["realised_coverage"] is not None
    assert coverage["acted_on"] == 100
    assert coverage["fitted_by"] == "live loop", (
        "§1a L2 turns on distinguishing what the live loop fitted from what a "
        "retrainer set; the label must ride with the numbers")


def test_calibration_survives_a_restart():
    """A bot that forgot how wrong it had been would be permanently uncalibrated."""
    import tempfile

    calibration = OnlineCalibration(segment="perp")
    for i in range(60):
        calibration.observe(score=0.6, outcome=i % 2, acted=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        calibration.save(path)
        resumed = OnlineCalibration.load(path, segment="perp")
    assert resumed.updates == calibration.updates
    assert resumed.acted == calibration.acted
    assert resumed.quantile() == calibration.quantile()


def test_a_corrupt_calibration_starts_fresh_rather_than_stopping_the_bot():
    """The safe state is already the conservative one: no record means abstain."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.json"
        path.write_text("{not json", encoding="utf-8")
        resumed = OnlineCalibration.load(path, segment="perp")
    assert resumed.updates == 0
    assert resumed.quantile() is None


# ------------------------------------------------------------- features (LB-01)

def _window(n=60, drift=0.1):
    return [MarketBar(event_time_ns=i * MINUTE, open=100 + i * drift,
                      high=100.5 + i * drift, low=99.5 + i * drift,
                      close=100.2 + i * drift, volume=10.0, trades=5.0)
            for i in range(n)]


def test_a_feature_row_has_exactly_the_declared_names():
    """Order is the contract: a booster splits on column indices."""
    features = compute_features(_window())
    assert features is not None
    assert len(features) == len(FEATURE_NAMES)


def test_a_short_window_refuses_rather_than_padding():
    """A padded window would mean horizons that are not the ones the model learned."""
    assert compute_features(_window(n=10)) is None


def test_every_feature_is_finite():
    """A NaN reaching a booster produces a prediction with no error raised."""
    import math

    for drift in (0.0, 0.1, -0.1):
        for value in compute_features(_window(drift=drift)):
            assert math.isfinite(value)


def test_the_live_bar_and_the_training_bar_are_the_same_shape():
    """Train/serve skew is prevented structurally, so the shapes must not drift."""
    import dataclasses

    live = [f.name for f in dataclasses.fields(_SealedBar)]
    training = [f.name for f in dataclasses.fields(MarketBar)]
    assert live == training
    assert LIVE_BAR_NS == 60_000_000_000, (
        "the live bar width must match bars_60000000000ns, the dataset the models "
        "are fitted on")


def test_the_live_bars_feed_the_training_feature_function_unchanged():
    """The same function, not a reimplementation that agrees today."""
    sealed = [_SealedBar(event_time_ns=i * MINUTE, open=100 + i * 0.1,
                         high=100.5 + i * 0.1, low=99.5 + i * 0.1,
                         close=100.2 + i * 0.1, volume=10.0, trades=5.0)
              for i in range(60)]
    assert compute_features(sealed) == compute_features(_window())


# ----------------------------------------------------------- axis probes (LB-07)

def test_out_of_regime_is_reported_as_failing_and_never_omitted():
    """An absent test reads as a passed one on a board (Rule 8)."""
    result = probe_out_of_regime({"days_covered": ["2026-08-17", "2026-08-18"]})
    assert result.verdict == FAIL
    assert result.axis_test == "L6"
    assert result.detail["n_days"] == 2


def test_coverage_with_no_resolved_predictions_is_not_measured_rather_than_green():
    calibration = OnlineCalibration(segment="perp")
    result = probe_realised_coverage(calibration)
    assert result.verdict == NOT_MEASURED


def test_coverage_passes_only_when_the_promise_actually_held():
    kept = OnlineCalibration(segment="perp", alpha=0.35)
    for _ in range(100):
        kept.observe(score=0.9, outcome=1, acted=True)
    assert probe_realised_coverage(kept).verdict == PASS

    broken = OnlineCalibration(segment="perp", alpha=0.05)
    for i in range(100):
        broken.observe(score=0.9, outcome=1 if i % 2 else 0, acted=True)
    assert probe_realised_coverage(broken).verdict == FAIL
