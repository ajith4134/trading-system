"""A champion must prove it was fitted on the segment it is about to decide for.

Measured 2026-08-18: spot's own-venue fit found no edge (p=0.4645) and was
correctly refused, while an earlier POOLED model still occupied the
`spot-direction-champion` alias. The spot bot loaded it and started deciding from
a model fitted partly on binance FUTURES bars - and every board it appeared on
read `learned`, because the existence of a champion had been taken as evidence of
its provenance.

RL-019 makes each segment its own bot with its own data. The load is the only
place that can hold that line once an alias exists, so these tests pin the
refusal and pin that the refusal is recorded rather than logged and forgotten.
"""
from dataclasses import dataclass

from learn import learned_brains
from segment import bot_registry


@dataclass
class _FakeChampion:
    version_id: str
    metrics: dict
    booster: object = None


class _StubBrain:
    """Stands in for the learned brains so the test needs no LightGBM artefact."""

    def __init__(self, *args, **kwargs):
        self.name = "stub-learned"
        self.makes_edge_claim = True

    def __call__(self, *args, **kwargs):
        raise AssertionError("the brain is not called by loading")


class _StubTail(_StubBrain):
    """The rule PROFIT-TAIL, whose horizons the learned half inherits."""

    segment = "perp"
    name = "perp-profit-tail-fast"
    take_profit = 0.004
    ratchet_trigger = 0.002
    ratchet_give_back = 0.5
    signal_expiry_ns = 30_000_000_000
    max_hold_ns = 300_000_000_000


def _load_with(monkeypatch, metrics, segment="perp"):
    """Run the champion swap against a fabricated registry entry."""
    monkeypatch.setattr(learned_brains, "load_champion",
                        lambda seg, **kw: _FakeChampion("deadbeef", metrics))
    # The tail champion is a mapping in the registry, not a LoadedModel.
    monkeypatch.setattr(learned_brains, "load_tail_champion",
                        lambda seg, **kw: {"version_id": "tail0001"})
    monkeypatch.setattr(learned_brains, "LearnedBullBrain", _StubBrain)
    monkeypatch.setattr(learned_brains, "LearnedBearBrain", _StubBrain)
    monkeypatch.setattr(learned_brains, "LearnedProfitTail", _StubBrain)

    bot = bot_registry.SegmentBot(
        segment=segment, venue="binance-futures", build_feed=lambda: None,
        bull=_StubBrain(), bear=_StubBrain(), profit_tail=_StubTail(),
        admit=lambda frames: [], describe_universe=lambda decisions: {},
        quantity=bot_registry.Decimal("0.001"),
        min_confidence=bot_registry.Decimal("0.55"),
        min_margin=bot_registry.Decimal("0.05"), max_loss_tail=None, band="fast")
    return bot_registry._with_learned_brains(bot)


def test_a_pooled_champion_is_refused_for_a_segment_that_declares_its_own_venues(
        monkeypatch):
    loaded = _load_with(monkeypatch, {"fitted_on_venues": "POOLED"})

    assert not loaded.learned, "a pooled model must not make a segment read learned"
    assert loaded.extra["champion_refused"]["fitted_on_venues"] == "POOLED"


def test_a_champion_fitted_on_another_segments_venues_is_refused(monkeypatch):
    loaded = _load_with(monkeypatch, {"fitted_on_venues": ["deribit"]})

    refusal = loaded.extra["champion_refused"]
    assert refusal["model_version"] == "deadbeef"
    assert "another segment's data" in refusal["why"]


def test_a_champion_with_no_provenance_field_at_all_is_refused(monkeypatch):
    loaded = _load_with(monkeypatch, {})

    assert loaded.extra["champion_refused"]["fitted_on_venues"] is None


def test_the_refusal_is_recorded_on_the_bot_rather_than_only_logged(monkeypatch):
    loaded = _load_with(monkeypatch, {"fitted_on_venues": "POOLED"})

    refusal = loaded.extra["champion_refused"]
    assert set(refusal) == {"model_version", "fitted_on_venues", "segment_venues", "why"}
    assert refusal["segment_venues"], "the refusal must name what was expected"


def test_a_champion_fitted_on_this_segments_own_venues_is_loaded(monkeypatch, tmp_path):
    from learn.training_set import SEGMENT_VENUES
    monkeypatch.setattr(bot_registry, "CALIBRATION_PATH",
                        str(tmp_path / "{segment}" / "calibration.json"))

    loaded = _load_with(monkeypatch,
                        {"fitted_on_venues": sorted(SEGMENT_VENUES["perp"]),
                         "conformal_scores": [0.4, 0.5]})

    assert loaded.learned
    assert loaded.model_version == "deadbeef"
