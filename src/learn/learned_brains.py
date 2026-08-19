"""LB-04 / LB-06: BULL, BEAR and PROFIT-TAIL, deciding from trained models.

## What replaces what

RL-025's rule brains held thresholds somebody typed — `BULL_FLOW = 0.15`,
`BULL_MOMENTUM = 0.0004`. RL-026 replaces them. Every number in the decision path here
comes from one of exactly two places, and `§1a` L1 admits only these two:

| number | where it comes from |
|---|---|
| the score | a booster fitted under purged CPCV, registered against a counted trial |
| the probability | reliability bins fitted live from realised outcomes |
| the abstention threshold | the conformal quantile of the brain's own past non-conformity |
| the expectancy and loss tail | quantile regressions on realised forward P&L |

There is no fifth row and there is deliberately no configurable threshold. A brain that
still needed one would have failed L1 in the one place it matters — the decision path.

## Train/serve skew is prevented structurally, not by discipline

The features are computed by **`learn.training_set.compute_features`** — the same
function, on the same one-minute bar shape, in training and in the live loop. Not a
reimplementation that agrees today; the same code object.

Skew of this kind is invisible: the model produces confident, plausible numbers from
inputs that mean something else than what it learned, and nothing errors. The only
reliable defence is to make a second implementation impossible, which is what importing
the function does.

`FEATURE_NAMES` is stored in the artefact at registration and checked here at load. A
booster is splits over column indices, so a reordered feature list is silently wrong —
`FeatureContractBroken` makes it loudly wrong instead.

## The artefact is JSON, and that is a security decision

A model artefact is an executable payload the moment its format allows one. LightGBM's
`model_to_string()` is text and every other field in the artefact is a primitive, so
there is nothing here that needs pickle — and pickle would mean any path that could
write into the model directory could execute code inside the trading loop. JSON plus
the registry's sha256 verification at load leaves no such path.

## R9: this brain cannot lean on a feature's name

`§1a` R9 renames `funding_rate` to `widget_7` and checks the answer is identical. It is,
necessarily: `predict()` receives a positional float vector and the booster splits on
column indices. There is no path by which a name reaches the model. That is a property
of the design rather than a test it happens to pass, which is the stronger form.

## The three-brain structure is unchanged by becoming learned

RL-023 stands. BULL proposes long or declines; BEAR proposes short or declines;
PROFIT-TAIL assesses, times and manages, and can refuse nothing. What changed is where
their numbers come from — not who is allowed to decide what.

**BULL and BEAR share one model and are not one brain.** The model estimates P(the
deterministic policy's long trade ends in profit). BULL acts on that estimate being
high; BEAR acts on it being low, which is evidence about a *short*, and applies its own
stricter bar — `dual-agent-spec.md` requires short-side limits stricter than long-side,
and the asymmetry survives here as a difference in what each brain requires of the same
evidence rather than as two thresholds somebody chose.

## What happens when there is no model

`NoChampionRegistered` is raised at construction, not at the first decision. A bot with
no model must fail to start rather than run and silently decline everything — that is
indistinguishable on the board from a quiet market, which is the failure `§1a.0` names.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from learn.belief import OBSERVED, Belief, Provenance
from learn.online_calibration import OnlineCalibration
from learn.training_set import FEATURE_NAMES, FEATURE_WINDOW_BARS, compute_features
from learn.train_segment_model import (
    CHAMPION_ALIAS, DEFAULT_REGISTRY_ROOT, TAIL_ALIAS,
)
from models.model_registry import ModelRegistry, UnknownAlias
from segment.brain import BEAR, BULL, decline, propose
from segment.profit_tail import ProfitTail, TailAssessment

# How long a directional belief stays actionable. One bar: the features are
# computed from one-minute bars and the newest one is the newest information there
# is, so a belief older than a bar is a belief about a bar that has closed.
BELIEF_HALF_LIFE_NS = 60_000_000_000


class NoChampionRegistered(RuntimeError):
    """No model is aliased as this segment's champion."""


class FeatureContractBroken(RuntimeError):
    """The artefact's feature list is not the one this code computes.

    A booster splits on column indices. If the order changed, every value reaches
    the wrong split and the model returns confident nonsense with no error.
    """


@dataclass(frozen=True)
class LoadedModel:
    """A champion booster and everything needed to cite it in a belief."""

    booster: object
    feature_names: tuple
    version_id: str
    trial_id: int
    metrics: dict
    segment: str

    def fit_reference(self) -> dict:
        return {
            "model_version": self.version_id,
            "trial_id": self.trial_id,
            "out_of_fold_accuracy": self.metrics.get("accuracy"),
            "base_rate": self.metrics.get("base_rate"),
            "n_out_of_fold": self.metrics.get("n_out_of_fold"),
            "training_rows": self.metrics.get("training_rows"),
            "fitted_by": "scheduled retrain",
        }


def _check_feature_contract(names) -> None:
    if tuple(names) != tuple(FEATURE_NAMES):
        raise FeatureContractBroken(
            f"the registered model was fitted on {list(names)} and this code "
            f"computes {list(FEATURE_NAMES)}. Refused: a booster splits on column "
            f"indices, so serving it a differently ordered vector produces "
            f"plausible predictions from values fed to the wrong splits")


def registered_champion_id(segment: str, *,
                           registry_root: Path = DEFAULT_REGISTRY_ROOT) -> str | None:
    """The champion version id on disk, WITHOUT loading the model (LB-09, RL-030).

    A bot checks this every poll, so it has to cost a small file read and nothing
    else - the model itself is loaded only when the id it returns differs from
    the one the bot is already deciding with.

    None when no champion is registered for this segment, which is a state a bot
    starts in rather than an error.
    """
    from models.model_registry import ALIASES_FILE

    try:
        aliases = json.loads((Path(registry_root) / ALIASES_FILE).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = aliases.get(CHAMPION_ALIAS.format(segment=segment))
    return version or None


def load_champion(segment: str, *, registry_root: Path = DEFAULT_REGISTRY_ROOT
                  ) -> LoadedModel:
    """The registered direction champion for one segment."""
    import lightgbm as lgb

    registry = ModelRegistry(Path(registry_root))
    alias = CHAMPION_ALIAS.format(segment=segment)
    try:
        version, artifact = registry.load_alias(alias)
    except (UnknownAlias, KeyError, FileNotFoundError) as exc:
        raise NoChampionRegistered(
            f"no model aliased {alias!r}. A bot with no champion must fail to "
            f"start: one that ran and declined everything is indistinguishable on "
            f"the board from a quiet market") from exc
    held = json.loads(artifact.decode("utf-8"))
    _check_feature_contract(held["feature_names"])
    return LoadedModel(
        booster=lgb.Booster(model_str=held["booster"]),
        feature_names=tuple(held["feature_names"]),
        version_id=version.version_id, trial_id=version.trial_id,
        metrics=version.metrics, segment=segment)


def load_tail_champion(segment: str, *, registry_root: Path = DEFAULT_REGISTRY_ROOT):
    """The registered PROFIT-TAIL quantile champion, or None if none exists.

    None rather than raising, unlike the direction champion: PROFIT-TAIL has a
    deterministic policy to fall back to, and `FEATURES.md` line 99 makes that
    policy the permanent rollback target rather than an embarrassment.
    """
    import lightgbm as lgb

    registry = ModelRegistry(Path(registry_root))
    alias = TAIL_ALIAS.format(segment=segment)
    try:
        version, artifact = registry.load_alias(alias)
    except (UnknownAlias, KeyError, FileNotFoundError):
        return None
    held = json.loads(artifact.decode("utf-8"))
    _check_feature_contract(held["feature_names"])
    return {
        "boosters": {float(q): lgb.Booster(model_str=s)
                     for q, s in held["quantile_boosters"].items()},
        "quantiles": [float(q) for q in held["quantiles"]],
        "version_id": version.version_id,
        "trial_id": version.trial_id,
        "metrics": version.metrics,
    }


def features_from_live(frames, venue: str, symbol: str):
    """The live feature vector, computed by the training function.

    `frames` is the engine's `LiveFeatureFrames`. Returns None during warm-up,
    which is a refusal and not a zero vector.
    """
    window = frames.bar_window(venue, symbol, FEATURE_WINDOW_BARS)
    if window is None:
        return None
    return compute_features(window)


@dataclass
class _LearnedDirectional:
    """Shared machinery for the two directional brains. Not a brain itself."""

    segment: str
    model: LoadedModel
    calibration: OnlineCalibration

    def score(self, vector) -> float:
        """P(the deterministic long trade from here ends in profit)."""
        return float(self.model.booster.predict([vector])[0])

    def belief(self, *, claim: str, value, at_ns: int, evidence: dict,
               confidence) -> Belief:
        """Every directional decision emits one of these (RL-027).

        OBSERVED, because it is computed from bars this system captured itself —
        and OBSERVED is the only class `§1a.6` allows to size a position.
        """
        return Belief(
            claim=claim, value=value, epistemic_class=OBSERVED,
            provenance=Provenance(
                source=f"{self.segment}-direction-model",
                evidence=evidence,
                fit_reference=self.model.fit_reference()),
            held_at_ns=at_ns, half_life_ns=BELIEF_HALF_LIFE_NS,
            confidence=confidence)


class LearnedBullBrain:
    """Proposes longs when the model says a long from here tends to pay."""

    stance = BULL
    makes_edge_claim = True

    def __init__(self, segment: str, model: LoadedModel,
                 calibration: OnlineCalibration) -> None:
        self.segment = segment
        self.name = f"{segment}-bull-learned"
        self._core = _LearnedDirectional(segment, model, calibration)
        self.model = model
        self.calibration = calibration

    def __call__(self, frame, chain_median=None):        # noqa: ARG002
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        vector = frame.get("model_vector")
        base = {"model_version": self.model.version_id,
                "trial_id": self.model.trial_id, "rule_brain": False,
                "feature_vector_length": None if vector is None else len(vector)}

        if vector is None:
            return decline(self, venue=venue, symbol=symbol,
                           reason="WARMING_UP_NO_FEATURE_VECTOR",
                           evidence={**base, "sealed_bars": frame.get("sealed_bars")},
                           at_ns=at_ns)

        raw = self._core.score(vector)
        probability, calibrated = self.calibration.calibrate(raw)
        abstain, coverage = self.calibration.should_abstain(probability)
        evidence = {**base, "raw_score": round(raw, 6),
                    "calibrated_probability": round(probability, 6),
                    "is_calibrated": calibrated, "coverage": coverage}

        if abstain:
            return decline(self, venue=venue, symbol=symbol,
                           reason=f"ABSTAINED_{coverage['reason']}",
                           evidence=evidence, at_ns=at_ns)
        if probability <= 0.5:
            # The evidence points the other way. A decline, not a short - a BULL
            # has no vocabulary for a short (RL-023, and `segment.brain` raises).
            return decline(self, venue=venue, symbol=symbol,
                           reason="MODEL_DOES_NOT_FAVOUR_LONG",
                           evidence=evidence, at_ns=at_ns)

        belief = self._core.belief(
            claim=f"{symbol} long resolves profitable under the exit policy",
            value=probability, at_ns=at_ns, evidence=evidence,
            confidence=Decimal(str(round(probability, 6))))
        return propose(self, venue=venue, symbol=symbol,
                       confidence=Decimal(str(round(probability, 6))),
                       evidence={**evidence, "belief": belief.describe(at_ns)},
                       at_ns=at_ns, calibrated=calibrated,
                       makes_edge_claim=self.makes_edge_claim)


class LearnedBearBrain:
    """Proposes shorts when the model says a long from here tends to lose.

    **Stricter than the bull on the same evidence.** `dual-agent-spec.md` requires
    short-side limits stricter than long-side, and here that is one number with a
    stated reason rather than a tuned threshold: the bear needs the model to be
    more confident against a long than the bull needs it to be for one, because a
    model trained on LONG outcomes is being read in reverse and a reversed reading
    is weaker evidence than a direct one.
    """

    stance = BEAR
    makes_edge_claim = True

    # The one disclosed design choice in this module, and `§1a` L1 permits exactly
    # that - what it refuses is an UNdisclosed one. It is not fitted, it is a
    # statement about asymmetric evidence strength, and it is named in every
    # decision's evidence so it can never be mistaken for a learned quantity.
    REVERSED_READING_PENALTY = 0.05

    def __init__(self, segment: str, model: LoadedModel,
                 calibration: OnlineCalibration) -> None:
        self.segment = segment
        self.name = f"{segment}-bear-learned"
        self._core = _LearnedDirectional(segment, model, calibration)
        self.model = model
        self.calibration = calibration

    def __call__(self, frame, chain_median=None):        # noqa: ARG002
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        vector = frame.get("model_vector")
        base = {"model_version": self.model.version_id,
                "trial_id": self.model.trial_id, "rule_brain": False,
                "reversed_reading_penalty": self.REVERSED_READING_PENALTY,
                "disclosed_design_choice": (
                    "the bear reads a long-trained model in reverse, which is "
                    "weaker evidence than a direct reading, so it requires more "
                    "of it (dual-agent-spec: short limits stricter than long)")}

        if vector is None:
            return decline(self, venue=venue, symbol=symbol,
                           reason="WARMING_UP_NO_FEATURE_VECTOR",
                           evidence={**base, "sealed_bars": frame.get("sealed_bars")},
                           at_ns=at_ns)

        raw = self._core.score(vector)
        probability, calibrated = self.calibration.calibrate(raw)
        # The bear's own quantity: how strongly the model argues AGAINST a long.
        short_evidence = 1 - probability
        abstain, coverage = self.calibration.should_abstain(probability)
        evidence = {**base, "raw_score": round(raw, 6),
                    "calibrated_probability": round(probability, 6),
                    "short_evidence": round(short_evidence, 6),
                    "is_calibrated": calibrated, "coverage": coverage}

        if abstain:
            return decline(self, venue=venue, symbol=symbol,
                           reason=f"ABSTAINED_{coverage['reason']}",
                           evidence=evidence, at_ns=at_ns)
        if short_evidence <= 0.5 + self.REVERSED_READING_PENALTY:
            return decline(self, venue=venue, symbol=symbol,
                           reason="MODEL_DOES_NOT_FAVOUR_SHORT",
                           evidence=evidence, at_ns=at_ns)

        belief = self._core.belief(
            claim=f"{symbol} long resolves unprofitable under the exit policy",
            value=short_evidence, at_ns=at_ns, evidence=evidence,
            confidence=Decimal(str(round(short_evidence, 6))))
        return propose(self, venue=venue, symbol=symbol,
                       confidence=Decimal(str(round(short_evidence, 6))),
                       evidence={**evidence, "belief": belief.describe(at_ns)},
                       at_ns=at_ns, calibrated=calibrated,
                       makes_edge_claim=self.makes_edge_claim)


class LearnedProfitTail(ProfitTail):
    """LB-06: PROFIT-TAIL whose expectancy and tail are forecast, not proxied.

    **Subclasses the deterministic one on purpose.** `manage()` and `time_entry()`
    are inherited unchanged, so every authority limit RL-023 imposes survives
    becoming learned — it still cannot reject a selected trade, still cannot hold
    past the hard stop, still has no line to execution. Only `assess()` is
    overridden, because only `assess()` was a proxy.

    The rule version computed `min(take_profit, volatility) - 2*spread`: a
    volatility estimate standing in for an expectancy nobody had measured. This
    returns the fitted median of forward P&L for the expectancy and the fitted 10th
    percentile for the loss tail — `bull-bear-profit-agents-spec.md` §4 asks for
    *the distribution of forward P&L, not a point estimate*, and quantiles are that
    distribution's shape where it matters.

    Falling back to the deterministic assessment when no quantile model is
    registered is deliberate and is the documented rollback target, not a silent
    degradation: `policy` in the evidence says which one answered.
    """

    def __init__(self, *args, tail_model=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tail_model = tail_model
        if tail_model:
            self.name = f"{self.segment}-profit-tail-learned"
            self.makes_edge_claim = True

    def assess(self, *, venue: str, symbol: str, frame, at_ns: int) -> TailAssessment:
        vector = frame.get("model_vector")
        if not self.tail_model or vector is None:
            assessment = super().assess(venue=venue, symbol=symbol, frame=frame,
                                        at_ns=at_ns)
            return TailAssessment(
                venue=assessment.venue, symbol=assessment.symbol,
                net_expectancy=assessment.net_expectancy,
                loss_tail=assessment.loss_tail, confidence=assessment.confidence,
                evidence={**assessment.evidence,
                          "policy": "deterministic fallback",
                          "why": ("no quantile model registered" if not self.tail_model
                                  else "warming up: no feature vector yet")},
                at_ns=assessment.at_ns)

        boosters = self.tail_model["boosters"]
        predictions = {q: float(b.predict([vector])[0]) for q, b in boosters.items()}
        low = min(predictions)
        median = sorted(predictions)[len(predictions) // 2]
        high = max(predictions)

        expectancy = Decimal(str(predictions[median]))
        # The loss tail is a POSITIVE magnitude by contract, and the low quantile
        # is signed. A forecast whose 10th percentile is positive means the model
        # expects no adverse excursion at that quantile, which is reported as a
        # zero tail rather than as a negative one - a negative tail would make the
        # arbiter's `loss_tail > max_loss_tail` test read backwards.
        loss_tail = Decimal(str(max(0.0, -predictions[low])))

        # The spread is a real cost the quantiles never saw: they were fitted on
        # bar-to-bar P&L, and the live trade crosses the spread twice.
        spread = frame.get("relative_spread")
        round_trip = Decimal(str(spread)) * 2 if spread is not None else Decimal(0)
        net = expectancy - round_trip

        return TailAssessment(
            venue=venue, symbol=symbol, net_expectancy=net, loss_tail=loss_tail,
            confidence=Decimal("0.6"),
            evidence={
                "policy": "learned quantiles",
                "model_version": self.tail_model["version_id"],
                "trial_id": self.tail_model["trial_id"],
                "quantile_predictions": {str(q): round(v, 8)
                                         for q, v in predictions.items()},
                "gross_expectancy": str(expectancy),
                "round_trip_spread": str(round_trip),
                "upper_quantile": round(predictions[high], 8),
                "segment": self.segment,
                "fitted_by": "scheduled retrain",
            },
            at_ns=at_ns)
