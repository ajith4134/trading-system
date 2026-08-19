"""LB-02: fit the segment model, count the trial, register the artefact with its loss.

## What §1a L1 actually demands, and how this satisfies it

> Every numeric parameter in the decision path traces to a `fit()`/`train()` call with a
> **logged loss and dataset reference**, or to an explicitly disclosed design choice.

So a trained model is not enough. What has to exist afterwards is an auditable chain:

    trial id  →  out-of-fold score  →  model version  →  the bytes a brain loaded

`models.model_registry` already enforces the last link — it refuses `register()` with
`trial_id=None`, because *"a model whose trial was never counted is a model whose N
nobody knows, and every promotion gate divides its protection by N"*. This module
supplies the first three.

## Two fits, and why the scoring one is not the one that gets deployed

`models.gradient_boosted_trees.train_gbt` is a **scoring harness**: it runs combinatorial
purged cross-validation, scores out of fold against the majority class, and returns a
`GbtResult`. It deliberately returns no booster — an honest out-of-fold score comes from
models that never saw the rows they scored, and there are `n_folds` of those.

Deployment needs one model that saw everything. So:

1. **`train_gbt` decides whether there is anything here at all** — counted as a trial,
   purged, scored against the base rate. Its `beats_majority_class` and
   `reproduced_majority_class` flags are the verdict.
2. **A final fit on all rows produces the artefact**, registered against that same trial
   id, carrying the out-of-fold score in its metrics.

The out-of-fold score therefore describes the *procedure*, not the deployed booster
itself, and that is the honest reading. A score computed on the deployed model's own
training rows would be an in-sample number wearing an out-of-sample name.

## The majority-class trap is reported, never smoothed

`GbtResult.reproduced_majority_class` catches the classifier that predicted the same
label on every row. On a 55/45 split that scores 55% and learned nothing. This module
**refuses to register such a model** rather than registering it with a caveat: a
registered model gets loaded, and a caveat in a metrics dict is not read at 3am.

## Feature order is part of the artefact

A booster is splits over column *indices*. Reorder `FEATURE_NAMES` and every value goes
to the wrong split, silently, with no error and plausible-looking predictions.
`FEATURE_NAMES` is written into the model's metrics at registration and checked at load
by `learn.learned_brains`, so a mismatch is a refusal rather than a wrong answer.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learn.training_set import FEATURE_NAMES
from features.sample_uniqueness import average_uniqueness_by_group
from models.gradient_boosted_trees import train_gbt
from models.model_registry import ModelRegistry
from validation.champion_promotion import (
    ChampionVerdict, evaluate_champion, evaluate_tail_champion,
)
from validation.trial_registry import TrialRegistry, TrialSpec

DEFAULT_REGISTRY_ROOT = Path.home() / "capture" / "models"
DEFAULT_TRIALS_ROOT = Path.home() / "capture" / "trials"

# The champion alias each segment's brains load. One alias per segment because
# RL-019 makes each segment its own bot - a shared champion would be one model
# wearing four names.
CHAMPION_ALIAS = "{segment}-direction-champion"
TAIL_ALIAS = "{segment}-profit-tail-champion"

FAMILY_DIRECTION = "segment-direction"
FAMILY_TAIL = "segment-profit-tail"


class NothingLearned(RuntimeError):
    """The fit did not beat its base rate, or reproduced the majority class.

    Raised rather than registered. §1a's master test asks whether a capability
    changes what the system does when it is wrong; a model that predicts the
    majority class on every row changes nothing and would occupy the champion
    alias, which is worse than no model because the board would read LEARNED.
    """


@dataclass(frozen=True)
class TrainedModel:
    """What a successful training run produced, and what it is entitled to claim."""

    version_id: str
    trial_id: int
    segment: str
    family: str
    accuracy: float
    base_rate: float
    n_out_of_fold: int
    rows: int
    feature_names: tuple
    describe_dataset: dict

    def as_fit_reference(self) -> dict:
        """The `Provenance.fit_reference` a belief from this model carries."""
        return {
            "model_version": self.version_id,
            "trial_id": self.trial_id,
            "family": self.family,
            "out_of_fold_accuracy": self.accuracy,
            "base_rate": self.base_rate,
            "n_out_of_fold": self.n_out_of_fold,
            "training_rows": self.rows,
        }


def _promote_or_keep(models, alias: str, version_id: str, metrics: dict,
                    *, evaluate, segment: str) -> "ChampionVerdict":
    """Assign the alias only if the challenger beats what is already serving.

    **CG-02 / RL-044.** This used to be a bare `assign_alias`, so the champion was
    whatever trained LAST rather than what was best, and a retrain producing a
    worse model silently replaced a better one. Measured 2026-08-19: perp's edge
    fell from 0.01695 to 0.01227 in exactly that way and the bot stopped trading.

    A refused promotion is NOT an error and does not raise. The trained version
    stays in the registry with its metrics, so it can be compared again later or
    promoted by hand; only the alias is withheld. Raising would lose the model and
    make a routine "no improvement this cycle" look like a failed run.
    """
    incumbent_id = models.aliases().get(alias)
    incumbent_metrics = None
    if incumbent_id:
        try:
            incumbent_metrics = dict(models.version(incumbent_id).metrics)
        except (KeyError, LookupError, ValueError):
            # An alias pointing at a version the registry cannot produce is its
            # own problem, and it must not silently read as "no incumbent" - the
            # verdict records it as replacing an unmeasured model.
            incumbent_metrics = None

    verdict = evaluate(challenger_metrics=metrics,
                       incumbent_metrics=incumbent_metrics,
                       incumbent_version_id=incumbent_id)
    if verdict.promote:
        models.assign_alias(alias, version_id, reason=verdict.describe())
    else:
        print(f"champion NOT promoted for {segment} ({alias}): "
              f"{verdict.describe()}", flush=True)
    return verdict


def _final_fit(features: np.ndarray, labels: np.ndarray, weights: np.ndarray,
               params: dict, boost_rounds: int, seed: int):
    """One booster over every row, for deployment. Scored by the CPCV run above."""
    import lightgbm as lgb

    from models.gradient_boosted_trees import DETERMINISM_PARAMS
    dataset = lgb.Dataset(features, label=labels, weight=weights, free_raw_data=False)
    return lgb.train({**params, **DETERMINISM_PARAMS, "seed": seed},
                     dataset, num_boost_round=boost_rounds)


def train_direction_model(rows, *, segment: str, venues=None,
                          registry_root: Path = DEFAULT_REGISTRY_ROOT,
                          trials_root: Path = DEFAULT_TRIALS_ROOT,
                          params: dict | None = None,
                          boost_rounds: int = 200,
                          seed: int = 0) -> TrainedModel:
    """Fit, score, register. The whole L1 chain in one call.

    Returns a `TrainedModel` whose `as_fit_reference()` is what every belief the
    brain later emits will carry, so a decision in the journal can be traced to the
    trial that counted the look at the data which produced it.
    """
    if len(rows) < 500:
        raise NothingLearned(
            f"{segment}: {len(rows)} training row(s). Refused - a purged CPCV over "
            f"six groups needs enough rows that a fold is not a handful, and a "
            f"model fitted on fewer would carry an accuracy nobody should read")

    features = np.asarray(rows.features, dtype=float)
    labels = np.asarray(rows.labels, dtype=int)
    # **Uniqueness WITHIN each symbol, never across the pooled cross-section.**
    # Every span's index is local to its own symbol, so a single pooled timeline
    # treats symbol A's bar 500 and symbol B's bar 500 as the same bar and divides
    # uniqueness by the symbol count. Measured 2026-08-19: perp reported 0.00223
    # over 573 symbols, whose implied per-symbol uniqueness of 1.28 is above the
    # 1.0 ceiling the quantity has - the arithmetic proof that pooling did it.
    weights = np.asarray(
        average_uniqueness_by_group(rows.spans, rows.symbols), dtype=float)

    trials = TrialRegistry(Path(trials_root))
    params = params or {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    # Counted BEFORE the first tree grows - `train_gbt` runs inside
    # `registry.evaluate`, so an exception still leaves a counted, abandoned trial.
    # §1a L7: the count keys on candidates EVALUATED, never on candidates retained.
    result = train_gbt(
        features, labels, rows.spans, span_groups=rows.symbols,
        family=FAMILY_DIRECTION, registry=trials,
        trial_name=f"{segment}-direction-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        params=params, boost_rounds=boost_rounds, seed=seed)

    if result.reproduced_majority_class:
        raise NothingLearned(
            f"{segment}: the fit predicted the same label on every row and scored "
            f"{result.accuracy:.4f} against a {result.base_rate:.4f} base rate. "
            f"That is the majority class, not a model. {result.describe()}")
    if not result.beats_majority_class:
        raise NothingLearned(
            f"{segment}: {result.describe()}. Refused rather than registered - a "
            f"champion alias pointing at this would make the board read LEARNED "
            f"for a model that adds nothing over guessing the base rate")

    # **Split conformal: a held-out calibration set, so the brain is not born mute.**
    #
    # `learn.online_calibration` abstains on everything until it has a track record
    # of its own non-conformity - which is correct, and on its own is a deadlock: no
    # trades means no outcomes means no track record means no trades. Verified
    # 2026-08-18: a freshly loaded brain declined every symbol with
    # NO_CONFORMAL_TRACK_RECORD.
    #
    # Split conformal is the standard answer and it is what the out-of-fold rows
    # already are: fit on the earlier portion, predict the later one, and the
    # resulting scores are honest non-conformity observations on data the model did
    # not see. The brain seeds from them and updates from live outcomes afterwards,
    # so the guarantee starts real rather than starting absent.
    #
    # Split by TIME, never at random. A random split would put a row's neighbours
    # on both sides and leak the label horizon across the boundary - the same
    # one-sided-purge defect found in donor code on 2026-08-08.
    split = int(len(features) * 0.8)
    calibration_scores: list = []
    if split > 100 and len(features) - split > 50:
        warm = _final_fit(features[:split], labels[:split], weights[:split],
                          params, boost_rounds, seed)
        held_out = warm.predict(features[split:])
        for score, truth in zip(held_out, labels[split:]):
            confidence = max(float(score), 1 - float(score))
            was_right = (float(score) >= 0.5) == bool(truth)
            calibration_scores.append(round(1 - confidence if was_right else 1.0, 6))

    booster = _final_fit(features, labels, weights, params, boost_rounds, seed)
    # JSON, never pickle. `Booster.model_to_string()` is text and every other
    # field is a primitive, so the artefact needs no code execution to load - and
    # a model artefact IS an executable payload if you let it be one. The registry
    # verifies the sha256 at load on top of that.
    artifact = json.dumps({
        "booster": booster.model_to_string(),
        # The contract that stops a reordered feature list becoming silent
        # nonsense. Checked at load, refused on mismatch.
        "feature_names": list(FEATURE_NAMES),
        "segment": segment,
        "trained_at_ns": time.time_ns(),
    }).encode("utf-8")

    models = ModelRegistry(Path(registry_root))
    trial_id = _last_trial_id(trials)
    version = models.register(
        artifact, trial_id=trial_id, family=FAMILY_DIRECTION,
        name=f"{segment}-direction",
        metrics={**result.as_registry_result(),
                 "feature_names": list(FEATURE_NAMES),
                 "training_rows": len(rows),
                 # The conformal calibration set, so a freshly loaded brain has a
                 # coverage guarantee instead of a deadlock. Capped: this rides in
                 # the model metadata and is not a place to store a dataset.
                 "conformal_scores": calibration_scores[-500:],
                 "conformal_split": "time-ordered 80/20 holdout",
                 # **The venues this model was fitted on, recorded so a segment
                 # cannot load a champion trained on somebody else's data.**
                 # Measured 2026-08-18: spot's own-venue fit found nothing
                 # (p=0.4645) while an earlier POOLED model sat in its champion
                 # alias, so the spot bot was running a model fitted partly on
                 # binance futures bars. RL-019 makes that wrong, and existence of
                 # a champion is not evidence of its provenance - this field is.
                 "fitted_on_venues": sorted(venues) if venues else "POOLED",
                 "dataset": rows.describe()},
        notes=(f"pooled cross-sectional direction model for the {segment} bot; "
               f"labels are the deterministic exit policy's realised sign"))
    _promote_or_keep(
        models, CHAMPION_ALIAS.format(segment=segment), version.version_id,
        metrics={**result.as_registry_result(),
                 "accuracy": result.accuracy, "base_rate": result.base_rate,
                 "beats_majority_class": result.beats_majority_class},
        evaluate=evaluate_champion, segment=segment)

    return TrainedModel(
        version_id=version.version_id, trial_id=trial_id, segment=segment,
        family=FAMILY_DIRECTION, accuracy=result.accuracy,
        base_rate=result.base_rate, n_out_of_fold=result.n_out_of_fold,
        rows=len(rows), feature_names=FEATURE_NAMES,
        describe_dataset=rows.describe())


def train_profit_tail_model(rows, *, segment: str,
                            registry_root: Path = DEFAULT_REGISTRY_ROOT,
                            trials_root: Path = DEFAULT_TRIALS_ROOT,
                            quantiles=(0.1, 0.5, 0.9),
                            boost_rounds: int = 200,
                            seed: int = 0) -> TrainedModel:
    """LB-06: fit quantiles of forward P&L, so PROFIT-TAIL forecasts a distribution.

    `bull-bear-profit-agents-spec.md` §4 asks for `forecast_profit_tail` — *the
    distribution of forward P&L, not a point estimate*. Three quantile regressions
    give that shape: the median is what the trade is expected to make, the 10th
    percentile is the loss tail the arbiter consumes, and the 90th is what a
    ratchet is trying to keep.

    This replaces the rule version's `min(take_profit, volatility) - 2*spread`,
    which was a volatility proxy standing in for an expectancy nobody had measured.
    """
    import lightgbm as lgb

    from models.gradient_boosted_trees import DETERMINISM_PARAMS

    if len(rows) < 500:
        raise NothingLearned(f"{segment}: {len(rows)} row(s) is too few for quantiles")

    features = np.asarray(rows.features, dtype=float)
    outcomes = np.asarray(rows.outcomes, dtype=float)
    # **Uniqueness WITHIN each symbol, never across the pooled cross-section.**
    # Every span's index is local to its own symbol, so a single pooled timeline
    # treats symbol A's bar 500 and symbol B's bar 500 as the same bar and divides
    # uniqueness by the symbol count. Measured 2026-08-19: perp reported 0.00223
    # over 573 symbols, whose implied per-symbol uniqueness of 1.28 is above the
    # 1.0 ceiling the quantity has - the arithmetic proof that pooling did it.
    weights = np.asarray(
        average_uniqueness_by_group(rows.spans, rows.symbols), dtype=float)

    trials = TrialRegistry(Path(trials_root))
    spec = TrialSpec(
        family=FAMILY_TAIL,
        name=f"{segment}-profit-tail-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        params={"quantiles": list(quantiles), "boost_rounds": boost_rounds,
                "rows": len(rows), "seed": seed})

    def _fit(_spec) -> dict:
        boosters, scores = {}, {}
        for q in quantiles:
            dataset = lgb.Dataset(features, label=outcomes, weight=weights,
                                  free_raw_data=False)
            booster = lgb.train(
                {"objective": "quantile", "alpha": q, "learning_rate": 0.05,
                 "num_leaves": 31, "min_data_in_leaf": 50, "verbose": -1,
                 **DETERMINISM_PARAMS, "seed": seed},
                dataset, num_boost_round=boost_rounds)
            boosters[q] = booster.model_to_string()
            predicted = booster.predict(features)
            # Pinball loss - the loss the quantile objective actually minimises,
            # logged because §1a L1 asks for the loss and not only the model.
            diff = outcomes - predicted
            scores[str(q)] = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
        _fit.boosters = boosters
        return {"sharpe": None, "pinball_loss": scores, "rows": len(rows)}

    result = trials.evaluate(spec, _fit)
    trial_id = _last_trial_id(trials)

    artifact = json.dumps({
        "quantile_boosters": {str(q): s for q, s in _fit.boosters.items()},
        "quantiles": list(quantiles),
        "feature_names": list(FEATURE_NAMES),
        "segment": segment,
        "trained_at_ns": time.time_ns(),
    }).encode("utf-8")
    models = ModelRegistry(Path(registry_root))
    version = models.register(
        artifact, trial_id=trial_id, family=FAMILY_TAIL,
        name=f"{segment}-profit-tail",
        metrics={**result, "feature_names": list(FEATURE_NAMES),
                 "dataset": rows.describe()},
        notes=("quantile forecast of forward P&L per unit; the 0.1 quantile is the "
               "loss tail the arbiter consumes and the 0.5 is the expectancy"))
    _promote_or_keep(
        models, TAIL_ALIAS.format(segment=segment), version.version_id,
        metrics=dict(result), evaluate=evaluate_tail_champion, segment=segment)

    return TrainedModel(
        version_id=version.version_id, trial_id=trial_id, segment=segment,
        family=FAMILY_TAIL, accuracy=float("nan"), base_rate=float("nan"),
        n_out_of_fold=0, rows=len(rows), feature_names=FEATURE_NAMES,
        describe_dataset=rows.describe())


def _last_trial_id(trials: TrialRegistry) -> int:
    """The id of the trial just recorded.

    Read back from the ledger rather than returned by `train_gbt`, which reports a
    score and not an id. Reading it back is also the stronger check: it confirms the
    trial reached disk before a model is registered against it.
    """
    entries = trials.trials()
    if not entries:
        raise NothingLearned(
            "no trial reached the ledger; refusing to register a model whose N "
            "nobody knows (model_registry refuses trial_id=None for the same reason)")
    return int(entries[-1]["trial_id"])


def retrain_segment(segment: str, *, n_hours: int = 14,
                    registry_root: Path = DEFAULT_REGISTRY_ROOT,
                    trials_root: Path = DEFAULT_TRIALS_ROOT) -> dict:
    """Read, label, fit, register — one segment, nobody invoking anything.

    Returns a report rather than raising on `NothingLearned`. A refit that found
    no edge is a NORMAL outcome and must not stop the supervisor: the champion
    alias keeps pointing at whatever last passed, which is the correct behaviour
    when today's fit is worse than yesterday's.
    """
    from learn.training_set import build_rows, default_policy, read_recent_bars

    from learn.training_set import SEGMENT_VENUES

    venues = SEGMENT_VENUES.get(segment)
    if venues is None:
        # A NORMAL outcome, reported rather than raised. dated and options reason
        # about basis, time to expiry and implied volatility, none of which is in a
        # bar, so there is nothing here for them to be fitted on yet. Raising made
        # the supervisor log an error every cycle for a state that is correct.
        return {"segment": segment, "skipped": "NO_DATASET_FOR_THIS_SEGMENT",
                "why": ("its brains reason about inputs a bar does not carry; a "
                        "bar-fitted direction model would answer a question they "
                        "do not ask. Rule brains stand until it has its own dataset"),
                "keeps": "rule brains"}

    started = time.time()
    bars, read_report = read_recent_bars(n_hours=n_hours, venues=venues)
    rows = build_rows(bars, policy=default_policy())
    report = {"segment": segment, "read": read_report,
              "dataset": rows.describe(), "seconds": round(time.time() - started, 1)}

    try:
        direction = train_direction_model(rows, segment=segment, venues=venues,
                                          registry_root=registry_root,
                                          trials_root=trials_root)
        report["direction"] = direction.as_fit_reference()
    except NothingLearned as exc:
        # Counted as a trial regardless - `train_gbt` registers before it fits, so
        # a refused model still increments N and still deflates every future
        # Sharpe. §1a L7: the count keys on candidates EVALUATED.
        report["direction"] = {"refused": str(exc)}
        return report

    try:
        tail = train_profit_tail_model(rows, segment=segment,
                                       registry_root=registry_root,
                                       trials_root=trials_root)
        report["profit_tail"] = tail.as_fit_reference()
    except NothingLearned as exc:
        report["profit_tail"] = {"refused": str(exc)}

    # LB-07: the §1a axis probes, run against the model that was just registered.
    #
    # Run HERE rather than by hand, and the difference is not convenience. §1a.6
    # says no module ships without an axis verdict and the wall carries it; a
    # verdict produced by somebody typing a command is a verdict that stops being
    # produced the week nobody types it. This is also what makes `learn.axis_probes`
    # a reachable module rather than one whose depth verdict rests on a shell
    # session - the repo's own reachability audit refuses that, correctly.
    report["axis_probes"] = _probe_registered_model(segment, rows, registry_root)
    return report


def _probe_registered_model(segment: str, rows, registry_root: Path) -> dict:
    """Run the §1a probes against the champion, on rows it was fitted to serve."""
    from learn.axis_probes import run_all
    from learn.learned_brains import load_champion
    from learn.online_calibration import OnlineCalibration

    try:
        model = load_champion(segment, registry_root=registry_root)
    except Exception as exc:                                   # noqa: BLE001
        return {"verdict": "NOT MEASURED", "why": f"no champion: {exc}"}

    calibration = OnlineCalibration(segment=segment)
    calibration.scores = list(model.metrics.get("conformal_scores") or [])
    # The most recent rows: the probes should ask about the data the model will
    # actually meet, not the oldest slice of its training set.
    sample = min(2000, len(rows.features))
    return run_all(model=model, calibration=calibration,
                   vectors=rows.features[-sample:], labels=rows.labels[-sample:],
                   dataset_description=rows.describe())


def main(argv=None) -> int:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segment", required=True)
    parser.add_argument("--hours", type=int, default=14)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--trials-root", type=Path, default=DEFAULT_TRIALS_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    report = retrain_segment(args.segment, n_hours=args.hours,
                             registry_root=args.registry_root,
                             trials_root=args.trials_root)
    text = _json.dumps(report, indent=2, default=str)
    print(text, flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
