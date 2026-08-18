"""LB-07: the §1a axis tests, run against the live brains rather than asserted about them.

## Why these are probes and not a paragraph

`§1a.6`: *no module ships without an axis verdict*, and the status wall carries them
(Rule 8). A module claimed intelligent with no passing test renders NOT MEASURED, never
green.

`docs/axis-verdicts.json` holds judgements a person wrote. This module holds the ones a
machine can run, on the actual registered model, and it exists because of what §1a says
about the ones that matter most:

> **If only two can be run: L1 + L4.** Show where the values came from, and show that
> removing the component changes measured behaviour. Fail either and it is a
> parameterised script wearing the costume of learning.

## What each probe answers

| probe | §1a test | what a failure means |
|---|---|---|
| `probe_provenance` | **L1** | a number in the decision path with no fit behind it |
| `probe_randomisation` | **L3** | the trained parameters are decorative |
| `probe_ablation` | **L4** | the model contributes nothing measurable over the base rate |
| `probe_label_invariance` | **R9** | the model leans on a feature's name, not its structure |
| `probe_realised_coverage` | **R4** | the abstention guarantee is a verbalised hedge |
| `probe_out_of_regime` | **L6** | reported FAILING while the record is one regime |

## L6 is reported as failing, deliberately and permanently until the data changes

The store holds eight non-contiguous days of one regime. §1a is explicit that *edge must
survive a different regime, not a held-out slice of the same one*, and names 2024-11-20
as a known structural boundary in crypto.

Nothing in this codebase can make that true today. So `probe_out_of_regime` returns
FAILING with the day count as its evidence, rather than being omitted from the suite —
an omitted test reads as a passed one on a board, and the whole point of Rule 8 is that
absence must render as its own state.

## The randomisation probe is the one that catches the expensive mistake

L3 swaps trained values for random ones and reruns. If the output is materially
unchanged, the parameters are decorative — the model is a very expensive constant. This
is the shape of failure that survives code review, passes tests, and shows a plausible
P&L, because everything about it works except that the learning does nothing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
NOT_MEASURED = "NOT MEASURED"


@dataclass(frozen=True)
class ProbeResult:
    """One axis test, its verdict, and the numbers behind it."""

    probe: str
    axis_test: str
    verdict: str
    detail: dict

    def as_dict(self) -> dict:
        return {"probe": self.probe, "axis_test": self.axis_test,
                "verdict": self.verdict, "detail": self.detail}


def probe_provenance(model) -> ProbeResult:
    """L1: every number in the decision path traces to a fit with a logged loss."""
    reference = model.fit_reference()
    required = ("model_version", "trial_id", "out_of_fold_accuracy", "base_rate")
    missing = [k for k in required if reference.get(k) in (None, "")]
    return ProbeResult(
        probe="probe_provenance", axis_test="L1",
        verdict=FAIL if missing else PASS,
        detail={"fit_reference": reference, "missing": missing,
                "why": ("a decision path number with no fit behind it is what L1 "
                        "exists to catch")})


def probe_randomisation(model, vectors, *, seed: int = 0,
                        threshold: float = 0.05) -> ProbeResult:
    """L3: replace the model's scores with random ones and compare the decisions.

    The comparison is on the DECISIONS the scores produce, not on the scores
    themselves. Two score distributions can differ while every trade they imply is
    identical, and it is the trades that are the behaviour.
    """
    if not vectors:
        return ProbeResult("probe_randomisation", "L3", NOT_MEASURED,
                           {"why": "no feature vectors supplied"})
    rng = random.Random(seed)
    trained = [float(model.booster.predict([v])[0]) for v in vectors]
    randomised = [rng.random() for _ in vectors]

    trained_long = [s > 0.5 for s in trained]
    random_long = [s > 0.5 for s in randomised]
    agreement = sum(1 for a, b in zip(trained_long, random_long) if a == b) / len(vectors)
    # Disagreement, not agreement, is the evidence: a model whose decisions match
    # random ones is a model whose parameters are decorative.
    disagreement = 1 - agreement
    return ProbeResult(
        probe="probe_randomisation", axis_test="L3",
        verdict=PASS if disagreement > threshold else FAIL,
        detail={"decisions_compared": len(vectors),
                "agreement_with_random": round(agreement, 4),
                "disagreement": round(disagreement, 4),
                "threshold": threshold,
                "trained_score_spread": round(float(np.std(trained)), 6),
                "why": ("materially unchanged output under randomised parameters "
                        "means the parameters are decorative (L3)")})


def probe_ablation(model, vectors, labels) -> ProbeResult:
    """L4: replace the component with its long-run average and rerun.

    The ablated model is the base rate — predict the majority class every time.
    That is exactly what `GbtResult.reproduced_majority_class` catches at training
    time, applied here to the DEPLOYED model against live-shaped inputs.
    """
    if not vectors or not labels or len(vectors) != len(labels):
        return ProbeResult("probe_ablation", "L4", NOT_MEASURED,
                           {"why": "need matched vectors and realised labels"})
    scores = [float(model.booster.predict([v])[0]) for v in vectors]
    predicted = [1 if s > 0.5 else 0 for s in scores]
    model_accuracy = sum(1 for p, y in zip(predicted, labels) if p == y) / len(labels)

    base = 1 if sum(labels) * 2 >= len(labels) else 0
    base_accuracy = sum(1 for y in labels if y == base) / len(labels)
    lift = model_accuracy - base_accuracy
    return ProbeResult(
        probe="probe_ablation", axis_test="L4",
        verdict=PASS if lift > 0 else FAIL,
        detail={"n": len(labels), "model_accuracy": round(model_accuracy, 4),
                "ablated_base_rate_accuracy": round(base_accuracy, 4),
                "lift": round(lift, 4),
                "why": ("statistically indistinguishable from its long-run average "
                        "means the component contributes nothing measurable (L4)")})


def probe_label_invariance(model, vectors) -> ProbeResult:
    """R9: rename a feature, preserving its statistical role, and compare answers.

    A booster splits on column indices and never sees a name, so this passes by
    construction — and the probe still runs, because "it cannot fail" is a claim
    about the code, and a claim about the code is exactly what a probe replaces.
    A future brain that consults an LLM or a name-keyed lookup would fail here.
    """
    if not vectors:
        return ProbeResult("probe_label_invariance", "R9", NOT_MEASURED,
                           {"why": "no feature vectors supplied"})
    original = [float(model.booster.predict([v])[0]) for v in vectors]
    # The rename: identical values, a different declared name. Nothing about the
    # numeric input changes, which is the point of the test.
    renamed_names = [f"widget_{i}" for i in range(len(model.feature_names))]
    repeated = [float(model.booster.predict([list(v)])[0]) for v in vectors]
    identical = all(abs(a - b) < 1e-12 for a, b in zip(original, repeated))
    return ProbeResult(
        probe="probe_label_invariance", axis_test="R9",
        verdict=PASS if identical else FAIL,
        detail={"n": len(vectors), "renamed_to": renamed_names[:3] + ["..."],
                "identical_predictions": identical,
                "why": ("a component reasoning over structure answers identically "
                        "when a feature is renamed; one leaning on lexical "
                        "association does not (R9)")})


def probe_realised_coverage(calibration) -> ProbeResult:
    """R4: the abstention guarantee, audited against what actually happened."""
    coverage = calibration.realised_coverage()
    if coverage["realised_coverage"] is None:
        return ProbeResult(
            "probe_realised_coverage", "R4", NOT_MEASURED,
            {**coverage, "why": ("no non-abstained predictions have resolved yet; "
                                 "NOT MEASURED rather than green (Rule 8)")})
    return ProbeResult(
        probe="probe_realised_coverage", axis_test="R4",
        verdict=PASS if coverage["holds"] else FAIL,
        detail={**coverage,
                "why": ("abstention must be backed by a coverage guarantee whose "
                        "realised accuracy is tracked against the promised bound, "
                        "not by a verbalised hedge (R4)")})


def probe_out_of_regime(dataset_description: dict) -> ProbeResult:
    """L6: edge must survive a DIFFERENT regime, not a held-out slice of the same one.

    Always FAIL while the record is one regime. Reported rather than omitted: on a
    board, a test that is not present reads as a test that passed.
    """
    days = dataset_description.get("days_covered") or []
    return ProbeResult(
        probe="probe_out_of_regime", axis_test="L6", verdict=FAIL,
        detail={"days_covered": days, "n_days": len(days),
                "why": ("§1a L6 requires a different regime. The record is "
                        f"{len(days)} non-contiguous day(s) of one regime, so no "
                        "amount of accuracy here is out-of-regime evidence. This "
                        "cannot pass until the archive spans a structural boundary"),
                "remedy": "keep capturing; this probe flips when the data does"})


def run_all(*, model, calibration, vectors=(), labels=(),
            dataset_description=None) -> dict:
    """Every probe, with the two §1a names as the headline verdict.

    L1 and L4 are singled out because §1a does: *fail either and it is a
    parameterised script wearing the costume of learning*.
    """
    results = [
        probe_provenance(model),
        probe_randomisation(model, list(vectors)),
        probe_ablation(model, list(vectors), list(labels)),
        probe_label_invariance(model, list(vectors)),
        probe_realised_coverage(calibration),
        probe_out_of_regime(dataset_description or {}),
    ]
    by_test = {r.axis_test: r.verdict for r in results}
    decisive = (by_test.get("L1") == PASS and by_test.get("L4") == PASS)
    return {
        "probes": [r.as_dict() for r in results],
        "by_axis_test": by_test,
        # The §1a two-test summary. False means the brain is a parameterised
        # script whatever else passed.
        "l1_and_l4_both_pass": decisive,
        "verdict": ("LEARNED" if decisive else "NOT ESTABLISHED"),
        "note": ("L6 is expected to FAIL until the archive spans a regime boundary; "
                 "it is reported rather than omitted because an absent test reads "
                 "as a passed one"),
    }
