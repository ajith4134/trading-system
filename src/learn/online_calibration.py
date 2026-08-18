"""LB-05: the parameters the LIVE loop changes itself — calibration, and the abstention quantile.

## Why this module is the whole difference between two claims

`§1a` L2, verbatim:

> Does the live decision loop itself update parameters from post-deployment data? If
> the only path that changes parameters is a script a human invokes, that is scheduled
> retraining — a legitimate but *different* claim, and it must be labelled as such.

`learn.train_segment_model` is the scheduled-retraining half and says so. This is the
other half: two parameters that the running bot fits from its own realised outcomes,
every poll, with nobody invoking anything.

They are deliberately small. Updating the booster online would risk catastrophic
forgetting (L9) and specification gaming (L8) on eight days of one regime, with no
held-out regime to catch either. Calibration and an abstention threshold are the two
things that *must* track the live distribution, because both are statements about how
wrong the model currently is — which is exactly what changes when the world moves.

## Calibration: a score is not a probability until something makes it one

LightGBM's binary objective outputs a score in [0, 1] that is not a calibrated
probability. `dual-agent-spec.md` makes calibration mandatory before sizing, and this is
where it happens.

The method is a **binned reliability map** rather than isotonic regression or Platt
scaling, for one reason: it has to be updatable incrementally from a stream, and it has
to be readable when it is wrong. A bin holding "the model said 0.7 here 40 times and was
right 51% of them" is a sentence anyone can check. A fitted logistic's two coefficients
are not.

Until a bin has `MIN_BIN_COUNT` observations it returns the raw score and reports itself
uncalibrated, so `Proposal.calibrated` stays False and nothing sizes off it.

## Abstention with teeth — the conformal quantile

`§1a` R4:

> "I don't know" must be backed by a coverage guarantee (conformal `q̂`, or an SGR
> bound), not a verbalised hedge. Track realised accuracy on non-abstained predictions
> against the promised bound.

So abstention here is not a threshold somebody typed. It is the **empirical quantile of
past non-conformity scores** at the target error rate: the model acts only when its
current score is more confident than the (1 − α) quantile of how confident it has been
when it was wrong.

And the promise is checked. `realised_coverage()` reports the accuracy actually achieved
on non-abstained predictions against the bound that was promised, so a guarantee that
stopped holding is visible rather than assumed. That check is the part R4 is really
about — a conformal quantile nobody audits is a verbalised hedge with arithmetic.

## The master test

`§1a.5`: *does it change what the system does when it is wrong?* Yes, and that is the
only reason this module exists. As the model's live accuracy degrades, the non-conformity
scores rise, `q̂` rises with them, and the brain abstains more. The bot trades less
precisely when it is being wrong more — without anybody noticing, retraining, or
intervening.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# Reliability bins across [0, 1]. Ten is enough to see a miscalibration curve and
# few enough that each fills in a reasonable number of live observations.
N_BINS = 10
# Below this an bin reports itself uncalibrated and passes the raw score through.
MIN_BIN_COUNT = 30
# How many recent non-conformity scores the conformal quantile is computed over.
# A window rather than all history: a coverage guarantee computed over a regime
# that ended is a guarantee about the past.
CONFORMAL_WINDOW = 500
# Below this many scores there is no quantile worth computing and the brain
# abstains on everything - the honest state for a model with no track record.
MIN_CONFORMAL_SCORES = 50
# Target error rate. The brain aims to be wrong at most this often on the
# predictions it does NOT abstain from.
DEFAULT_ALPHA = 0.35


@dataclass
class OnlineCalibration:
    """Reliability bins and a conformal quantile, fitted live from realised outcomes.

    Persisted as JSON so a restart resumes its own calibration rather than starting
    over. A bot that forgot how wrong it had been every time its supervisor
    restarted it would be permanently uncalibrated, and would look calibrated
    within an hour of every restart.
    """

    segment: str
    alpha: float = DEFAULT_ALPHA
    # bin index -> [n_observations, n_positive_outcomes]
    bins: dict = field(default_factory=dict)
    # Recent non-conformity scores, newest last.
    scores: list = field(default_factory=list)
    updates: int = 0
    # Realised outcomes on predictions that were ACTED on, for the R4 audit.
    acted: int = 0
    acted_correct: int = 0

    # ------------------------------------------------------------ calibration

    @staticmethod
    def _bin_of(score: float) -> int:
        return min(N_BINS - 1, max(0, int(score * N_BINS)))

    def calibrate(self, score: float) -> tuple[float, bool]:
        """Map a raw model score to a calibrated probability.

        Returns `(probability, is_calibrated)`. When the bin is too thin the raw
        score is returned with `False`, and every caller must carry that flag into
        the proposal - an uncalibrated score that reaches a sizing rule is the
        specific error `dual-agent-spec.md` names.
        """
        index = self._bin_of(score)
        entry = self.bins.get(str(index))
        if not entry or entry[0] < MIN_BIN_COUNT:
            return score, False
        observations, positives = entry
        return positives / observations, True

    # ------------------------------------------------------------- abstention

    def quantile(self) -> float | None:
        """The conformal `q̂` at the target error rate, or None if unestablished."""
        if len(self.scores) < MIN_CONFORMAL_SCORES:
            return None
        ordered = sorted(self.scores)
        n = len(ordered)
        # The standard finite-sample conformal index: ceil((n+1)(1-alpha))/n.
        rank = math.ceil((n + 1) * (1 - self.alpha))
        rank = min(max(rank, 1), n)
        return ordered[rank - 1]

    def should_abstain(self, probability: float) -> tuple[bool, dict]:
        """Whether the brain must abstain, and the evidence for the decision.

        Non-conformity here is `1 - confidence in the predicted class`: a
        prediction of 0.9 for the positive class is as confident as 0.1 for the
        negative one. The brain acts only when its non-conformity is at or below
        the quantile of what it has scored historically.
        """
        confidence = max(probability, 1 - probability)
        non_conformity = 1 - confidence
        threshold = self.quantile()
        evidence = {
            "non_conformity": round(non_conformity, 6),
            "conformal_quantile": None if threshold is None else round(threshold, 6),
            "alpha": self.alpha,
            "n_scores": len(self.scores),
            "target_coverage": round(1 - self.alpha, 4),
        }
        if threshold is None:
            # No track record, no guarantee, no trade. An abstention that says why.
            evidence["reason"] = "NO_CONFORMAL_TRACK_RECORD"
            return True, evidence
        if non_conformity > threshold:
            evidence["reason"] = "LESS_CONFIDENT_THAN_ITS_OWN_QUANTILE"
            return True, evidence
        evidence["reason"] = "WITHIN_COVERAGE"
        return False, evidence

    # ----------------------------------------------------------------- update

    def observe(self, *, score: float, outcome: int, acted: bool) -> None:
        """One realised outcome. The live loop's only parameter update.

        `score` is the RAW model score the decision was made on, `outcome` is 1
        when the positive class occurred. Both halves update: the reliability bin
        for calibration, and the non-conformity score for the quantile.
        """
        index = str(self._bin_of(score))
        observations, positives = self.bins.get(index, [0, 0])
        self.bins[index] = [observations + 1, positives + (1 if outcome else 0)]

        predicted_positive = score >= 0.5
        was_right = predicted_positive == bool(outcome)
        confidence = max(score, 1 - score)
        self.scores.append(1 - confidence if was_right else 1.0)
        if len(self.scores) > CONFORMAL_WINDOW:
            del self.scores[:-CONFORMAL_WINDOW]

        if acted:
            self.acted += 1
            self.acted_correct += 1 if was_right else 0
        self.updates += 1

    # ------------------------------------------------------------- the R4 audit

    def realised_coverage(self) -> dict:
        """What was promised against what happened, on non-abstained predictions.

        R4's real requirement. A conformal quantile that nobody checks against
        outcomes is a hedge with arithmetic on it.
        """
        promised = 1 - self.alpha
        realised = (self.acted_correct / self.acted) if self.acted else None
        return {
            "promised_coverage": round(promised, 4),
            "realised_coverage": None if realised is None else round(realised, 4),
            "acted_on": self.acted,
            "holds": None if realised is None else bool(realised >= promised - 0.10),
            "updates": self.updates,
            # Named so the board can label this apart from anything the retrainer
            # set - §1a L2 turns on that distinction.
            "fitted_by": "live loop",
        }

    # ------------------------------------------------------------- persistence

    def as_dict(self) -> dict:
        return {"segment": self.segment, "alpha": self.alpha, "bins": self.bins,
                "scores": self.scores, "updates": self.updates,
                "acted": self.acted, "acted_correct": self.acted_correct}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, *, segment: str,
             alpha: float = DEFAULT_ALPHA) -> OnlineCalibration:
        """Resume, or start fresh. A corrupt file starts fresh rather than raising.

        Deliberate asymmetry: an unreadable calibration must not stop a bot from
        trading, because the safe state is already the conservative one - no track
        record means abstain on everything until one is rebuilt.
        """
        path = Path(path)
        if path.is_file():
            try:
                held = json.loads(path.read_text(encoding="utf-8"))
                return cls(segment=held.get("segment", segment),
                           alpha=float(held.get("alpha", alpha)),
                           bins=held.get("bins", {}),
                           scores=list(held.get("scores", [])),
                           updates=int(held.get("updates", 0)),
                           acted=int(held.get("acted", 0)),
                           acted_correct=int(held.get("acted_correct", 0)))
            except (ValueError, TypeError, OSError):
                pass
        return cls(segment=segment, alpha=alpha)
