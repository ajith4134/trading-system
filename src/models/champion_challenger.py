"""Shadow before swap, and the labels arrive late — which is the whole problem.

`FEATURES.md` §3 (P2): *"Champion / challenger with delayed-label comparison —
'Shadow Before Swap'"*. Ledger MD-035.

## Why this is not just "compare two accuracies"

In most machine learning the label is available at prediction time or shortly
after. In trading it is not: a triple-barrier label matures when a barrier is
touched or the vertical barrier expires, which for a carry family is 480 bars —
eight hours — after the decision. So at any instant the record holds three kinds
of decision, and confusing them is the defect this module exists to prevent:

* **matured** — the outcome is known and both models can be scored on it
* **pending** — the horizon has not elapsed. Not a loss, not a win, not data
* **unsettled past maturity** — the horizon elapsed and nobody wrote the outcome
  down. That is an operational fault, and it is counted apart from `pending`
  because it means something upstream stopped

The naive implementation scores pending decisions as wrong, or drops them
silently. The first punishes whichever model made more recent predictions —
which is always the challenger. The second makes the comparison quietly shrink
whenever the pipeline is behind.

## Both models are scored on exactly the same decisions

A challenger that started shadowing on Tuesday has no opinion about Monday, and a
comparison that lets each model be scored on "whatever it has" compares two
different weeks and calls it a model difference. `compare()` uses only decisions
where **both** predictions are present and the label has matured, and reports
that count — so a comparison over eleven aligned decisions cannot be mistaken for
one over a thousand.

## The swap verdict is advisory, and the alias move is somebody else's

This module returns `SwapVerdict`. It does not call
`models.model_registry.assign_alias`, and it does not import the registry at all.

That separation is the same one the registry itself keeps by owning no promotion
logic: `validation.promotion_pipeline` is where a promotion decision lives, and a
second place for it is the place that drifts. What this owns is the *comparison*,
which is a fact about two models; what to do about it is a decision about
capital.

## A significance test, not a difference

`validation.superior_predictive_ability` again — one bootstrap implementation in
this project, one place for it to be wrong. On a few hundred matured decisions a
two-point accuracy difference is routinely noise, and swapping the model that
prices real positions on noise is how a system churns.

`MIN_MATURED_DECISIONS` is a floor beneath which no verdict is issued at all.
Refused rather than reported as "no evidence to swap", because those read the
same to a caller and only one of them means the challenger might be better.

## What is measured but not judged here

`window_span_ns` — how much wall-clock time the matured decisions cover — rides
every verdict, and it is the number that says whether the challenger has been
tested against anything. A challenger that beat the champion across six calm
hours has beaten it across six calm hours. Judging that is ledger **MD-022**, the
regime-coverage tracker, which does not exist yet; reporting the span without
claiming to have judged it is the honest half that can be built today.

## The record is on disk and append-only

A champion/challenger comparison spans days. Held in memory it would be a
session, and the first restart would silently reset the evidence to nothing while
the code continued to work. Settlements are appended rather than written over the
decision they settle, for the reason `validation.trial_registry` appends: an
append-only file is what stops a disappointing outcome being negotiated out of
the record later.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from validation.superior_predictive_ability import (
    NotEnoughPaths, superior_predictive_ability,
)

LEDGER_FILE = "shadow-decisions.ndjson"

KIND_DECISION = "decision"
KIND_SETTLEMENT = "settlement"

# Below this, a difference between two models is noise and swapping the model
# that prices real positions on noise is how a system churns. No verdict is
# issued at all beneath it - see the module docstring on why that is not the same
# as "no evidence to swap".
MIN_MATURED_DECISIONS = 30


class UnknownDecision(KeyError):
    """A settlement names a decision that was never recorded.

    Raised rather than ignored. A settlement with no decision means the two
    halves of this record were written by processes that disagree about what
    happened, and silently dropping it would leave the comparison short by
    however many such disagreements there are.
    """


class DecisionAlreadySettled(ValueError):
    """A decision already has an outcome.

    Refused rather than overwritten. The outcome is what the comparison rests on,
    and a second write is either a duplicate - harmless but worth seeing - or a
    revision, which is exactly the negotiation an append-only record exists to
    prevent.
    """


@dataclass(frozen=True)
class ShadowDecision:
    """One moment where both models had an opinion, and what became of it."""
    decision_id: str
    made_at_ns: int
    matures_at_ns: int
    champion_prediction: int
    challenger_prediction: int
    actual: int | None = None
    settled_at_ns: int | None = None

    def is_matured(self, now_ns: int) -> bool:
        return now_ns >= self.matures_at_ns

    @property
    def is_settled(self) -> bool:
        return self.actual is not None


@dataclass(frozen=True)
class SwapVerdict:
    """Whether the challenger has earned the alias, and what it rests on.

    `should_swap` is advisory. Moving the alias is the caller's act, through
    `models.model_registry`, and it is deliberately not done here.
    """
    should_swap: bool
    champion_accuracy: float
    challenger_accuracy: float
    p_value: float | None
    matured_decisions: int
    pending_decisions: int
    unsettled_past_maturity: int
    window_span_ns: int
    detail: str


@dataclass(frozen=True)
class NoVerdict:
    """Not enough matured, aligned decisions to say anything.

    A type of its own rather than a `SwapVerdict` with `should_swap=False`,
    because the two are different findings and the second one reads as "the
    challenger is not better" when the truth is "nobody has looked yet".
    """
    matured_decisions: int
    pending_decisions: int
    unsettled_past_maturity: int
    detail: str


class ShadowLedger:
    """The append-only record of every decision both models were asked about."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._path = self._root / LEDGER_FILE

    # --- writing ----------------------------------------------------------

    def record_decision(self, decision_id: str, *, made_at_ns: int,
                        matures_at_ns: int, champion_prediction: int,
                        challenger_prediction: int) -> ShadowDecision:
        """Append one decision, before its outcome is knowable.

        `matures_at_ns` is supplied by the caller rather than derived here,
        because the horizon belongs to the label definition - a triple-barrier
        event's vertical barrier - and deriving it from a family name in this
        module would put the label's own contract in a second place.
        """
        if matures_at_ns <= made_at_ns:
            raise ValueError(
                f"decision {decision_id!r} matures at or before it was made "
                f"({matures_at_ns} <= {made_at_ns}). A label available at "
                f"decision time is not a delayed label, and this whole module is "
                f"about the delay")
        decision = ShadowDecision(
            decision_id=decision_id, made_at_ns=int(made_at_ns),
            matures_at_ns=int(matures_at_ns),
            champion_prediction=int(champion_prediction),
            challenger_prediction=int(challenger_prediction))
        self._append({
            "kind": KIND_DECISION, "decision_id": decision.decision_id,
            "made_at_ns": decision.made_at_ns,
            "matures_at_ns": decision.matures_at_ns,
            "champion_prediction": decision.champion_prediction,
            "challenger_prediction": decision.challenger_prediction,
        })
        return decision

    def settle(self, decision_id: str, actual: int, *,
               settled_at_ns: int | None = None) -> None:
        """Append the outcome for a decision already on the record."""
        decisions = self.decisions()
        if decision_id not in decisions:
            raise UnknownDecision(
                f"settlement for {decision_id!r}, which was never recorded. The "
                f"two halves of this record were written by processes that "
                f"disagree about what happened")
        if decisions[decision_id].is_settled:
            raise DecisionAlreadySettled(
                f"decision {decision_id!r} already has an outcome. Refused "
                f"rather than overwritten - a second write is either a duplicate "
                f"worth seeing or a revision, and a revision is the negotiation "
                f"an append-only record exists to prevent")
        self._append({
            "kind": KIND_SETTLEMENT, "decision_id": decision_id,
            "actual": int(actual),
            "settled_at_ns": int(settled_at_ns if settled_at_ns is not None
                                 else time.time_ns()),
        })

    def _append(self, row: dict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # --- reading ----------------------------------------------------------

    def decisions(self) -> dict[str, ShadowDecision]:
        """Every decision, folded with its settlement, in insertion order.

        Folding on read is what keeps a decision one row rather than two - the
        same reason `validation.trial_registry` folds registrations and
        settlements instead of counting lines.
        """
        if not self._path.is_file():
            return {}
        out: dict[str, ShadowDecision] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only the final line can be torn - a process killed mid-append.
                continue
            if row["kind"] == KIND_DECISION:
                out[row["decision_id"]] = ShadowDecision(
                    decision_id=row["decision_id"],
                    made_at_ns=row["made_at_ns"],
                    matures_at_ns=row["matures_at_ns"],
                    champion_prediction=row["champion_prediction"],
                    challenger_prediction=row["challenger_prediction"])
            elif row["kind"] == KIND_SETTLEMENT:
                existing = out.get(row["decision_id"])
                if existing is None or existing.is_settled:
                    continue
                out[row["decision_id"]] = ShadowDecision(
                    decision_id=existing.decision_id,
                    made_at_ns=existing.made_at_ns,
                    matures_at_ns=existing.matures_at_ns,
                    champion_prediction=existing.champion_prediction,
                    challenger_prediction=existing.challenger_prediction,
                    actual=row["actual"], settled_at_ns=row["settled_at_ns"])
        return out

    # --- the comparison ---------------------------------------------------

    def compare(self, now_ns: int | None = None, *, alpha: float = 0.05,
                seed: int = 0) -> SwapVerdict | NoVerdict:
        """Score both models on the decisions where both have an answer.

        Three populations are separated and all three are reported:

          * scored - matured AND settled AND both models predicted
          * pending - the horizon has not elapsed. Not a loss and not data
          * unsettled past maturity - the horizon elapsed and nobody wrote the
            outcome down, which is an operational fault rather than a result
        """
        now_ns = int(now_ns if now_ns is not None else time.time_ns())
        decisions = list(self.decisions().values())

        scored = [d for d in decisions if d.is_matured(now_ns) and d.is_settled]
        pending = sum(1 for d in decisions if not d.is_matured(now_ns))
        stale = sum(1 for d in decisions
                    if d.is_matured(now_ns) and not d.is_settled)

        if len(scored) < MIN_MATURED_DECISIONS:
            return NoVerdict(
                matured_decisions=len(scored), pending_decisions=pending,
                unsettled_past_maturity=stale,
                detail=(f"{len(scored)} matured and settled decision(s), need "
                        f">= {MIN_MATURED_DECISIONS}. {pending} still pending "
                        f"and {stale} matured but unsettled. Nobody has looked "
                        f"yet - which is not the same finding as the challenger "
                        f"not being better"))

        champion = [1.0 if d.champion_prediction == d.actual else 0.0
                    for d in scored]
        challenger = [1.0 if d.challenger_prediction == d.actual else 0.0
                      for d in scored]
        champion_accuracy = sum(champion) / len(champion)
        challenger_accuracy = sum(challenger) / len(challenger)
        span = (max(d.matures_at_ns for d in scored)
                - min(d.made_at_ns for d in scored))

        if champion == challenger:
            # Identical on every decision. The bootstrap has no variance to work
            # with, and "indistinguishable" is the answer rather than an error.
            return SwapVerdict(
                should_swap=False, champion_accuracy=champion_accuracy,
                challenger_accuracy=challenger_accuracy, p_value=None,
                matured_decisions=len(scored), pending_decisions=pending,
                unsettled_past_maturity=stale, window_span_ns=span,
                detail=(f"the challenger agreed with the champion on all "
                        f"{len(scored)} matured decision(s) - there is nothing "
                        f"to swap to"))

        try:
            spa = superior_predictive_ability(
                incumbent=champion, challengers={"challenger": challenger},
                seed=seed, alpha=alpha)
            should_swap = "challenger" in spa.superior
            p_value = spa.p_value
        except NotEnoughPaths:                     # pragma: no cover - floored above
            return NoVerdict(
                matured_decisions=len(scored), pending_decisions=pending,
                unsettled_past_maturity=stale,
                detail="too few paths for the bootstrap")

        hours = span / 3_600_000_000_000
        verdict = "SWAP" if should_swap else "hold"
        return SwapVerdict(
            should_swap=should_swap, champion_accuracy=champion_accuracy,
            challenger_accuracy=challenger_accuracy, p_value=p_value,
            matured_decisions=len(scored), pending_decisions=pending,
            unsettled_past_maturity=stale, window_span_ns=span,
            detail=(f"{verdict}: challenger {challenger_accuracy:.4f} against "
                    f"champion {champion_accuracy:.4f} over {len(scored)} "
                    f"matured decision(s) spanning {hours:.1f}h "
                    f"(p={p_value:.4f}); {pending} pending, {stale} matured but "
                    f"unsettled. The span is reported, not judged - whether "
                    f"{hours:.1f}h covered a regime is MD-022's question and "
                    f"MD-022 does not exist yet"))
