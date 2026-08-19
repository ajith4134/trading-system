"""Whether a freshly trained model may take the champion alias — CG-01, RL-044.

## The hole this fills

Measured 2026-08-19: `learn.train_segment_model` assigned the champion alias
UNCONDITIONALLY, and `models.model_registry.assign_alias` read the incumbent only
to record it as `previous_version_id`. Nothing compared them. The champion was
therefore whatever trained LAST, and a retrain producing a worse model silently
replaced a better one.

What that cost, the same day it was found. perp's champion moved
`976ee2399a5c` to `9fa37695dae0` at 15:34:

    accuracy    0.54856  ->  0.54862      looks like a tie
    base rate   0.53161  ->  0.53635      the market got easier to guess
    EDGE        0.01695  ->  0.01227      the actual skill fell 28%
    out-of-fold 158,647  ->   91,916      on 42% less data

The outgoing model had opened 3,352 positions. The incoming one proposed 10 times
in 600 polls and selected none.

## Why EDGE and not accuracy

Accuracy alone is the number that made the swap look harmless. A model scoring
0.5486 against a base rate of 0.5363 knows less than one scoring 0.5486 against
0.5316 - identical accuracy, different problem. The base rate is what a coin
weighted to the majority class would score, so the only thing a model is entitled
to claim is the distance above it.

## Why this is not `evaluate_for_promotion`

`validation.promotion_gate` is built for a candidate with a RETURN SERIES: it
takes a `backtest_fold` returning per-path returns and computes a deflated Sharpe
against a trial count. A direction classifier has no return series - it emits
predictions, which is why `walk_forward.as_registry_result` sets `sharpe: None`
in its own words. Routing a classifier through it would mean inventing returns to
satisfy an interface, and a fabricated series is worse than no gate because it
produces a number that looks like a Sharpe.

This is the classifier-shaped sibling, and it lives in `validation/` for the
reason `models.champion_challenger` states: promotion decisions live here.

## The check that deliberately does NOT gate

Effective sample fraction was **0.0022 for the model that was replaced and 0.0038
for the one that replaced it** - so a floor on it would have blocked the BETTER
model. Both sit around a quarter of one percent of nominal, which is its own
problem and a real one. But a threshold there has the power to stop all promotion
and leave every bot on whatever it happens to be running, so it is MEASURED and
REPORTED on every verdict and left to a human to set. A gate that quietly picked
that number would be making a decision of that size on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# What a challenger must clear to take the alias. Deliberately small: this gate
# exists to stop a REGRESSION, not to raise the bar on what may be trained. A
# challenger equal to the incumbent does not promote - ties go to the model
# already serving, because a swap costs a warm-up the incumbent has already paid.
DEFAULT_MIN_EDGE_IMPROVEMENT = 0.0


@dataclass(frozen=True)
class Check:
    """One question the gate asked, its answer, and the numbers behind it."""

    name: str
    passed: bool
    detail: str
    measured: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail,
                "measured": self.measured, "threshold": self.threshold}


@dataclass(frozen=True)
class ChampionVerdict:
    """Whether the challenger may serve, and everything that decided it."""

    promote: bool
    checks: list[Check] = field(default_factory=list)
    challenger_edge: float | None = None
    incumbent_edge: float | None = None
    incumbent_version_id: str | None = None

    @property
    def refusals(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed]

    def describe(self) -> str:
        """One line for the alias reason, or for a refusal record."""
        if self.promote:
            if self.incumbent_edge is None:
                return (f"promoted: out-of-fold edge {self.challenger_edge:.5f}, "
                        f"no incumbent to beat")
            return (f"promoted: out-of-fold edge {self.challenger_edge:.5f} beats "
                    f"incumbent {self.incumbent_version_id} at "
                    f"{self.incumbent_edge:.5f}")
        return "refused: " + "; ".join(self.refusals)

    def as_dict(self) -> dict:
        return {"promote": self.promote,
                "challenger_edge": self.challenger_edge,
                "incumbent_edge": self.incumbent_edge,
                "incumbent_version_id": self.incumbent_version_id,
                "checks": [c.as_dict() for c in self.checks]}


def out_of_fold_edge(metrics) -> float | None:
    """Accuracy above the base rate, or None when it cannot be computed.

    None is not zero. A model whose metrics do not carry both numbers has not
    been measured, and treating that as an edge of zero would let an unmeasured
    challenger tie with a measured incumbent and lose only on the tie-break -
    when the honest answer is that it cannot be compared at all.
    """
    if not isinstance(metrics, dict):
        return None
    accuracy = metrics.get("accuracy")
    base_rate = metrics.get("base_rate")
    if accuracy is None or base_rate is None:
        return None
    try:
        return float(accuracy) - float(base_rate)
    except (TypeError, ValueError):
        return None


def evaluate_champion(*, challenger_metrics: dict,
                      incumbent_metrics: dict | None = None,
                      incumbent_version_id: str | None = None,
                      min_edge_improvement: float = DEFAULT_MIN_EDGE_IMPROVEMENT,
                      ) -> ChampionVerdict:
    """Decide whether the challenger may take the alias from the incumbent.

    The first champion is a special case and is promoted on its own merits: there
    is nothing to regress against, and refusing it would leave the bot with no
    model at all - a worse state than an unproven one.
    """
    checks: list[Check] = []
    challenger_edge = out_of_fold_edge(challenger_metrics)

    # Effective sample is REPORTED, never gated. See the module docstring: on the
    # measured case a floor would have blocked the better model.
    effective = (challenger_metrics or {}).get("effective_sample_fraction")
    if effective is not None:
        checks.append(Check(
            name="effective_sample_reported", passed=True,
            measured=float(effective), threshold=None,
            detail=(f"effective sample {float(effective):.4f} of nominal - reported, "
                    f"not gated; a threshold here can stop all promotion and is a "
                    f"human's to set")))

    if challenger_edge is None:
        checks.append(Check(
            name="challenger_measured", passed=False,
            detail=("the challenger carries no accuracy/base_rate pair, so its edge "
                    "cannot be computed and it cannot be compared to anything")))
        return ChampionVerdict(promote=False, checks=checks,
                               incumbent_version_id=incumbent_version_id)

    checks.append(Check(
        name="challenger_measured", passed=True, measured=challenger_edge,
        detail=f"challenger out-of-fold edge {challenger_edge:.5f}"))

    beats_majority = (challenger_metrics or {}).get("beats_majority_class")
    if beats_majority is False:
        checks.append(Check(
            name="beats_majority_class", passed=False,
            detail="the challenger does not beat the majority class"))
        return ChampionVerdict(promote=False, checks=checks,
                               challenger_edge=challenger_edge,
                               incumbent_version_id=incumbent_version_id)
    checks.append(Check(name="beats_majority_class", passed=True,
                        detail="the challenger beats the majority class"))

    incumbent_edge = out_of_fold_edge(incumbent_metrics)

    if incumbent_version_id is None:
        checks.append(Check(
            name="no_regression", passed=True,
            detail="no incumbent holds this alias; the first champion promotes on "
                   "its own merits"))
        return ChampionVerdict(promote=True, checks=checks,
                               challenger_edge=challenger_edge)

    if incumbent_edge is None:
        # An incumbent nobody can score cannot be defended. Promoting is the right
        # call - but it is recorded as such, so "we replaced a model we could not
        # measure" never reads later as "we beat it".
        checks.append(Check(
            name="no_regression", passed=True,
            detail=(f"incumbent {incumbent_version_id} carries no measurable edge, "
                    f"so the challenger is not being compared - it is replacing an "
                    f"unmeasured model")))
        return ChampionVerdict(promote=True, checks=checks,
                               challenger_edge=challenger_edge,
                               incumbent_version_id=incumbent_version_id)

    improvement = challenger_edge - incumbent_edge
    passed = improvement > min_edge_improvement
    checks.append(Check(
        name="no_regression", passed=passed, measured=improvement,
        threshold=min_edge_improvement,
        detail=(f"challenger edge {challenger_edge:.5f} against incumbent "
                f"{incumbent_version_id} at {incumbent_edge:.5f}: "
                f"{'an improvement of' if passed else 'a change of'} "
                f"{improvement:+.5f}"
                + ("" if passed else
                   " - the model already serving keeps the alias, and a tie goes to "
                   "it because a swap costs a warm-up it has already paid"))))

    return ChampionVerdict(promote=passed, checks=checks,
                           challenger_edge=challenger_edge,
                           incumbent_edge=incumbent_edge,
                           incumbent_version_id=incumbent_version_id)


# The quantile the tail champion is judged on. The 0.5 pinball loss is the
# median forecast's error; the 0.1 tail is what the risk gate consumes, but a
# model can look good there by forecasting a wide tail on everything, so the
# median is the one that cannot be gamed by widening.
TAIL_SCORE_QUANTILE = "0.5"


def tail_pinball_loss(metrics, quantile: str = TAIL_SCORE_QUANTILE) -> float | None:
    """The profit-tail model's loss at one quantile, or None if unmeasured."""
    if not isinstance(metrics, dict):
        return None
    losses = metrics.get("pinball_loss")
    if not isinstance(losses, dict) or quantile not in losses:
        return None
    try:
        return float(losses[quantile])
    except (TypeError, ValueError):
        return None


def evaluate_tail_champion(*, challenger_metrics: dict,
                           incumbent_metrics: dict | None = None,
                           incumbent_version_id: str | None = None,
                           ) -> ChampionVerdict:
    """The same no-regression rule for a model scored by LOSS, not accuracy.

    The profit-tail model forecasts quantiles of forward P&L, so it has no
    accuracy and no base rate - `evaluate_champion` would find no edge on it and
    refuse every promotion, which would freeze the tail brains on whatever they
    happened to be running.

    Lower loss is better here, which is the only thing that inverts. Ties still go
    to the incumbent for the same reason: a swap costs a warm-up already paid.

    The verdict's `challenger_edge` and `incumbent_edge` carry NEGATED loss, so
    that on both this gate and the accuracy one, larger always means better. A
    field that meant "higher is worse" on one path and "higher is better" on the
    other is the sign error waiting to happen in whatever reads them next.
    """
    checks: list[Check] = []
    challenger_loss = tail_pinball_loss(challenger_metrics)

    if challenger_loss is None:
        checks.append(Check(
            name="challenger_measured", passed=False,
            detail=(f"the challenger carries no pinball loss at quantile "
                    f"{TAIL_SCORE_QUANTILE}, so it cannot be compared")))
        return ChampionVerdict(promote=False, checks=checks,
                               incumbent_version_id=incumbent_version_id)

    checks.append(Check(
        name="challenger_measured", passed=True, measured=challenger_loss,
        detail=f"challenger pinball loss {challenger_loss:.8f} at q"
               f"{TAIL_SCORE_QUANTILE}"))

    incumbent_loss = tail_pinball_loss(incumbent_metrics)

    if incumbent_version_id is None:
        checks.append(Check(
            name="no_regression", passed=True,
            detail="no incumbent holds this alias; the first champion promotes on "
                   "its own merits"))
        return ChampionVerdict(promote=True, checks=checks,
                               challenger_edge=-challenger_loss)

    if incumbent_loss is None:
        checks.append(Check(
            name="no_regression", passed=True,
            detail=(f"incumbent {incumbent_version_id} carries no pinball loss, so "
                    f"the challenger is replacing an unmeasured model rather than "
                    f"beating one")))
        return ChampionVerdict(promote=True, checks=checks,
                               challenger_edge=-challenger_loss,
                               incumbent_version_id=incumbent_version_id)

    passed = challenger_loss < incumbent_loss
    checks.append(Check(
        name="no_regression", passed=passed,
        measured=incumbent_loss - challenger_loss, threshold=0.0,
        detail=(f"challenger loss {challenger_loss:.8f} against incumbent "
                f"{incumbent_version_id} at {incumbent_loss:.8f}: "
                f"{'lower, so it serves' if passed else 'not lower, so the model '
                   'already serving keeps the alias'}")))

    return ChampionVerdict(promote=passed, checks=checks,
                           challenger_edge=-challenger_loss,
                           incumbent_edge=-incumbent_loss,
                           incumbent_version_id=incumbent_version_id)
