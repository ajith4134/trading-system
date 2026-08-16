"""A tail cap that adapts — on paper only, bounded, and with a record of every move.

Added 2026-08-16 at the user's instruction: *"for paper trading it is for
practising and experimenting so make these dynamic or auto adjust"*.

## Why this is a separate module and not a mode inside `risk.tail_cap`

`risk.tail_cap` has exactly one function that writes the ceiling file —
`seed_ceiling` — and it refuses to overwrite one that exists. That refusal is
§6 made mechanical: *"no code in this system raises it"*. Adding an adaptive
path inside that module would put a widening function next to the invariant it
must not touch, and the first refactor that blurred the two would be invisible.

So the live ceiling keeps its single writer and its absolute meaning, and this
module derives a **separate, paper-only** cap beside it. There is no function
here that returns a live cap, no argument that selects live, and
`PaperCap.mode` is the constant `PAPER`. `models.champion_challenger` keeps the
same kind of distance from the model registry, for the same reason.

## Why adapting is right on paper and wrong live

Paper is where a strategy is practised, and a practice environment has two
useless extremes. A cap far too tight halts every experiment before it says
anything, so nothing is learned. A cap far too loose is never touched, so the
strategy never demonstrates it can live inside a limit — and *"it never
breached"* against a limit that could not be reached is not evidence.

The seeded live ceiling (daily 1.0%, drawdown 3.0%) is a **proposal** the system
wrote on 2026-08-09 from §0's ~2.8% annualised carry profile, and its own note
says to revise it from measured volatility once there is any. On paper there now
is some. Deriving the paper cap from what the strategy actually does is the
honest version of that sentence.

Live is the opposite case. There the number is the user's, the consequences are
real, and a cap that moves is a cap nobody agreed to.

## Bounded, because a limit that widens with volatility is how accounts die

This is the danger and it is not hypothetical: a risk limit derived from recent
realised risk **rises exactly when risk rises**, so it stops binding at the
moment it was for. The bound is therefore not optional.

* `MAX_WIDENING_MULTIPLE` — the paper cap may never exceed this multiple of the
  live ceiling. Widening is capped, and hitting the cap is reported, so *"the
  measured drawdown wants more room than we will give it"* is visible rather
  than silently granted.
* `MIN_TIGHTENING_MULTIPLE` — nor may it collapse. A cap derived from an
  unusually calm stretch would otherwise halt on the first ordinary day, which
  teaches the same nothing as halting on the first tick.
* `MIN_OBSERVATIONS` — below it the live ceiling is returned unchanged, with the
  reason. A cap derived from twenty paper minutes is not adaptive, it is random,
  and the tidy percentile it produces is what would make it convincing.

## The derivation is the same bootstrap the ladder already uses

`risk.drawdown_distribution.block_bootstrap_max_drawdowns` — a **block**
bootstrap, because IID resampling destroys the autocorrelation that makes
drawdowns deep and would understate the distribution it is being asked about.
The drawdown fraction is taken at `DRAWDOWN_PERCENTILE` of that distribution, not
at the realised maximum: the realised maximum is one draw, and sizing on it plans
for a past that happened to be lucky or unlucky.

The daily fraction is derived from the same series at `DAILY_PERCENTILE` of the
absolute single-period losses, keeping the two limits in the same relationship
the seeded pair has rather than inventing a second methodology for it.

## Every move is appended, never overwritten

`paper-tail-cap-history.ndjson` records each derivation: the values, the number
of observations behind them, whether either bound bit, and what the previous
values were.

Without it, a paper result cannot be read afterwards. *"The strategy never
breached"* is meaningless unless what it was measured against on that day is
recoverable, and a cap that moves silently makes every past paper run
uninterpretable. Same reasoning as `models.model_registry`'s alias history, and
the same append-only shape.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from risk.drawdown_distribution import NotEnoughHistory, block_bootstrap_max_drawdowns
from risk.tail_cap import PAPER, TailCap

HISTORY_FILE = "paper-tail-cap-history.ndjson"

# The paper cap may not exceed this multiple of the live ceiling, whatever the
# measured drawdowns want. See the module docstring: a limit derived from recent
# realised risk rises exactly when risk rises.
MAX_WIDENING_MULTIPLE = Decimal("3")

# Nor may it collapse below this multiple. A cap derived from an unusually calm
# stretch would halt on the first ordinary day, which teaches nothing.
MIN_TIGHTENING_MULTIPLE = Decimal("0.25")

# Below this the live ceiling is returned unchanged. Mirrors the bootstrap's own
# floor and exists for the same reason: resampling a short sample manufactures
# confidence, and the tidy percentile it produces is what makes it dangerous.
MIN_OBSERVATIONS = 100

# Where on the bootstrapped distribution each limit sits. Declared, not searched:
# a percentile chosen until the cap felt comfortable would be the system setting
# its own limit by way of a hyperparameter.
DRAWDOWN_PERCENTILE = 0.90
DAILY_PERCENTILE = 0.99


@dataclass(frozen=True)
class PaperCap:
    """A derived, paper-only cap and the evidence behind it.

    `mode` is the constant PAPER and there is no path that sets it otherwise.
    `is_derived` is False when the live ceiling was returned unchanged, so a
    caller can tell an adapted cap from a passed-through one - they are the same
    numbers in the common early case and mean entirely different things.
    """
    cap: TailCap
    mode: str
    is_derived: bool
    observations: int
    widening_bound_hit: bool
    tightening_bound_hit: bool
    measured_drawdown_fraction: Decimal | None
    measured_daily_fraction: Decimal | None
    reason: str

    def describe(self) -> str:
        if not self.is_derived:
            return f"live ceiling passed through: {self.reason}"
        bounds = []
        if self.widening_bound_hit:
            bounds.append("widening bound bit")
        if self.tightening_bound_hit:
            bounds.append("tightening bound bit")
        note = f" [{', '.join(bounds)}]" if bounds else ""
        return (f"paper cap {self.cap.describe()} derived from "
                f"{self.observations} observation(s); measured drawdown "
                f"{self.measured_drawdown_fraction * 100:.2f}%{note}")


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values to take a percentile of")
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _bounded(measured: Decimal, live: Decimal) -> tuple[Decimal, bool, bool]:
    """Clamp a derived fraction into the band around the live ceiling."""
    widest = live * MAX_WIDENING_MULTIPLE
    tightest = live * MIN_TIGHTENING_MULTIPLE
    if measured > widest:
        return widest, True, False
    if measured < tightest:
        return tightest, False, True
    return measured, False, False


def derive_paper_cap(live_ceiling: TailCap, returns: Sequence[float], *,
                     n_samples: int = 1000, block_size: int = 50,
                     seed: int = 0) -> PaperCap:
    """A paper cap from the strategy's own measured returns, bounded both ways.

    Returns the live ceiling unchanged - with `is_derived=False` and the reason -
    when there is too little history, rather than raising. The early state of
    every paper run is "not enough yet", and a caller should be able to render
    that on a board without a try block, the same posture
    `models.ledger_meta_model.FitRefused` takes.
    """
    if len(returns) < MIN_OBSERVATIONS:
        return PaperCap(
            cap=live_ceiling, mode=PAPER, is_derived=False,
            observations=len(returns), widening_bound_hit=False,
            tightening_bound_hit=False, measured_drawdown_fraction=None,
            measured_daily_fraction=None,
            reason=(f"{len(returns)} observation(s), need >= "
                    f"{MIN_OBSERVATIONS}; a cap derived from this much history "
                    f"is random rather than adaptive"))
    try:
        drawdowns = block_bootstrap_max_drawdowns(
            list(returns), n_samples=n_samples, block_size=block_size, seed=seed)
    except (NotEnoughHistory, ValueError) as error:
        return PaperCap(
            cap=live_ceiling, mode=PAPER, is_derived=False,
            observations=len(returns), widening_bound_hit=False,
            tightening_bound_hit=False, measured_drawdown_fraction=None,
            measured_daily_fraction=None,
            reason=f"the bootstrap refused this series: {error}")

    measured_drawdown = Decimal(str(_percentile(drawdowns, DRAWDOWN_PERCENTILE)))
    losses = [abs(r) for r in returns if r < 0] or [0.0]
    measured_daily = Decimal(str(_percentile(losses, DAILY_PERCENTILE)))

    drawdown, drawdown_wide, drawdown_tight = _bounded(
        measured_drawdown, live_ceiling.drawdown_fraction)
    daily, daily_wide, daily_tight = _bounded(
        measured_daily, live_ceiling.daily_loss_fraction)

    return PaperCap(
        cap=TailCap(daily_loss_fraction=daily, drawdown_fraction=drawdown),
        mode=PAPER, is_derived=True, observations=len(returns),
        widening_bound_hit=drawdown_wide or daily_wide,
        tightening_bound_hit=drawdown_tight or daily_tight,
        measured_drawdown_fraction=measured_drawdown,
        measured_daily_fraction=measured_daily,
        reason=(f"p{DRAWDOWN_PERCENTILE:.2f} of {len(drawdowns)} block-bootstrap "
                f"max drawdowns over {len(returns)} observation(s)"))


def record_paper_cap(root: Path, derived: PaperCap,
                     previous: TailCap | None = None) -> None:
    """Append this derivation to the history. Never rewrites.

    Without this file a past paper run cannot be read: "the strategy never
    breached" is meaningless unless the limit it was measured against that day is
    recoverable, and a cap that moves silently makes every earlier run
    uninterpretable.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at_ns": time.time_ns(),
        "mode": derived.mode,
        "is_derived": derived.is_derived,
        "observations": derived.observations,
        "daily_loss_fraction": str(derived.cap.daily_loss_fraction),
        "drawdown_fraction": str(derived.cap.drawdown_fraction),
        "previous_daily_loss_fraction": (
            str(previous.daily_loss_fraction) if previous else None),
        "previous_drawdown_fraction": (
            str(previous.drawdown_fraction) if previous else None),
        "measured_drawdown_fraction": (
            str(derived.measured_drawdown_fraction)
            if derived.measured_drawdown_fraction is not None else None),
        "widening_bound_hit": derived.widening_bound_hit,
        "tightening_bound_hit": derived.tightening_bound_hit,
        "reason": derived.reason,
    }
    with open(root / HISTORY_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_paper_cap_history(root: Path) -> list[dict]:
    """Every derivation ever recorded, oldest first.

    A torn final line is skipped rather than fatal - a process killed mid-append
    must not make unreadable the file that explains what a past paper run was
    measured against.
    """
    path = Path(root) / HISTORY_FILE
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
