"""The tail-loss ceiling — enforced on real money, measured on paper.

`§0` of the goal spec states the objective as a shape, not a number:

    Maximise the fraction of green days, subject to a hard tail-loss cap the
    system cannot raise.

and says why the cap is part of the objective rather than a safety tax bolted
on: *"maximise green days" on its own is the objective that selects for selling
insurance* - short vol, premium harvest, green almost every day, and the whole
multi-year gain returned in one session. The cap is the term that makes the
objective safe to optimise against.

`§6` makes it a human gate: *"the system may raise and lower its own risk limits,
and clear its own halts, inside a ceiling the user sets once."* So this module
READS the ceiling and never writes it. There is no function here that raises it,
and that absence is the guarantee - not a permission check something could be
persuaded past.

## Two caps, because they fail differently

A **daily** cap answers §0 literally: no single session undoes a year. A
**peak-to-trough** ceiling catches what a daily cap never trips - twenty
consecutive 0.4% days is 8% gone without a single breach, and that is close to
the shape of the prior bot's **net -$837 over 10,240 live trades**, where no
individual day was dramatic.

Neither covers the other, so both exist.

## Paper is unconstrained, and that is a ruling rather than an oversight

Decided 2026-08-09: **the caps bind only on real money.** Paper trades fake
capital, and capital that cannot be lost does not need protecting. Constraining
exploration there would cost the thing §0 calls the asset - *"the failures are
the asset"* - by refusing the experiments that find where the edges are.

But paper is still MEASURED against the live ceiling, and that matters more than
it sounds. A candidate tuned with no ceiling has been optimised against a
different objective from the one it must satisfy with money behind it. Promoting
it would hand capital to a strategy that has never once been tested against the
constraint it is about to live under. So `PAPER` records every breach it would
have suffered and `would_have_breached` is a promotion input - the strategy has
to show it could have lived inside the cap, even though nothing stopped it.

## What a breach does

Trips `ops.watchdog`, which writes the kill file every trading path already
consults, and only a human clears it. That is `§6`'s other human gate - *"paper →
real money"* - read in the direction nobody enjoys.

It also gives the watchdog the trigger it has never had. `DECISIONS.md` §13
records the trip side as unarmed because *"a hard cap needs capital to have a
hard cap over"*. This is that cap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

# Where the ceiling lives. A file the user owns, read here and written nowhere in
# this codebase - the same shape as `capture/eviction-keep-days`, and for a
# stronger reason.
CEILING_FILE = "tail-cap.json"

PAPER = "paper"
LIVE = "live"

# Proposed defaults, and they are a PROPOSAL - §6 says the ceiling is the user's
# to set once, so these are what the file is seeded with rather than what the
# system decided.
#
# Both derive from one number in §0's own table: market-neutral carry returns
# **~2.8% annualised**. That makes 2.8% in a day the definition of "a year
# undone", so it is the ceiling on any sane answer rather than a candidate.
#
#   daily 1.0%     ~4.3 months of expected return in one session. Painful,
#                  survivable, and comfortably short of undoing the year.
#   drawdown 3.0%  slightly more than a year's expected return given back. Past
#                  that, continuing is not defensible without a human looking at
#                  it.
#
# Stated as a first setting, not a derivation: the honest version of both numbers
# comes from measured live volatility, and there is none yet. Revisit once there
# is - that is Rule 0 applied to the risk limits themselves.
DEFAULT_DAILY_LOSS_FRACTION = Decimal("0.010")
DEFAULT_DRAWDOWN_FRACTION = Decimal("0.030")


class CeilingNotSet(RuntimeError):
    """No ceiling on disk. Live trading must not proceed without one."""


@dataclass(frozen=True)
class TailCap:
    """The ceiling, as fractions of NAV. §4: never an absolute."""

    daily_loss_fraction: Decimal
    drawdown_fraction: Decimal

    def describe(self) -> str:
        return (f"daily {self.daily_loss_fraction * 100:.2f}% of NAV, "
                f"peak-to-trough {self.drawdown_fraction * 100:.2f}%")


@dataclass(frozen=True)
class CapAssessment:
    """What the caps say about one moment. Every field is measured."""

    mode: str
    nav: Decimal
    day_open_nav: Decimal
    peak_nav: Decimal
    daily_loss_fraction: Decimal
    drawdown_fraction: Decimal
    daily_breached: bool
    drawdown_breached: bool

    @property
    def breached(self) -> bool:
        return self.daily_breached or self.drawdown_breached

    @property
    def enforced(self) -> bool:
        """Whether a breach here stops anything. False on paper, by ruling."""
        return self.mode == LIVE

    @property
    def would_have_breached(self) -> bool:
        """A paper breach: nothing stopped, and a promotion input.

        A strategy that could not have lived inside the ceiling has not earned
        capital, however well it did without one.
        """
        return self.breached and not self.enforced

    def reason(self) -> str:
        parts = []
        if self.daily_breached:
            parts.append(f"day down {self.daily_loss_fraction * 100:.2f}%")
        if self.drawdown_breached:
            parts.append(f"peak-to-trough {self.drawdown_fraction * 100:.2f}%")
        return " and ".join(parts) or "within the ceiling"


def read_ceiling(root: Path) -> TailCap:
    """The ceiling from disk. Raises rather than defaulting.

    A missing ceiling is not zero risk appetite and it is not the default
    either - it is a question nobody answered, and answering it on the user's
    behalf is precisely what §6 reserves to them. `seed_ceiling` writes the
    proposal; something has to call it deliberately.
    """
    path = Path(root) / CEILING_FILE
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CeilingNotSet(
            f"no tail-loss ceiling at {path}. §6 reserves it to the user: seed "
            f"the proposal with `risk.tail_cap.seed_ceiling` and edit it, or "
            f"write the file by hand. Nothing trades real money without one."
        ) from None
    except (OSError, ValueError) as exc:
        raise CeilingNotSet(f"tail-loss ceiling at {path} is unreadable: {exc}") from exc

    try:
        cap = TailCap(
            daily_loss_fraction=Decimal(str(body["daily_loss_fraction"])),
            drawdown_fraction=Decimal(str(body["drawdown_fraction"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CeilingNotSet(f"tail-loss ceiling at {path} is malformed: {exc}") from exc

    if cap.daily_loss_fraction <= 0 or cap.drawdown_fraction <= 0:
        raise CeilingNotSet(
            f"tail-loss ceiling at {path} is non-positive, which would halt on "
            f"the first tick rather than bound anything")
    return cap


def seed_ceiling(root: Path,
                 daily_loss_fraction: Decimal = DEFAULT_DAILY_LOSS_FRACTION,
                 drawdown_fraction: Decimal = DEFAULT_DRAWDOWN_FRACTION) -> Path:
    """Write the proposed ceiling, once, and never overwrite one that exists.

    Refusing to overwrite is the whole point. This is the only function here
    that writes the file at all, and it will not raise a ceiling that is already
    set - so no code path in this system can widen its own limit, which is what
    §6 means by a ceiling the system cannot raise.
    """
    path = Path(root) / CEILING_FILE
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "daily_loss_fraction": str(daily_loss_fraction),
        "drawdown_fraction": str(drawdown_fraction),
        "_note": "§6: the user sets this once; no code in this system raises it. "
                 "Fractions of NAV (§4). Seeded 2026-08-09 as a proposal derived "
                 "from §0's ~2.8% annualised carry profile - revise from measured "
                 "live volatility once there is any.",
    }, indent=2) + "\n", encoding="utf-8")
    return path


def assess(cap: TailCap, nav: Decimal, day_open_nav: Decimal, peak_nav: Decimal,
           mode: str = PAPER) -> CapAssessment:
    """Measure this moment against the ceiling. Decides nothing on its own.

    Losses are fractions of the reference they are measured from - the day's
    opening NAV for the daily cap, the running peak for the drawdown - because
    §4 says every size is a fraction of NAV and a cap in currency would mean
    something different at each end of a dial that runs $2k to $100k+.
    """
    daily_loss = ((day_open_nav - nav) / day_open_nav
                  if day_open_nav > 0 else Decimal("0"))
    drawdown = ((peak_nav - nav) / peak_nav
                if peak_nav > 0 else Decimal("0"))
    return CapAssessment(
        mode=mode,
        nav=nav,
        day_open_nav=day_open_nav,
        peak_nav=peak_nav,
        daily_loss_fraction=daily_loss,
        drawdown_fraction=drawdown,
        daily_breached=daily_loss >= cap.daily_loss_fraction,
        drawdown_breached=drawdown >= cap.drawdown_fraction,
    )


def enforce(assessment: CapAssessment, watchdog, pids=()) -> dict | None:
    """Trip the kill on a LIVE breach. Returns the watchdog's outcome, or None.

    Paper returns None however badly it breached - the ruling of 2026-08-09,
    and the caller is expected to record `would_have_breached` rather than treat
    the None as an all-clear.
    """
    if not (assessment.breached and assessment.enforced):
        return None
    return watchdog.trip(
        reason="tail_cap",
        detail=(f"{assessment.reason()} against a ceiling of "
                f"{assessment.daily_loss_fraction * 100:.2f}%/day observed; "
                f"NAV {assessment.nav} from day open {assessment.day_open_nav}, "
                f"peak {assessment.peak_nav}"),
        pids=list(pids),
    )
