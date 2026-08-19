"""How much leverage one trade takes, and whether its stop survives it — CL-06.

RL-041: both bots lever, and the multiple is chosen PER TRADE under a ceiling
declared per segment. Leverage did not exist anywhere in this codebase before this
module - no leverage, no margin, no liquidation - so nothing here is exposing a
setting that was already there.

## The three things that make leverage safe, and none is optional

**1. The choosing rule is declared and journalled.** `capital.json` names it, this
module implements it, and the fill records which rule ran and on what inputs. A
multiple nobody can trace is a multiple nobody can argue with afterwards.

**2. The stop has to fit inside the liquidation distance.** At `L` leverage an
isolated position is liquidated on roughly a `1/L` adverse move, less maintenance
margin. A hard stop WIDER than that can never fire - the position dies first, and
the risk gate goes on believing in protection it does not have. So the stop is
checked against the distance, leverage is REDUCED until it fits, and if it still
does not fit at the floor the trade is refused. Reducing before refusing matters:
the trade is usually fine at 3x and only impossible at 20x.

**3. Leveraged spot is borrowed money and is charged for.** Not charging makes the
spot bot's returns a free loan, and - this is the part that makes it a bug rather
than an omission - it inflates them MORE the more leverage is taken. An accounting
error that rewards recklessness will be found by the optimiser long before it is
found by a person.

## Why volatility-targeted and not confidence-scaled

The default rule sizes leverage as `target vol / this instrument's vol`, so a quiet
instrument gets more and a violent one gets less, and every position carries about
the same risk. That is the professional default and it is also the CAUTIOUS one
here, because the alternative is worse than it looks: both bots journal
`calibrated=False`, so a confidence of 0.9 is not a 90% chance of anything. Scaling
leverage by it would compound a miscalibration precisely where it costs most, on
the trades the model is most sure about.

## The units conversion, done once, here

`live_features` publishes `window_volatility` - movement over the feature window,
about 180 seconds - and deliberately does NOT annualise, because "one conversion
per caller is one chance each to get it wrong". The declaration states an ANNUAL
target because that is the number a person can reason about. So exactly one
conversion exists, it lives in `annualise_window_volatility`, and it is tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

# Crypto trades every day, so the year is a calendar year and not 252 sessions.
# Using a 252-day year here would overstate annualised volatility by ~20% and
# quietly hand every instrument more leverage than the target asked for.
SECONDS_PER_YEAR = Decimal(365 * 24 * 3600)

# How much of the liquidation distance the hard stop is allowed to occupy. At 1.0
# the stop and the liquidation price coincide, which is not a stop - it is a race
# between two events on the same tick, decided by the venue rather than by us.
STOP_HEADROOM_FRACTION = Decimal("0.8")

VOLATILITY_TARGETED = "volatility_targeted"
FIXED = "fixed"

STOP_OUTSIDE_LIQUIDATION = "STOP_OUTSIDE_LIQUIDATION"


@dataclass(frozen=True)
class LeverageChoice:
    """The multiple, and everything needed to argue with it later."""

    leverage: Decimal
    rule: str
    reason: str
    reduced_from: Decimal | None = None
    refusal: str | None = None
    evidence: dict | None = None

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def as_dict(self) -> dict:
        return {
            "leverage": str(self.leverage),
            "leverage_rule": self.rule,
            "leverage_reason": self.reason,
            "leverage_reduced_from": (None if self.reduced_from is None
                                      else str(self.reduced_from)),
            "leverage_refusal": self.refusal,
            "leverage_evidence": self.evidence or {},
        }


def annualise_window_volatility(window_volatility, window_ns: int) -> Decimal | None:
    """A window's movement, expressed as an annual standard deviation.

    Square-root-of-time scaling: sigma over `window` seconds becomes sigma times
    the square root of how many such windows fit in a year. Returns `None` when
    there is nothing to scale, so a caller can tell "no estimate" apart from an
    estimate of zero - a zero would divide into infinite leverage.
    """
    if window_volatility is None or window_ns is None or window_ns <= 0:
        return None
    volatility = Decimal(str(window_volatility))
    if volatility <= 0:
        return None
    windows_per_year = SECONDS_PER_YEAR / (Decimal(window_ns) / Decimal(1_000_000_000))
    return volatility * Decimal(str(math.sqrt(float(windows_per_year))))


def choose_leverage(*, declaration, segment: str, window_volatility=None,
                    window_ns: int = 0) -> LeverageChoice:
    """The multiple for one trade, before the stop is checked against it."""
    floor = declaration.leverage_floor
    ceiling = declaration.ceiling_for(segment)

    if declaration.leverage_rule == FIXED:
        return LeverageChoice(
            leverage=ceiling, rule=FIXED,
            reason="declared fixed at the segment ceiling",
            evidence={"ceiling": str(ceiling)})

    annual = annualise_window_volatility(window_volatility, window_ns)
    if annual is None:
        # **Unknown risk is not an argument for more leverage.** A symbol with too
        # few observations to estimate volatility gets the FLOOR, and the reason is
        # journalled. Defaulting to the ceiling here would hand the most leverage
        # to the instruments least understood, which is the exact inversion of what
        # the rule is for.
        return LeverageChoice(
            leverage=floor, rule=VOLATILITY_TARGETED,
            reason="no volatility estimate; floored rather than assumed",
            evidence={"window_volatility": None, "window_ns": window_ns})

    annual_pct = annual * 100
    raw = declaration.target_annual_vol_pct / annual_pct
    leverage = max(floor, min(ceiling, raw))
    return LeverageChoice(
        leverage=leverage, rule=VOLATILITY_TARGETED,
        reason="target annual volatility divided by this instrument's own",
        evidence={"annualised_volatility_pct": str(round(annual_pct, 4)),
                  "target_annual_vol_pct": str(declaration.target_annual_vol_pct),
                  "unclamped": str(round(raw, 4)),
                  "floor": str(floor), "ceiling": str(ceiling)})


def liquidation_distance(leverage: Decimal, maintenance_margin_rate: Decimal) -> Decimal:
    """The adverse move, as a fraction of entry, that wipes an isolated position."""
    if leverage <= 0:
        return Decimal(1)
    return Decimal(1) / leverage - maintenance_margin_rate


def fit_leverage_to_stop(choice: LeverageChoice, *, stop_fraction: Decimal,
                         declaration) -> LeverageChoice:
    """Reduce the multiple until the hard stop fits inside liquidation, or refuse.

    Reducing rather than refusing outright is the whole point. A 4% stop is fine at
    3x and impossible at 20x, so refusing the trade would be discarding a position
    that the declared risk appetite has no problem with - it is the LEVERAGE that
    did not fit, not the trade.

    Both the reduction and the refusal carry their numbers, because "refused" with
    no figures is a message that sends the reader to guess which dial to turn.
    """
    maintenance = declaration.maintenance_margin_rate
    floor = declaration.leverage_floor
    original = choice.leverage

    leverage = original
    while leverage >= floor:
        allowed = liquidation_distance(leverage, maintenance) * STOP_HEADROOM_FRACTION
        if stop_fraction < allowed:
            if leverage == original:
                return choice
            return LeverageChoice(
                leverage=leverage, rule=choice.rule,
                reason=(f"reduced from {original} so the {stop_fraction} hard stop "
                        f"sits inside the liquidation distance"),
                reduced_from=original,
                evidence={**(choice.evidence or {}),
                          "stop_fraction": str(stop_fraction),
                          "allowed_stop_fraction": str(round(allowed, 8)),
                          "maintenance_margin_rate": str(maintenance)})
        # Step down a whole multiple. Fractional leverage is not a thing a venue
        # offers, and searching a continuum would produce a number no exchange
        # would accept.
        leverage = (leverage - 1).quantize(Decimal("1"))

    allowed_at_floor = liquidation_distance(floor, maintenance) * STOP_HEADROOM_FRACTION
    return LeverageChoice(
        leverage=floor, rule=choice.rule,
        reason="the hard stop does not fit even at the leverage floor",
        reduced_from=original, refusal=STOP_OUTSIDE_LIQUIDATION,
        evidence={**(choice.evidence or {}),
                  "stop_fraction": str(stop_fraction),
                  "allowed_stop_fraction_at_floor": str(round(allowed_at_floor, 8)),
                  "floor": str(floor),
                  "maintenance_margin_rate": str(maintenance)})


def borrow_interest(*, notional_usdt: Decimal, margin_usdt: Decimal,
                    annual_pct: Decimal, held_ns: int) -> Decimal:
    """What the borrowed part of a leveraged spot position cost while it was held.

    Charged on `notional - margin`, which is the money that was actually borrowed;
    charging it on the whole notional would bill the trader interest on their own
    capital. An unleveraged position borrows nothing and pays nothing, so this
    returns exactly zero rather than a small number that looks like a rounding
    artefact.
    """
    borrowed = notional_usdt - margin_usdt
    if borrowed <= 0 or held_ns <= 0 or annual_pct <= 0:
        return Decimal(0)
    years = Decimal(held_ns) / Decimal(1_000_000_000) / SECONDS_PER_YEAR
    return borrowed * (annual_pct / 100) * years
