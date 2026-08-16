"""The pre-trade gate: every order passes here or dies here, before it is sent.

`FEATURES.md` §6 (P0): *"Pre-trade gate: notional, leverage, position cap —
blocks the order **before** it is sent"*. Ledger RX-001 and RX-037, the second
of which carries the SEC Rule 15c3-5 anchor for the same bundle.

## The concrete gap this closes

The forward engine buys on every bar of every symbol with no bound at all. On
2026-08-16 that was **363 open positions and 14,107 fills**, opened by a strategy
that makes no edge claim. Nothing anywhere said no.

## The donor was read, and it does not do what the ledger says it does

`CLAUDE.md` requires reading donor code rather than the ledger's note about it,
and this is the third time that has mattered. RX-001 records *"prior-art
available to port: `pre_trade_risk_gate.py` (nse-botonly)"*. Fetched from GitHub
2026-08-16, it is an **NSE options and equity** gate: it rejects undefined-risk
option combinations, refuses fresh positions on the F&O ban list, and sizes lots
against an intraday cash-margin fraction. It contains **no notional cap, no
position cap, no order-rate limit, no price collar, no leverage cap and no
aggregate exposure limit** — none of the controls the row it is attached to
describes.

So the port is DECLINED and the reason is written down. What is adopted is its
**shape**, which is good and is the part worth taking:

* an explicit decision object rather than a boolean, carrying machine-readable
  rejection reasons so a board can report what was blocked and why;
* **refuse, never round.** Its comment — *"a trade the risk budget can't afford
  at least 1 share/lot of is rejected, not rounded up"* — is the right rule and
  it generalises: a silently trimmed order is a different order from the one the
  strategy asked for, and the strategy's expectancy was computed on the size it
  asked for.

## A cap never blocks an order that reduces exposure

Found by a test rather than by reasoning. Applying the position caps to every
order meant a position **already over its limit could never be reduced** — the
book trapped at its own cap, in the one state where an exit matters most. A cap
exists to stop exposure growing, not to stop it shrinking, so an order that moves
toward compliance passes even when the result is still over the line.

The order-size cap and the rate limit still apply to reducing orders, because
those are about the *order* rather than the position: a fat-fingered exit is
still a fat finger, and a runaway loop that only sells is still a runaway loop.

## Every failing rule is reported, not just the first

A gate that returns on the first violation makes a caller fix one thing,
resubmit, and discover the next — which on a rate limit means a caller learns
about it one rejected order at a time. All applicable rules are evaluated and
every breach is named.

## What this gate does NOT do, and where those live instead

* **Daily loss and drawdown kills** belong to `risk.tail_cap`, which reads a
  ceiling the user owns and that no code here may raise. Duplicating them would
  put the §6 limit in two places, and the copy that drifts is the one nobody
  re-derived.
* **Leverage** needs margin state from the venue, which this system does not
  have. Declared absent rather than approximated: a leverage check computed from
  a number nobody fetched is a check that passes.
* **Correlation-aware limits** need `features.beta_to_btc` wired to a portfolio,
  which does not exist. RX-001 names them and they are not here.
* **The price collar** applies only to limit orders. A market order carries no
  price to band, so the fat-finger guard for one is `max_order_notional` — stated
  because a collar that silently skips every market order is a control that
  reads as active and is not.

## NAV is supplied and never defaulted

Every fractional limit is a fraction of net asset value, and a gate that invented
a NAV would compute fractions of a number nobody set. `NavNotSupplied` refuses
rather than assuming, in the same posture `risk.tail_cap.read_ceiling` takes to a
missing ceiling: a missing limit is a question nobody answered, not a permission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path

BUY = "BUY"
SELL = "SELL"

# Where the limits live. A file the user owns, read here and written only by
# `seed_gate_limits`, which refuses to overwrite - the same shape
# `risk.tail_cap` uses for the §6 ceiling, and for the same reason: these are
# decisions about capital, and a decision about capital does not belong buried
# in a shell script's arguments where nobody reads it again.
LIMITS_FILE = "paper-gate.json"

# A PROPOSAL, seeded so the gate can run at all, and not a number the system
# decided. `nav` here is a declared paper book size rather than real capital -
# paper is for practising, so the figure exists to make the fractions mean
# something, and it is in the file precisely so it can be argued with.
PROPOSED_LIMITS = {
    "nav": "10000",
    "max_order_notional": "500",
    "max_position_notional": "1000",
    "max_position_nav_fraction": "0.10",
    "max_aggregate_nav_fraction": "0.50",
    "max_open_instruments": "25",
    "max_orders_per_window": "120",
    "order_rate_window_ns": str(60 * 1_000_000_000),
    "price_collar_fraction": "0.05",
}


class RejectionReason(str, Enum):
    """Machine-readable, so a board can report what was blocked and why."""
    ORDER_NOTIONAL_TOO_LARGE = "order_notional_too_large"
    POSITION_CAP_EXCEEDED = "position_cap_exceeded"
    POSITION_NAV_FRACTION_EXCEEDED = "position_nav_fraction_exceeded"
    AGGREGATE_EXPOSURE_EXCEEDED = "aggregate_exposure_exceeded"
    TOO_MANY_OPEN_INSTRUMENTS = "too_many_open_instruments"
    ORDER_RATE_EXCEEDED = "order_rate_exceeded"
    PRICE_OUTSIDE_COLLAR = "price_outside_collar"
    NO_REFERENCE_PRICE = "no_reference_price"


class LimitsNotSet(RuntimeError):
    """No gate limits on disk.

    Raised rather than defaulted so a caller decides between running gated and
    running ungated, out loud. `seed_gate_limits` writes the proposal; something
    has to call it deliberately.
    """


class NavNotSupplied(ValueError):
    """No net asset value to compute fractional limits against.

    Refused rather than defaulted. A gate that invents a NAV computes fractions
    of a number nobody set, and every limit derived from it is a number that
    looks measured and is not.
    """


@dataclass(frozen=True)
class GateLimits:
    """The bundle. Every value is declared by whoever owns the capital.

    Fractions rather than absolutes wherever the quantity scales with the book,
    per `FEATURES.md` §4's rule that a cap is never an absolute - a limit in
    dollars means something different at every account size, and the one that
    was right last month silently is not.
    """
    max_order_notional: Decimal
    max_position_notional: Decimal
    max_position_nav_fraction: Decimal
    max_aggregate_nav_fraction: Decimal
    max_open_instruments: int
    max_orders_per_window: int
    order_rate_window_ns: int
    # How far a LIMIT price may sit from the reference before it is a fat finger.
    # Does not apply to market orders - see the module docstring.
    price_collar_fraction: Decimal


@dataclass(frozen=True)
class GateDecision:
    """Approve or refuse, with every reason and the numbers behind them.

    `approved_quantity` equals the requested quantity or is zero. It is never a
    reduced size: a silently trimmed order is a different order from the one the
    strategy asked for, and the strategy's expectancy was computed on the size it
    asked for.
    """
    approved: bool
    reasons: tuple[RejectionReason, ...]
    approved_quantity: Decimal
    order_notional: Decimal
    resulting_position_notional: Decimal
    detail: str

    def describe(self) -> str:
        if self.approved:
            return (f"approved {self.approved_quantity} "
                    f"({self.order_notional:,.2f} notional)")
        return f"REFUSED: {', '.join(r.value for r in self.reasons)}. {self.detail}"


@dataclass
class OrderRateWindow:
    """A sliding count of orders, per key and in aggregate.

    A rate limit is a control against a runaway loop, and a runaway loop is the
    failure mode a paper engine reaching a broken tape actually has. Counted per
    instrument AND overall, because a loop that hits one symbol and a loop that
    sprays the universe are the same defect wearing different shapes.
    """
    window_ns: int
    _stamps: dict[str, list[int]] = field(default_factory=dict)

    def record(self, key: str, now_ns: int) -> None:
        stamps = self._stamps.setdefault(key, [])
        stamps.append(int(now_ns))
        self._evict(key, now_ns)

    def count(self, key: str, now_ns: int) -> int:
        self._evict(key, now_ns)
        return len(self._stamps.get(key, []))

    def _evict(self, key: str, now_ns: int) -> None:
        cutoff = int(now_ns) - self.window_ns
        stamps = self._stamps.get(key)
        if stamps is None:
            return
        self._stamps[key] = [s for s in stamps if s > cutoff]


class PreTradeGate:
    """Every order passes here or dies here.

    Consulted BEFORE the intent WAL, so a refused order is never written as
    something that was sent - `FEATURES.md` §6's own words are "blocks the order
    *before* it is sent", and an order recorded as intended and then blocked is a
    different claim from one that never existed.
    """

    def __init__(self, limits: GateLimits) -> None:
        self._limits = limits
        self._rate = OrderRateWindow(window_ns=limits.order_rate_window_ns)
        self.refusals_by_reason: dict[str, int] = {}
        self.approved = 0

    def evaluate(self, *, venue: str, symbol: str, side: str,
                 quantity: Decimal, reference_price: Decimal | None,
                 limit_price: Decimal | None, nav: Decimal | None,
                 open_positions: dict[tuple[str, str], Decimal],
                 now_ns: int) -> GateDecision:
        """Judge one order against every applicable rule.

        `open_positions` maps (venue, symbol) to signed quantity, and
        `reference_price` is what the instrument last printed - supplied by the
        caller because this module reads no market data, which is what keeps it
        a gate rather than a second view of the tape.
        """
        if nav is None:
            raise NavNotSupplied(
                "no NAV supplied. Every fractional limit here is a fraction of "
                "it, and a gate that invented one would compute fractions of a "
                "number nobody set")

        reasons: list[RejectionReason] = []
        notes: list[str] = []

        if reference_price is None or reference_price <= 0:
            # Without a price there is no notional, and every remaining rule is
            # about notional. Refused rather than skipped: a gate that waves an
            # order through because it could not price it is not a gate.
            return self._refuse(
                [RejectionReason.NO_REFERENCE_PRICE], quantity, Decimal(0),
                Decimal(0),
                "no reference price, so no rule that depends on notional could "
                "be applied - refused rather than waved through")

        order_notional = quantity * reference_price
        held = open_positions.get((venue, symbol), Decimal(0))
        direction = Decimal(1) if side.upper() == BUY else Decimal(-1)
        resulting = abs(held + direction * quantity) * reference_price
        # Does this order make the position BIGGER? Found by a test: applying
        # the caps to every order meant a position already over its limit could
        # never be reduced - the book would be trapped at its own cap, in the one
        # state where an exit matters most. A cap exists to stop exposure
        # growing, not to stop it shrinking, so an order that moves toward
        # compliance is never blocked by one even when the result is still over.
        increases = abs(held + direction * quantity) > abs(held)
        aggregate = sum(
            (abs(q) * reference_price for (v, s), q in open_positions.items()
             if q != 0), start=Decimal(0))

        limits = self._limits
        if order_notional > limits.max_order_notional:
            reasons.append(RejectionReason.ORDER_NOTIONAL_TOO_LARGE)
            notes.append(f"order {order_notional:,.2f} > "
                         f"{limits.max_order_notional:,.2f}")
        if increases and resulting > limits.max_position_notional:
            reasons.append(RejectionReason.POSITION_CAP_EXCEEDED)
            notes.append(f"position would reach {resulting:,.2f} > "
                         f"{limits.max_position_notional:,.2f}")
        if increases and resulting > nav * limits.max_position_nav_fraction:
            reasons.append(RejectionReason.POSITION_NAV_FRACTION_EXCEEDED)
            notes.append(f"position would reach "
                         f"{resulting / nav:.1%} of NAV > "
                         f"{limits.max_position_nav_fraction:.1%}")
        if (increases
                and aggregate + order_notional
                > nav * limits.max_aggregate_nav_fraction):
            reasons.append(RejectionReason.AGGREGATE_EXPOSURE_EXCEEDED)
            notes.append(f"aggregate would reach "
                         f"{(aggregate + order_notional) / nav:.1%} of NAV > "
                         f"{limits.max_aggregate_nav_fraction:.1%}")

        # Likewise breadth: adding an instrument is what the cap is about, and a
        # full book must still be able to manage what it already holds.
        opening_new = held == 0
        open_count = sum(1 for q in open_positions.values() if q != 0)
        if opening_new and open_count >= limits.max_open_instruments:
            reasons.append(RejectionReason.TOO_MANY_OPEN_INSTRUMENTS)
            notes.append(f"{open_count} instrument(s) already open >= "
                         f"{limits.max_open_instruments}")

        for key in (f"{venue}:{symbol}", "*"):
            if self._rate.count(key, now_ns) >= limits.max_orders_per_window:
                reasons.append(RejectionReason.ORDER_RATE_EXCEEDED)
                notes.append(f"{limits.max_orders_per_window} order(s) already "
                             f"sent for {key} inside the window")
                break

        # The collar applies to LIMIT orders only. A market order has no price to
        # band, and its fat-finger guard is `max_order_notional` above.
        if limit_price is not None:
            drift = abs(limit_price - reference_price) / reference_price
            if drift > limits.price_collar_fraction:
                reasons.append(RejectionReason.PRICE_OUTSIDE_COLLAR)
                notes.append(f"limit {limit_price} is {drift:.2%} from the "
                             f"reference > {limits.price_collar_fraction:.2%}")

        if reasons:
            return self._refuse(reasons, quantity, order_notional, resulting,
                                "; ".join(notes))

        self._rate.record(f"{venue}:{symbol}", now_ns)
        self._rate.record("*", now_ns)
        self.approved += 1
        return GateDecision(
            approved=True, reasons=(), approved_quantity=quantity,
            order_notional=order_notional, resulting_position_notional=resulting,
            detail="within every limit")

    def _refuse(self, reasons, quantity, order_notional, resulting,
                detail) -> GateDecision:
        for reason in reasons:
            self.refusals_by_reason[reason.value] = (
                self.refusals_by_reason.get(reason.value, 0) + 1)
        return GateDecision(
            approved=False, reasons=tuple(reasons),
            # Zero, never a reduced size. See `GateDecision`.
            approved_quantity=Decimal(0), order_notional=order_notional,
            resulting_position_notional=resulting, detail=detail)

    def describe(self) -> str:
        if not self.refusals_by_reason:
            return f"{self.approved} order(s) approved, none refused"
        refusals = ", ".join(f"{reason} {count}" for reason, count
                             in sorted(self.refusals_by_reason.items(),
                                       key=lambda kv: -kv[1]))
        return (f"{self.approved} approved, "
                f"{sum(self.refusals_by_reason.values())} refused ({refusals})")


def seed_gate_limits(root: Path, limits: dict | None = None) -> Path:
    """Write the proposed limits once, and never overwrite ones that exist.

    Refusing to overwrite is the point, exactly as in `risk.tail_cap`: no code
    path in this system may widen its own limit. Editing the file by hand is how
    it changes, which keeps the decision with whoever owns the capital.
    """
    path = Path(root) / LIMITS_FILE
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(PROPOSED_LIMITS)
    body.update({k: str(v) for k, v in (limits or {}).items()})
    body["_note"] = (
        "Pre-trade gate limits (FEATURES.md §6, ledger RX-001/RX-037). Seeded "
        "2026-08-16 as a PROPOSAL, not a decision - `nav` is a declared paper "
        "book size rather than real capital. No code in this system raises "
        "these; edit the file. Daily-loss and drawdown kills are NOT here: they "
        "belong to risk/tail-cap.json.")
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def read_gate_limits(root: Path) -> tuple[GateLimits, Decimal]:
    """The limits and the NAV they are fractions of, or `LimitsNotSet`."""
    path = Path(root) / LIMITS_FILE
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LimitsNotSet(
            f"no pre-trade gate limits at {path}. Seed the proposal with "
            f"`risk.pre_trade_gate.seed_gate_limits` and edit it, or run "
            f"ungated deliberately - but not by accident") from None
    except (OSError, ValueError) as exc:
        raise LimitsNotSet(f"gate limits at {path} are unreadable: {exc}") from exc

    try:
        limits = GateLimits(
            max_order_notional=Decimal(str(body["max_order_notional"])),
            max_position_notional=Decimal(str(body["max_position_notional"])),
            max_position_nav_fraction=Decimal(
                str(body["max_position_nav_fraction"])),
            max_aggregate_nav_fraction=Decimal(
                str(body["max_aggregate_nav_fraction"])),
            max_open_instruments=int(body["max_open_instruments"]),
            max_orders_per_window=int(body["max_orders_per_window"]),
            order_rate_window_ns=int(body["order_rate_window_ns"]),
            price_collar_fraction=Decimal(str(body["price_collar_fraction"])))
        nav = Decimal(str(body["nav"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LimitsNotSet(f"gate limits at {path} are malformed: {exc}") from exc

    if nav <= 0:
        raise LimitsNotSet(
            f"gate limits at {path} declare a non-positive NAV, which would "
            f"make every fractional limit zero and refuse the first order")
    return limits, nav
