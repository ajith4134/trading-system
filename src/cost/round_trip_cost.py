"""The reality filter: what a round trip costs, or why it cannot be priced.

`ARCHITECTURE.md` Layer 1 states the reason this exists before any strategy
does: **fees dominate breakeven by roughly 5-10× over slippage and adverse
selection at this size.** Without a cost engine gating signals, paper mode
manufactures edge that evaporates live.

Two properties carry the design.

**A refusal is a value, not an exception.** `CostRefused` is returned alongside
`CostQuote`, has no `breakeven_bps`, and will not coerce to a number. A caller
cannot accidentally read "we do not know" as "zero", which is the specific way
a cost gate stops being a gate.

**Provenance survives the arithmetic.** Every component names where it came
from, how old it is, and whether anyone verified it. A quote resting on a fee
someone typed off a web page reports `is_verified = False` all the way out —
Rule 8 applied to a number: a quote is a measurement with provenance, not a
figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from cost.fee_schedule import DECLARED_SCHEDULES, FeeSchedule
from cost.funding_carry import (
    NoFundingAvailable, funding_cost_bps, load_funding_rates_as_of,
    settlements_between,
)

_ORDER_TYPES = ("maker", "taker")


@dataclass(frozen=True)
class CostInput:
    """One thing a quote rests on, and how much it should be trusted."""

    name: str                   # "fee", "spread", "impact", "funding"
    detail: str                 # the endpoint, document or dataset it came from
    verified: bool              # fetched from the venue, not transcribed by a human
    age_ns: int | None          # None when nothing was fetched to be old


@dataclass(frozen=True)
class CostQuote:
    """What a round trip costs, itemised, with every source named."""

    venue: str
    symbol: str
    notional: Decimal
    fee_bps: Decimal
    spread_bps: Decimal
    impact_bps: Decimal
    funding_bps: Decimal
    inputs: tuple[CostInput, ...]

    @property
    def breakeven_bps(self) -> Decimal:
        """What the edge has to clear before the trade makes money."""
        return self.fee_bps + self.spread_bps + self.impact_bps + self.funding_bps

    @property
    def is_verified(self) -> bool:
        """False if *any* input was not fetched from the venue.

        Deliberately pessimistic: a quote is only as trustworthy as its least
        trustworthy component, and averaging confidence is how an unverified
        number becomes a trusted one.
        """
        return all(source.verified for source in self.inputs)


@dataclass(frozen=True)
class CostRefused:
    """The trade cannot be priced, and why.

    No `breakeven_bps`, and `__float__` raises. Both are deliberate: the failure
    this type exists to prevent is a caller reaching for a number on a refusal
    and finding one.
    """

    venue: str
    symbol: str
    reason: str
    missing: str                # which input could not be established

    def __float__(self):
        raise TypeError(
            f"a refusal is not a cost: {self.reason}. Handle CostRefused "
            f"explicitly; there is no numeric fallback by design")


def quote_round_trip_cost(venue: str, symbol: str, notional: Decimal, *,
                          order_type: str, at_ns: int,
                          instrument_kind: str = "perp",
                          holding_ns: int = 0,
                          side: str = "long",
                          schedule: FeeSchedule | None = None,
                          store_root: Path | None = None) -> CostQuote | CostRefused:
    """Price both legs, or refuse and name what was missing.

    Both legs, because that is the only number that decides whether an edge is
    real — a strategy pays to get in and pays again to get out.

    Spread and impact are currently zero **and reported as unverified inputs**,
    not silently omitted: Layer 1 holds no book dataset, so there is nothing to
    price them from. That understates cost, which is the dangerous direction, so
    it is stated in the provenance rather than hidden in the total. When the
    book dataset exists this is where it plugs in.
    """
    if order_type not in _ORDER_TYPES:
        raise ValueError(f"order_type must be one of {_ORDER_TYPES}, got {order_type!r}")

    # A caller that has fetched a schedule passes it in, which is how the one
    # venue with a public schedule (Hyperliquid) produces a verified quote at
    # all. Falling back to the declared table keeps every other venue honest
    # about resting on a number a human transcribed.
    schedule = schedule or _fee_schedule(venue, instrument_kind)
    if schedule is None:
        return CostRefused(
            venue=venue, symbol=symbol, missing="fee",
            reason=(f"no fee schedule for {venue}:{instrument_kind}. Refusing "
                    f"rather than borrowing another venue's rates - venue "
                    f"selection outranks every execution algorithm at this size"))

    maker = order_type == "maker"
    fee_bps = schedule.round_trip_bps(maker_in=maker, maker_out=maker)

    funding_bps = Decimal(0)
    funding_input = None
    if holding_ns > 0:
        try:
            settlements = settlements_between(venue, at_ns, at_ns + holding_ns)
        except NoFundingAvailable as exc:
            return CostRefused(venue=venue, symbol=symbol, missing="funding",
                               reason=str(exc))
        if settlements:
            # Held across a settlement, so funding is a real cost line and
            # cannot be skipped. Charging zero here would understate exactly
            # the family the prime directive rests on.
            try:
                rates = load_funding_rates_as_of(
                    store_root or Path.home() / "capture" / "store",
                    venue, symbol, at_ns, at_ns + holding_ns)
            except NoFundingAvailable as exc:
                return CostRefused(venue=venue, symbol=symbol, missing="funding",
                                   reason=str(exc))
            # Assigned, not merely computed. An earlier version loaded the
            # rates and dropped them on the floor, so every carry priced as
            # though funding were free - the one cost line that decides
            # whether a carry trade is worth doing at all.
            funding_bps = funding_cost_bps(rates, side=side)
            funding_input = CostInput(
                name="funding", verified=True, age_ns=None,
                detail=f"{len(rates)} archived settlement rate(s) from the "
                       f"clock-gated funding dataset")

    inputs = [
        CostInput(name="fee", detail=schedule.source_detail,
                  verified=schedule.is_verified, age_ns=(
                      None if schedule.fetched_at_ns is None
                      else at_ns - schedule.fetched_at_ns)),
        CostInput(name="spread", verified=False, age_ns=None,
                  detail="no book dataset in Layer 1; charged as 0 and flagged "
                         "unverified, which understates cost"),
        CostInput(name="impact", verified=False, age_ns=None,
                  detail="no book dataset in Layer 1; charged as 0 and flagged "
                         "unverified, which understates cost"),
    ]
    if funding_input is not None:
        inputs.append(funding_input)

    return CostQuote(venue=venue, symbol=symbol, notional=notional,
                     fee_bps=fee_bps, spread_bps=Decimal(0),
                     impact_bps=Decimal(0), funding_bps=funding_bps,
                     inputs=tuple(inputs))


def is_signal_viable(expected_edge_bps: Decimal,
                     quote: CostQuote | CostRefused) -> bool:
    """The one call a strategy makes before a signal may become an order.

    A refusal is never viable, however large the claimed edge. That single line
    is what makes this a gate rather than a suggestion: an unpriceable trade is
    not a profitable trade, and a strategy confident enough to override that is
    exactly the strategy the gate is for.
    """
    if isinstance(quote, CostRefused):
        return False
    return Decimal(expected_edge_bps) > quote.breakeven_bps


def _fee_schedule(venue: str, instrument_kind: str) -> FeeSchedule | None:
    return DECLARED_SCHEDULES.get((venue, instrument_kind))
