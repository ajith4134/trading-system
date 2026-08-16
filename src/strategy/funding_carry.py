"""Delta-hedged funding carry: one expression, every symbol, default stand aside.

`FEATURES.md` §4 (P1) — *"Funding-rate carry — **start here**, latency-immune,
viable at this size"*. Ledger SP-056. Spec:
`docs/superpowers/specs/2026-08-09-phase-4-carry-first-pass.md`, whose §11 Q2 the
user answered on 2026-08-16 by not narrowing the watch list: **delta-hedged**,
accepting that far fewer symbols clear a doubled cost gate, with universe-wide
observation kept separate in `features.universe_coverage`.

## One hypothesis with N samples, not N hypotheses

Goal-doc §5a.5, the sentence this module is built around:

> Setup definitions are universal and parameter-free across symbols. Per-symbol
> variation comes only from *normalisation* — z-scores or percentiles against
> that symbol's own history — never from fitted per-symbol parameters. One setup
> on 1,290 symbols is **one hypothesis with 1,290 samples**. It becomes 1,290
> hypotheses the moment per-symbol tuning is allowed. This is the single most
> dangerous thing that could be implemented here.

So there is exactly one expression here and it is evaluated identically on every
candidate. The only per-symbol quantity is the funding percentile, which comes
from `features.funding_basis` and is a rank against that symbol's own visible
history. `test_no_per_symbol_parameter_exists` pins the absence.

## The gate compares carry EARNED to cost PAID, over a declared holding period

This is the defect the first shipped version had, and it is worth stating in
full because it is the exact shape this project exists to catch.

That version compared the **annualised** carry against the **one-off** round-trip
cost: `1095 bps - 24 bps = +1071 bps, PASS`. But an annualised rate is only
earned by holding for a year. Over one 8-hour settlement the same trade earns
**1 bp** and pays **24 bps** — a 23 bp loss — and it needs **24 settlements, 8
days**, merely to break even. The gate passed trades that lose money on every
realistic holding period, and the output looked outstanding.

So the gate now uses the carry **earned over `holding_settlements`**, which is a
declared property of the trade rather than a unit conversion. `FEATURES.md`'s own
spec §4 step 3 states the rule this restores: *"expected carry per settlement
must exceed `quote_round_trip_cost` for the round trip"*.

`annualised_carry_bps` is still reported, because it is the figure that compares
two instruments with different settlement schedules — but it is **not** the gate
input, and `CarryProposal` keeps the two in separate fields so they cannot be
confused again.

## The carry number is a FORECAST, and it is labelled as one

`expected_carry_bps` is the latest observed funding rate over the holding period.
The expectation embedded in that is *"the rate at every settlement in the hold
equals the rate we last saw"*, which is a random walk on funding and is not a
measurement of anything.

It is named `expected_` rather than `realised_`, `CarryProposal` carries
`rate_observed_at_ns` so the age of that assumption is visible, and the module
reports it as a forecast in `describe()`. A carry table that presented this as
measured would be the most convincing wrong number available here — funding is
persistent, so the forecast is usually nearly right, and "usually nearly right"
is exactly the property that stops anyone checking.

## The cost gate prices BOTH legs, because the trade has two

A delta-hedged carry is short the perp and long the spot. It pays to enter and
exit **twice**, and funding is charged on the perp leg across every settlement
held. Pricing one leg is the flattering error and it is roughly a factor of two:
`quote_round_trip_cost` is called for the perp and for the spot, and the gate
compares the carry against their **sum**.

`is_signal_viable` is what makes this a gate rather than a suggestion — a refused
quote is never viable however large the claimed carry, because an unpriceable
trade is not a profitable trade.

## The acceptance threshold rises with opportunity flow

§5a.5 again: the more symbols qualifying, the pickier each must be. A fixed
threshold takes the same trade whether it is one of three opportunities or one of
three hundred, which spends the book on the first thing that clears.

The rule here has one declared number — `capacity`, how many positions the book
can hold — and no fitted ones. The threshold is the **capacity-th best** net
carry among everything that cleared the cost gate:

* 3 candidates, capacity 10 → the threshold is just the cost gate. Take them all.
* 300 candidates, capacity 10 → the threshold is the 10th best, and 290 proposals
  that would each have been accepted on a quiet day are declined today.

**Ties are not broken here.** On the first live run the threshold landed on 1071
bps and 58 symbols cleared it, because 48 of them sit at binance's 1 bp default
funding rate — identical opportunities that no ranking can separate. Choosing
between them would be inventing a preference in a module whose whole contract is
that it proposes and the arbiter selects. `over_capacity` reports the overflow
instead, and it is a finding in its own right: a screen whose acceptance
threshold lands on the venue's default rate is ranking noise, not carry.

`capacity` is a portfolio constraint rather than a tuned number, and it lands in
the Trial Registry with every run so a search over it is counted.

## Default action is stand aside, and an empty result says why

`CarrySelection.declined` counts every candidate that fell out and at which
stage — not dollar-quoted, no spot leg, no funding rank, cost refused, below the
cost gate, below the flow threshold. An empty proposal list from a quiet market
and one from a broken funding feed are the same empty list, and the counts are
the only thing that separates them.

## It proposes. It does not size, and it does not order.

No notional, no leverage, no order. `ARCHITECTURE.md` puts trade selection in the
arbiter and sizing behind the risk gate, and `notional` here is an *input* to the
cost quote — the size at which the question was asked — not a position. A
strategy that sized its own positions would put the tail cap's job inside the
thing the tail cap exists to bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from cost.round_trip_cost import CostRefused, quote_round_trip_cost
from features.funding_basis import compute_funding_basis
from validation.trial_registry import TrialRegistry, TrialSpec

FAMILY = "carry"

# The hedge pairing, and a vocabulary collision this had to make explicit.
#
# "venue" means two different things in this repo and they do not match. The
# STORE's venue is a FEED - `binance` and `binance-spot` are separate capture
# processes writing separate rows. The FEE table's venue is an EXCHANGE, keyed
# `("binance", "perp")` and `("binance", "spot")`, with the instrument kind
# carrying the distinction the store puts in the venue name.
#
# Found by running this: every one of 868 binance perps declined `cost_refused`,
# because the spot leg asked `quote_round_trip_cost("binance-spot", ...)` and no
# such fee venue exists. Nothing was wrong with either table - they disagree
# about what a word means, and the failure was total and silent.
#
# So the pairing declares both halves: which FEED carries the spot leg, and which
# (exchange, instrument kind) prices each leg. An unlisted perp venue is refused
# rather than paired by guess - a hedge on the wrong exchange is not a hedge, and
# it would look like an ordinary position until the legs moved apart.
HEDGE_VENUES = {
    # perp feed -> (spot feed, perp fee key, spot fee key)
    "binance": ("binance-spot", ("binance", "perp"), ("binance", "spot")),
}

# The hedge is matched on EXACT ticker equality across that venue pair, never by
# stripping a suffix or splitting a symbol. Exact equality asserts only that two
# venues use the same string, which is checkable; a suffix rule would assert
# something about structure, which is what `features.term_structure` refuses to
# do and for the same reason. The residual risk is stated rather than hidden: if
# binance ever lists a spot ticker that is not the same asset as the perp of that
# name, this pairs them wrongly. The dollar-quote filter removes most of it,
# since the ticker encodes the quote asset.
_HEDGE_MATCH = "exact ticker equality across the declared venue pair"

# The notional the cost question is asked at. An INPUT to the quote, not a
# position: fees are per-notional and some are tiered, so the gate has to be
# asked at a size. Declared here rather than passed by every caller so that two
# runs are comparable, and recorded in the trial.
DEFAULT_QUOTE_NOTIONAL = Decimal("1000")

# Maker on both legs. Carry is latency-immune by construction - there is no race
# to be in it - so resting is the correct assumption, and taking would price a
# trade nobody has to do in a hurry as though they did.
DEFAULT_ORDER_TYPE = "maker"

# How many funding settlements the trade is assumed to be held across. ONE by
# default, which is the spec's own rule - "expected carry per settlement must
# exceed quote_round_trip_cost for the round trip" - and the strictest honest
# gate, because a trade that pays for its round trip in a single settlement is
# not exposed to the rate changing afterwards.
#
# It is a declared property of the trade, not a unit conversion, and it lands in
# the Trial Registry: raising it is a real loosening of the gate and has to be
# counted as the search it is.
DEFAULT_HOLDING_SETTLEMENTS = 1

_DECLINE_REASONS = ("no_funding_rank", "no_spot_leg", "unknown_perp_venue",
                    "not_dollar_quoted", "cost_refused", "below_cost_gate",
                    "below_flow_threshold")


class UniverseNotSupplied(ValueError):
    """No dollar-quoted symbol set for a venue this setup needs.

    Refused rather than defaulted. The filter is what establishes that a hedge
    leg exists; without it the setup proposes naked shorts and calls them hedged,
    which is the one error here that cannot be seen in the output.
    """


class UnknownPerpVenue(KeyError):
    """No spot venue declared for this perp venue.

    Refused rather than paired by guess. A hedge on the wrong exchange is not a
    hedge, and the failure would look like an ordinary position until the two
    legs moved apart.
    """


@dataclass(frozen=True)
class CarryProposal:
    """One symbol the setup would take, and everything the number rests on.

    `expected_carry_bps` is a FORECAST - the last observed rate, annualised. The
    expectation inside it is that the next settlement's rate equals the last one,
    which is a random walk on funding. `rate_observed_at_ns` is carried so the
    age of that assumption is visible rather than implied.
    """
    perp_venue: str
    spot_venue: str
    symbol: str
    # Carry EARNED over the holding period. This is the gate input.
    expected_carry_bps: Decimal
    # The same rate scaled to a year. Comparable across instruments with
    # different settlement schedules, and NOT the gate input - kept in its own
    # field so the two cannot be confused, which is how the first version passed
    # trades that lose money on every realistic hold.
    annualised_carry_bps: Decimal
    holding_settlements: int
    funding_rate_bps: Decimal
    funding_percentile: float
    rate_observed_at_ns: int
    perp_cost_bps: Decimal
    spot_cost_bps: Decimal
    net_carry_bps: Decimal
    observations: int

    @property
    def total_cost_bps(self) -> Decimal:
        return self.perp_cost_bps + self.spot_cost_bps

    def describe(self) -> str:
        return (f"{self.perp_venue}/{self.symbol}: forecast carry "
                f"{self.expected_carry_bps:.2f} bps over "
                f"{self.holding_settlements} settlement(s) (last rate "
                f"{self.funding_rate_bps:.3f} bps, its own p"
                f"{self.funding_percentile:.2f}; "
                f"{self.annualised_carry_bps:.0f} bps annualised, which is NOT "
                f"the gate) less {self.total_cost_bps:.1f} bps of two-leg round "
                f"trip = {self.net_carry_bps:.2f} bps net")


@dataclass(frozen=True)
class CarrySelection:
    """What the setup proposes, and what fell out at every stage.

    An empty `proposals` from a quiet market and one from a broken funding feed
    are the same empty list. `declined` is the only thing that separates them,
    which is why it counts by stage rather than in total.
    """
    proposals: list[CarryProposal]
    declined: dict[str, int]
    candidates: int
    cleared_cost_gate: int
    capacity: int
    flow_threshold_bps: Decimal | None
    # How many proposals exceed `capacity` because they TIE at the threshold.
    # Not broken here: this module proposes and the arbiter selects, so choosing
    # between identical opportunities would be inventing a preference. Reported
    # because a run that returns 58 proposals at capacity 10 has told the caller
    # something real - that the threshold landed on a value dozens of symbols
    # share, which on binance is the 1 bp default rate, and a screen whose top is
    # the default rate is ranking noise.
    over_capacity: int

    def describe(self) -> str:
        if not self.proposals:
            worst = (max(self.declined.items(), key=lambda kv: kv[1])
                     if any(self.declined.values()) else None)
            tail = (f"; the largest loss was {worst[0]} on {worst[1]}"
                    if worst else "")
            return (f"STAND ASIDE - {self.candidates} candidate(s), none "
                    f"proposed{tail}")
        threshold = ("the cost gate alone" if self.flow_threshold_bps is None
                     else f"{self.flow_threshold_bps:.1f} bps")
        tied = (f"; {self.over_capacity} of them TIE at the threshold, so the "
                f"book cannot be filled by carry alone - the arbiter has to "
                f"choose between identical opportunities"
                if self.over_capacity else "")
        return (f"{len(self.proposals)} proposal(s) of {self.candidates} "
                f"candidate(s); {self.cleared_cost_gate} cleared the two-leg "
                f"cost gate and the flow threshold was {threshold} at capacity "
                f"{self.capacity}{tied}")


def flow_threshold(net_carries: Sequence[Decimal],
                   capacity: int) -> Decimal | None:
    """The capacity-th best net carry, or None when flow is below capacity.

    This is §5a.5's *"the threshold must rise with opportunity flow"* made
    mechanical, with one declared number and no fitted ones. None means the
    market offered fewer opportunities than the book can hold, so the cost gate
    is the only threshold - which is the correct answer on a quiet day and is
    reported as such rather than as a threshold of zero.
    """
    if capacity <= 0:
        raise ValueError(
            f"capacity must be > 0, got {capacity}; a book that can hold nothing "
            f"declines everything, and calling that a threshold hides it")
    if len(net_carries) <= capacity:
        return None
    return sorted(net_carries, reverse=True)[capacity - 1]


def _required_venues() -> list[str]:
    """Every feed whose universe must be supplied before the setup may run."""
    venues = []
    for perp_venue, (spot_venue, _perp_fee, _spot_fee) in HEDGE_VENUES.items():
        venues += [perp_venue, spot_venue]
    return sorted(set(venues))


def _cost_bps(venue: str, symbol: str, notional: Decimal, at_ns: int,
              instrument_kind: str, holding_ns: int,
              store_root: Path) -> Decimal | None:
    """Breakeven bps for one leg, or None if the quote was refused."""
    quote = quote_round_trip_cost(
        venue, symbol, notional, order_type=DEFAULT_ORDER_TYPE, at_ns=at_ns,
        instrument_kind=instrument_kind, holding_ns=holding_ns,
        store_root=store_root)
    if isinstance(quote, CostRefused):
        return None
    return quote.breakeven_bps


def select(store_root: Path, as_of_ns: int, *, capacity: int,
           registry: TrialRegistry, trial_name: str,
           dollar_quoted: dict[str, frozenset[str]],
           notional: Decimal = DEFAULT_QUOTE_NOTIONAL,
           holding_settlements: int = DEFAULT_HOLDING_SETTLEMENTS,
           custodian=None) -> CarrySelection:
    """Evaluate the carry setup across the universe and propose what survives.

    Counted in the Trial Registry before it runs: §5a.5 says every scan counts,
    and a selection pass is a look at the data whatever it proposes. The count is
    what keeps a later deflated Sharpe honest about how many things were tried.

    `dollar_quoted` maps a FEED venue to the symbols known to be dollar-quoted at
    this clock, and it is **required**. An earlier version defaulted it to None
    and skipped the filter, which is fail-open in the worst place: the check that
    a hedge leg exists at all was silently not run, and the module happily
    proposed shorting microcap perps against a spot market that may not list
    them. The spec calls universe selection *"a hard requirement, not a detail"*
    and this is why.

    Passed in rather than fetched here because
    `store.quote_currency.dollar_quoted_symbols` reads the CAPTURE root and this
    module reads the STORE; a module reaching into both would be two dependencies
    pretending to be one.
    """
    missing = [venue for venue in _required_venues()
               if venue not in dollar_quoted]
    if missing:
        raise UniverseNotSupplied(
            f"no dollar-quoted symbol set for {', '.join(missing)}. Refused "
            f"rather than defaulted to 'everything': skipping the filter skips "
            f"the check that a hedge leg exists, and a carry proposal with no "
            f"spot leg is a naked short wearing a hedge's name")
    spec = TrialSpec(
        name=trial_name, family=FAMILY,
        params={"setup": "delta-hedged funding carry", "capacity": capacity,
                "notional": str(notional), "order_type": DEFAULT_ORDER_TYPE,
                "holding_settlements": holding_settlements,
                "hedge_match": _HEDGE_MATCH})

    def evaluate(_spec: TrialSpec) -> dict:
        selection = _run(store_root, as_of_ns, capacity, dollar_quoted,
                         notional, holding_settlements, custodian)
        return {
            "sharpe": None,
            "proposals": len(selection.proposals),
            "candidates": selection.candidates,
            "cleared_cost_gate": selection.cleared_cost_gate,
            "flow_threshold_bps": (str(selection.flow_threshold_bps)
                                   if selection.flow_threshold_bps is not None
                                   else None),
            "over_capacity": selection.over_capacity,
            "declined": selection.declined,
        }

    registry.evaluate(spec, evaluate)
    return _run(store_root, as_of_ns, capacity, dollar_quoted, notional,
                holding_settlements, custodian)


def _run(store_root: Path, as_of_ns: int, capacity: int,
         dollar_quoted: dict[str, frozenset[str]],
         notional: Decimal, holding_settlements: int,
         custodian) -> CarrySelection:
    store_root = Path(store_root)
    as_of_ns = int(as_of_ns)
    if holding_settlements < 1:
        raise ValueError(
            f"holding_settlements must be >= 1, got {holding_settlements}; a "
            f"trade held across no settlement earns no funding, and gating on "
            f"zero carry would pass everything")
    declined = {reason: 0 for reason in _DECLINE_REASONS}

    basis = compute_funding_basis(store_root, as_of_ns, custodian=custodian)
    rows = basis.rows
    if rows.empty:
        return CarrySelection(proposals=[], declined=declined, candidates=0,
                              cleared_cost_gate=0, capacity=capacity,
                              flow_threshold_bps=None, over_capacity=0)

    priced: list[CarryProposal] = []
    for row in rows.itertuples(index=False):
        perp_venue = str(row.venue)
        symbol = str(row.symbol)
        pairing = HEDGE_VENUES.get(perp_venue)
        if pairing is None:
            declined["unknown_perp_venue"] += 1
            continue
        spot_venue, perp_fee, spot_fee = pairing
        if symbol not in dollar_quoted.get(perp_venue, frozenset()):
            declined["not_dollar_quoted"] += 1
            continue
        if symbol not in dollar_quoted.get(spot_venue, frozenset()):
            # Listed as a dollar-quoted perp and not as a dollar-quoted spot.
            # There is no hedge leg to buy, so this is a naked short with a
            # hedge's name - which is exactly what the highest-carry rows of any
            # naive screen are, because that is what the funding is paying for.
            declined["no_spot_leg"] += 1
            continue

        # Priced against the FEE identity, not the feed name - see HEDGE_VENUES.
        # NO funding in the cost quote, on either leg, and this is a correction
        # rather than an omission. `quote_round_trip_cost` charges funding as a
        # COST for a long. This trade is SHORT the perp, so funding is its
        # REVENUE - it is the whole edge - and charging it in the quote would
        # subtract the edge from itself with the sign reversed. It also made the
        # selection unrunnable: the funding path re-reads the whole clock-gated
        # dataset per symbol, which is 1,100 full reads for one pass.
        perp_cost = _cost_bps(perp_fee[0], symbol, notional, as_of_ns,
                              perp_fee[1], 0, store_root)
        # Spot has no settlements at all, so the same zero here is the
        # ordinary case rather than a correction.
        spot_cost = _cost_bps(spot_fee[0], symbol, notional, as_of_ns,
                              spot_fee[1], 0, store_root)
        if perp_cost is None or spot_cost is None:
            declined["cost_refused"] += 1
            continue

        rate = Decimal(str(row.funding_rate_bps))
        # Carry EARNED over the hold, not the annualised rate. See the module
        # docstring: comparing an annual rate to a one-off cost passed trades
        # that lose 23 bps on every settlement they are actually held for.
        expected_carry = rate * Decimal(holding_settlements)
        net = expected_carry - (perp_cost + spot_cost)
        priced.append(CarryProposal(
            perp_venue=perp_venue, spot_venue=spot_venue, symbol=symbol,
            expected_carry_bps=expected_carry,
            annualised_carry_bps=Decimal(str(row.funding_annualised_bps)),
            holding_settlements=holding_settlements,
            funding_rate_bps=rate,
            funding_percentile=float(row.funding_percentile),
            rate_observed_at_ns=int(row.event_time_ns),
            perp_cost_bps=perp_cost, spot_cost_bps=spot_cost,
            net_carry_bps=net, observations=int(row.observations)))

    candidates = len(rows)
    cleared = [p for p in priced if p.net_carry_bps > 0]
    declined["below_cost_gate"] += len(priced) - len(cleared)

    threshold = flow_threshold([p.net_carry_bps for p in cleared], capacity)
    if threshold is None:
        proposals = sorted(cleared, key=lambda p: p.net_carry_bps, reverse=True)
    else:
        proposals = sorted((p for p in cleared if p.net_carry_bps >= threshold),
                           key=lambda p: p.net_carry_bps, reverse=True)
        declined["below_flow_threshold"] += len(cleared) - len(proposals)

    return CarrySelection(
        proposals=proposals, declined=declined, candidates=candidates,
        cleared_cost_gate=len(cleared), capacity=capacity,
        flow_threshold_bps=threshold,
        over_capacity=max(0, len(proposals) - capacity))
