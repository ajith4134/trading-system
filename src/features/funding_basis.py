"""Funding and basis as features: the level, the spread between them, and rank.

`FEATURES.md` §2 (P0) — *"Funding / basis spread features"*. Ledger FE-003,
whose note read *"blocked on DM-017 (spot capture) for the basis half"*; that
block is gone — `binance-spot` has been captured since 2026-08-02 and
`features.spot_perp_basis` already prices the basis off the funding dataset's
own reference columns.

## What this adds that `spot_perp_basis` does not

`features.spot_perp_basis` answers *"what is the basis right now"*. That is a
state, and a state is not yet a feature: a model cannot use `basis_bps = 4.1`
without knowing whether 4.1 is high for this instrument, and it cannot use a
funding rate without knowing what holding through it actually costs on this
venue's schedule. Three things are added here, and each is the difference
between a number and a usable one:

* **The spread.** `carry_spread_bps = basis_bps - funding_rate_bps` — what the
  perp trades above its reference, less what the holder is charged per
  settlement for the privilege. That difference *is* the cash-and-carry edge
  per interval. Either half alone is a number that moves with the other and
  tells you nothing on its own.
* **Annualisation on the venue's real schedule.** A funding rate is per
  settlement, and Binance settles three times a day while Hyperliquid settles
  twenty-four. Annualising both at one rate — the mistake `cost.funding_carry`
  exists to prevent — is wrong by 8× on one of them, in the direction that
  turns a losing carry into a winner. The settlement count comes from that
  module's own `SETTLEMENT_HOURS`, not from a constant retyped here.
* **Rank against the instrument's own history.** See below.

## Percentiles, and why the normalisation is a rank and not a z-score

Goal-doc §5a.5, carried as ledger FE-012, calls fitted per-symbol parameters
*"the single most dangerous thing that could be implemented here"* and permits
per-symbol variation only through normalisation against a symbol's own history.
This module normalises by **percentile rank within the key's own visible
history** — no mean, no standard deviation, no fitted scale.

A z-score would have been the obvious alternative and is worse here for a
reason specific to this data: funding rates are bounded, heavily zero-inflated
and occasionally violent, so their standard deviation is dominated by a handful
of settlements and a z-score computed across a regime change reports a number
that says more about the sample's tail than about where the rate sits today. A
rank is invariant to that entirely. It is also directly comparable across
instruments — the whole point of normalising — where a z-score is only
comparable if the distributions have the same shape, which these do not.

The rank is computed over the history the **clock gate** made visible, so the
percentile a backtest sees at a past clock is the percentile that was knowable
then, not one computed against a future the model had not lived through.

## A key with too little history is refused entirely, level included

The tempting design is to report the levels and leave the percentile null. It is
refused here, because a consumer that takes the level and skips the missing rank
is doing exactly what FE-012 forbids: reading a raw funding rate whose meaning
is per-instrument. `MIN_HISTORY_OBSERVATIONS` is the floor, and a key below it
is counted under `too_few_observations` rather than half-served. The levels are
not lost — `features.spot_perp_basis` reports them, and reports them as a state,
which is what they are.

## Decimal for the levels, float for the ranks, and the boundary is deliberate

Prices, rates and bps stay `Decimal` exactly as every sibling keeps them. The
percentile is computed in float, and that is safe for a reason that does not
generalise: a percentile is a **rank statistic**, so it depends only on the
ordering of the sample, and float ordering disagrees with Decimal ordering only
for values closer together than float can resolve — where the resulting rank
difference is far below the resolution this feature reports. Doing it in Decimal
would mean sorting millions of rows through Python objects for no gain that any
consumer could observe.

## Staleness (FE-001)

Stamped per (venue, symbol) against the whole visible funding history for that
key, matching `features.spot_perp_basis`: a funding poll that died an hour ago
still yields a valid percentile, and the stamp is the only thing that says the
level behind it is an hour old.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from cost.funding_carry import SETTLEMENT_HOURS
from features.spot_perp_basis import REFERENCE_COLUMN_BY_FUNDS_ON
from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "funding"
_BPS = Decimal(10_000)

# Below this, a percentile is a rank among a handful of neighbours rather than a
# position in a distribution, and reporting it would give a consumer a number
# whose confidence it cannot see. Chosen as the smallest sample in which a
# percentile has a resolution finer than a decile - the coarsest granularity any
# consumer here would act on.
MIN_HISTORY_OBSERVATIONS = 30

_COLUMNS = ("venue", "symbol", "event_time_ns", "reference",
            "funding_rate_bps", "settlements_per_year",
            "funding_annualised_bps", "basis_bps", "carry_spread_bps",
            "funding_percentile", "basis_percentile",
            "carry_spread_percentile", "observations")

_REFUSAL_REASONS = ("no_mark_price", "no_reference_price",
                    "unrecognised_funds_on", "unparseable_price",
                    "unknown_settlement_schedule", "no_funding_settlements",
                    "too_few_observations")


@dataclass(frozen=True)
class FundingBasisTable:
    """Funding, basis and their spread per (venue, symbol), each with its rank.

    One row per key that cleared the history floor. `refused` counts every key
    that did not, by reason, for the reason every sibling counts them: a table
    holding four keys where twelve venues-and-symbols were visible reads exactly
    like a market where eight of them were quiet.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def settlements_per_year(venue: str) -> int | None:
    """How many funding settlements a year holds on this venue's schedule.

    None for a venue with no schedule on record, and None for a venue with a
    schedule holding no settlements at all (spot). Both are refusals at the
    caller rather than a zero here: annualising by zero settlements would report
    every funding rate on that venue as costing nothing to hold, which is the
    flattering direction and is silent.
    """
    hours = SETTLEMENT_HOURS.get(venue)
    if hours is None or not hours:
        return None
    return len(hours) * 365


def percentile_rank(history, value: float) -> float:
    """Where `value` sits in `history`, as a fraction in [0, 1].

    Midrank on ties - the mean of the fraction strictly below and the fraction
    at or below - so a rate sitting at a value the instrument has printed a
    thousand times reports the middle of that plateau rather than its top or its
    bottom. A strictly-below definition would report 0.0 for a funding rate that
    has been exactly zero all week, and a rate at the bottom of its range is a
    different claim from a rate at the centre of a spike of zeros.
    """
    total = len(history)
    below = sum(1 for item in history if item < value)
    at_or_below = below + sum(1 for item in history if item == value)
    return (below + at_or_below) / (2 * total)


def _decimal(value) -> Decimal | None:
    """A stored price or rate as a Decimal, or None if it cannot be trusted.

    `str(value)` first, matching every sibling over this dataset: constructing
    from a float imports that float's own binary rounding as a digit of the
    stored number.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _basis_bps(mark: Decimal, reference: Decimal) -> Decimal:
    return (mark - reference) / reference * _BPS


def compute_funding_basis(store_root: Path, as_of_ns: int,
                          custodian=None) -> FundingBasisTable:
    """Funding, basis and carry-spread features per (venue, symbol) at this clock.

    Reads the `funding` dataset once through `ClockGatedReader` and derives both
    the newest values and the history the percentiles rank them against from
    that same visible frame - so the rank cannot see a row the level could not.
    Assembling the history through a second read at a different clock is the
    shape that leaks, and it leaks quietly.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return FundingBasisTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        per_year = settlements_per_year(venue)
        if venue not in SETTLEMENT_HOURS:
            refused["unknown_settlement_schedule"] += 1
            continue
        if per_year is None:
            # A venue with a schedule holding no settlements - spot. Not an
            # error and not a feature: there is no funding to rank.
            refused["no_funding_settlements"] += 1
            continue

        ordered = group.sort_values("event_time_ns")
        if len(ordered) < MIN_HISTORY_OBSERVATIONS:
            refused["too_few_observations"] += 1
            continue

        newest = ordered.iloc[-1]
        reference_column = REFERENCE_COLUMN_BY_FUNDS_ON.get(newest["funds_on"])
        if reference_column is None:
            refused["unrecognised_funds_on"] += 1
            continue

        mark = _decimal(newest["mark_price"])
        if mark is None:
            refused["no_mark_price"] += 1
            continue
        reference = _decimal(newest[reference_column])
        if reference is None:
            refused["no_reference_price"] += 1
            continue
        if reference <= 0:
            # The placeholder-price defect wearing a reference's coat. Dividing
            # by it is either an exception or an absurd number by sign.
            refused["unparseable_price"] += 1
            continue
        rate = _decimal(newest["funding_rate"])
        if rate is None:
            refused["unparseable_price"] += 1
            continue

        funding_bps = rate * _BPS
        basis_bps = _basis_bps(mark, reference)
        carry_spread_bps = basis_bps - funding_bps

        # The history the ranks are taken against, built from the SAME visible
        # frame. Rows whose reference or mark cannot be parsed are dropped from
        # the basis history only - a bad price is not a reason to discard the
        # funding rate printed beside it.
        funding_history: list[float] = []
        basis_history: list[float] = []
        spread_history: list[float] = []
        for row in ordered.itertuples(index=False):
            historical_rate = _decimal(row.funding_rate)
            if historical_rate is None:
                continue
            historical_funding_bps = float(historical_rate * _BPS)
            funding_history.append(historical_funding_bps)

            column = REFERENCE_COLUMN_BY_FUNDS_ON.get(row.funds_on)
            if column is None:
                continue
            historical_mark = _decimal(row.mark_price)
            historical_reference = _decimal(getattr(row, column))
            if (historical_mark is None or historical_reference is None
                    or historical_reference <= 0):
                continue
            historical_basis = float(
                _basis_bps(historical_mark, historical_reference))
            basis_history.append(historical_basis)
            spread_history.append(historical_basis - historical_funding_bps)

        if (len(funding_history) < MIN_HISTORY_OBSERVATIONS
                or len(basis_history) < MIN_HISTORY_OBSERVATIONS):
            # The key had enough ROWS and not enough usable ones. Counted the
            # same way, because the consequence is the same: no rank.
            refused["too_few_observations"] += 1
            continue

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["event_time_ns"].append(int(newest["event_time_ns"]))
        out["reference"].append(reference_column)
        out["funding_rate_bps"].append(funding_bps)
        out["settlements_per_year"].append(per_year)
        out["funding_annualised_bps"].append(funding_bps * Decimal(per_year))
        out["basis_bps"].append(basis_bps)
        out["carry_spread_bps"].append(carry_spread_bps)
        out["funding_percentile"].append(
            percentile_rank(funding_history, float(funding_bps)))
        out["basis_percentile"].append(
            percentile_rank(basis_history, float(basis_bps)))
        out["carry_spread_percentile"].append(
            percentile_rank(spread_history, float(carry_spread_bps)))
        out["observations"].append(len(basis_history))

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return FundingBasisTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                             refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
