"""Funding paid or earned across settlements — the carry the directive rests on.

The venue asymmetry encoded here is marked `[MISSED]` in `FEATURES.md` §1, and
it is the reason this is a module rather than a constant. Binance USDⓈ-M settles
**8-hourly on the mark price**; Hyperliquid settles **hourly on the oracle
price**, capped at 4%/hour. A single shared schedule is wrong at both ends, and
it is wrong in the direction that flatters the strategy: pricing a Hyperliquid
carry on Binance's schedule undercounts settlements eightfold, which can turn a
losing carry into an apparent winner.

An unknown venue refuses rather than inheriting a default. Defaulting to
8-hourly is exactly the mistake the module exists to prevent, and it would be
silent.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Sequence

_BPS = Decimal(10_000)
_NS_PER_HOUR = 3_600_000_000_000
_SIDES = ("long", "short")

# The Layer 1 dataset this reads. One name, so the builder and the reader cannot
# drift apart into two datasets that look like one.
_DATASET = "funding"

# Hours past midnight UTC at which each venue settles funding. Encoded as a
# schedule rather than an interval because Binance's settlements are at fixed
# wall-clock hours, not every-8-hours-from-whenever-you-opened.
#
# Public because `features.funding_basis` annualises a funding rate by the
# number of settlements a year actually holds, and that count has to come from
# the same schedule this module charges against. A private copy there would put
# the eightfold Binance/Hyperliquid asymmetry in two places, and the copy that
# drifts is the one nobody re-derived - which is the exact defect the docstring
# above says this module exists to prevent.
SETTLEMENT_HOURS = {
    "binance": (0, 8, 16),
    "binance-spot": (),          # spot has no funding at all
    "hyperliquid": tuple(range(24)),
}


class NoFundingAvailable(Exception):
    """The funding a position would pay cannot be established.

    Raised for a venue whose schedule is unknown, and for a period the store
    cannot serve. Both are refusals rather than zeros: charging zero funding to
    a carry strategy is not a conservative default, it is the optimistic one.
    """


def settlements_between(venue: str, held_from_ns: int, held_to_ns: int) -> list[int]:
    """Every funding settlement instant strictly after open and at or before close.

    Inclusive at the close because a position held exactly to 08:00:00 was open
    when the settlement landed and pays it. Exclusive at the open for the mirror
    reason: opening exactly at 08:00:00 is opening after that settlement.
    """
    if venue not in SETTLEMENT_HOURS:
        raise NoFundingAvailable(
            f"no funding schedule known for venue {venue!r}. Refusing rather "
            f"than assuming a default - the schedules genuinely differ "
            f"(Binance 8-hourly on mark, Hyperliquid hourly on oracle) and a "
            f"wrong one misprices carry in the flattering direction")

    hours = SETTLEMENT_HOURS[venue]
    if not hours or held_to_ns <= held_from_ns:
        return []

    # Walk day boundaries rather than stepping by a fixed interval, so a
    # schedule tied to wall-clock hours stays tied to them.
    day_ns = 24 * _NS_PER_HOUR
    first_day = (held_from_ns // day_ns) * day_ns
    crossed = []
    day = first_day
    while day <= held_to_ns:
        for hour in hours:
            instant = day + hour * _NS_PER_HOUR
            if held_from_ns < instant <= held_to_ns:
                crossed.append(instant)
        day += day_ns
    return sorted(crossed)


def funding_cost_bps(rates: Sequence[Decimal], side: str) -> Decimal:
    """Total funding in basis points across the settlements a position crossed.

    `rates` are the venue's own per-settlement unit rates, one per settlement
    actually crossed — **not the current rate repeated**. Funding moves, and a
    carry priced on today's rate held constant is a forecast wearing a
    measurement's clothes.

    Positive means longs pay shorts, which is the usual state of a crypto perp
    in a bull market. A cost is positive bps, so a short earning funding
    returns a negative cost.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    total = sum((Decimal(rate) for rate in rates), Decimal(0)) * _BPS
    return total if side == "long" else -total


def load_funding_rates_as_of(store_root: Path, venue: str, symbol: str,
                             held_from_ns: int, held_to_ns: int) -> list[Decimal]:
    """The archived rate at each settlement crossed, through the clock gate only.

    Read through `ClockGatedReader` and nothing else. The reader is asked for
    what was knowable **at each settlement**, not at the end of the holding
    period, so a rate published after the settlement it would be charged
    against is invisible here rather than merely discouraged. That is the whole
    reason this goes through Layer 1 instead of the raw archive: backtest and
    live share one access path, and an off-by-one in windowing cannot produce a
    cheaper backtest than live.

    Refuses when the dataset is absent or has nothing at a settlement. Charging
    zero would not be a conservative default - it is the optimistic one, and it
    flatters exactly the family the prime directive rests on.
    """
    from store.clock_gated_reader import ClockGatedReader
    from store.funding_rates import FundingObservation, rates_at_settlements
    from store.temporal_schema import INGESTION_TIME, SYMBOL

    settlements = settlements_between(venue, held_from_ns, held_to_ns)
    if not settlements:
        return []

    try:
        # Read as of the last settlement: anything later is not knowable at any
        # settlement in this window, so the gate excludes it before we look.
        visible = ClockGatedReader(Path(store_root), _DATASET).read_as_of(
            settlements[-1], symbols=[symbol])
    except FileNotFoundError as exc:
        raise NoFundingAvailable(
            f"no funding dataset in the store for {venue}:{symbol}. "
            f"premiumIndex is in the raw archive but has not been built into a "
            f"clock-gated dataset ({exc})") from exc

    if visible.empty:
        raise NoFundingAvailable(
            f"the funding dataset holds nothing for {venue}:{symbol} knowable "
            f"by {settlements[-1]}. Refusing rather than charging zero")

    rows = visible[visible["venue"] == venue] if "venue" in visible else visible
    observations = [
        FundingObservation(
            symbol=row[SYMBOL], venue=venue,
            funding_rate=Decimal(str(row["funding_rate"])),
            mark_price=Decimal(str(row.get("mark_price", "0"))),
            index_price=Decimal(str(row.get("index_price", "0"))),
            next_funding_time_ns=int(row.get("next_funding_time_ns", 0)),
            event_time_ns=int(row.get("event_time_ns", 0)),
            ingestion_time_ns=int(row[INGESTION_TIME]))
        for _, row in rows.iterrows()
    ]

    try:
        return rates_at_settlements(observations, settlements)
    except LookupError as exc:
        raise NoFundingAvailable(
            f"funding dataset for {venue}:{symbol} has a settlement with no "
            f"prior observation: {exc}") from exc
