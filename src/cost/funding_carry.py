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

# Hours past midnight UTC at which each venue settles funding. Encoded as a
# schedule rather than an interval because Binance's settlements are at fixed
# wall-clock hours, not every-8-hours-from-whenever-you-opened.
_SETTLEMENT_HOURS = {
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
    if venue not in _SETTLEMENT_HOURS:
        raise NoFundingAvailable(
            f"no funding schedule known for venue {venue!r}. Refusing rather "
            f"than assuming a default - the schedules genuinely differ "
            f"(Binance 8-hourly on mark, Hyperliquid hourly on oracle) and a "
            f"wrong one misprices carry in the flattering direction")

    hours = _SETTLEMENT_HOURS[venue]
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

    Always refuses today. `premiumIndex` is captured into the raw archive and
    carries exactly what this needs — `lastFundingRate` and `nextFundingTime` —
    but it has never been built into a clock-gated dataset, so there is nothing
    for the reader to serve.

    Reading the raw archive directly would answer the question and break the
    guarantee that makes Layer 1 worth having: backtest and live must share one
    access path. A refusal costs a carry that cannot be priced yet; a direct
    read costs the property the whole layer exists for.
    """
    raise NoFundingAvailable(
        f"no funding dataset in the store for {venue}:{symbol} between "
        f"{held_from_ns} and {held_to_ns}. premiumIndex is in the raw archive "
        f"but has not been built into a clock-gated dataset. Refusing rather "
        f"than reading the archive directly or charging zero")
