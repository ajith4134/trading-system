"""What a book can fill, and what crossing it costs.

Two rules run through everything here, and both are the difference between a
cost engine and a decoration.

**Nothing is extrapolated past the last visible level.** A notional larger than
the book raises `BookTooThin` rather than pricing the remainder at the last
price seen. Extrapolation invents liquidity, and a strategy sized on invented
liquidity is sized on nothing - which is precisely how paper mode manufactures
edge that evaporates live.

**Impact is measured from the touch, not from the mid.** The half-spread already
charges for crossing to the touch; charging it again inside impact would double
count the one cost this layer exists to state honestly.

The pure functions below take an explicit book, so they are testable without a
store and cannot silently read anything. `load_book_as_of` is the only I/O door,
and it reads the Layer 1 `book` dataset through the clock gate - never the raw
archive, and never a book that arrived after the moment being priced. When no
snapshot is visible it refuses rather than borrowing one, because a spread taken
from the wrong instant is worse than no spread at all.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Sequence

# (price, quantity), best first. Quantity is in base units, matching how every
# venue publishes a book.
Level = tuple[Decimal, Decimal]

_BPS = Decimal(10_000)
_SIDES = ("buy", "sell")

# The Layer 1 dataset this reads. One name, so the builder and the reader cannot
# drift into two datasets that look like one.
_DATASET = "book"


class NoBookAvailable(Exception):
    """No usable book at or before the requested instant.

    Distinct from `BookTooThin`: this is "we cannot see the market", that is
    "we can see it and it is not big enough". A caller may reasonably wait for
    the first and must resize for the second.
    """


class BookTooThin(Exception):
    """The visible depth cannot fill the notional. Refused, never extrapolated."""


def mid_price(bids: Sequence[Level], asks: Sequence[Level]) -> Decimal:
    """The midpoint of the touch.

    Refuses a one-sided book rather than falling back to the side that exists,
    because a mid taken from one side prices a round trip against a price
    nobody is offering.
    """
    if not bids or not asks:
        raise NoBookAvailable(
            f"one-sided book: {len(bids)} bid levels, {len(asks)} ask levels")
    return (bids[0][0] + asks[0][0]) / Decimal(2)


def half_spread_bps(bids: Sequence[Level], asks: Sequence[Level]) -> Decimal:
    """Cost in basis points of crossing from the mid to the touch, one side."""
    mid = mid_price(bids, asks)
    return ((asks[0][0] - bids[0][0]) / Decimal(2)) / mid * _BPS


def impact_bps(bids: Sequence[Level], asks: Sequence[Level],
               notional: Decimal, side: str) -> Decimal:
    """Cost beyond the touch of filling `notional` by walking the book.

    `side` is the direction of the order: a buy lifts asks, a sell hits bids.
    Pricing a sell off the ask would report the wrong cost in exactly the
    situation where it matters, which is a book deep on one side and thin on
    the other.

    Returns zero for a clip that never leaves the first level: it has not moved
    the book, and charging it impact would overstate the cost of the small
    trades this account actually does.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if notional <= 0:
        raise ValueError(f"notional must be positive, got {notional}")

    mid = mid_price(bids, asks)
    levels = asks if side == "buy" else bids
    touch = levels[0][0]

    remaining = notional
    spent = Decimal(0)
    filled_base = Decimal(0)
    for price, quantity in levels:
        level_notional = price * quantity
        if level_notional >= remaining:
            take_base = remaining / price
            spent += remaining
            filled_base += take_base
            remaining = Decimal(0)
            break
        spent += level_notional
        filled_base += quantity
        remaining -= level_notional

    if remaining > 0:
        visible = notional - remaining
        raise BookTooThin(
            f"{side} of {notional} exceeds visible depth: the book can fill "
            f"{visible} across {len(levels)} levels. Refusing rather than "
            f"extrapolating past the last level")

    average_price = spent / filled_base
    # Distance from the touch, signed so that a worse fill is always a positive
    # cost on either side.
    slip = (average_price - touch) if side == "buy" else (touch - average_price)
    return slip / mid * _BPS


def load_book_as_of(store_root: Path, venue: str, symbol: str,
                    at_ns: int) -> tuple[list[Level], list[Level]]:
    """The book as it was knowable at `at_ns`, through the clock gate only.

    Through `ClockGatedReader` and nothing else - never the raw archive, which
    is the one thing `ARCHITECTURE.md` Layer 0 forbids: backtest and live must
    share a single access path, or an off-by-one in windowing yields a great
    backtest and a broken system.

    The newest snapshot knowable at `at_ns` wins. Anything that arrived later is
    invisible rather than merely discouraged, and there is no interpolation and
    no reaching forward - pricing against a book that did not exist yet is the
    leakage this layer is built to make impossible.
    """
    from store.book_snapshots import levels_from_row
    from store.clock_gated_reader import ClockGatedReader
    from store.temporal_schema import AVAILABILITY_TIME

    try:
        visible = ClockGatedReader(Path(store_root), _DATASET).read_as_of(
            at_ns, symbols=[symbol])
    except FileNotFoundError as exc:
        raise NoBookAvailable(
            f"no book dataset in the store for {venue}:{symbol}. Depth "
            f"snapshots are in the raw archive but have not been built into a "
            f"clock-gated dataset ({exc})") from exc

    if venue and "venue" in visible:
        visible = visible[visible["venue"] == venue]
    if visible.empty:
        raise NoBookAvailable(
            f"no book for {venue}:{symbol} knowable at {at_ns}. Refusing "
            f"rather than reaching forward to one that arrived later")

    newest = visible.sort_values(AVAILABILITY_TIME).iloc[-1]
    return levels_from_row(newest)
