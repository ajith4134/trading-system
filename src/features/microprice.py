"""The depth-weighted price between the touch prices, never the naive mid.

`FEATURES.md` §2 (P2) files this as *"volume-adjusted variant outperforms the
original"*. The "original" is Stoikov's level-1 microprice - the touch prices
weighted by the OPPOSITE side's resting size:

    microprice = (bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)

Weighting each side's price by the OTHER side's size, not its own, is the
part that is easy to get backwards while still producing a number that looks
like a price. A wall of resting bids does not pull the microprice DOWN toward
the bid - it signals buying pressure that will consume the ask, so it pulls
the microprice UP, toward the ask. Reversing the weighting produces a number
that sits on the wrong side of the mid and would be indistinguishable from
the correct one without a test built specifically to catch the swap - see
`test_bid_size_far_greater_than_ask_size_pulls_microprice_toward_the_ask`.

## The volume-adjusted variant, and exactly how many levels it uses

The catalogue asks for volume-adjustment beyond level 1. `FEATURES.md` §2 is
explicit about why level-1 alone is not enough for an order-flow number:
**"Never level-1 OBI - demonstrably spoofable"** (the OFI row, same section).
A single resting order at the touch costs nothing to place and cancel; moving
the number here requires resting size at `LEVELS_USED` price levels
simultaneously, which is a materially more expensive thing to fake.

So the weight on each side is not that side's level-1 size, it is the
CUMULATIVE size across the first `LEVELS_USED` levels - still applied to the
level-1 (touch) PRICES, because a microprice several levels off the touch is
not a microprice, it is a VWAP wearing this module's name:

    microprice = (bid_touch * cum_ask_size + ask_touch * cum_bid_size)
                 / (cum_bid_size + cum_ask_size)

`LEVELS_USED = 5`, disclosed here rather than fitted. The archive
(`store.book_snapshots`) keeps 20 levels a side, sized for what this
account's own clip sizes can reach "by orders of magnitude" - see that
module's docstring. Microprice is not a fill-cost estimate for one clip
(`cost.spread_and_depth` already owns that job, walking the full visible
book); it is a fair-value estimate for the current instant, and liquidity
resting fifteen levels deep is not evidence about the next tick - it is
mostly there to be cancelled before it is ever reached. Five levels is enough
to raise the cost of spoofing the number well past level 1 while staying
inside the part of the book that plausibly trades soon.

## What is refused, and what is exposed but never the headline

A book missing a side, crossed (`bid >= ask`), locked (`bid == ask`), or
carrying zero size at the touch is REFUSED by its own named reason and
counted - never priced from a default. A crossed or locked book is a real
market state (not a parsing failure) and gets its own reason,
`crossed_or_locked_book`, rather than being folded into a generic "bad book"
bucket a consumer cannot distinguish from a data corruption.

The level-1-only number is computed too, because it is occasionally useful
for comparison against published microprice research that only ever used
level 1. It rides the result as `microprice_level1_spoofable` - named for
what it is, per the house rule that the spoofable variant must never be the
headline number. `microprice` is always the depth-weighted column.

## Staleness (FE-001)

Every row is stamped via `features.staleness`, keyed on (venue, symbol). A
microprice computed from a book that stopped updating carries the same
danger `spot_perp_basis` and `consolidated_price` guard against - a
confident number in the same column as a fresh one - so this module does not
get to skip the stamp because it is "just arithmetic on a book".

Reads the `book` dataset through `ClockGatedReader`, the only door - no
direct Parquet reads, no live REST.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from features.staleness import measure_staleness, stamp
from store.book_snapshots import levels_from_row
from store.clock_gated_reader import ClockGatedReader

_DATASET = "book"

# Disclosed, not fitted - see the module docstring for why 5 and not the
# archive's full 20.
LEVELS_USED = 5


@dataclass(frozen=True)
class MicropriceTable:
    """The computed microprice per (venue, symbol) and an account of every
    book that could not be priced.

    `rows` carries the depth-weighted `microprice` (the headline number, over
    `LEVELS_USED` levels), the level-1-only `microprice_level1_spoofable` for
    comparison, the touch prices themselves, and how many levels each side
    actually contributed (a book thinner than `LEVELS_USED` uses what it has
    rather than being refused for it). `refused` counts what was left out, by
    reason - a microprice table that silently dropped every crossed book
    would look identical to a market with no crossed books.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def compute_microprice(store_root: Path, as_of_ns: int,
                       custodian=None) -> MicropriceTable:
    """The newest depth-weighted microprice per (venue, symbol) at `as_of_ns`.

    Newest-per-key, the same shape as `compute_spot_perp_basis`: the
    microprice is a state, not a series, and a caller wanting its history
    calls this at each clock it cares about - every read is clock-gated, so
    the assembled history is leak-free by construction.
    """
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    refused = {"no_bid": 0, "no_ask": 0, "crossed_or_locked_book": 0,
               "zero_size": 0, "unparseable_book": 0}
    if frame.empty:
        return MicropriceTable(rows=_empty_rows(), refused=refused)

    # Kept whole for the staleness cadence, the same reasoning
    # `compute_spot_perp_basis` uses: the newest row gives the value, the
    # full visible history gives the cadence that says whether it is current.
    history = frame
    newest = (frame.sort_values("event_time_ns")
                   .groupby(["venue", "symbol"], as_index=False).last())

    out = {"venue": [], "symbol": [], "event_time_ns": [],
           "microprice": [], "microprice_level1_spoofable": [],
           "bid_touch": [], "ask_touch": [],
           "bid_levels_used": [], "ask_levels_used": []}

    for row in newest.itertuples(index=False):
        parsed = _parse_levels(row.bids, row.asks)
        if parsed is None:
            refused["unparseable_book"] += 1
            continue
        bids, asks = parsed
        if not bids:
            refused["no_bid"] += 1
            continue
        if not asks:
            refused["no_ask"] += 1
            continue

        bid_touch, bid_touch_size = bids[0]
        ask_touch, ask_touch_size = asks[0]

        if bid_touch >= ask_touch:
            # Real market state, not a parsing failure - a crossed book is
            # locked when equal, inverted when the bid is above the ask, and
            # neither has a mid worth reporting. Named separately from every
            # other refusal so a consumer can tell "the market did this" from
            # "the data was broken".
            refused["crossed_or_locked_book"] += 1
            continue
        if bid_touch_size <= 0 or ask_touch_size <= 0:
            # A level that exists but carries no size is the placeholder-price
            # defect's sibling: dividing by it is either an exception or an
            # absurd weighting, and pretending the touch has size it does not
            # have is exactly the substituted number this module refuses to
            # produce.
            refused["zero_size"] += 1
            continue

        bid_window = bids[:LEVELS_USED]
        ask_window = asks[:LEVELS_USED]
        cum_bid_size = sum((size for _, size in bid_window), Decimal(0))
        cum_ask_size = sum((size for _, size in ask_window), Decimal(0))

        depth_weighted = ((bid_touch * cum_ask_size + ask_touch * cum_bid_size)
                          / (cum_bid_size + cum_ask_size))
        level1_only = ((bid_touch * ask_touch_size + ask_touch * bid_touch_size)
                       / (bid_touch_size + ask_touch_size))

        out["venue"].append(row.venue)
        out["symbol"].append(row.symbol)
        out["event_time_ns"].append(int(row.event_time_ns))
        out["microprice"].append(depth_weighted)
        out["microprice_level1_spoofable"].append(level1_only)
        out["bid_touch"].append(bid_touch)
        out["ask_touch"].append(ask_touch)
        out["bid_levels_used"].append(len(bid_window))
        out["ask_levels_used"].append(len(ask_window))

    # FE-001: every value says how old the book behind it was, judged against
    # that (venue, symbol)'s own snapshot cadence.
    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           int(as_of_ns))
        for (venue, symbol), group in history.groupby(["venue", "symbol"], sort=False)
    }
    return MicropriceTable(rows=stamp(rows, ages, ["venue", "symbol"]), refused=refused)


def _parse_levels(bids_raw, asks_raw) -> tuple[list, list] | None:
    """Bid/ask levels as Decimals, best first - the same parser
    `cost.spread_and_depth` reads a book through, so this module and that one
    cannot silently disagree about what a stored level means. `None` on
    anything unparseable: a null column, a non-JSON string, or a value that
    is not a decimal - the row is refused rather than guessed at.
    """
    if not isinstance(bids_raw, str) or not isinstance(asks_raw, str):
        return None
    try:
        bids, asks = levels_from_row({"bids": bids_raw, "asks": asks_raw})
    except (TypeError, ValueError, InvalidOperation, KeyError):
        return None
    return bids, asks


def _empty_rows() -> pd.DataFrame:
    empty = pd.DataFrame({
        "venue": [], "symbol": [], "event_time_ns": [],
        "microprice": [], "microprice_level1_spoofable": [],
        "bid_touch": [], "ask_touch": [],
        "bid_levels_used": [], "ask_levels_used": [],
    })
    return stamp(empty, {}, ["venue", "symbol"])
