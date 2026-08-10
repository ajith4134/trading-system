"""Depth-weighted book imbalance - never level-1, because level-1 is cheap to fake.

The catalogue is explicit about why this module exists in this shape
(`FEATURES.md` §2, P1): **"Never level-1 OBI - demonstrably spoofable."** A
level-1 imbalance is one bid and one ask, and either can be a resting order
cancelled before it is ever hit. Painting a lopsided level-1 book costs one
order and milliseconds of exposure; a strategy reading only the touch cannot
tell that book from a genuine one. Making the same lie true ten levels deep -
real size, resting, at real risk of being filled - costs real capital at real
risk for as long as the illusion holds. That asymmetry is the whole reason to
weight across levels instead of reading the top of the book: it does not make
spoofing impossible, it makes it expensive, and expensive is what changes who
bothers.

## The weighting scheme, and why it carries no decay constant

Each paired level `i` (1-indexed, best first) is weighted `1/i` - harmonic
decay by RANK, not by price distance or a chosen half-life. This is a
deliberate, disclosed design choice, not a default: an exponential weighting
would need someone to pick a decay rate, and there is no principled value for
one here - book depth is polled, not streamed (see below), and tick spacing
between levels differs by venue and by symbol, so a distance-based decay would
implicitly assume a tick size nothing in this dataset asserts. Rank-based
harmonic decay needs no such number: level 2 counts for half of level 1,
level 3 for a third, with nothing typed and nothing tuned. The trade-off is
disclosed too - two books that differ only in how their size is distributed
between adjacent levels of equal *rank* can score identically even if their
absolute price gaps differ, because rank, not price, is what is weighted.

## This is a snapshot measure, not the event-based OFI of the literature

`store.book_snapshots` polls each venue's book roughly once a minute; it does
not see every order placed, cancelled or filled between polls. The
order-flow-imbalance of Cont, Kukanov & Stoikov (2014) is defined over that
missing stream - a running sum of every book-changing event - and needs a
continuous feed to mean what it says. What this module computes is the
depth-weighted imbalance of the book STATE at each poll: a snapshot, not a
flow. Calling it event-based OFI would be a claim this dataset cannot support,
so it is not made - the docstring, the module and the axis verdict all say
"snapshot" rather than borrow the stronger literature term.

## What is refused, and why a refusal beats a number

A row with a missing side, all-zero depth on the levels it does carry, or a
crossed touch (best bid at or above best ask) is REFUSED and counted, never
priced. A crossed book is a broken snapshot, not a quiet market, and reporting
an imbalance for one dresses up a data defect as a signal. `levels_present`
rides every surviving row - the count of levels actually paired between bid
and ask sides, capped by whichever side is shorter - because a value computed
over 2 levels and one computed over 20 are different measurements and must
not share a column unlabelled. The archive keeps 20-level snapshots for a
small set of symbols and nothing deeper for the rest; `levels_present` is how
a consumer tells which measurement it is holding.

`level1_imbalance_spoofable` rides alongside the depth-weighted figure for
exactly one reason: to make the number the catalogue warns against visible
and named, never the headline. A consumer that reaches for it instead of
`depth_weighted_imbalance` has to type the word `spoofable` to get there.

Reads the Layer 1 `book` dataset through `ClockGatedReader` - the only door -
and stamps every row via `features.staleness` (FE-001): a frozen book's
imbalance is a stale signal exactly the way a frozen consolidated price is,
and a value with no age attached is indistinguishable from one computed a
second ago.
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


@dataclass(frozen=True)
class OrderFlowImbalanceTable:
    """The depth-weighted imbalance per (venue, symbol), and an account of
    what was refused.

    `rows` carries, per row: `depth_weighted_imbalance` (the headline,
    harmonic-by-rank across `levels_present` paired levels, range [-1, 1]),
    `level1_imbalance_spoofable` (the catalogue's forbidden headline, kept
    only as a labelled comparison), and `levels_present` (how many paired
    levels the depth-weighted figure actually drew on). `refused` counts rows
    that could not be priced, by reason - a consumer who cannot see the
    refusals cannot tell a thin book from a broken one.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


_COLUMNS = ("venue", "symbol", "event_time_ns", "depth_weighted_imbalance",
           "level1_imbalance_spoofable", "levels_present")


def compute_order_flow_imbalance(store_root: Path, as_of_ns: int,
                                 custodian=None) -> OrderFlowImbalanceTable:
    """The newest depth-weighted book imbalance per (venue, symbol) knowable
    at `as_of_ns`.

    Newest-per-key, like every other feature reading this dataset: the
    imbalance is a state of the book at the moment it was polled, and a
    caller wanting its history calls this at the clocks it cares about, each
    read separately clock-gated.
    """
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    refused = {"one_sided_book": 0, "zero_total_depth": 0,
               "crossed_book": 0, "unparseable": 0}
    if frame.empty:
        return OrderFlowImbalanceTable(rows=_empty_rows(), refused=refused)

    # Kept for staleness, alongside the newest-per-key frame used to price.
    history = frame
    newest = (frame.sort_values("event_time_ns")
                   .groupby(["venue", "symbol"], as_index=False).last())

    out = {column: [] for column in _COLUMNS}
    for _, row in newest.iterrows():
        try:
            bids, asks = levels_from_row(row)
        except (TypeError, ValueError, KeyError, InvalidOperation):
            refused["unparseable"] += 1
            continue

        if not bids or not asks:
            refused["one_sided_book"] += 1
            continue
        if bids[0][0] >= asks[0][0]:
            refused["crossed_book"] += 1
            continue

        levels_present = min(len(bids), len(asks))
        numerator = Decimal(0)
        denominator = Decimal(0)
        for rank in range(levels_present):
            weight = Decimal(1) / Decimal(rank + 1)
            _, bid_size = bids[rank]
            _, ask_size = asks[rank]
            numerator += weight * (bid_size - ask_size)
            denominator += weight * (bid_size + ask_size)

        if denominator == 0:
            refused["zero_total_depth"] += 1
            continue

        level1_denominator = bids[0][1] + asks[0][1]
        level1_imbalance = (
            (bids[0][1] - asks[0][1]) / level1_denominator
            if level1_denominator != 0 else None
        )

        out["venue"].append(row["venue"])
        out["symbol"].append(row["symbol"])
        out["event_time_ns"].append(int(row["event_time_ns"]))
        out["depth_weighted_imbalance"].append(numerator / denominator)
        out["level1_imbalance_spoofable"].append(level1_imbalance)
        out["levels_present"].append(levels_present)

    # FE-001: every value says how old the book behind it was, judged against
    # that (venue, symbol)'s own poll cadence.
    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           int(as_of_ns))
        for (venue, symbol), group in history.groupby(["venue", "symbol"], sort=False)
    }
    return OrderFlowImbalanceTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                                   refused=refused)


def _empty_rows() -> pd.DataFrame:
    empty = pd.DataFrame({column: [] for column in _COLUMNS})
    return stamp(empty, {}, ["venue", "symbol"])
