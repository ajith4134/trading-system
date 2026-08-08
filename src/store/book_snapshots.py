"""Archived depth snapshots, turned into a clock-gated book dataset.

**Built from snapshots, not by replaying diffs onto them.** A snapshot is a true
book as the venue reported it, once a minute. Diff replay would give sub-second
resolution, and it is a much larger and much riskier piece of work: one
mis-sequenced update and the reconstructed book diverges from reality silently,
with nothing to compare it against. A book that is right once a minute is worth
more than a book that is plausible continuously - and the raw diffs stay in the
archive, so replay remains possible later without recapturing anything.

**A book is knowable when we received it, never when the venue stamped it.**
Same rule as funding, for the same reason: availability keyed on the venue's
clock lets a backtest price against a book that had not arrived yet.

**Only the top of the book is kept.** A snapshot is 1,000 levels a side and the
whole archive holds the original, so this dataset is a derived view sized for
what it is for: at this account's clip sizes the far side of a 1,000-level book
is never reached, and storing it would multiply the dataset by fifty for depth
nothing will walk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

import pandas as pd

from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    SYMBOL,
    VENUE,
)

_MS_TO_NS = 1_000_000

# Levels kept per side. Twenty covers every clip this account can place against
# a liquid perp by orders of magnitude; `impact_bps` refuses rather than
# extrapolating when a notional exceeds what is stored, so a clip too large for
# twenty levels is answered honestly rather than approximated.
_LEVELS_KEPT = 20

Level = tuple[Decimal, Decimal]


@dataclass(frozen=True)
class BookSnapshot:
    """One full-book response, as it was received."""

    symbol: str
    venue: str
    bids: tuple[Level, ...]        # best first, descending
    asks: tuple[Level, ...]        # best first, ascending
    last_update_id: int
    event_time_ns: int
    ingestion_time_ns: int


def _levels(raw, reverse: bool) -> tuple[Level, ...]:
    """Parse and order one side, best first.

    Ordered here rather than trusted from the venue: `impact_bps` walks from the
    touch outward, and a reversed side would price a large clip against the far
    end of the book and report it cheap - a silent understatement in exactly the
    case where cost matters most.
    """
    parsed = [(Decimal(str(price)), Decimal(str(size))) for price, size in raw]
    parsed.sort(key=lambda level: level[0], reverse=reverse)
    return tuple(parsed[:_LEVELS_KEPT])


def extract_book_snapshot(payload: str, entry, venue: str,
                          symbol: str) -> list[BookSnapshot]:
    """Read one archived frame, or nothing if it is not a full-book snapshot.

    A depth *diff* carries `e` and `U`; a snapshot carries `lastUpdateId` and
    neither. Misreading a diff as a book would price against a changeset, so the
    distinction is made on the fields rather than on the filename.

    A one-sided book is refused rather than stored. It has no mid, and admitting
    one would put a row in the dataset every consumer then has to defend
    against.
    """
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(body, dict) or "lastUpdateId" not in body:
        return []
    if "e" in body or "U" in body:          # a diff, not a snapshot
        return []

    bids_raw, asks_raw = body.get("bids"), body.get("asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return []
    if not bids_raw or not asks_raw:
        return []

    received = int(entry.t_recv_ns)
    # Futures stamps `E`; spot stamps nothing at all. Falling back to receipt
    # rather than inventing an event time - admitting we only know when it
    # landed is better than a number nobody measured.
    stamped_ms = body.get("E")
    event_ns = stamped_ms * _MS_TO_NS if isinstance(stamped_ms, int) else received

    return [BookSnapshot(
        symbol=symbol, venue=venue,
        bids=_levels(bids_raw, reverse=True),
        asks=_levels(asks_raw, reverse=False),
        last_update_id=int(body["lastUpdateId"]),
        event_time_ns=event_ns,
        ingestion_time_ns=received,
    )]


def build_book_frame(snapshots: Iterable[BookSnapshot]) -> pd.DataFrame:
    """One bitemporal row per snapshot, levels carried as text.

    Text rather than floats: a representation error here lands directly in the
    spread that gates every strategy, which is the same reason `fee_schedule`
    parses from strings.
    """
    rows = list(snapshots)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: s.symbol,
        VENUE: s.venue,
        "bids": json.dumps([[str(p), str(q)] for p, q in s.bids]),
        "asks": json.dumps([[str(p), str(q)] for p, q in s.asks]),
        "last_update_id": s.last_update_id,
        EVENT_TIME: s.event_time_ns,
        INGESTION_TIME: s.ingestion_time_ns,
        # A snapshot is knowable the moment it lands. Nothing to close, nothing
        # to wait for.
        AVAILABILITY_TIME: s.ingestion_time_ns,
    } for s in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME, "last_update_id"):
        frame[column] = frame[column].astype("int64")
    return frame


def levels_from_row(row) -> tuple[list[Level], list[Level]]:
    """Recover a book from a stored row, as Decimals, best first."""
    def parse(text):
        return [(Decimal(price), Decimal(size)) for price, size in json.loads(text)]

    return parse(row["bids"]), parse(row["asks"])
