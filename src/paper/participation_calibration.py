"""How much of a print a resting order would actually have received.

`fill_model` caps every fill by a participation fraction. This module is where
that fraction comes from, and the whole reason it exists is that the alternative
is inventing one.

Measured, or refused. There is no default: the archive holds depth for three
symbols out of 2,123, and a fraction defaulted for the other 2,120 would be
indistinguishable downstream from a fraction that was measured.
"""
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from store.book_snapshots import levels_from_row
from store.clock_gated_reader import ClockGatedReader
from store.temporal_schema import EVENT_TIME, VENUE

RECEIPT_FILE = "latest.json"


@dataclass(frozen=True)
class Calibration:
    """A measured queue share, with the unit it was measured on.

    `event_unit_ns` is not decoration. The share is "what is left of a bar's volume
    once the queue ahead is consumed", and that is only true for an order resting
    across a whole bar of that length. Applied to per-trade events it would be
    wildly optimistic, so the unit travels with the number.
    """

    venue: str
    symbol: str
    fraction: Decimal
    event_unit_ns: int
    n_observations: int
    as_of_ns: int
    detail: str


@dataclass(frozen=True)
class CalibrationRefused:
    """The queue share cannot be measured here, and what was missing."""

    venue: str
    symbol: str
    reason: str
    missing: str

    def __float__(self):
        raise TypeError(
            f"a refusal is not a participation rate: {self.reason}. Handle "
            f"CalibrationRefused explicitly; there is no numeric fallback by design")


def calibrate_participation(book_reader: ClockGatedReader,
                            bar_reader: ClockGatedReader, *,
                            venue: str, symbol: str, as_of_ns: int,
                            bar_ns: int) -> "Calibration | CalibrationRefused":
    """Measure the share of printed volume a resting order would have taken."""
    books = _rows_for(book_reader.read_as_of(as_of_ns, symbols=[symbol]), venue)
    if books.empty:
        return CalibrationRefused(
            venue=venue, symbol=symbol, missing="book",
            reason=(f"no depth snapshots for {venue}:{symbol} at or before "
                    f"{as_of_ns}; the queue ahead of a resting order is exactly "
                    f"what depth shows and nothing else does"))

    bars = _rows_for(bar_reader.read_as_of(as_of_ns, symbols=[symbol]), venue)
    shares: list[Decimal] = []
    for bar in bars.itertuples():
        volume = Decimal(str(bar.volume))
        queue_ahead = _queue_ahead(books, getattr(bar, EVENT_TIME), bar_ns)
        if queue_ahead is None:
            continue
        reached = max(Decimal("0"), volume - queue_ahead)
        shares.append(reached / volume)

    if not shares:
        return CalibrationRefused(
            venue=venue, symbol=symbol, missing="overlapping bars",
            reason=(f"{len(books)} depth snapshots and {len(bars)} bars for "
                    f"{venue}:{symbol}, but none of the bars contains a snapshot; "
                    f"a share measured on bars that happen to line up would "
                    f"describe a different window than the one it names"))

    return Calibration(
        venue=venue, symbol=symbol, fraction=_median(shares),
        event_unit_ns=bar_ns, n_observations=len(shares), as_of_ns=as_of_ns,
        detail=(f"median over {len(shares)} bars of "
                f"max(0, volume - deeper touch side) / volume, from depth "
                f"snapshots joined to {bar_ns}ns bars"))


def record_calibration_receipt(receipt_dir: Path,
                               results: Sequence["Calibration | CalibrationRefused"],
                               *, measured_at_ns: int) -> Path:
    """Write what was measured, so the wall reads a probe instead of a claim.

    Fractions are serialised as strings. The fraction multiplies the quantity of
    every paper fill, and a receipt that round-trips 0.95 through a float and back
    puts a representation error into the size of every trade scored against it.

    Written to a temporary file and renamed, for the reason the fee receipt is: a
    half-written receipt is one the wall would read as a finished measurement.
    """
    symbols: dict[str, dict] = {}
    for result in results:
        key = f"{result.venue}:{result.symbol}"
        if isinstance(result, CalibrationRefused):
            symbols[key] = {"measured": False, "missing": result.missing,
                            "reason": result.reason}
            continue
        symbols[key] = {"measured": True, "fraction": str(result.fraction),
                        "event_unit_ns": result.event_unit_ns,
                        "n_observations": result.n_observations,
                        "as_of_ns": result.as_of_ns, "detail": result.detail}

    receipt = {"measured_at_ns": int(measured_at_ns), "symbols": symbols}
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    out = receipt_dir / RECEIPT_FILE
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out


def _queue_ahead(books: pd.DataFrame, bar_open_ns: int,
                 bar_ns: int) -> Decimal | None:
    """The deeper of the two touch sides, from a snapshot inside this bar.

    The deeper side on purpose: a strategy trades both, and taking the thinner one
    would report the easier half of the book as though it were the whole of it.
    """
    inside = books[(books[EVENT_TIME] >= bar_open_ns)
                   & (books[EVENT_TIME] < bar_open_ns + bar_ns)]
    if inside.empty:
        return None
    bids, asks = levels_from_row(inside.iloc[0])
    return max(_touch_size(bids), _touch_size(asks))


def _touch_size(levels) -> Decimal:
    """Size at the best level, or zero for a side with no levels at all."""
    if not levels:
        return Decimal("0")
    return levels[0][1]


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


def _rows_for(frame: pd.DataFrame, venue: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[frame[VENUE] == venue]
