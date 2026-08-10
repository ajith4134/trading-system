"""One price from several venues, weighted by the depth behind each of them.

`FEATURES.md` §1 gives this row its reason in six words: **naive N-source
averaging is what broke Mango.** An equal-weighted mean of venue prices is an
invitation - move the thinnest venue and drag the consolidated price with a
fraction of the capital the deepest venue would need. So nothing here averages.
Each venue's contribution is its measured depth, and a venue with no depth
contributes nothing no matter how loudly it quotes.

## The two ways a venue lies, and what stops each

**A thin venue quoting a moved price.** Answered by weighting: the weight is
the notional actually resting within a band of the mid, so a venue holding
$3,000 of depth cannot outvote one holding $3,000,000. The band is not a
number chosen here - it is the widest top-of-book spread among the venues
contributing to this symbol, so the comparison is made over a region every
contributing venue actually quotes. Measuring one venue's depth to five bps
and another's to fifty would be an arbitrary handicap wearing arithmetic.

**A stale venue quoting a price that no longer exists.** A frozen book keeps
its depth and therefore its weight, which makes staleness the more dangerous
of the two. Answered by excluding any venue whose newest snapshot at the clock
is older than three times its OWN measured update interval - the same
`_SILENCE_STALL_MULTIPLE` the recorder already judges a dead stream by, reused
rather than re-chosen, and applied to a cadence measured per venue rather than
assumed.

## What it refuses to do

It never fills in a venue that is missing, stale, or empty - those are counted
in `excluded` by reason and the price is computed from what is left, with
`venues_used` naming how many that was. A one-venue consolidated price is that
venue's price and says so; it is not silently presented as a consensus.

`disagreement_bps` rides every row: the spread between the highest and lowest
contributing venue mid. A consolidated price is a summary, and a summary that
hides that its inputs disagreed by 200 bps has hidden the only interesting
thing about that minute.

## Coverage, stated rather than implied

Book snapshots exist for the three core symbols on two venues - that is what
Layer 0 captures, and the wide universe has trades but no depth. So this
covers three symbols today. Weighting the rest by traded volume instead was
considered and refused: volume is the number wash trading inflates (§1's own
`[MISSED]` row), and mixing two weighting methods under one column name would
make the weaker one invisible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from features.staleness import STALE, measure_staleness
from store.clock_gated_reader import ClockGatedReader

_DATASET = "book"


@dataclass(frozen=True)
class ConsolidatedPrices:
    """Per-symbol consolidated price, and an account of every venue left out."""
    rows: pd.DataFrame
    excluded: dict[str, int]


def consolidate_prices(store_root: Path, as_of_ns: int,
                       custodian=None) -> ConsolidatedPrices:
    """The depth-weighted price per symbol at `as_of_ns`."""
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    excluded = {"stale": 0, "no_book": 0, "unparseable": 0, "no_depth_in_band": 0}
    if frame.empty:
        return ConsolidatedPrices(rows=_empty_rows(), excluded=excluded)

    out = {"symbol": [], "price": [], "venues_used": [], "venues": [],
           "total_depth_notional": [], "disagreement_bps": [],
           "newest_event_ns": []}

    for symbol, symbol_rows in frame.groupby("symbol", sort=True):
        quotes = []
        for venue, venue_rows in symbol_rows.groupby("venue", sort=True):
            ordered = venue_rows.sort_values("event_time_ns")
            newest = ordered.iloc[-1]

            if _is_stale(ordered, int(as_of_ns)):
                excluded["stale"] += 1
                continue
            book = _parse_book(newest)
            if book is None:
                excluded["unparseable"] += 1
                continue
            bids, asks = book
            if not bids or not asks:
                excluded["no_book"] += 1
                continue
            quotes.append({
                "venue": venue, "bids": bids, "asks": asks,
                "mid": (bids[0][0] + asks[0][0]) / 2.0,
                "spread_fraction": (asks[0][0] - bids[0][0]) / ((bids[0][0] + asks[0][0]) / 2.0),
                "event_time_ns": int(newest["event_time_ns"]),
            })

        if not quotes:
            continue

        # The band every contributing venue actually quotes across. Taking the
        # widest spread rather than the tightest means no venue is measured
        # over a region it does not participate in.
        band = max(q["spread_fraction"] for q in quotes)

        weighted, total_depth = 0.0, 0.0
        used = []
        for quote in quotes:
            depth = _depth_notional(quote["bids"], quote["asks"], quote["mid"], band)
            if depth <= 0:
                excluded["no_depth_in_band"] += 1
                continue
            weighted += quote["mid"] * depth
            total_depth += depth
            used.append(quote)

        if not used or total_depth <= 0:
            continue

        mids = [q["mid"] for q in used]
        out["symbol"].append(symbol)
        out["price"].append(weighted / total_depth)
        out["venues_used"].append(len(used))
        out["venues"].append(",".join(q["venue"] for q in used))
        out["total_depth_notional"].append(total_depth)
        out["disagreement_bps"].append(
            (max(mids) - min(mids)) / (sum(mids) / len(mids)) * 10_000.0)
        out["newest_event_ns"].append(max(q["event_time_ns"] for q in used))

    return ConsolidatedPrices(rows=pd.DataFrame(out), excluded=excluded)


def _is_stale(ordered: pd.DataFrame, as_of_ns: int) -> bool:
    """Is this venue's newest snapshot older than its own cadence allows?

    The rule itself now lives in `features.staleness`, which is FE-001 - every
    feature value carries this measurement, so keeping a second copy here would
    be two numbers answering one question. This wrapper is what remains: the
    per-venue judgement, and the decision about the third state.

    UNKNOWN_CADENCE is admitted rather than excluded. A venue that has published
    exactly one book has demonstrated nothing about its cadence, and refusing it
    would drop a venue for being new. That is a choice this consumer makes
    knowingly - `measure_staleness` reports the state rather than folding it
    into "fresh".
    """
    return measure_staleness(
        ordered["event_time_ns"].astype("int64"), as_of_ns).verdict == STALE


def _parse_book(row) -> tuple[list, list] | None:
    """Levels as (price, size) floats, or None when the row cannot be read."""
    try:
        bids = json.loads(row["bids"]) if isinstance(row["bids"], str) else row["bids"]
        asks = json.loads(row["asks"]) if isinstance(row["asks"], str) else row["asks"]
        return ([(float(p), float(s)) for p, s in bids],
                [(float(p), float(s)) for p, s in asks])
    except (TypeError, ValueError, KeyError):
        return None


def _depth_notional(bids: list, asks: list, mid: float, band: float) -> float:
    """Notional resting within `band` of the mid, both sides.

    Notional rather than size, because size in coins is not comparable across
    symbols and a venue quoting a different contract size would otherwise
    carry a weight that is an artefact of its units.
    """
    low, high = mid * (1.0 - band), mid * (1.0 + band)
    total = 0.0
    for price, size in bids:
        if price >= low:
            total += price * size
    for price, size in asks:
        if price <= high:
            total += price * size
    return total


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({"symbol": [], "price": [], "venues_used": [],
                         "venues": [], "total_depth_notional": [],
                         "disagreement_bps": [], "newest_event_ns": []})
