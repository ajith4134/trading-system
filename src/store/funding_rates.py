"""Archived `premiumIndex` polls, turned into a clock-gated funding dataset.

Funding is the carry the prime directive rests on, and it was the one cost the
engine had to refuse for want of somewhere to read it from.

Two properties matter more than anything else here.

**A rate is knowable when we received it, not when the venue stamped it.** The
venue's `time` field runs ahead of our receipt by however long the round trip
took - measured at ~500ms on a real frame. Keying availability on the venue's
clock would let a backtest read a rate before it arrived, which is precisely
the leakage Layer 1 exists to make structurally impossible rather than a matter
of discipline.

**Mark and index are both kept.** `FEATURES.md` §1 marks mark-vs-index-vs-oracle
`[MISSED]`: Hyperliquid funds on the *oracle* price, Binance on *mark*, and the
gap between them is the basis trade itself. Dropping either at ingest makes it
unrecoverable, and this archive cannot be rebuilt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

import pandas as pd

from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    SYMBOL,
    VENUE,
)

_MS_TO_NS = 1_000_000


@dataclass(frozen=True)
class FundingObservation:
    """One `premiumIndex` poll, as it was received."""

    symbol: str
    venue: str
    funding_rate: Decimal          # the rate applied at the last settlement
    mark_price: Decimal
    index_price: Decimal
    next_funding_time_ns: int
    event_time_ns: int             # the venue's stamp
    ingestion_time_ns: int         # when we received it


def extract_funding(payload: str, entry, venue: str,
                    symbol: str) -> list[FundingObservation]:
    """Read one archived frame, or return nothing if it is not a funding poll.

    The archive holds several streams side by side, and misreading a trade as a
    funding rate would quietly poison every carry cost downstream. A frame is
    recognised by the fields it is fetched for rather than by the file it came
    from, matching how `BinanceVenue.extract` routes the same payload.

    Rates are parsed straight from the venue's decimal string. Going via float
    would put a representation error into the number that prices every carry
    trade - the same reason `fee_schedule` takes strings.
    """
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(body, dict):
        return []
    # A polled REST body is bare - no "data" envelope and no event field.
    if "lastFundingRate" not in body or "markPrice" not in body:
        return []

    venue_time_ms = body.get("time")
    if not isinstance(venue_time_ms, int):
        return []

    return [FundingObservation(
        symbol=body.get("symbol") or symbol,
        venue=venue,
        funding_rate=Decimal(str(body["lastFundingRate"])),
        mark_price=Decimal(str(body["markPrice"])),
        index_price=Decimal(str(body.get("indexPrice", body["markPrice"]))),
        next_funding_time_ns=int(body.get("nextFundingTime", 0)) * _MS_TO_NS,
        event_time_ns=venue_time_ms * _MS_TO_NS,
        ingestion_time_ns=int(entry.t_recv_ns),
    )]


def build_funding_frame(observations: Iterable[FundingObservation]) -> pd.DataFrame:
    """One bitemporal row per observation.

    No observations is an empty frame, never a zero rate. Charging zero funding
    to a carry strategy is not a conservative default - it is the optimistic
    one, and it flatters exactly the family this dataset exists to price.
    """
    rows = list(observations)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: o.symbol,
        VENUE: o.venue,
        "funding_rate": str(o.funding_rate),      # Decimal survives as its own text
        "mark_price": str(o.mark_price),
        "index_price": str(o.index_price),
        "next_funding_time_ns": o.next_funding_time_ns,
        EVENT_TIME: o.event_time_ns,
        INGESTION_TIME: o.ingestion_time_ns,
        # A polled value is knowable the moment it lands, and not one nanosecond
        # earlier. There is no bar to close and nothing to wait for, so this is
        # ingestion time exactly rather than a max over anything.
        AVAILABILITY_TIME: o.ingestion_time_ns,
    } for o in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME,
                   "next_funding_time_ns"):
        frame[column] = frame[column].astype("int64")
    return frame


def rates_at_settlements(observations: Sequence[FundingObservation],
                         settlements_ns: Sequence[int]) -> list[Decimal]:
    """The rate that was known at each settlement, in order.

    The last observation received *at or before* the settlement, never the
    newest one overall. Pricing a carry against a rate that had not been
    published when it was charged is look-ahead, and it is invisible: the
    numbers look plausible and the backtest is simply wrong.

    A settlement with no prior observation raises rather than defaulting.
    Borrowing a later rate, or substituting zero, prices the trade against a
    number that did not exist yet.
    """
    ordered = sorted(observations, key=lambda o: o.ingestion_time_ns)
    rates: list[Decimal] = []
    for settlement in settlements_ns:
        known = [o for o in ordered if o.ingestion_time_ns <= settlement]
        if not known:
            raise LookupError(
                f"no funding observation known at or before settlement "
                f"{settlement}; refusing rather than borrowing a later rate")
        rates.append(known[-1].funding_rate)
    return rates
