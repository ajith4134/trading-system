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


# Which price a venue computes its funding against. Not decoration: Binance
# funds on MARK and Hyperliquid on ORACLE, and `FEATURES.md` §1 flags the
# distinction [MISSED] precisely because the gap between those prices is the
# basis trade itself. A cross-venue carry number that does not know which price
# each side was funded on is comparing two different quantities.
FUNDS_ON_MARK = "mark"
FUNDS_ON_ORACLE = "oracle"


@dataclass(frozen=True)
class FundingObservation:
    """One funding poll, as it was received.

    Four fields carry what differs BETWEEN venues rather than between rows, and
    they exist so that a reader never has to know which venue it is holding in
    order to read the row correctly.
    """

    symbol: str
    venue: str
    funding_rate: Decimal          # the rate applied at the last settlement
    mark_price: Decimal
    index_price: Decimal | None    # None where the venue publishes none
    next_funding_time_ns: int
    event_time_ns: int             # the venue's stamp, or our receipt - see the flag
    ingestion_time_ns: int         # when we received it
    # The oracle price, where the venue has one. Hyperliquid funds on it; Binance
    # publishes none, so this is None there rather than a copy of mark.
    oracle_price: Decimal | None = None
    # Which of the above the funding rate is computed against.
    funds_on: str = FUNDS_ON_MARK
    # True when `event_time_ns` is OUR receipt rather than the venue's stamp.
    #
    # Hyperliquid's asset context carries no timestamp of any kind - measured
    # against the live endpoint 2026-08-09, the keys are exactly funding,
    # openInterest, prevDayPx, dayNtlVlm, premium, oraclePx, markPx, midPx,
    # impactPxs and dayBaseVlm. So there is no venue clock to key on and the
    # only honest event time is when the poll landed.
    #
    # Flagged rather than silently equal, because a consumer comparing event
    # times across venues would otherwise be comparing a venue stamp against a
    # receipt and would have no way to tell. Rule 8, one layer down: the absence
    # of a measurement is its own state.
    event_time_is_receipt: bool = False
    # True when `next_funding_time_ns` is unknown rather than known to be zero.
    next_funding_time_unknown: bool = False


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


def extract_hyperliquid_funding(payload: str, entry, venue: str,
                                symbol: str) -> list[FundingObservation]:
    """Read one archived `assetCtx` record into the same shape as a Binance poll.

    Same dataset, different venue model, and three of the differences are real
    rather than cosmetic:

    **It funds on the ORACLE price, hourly, capped at 4%/hour** - against
    Binance's mark price every eight hours. Recorded in `funds_on` so a
    cross-venue carry number knows which price each side was funded against
    instead of assuming they match.

    **There is no venue timestamp.** Measured against the live endpoint
    2026-08-09, an asset context carries exactly funding, openInterest,
    prevDayPx, dayNtlVlm, premium, oraclePx, markPx, midPx, impactPxs and
    dayBaseVlm - no clock of any kind. So the event time is our receipt, and
    `event_time_is_receipt` says so rather than letting it pass as a venue
    stamp. Availability is unaffected: it was already keyed on receipt for every
    venue, which is what makes the gate safe.

    **There is no next settlement time.** Hyperliquid settles hourly, so it is
    derivable from the clock - and deriving it here would put an assumption
    about venue behaviour into a row that reads like an observation. Left
    unknown and flagged; whoever needs it can apply the venue's schedule
    knowingly.

    Recognised by its own fields rather than by the file it came from, matching
    `extract_funding`: misreading a trade as a funding rate would poison every
    carry cost downstream.
    """
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(body, dict):
        return []
    if "funding" not in body or "markPx" not in body:
        return []

    return [FundingObservation(
        symbol=body.get("coin") or symbol,
        venue=venue,
        funding_rate=Decimal(str(body["funding"])),
        mark_price=Decimal(str(body["markPx"])),
        # This venue publishes no index price. None, never a copy of mark.
        index_price=None,
        oracle_price=(Decimal(str(body["oraclePx"]))
                      if body.get("oraclePx") is not None else None),
        funds_on=FUNDS_ON_ORACLE,
        next_funding_time_ns=0,
        next_funding_time_unknown=True,
        event_time_ns=int(entry.t_recv_ns),
        event_time_is_receipt=True,
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
        # None stays None rather than becoming a copy of mark. A venue that
        # publishes no index has not published one, and filling it from a
        # neighbouring price is the kind of quiet substitution that reads as
        # data forever after.
        "index_price": None if o.index_price is None else str(o.index_price),
        "oracle_price": None if o.oracle_price is None else str(o.oracle_price),
        "funds_on": o.funds_on,
        "event_time_is_receipt": o.event_time_is_receipt,
        "next_funding_time_unknown": o.next_funding_time_unknown,
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
    for column in ("event_time_is_receipt", "next_funding_time_unknown"):
        frame[column] = frame[column].astype("bool")
    # Pinned to a string dtype even when every value in the partition is null.
    #
    # Not defensive typing - it broke the live dataset. Hyperliquid publishes no
    # index price, so its first partition had `index_price` null for all 3,248
    # rows, pyarrow inferred the column's type as `null`, and reading the dataset
    # afterwards failed outright:
    #
    #   ArrowNotImplementedError: Unsupported cast from large_string to null
    #
    # One venue's absent field made every venue's funding unreadable, because a
    # dataset is read across its partitions and their schemas have to unify. A
    # column whose type depends on whether a particular day happened to carry a
    # value is not a schema.
    for column in ("funding_rate", "mark_price", "index_price", "oracle_price",
                   "funds_on"):
        frame[column] = frame[column].astype("string")
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
