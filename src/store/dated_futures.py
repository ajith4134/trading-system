"""Archived `linearTickers` polls for the contracts that expire, kept as their own dataset.

The funding reader drops these on purpose - a dated contract pays no funding and
says so, `fundingRate` is the empty string - and until now that drop was the end
of them. 40 dated contracts across 9 underlyings were being captured every 60
seconds and read by nothing, which is the shape this repo's own CLAUDE.md warns
about: built, archived, called by nobody.

They belong in a separate dataset rather than as funding rows with a null rate.
`funding` answers "what carry is charged against a perpetual"; every consumer of
it - `cost.funding_carry`, `features.spot_perp_basis` - would then have to learn
that some of its rows are not perpetuals at all. A calendar contract is a
different instrument with a different question attached (what does the market
charge to hold this until a date), and it gets its own door.

## What the venue publishes, measured on a real frame 2026-08-10

`BTCUSDT-14AUG26`: lastPrice 64976.8, indexPrice 64999.5, markPrice 65011.9,
deliveryTime 1786694400000, basis -22.71, basisRate 0.00011907, basisRateYear
0.01723021, fundingRate "" and time 1786336806394.

The venue computes a basis of its own and this module stores it **without using
it as the answer**. Its `basis` of -22.71 is nearly `lastPrice - indexPrice`
(64976.8 - 64999.5 = -22.70) and off by a cent, while `basisRateYear` annualises
something nearer the mark than the last price - so the venue's own three basis
fields reconcile neither with each other nor with the prices published in the
same frame. Storing them alongside our own computation is what makes that
checkable later; substituting them for it would import an inconsistency we would
then have to trust.

Delivery time is the field that makes a term structure possible at all, and it
is the venue's own declaration rather than a date parsed out of the symbol
suffix. `BTCUSDT-14AUG26` is a naming convention and would break the day it
changed; `deliveryTime` is a number the matching engine settles on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
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


@dataclass(frozen=True)
class DatedFutureObservation:
    """One poll of one expiring contract, as it was received.

    `delivery_time_ns` is the whole reason this dataset exists - it is what turns
    a price into a point on a curve. Everything else is here because dropping it
    at ingest makes it unrecoverable: this archive is evicted after seven days
    and cannot be rebuilt.
    """

    symbol: str
    venue: str
    mark_price: Decimal
    index_price: Decimal | None     # None where the venue publishes none
    last_price: Decimal | None
    delivery_time_ns: int
    event_time_ns: int              # the venue's own stamp - bybit sends one
    ingestion_time_ns: int          # when we received it
    open_interest: Decimal | None = None
    # The venue's own basis arithmetic, stored for corroboration and never used
    # as the answer. `venue_basis_rate_year` is a fraction, not basis points.
    venue_basis: Decimal | None = None
    venue_basis_rate: Decimal | None = None
    venue_basis_rate_year: Decimal | None = None


def _decimal_or_none(value) -> Decimal | None:
    """A price the venue may send as an empty string, and often does.

    Empty is absent, not zero. A zero price on a curve point is not a
    conservative default - it prices the contract at a 100% discount to index,
    which is the most attractive carry the arithmetic can produce.
    """
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def extract_bybit_dated_future(payload: str, entry, venue: str,
                               symbol: str) -> list[DatedFutureObservation]:
    """Read one archived Bybit linear ticker, if it is a contract that expires.

    The mirror image of `funding_rates.extract_bybit_funding`, and it decides
    the same way that one does: on the venue's own say-so. A perpetual carries a
    non-empty `fundingRate`; an expiring contract carries an empty one and a
    non-zero `deliveryTime`. Both conditions are required rather than either,
    because a perpetual with a momentarily blank rate must not become a curve
    point with a delivery date of 1970.
    """
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(body, dict) or "markPrice" not in body:
        return []

    rate = body.get("fundingRate")
    if not (isinstance(rate, str) and rate == ""):
        # A perpetual. `funding` owns it, and this is not an error either.
        return []

    delivery_ms = body.get("deliveryTime")
    try:
        delivery_ms = int(delivery_ms) if delivery_ms not in (None, "") else 0
    except (TypeError, ValueError):
        return []
    if delivery_ms <= 0:
        # No funding AND no delivery date. Not a perpetual and not a dated
        # contract either - refused rather than filed as either one.
        return []

    venue_time_ms = body.get("time")
    if not isinstance(venue_time_ms, int):
        return []

    mark = _decimal_or_none(body.get("markPrice"))
    if mark is None:
        return []

    return [DatedFutureObservation(
        symbol=body.get("symbol") or symbol,
        venue=venue,
        mark_price=mark,
        index_price=_decimal_or_none(body.get("indexPrice")),
        last_price=_decimal_or_none(body.get("lastPrice")),
        delivery_time_ns=delivery_ms * _MS_TO_NS,
        event_time_ns=venue_time_ms * _MS_TO_NS,
        ingestion_time_ns=int(entry.t_recv_ns),
        open_interest=_decimal_or_none(body.get("openInterest")),
        venue_basis=_decimal_or_none(body.get("basis")),
        venue_basis_rate=_decimal_or_none(body.get("basisRate")),
        venue_basis_rate_year=_decimal_or_none(body.get("basisRateYear")),
    )]


_DECIMAL_COLUMNS = ("mark_price", "index_price", "last_price", "open_interest",
                    "venue_basis", "venue_basis_rate", "venue_basis_rate_year")


def build_dated_futures_frame(
        observations: Iterable[DatedFutureObservation]) -> pd.DataFrame:
    """One bitemporal row per observation.

    Prices travel as their own text, like every other Decimal in this store:
    a float mark price on a five-figure contract loses the cents that the basis
    is measured in.
    """
    rows = list(observations)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: o.symbol,
        VENUE: o.venue,
        "mark_price": str(o.mark_price),
        "index_price": None if o.index_price is None else str(o.index_price),
        "last_price": None if o.last_price is None else str(o.last_price),
        "open_interest": None if o.open_interest is None else str(o.open_interest),
        "venue_basis": None if o.venue_basis is None else str(o.venue_basis),
        "venue_basis_rate": (None if o.venue_basis_rate is None
                             else str(o.venue_basis_rate)),
        "venue_basis_rate_year": (None if o.venue_basis_rate_year is None
                                  else str(o.venue_basis_rate_year)),
        "delivery_time_ns": o.delivery_time_ns,
        EVENT_TIME: o.event_time_ns,
        INGESTION_TIME: o.ingestion_time_ns,
        # A polled value is knowable the moment it lands. Same rule as funding:
        # availability is ingestion exactly, never the venue's stamp, which runs
        # ahead of our receipt by the round trip.
        AVAILABILITY_TIME: o.ingestion_time_ns,
    } for o in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME,
                   "delivery_time_ns"):
        frame[column] = frame[column].astype("int64")
    # Pinned to string even when a whole partition is null, for the reason the
    # funding dataset learned the hard way: pyarrow infers an all-null column as
    # type `null`, and the next partition that carries a value cannot unify with
    # it - `Unsupported cast from large_string to null` made one venue's absent
    # field break every venue's read.
    for column in _DECIMAL_COLUMNS:
        frame[column] = frame[column].astype("string")
    return frame
