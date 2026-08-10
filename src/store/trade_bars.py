"""Raw venue frames to OHLCV bars, each carrying when it became knowable.

The load-bearing line in this module is one expression:

    availability = max(bar_close, latest ingestion of the trades in the bar)

A bar is not usable at its close if the trades composing it had not arrived by
then. Measured on the real archive (2026-08-02): binance trade frames land 70 ms
after the venue timestamp at the median and never later than 205 ms, but 4 of
6,548 hyperliquid frames arrived over 10 seconds late, one by 33.8 seconds, and a
single frame carried trades spanning 32.4 seconds of venue time. Those are
reconnect backfill, they are rare, and they are exactly the rows that make a
backtest look better than the system can be.

Venue formats differ structurally and are not unified by guessing: binance sends
one trade per frame, hyperliquid sends an array. An unrecognised venue raises
rather than returning nothing, because a silent empty list reads downstream as a
quiet market.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from capture.frame_codec import IndexEntry
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

_MS_TO_NS = 1_000_000


@dataclass(frozen=True)
class Trade:
    symbol: str
    venue: str
    price: float
    size: float
    event_time_ns: int
    ingestion_time_ns: int


class UnknownVenueFormat(ValueError):
    """No extractor for this venue's frame shape.

    Raising here, instead of returning an empty list, is deliberate: a venue
    silently yielding zero trades reads downstream as a quiet market rather than
    as a gap in this module's coverage, and the two must never look the same.
    """


def _extract_binance(body: dict, entry: IndexEntry, symbol: str, venue: str) -> list[Trade]:
    data = body.get("data", {})
    if data.get("e") != "trade":
        return []
    # "T" is trade time; "E" is event-emission time. They are equal on this feed
    # today, but T is the one that describes the trade itself.
    event_ms = data.get("T", data.get("E"))
    return [Trade(
        symbol=data.get("s", symbol),
        venue=venue,
        price=float(data["p"]),
        size=float(data["q"]),
        event_time_ns=int(event_ms) * _MS_TO_NS,
        ingestion_time_ns=entry.t_recv_ns,
    )]


def _extract_hyperliquid(body: dict, entry: IndexEntry, symbol: str, venue: str) -> list[Trade]:
    if body.get("channel") != "trades":
        return []
    trades = []
    for item in body.get("data", []) or []:
        if "px" not in item or "time" not in item:
            continue
        trades.append(Trade(
            symbol=item.get("coin", symbol),
            venue=venue,
            price=float(item["px"]),
            size=float(item["sz"]),
            event_time_ns=int(item["time"]) * _MS_TO_NS,
            # Every trade in the batch shares the frame's arrival: they became
            # knowable together, whatever their individual venue timestamps say.
            # Assigning each its own arrival would fabricate an arrival that
            # never happened.
            ingestion_time_ns=entry.t_recv_ns,
        ))
    return trades


def _extract_coinbase(body: dict, entry: IndexEntry, symbol: str, venue: str) -> list[Trade]:
    """One `match` frame, one trade.

    Two shapes carry a trade here and both are taken: `match` is the live tape,
    and `last_match` is the one frame the venue sends at subscribe carrying the
    most recent trade. Dropping `last_match` would discard the only trade some
    quiet products publish in an hour - and a product with no bar at all reads
    downstream as a product nobody captured.

    The timestamp is an ISO8601 string rather than a number, and it is parsed
    rather than approximated from receipt: this venue's clock is what orders its
    trades against the other three.
    """
    if body.get("type") not in ("match", "last_match"):
        return []
    price, size = body.get("price"), body.get("size")
    event_ms = _coinbase_event_ms(body.get("time"))
    if price is None or size is None or event_ms is None:
        # A trade missing its price, size or clock is not a trade. Refused
        # rather than defaulted - a zero-price fill would price everything that
        # reads the bar it landed in.
        return []
    return [Trade(
        symbol=body.get("product_id", symbol),
        venue=venue,
        price=float(price),
        size=float(size),
        event_time_ns=event_ms * _MS_TO_NS,
        ingestion_time_ns=entry.t_recv_ns,
    )]


def _coinbase_event_ms(value) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return int(stamp.timestamp() * 1000)


_EXTRACTORS = {
    "binance": _extract_binance,
    # Spot shares the futures frame shape exactly for the fields read above:
    # `e`, `T`, `s`, `p`, `q`. It adds `M` and omits `X`/`st`, none of which are
    # touched. Verified against a captured frame rather than assumed, because the
    # cost of being wrong is a venue that reads as a quiet market.
    #
    # It stays a separate venue rather than folding into "binance": BTCUSDT is
    # listed on both and they are different instruments with different prices and
    # different fees. `build_bars` keys on (symbol, venue, bar) and the reader
    # de-duplicates on (symbol, venue, event_time), so the two remain distinct
    # everywhere downstream - but only because the venue is carried, never
    # normalised away.
    "binance-spot": _extract_binance,
    "hyperliquid": _extract_hyperliquid,
    # Spot, and the only tape here that is not binance. Its frame shape shares
    # nothing with either: `type`, `product_id`, `price`, `size` and an ISO8601
    # `time`, against binance's single-letter keys and millisecond integers.
    "coinbase": _extract_coinbase,
}


def is_tradeable(trade: Trade) -> bool:
    """Whether this row describes a transaction that actually happened.

    Binance emits frames on its `trade` stream that are shaped like trades and are
    not: price "0", quantity "0", `X` "NA". Captured verbatim from the live tape,
    2026-08-03T18 on BTCUSDT, 60 of 17,227 frames in that hour. Layer 0 is right to
    store them - it records what the venue said, never what we wish it had said -
    and this is the layer that decides what they mean.

    Measured damage before this existed: 746 of the real store's 1,671 bars carried
    `low <= 0` - BTCUSDT 237 of 278, ETHUSDT 268 of 278, SOLUSDT 241 of 278, with 25
    zero opens and 19 zero closes. Hyperliquid's 837 bars were clean, because only
    this feed emits them. `low=("price", "min")` needs one zero to ruin a bar, and
    the bar looks entirely normal otherwise: correct open, correct high, hundreds of
    trades. It was found by a paper-plumbing run refusing to divide by a zero price,
    not by anything watching the store.

    A non-positive SIZE is refused on the same grounds: a fill of nothing is not a
    fill, and it would inflate the trade count while contributing no volume.
    """
    return trade.price > 0 and trade.size > 0


def extract_trades(payload: str, entry: IndexEntry, venue: str, symbol: str) -> list[Trade]:
    """Trades carried by one captured frame. Raises on a venue with no extractor.

    Rows the venue reported but which cannot be transactions are dropped here, by
    `is_tradeable`, rather than in the aggregation - a zero that reaches `min()` is
    already indistinguishable from a cheap fill.
    """
    extractor = _EXTRACTORS.get(venue)
    if extractor is None:
        raise UnknownVenueFormat(
            f"no trade extractor for venue '{venue}'; add one rather than letting "
            f"its frames read downstream as a quiet market")
    if entry.kind != "data":
        return []
    try:
        body = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return [trade for trade in extractor(body, entry, symbol, venue)
            if is_tradeable(trade)]


def build_bars(trades: Iterable[Trade], interval_ns: int) -> pd.DataFrame:
    """OHLCV per (symbol, venue, interval), stamped with when each bar became knowable."""
    rows = list(trades)
    if not rows:
        # No trades is not a flat bar. Zero-filling would invent liquidity that
        # a backtest would then assume it could trade against.
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: t.symbol, VENUE: t.venue, "price": t.price, "size": t.size,
        EVENT_TIME: t.event_time_ns, INGESTION_TIME: t.ingestion_time_ns,
    } for t in rows])
    frame["bar_open_ns"] = (frame[EVENT_TIME] // interval_ns) * interval_ns

    # Sorted by event time so open and close follow the venue's clock rather than
    # the order frames happened to arrive in.
    frame = frame.sort_values(EVENT_TIME, kind="mergesort")

    grouped = frame.groupby([SYMBOL, VENUE, "bar_open_ns"], sort=True)
    bars = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trades=("price", "size"),
        latest_ingestion=(INGESTION_TIME, "max"),
    ).reset_index()

    bar_close = (bars["bar_open_ns"] + interval_ns).astype("int64")
    bars[EVENT_TIME] = bars["bar_open_ns"].astype("int64")
    bars[INGESTION_TIME] = bars["latest_ingestion"].astype("int64")
    # The whole point of this module: a bar is not available at its close if a
    # trade composing it arrived later than that. np.maximum on two int64 arrays
    # is an unambiguous element-wise max - no reduction axis to get backwards,
    # unlike a DataFrame.max(axis=...) which silently answers the wrong question
    # if the axis argument is ever transposed.
    bars[AVAILABILITY_TIME] = np.maximum(
        bar_close.to_numpy(), bars["latest_ingestion"].to_numpy()
    ).astype("int64")

    return bars.drop(columns=["bar_open_ns", "latest_ingestion"]).reset_index(drop=True)
