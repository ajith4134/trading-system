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


class BarAccumulator:
    """Folds trades into their bars as they arrive, holding no tape.

    `build_bars_for_day` accumulated every trade of a batch into one Python list
    and aggregated at the end. On 2026-08-10 the binance build for 2026-08-09
    was OOM-killed at 20.3 GB on a 29 GB box and left 69 of 569 symbols unbuilt,
    with nothing saying the day was partial rather than absent.

    ## What was measured, including the part that is still wrong

    Three hypotheses, in the order they were tried, on the same symbol -
    TUTUSDT, 226 MB of compressed tape carrying **25,963,564 trades** in one
    day, which is what the old code was trying to hold as Python objects:

    1. **The per-trade dicts in `build_bars`.** Building the frame column by
       column instead removed a full copy. Measured: the build still OOM-killed,
       at 12.0 GB. Removing a copy of the thing that does not fit still does not
       fit.
    2. **The list of `Trade` objects.** That is this class. Measured: the same
       build COMPLETES, exit 0, peaking at 4.09 GB against 12.0 GB killed - and
       26 million trades reduce to 856 bars.
    3. **`capture.raw_writer.read_pair`, which returns `list[tuple[str,
       IndexEntry]]` - a whole hour file, every payload string and an object per
       frame, materialised before one trade is examined.** That is what the
       remaining 4.1 GB is, and it is not fixed here. Peak is set by the largest
       HOUR FILE in a batch, so a symbol's whole day no longer has to fit but
       one of its hours still does.

    So this bounds memory by the busiest hour rather than by the busiest day. It
    is an improvement with a number behind it, not a cure, and the next bound is
    named above rather than left for someone to rediscover from an OOM log.

    ## The ordering this has to reproduce exactly

    The frame version sorted by event time with a STABLE sort and took `first`
    and `last` per group, so:

    * open is the price of the earliest event time, and among ties the trade
      seen first;
    * close is the price of the latest event time, and among ties the trade seen
      last.

    That is not pedantry. Venue frames arrive out of event order - hyperliquid
    reconnect backfill was measured carrying trades spanning 32.4 seconds of
    venue time in one frame - so "the last trade added" and "the last trade of
    the minute" are different rows, and taking the wrong one puts a stale price
    in the close that every return is computed from. Hence the strict `<` for
    open and the `>=` for close below: they are what a stable sort does.
    """

    __slots__ = ("_interval_ns", "_bars")

    def __init__(self, interval_ns: int) -> None:
        self._interval_ns = int(interval_ns)
        # (symbol, venue, bar_open_ns) -> mutable bucket. A plain list rather
        # than a dataclass: there is one per bar and they are written on every
        # trade, so the attribute lookup and the object header both matter.
        # Layout: [open, open_event, high, low, close, close_event, volume,
        #          count, latest_ingestion]
        self._bars: dict[tuple[str, str, int], list] = {}

    def add(self, trade: Trade) -> None:
        bar_open = (trade.event_time_ns // self._interval_ns) * self._interval_ns
        key = (trade.symbol, trade.venue, bar_open)
        bucket = self._bars.get(key)
        if bucket is None:
            self._bars[key] = [trade.price, trade.event_time_ns, trade.price,
                               trade.price, trade.price, trade.event_time_ns,
                               trade.size, 1, trade.ingestion_time_ns]
            return
        price = trade.price
        if trade.event_time_ns < bucket[1]:
            bucket[0], bucket[1] = price, trade.event_time_ns
        if price > bucket[2]:
            bucket[2] = price
        if price < bucket[3]:
            bucket[3] = price
        if trade.event_time_ns >= bucket[5]:
            bucket[4], bucket[5] = price, trade.event_time_ns
        bucket[6] += trade.size
        bucket[7] += 1
        if trade.ingestion_time_ns > bucket[8]:
            bucket[8] = trade.ingestion_time_ns

    def extend(self, trades: Iterable[Trade]) -> None:
        add = self.add
        for trade in trades:
            add(trade)

    def __len__(self) -> int:
        return len(self._bars)

    def to_frame(self) -> pd.DataFrame:
        """The bars, ordered as the grouped-and-sorted frame version ordered them."""
        if not self._bars:
            # No trades is not a flat bar. Zero-filling would invent liquidity
            # that a backtest would then assume it could trade against.
            return pd.DataFrame()

        keys = sorted(self._bars)
        buckets = [self._bars[key] for key in keys]
        frame = pd.DataFrame.from_dict({
            SYMBOL: [key[0] for key in keys],
            VENUE: [key[1] for key in keys],
            "open": [b[0] for b in buckets],
            "high": [b[2] for b in buckets],
            "low": [b[3] for b in buckets],
            "close": [b[4] for b in buckets],
            "volume": [b[6] for b in buckets],
            "trades": [b[7] for b in buckets],
            EVENT_TIME: [key[2] for key in keys],
            INGESTION_TIME: [b[8] for b in buckets],
        })
        bar_close = (frame[EVENT_TIME] + self._interval_ns).astype("int64")
        # The whole point of this module: a bar is not available at its close if
        # a trade composing it arrived later than that. np.maximum on two int64
        # arrays is an unambiguous element-wise max - no reduction axis to get
        # backwards, unlike a DataFrame.max(axis=...) which silently answers the
        # wrong question if the axis argument is ever transposed.
        frame[AVAILABILITY_TIME] = np.maximum(
            bar_close.to_numpy(), frame[INGESTION_TIME].to_numpy()).astype("int64")
        return frame


def build_bars(trades: Iterable[Trade], interval_ns: int) -> pd.DataFrame:
    """OHLCV per (symbol, venue, interval), stamped with when each bar became knowable.

    Kept as the one-shot form for callers that already hold their trades - the
    tests, and anything with a list small enough not to matter. A caller reading
    a day off disk should feed a `BarAccumulator` instead and never build the
    list at all.
    """
    accumulator = BarAccumulator(interval_ns)
    accumulator.extend(trades)
    return accumulator.to_frame()
