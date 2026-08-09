"""Hyperliquid funding into the same dataset as Binance, without pretending the
two venues are the same thing.

Captured from 2026-08-09. Three of the differences are real rather than
cosmetic, and each one is a column rather than a lossy mapping:

  - it funds on the ORACLE price hourly, capped at 4%/hour; Binance funds on
    MARK every eight hours, and the gap between those prices is the basis trade
  - an asset context carries NO timestamp of any kind, so the event time is our
    receipt and has to say so
  - it publishes no next-settlement time, and deriving one from the venue's
    hourly schedule would put an assumption into a row that reads as an
    observation
"""
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from store.build_polled import DATASETS, source_for, polled_symbols
from store.funding_rates import (
    FUNDS_ON_MARK, FUNDS_ON_ORACLE, build_funding_frame, extract_funding,
    extract_hyperliquid_funding,
)

RECEIVED_NS = 1786276800_000_000_000


@dataclass(frozen=True)
class _Entry:
    t_recv_ns: int


# The real shape, copied from a live poll on 2026-08-09.
LIVE_CTX = json.dumps({
    "coin": "BTC", "funding": "0.0000036476", "openInterest": "34493.53682",
    "prevDayPx": "64926.0", "dayNtlVlm": "384537761.47", "premium": "-0.0004488933",
    "oraclePx": "64826.1", "markPx": "64795.0", "midPx": "64796.5",
    "impactPxs": ["64796.0", "64797.0"], "dayBaseVlm": "5918.33974",
})


def _one(payload=LIVE_CTX):
    return extract_hyperliquid_funding(payload, _Entry(RECEIVED_NS), "hyperliquid", "BTC")[0]


def test_the_rate_and_both_prices_survive():
    o = _one()
    assert o.symbol == "BTC"
    assert o.funding_rate == Decimal("0.0000036476")
    assert o.mark_price == Decimal("64795.0")
    assert o.oracle_price == Decimal("64826.1")


def test_it_records_that_it_funds_on_the_oracle():
    """A cross-venue carry number that does not know which price each side was
    funded against is comparing two different quantities."""
    assert _one().funds_on == FUNDS_ON_ORACLE


def test_no_index_price_is_none_rather_than_a_copy_of_mark():
    """This venue publishes no index. Filling it from a neighbouring price is
    the kind of quiet substitution that reads as data forever after."""
    assert _one().index_price is None


def test_the_event_time_is_our_receipt_and_says_so():
    """An asset context carries no clock. Letting the receipt pass as a venue
    stamp would leave a consumer comparing event times across venues with no way
    to tell it was comparing a stamp against a receipt."""
    o = _one()
    assert o.event_time_ns == RECEIVED_NS
    assert o.ingestion_time_ns == RECEIVED_NS
    assert o.event_time_is_receipt is True


def test_the_next_settlement_is_unknown_rather_than_zero():
    """Hyperliquid settles hourly, so it IS derivable - and deriving it here
    would put an assumption about venue behaviour into a row that reads like an
    observation."""
    o = _one()
    assert o.next_funding_time_unknown is True
    assert o.next_funding_time_ns == 0


def test_a_frame_that_is_not_a_funding_context_is_ignored():
    """The archive holds several streams side by side, and misreading a trade as
    a funding rate poisons every carry cost downstream."""
    trade = json.dumps({"coin": "BTC", "px": "64795.0", "sz": "0.1"})
    assert extract_hyperliquid_funding(trade, _Entry(1), "hyperliquid", "BTC") == []
    assert extract_hyperliquid_funding("not json", _Entry(1), "hyperliquid", "BTC") == []
    assert extract_hyperliquid_funding("[]", _Entry(1), "hyperliquid", "BTC") == []


def test_binance_still_reads_as_funding_on_mark():
    """The venue that was here first must be unchanged by the one that arrived."""
    body = json.dumps({"symbol": "BTCUSDT", "markPrice": "64965.9",
                       "indexPrice": "65007.98", "lastFundingRate": "0.00004093",
                       "nextFundingTime": 1786291200000, "time": 1786276800000})
    o = extract_funding(body, _Entry(RECEIVED_NS), "binance", "BTCUSDT")[0]

    assert o.funds_on == FUNDS_ON_MARK
    assert o.index_price == Decimal("65007.98")
    assert o.oracle_price is None
    assert o.event_time_is_receipt is False
    assert o.next_funding_time_unknown is False


def test_a_partition_of_only_nulls_still_has_a_string_column():
    """This broke the live dataset. Hyperliquid publishes no index price, so its
    first partition had `index_price` null for all 3,248 rows, pyarrow inferred
    the column type as `null`, and reading the dataset afterwards failed with
    `Unsupported cast from large_string to null`. One venue's absent field made
    every venue's funding unreadable.
    """
    frame = build_funding_frame([_one()])
    assert str(frame["index_price"].dtype) == "string"
    assert str(frame["oracle_price"].dtype) == "string"
    assert frame["index_price"].isna().all()


def test_both_venues_write_a_schema_that_unifies(tmp_path):
    """The real failure was across partitions, so the test has to be too."""
    from store.parquet_partition import append_partition, read_dataset

    binance_body = json.dumps({"symbol": "BTCUSDT", "markPrice": "64965.9",
                               "indexPrice": "65007.98", "lastFundingRate": "0.00004",
                               "nextFundingTime": 1786291200000, "time": 1786276800000})
    append_partition(tmp_path, "funding", build_funding_frame(
        extract_funding(binance_body, _Entry(RECEIVED_NS), "binance", "BTCUSDT")), "b")
    append_partition(tmp_path, "funding", build_funding_frame([_one()]), "h")

    both = read_dataset(tmp_path, "funding")
    assert sorted(both["venue"].unique()) == ["binance", "hyperliquid"]
    assert set(both["funds_on"]) == {FUNDS_ON_MARK, FUNDS_ON_ORACLE}


def test_the_dataset_routes_each_venue_to_its_own_stream_and_reader():
    """One dataset, several venue models. Splitting them into two datasets would
    push "which venue am I holding" onto every consumer, and cross-venue carry
    is a strategy in its own right (ledger SP-059)."""
    assert source_for("funding", "binance") == (DATASETS["funding"][0],
                                                DATASETS["funding"][1])
    stream, reader = source_for("funding", "hyperliquid")
    assert stream == "assetCtx"
    assert reader is extract_hyperliquid_funding


def test_the_polled_universe_is_read_off_the_archive(tmp_path):
    """A hand-kept list is what cost the trade tape 99.6% of itself. Funding now
    covers 863 instruments on binance and no list is going to track that."""
    folder = tmp_path / "raw" / "hyperliquid" / "2026-08-09"
    folder.mkdir(parents=True)
    for name in ("assetCtx_BTC_2026-08-09T00.ndjson.zst",
                 "assetCtx_ETH_2026-08-09T01.ndjson.zst",
                 "trades_BTC_2026-08-09T00.ndjson.zst"):
        (folder / name).write_bytes(b"x")

    assert polled_symbols(tmp_path, "hyperliquid", "2026-08-09", "assetCtx") == ["BTC", "ETH"]
