"""The basis is one subtraction; the module exists to refuse the wrong one.

Each venue's basis is priced against the reference it actually funds on - the
declared `funds_on` in the dataset row, never the venue name and never a shared
assumption. The fixtures use the real dataset schema and the real store writer,
so a schema drift breaks these tests rather than silently breaking the basis.
"""
from decimal import Decimal
from pathlib import Path

import pandas as pd

from features.spot_perp_basis import compute_spot_perp_basis
from store.parquet_partition import append_partition


def _funding_row(venue, symbol, funds_on, mark, index, oracle, event_ns,
                 avail_ns=None):
    return {
        "venue": venue, "symbol": symbol, "funds_on": funds_on,
        "funding_rate": Decimal("0.0001"),
        "mark_price": mark, "index_price": index, "oracle_price": oracle,
        "event_time_is_receipt": False, "next_funding_time_unknown": False,
        "next_funding_time_ns": event_ns + 1, "funding_interval_hours": 8,
        "event_time_ns": event_ns,
        "ingestion_time_ns": event_ns,
        "availability_time_ns": avail_ns if avail_ns is not None else event_ns,
    }


def _write_funding(tmp_path, rows) -> Path:
    frame = pd.DataFrame(rows)
    append_partition(tmp_path, "funding", frame, snapshot_id="basis-test")
    return tmp_path


def test_each_venue_is_priced_against_the_reference_it_funds_on(tmp_path):
    store = _write_funding(tmp_path, [
        # mark 101 vs index 100 -> +100 bps
        _funding_row("binance", "BTCUSDT", "mark",
                     Decimal("101.000"), Decimal("100.000"), None, 1_000),
        # mark 198 vs oracle 200 -> -100 bps; the index column is absent, and
        # for an oracle venue that must not matter. (Both marks carry six
        # digits: pyarrow infers decimal precision per part, and per-symbol
        # parts with different precisions refuse to merge on read.)
        _funding_row("hyperliquid", "BTC", "oracle",
                     Decimal("198.000"), None, Decimal("200.000"), 1_000),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    by_venue = {row.venue: row for row in table.rows.itertuples()}
    assert by_venue["binance"].basis_bps == Decimal("100.000")
    assert by_venue["binance"].reference == "index_price"
    assert by_venue["hyperliquid"].basis_bps == Decimal("-100")
    assert by_venue["hyperliquid"].reference == "oracle_price"
    assert sum(table.refused.values()) == 0


def test_missing_reference_is_refused_and_counted_never_defaulted(tmp_path):
    store = _write_funding(tmp_path, [
        # an oracle venue whose oracle is missing - pricing it against the
        # index that happens to be present would be the venue-blind basis
        # this module exists to refuse
        _funding_row("hyperliquid", "BTC", "oracle",
                     Decimal("99.000"), Decimal("100.000"), None, 1_000),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    assert len(table.rows) == 0
    assert table.refused["no_reference_price"] == 1


def test_unrecognised_funds_on_is_refused_not_guessed(tmp_path):
    store = _write_funding(tmp_path, [
        _funding_row("newvenue", "XUSDT", "twap",
                     Decimal("101.000"), Decimal("100.000"), Decimal("100.000"), 1_000),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    assert len(table.rows) == 0
    assert table.refused["unrecognised_funds_on"] == 1


def test_zero_reference_is_refused_not_divided_by(tmp_path):
    store = _write_funding(tmp_path, [
        _funding_row("binance", "BADUSDT", "mark",
                     Decimal("101.000"), Decimal("0.000"), None, 1_000),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    assert len(table.rows) == 0
    assert table.refused["unparseable_price"] == 1


def test_the_clock_gates_the_basis(tmp_path):
    """A row available after the as-of clock does not exist for this read."""
    store = _write_funding(tmp_path, [
        _funding_row("binance", "BTCUSDT", "mark",
                     Decimal("101.000"), Decimal("100.000"), None, 1_000,
                     avail_ns=1_000),
        _funding_row("binance", "BTCUSDT", "mark",
                     Decimal("150.000"), Decimal("100.000"), None, 3_000,
                     avail_ns=3_000),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    assert len(table.rows) == 1
    assert table.rows.iloc[0].basis_bps == Decimal("100.000"), (
        "the 5000-bps row lands at 3000ns and must be invisible at 2000ns")


def test_newest_row_per_symbol_wins(tmp_path):
    store = _write_funding(tmp_path, [
        _funding_row("binance", "BTCUSDT", "mark",
                     Decimal("101.000"), Decimal("100.000"), None, 1_000),
        _funding_row("binance", "BTCUSDT", "mark",
                     Decimal("102.000"), Decimal("100.000"), None, 1_500),
    ])
    table = compute_spot_perp_basis(store, as_of_ns=2_000)
    assert len(table.rows) == 1
    assert table.rows.iloc[0].basis_bps == Decimal("200")


def test_an_empty_store_returns_an_empty_table_with_zero_refusals(tmp_path):
    table = compute_spot_perp_basis(tmp_path, as_of_ns=2_000)
    assert len(table.rows) == 0
    assert sum(table.refused.values()) == 0
