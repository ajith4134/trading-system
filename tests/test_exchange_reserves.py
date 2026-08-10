"""Exchange reserves and netflow, and the number the prior art invented.

The row DM-025 points at (`analysis/onchain_advanced.py`, early repos) returned
`circulating_supply * 0.15` as `total_reserve` and a hardcoded `0` as the 24h
change whenever it had no API key. `test_an_exchange_with_nothing_measured_is_
refused_not_zeroed` is that defect written as a test: an exchange holding
billions must never be reported as holding nothing.

Every fixture below is a real row from the live source, 2026-08-10.
"""
import json

import pandas as pd
import pytest

from store.exchange_reserves import (
    DATASET, SOURCE, ExchangeReserve, build_reserves_frame,
    parse_cex_transparency, poll_exchange_reserves,
)
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
    validate_temporal_frame,
)

FETCHED_NS = 1_786_342_812_900_709_269

BINANCE = {
    "name": "Binance", "slug": "Binance-CEX", "coin": "BNB",
    "currentTvl": 139684360969.39758, "cleanAssetsTvl": 118724669008.50696,
    "inflows_24h": 4403471.619324267, "inflows_1w": 475612052.84525293,
    "inflows_1m": 175777146.42118216, "spotVolume": 4475543753.92344,
    "oi": 25798373658.25948,
}
BYBIT = {
    "name": "Bybit", "slug": "bybit", "coin": "BIT",
    "currentTvl": 13073538118.423311, "cleanAssetsTvl": 11908975092.202686,
    "inflows_24h": -27319332.16195826, "inflows_1w": -75111070.0,
    "inflows_1m": -1169929341.0, "spotVolume": 642301629.98, "oi": 6674871281.15,
}
# Listed by name and measured for nothing - it publishes no wallet set. Eleven
# others on the same response look identical.
COINBASE = {
    "name": "Coinbase", "slug": None, "currentTvl": None,
    "cleanAssetsTvl": None, "inflows_24h": None, "inflows_1w": None,
    "inflows_1m": None,
}


def _payload(*entries):
    return json.dumps({"cexs": list(entries)})


def _parse(*entries):
    return parse_cex_transparency(_payload(*entries), FETCHED_NS)


# --- reading the response -------------------------------------------------

def test_an_exchange_becomes_one_observation():
    rows, refused = _parse(BINANCE)
    assert len(rows) == 1
    row = rows[0]
    assert row.exchange == "Binance"
    assert row.total_reserve_usd == pytest.approx(139684360969.39758)
    assert row.clean_reserve_usd == pytest.approx(118724669008.50696)
    assert refused == {"no_total_reserve": 0, "no_name": 0, "malformed": 0}


def test_an_exchange_with_nothing_measured_is_refused_not_zeroed():
    """The defect this module exists not to repeat. A zero reserve for an
    exchange holding billions is the most dangerous number here, and the prior
    art produced exactly it - `circulating_supply * 0.15` and a hardcoded zero
    change - whenever its paid source was unavailable."""
    rows, refused = _parse(BINANCE, COINBASE)

    assert [r.exchange for r in rows] == ["Binance"]
    assert refused["no_total_reserve"] == 1


def test_a_refusal_is_counted_because_a_silent_drop_looks_like_a_delisting():
    rows, refused = _parse(COINBASE, COINBASE)
    assert rows == []
    assert refused["no_total_reserve"] == 2


def test_a_malformed_payload_is_refused_rather_than_raising():
    assert parse_cex_transparency("not json", FETCHED_NS)[0] == []
    assert parse_cex_transparency(json.dumps([1, 2]), FETCHED_NS)[0] == []
    assert parse_cex_transparency(json.dumps({"cexs": "nope"}), FETCHED_NS)[0] == []


def test_a_nan_reserve_is_absent_not_zero():
    broken = dict(BINANCE, cleanAssetsTvl=float("nan"))
    rows, _ = _parse(broken)
    assert rows[0].clean_reserve_usd is None


def test_a_row_without_a_name_cannot_be_attributed_to_anyone():
    rows, refused = _parse(dict(BINANCE, name=""))
    assert rows == []
    assert refused["no_name"] == 1


# --- what the two reserve numbers are for ---------------------------------

def test_the_own_token_share_is_the_gap_between_total_and_clean():
    """Collateral the exchange printed, priced at market. That arithmetic
    ended FTX, and Binance carries 15% of $139.7bn in it."""
    rows, _ = _parse(BINANCE)
    assert rows[0].own_token_share == pytest.approx(0.1500, abs=0.0005)


def test_a_missing_clean_figure_leaves_the_share_unknown_not_zero():
    """Zero here reads as an exchange holding no self-issued collateral - the
    most reassuring thing this module could say and the least safe to guess."""
    rows, _ = _parse(dict(BINANCE, cleanAssetsTvl=None))
    assert rows[0].clean_reserve_usd is None
    assert rows[0].own_token_share is None


def test_netflow_keeps_its_sign_with_the_venue_as_the_destination():
    """Positive is value moving TO the exchange. A convention nobody wrote
    down is one that gets read backwards, and the sign is the whole content of
    the number: Bybit was losing $27m over 24h on this reading."""
    rows, _ = _parse(BINANCE, BYBIT)
    by_name = {r.exchange: r for r in rows}
    assert by_name["Binance"].netflow_24h_usd > 0
    assert by_name["Bybit"].netflow_24h_usd == pytest.approx(-27319332.16195826)


# --- the stored frame -----------------------------------------------------

def test_the_frame_satisfies_the_temporal_contract():
    rows, _ = _parse(BINANCE)
    frame = build_reserves_frame(rows)
    validate_temporal_frame(frame)
    assert frame.iloc[0][VENUE] == "Binance-CEX"
    assert frame.iloc[0][SYMBOL] == "ALL"


def test_the_reading_is_stamped_at_the_only_instant_it_can_honestly_claim():
    """The source publishes no timestamp of its own, so event time is the
    fetch too. Claiming an event time we were not given is the same error as
    stamping a backfill at bar close."""
    rows, _ = _parse(BINANCE)
    row = build_reserves_frame(rows).iloc[0]
    assert row[EVENT_TIME] == row[INGESTION_TIME] == row[AVAILABILITY_TIME] == FETCHED_NS


def test_every_row_names_whose_arithmetic_it_is():
    """A third party's measurement of somebody else's wallets, not the venue's
    own statement - carried per row so a frame filtered or joined elsewhere
    still says so."""
    rows, _ = _parse(BINANCE, BYBIT)
    frame = build_reserves_frame(rows)
    assert (frame["source"] == SOURCE).all()


def test_no_rows_is_an_empty_frame():
    assert build_reserves_frame([]).empty


# --- polling --------------------------------------------------------------

def test_a_poll_lands_as_one_partition(tmp_path):
    result = poll_exchange_reserves(
        tmp_path, fetch=lambda url, timeout=25.0: _payload(BINANCE, BYBIT, COINBASE),
        now_ns=lambda: FETCHED_NS)

    assert result["exchanges"] == 2
    assert result["appended"] is True
    assert result["refused"]["no_total_reserve"] == 1
    assert (tmp_path / DATASET).exists()


def test_a_response_measuring_nothing_writes_no_partition(tmp_path):
    """An empty poll is a fact about the source. A zero-row partition would
    look like a completed reading."""
    result = poll_exchange_reserves(
        tmp_path, fetch=lambda url, timeout=25.0: _payload(COINBASE),
        now_ns=lambda: FETCHED_NS)

    assert result["appended"] is False
    assert not (tmp_path / DATASET).exists()


def test_a_failed_fetch_is_reported_not_raised(tmp_path):
    """This runs on a loop beside builds that must not stop because a
    third-party endpoint had a bad minute."""
    def _boom(url, timeout=25.0):
        raise TimeoutError("read timed out")

    result = poll_exchange_reserves(tmp_path, fetch=_boom, now_ns=lambda: FETCHED_NS)

    assert result["appended"] is False
    assert "read timed out" in result["error"]
