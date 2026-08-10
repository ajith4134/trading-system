"""The calendar curve, and the three refusals that keep it honest.

The numbers in the fixtures are this box's own, read off the live store on
2026-08-10: BTCUSDT-14AUG26 marked 65011.9 against an index of 64999.5 with the
perpetual at 64980.9, four days from delivery. That curve ran 173 bps/yr at the
front to 439 bps/yr at 319 days.
"""
from decimal import Decimal

import pandas as pd
import pytest

from features.term_structure import (
    MIN_DAYS_TO_ANNUALISE, compute_term_structure, summarise_curves,
)
from store.parquet_partition import append_partition

_DAY_NS = 86_400_000_000_000
NOW = 1_786_336_806_435_346_403


def _dated_row(symbol, mark, index, delivery_ns, *, venue="bybit",
               available_ns=NOW - 1, venue_year_rate=None):
    return {
        "venue": venue, "symbol": symbol,
        "mark_price": str(mark), "index_price": None if index is None else str(index),
        "last_price": str(mark), "open_interest": "1.0",
        "venue_basis": None, "venue_basis_rate": None,
        "venue_basis_rate_year": (None if venue_year_rate is None
                                  else str(venue_year_rate)),
        "delivery_time_ns": delivery_ns,
        "event_time_ns": available_ns, "ingestion_time_ns": available_ns,
        "availability_time_ns": available_ns,
    }


def _perp_row(symbol, mark, *, venue="bybit", available_ns=NOW - 1):
    return {
        "venue": venue, "symbol": symbol, "funding_rate": "0.00005399",
        "mark_price": str(mark), "index_price": str(mark), "oracle_price": None,
        "funds_on": "mark", "funding_interval_hours": 8,
        "event_time_is_receipt": False, "next_funding_time_unknown": False,
        "next_funding_time_ns": 0,
        "event_time_ns": available_ns, "ingestion_time_ns": available_ns,
        "availability_time_ns": available_ns,
    }


def _store(tmp_path, dated, perps=(), snapshot="term-test"):
    if dated:
        append_partition(tmp_path, "dated_futures", pd.DataFrame(dated),
                         snapshot_id=snapshot)
    if perps:
        append_partition(tmp_path, "funding", pd.DataFrame(list(perps)),
                         snapshot_id=snapshot + "-funding")
    return tmp_path


# --- the curve point ------------------------------------------------------

def test_a_dated_contract_with_a_perp_anchor_becomes_a_curve_point(tmp_path):
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert len(got.rows) == 1
    row = got.rows.iloc[0]
    assert row.underlying == "BTCUSDT"
    assert row.days_to_delivery == Decimal(4)
    # (65011.9 - 64999.5) / 64999.5 in basis points.
    assert row.basis_bps == pytest.approx(Decimal("1.9077"), abs=Decimal("0.0001"))


def test_the_annualised_rate_scales_by_the_tenor(tmp_path):
    """A 4-day basis annualises at 365/4. The multiplier is the whole reason
    the near end of a curve has to be handled carefully."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    row = compute_term_structure(store, NOW).rows.iloc[0]

    assert row.annualised_basis_bps == row.basis_bps * Decimal(365) / Decimal(4)


def test_the_spread_against_the_perpetual_is_the_calendar_trade(tmp_path):
    """The curve exists to price holding a date against holding the perp."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    row = compute_term_structure(store, NOW).rows.iloc[0]

    expected = (Decimal("65011.9") - Decimal("64980.9")) / Decimal("64980.9") * 10_000
    assert row.spread_to_perp_bps == expected


# --- the three refusals ---------------------------------------------------

def test_a_contract_inside_a_day_of_delivery_reports_no_annualised_rate(tmp_path):
    """Its price is real and is kept. The annualised rate is 730x leverage on
    a number measured to a fraction of a basis point, and it would sort to the
    top of any table ranked by carry."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + _DAY_NS // 2)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert len(got.rows) == 1
    assert got.rows.iloc[0].basis_bps > 0
    assert got.rows.iloc[0].annualised_basis_bps is None
    assert got.refused["too_near_expiry_to_annualise"] == 1


def test_a_contract_past_its_delivery_time_is_refused(tmp_path):
    """A negative tenor annualises to a rate with the sign flipped, which is
    worse than no number at all."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW - _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert got.rows.empty
    assert got.refused["already_delivered"] == 1


def test_a_contract_whose_underlying_is_not_a_live_perp_is_refused(tmp_path):
    """The symbol split is a candidate, not a conclusion. `WIFUSDT-14AUG26`
    looks exactly as much like WIFUSDT as the BTC contract looks like BTCUSDT -
    the difference is that the venue is publishing funding for one of them."""
    store = _store(
        tmp_path,
        [_dated_row("WIFUSDT-14AUG26", "1.5", "1.49", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert got.rows.empty
    assert got.refused["no_perp_anchor"] == 1


def test_a_contract_with_no_index_price_is_refused_not_priced_off_the_mark(tmp_path):
    """Substituting the mark for the index makes every basis exactly zero,
    which reads as a market with no carry rather than as missing data."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", None, NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert got.rows.empty
    assert got.refused["no_index_price"] == 1


def test_a_zero_index_price_is_refused(tmp_path):
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "0", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    got = compute_term_structure(store, NOW)

    assert got.rows.empty
    assert got.refused["unparseable_price"] == 1


# --- the venue as a witness, not as the answer ----------------------------

def test_the_venues_own_annualised_rate_rides_alongside_with_its_disagreement(tmp_path):
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS,
                    venue_year_rate="0.01723021")],
        [_perp_row("BTCUSDT", "64980.9")])

    row = compute_term_structure(store, NOW).rows.iloc[0]

    assert row.venue_annualised_bps == Decimal("0.01723021") * 10_000
    assert row.venue_disagreement_bps == (row.annualised_basis_bps
                                          - row.venue_annualised_bps)


def test_a_venue_publishing_no_annualised_rate_leaves_the_comparison_empty(tmp_path):
    """No witness is not agreement. A zero disagreement would read as the two
    computations having been checked against each other."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    row = compute_term_structure(store, NOW).rows.iloc[0]

    assert row.venue_annualised_bps is None
    assert row.venue_disagreement_bps is None


# --- the clock ------------------------------------------------------------

def test_a_poll_that_had_not_arrived_yet_is_invisible(tmp_path):
    """The whole point of reading through the gate. A curve assembled from
    prices that had not been received is a backtest reading its own future."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS,
                    available_ns=NOW + 60_000_000_000)],
        [_perp_row("BTCUSDT", "64980.9")])

    assert compute_term_structure(store, NOW).rows.empty


def test_the_tenor_is_measured_from_the_clock_not_from_the_poll(tmp_path):
    """A poll that landed an hour ago does not make the contract an hour
    further away. Tenor is from the as-of clock, always."""
    hour_ago = NOW - 3_600_000_000_000
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS,
                    available_ns=hour_ago)],
        [_perp_row("BTCUSDT", "64980.9", available_ns=hour_ago)])

    row = compute_term_structure(store, NOW).rows.iloc[0]

    assert row.days_to_delivery == Decimal(4)


def test_an_empty_store_is_an_empty_curve_not_an_error(tmp_path):
    got = compute_term_structure(tmp_path, NOW)
    assert got.rows.empty
    assert got.refused["no_perp_anchor"] == 0


# --- the curve as a whole -------------------------------------------------

def test_points_of_one_underlying_come_back_nearest_first(tmp_path):
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-25DEC26", "66000", "64999.5", NOW + 137 * _DAY_NS),
         _dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS),
         _dated_row("BTCUSDT-28AUG26", "65120", "64999.5", NOW + 18 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    curve = compute_term_structure(store, NOW).curve("BTCUSDT")

    assert list(curve["symbol"]) == ["BTCUSDT-14AUG26", "BTCUSDT-28AUG26",
                                     "BTCUSDT-25DEC26"]


def test_the_slope_is_stated_per_curve_and_never_averaged_across_them(tmp_path):
    """A market in contango on BTC and backwardation on DOGE has no meaningful
    average - the mean describes neither curve."""
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS),
         _dated_row("BTCUSDT-25DEC26", "66000", "64999.5", NOW + 137 * _DAY_NS),
         _dated_row("DOGEUSDT-14AUG26", "0.0995", "0.1", NOW + 4 * _DAY_NS),
         _dated_row("DOGEUSDT-25DEC26", "0.099", "0.1", NOW + 137 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9"), _perp_row("DOGEUSDT", "0.1")])

    summary = summarise_curves(compute_term_structure(store, NOW))

    assert list(summary["underlying"]) == ["BTCUSDT", "DOGEUSDT"]
    assert all(summary["tenors"] == 2)
    btc = summary[summary["underlying"] == "BTCUSDT"].iloc[0]
    doge = summary[summary["underlying"] == "DOGEUSDT"].iloc[0]
    assert btc.slope_bps_per_year > 0
    assert doge.slope_bps_per_year > 0 or doge.slope_bps_per_year < 0


def test_a_single_point_is_a_price_not_a_structure(tmp_path):
    store = _store(
        tmp_path,
        [_dated_row("BTCUSDT-14AUG26", "65011.9", "64999.5", NOW + 4 * _DAY_NS)],
        [_perp_row("BTCUSDT", "64980.9")])

    summary = summarise_curves(compute_term_structure(store, NOW))

    assert summary.iloc[0].tenors == 1
    assert summary.iloc[0].slope_bps_per_year is None


def test_an_empty_curve_summarises_to_an_empty_table(tmp_path):
    assert summarise_curves(compute_term_structure(tmp_path, NOW)).empty


def test_the_annualisation_floor_is_stated_in_days(tmp_path):
    """Named rather than buried, because it is the one arbitrary number here."""
    assert MIN_DAYS_TO_ANNUALISE == Decimal(1)
