"""Funding and basis become features only once they are ranked and netted.

The failures defended here are the ones that leave a plausible number behind:
a funding rate annualised on the wrong venue's schedule (wrong by 8x, in the
flattering direction), a percentile taken against a history the clock had not
served, a rank that reports 0.0 for an instrument whose rate has been flat, and
a key half-served with a level but no rank - which is the raw per-symbol number
FE-012 exists to forbid.
"""
from decimal import Decimal

import pandas as pd
import pytest

from cost.funding_carry import SETTLEMENT_HOURS
from features.funding_basis import (
    MIN_HISTORY_OBSERVATIONS,
    compute_funding_basis,
    percentile_rank,
    settlements_per_year,
)
from store.parquet_partition import append_partition


def _funding_row(venue, symbol, funds_on, mark, index, oracle, event_ns,
                 rate=Decimal("0.0001"), avail_ns=None):
    return {
        "venue": venue, "symbol": symbol, "funds_on": funds_on,
        "funding_rate": rate,
        "mark_price": mark, "index_price": index, "oracle_price": oracle,
        "event_time_is_receipt": False, "next_funding_time_unknown": False,
        "next_funding_time_ns": event_ns + 1, "funding_interval_hours": 8,
        "event_time_ns": event_ns,
        "ingestion_time_ns": event_ns,
        "availability_time_ns": avail_ns if avail_ns is not None else event_ns,
    }


def _history(venue, symbol, funds_on, marks, *, reference=Decimal("100.000"),
             rates=None, start_ns=1_000, step_ns=1_000, avail_ns=None):
    """One row per mark, all against a fixed reference, evenly spaced."""
    rows = []
    for i, mark in enumerate(marks):
        rate = Decimal("0.0001") if rates is None else rates[i]
        index = reference if funds_on == "mark" else None
        oracle = reference if funds_on == "oracle" else None
        event = start_ns + i * step_ns
        rows.append(_funding_row(venue, symbol, funds_on, mark, index, oracle,
                                 event, rate=rate,
                                 avail_ns=avail_ns if avail_ns is None
                                 else avail_ns))
    return rows


def _write(tmp_path, rows, snapshot="funding-basis-test"):
    append_partition(tmp_path, "funding", pd.DataFrame(rows),
                     snapshot_id=snapshot)
    return tmp_path


def _flat_marks(n, value="100.500"):
    return [Decimal(value)] * n


# --- the venue asymmetry, which is the whole [MISSED] on this feature -----

def test_annualisation_uses_each_venue_s_own_settlement_count():
    """Binance settles 3x a day, Hyperliquid 24x. One shared number is wrong by
    8x on whichever venue it was not written for, and wrong in the direction
    that turns a losing carry into a winner."""
    assert settlements_per_year("binance") == 3 * 365
    assert settlements_per_year("hyperliquid") == 24 * 365


def test_a_venue_with_no_schedule_annualises_to_nothing_rather_than_zero():
    """Spot has no funding at all, and an unknown venue has no schedule on
    record. Both must refuse - annualising by zero settlements would report
    every rate on that venue as free to hold."""
    assert settlements_per_year("binance-spot") is None
    assert settlements_per_year("kraken-perp-that-does-not-exist") is None


def test_the_schedule_comes_from_the_module_that_charges_against_it():
    """Two copies of the settlement schedule drift, and the drift is silent."""
    assert set(SETTLEMENT_HOURS) >= {"binance", "binance-spot", "hyperliquid"}
    assert settlements_per_year("binance") == len(SETTLEMENT_HOURS["binance"]) * 365


# --- the rank ------------------------------------------------------------

def test_a_flat_history_ranks_at_the_middle_not_at_zero():
    """A funding rate that has been exactly zero all week sits in the CENTRE of
    its own distribution, not at the bottom of it. A strictly-below definition
    reports 0.0 and a consumer reads that as 'unusually low'."""
    assert percentile_rank([0.0] * 50, 0.0) == pytest.approx(0.5)


def test_the_rank_is_a_midrank_over_a_plateau():
    """Half the sample below, a plateau at the value: the answer is the middle
    of the plateau, which is what a tie means."""
    history = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    # Two strictly below, five at or below -> (2 + 5) / 12.
    assert percentile_rank(history, 2.0) == pytest.approx(7 / 12)


def test_the_extremes_are_reachable():
    """A rank that can never reach its ends is a rank nobody can threshold."""
    history = [1.0, 2.0, 3.0, 4.0]
    assert percentile_rank(history, 0.0) == pytest.approx(0.0)
    assert percentile_rank(history, 9.0) == pytest.approx(1.0)


# --- end to end ----------------------------------------------------------

def test_the_carry_spread_is_the_basis_net_of_what_holding_it_costs(tmp_path):
    """mark 101 against index 100 is +100 bps; a 0.0001 rate is 1 bp per
    settlement; the tradable edge per interval is the difference, not either."""
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    rows[-1]["mark_price"] = Decimal("101.000")
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    assert len(table.rows) == 1, table.refused
    row = table.rows.iloc[0]
    assert row["basis_bps"] == pytest.approx(Decimal("100"))
    assert row["funding_rate_bps"] == pytest.approx(Decimal("1"))
    assert row["carry_spread_bps"] == pytest.approx(Decimal("99"))
    assert row["settlements_per_year"] == 3 * 365
    assert row["funding_annualised_bps"] == pytest.approx(Decimal(3 * 365))
    assert row["reference"] == "index_price"


def test_a_basis_at_the_top_of_its_own_history_ranks_near_one(tmp_path):
    """The rank is what makes the level usable. 101 after thirty prints at
    100.5 is high FOR THIS INSTRUMENT, which no absolute threshold could say."""
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    rows[-1]["mark_price"] = Decimal("101.000")
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    row = table.rows.iloc[0]
    assert row["basis_percentile"] > 0.95
    assert row["observations"] == MIN_HISTORY_OBSERVATIONS + 5


def test_the_rank_cannot_see_a_row_the_clock_gate_withheld(tmp_path):
    """The leak this module would otherwise have: a percentile computed against
    a history that includes rows not yet available reports where today's value
    sits in a distribution the model had not lived through.
    """
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    rows[-1]["mark_price"] = Decimal("101.000")
    # Three enormous prints that HAPPENED early and ARRIVED late - the
    # bitemporal shape a correction actually has. Their event times sit inside
    # the history, so the newest row is the same one at both clocks and only the
    # distribution behind it changes; a test where the newest row also moved
    # would pass without the gate doing anything.
    late = _history("binance", "BTCUSDT", "mark",
                    [Decimal("900.000")] * 3, start_ns=500, step_ns=1)
    for row in late:
        row["availability_time_ns"] = 10**12
    store = _write(tmp_path, rows + late)

    gated = compute_funding_basis(store, as_of_ns=10**9).rows.iloc[0]
    ungated = compute_funding_basis(store, as_of_ns=10**13).rows.iloc[0]

    assert gated["basis_percentile"] > 0.95, "top of the history it could see"
    assert ungated["basis_percentile"] < 0.95, (
        "the same value is no longer extreme once the future is visible - "
        "which is exactly why the gated answer must not equal this one")


def test_a_key_with_too_little_history_is_refused_not_half_served(tmp_path):
    """Reporting the level without the rank hands a consumer the raw per-symbol
    number FE-012 forbids, and hands it over looking complete."""
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS - 1))
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    assert table.rows.empty
    assert table.refused["too_few_observations"] == 1


def test_spot_is_refused_as_having_no_funding_rather_than_ranked(tmp_path):
    """`binance-spot` has an empty schedule on purpose: there is no funding to
    annualise, so there is no feature here - and that is not an error."""
    rows = _history("binance-spot", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    assert table.rows.empty
    assert table.refused["no_funding_settlements"] == 1


def test_a_venue_with_no_schedule_on_record_is_refused(tmp_path):
    """Guessing 8-hourly for an unrecognised venue is the exact mistake
    `cost.funding_carry` was written to prevent, and it would be silent."""
    rows = _history("okx", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    assert table.rows.empty
    assert table.refused["unknown_settlement_schedule"] == 1


def test_an_oracle_venue_is_ranked_against_its_oracle(tmp_path):
    """Hyperliquid funds on oracle. Pricing its basis off an index it does not
    fund on is wrong at one end, and the sign of the error follows whichever
    reference happens to sit nearer the mark."""
    rows = _history("hyperliquid", "BTC", "oracle",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5),
                    reference=Decimal("200.000"))
    rows[-1]["mark_price"] = Decimal("198.000")
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    row = table.rows.iloc[0]
    assert row["reference"] == "oracle_price"
    assert row["basis_bps"] == pytest.approx(Decimal("-100"))
    assert row["settlements_per_year"] == 24 * 365


def test_a_zero_reference_refuses_rather_than_dividing(tmp_path):
    """The placeholder-price defect wearing a reference's coat."""
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    rows[-1]["index_price"] = Decimal("0.000")
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    assert table.rows.empty
    assert table.refused["unparseable_price"] == 1


def test_every_row_carries_its_staleness(tmp_path):
    """FE-001 is the Phase B contract every later feature inherits."""
    rows = _history("binance", "BTCUSDT", "mark",
                    _flat_marks(MIN_HISTORY_OBSERVATIONS + 5))
    table = compute_funding_basis(_write(tmp_path, rows), as_of_ns=10**9)

    row = table.rows.iloc[0]
    assert row["age_ns"] is not None
    assert row["freshness"] is not None


def test_an_empty_store_measures_nothing_and_refuses_nothing(tmp_path):
    """No input is not a refusal - there was nothing to refuse."""
    (tmp_path / "funding").mkdir(parents=True)
    table = compute_funding_basis(tmp_path, 10**18)

    assert table.rows.empty
    assert set(table.refused.values()) == {0}
