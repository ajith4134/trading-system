"""Funding, which is the carry the prime directive rests on.

The venue asymmetry is the whole point of this module. `FEATURES.md` §1 marks it
`[MISSED]`: Binance USDⓈ-M settles 8-hourly on the mark price, Hyperliquid
hourly on the *oracle* price and capped at 4%/hour. One shared assumption is
wrong at both ends, and it is wrong in the direction of overstating the edge.
"""
import datetime as dt
from decimal import Decimal

import pytest

from cost.funding_carry import (
    NoFundingAvailable,
    funding_cost_bps,
    load_funding_rates_as_of,
    settlements_between,
)


def ns(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso).replace(
        tzinfo=dt.timezone.utc).timestamp() * 1e9)


def test_binance_settles_three_times_a_day():
    """00:00, 08:00 and 16:00 UTC. A position held a full day crosses three."""
    crossed = settlements_between("binance", ns("2026-08-08T00:00:01"),
                                  ns("2026-08-09T00:00:00"))
    assert len(crossed) == 3


def test_hyperliquid_settles_every_hour():
    """Hourly, not 8-hourly. Pricing a Hyperliquid carry on Binance's schedule
    undercounts settlements eightfold and turns a losing carry into a winner."""
    crossed = settlements_between("hyperliquid", ns("2026-08-08T00:00:01"),
                                  ns("2026-08-08T12:00:00"))
    assert len(crossed) == 12


def test_a_position_closed_before_any_settlement_pays_no_funding():
    """The common case for anything intraday, and getting it wrong charges
    every scalp a funding leg it never paid."""
    assert settlements_between("binance", ns("2026-08-08T00:00:01"),
                               ns("2026-08-08T07:59:59")) == []


def test_the_settlement_boundary_is_inclusive_of_the_instant_it_lands():
    """Held exactly to 08:00:00 means the settlement happened."""
    crossed = settlements_between("binance", ns("2026-08-08T00:00:01"),
                                  ns("2026-08-08T08:00:00"))
    assert len(crossed) == 1


def test_an_unknown_venue_refuses_rather_than_assuming_a_schedule():
    """Defaulting to 8-hourly would silently misprice every venue that is not
    Binance - which is the mistake this module was written to prevent."""
    with pytest.raises(NoFundingAvailable):
        settlements_between("kraken", ns("2026-08-08T00:00:01"),
                            ns("2026-08-09T00:00:00"))


def test_a_long_pays_funding_when_the_rate_is_positive():
    """Positive rate means longs pay shorts. A cost is positive bps."""
    rates = [Decimal("0.0001"), Decimal("0.0001")]      # 1 bp each
    assert funding_cost_bps(rates, side="long") == Decimal(2)


def test_a_short_earns_what_the_long_pays():
    rates = [Decimal("0.0001"), Decimal("0.0001")]
    assert funding_cost_bps(rates, side="short") == Decimal(-2)


def test_each_settlement_uses_its_own_archived_rate():
    """Not three copies of the latest one. Funding moves, and a carry priced on
    the current rate held constant is a forecast wearing a measurement's
    clothes."""
    rates = [Decimal("0.0001"), Decimal("-0.0002"), Decimal("0.0003")]
    assert funding_cost_bps(rates, side="long") == Decimal(2)


def test_no_settlements_costs_nothing():
    assert funding_cost_bps([], side="long") == Decimal(0)


def test_an_unknown_side_refuses():
    with pytest.raises(ValueError):
        funding_cost_bps([Decimal("0.0001")], side="flat")


def test_loading_rates_refuses_when_no_funding_dataset_exists(tmp_path):
    """premiumIndex is captured into the raw archive but has never been built
    into a clock-gated dataset. Until it is, the honest answer is a refusal."""
    with pytest.raises(NoFundingAvailable) as excinfo:
        load_funding_rates_as_of(tmp_path, "binance", "BTCUSDT",
                                 ns("2026-08-08T00:00:00"), ns("2026-08-09T00:00:00"))
    assert "funding" in str(excinfo.value).lower()


# --- reading the dataset once it exists --------------------------------------

def _write_funding_dataset(store_root, rows):
    """A real dataset written through the real partition writer."""
    from decimal import Decimal as D
    from store.funding_rates import FundingObservation, build_funding_frame
    from store.parquet_partition import append_partition
    frame = build_funding_frame([
        FundingObservation(symbol="BTCUSDT", venue="binance",
                           funding_rate=D(rate), mark_price=D("1"),
                           index_price=D("1"), next_funding_time_ns=0,
                           event_time_ns=at, ingestion_time_ns=at)
        for rate, at in rows])
    append_partition(store_root, "funding", frame, snapshot_id="test")


def test_rates_are_served_once_the_dataset_exists(tmp_path):
    """The refusal is about a missing dataset, not a permanent state. Once
    Layer 1 holds funding, the same call must answer."""
    settlement = ns("2026-08-08T08:00:00")
    _write_funding_dataset(tmp_path, [("0.0001", settlement - 10**9)])

    got = load_funding_rates_as_of(tmp_path, "binance", "BTCUSDT",
                                   ns("2026-08-08T00:00:01"),
                                   ns("2026-08-08T08:00:00"))
    assert got == [Decimal("0.0001")]


def test_a_rate_published_after_the_settlement_is_not_used(tmp_path):
    """The leakage case, end to end through the clock gate. A rate that landed
    after the settlement it would be charged against must not be visible."""
    settlement = ns("2026-08-08T08:00:00")
    _write_funding_dataset(tmp_path, [("0.0001", settlement - 10**9),
                                      ("0.0009", settlement + 60 * 10**9)])

    got = load_funding_rates_as_of(tmp_path, "binance", "BTCUSDT",
                                   ns("2026-08-08T00:00:01"),
                                   ns("2026-08-08T08:00:00"))
    assert got == [Decimal("0.0001")], "used a rate published after the settlement"
