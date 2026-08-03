"""A fee is a claim. This is where it is forced to say who made it."""
from decimal import Decimal

from cost.fee_schedule import DECLARED_SCHEDULES, FeeRate, FeeSchedule, FeeSource

HOUR_NS = 3_600 * 10**9


def _fetched(now_ns: int = 1_000_000_000_000) -> FeeSchedule:
    return FeeSchedule(
        venue="hyperliquid", instrument_kind="perp",
        rate=FeeRate.from_unit_rates("0.00015", "0.00045"),
        tier="base", source=FeeSource.VENUE_API,
        source_detail="POST /info", fetched_at_ns=now_ns)


def test_unit_rates_convert_to_basis_points_exactly():
    """Hyperliquid's live base rates, 2026-08-03: 0.00015 add / 0.00045 cross."""
    rate = FeeRate.from_unit_rates("0.00015", "0.00045")
    assert rate.maker_bps == Decimal("1.5")
    assert rate.taker_bps == Decimal("4.5")


def test_rates_are_decimal_not_float():
    """A float breakeven gates every strategy on a representation error."""
    rate = FeeRate.from_unit_rates("0.00015", "0.00045")
    assert isinstance(rate.maker_bps, Decimal)
    assert isinstance(rate.taker_bps, Decimal)


def test_a_declared_schedule_never_reports_itself_verified():
    for schedule in DECLARED_SCHEDULES.values():
        assert schedule.source is FeeSource.DECLARED
        assert not schedule.is_verified
        assert schedule.fetched_at_ns is None


def test_a_declared_schedule_is_stale_from_birth():
    """Staleness asks 'could this have changed without us noticing?'. For a
    number nobody fetches the answer is always yes, however recently it was
    typed - so no max_age can make it fresh."""
    declared = DECLARED_SCHEDULES[("binance", "perp")]
    assert declared.is_stale(now_ns=0, max_age_ns=10**18)


def test_a_fetched_schedule_is_fresh_until_its_age_passes():
    schedule = _fetched(now_ns=1_000 * HOUR_NS)
    assert not schedule.is_stale(now_ns=1_001 * HOUR_NS, max_age_ns=2 * HOUR_NS)
    assert schedule.is_stale(now_ns=1_005 * HOUR_NS, max_age_ns=2 * HOUR_NS)


def test_staleness_is_measured_from_the_fetch_not_from_now():
    """A schedule fetched days ago must not read fresh merely because the
    process restarted."""
    schedule = _fetched(now_ns=0)
    assert schedule.is_stale(now_ns=100 * HOUR_NS, max_age_ns=HOUR_NS)


def test_round_trip_charges_both_legs():
    schedule = _fetched()
    assert schedule.round_trip_bps(maker_in=False, maker_out=False) == Decimal("9.0")
    assert schedule.round_trip_bps(maker_in=True, maker_out=True) == Decimal("3.0")
    assert schedule.round_trip_bps(maker_in=True, maker_out=False) == Decimal("6.0")


def test_the_declared_binance_spot_rate_matches_the_architecture_document():
    """ARCHITECTURE.md Layer 1 states a Binance spot taker round trip of ~20bps
    and sizes the whole cost argument on it. If the table and the design
    document disagree, one of them is wrong and nobody finds out."""
    spot = DECLARED_SCHEDULES[("binance", "spot")]
    assert spot.round_trip_bps(maker_in=False, maker_out=False) == Decimal("20.0")
