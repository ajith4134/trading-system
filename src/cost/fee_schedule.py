"""What a venue charges, and how much that claim can be trusted.

The distinction this module exists to keep is between a fee that was *fetched
from the venue* and a fee that was *typed in from a documentation page*. Both are
usable; only one is evidence. A cost engine that cannot tell them apart will
eventually gate a live strategy on a number nobody measured, which is the
failure the whole reality-filter layer exists to prevent.

Measured 2026-08-03: Hyperliquid publishes its full schedule unauthenticated,
Binance does not (`/fapi/v1/commissionRate` -> 401 without a key). So the
unverified case is not hypothetical - it is the state Binance is in until a
read-only key exists, and it has to be representable rather than papered over.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

# Basis points per unit rate: a venue quoting 0.00045 charges 4.5 bps.
_BPS_PER_UNIT = Decimal(10_000)


class FeeSource(str, Enum):
    """Where a rate came from. Not cosmetic - it decides whether a quote built
    on this rate may ever be presented as verified."""

    VENUE_API = "venue_api"      # fetched from the venue, with a timestamp
    DECLARED = "declared"        # written down by a human from a fee page


@dataclass(frozen=True)
class FeeRate:
    maker_bps: Decimal
    taker_bps: Decimal

    @classmethod
    def from_unit_rates(cls, maker: str, taker: str) -> FeeRate:
        """Venues quote unit rates as decimal strings ("0.00045"). Parsed as
        `Decimal` straight from the string - going via float would introduce a
        representation error into the number that gates every strategy."""
        return cls(Decimal(maker) * _BPS_PER_UNIT, Decimal(taker) * _BPS_PER_UNIT)


@dataclass(frozen=True)
class FeeSchedule:
    venue: str
    instrument_kind: str            # "perp" or "spot"
    rate: FeeRate
    tier: str
    source: FeeSource
    source_detail: str              # the endpoint or document it came from
    fetched_at_ns: int | None       # None for a declared rate: nothing was fetched

    @property
    def is_verified(self) -> bool:
        return self.source is FeeSource.VENUE_API and self.fetched_at_ns is not None

    def is_stale(self, now_ns: int, max_age_ns: int) -> bool:
        """A declared rate is stale from birth.

        Not a technicality. Staleness asks "could this have changed without us
        noticing?", and for a number nobody is fetching the answer is always yes,
        no matter how recently someone typed it.
        """
        if not self.is_verified:
            return True
        return (now_ns - self.fetched_at_ns) > max_age_ns

    def round_trip_bps(self, *, maker_in: bool, maker_out: bool) -> Decimal:
        """Both legs, which is the only number that decides whether an edge is
        real - a strategy pays to get in and pays again to get out."""
        leg_in = self.rate.maker_bps if maker_in else self.rate.taker_bps
        leg_out = self.rate.maker_bps if maker_out else self.rate.taker_bps
        return leg_in + leg_out


def _ns(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso).timestamp() * 1e9)


# Rates nobody fetched. Every entry names its source and is permanently
# unverified: Binance will not serve its schedule without an API key, so until
# one exists these are the best available claim and must read as exactly that.
DECLARED_SCHEDULES: dict[tuple[str, str], FeeSchedule] = {
    ("binance", "perp"): FeeSchedule(
        venue="binance", instrument_kind="perp",
        rate=FeeRate(Decimal("2.0"), Decimal("5.0")),
        tier="VIP 0",
        source=FeeSource.DECLARED,
        source_detail=("binance.com/en/fee/futureFee (USD-M VIP 0, 0.0200%/0.0500%); "
                       "/fapi/v1/commissionRate returns 401 without an API key, "
                       "measured 2026-08-03"),
        fetched_at_ns=None),
    ("binance", "spot"): FeeSchedule(
        venue="binance", instrument_kind="spot",
        rate=FeeRate(Decimal("10.0"), Decimal("10.0")),
        tier="VIP 0",
        source=FeeSource.DECLARED,
        source_detail=("binance.com/en/fee/schedule (spot VIP 0, 0.1000%/0.1000%); "
                       "matches the ~20bps round trip in ARCHITECTURE.md Layer 1"),
        fetched_at_ns=None),
}
