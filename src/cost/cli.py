"""Ask what a round trip costs, and see where every number came from.

Prints the itemised quote with its provenance rather than a single figure,
because a cost a strategy cannot audit is a cost it should not trust. A venue
whose schedule was never fetched prints `UNVERIFIED` next to the line that was
transcribed by a human, and a quote that cannot be built prints the refusal and
what was missing.

    python -m cost.cli --venue hyperliquid --symbol BTC --notional 10000
    python -m cost.cli --venue binance --symbol BTCUSDT --notional 10000 --kind spot
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from decimal import Decimal

from cost.fee_fetcher import FeeScheduleUnavailable, fetch_hyperliquid_schedule
from cost.round_trip_cost import CostRefused, quote_round_trip_cost

# Only Hyperliquid publishes a schedule without credentials - measured
# 2026-08-03, and the reason the cost engine is a declared table plus a
# verifier rather than a pure fetcher. Binance answers /fapi/v1/commissionRate
# with 401 and has no anonymous equivalent.
_FETCHERS = {"hyperliquid": fetch_hyperliquid_schedule}


def _live_schedule(venue: str, now_ns: int):
    """The venue's own schedule if it will serve one, otherwise None.

    A failed fetch returns None rather than raising: falling back to the
    declared table is correct, and the quote will mark itself unverified, which
    is the honest outcome. Silence would not be.
    """
    fetcher = _FETCHERS.get(venue)
    if fetcher is None:
        return None
    try:
        return asyncio.run(fetcher(now_ns=now_ns))
    except (FeeScheduleUnavailable, OSError) as exc:
        print(f"note: live fetch for {venue} failed ({exc}); "
              f"falling back to the declared table", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cost", description="Round-trip cost, itemised, with provenance.")
    parser.add_argument("--venue", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--notional", required=True, type=Decimal)
    parser.add_argument("--kind", default="perp", choices=("perp", "spot"))
    parser.add_argument("--order-type", default="taker", choices=("maker", "taker"))
    parser.add_argument("--holding-hours", type=float, default=0.0,
                        help="how long the position is held; anything crossing a "
                             "funding settlement needs the funding dataset")
    parser.add_argument("--no-fetch", action="store_true",
                        help="use the declared table only, never the venue")
    args = parser.parse_args(argv)

    now_ns = time.time_ns()
    schedule = None if args.no_fetch else _live_schedule(args.venue, now_ns)

    result = quote_round_trip_cost(
        args.venue, args.symbol, args.notional,
        order_type=args.order_type, at_ns=now_ns,
        instrument_kind=args.kind, schedule=schedule,
        holding_ns=int(args.holding_hours * 3_600_000_000_000))

    if isinstance(result, CostRefused):
        print(f"REFUSED  {result.venue}:{result.symbol}")
        print(f"  missing: {result.missing}")
        print(f"  {result.reason}")
        # Non-zero so a script cannot treat a refusal as a priced trade by
        # ignoring the exit code.
        return 2

    verdict = "VERIFIED" if result.is_verified else "UNVERIFIED"
    print(f"{verdict}  {result.venue}:{result.symbol}  "
          f"{args.order_type} round trip on {result.notional}")
    print(f"  breakeven      {result.breakeven_bps:>8} bps")
    print(f"    fee          {result.fee_bps:>8} bps")
    print(f"    spread       {result.spread_bps:>8} bps")
    print(f"    impact       {result.impact_bps:>8} bps")
    print(f"    funding      {result.funding_bps:>8} bps")
    print("  inputs:")
    for source in result.inputs:
        mark = "verified  " if source.verified else "UNVERIFIED"
        age = "" if source.age_ns is None else f"  age {source.age_ns / 1e9:.1f}s"
        print(f"    {mark} {source.name:<8}{age}  {source.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
