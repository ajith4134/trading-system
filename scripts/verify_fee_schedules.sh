#!/usr/bin/env bash
# Fetch each venue's fee schedule and report what could actually be verified.
#
# This exists as a named script rather than an inline command so that the
# permission to run it is narrow and reviewable. It reads credentials from the
# sops+age store through `cost.secret_store`, which pipes them into memory and
# never writes plaintext to disk, and it prints **rates only** - never a key,
# never a signature, never a request URL.
#
# Hyperliquid publishes its schedule unauthenticated. Binance does not: both
# `/fapi/v1/commissionRate` and `/sapi/v1/asset/tradeFee` answer -2014 without a
# signed request, and no public endpoint carries a maker/taker rate at all -
# re-measured 2026-08-08. So Binance is the only venue here that needs the key.
#
# Usage: scripts/verify_fee_schedules.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"

PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" - <<'PY'
import asyncio

from cost.fee_fetcher import (
    FeeScheduleUnavailable,
    fetch_binance_schedule,
    fetch_hyperliquid_schedule,
)


def report(label, schedule):
    verdict = "VERIFIED" if schedule.is_verified else "UNVERIFIED"
    print(f"  {verdict:11s} {label}")
    print(f"                maker {schedule.rate.maker_bps} bps | "
          f"taker {schedule.rate.taker_bps} bps | tier {schedule.tier}")
    print(f"                taker round trip "
          f"{schedule.round_trip_bps(maker_in=False, maker_out=False)} bps")
    print(f"                source {schedule.source.value} "
          f"({schedule.source_detail[:60]})")


async def main():
    for label, coro in (
            ("hyperliquid:perp", fetch_hyperliquid_schedule()),
            ("binance:perp", fetch_binance_schedule("BTCUSDT")),
    ):
        try:
            report(label, await coro)
        except FeeScheduleUnavailable as exc:
            # The whole point of the declared-table design: a venue that will
            # not answer leaves the quote unverified rather than inventing one.
            print(f"  REFUSED     {label}: {str(exc)[:160]}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  ERROR       {label}: {type(exc).__name__}: {str(exc)[:160]}")


asyncio.run(main())
PY
