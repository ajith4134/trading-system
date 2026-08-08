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
# It also leaves a receipt at $CAPTURE_ROOT/fee-verification/latest.json, which
# is what the status wall reads. The wall must not make signed API calls itself:
# a display that authenticates on every render puts credentials in the render
# path and spends rate limit to draw a tile. Verification is a separate act that
# records what it measured, and the tile reports that record plus its age - so a
# schedule verified once and never again reads as stale rather than as verified.
#
# The receipt holds rates and timestamps only. No key, no signature.
#
# Usage: scripts/verify_fee_schedules.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"
export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"

CAPTURE_ROOT="${CAPTURE_ROOT:-$HOME/capture}"
export RECEIPT_DIR="$CAPTURE_ROOT/fee-verification"
mkdir -p "$RECEIPT_DIR"

PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" - <<'PY'
import asyncio
import json
import os
import time
from pathlib import Path

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
    receipt = {"measured_at_ns": time.time_ns(), "venues": {}}
    for label, coro in (
            ("hyperliquid:perp", fetch_hyperliquid_schedule()),
            ("binance:perp", fetch_binance_schedule("BTCUSDT")),
    ):
        try:
            schedule = await coro
            report(label, schedule)
            receipt["venues"][label] = {
                "verified": schedule.is_verified,
                "maker_bps": str(schedule.rate.maker_bps),
                "taker_bps": str(schedule.rate.taker_bps),
                "tier": schedule.tier,
                "source": schedule.source.value,
                "fetched_at_ns": schedule.fetched_at_ns,
            }
        except FeeScheduleUnavailable as exc:
            # The whole point of the declared-table design: a venue that will
            # not answer leaves the quote unverified rather than inventing one.
            print(f"  REFUSED     {label}: {str(exc)[:160]}")
            receipt["venues"][label] = {"verified": False,
                                        "refused": str(exc)[:200]}
        except Exception as exc:                        # noqa: BLE001
            print(f"  ERROR       {label}: {type(exc).__name__}: {str(exc)[:160]}")
            receipt["venues"][label] = {"verified": False,
                                        "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    # Written last and atomically: a half-written receipt would make the wall
    # report a verification that did not finish.
    out = Path(os.environ["RECEIPT_DIR"]) / "latest.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"  receipt -> {out}")


asyncio.run(main())
PY
