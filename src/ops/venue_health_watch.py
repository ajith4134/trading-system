"""Feed every venue's health report into the halt registry.

`VenueHaltRegistry` has been complete since it was written and has never been
called. `observe()` and `assess_venue()` had zero callers anywhere in `src/`, so
no degradation could halt anything - and the status wall said "auto-halt armed"
on top of that, for a mechanism with no driver. This is the driver.

## Why this is not the wall

`statuswall.evidence` imports the registry and reads it. It must never write to
it: a display that decides whether a venue may be traded is a display whose
rendering cadence, error handling and refresh bugs become trading decisions. The
board was already the only importer, which is exactly how a read-only consumer
comes to look like the owner of a mechanism.

Separating them also separates the failure. A probe raising takes down a tile;
this loop dying takes down halting, and the wall says so, because the registry's
`last_observed_ns` goes stale and the venue-health tile grades on it. Neither
failure hides the other.

## What it will and will not halt

`assess_venue` halts on archive corruption alone, and deliberately not on silence
- see the reasoning at `_SILENCE_LIMIT`, where the count is inflated by thin tail
symbols and by a feed the venue withholds, so it cannot tell a sick venue from a
healthy one with a wide universe. Measured on this host 2026-08-09 before this
loop was first run: 0 corrupting events on all three venues against 1,570 and
6,987 silence events, so feeding the registry halts nothing and the observation
stamps become the tile's evidence instead of its absence.

That narrowness is the honest state of the check and is left alone here. Widening
it belongs with the per-stream staleness view Layer 3 actually specifies, not
with the loop that calls it.

## One pass per invocation

The loop is the shell script's, matching every other supervisor here. A crash
then costs one interval rather than the whole watch, and the restart is recorded
where an operator already looks.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from capture.capture_health import build_report, measure_daily_bytes
from ops.venue_halt import VenueHaltRegistry


def capture_dates_by_venue(capture_root: Path) -> dict[str, list[str]]:
    """UTC dates present per venue under `raw/`, newest last."""
    raw = Path(capture_root) / "raw"
    if not raw.is_dir():
        return {}
    found: dict[str, list[str]] = {}
    for venue_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        dates = sorted(p.name for p in venue_dir.iterdir() if p.is_dir())
        if dates:
            found[venue_dir.name] = dates
    return found


def observe_all_venues(capture_root: Path,
                       registry: VenueHaltRegistry | None = None) -> dict:
    """Build each venue's newest health report and feed it to the registry.

    A venue whose report cannot be built is skipped and named, not silently
    dropped: it is indistinguishable from a healthy one otherwise, and the
    registry would keep serving whatever verdict it last recorded while nothing
    said the input had stopped arriving.
    """
    capture_root = Path(capture_root)
    registry = registry or VenueHaltRegistry(capture_root / "ops")
    dates_by_venue = capture_dates_by_venue(capture_root)
    if not dates_by_venue:
        return {"observed": {}, "skipped": {}, "halted": []}

    free_bytes = shutil.disk_usage(capture_root).free
    daily_bytes = measure_daily_bytes(capture_root)

    observed: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for venue, dates in dates_by_venue.items():
        try:
            report = build_report(capture_root, venue, dates[-1], free_bytes, daily_bytes)
        except Exception as exc:
            skipped[venue] = f"{type(exc).__name__}: {exc}"
            continue
        registry.observe(venue, report)
        observed[venue] = {
            "date": dates[-1],
            "corrupting_non_gap": report.get("corrupting_non_gap", 0),
            "silent_streams": report.get("silent_streams", 0),
            "tradeable": registry.is_tradeable(venue),
            "halt_reason": registry.halt_reason(venue),
        }

    return {"observed": observed, "skipped": skipped,
            "halted": sorted(v for v, o in observed.items() if not o["tradeable"])}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Feed venue health reports into the halt registry - one pass.")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    args = parser.parse_args(argv)

    result = observe_all_venues(Path(args.capture_root))
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    # A halted venue is the mechanism working, not this process failing, so it is
    # not an error exit. A venue whose report could not be built IS one: the
    # registry went another interval with no input and nothing else would say so.
    return 1 if result["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
