"""Record one point-in-time universe snapshot, with the quote currencies.

Exists because the capture loop records a snapshot only at startup. Quote assets
began being captured on 2026-08-08, after three capture processes had already been
running for hours, and the alternative to this entry point was killing live
sockets to make a listing readable - Layer 0 cannot backfill a frame it was not
connected for, so a restart to fix a metadata gap costs real tape.

It is also the operational answer to `QuoteAssetsNotRecorded`: the refusal names
this command, and running it is what makes the day filterable.

    python -m capture.record_universe_snapshot --venue binance-spot

Deliberately not on a timer. A snapshot is a permanent append, the events it emits
are the record of what listed and delisted, and an hourly one would bury a real
listing among 24 identical lines a day. The capture loop's startup snapshot plus
this by hand is the whole intended cadence.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from capture.cli import _VENUES, fetch_instruments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="record_universe_snapshot",
        description="Fetch a venue's listing and record it as a point-in-time snapshot.")
    parser.add_argument("--venue", required=True, choices=sorted(_VENUES))
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    args = parser.parse_args(argv)

    # Imported here rather than at module scope so `--help` works without the
    # capture package's heavier dependencies being importable.
    from capture.universe_tracker import UniverseTracker

    venue = _VENUES[args.venue]()
    # One fetch answers both questions. Two would risk a listing landing between
    # them, leaving a quote map that covers symbols the recorded universe does not.
    payload = fetch_instruments(venue)
    symbols = venue.parse_instruments(payload)
    quote_assets = venue.parse_quote_assets(payload)

    # The timestamp is when the listing was OBSERVED, taken after the fetch
    # returned. Stamping it before would date the snapshot to a moment whose
    # answer had not arrived yet, and every point-in-time read of it would be
    # reading the future by however long the request took.
    ts_ns = time.time_ns()
    events = UniverseTracker(Path(args.root), venue.name).record_snapshot(
        symbols, ts_ns, quote_assets=quote_assets)

    print(f"{venue.name}: recorded {len(symbols)} symbol(s), "
          f"{len(quote_assets)} with a quote currency, at {ts_ns}")
    # Printed by kind and count rather than in full: a routine snapshot emits
    # none, and a listing wave emitting 40 would scroll the count off the screen.
    listed = [e.symbol for e in events if e.kind == "listed"]
    delisted = [e.symbol for e in events if e.kind == "delisted"]
    print(f"  {len(listed)} listed, {len(delisted)} delisted since the last snapshot")
    for kind, changed in (("listed", listed), ("delisted", delisted)):
        if changed:
            print(f"  {kind}: {', '.join(sorted(changed))}")
    return 0


if __name__ == "__main__":       # pragma: no cover - entry point
    raise SystemExit(main())
