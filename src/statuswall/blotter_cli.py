"""Regenerate the blotter on its own clock, not the status wall's.

Split out 2026-08-16 after the user opened `/blotter.html` and was served a page
**3.5 hours old** showing zero closed trades when four had closed.

## Why it was stale, and why a shorter interval would not have fixed it

The blotter rode `statuswall.cli`'s measurement pass, which is what the wall and
the build-progress board are built from. That pass takes **ten to nineteen
minutes**: every phase-B probe reads the whole bar dataset, several read the
funding dataset, and the universe watch list reads three. So the blotter could
never be fresher than the slowest probe on the board, and when a pass was started
by a process holding older code it was not written at all.

The blotter needs none of that. `paper.blotter.read_blotter` reads one directory
of NDJSON and returns in about **0.2 seconds**. Coupling a 0.2-second view to a
ten-minute measurement pass is what made it stale, and running the ten-minute
pass more often would only have burned CPU to fix a problem that was never about
the interval.

So this is a separate entry point with its own loop, and the two boards now age
independently — which is also more honest: the wall's age is the age of a
measurement sweep, and the blotter's is the age of a journal read. A single
timestamp covering both would be wrong about one of them.

## It writes the same page the wall pass does

`statuswall.cli` still writes the blotter on its own pass. Two writers of one
file is normally a smell, and here it is deliberate: whichever ran last wins, the
file is written whole rather than appended, and a reader gets one version or the
other. The alternative — removing it from the wall pass — would mean a board with
no blotter at all if this loop is not running, and the failure would be a missing
page rather than an old one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from paper.blotter import read_blotter
from statuswall.blotter_page import render_blotter_page


def render_once(capture_root: Path, out: Path) -> str:
    """Write the blotter and return the one-line summary it reports."""
    now = dt.datetime.now(dt.timezone.utc)
    view = read_blotter(capture_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_blotter_page(
        view=view,
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
        generated_at_epoch_s=int(now.timestamp()),
    ), encoding="utf-8")
    return view.describe()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="statuswall.blotter_cli",
        description="Regenerate the paper blotter from the engine's fill "
                    "journal. Fast, because it reads one directory of NDJSON "
                    "and nothing else.")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--out", default=str(
        Path.home() / "research" / "dashboard" / "blotter.html"))
    parser.add_argument("--interval-seconds", type=int, default=0,
                        help="0 renders once and exits; anything else loops")
    args = parser.parse_args(argv)

    capture_root, out = Path(args.capture_root), Path(args.out)
    while True:
        try:
            summary = render_once(capture_root, out)
        except Exception as error:                 # noqa: BLE001 - reported, not fatal
            # Reported and retried rather than fatal. A journal being appended
            # to while it is read is the ordinary case, and a loop that dies on
            # one bad read leaves the board frozen with nothing saying so - the
            # failure `boards_supervisor.sh` was written after.
            print(f"blotter render failed: {error}", file=sys.stderr, flush=True)
        else:
            print(f"wrote {out}: {summary}", flush=True)
        if args.interval_seconds <= 0:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
