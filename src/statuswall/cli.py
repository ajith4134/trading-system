"""Generates the status wall from a live measurement pass.

Run it to refresh the board:

    python -m statuswall.cli --out ~/research/dashboard/status-wall.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from statuswall.build_progress import (
    count_unresolved_ledger_rows, probe_forward_paper, render_build_progress_page,
    summarise_build_progress,
)
from statuswall.catalogue import read_catalogue
from statuswall.evidence import (
    NOT_BUILT, SEVERITY_ORDER, STATE_LABEL, assess, measure_system, verify_probe_coverage,
)
from statuswall.wall_page import WallInput, render_wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="statuswall",
        description="Render a status wall from measured state. Nothing here is hand-written.")
    parser.add_argument("--features", default=str(Path.home() / "research" / "FEATURES.md"),
                        help="the feature catalogue this wall must not disagree with")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--ledger", default=None,
                        help="the requirements ledger to check BUILT claims against; "
                             "defaults to ledger/merged beside the feature catalogue")
    parser.add_argument("--out", required=True, help="where to write the HTML")
    args = parser.parse_args(argv)

    features_path = Path(args.features)
    if not features_path.is_file():
        parser.error(f"feature catalogue not found: {features_path}")

    features = read_catalogue(features_path)
    if not features:
        parser.error(f"no features parsed from {features_path} - the wall would be empty")
    # Checked after the empty case so a parse failure reports itself as a parse
    # failure rather than as 11 simultaneously orphaned probes.
    verify_probe_coverage(features)

    # Derived from the catalogue rather than from $HOME: the two live together,
    # and a probe guessing an absolute path would report a clean bill of health
    # on any machine where the ledger sits somewhere else.
    ledger_root = (Path(args.ledger) if args.ledger
                   else features_path.parent / "ledger" / "merged")

    facts = measure_system(
        capture_root=Path(args.capture_root),
        repo_root=Path(args.repo_root),
        now=dt.datetime.now(dt.timezone.utc),
        ledger_root=ledger_root if ledger_root.is_dir() else None,
    )
    results = assess(features, facts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_wall(WallInput(features, results, facts)), encoding="utf-8")

    # The build-progress board rides the same measurement pass and lands beside
    # the wall, so the boards supervisor serves it without knowing it exists.
    paper_running, paper_evidence = probe_forward_paper(Path(args.capture_root))
    progress_out = out.parent / "build-progress.html"
    progress_out.write_text(render_build_progress_page(
        phases=summarise_build_progress(features, results),
        unresolved_rows=count_unresolved_ledger_rows(ledger_root),
        paper_running=paper_running,
        paper_evidence=paper_evidence,
        generated_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        # The epoch the page ages itself against in the reader's browser. A
        # server-rendered age cannot cover the case that matters: the server that
        # would render it is the one that stopped writing.
        generated_at_epoch_s=int(dt.datetime.now(dt.timezone.utc).timestamp()),
    ), encoding="utf-8")
    print(f"wrote {progress_out}")

    counts = {state: 0 for state in SEVERITY_ORDER}
    for result in results.values():
        counts[result.state] += 1
    measured = len(features) - counts[NOT_BUILT]
    summary = " · ".join(f"{STATE_LABEL[s]} {counts[s]}" for s in SEVERITY_ORDER if counts[s])
    print(f"wrote {out}")
    print(f"{measured}/{len(features)} features measured — {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
