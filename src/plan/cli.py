"""Render the plan board and the ruling-conformance board.

Its own process, deliberately, and not folded into `statuswall.cli`. Measured
2026-08-17: `statuswall.cli` holds 4.8 GB while capture holds ~9 GB on a 30 GB
box with no swap, and at 09:49:54Z the kernel OOM-killed the forward paper
engine (exit 137 after 7,183s), which then spent minutes re-feeding 2.3M events
instead of trading. Adding a 2,738-member corpus sweep to that same process
would buy a marginally tidier boards pass at the cost of the thing the whole
system exists to run.

Short-lived on purpose: it reads, renders, writes and exits, so its footprint
is not resident between passes.

Usage: python -m plan.cli [--out-dir ~/research/dashboard]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from plan.master_plan import read_master_plan
from plan.reconcile_sources import (
    read_catalogue_rows, read_design_sections, read_ledger_rows, reconcile,
)
from plan.rulings import read_rulings
from statuswall.evidence import NOT_MEASURED, ProbeResult
from statuswall.master_plan_board import render_master_plan_page
from statuswall.ruling_conformance import (
    PROBES, assess_rulings, render_ruling_conformance_page,
)
from statuswall.segment_probes import SEGMENT_PROBES

# BF-10: one registry, assembled here. The segment and learned-brain probes read
# journals and model registries and are deliberately not imported by
# `ruling_conformance`, which several boards import for its renderer alone.
ALL_PROBES = {**PROBES, **SEGMENT_PROBES}

REPO = Path(__file__).resolve().parents[2]
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"
REGISTER = REPO / "docs" / "rulings.json"
DECISIONS = REPO / "docs" / "reconciliation-decisions.json"


def measure_rows(slices) -> dict[str, ProbeResult]:
    """Run each row's named probe. A probe that raises measures nothing."""
    results: dict[str, ProbeResult] = {}
    for plan_slice in slices:
        for row in plan_slice.rows:
            if row.decided:
                # BLOCKED and DECLINED are decisions, not measurements, and the
                # row says so in the document rather than here.
                results[row.id] = ProbeResult(
                    NOT_MEASURED, f"{row.decided} by decision, not measured",
                    "docs/AJIT-MASTER-PLAN.md")
                continue
            if row.probe is None:
                results[row.id] = ProbeResult(
                    NOT_MEASURED, "no probe is named for this row",
                    "docs/AJIT-MASTER-PLAN.md")
                continue
            if row.probe not in ALL_PROBES:
                results[row.id] = ProbeResult(
                    NOT_MEASURED, f"{row.probe} is named but not implemented",
                    "statuswall.ruling_conformance.PROBES + "
                    "statuswall.segment_probes.SEGMENT_PROBES")
                continue
            try:
                results[row.id] = ALL_PROBES[row.probe]()
            except Exception as failure:
                results[row.id] = ProbeResult(
                    NOT_MEASURED,
                    f"{row.probe} raised {type(failure).__name__}: {failure}",
                    row.probe)
    return results


def read_decisions(path: Path = DECISIONS) -> dict[str, tuple[str, str]]:
    """Written resolutions. Absent is not an error - it means none yet."""
    if not path.is_file():
        return {}
    held = json.loads(path.read_text()).get("decisions", {})
    return {key: tuple(value) for key, value in held.items()}


def sweep(slices):
    """All three populations, classified against the spine."""
    members = (read_ledger_rows(Path.home() / "research" / "ledger" / "merged")
               + read_catalogue_rows(Path.home() / "research" / "FEATURES.md")
               + read_design_sections([Path.home() / "research",
                                       REPO / "docs" / "superpowers"]))
    return reconcile(members, slices, read_decisions())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "research" / "dashboard")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="render without the corpus sweep; the reconciliation "
                             "block then reads NOT MEASURED rather than zero")
    args = parser.parse_args()

    slices = read_master_plan(SPINE)
    rulings = read_rulings(REGISTER)
    results = measure_rows(slices)
    resolutions = [] if args.skip_sweep else sweep(slices)

    now_s = int(time.time())
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now_s))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ajit-master-plan.html").write_text(
        render_master_plan_page(slices, results, resolutions, now_s, stamp))
    (args.out_dir / "ruling-conformance.html").write_text(
        render_ruling_conformance_page(
            assess_rulings(rulings, slices, ALL_PROBES), now_s))

    unassigned = sum(1 for r in resolutions if r.outcome == "unassigned")
    print(json.dumps({
        "slices": len(slices),
        "rows": sum(len(s.rows) for s in slices),
        "members_swept": len(resolutions),
        "unassigned": unassigned,
        "rulings": len(rulings),
        "out_dir": str(args.out_dir),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
