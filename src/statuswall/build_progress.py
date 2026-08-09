"""The build-progress board: the master plan's phases, measured, never asserted.

`docs/superpowers/plans/2026-08-09-full-build-master-plan.md` orders the full
build as phases A-J. This page answers one question per phase - how much of it
is lit on the wall - by grouping the SAME feature assessments the status wall
renders, under a declared section-to-phase mapping. The mapping is a design
fact and lives here; every count under it is measured.

Two phases have no catalogue section and get their own probes instead:

- Phase I (ledger closure) is measured by counting UNRESOLVED rows in the
  merged requirements ledger - the number the phase exists to drive to zero.
- Phase J (paper trading) is measured by the forward-paper journal directory.
  No journal, no paper trading, and the row says so (Rule 8).

If FEATURES.md gains a section this mapping does not name, the build refuses
to render rather than silently dropping the section's features from every
phase count.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from statuswall.catalogue import Feature
from statuswall.evidence import NOT_BUILT, ProbeResult, SEVERITY_ORDER, STATE_LABEL

# Phase key, phase title, and the FEATURES.md sections whose features it owns.
# Every catalogue section appears exactly once; enforced by _verify_mapping.
PLAN_PHASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A", "Data & ingestion completion", ("1",)),
    ("B", "Feature engineering", ("2",)),
    ("C", "Models & strategy families", ("3", "4")),
    ("D", "BULL / BEAR / PROFIT-TAIL, brain 1", ("3b",)),
    ("E", "Portfolio & risk", ("6", "7")),
    ("F", "Execution, ops & live-prep", ("5", "9", "10")),
    ("G", "Brains 2-3 + intelligence layer", ("11", "5b")),
    ("H", "Dashboard & observability", ("13",)),
    ("I", "Ledger closure & governance", ("8", "12")),
)

_UNRESOLVED_CELL = "UNRESOLVED"


class UnmappedSection(Exception):
    """FEATURES.md carries a section the phase mapping does not assign."""


@dataclass(frozen=True)
class PhaseProgress:
    """Measured state of one plan phase: counts only, no judgement."""
    key: str
    title: str
    total: int
    state_counts: dict[str, int]

    @property
    def lit(self) -> int:
        """Features with any measured state at all - everything but NOT BUILT."""
        return self.total - self.state_counts.get(NOT_BUILT, 0)


def _verify_mapping(features: list[Feature]) -> None:
    assigned: dict[str, str] = {}
    for key, _title, sections in PLAN_PHASES:
        for section in sections:
            if section in assigned:
                raise UnmappedSection(
                    f"section {section} is claimed by phases {assigned[section]} and {key}")
            assigned[section] = key
    present = {feature.section_idx for feature in features}
    orphans = sorted(present - assigned.keys())
    if orphans:
        raise UnmappedSection(
            "FEATURES.md sections with no phase in PLAN_PHASES: "
            + ", ".join(orphans)
            + " - assign them or the board would silently under-count a phase")


def summarise_build_progress(
        features: list[Feature],
        results: dict[str, ProbeResult]) -> list[PhaseProgress]:
    """Group the wall's per-feature assessments into per-phase counts."""
    _verify_mapping(features)
    by_section: dict[str, list[Feature]] = {}
    for feature in features:
        by_section.setdefault(feature.section_idx, []).append(feature)

    phases = []
    for key, title, sections in PLAN_PHASES:
        owned = [f for section in sections for f in by_section.get(section, [])]
        counts: dict[str, int] = {}
        for feature in owned:
            state = results[feature.key].state
            counts[state] = counts.get(state, 0) + 1
        phases.append(PhaseProgress(key=key, title=title, total=len(owned),
                                    state_counts=counts))
    return phases


def count_unresolved_ledger_rows(ledger_root: Path) -> int | None:
    """Count UNRESOLVED rows across the merged ledger slices, or None if absent.

    Mirrors the rules of `research/scripts/build_ledger_index.py`, the reference
    parser, so this board and the ledger index can never disagree by method:
    a data row has three or more cells (which excludes each slice's two-column
    totals table - the first draft here counted those and over-reported by 4),
    the status is found by value in any cell, bold markers are stripped, and
    suffixes like "(partial)" or "as primary" are split off.
    """
    if not ledger_root.is_dir():
        return None
    unresolved = 0
    for slice_path in sorted(ledger_root.glob("*.md")):
        for line in slice_path.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            for cell in cells:
                token = cell.replace("**", "").strip()
                head = token.split("(")[0].split(" as ")[0].strip()
                if head == _UNRESOLVED_CELL:
                    unresolved += 1
                    break
    return unresolved


def probe_forward_paper(capture_root: Path) -> tuple[bool, str]:
    """Is forward paper trading journalling? (answer, evidence)."""
    journal_dir = capture_root / "paper" / "forward"
    if not journal_dir.is_dir():
        return False, f"no journal directory at {journal_dir}"
    journals = sorted(journal_dir.glob("*.ndjson"))
    if not journals:
        return False, f"{journal_dir} exists but holds no journal"
    return True, f"{len(journals)} journal file(s), latest {journals[-1].name}"


_PAGE_STYLE = """
  :root { color-scheme: dark; --ground:#080D11; --panel:#0F171D; --rule:#1E2C35;
    --ink:#E8F0F4; --ink-2:#93A7B3; --ink-3:#61737E; --lit:#3FD68A; --dark:#26343D;
    --warn:#E8B04F; --fail:#E85A4F;
    --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace; }
  @media (prefers-color-scheme: light) {
    :root { color-scheme: light; --ground:#EDF1F4; --panel:#FFFFFF; --rule:#D6DFE5;
      --ink:#0B1418; --ink-2:#465761; --ink-3:#6D808B; --lit:#12764A; --dark:#C9D4DB;
      --warn:#9A6B0F; --fail:#B3261E; } }
  * { box-sizing:border-box; } body { margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--mono); padding:clamp(16px,4vw,40px); }
  h1 { font-size:clamp(18px,4vw,26px); margin:0 0 4px; }
  .sub { color:var(--ink-3); font-size:12px; margin-bottom:24px; }
  table { border-collapse:collapse; width:100%; max-width:880px; }
  th,td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--rule);
    font-size:13px; vertical-align:top; }
  th { color:var(--ink-3); font-size:11px; letter-spacing:.14em; text-transform:uppercase; }
  .bar { display:flex; height:10px; border-radius:5px; overflow:hidden;
    background:var(--dark); min-width:160px; }
  .bar span { background:var(--lit); }
  .n { color:var(--ink-2); white-space:nowrap; }
  .zero { color:var(--ink-3); }
  .fail { color:var(--fail); } .warn { color:var(--warn); } .lit { color:var(--lit); }
  .foot { margin-top:24px; color:var(--ink-3); font-size:11px; max-width:880px; }
"""


def _phase_row(phase: PhaseProgress) -> str:
    pct = 0 if phase.total == 0 else round(100 * phase.lit / phase.total)
    breakdown = " · ".join(
        f"{STATE_LABEL[state]} {phase.state_counts[state]}"
        for state in SEVERITY_ORDER
        if phase.state_counts.get(state) and state != NOT_BUILT)
    return (
        f"<tr><td>{phase.key}</td>"
        f"<td>{html.escape(phase.title)}</td>"
        f"<td><div class='bar'><span style='width:{pct}%'></span></div></td>"
        f"<td class='n'>{phase.lit} / {phase.total} lit"
        f"{' — ' + html.escape(breakdown) if breakdown else ''}</td></tr>")


def render_build_progress_page(
        phases: list[PhaseProgress],
        unresolved_rows: int | None,
        paper_running: bool,
        paper_evidence: str,
        generated_at: str) -> str:
    """The whole page. Every number on it arrived through an argument that was
    measured by the caller; nothing here invents one."""
    rows = "".join(_phase_row(p) for p in phases)

    if unresolved_rows is None:
        ledger_cell = "<td class='warn' colspan='2'>NOT MEASURED — ledger not found</td>"
    else:
        cls = "lit" if unresolved_rows == 0 else "warn"
        ledger_cell = (f"<td><div class='bar'><span style='width:"
                       f"{100 if unresolved_rows == 0 else 0}%'></span></div></td>"
                       f"<td class='n {cls}'>{unresolved_rows} UNRESOLVED rows "
                       f"(done at 0)</td>")

    paper_cls = "lit" if paper_running else "fail"
    paper_label = "RUNNING" if paper_running else "NOT RUNNING"
    total = sum(p.total for p in phases)
    lit = sum(p.lit for p in phases)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Build progress — full-build master plan</title>
<style>{_PAGE_STYLE}</style></head><body>
<h1>Build progress — phases A–J</h1>
<div class="sub">generated {html.escape(generated_at)} ·
{lit} / {total} catalogue features lit ·
plan: docs/superpowers/plans/2026-08-09-full-build-master-plan.md</div>
<table>
<tr><th>Phase</th><th>Scope</th><th></th><th>Measured</th></tr>
{rows}
<tr><td>I+</td><td>Requirements ledger sweep</td>{ledger_cell}</tr>
<tr><td>J</td><td>Paper trading (forward journal)</td>
<td class="{paper_cls}">{paper_label}</td>
<td class="n">{html.escape(paper_evidence)}</td></tr>
</table>
<div class="foot">Every count is measured: phase rows group the status wall's
per-feature probe results under the declared section mapping in
statuswall/build_progress.py; the ledger row counts UNRESOLVED cells in
research/ledger/merged; the paper row probes the forward journal directory.
A dark bar is the correct bar for an unbuilt phase.</div>
</body></html>
"""
