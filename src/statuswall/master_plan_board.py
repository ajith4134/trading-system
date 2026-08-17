"""The plan board: how much is built, which slice is active, what is next.

The user asked "how much is complete and why is it taking so long" six times
between 2026-08-08 and 2026-08-17. This board exists so the question answers
itself. Everything on it is measured - counts from probes, reconciliation
totals from the sweep - and there are deliberately NO dates and NO effort
estimates, because an estimate on this project would be a guess presented as a
number, and a wrong one is what makes the question unanswerable rather than
answered.

Two renderings would flatter and both are refused:

  An EMPTY slice renders `0 / 0`, never a completion mark. A slice with no rows
  is unplanned work, not finished work - and four of the six slices are empty
  today, on purpose, until their inventory is reviewed with the user.

  The UNASSIGNED count is published rather than summarised away. It is large:
  2,738 corpus members swept and almost all of them unassigned. That number has
  been invisible for fifteen days, and hiding it again would defeat the board.

Page style is borrowed from `build_progress` rather than reinvented, so the
boards read as one system.
"""
from __future__ import annotations

import html

from plan.master_plan import PlanRow, PlanSlice
from statuswall.build_progress import _PAGE_STYLE
from statuswall.evidence import BUILT, NOT_MEASURED, OK, ProbeResult, STATE_LABEL
from statuswall.staleness_banner import render_staleness_banner

# A row counts as done when a probe measured it OK or BUILT. Nothing else does -
# PARTIAL is not done, and NOT MEASURED is emphatically not done.
_DONE = frozenset({OK, BUILT})


def _state_of(row: PlanRow, results: dict[str, ProbeResult]) -> ProbeResult:
    """The measured state of a row, or NOT MEASURED. Never a default of OK."""
    return results.get(row.id, ProbeResult(
        NOT_MEASURED, "no probe result for this row", "statuswall.master_plan_board"))


def _built_count(plan_slice: PlanSlice, results: dict[str, ProbeResult]) -> int:
    return sum(1 for row in plan_slice.rows
               if _state_of(row, results).state in _DONE)


def active_slice(slices: list[PlanSlice], results: dict[str, ProbeResult]) -> str:
    """The first slice not fully built. Slices are strict: no working ahead.

    An empty slice IS the active slice once the one above it is built, because
    empty means unplanned and unplanned is the next thing to do - not a slice to
    skip past.
    """
    for plan_slice in slices:
        if not plan_slice.rows:
            return plan_slice.key
        if _built_count(plan_slice, results) < len(plan_slice.rows):
            return plan_slice.key
    return slices[-1].key if slices else ""


def next_row(slices: list[PlanSlice],
             results: dict[str, ProbeResult]) -> PlanRow | None:
    """The first not-done row of the active slice, or None if it has no rows."""
    key = active_slice(slices, results)
    for plan_slice in slices:
        if plan_slice.key != key:
            continue
        for row in plan_slice.rows:
            if _state_of(row, results).state not in _DONE:
                return row
    return None


def _slice_row(plan_slice: PlanSlice, results: dict[str, ProbeResult],
               active: str) -> str:
    total = len(plan_slice.rows)
    built = _built_count(plan_slice, results)
    width = int(100 * built / total) if total else 0
    mark = " ←" if plan_slice.key == active else ""
    cls = "lit" if total and built == total else ("warn" if total else "zero")
    note = "" if total else " — inventory pending review"
    return (f"<tr><td>{html.escape(plan_slice.key)}{mark}</td>"
            f"<td>{html.escape(plan_slice.title)}{note}</td>"
            f"<td><div class='bar'><span style='width:{width}%'></span></div></td>"
            f"<td class='n {cls}'>{built} / {total}</td></tr>")


def _detail_row(row: PlanRow, results: dict[str, ProbeResult]) -> str:
    result = _state_of(row, results)
    label = STATE_LABEL.get(result.state, result.state).upper()
    cls = "lit" if result.state in _DONE else "warn"
    return (f"<tr><td>{html.escape(row.id)}</td>"
            f"<td>{html.escape(row.does)}</td>"
            f"<td>{html.escape(' '.join(row.satisfies))}</td>"
            f"<td class='n {cls}'>{html.escape(label)}</td>"
            f"<td class='n'>{html.escape(result.detail[:120])}</td></tr>")


def _reconciliation_block(resolutions) -> str:
    """Per-population outcome counts, with unassigned styled as the failure."""
    if not resolutions:
        return ("<p class='foot'>Reconciliation NOT MEASURED — the sweep did not "
                "run for this render.</p>")
    counts: dict[str, dict[str, int]] = {}
    for resolution in resolutions:
        bucket = counts.setdefault(resolution.member.population, {})
        bucket[resolution.outcome] = bucket.get(resolution.outcome, 0) + 1

    rows = []
    for population in sorted(counts):
        outcomes = counts[population]
        cells = " · ".join(
            f"<span class='{'fail' if k == 'unassigned' else 'n'}'>"
            f"{html.escape(k.upper())} {v}</span>"
            for k, v in sorted(outcomes.items()))
        rows.append(f"<tr><td>{html.escape(population)}</td>"
                    f"<td class='n'>{sum(outcomes.values())}</td>"
                    f"<td>{cells}</td></tr>")
    total_unassigned = sum(c.get("unassigned", 0) for c in counts.values())
    return (f"<h2>Reconciliation — {total_unassigned} UNASSIGNED</h2>"
            "<table><tr><th>Population</th><th>Members</th><th>Outcomes</th></tr>"
            + "".join(rows) + "</table>")


def render_master_plan_page(slices: list[PlanSlice],
                            results: dict[str, ProbeResult],
                            resolutions,
                            generated_at_epoch_s: int,
                            generated_at: str = "") -> str:
    """The whole page. Every number arrived measured; nothing here invents one."""
    active = active_slice(slices, results)
    following = next_row(slices, results)
    slice_rows = "".join(_slice_row(s, results, active) for s in slices)
    detail_rows = "".join(_detail_row(row, results)
                          for s in slices for row in s.rows)

    if following is None:
        next_block = (f"<div class='sub'>ACTIVE SLICE <b>{html.escape(active)}</b> — "
                      f"no rows yet; its inventory is reviewed with the user before "
                      f"the slice starts.</div>")
    else:
        next_block = (f"<div class='sub'>ACTIVE SLICE <b>{html.escape(active)}</b> · "
                      f"NEXT ROW <b>{html.escape(following.id)}</b> — "
                      f"{html.escape(following.does)}</div>")

    banner = render_staleness_banner(generated_at_epoch_s,
                                     generated_at or "unknown")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>AJIT MASTER PLAN — measured</title>
<style>{_PAGE_STYLE}</style></head><body>
<h1>AJIT MASTER PLAN</h1>
{banner}
{next_block}
<table>
<tr><th>Slice</th><th>Scope</th><th></th><th>Built</th></tr>
{slice_rows}
</table>
<h2>Rows</h2>
<table>
<tr><th>Row</th><th>Does</th><th>Satisfies</th><th>State</th><th>Detail</th></tr>
{detail_rows}
</table>
{_reconciliation_block(resolutions)}
<div class="foot">Counts are measured by the probe each row names; a row with no
probe result reads NOT MEASURED and is never counted as built. A slice with no
rows reads 0 / 0 — unplanned work, not finished work. No dates and no effort
estimates appear here on purpose: an estimate would be a guess presented as a
number. Plan: docs/AJIT-MASTER-PLAN.md. Rulings: docs/rulings.json.</div>
</body></html>"""
