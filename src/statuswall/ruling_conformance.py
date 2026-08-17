"""One board row per human ruling, measured or explicitly not measured.

A ruling honoured in a document and not in running code is the failure this
board exists to catch. RL-011 was given 2026-08-03 - three independent bots with
their own data, features and architecture - and measured at 2 of 22 built on
2026-08-17. Nothing on any board said so, because no board knew the ruling
existed.

Two numbers per ruling, answering different questions:

  COVERAGE  does the PLAN have rows for this, per its scope? Four of four
            segments, twelve of twelve brains, one row anywhere for shared.
  STATE     does the RUNNING SYSTEM do it? Measured by the named probe.

A ruling can be fully covered and still fail its probe. That is not a
contradiction - it is the difference between planned and true, and collapsing
the two is how a plan starts vouching for a system.

A ruling naming no probe renders NOT MEASURED and never green (Rule 8). A probe
that raises measures nothing rather than taking the board down: a board that
fails to render tells the reader less than a board with one dark row.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import time
from pathlib import Path

from plan.master_plan import PlanSlice, read_master_plan
from plan.rulings import Ruling
from plan.scope_coverage import Coverage, cover_ruling
from statuswall.evidence import (
    DEGRADED, NOT_MEASURED, OK, PARTIAL, ProbeResult, STATE_LABEL,
)
from statuswall.staleness_banner import render_staleness_banner

REPO = Path(__file__).resolve().parents[2]
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"
REGISTER = REPO / "docs" / "rulings.json"


# --- probes ---------------------------------------------------------------
#
# Each measures the RUNNING SYSTEM, never whether a test passed. A test proves
# the logic; a probe answers whether the thing is true on this box right now.

def probe_store_offloaded(bucket: str = "gs://capture-raw-data4134") -> ProbeResult:
    """RL-020: the observed record must exist somewhere other than this disk."""
    try:
        listing = subprocess.run(
            ["gcloud", "storage", "ls", f"{bucket}/store/funding/"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as failure:
        return ProbeResult(NOT_MEASURED, f"could not reach the bucket: {failure!r}",
                           f"gcloud storage ls {bucket}/store/funding/")
    if listing.returncode != 0:
        return ProbeResult(
            DEGRADED,
            "the store prefix is absent from the bucket - the observed funding "
            "record exists on one disk only",
            f"gcloud storage ls {bucket}/store/funding/ -> exit {listing.returncode}")
    lines = [l for l in listing.stdout.splitlines() if l.strip()]
    if not lines:
        return ProbeResult(DEGRADED, "the store prefix exists and is empty",
                           f"{bucket}/store/funding/ listed 0 objects")
    return ProbeResult(OK, f"{len(lines)} object(s) under store/funding",
                       f"{bucket}/store/funding/ listed {len(lines)} object(s)")


def probe_memory_reachable() -> ProbeResult:
    """RL-016: the memory carrying 'interview, do not assume' must be loadable.

    Memory is keyed by working directory. A session started from `/` reads a
    different key from one started at home, and on 2026-08-17 that key was
    empty while twelve real memories sat under the other one.
    """
    generic = Path.home() / ".claude" / "projects" / "-" / "memory"
    named = Path.home() / ".claude" / "projects" / "-home-anushadudekula71" / "memory"
    if not named.is_dir():
        return ProbeResult(NOT_MEASURED, "no project memory directory found", str(named))
    generic_count = len(list(generic.glob("*.md"))) if generic.is_dir() else 0
    named_count = len(list(named.glob("*.md")))
    if generic_count < named_count:
        return ProbeResult(
            DEGRADED,
            f"a session started from / sees {generic_count} of {named_count} memories",
            f"{generic} vs {named}")
    return ProbeResult(OK, f"{named_count} memories reachable from either key",
                       f"{generic} -> {generic_count}, {named} -> {named_count}")


def probe_paper_engine_running(
    state_dir: Path = Path.home() / "capture" / "paper" / "forward",
) -> ProbeResult:
    """RL-005, RL-017, RL-020: paper trading is the experimentation ground.

    PARTIAL rather than OK while the strategy makes no edge claim. The engine
    working and the strategy being a strategy are different facts, and a green
    tile over `plumbing-momentum` would be the second one asserted from the
    first.
    """
    heartbeat = state_dir / "heartbeat.json"
    if not heartbeat.is_file():
        return ProbeResult(NOT_MEASURED, "no heartbeat written", str(heartbeat))
    try:
        beat = json.loads(heartbeat.read_text())
        age_s = (time.time_ns() - int(beat["written_at_ns"])) / 1e9
    except (ValueError, OSError, KeyError) as failure:
        return ProbeResult(NOT_MEASURED, f"unreadable heartbeat: {failure!r}",
                           str(heartbeat))
    detail = (f"strategy {beat.get('strategy')!r}, edge claim "
              f"{bool(beat.get('makes_edge_claim'))}, {beat.get('fills', 0)} fill(s), "
              f"heartbeat {age_s:.0f}s old")
    if age_s > 600:
        return ProbeResult(DEGRADED, detail, str(heartbeat))
    if not beat.get("makes_edge_claim", False):
        return ProbeResult(PARTIAL, detail, str(heartbeat))
    return ProbeResult(OK, detail, str(heartbeat))


def probe_enforcement_live(
    hooks_dir: Path = Path.home() / ".claude" / "hooks",
    settings: Path = Path.home() / ".claude" / "settings.json",
) -> ProbeResult:
    """RL-021: the enforcement layer is a hook that fires, not a paragraph."""
    required = ("require-plan-row.sh", "inject-plan-index.sh")
    present = [n for n in required if (hooks_dir / n).is_file()]
    registered = settings.read_text() if settings.is_file() else ""
    wired = [n for n in required if n in registered]
    detail = f"{len(present)}/2 hook scripts present, {len(wired)}/2 registered"
    proof = f"{hooks_dir} and {settings}"
    if len(wired) == len(present) == len(required):
        return ProbeResult(OK, detail, proof)
    if present or wired:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


def probe_rulings_register_loads(register: Path = REGISTER) -> ProbeResult:
    """RL-021: the register is only a record if it can be read mechanically."""
    from plan.rulings import read_rulings
    if not register.is_file():
        return ProbeResult(NOT_MEASURED, "no register on disk", str(register))
    try:
        rulings = read_rulings(register)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the register does not parse: {failure!r}",
                           str(register))
    scoped = sum(1 for r in rulings if r.scope)
    return ProbeResult(OK, f"{len(rulings)} rulings parse, {scoped} scoped",
                       str(register))


def probe_spine_parses(spine: Path = SPINE) -> ProbeResult:
    """RL-007: an unparseable plan is a plan nothing can measure against."""
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document on disk", str(spine))
    try:
        slices = read_master_plan(spine)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the spine does not parse: {failure!r}",
                           str(spine))
    rows = sum(len(s.rows) for s in slices)
    empty = [s.key for s in slices if not s.rows]
    return ProbeResult(
        OK,
        f"{len(slices)} slices, {rows} rows, {len(empty)} slice(s) with no rows yet",
        str(spine))


def probe_spine_present(spine: Path = SPINE) -> ProbeResult:
    """RL-007: the plan states its own authority, or it has none."""
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document on disk", str(spine))
    body = spine.read_text()
    claims = "top of the authority chain" in body.lower()
    supersedes = "2026-08-09-full-build-master-plan.md" in body
    if claims and supersedes:
        return ProbeResult(OK, "claims authority and names what it supersedes",
                           str(spine))
    return ProbeResult(PARTIAL,
                       f"authority claimed: {claims}, supersession named: {supersedes}",
                       str(spine))


def probe_scope_arithmetic(spine: Path = SPINE,
                           register: Path = REGISTER) -> ProbeResult:
    """RL-021: coverage must read as a fraction, not as a yes.

    Reports how many rulings the plan actually covers. A low number is the
    correct result today and is the whole reason the arithmetic exists.
    """
    from plan.rulings import read_rulings
    if not (spine.is_file() and register.is_file()):
        return ProbeResult(NOT_MEASURED, "plan or register missing", str(spine))
    try:
        slices = read_master_plan(spine)
        rulings = read_rulings(register)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"{failure!r}", str(spine))
    covers = [cover_ruling(r, slices) for r in rulings]
    resolved = sum(1 for c in covers if c.resolved)
    partial = [f"{c.subject} {c.have}/{c.need}" for c in covers if not c.resolved]
    detail = f"{resolved}/{len(covers)} rulings covered; short: {', '.join(partial[:6])}"
    if resolved == len(covers):
        return ProbeResult(OK, detail, str(spine))
    return ProbeResult(PARTIAL, detail, str(spine))


def probe_authority_chain_consistent(repo: Path = REPO) -> ProbeResult:
    """RL-007, RL-019: exactly one document may claim the order of work."""
    old = repo / "docs" / "superpowers" / "plans" / "2026-08-09-full-build-master-plan.md"
    goal = (repo / "docs" / "superpowers" / "specs"
            / "2026-08-08-final-project-goal-design.md")
    superseded = old.is_file() and "SUPERSEDED" in old.read_text().upper()
    # The HEADING, not the string. A bare "3b" appears in unrelated prose - the
    # goal document cites `ARCHITECTURE.md §3b` about coinbase - and the loose
    # version of this check reported the section present before it was written.
    records_ruling = goal.is_file() and bool(
        re.search(r"^##\s+3b\.", goal.read_text(), re.MULTILINE))
    detail = (f"old plan marked superseded: {superseded}, "
              f"goal records §3b: {records_ruling}")
    proof = f"{old.name}, {goal.name}"
    if superseded and records_ruling:
        return ProbeResult(OK, detail, proof)
    if superseded or records_ruling:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(
        DEGRADED,
        "two documents still claim authority over the order of work: " + detail,
        proof)


def probe_reconciliation_sweeps(spine: Path = SPINE) -> ProbeResult:
    """RL-021: a sweep that does not run cannot report anything skipped.

    PARTIAL while members remain unassigned, which is the honest reading today
    and for a long time: 2,738 corpus members against 13 plan rows. OK would
    require every member assigned, declined or written off as prose.
    """
    from plan.cli import read_decisions
    from plan.reconcile_sources import (
        read_catalogue_rows, read_design_sections, read_ledger_rows, reconcile,
    )
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document to sweep against",
                           str(spine))
    try:
        members = (read_ledger_rows(Path.home() / "research" / "ledger" / "merged")
                   + read_catalogue_rows(Path.home() / "research" / "FEATURES.md")
                   + read_design_sections([Path.home() / "research",
                                           REPO / "docs" / "superpowers"]))
        resolutions = reconcile(members, read_master_plan(spine), read_decisions())
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the sweep failed: {failure!r}", str(spine))
    unassigned = sum(1 for r in resolutions if r.outcome == "unassigned")
    detail = f"{len(resolutions)} members swept, {unassigned} unassigned"
    proof = "plan.reconcile_sources over ledger, catalogue and design sections"
    return ProbeResult(OK if unassigned == 0 else PARTIAL, detail, proof)


def _board_freshness(name: str, out_dir: Path) -> ProbeResult:
    """A generated board is only evidence while it is recent."""
    page = out_dir / name
    if not page.is_file():
        return ProbeResult(NOT_MEASURED, f"{name} has never been generated", str(page))
    age_s = time.time() - page.stat().st_mtime
    detail = f"{name} generated {age_s / 60:.0f} min ago, {page.stat().st_size} bytes"
    if age_s > 24 * 3600:
        return ProbeResult(DEGRADED, detail + " — the generator has stopped", str(page))
    return ProbeResult(OK, detail, str(page))


def probe_plan_board_rendered(
        out_dir: Path = Path.home() / "research" / "dashboard") -> ProbeResult:
    """RL-012: the board is a display of measured state, so it must exist."""
    return _board_freshness("ajit-master-plan.html", out_dir)


def probe_ruling_conformance(
        out_dir: Path = Path.home() / "research" / "dashboard") -> ProbeResult:
    """RL-021: a conformance board nobody renders measures nothing."""
    return _board_freshness("ruling-conformance.html", out_dir)


PROBES = {
    "probe_store_offloaded": probe_store_offloaded,
    "probe_reconciliation_sweeps": probe_reconciliation_sweeps,
    "probe_plan_board_rendered": probe_plan_board_rendered,
    "probe_ruling_conformance": probe_ruling_conformance,
    "probe_memory_reachable": probe_memory_reachable,
    "probe_paper_engine_running": probe_paper_engine_running,
    "probe_enforcement_live": probe_enforcement_live,
    "probe_rulings_register_loads": probe_rulings_register_loads,
    "probe_spine_parses": probe_spine_parses,
    "probe_spine_present": probe_spine_present,
    "probe_scope_arithmetic": probe_scope_arithmetic,
    "probe_authority_chain_consistent": probe_authority_chain_consistent,
}


def assess_rulings(
    rulings: list[Ruling], slices: list[PlanSlice],
) -> list[tuple[Ruling, Coverage, ProbeResult]]:
    """Coverage from the plan, state from the probe, neither from the ruling."""
    assessments: list[tuple[Ruling, Coverage, ProbeResult]] = []
    for ruling in rulings:
        coverage = cover_ruling(ruling, slices)
        if ruling.probe is None:
            result = ProbeResult(NOT_MEASURED, "no probe is named for this ruling",
                                 "docs/rulings.json")
        elif ruling.probe not in PROBES:
            result = ProbeResult(
                NOT_MEASURED, f"{ruling.probe} is named but not implemented",
                "statuswall.ruling_conformance.PROBES")
        else:
            try:
                result = PROBES[ruling.probe]()
            except Exception as failure:
                # A broken probe measures nothing. It must not take the board
                # down: a board that fails to render tells the reader less than
                # a board carrying one dark row.
                result = ProbeResult(
                    NOT_MEASURED,
                    f"{ruling.probe} raised {type(failure).__name__}: {failure}",
                    ruling.probe)
        assessments.append((ruling, coverage, result))
    return assessments


def render_ruling_conformance_page(assessments, generated_at_epoch_s: int) -> str:
    """One row per ruling: what was said, how covered, and what is true."""
    rows = []
    for ruling, coverage, result in assessments:
        label = STATE_LABEL.get(result.state, result.state).upper()
        gap = ", ".join(coverage.missing[:4]) if coverage.missing else ""
        rows.append(
            "<tr>"
            f"<td class='id'>{html.escape(ruling.id)}</td>"
            f"<td class='date'>{html.escape(ruling.date)}</td>"
            f"<td class='said'>{html.escape(ruling.verbatim[:180])}</td>"
            f"<td class='scope'>{html.escape(ruling.scope)}</td>"
            f"<td class='cover'>{coverage.have} / {coverage.need}</td>"
            f"<td class='gap'>{html.escape(gap)}</td>"
            f"<td class='state s-{html.escape(result.state)}'>{html.escape(label)}</td>"
            f"<td class='detail'>{html.escape(result.detail)}</td>"
            f"<td class='proof'>{html.escape(result.proof)}</td>"
            "</tr>")

    banner = render_staleness_banner(
        generated_at_epoch_s,
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(generated_at_epoch_s)))

    return (
        "<h1>Ruling conformance</h1>"
        f"{banner}"
        "<table><thead><tr>"
        "<th>Ruling</th><th>Date</th><th>What the user said</th><th>Scope</th>"
        "<th>Covered</th><th>Missing</th><th>State</th><th>Detail</th><th>Proof</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p class='foot'>Covered counts plan rows against the ruling's scope - "
        "four segment slices for per-segment, twelve brains for per-brain. State "
        "is measured by the named probe and answers a different question: whether "
        "the running system does it. A ruling with no probe reads NOT MEASURED "
        "and is never green.</p>")
