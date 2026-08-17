"""The three populations every plan row is reconciled against.

The ledger has reconciled rows since 2026-08-08 and it works: 1,491 rows, each
CLAIMED, PLANNED, PRIOR-ART, DECLINED or UNRESOLVED. What nothing reconciled
was DESIGN SECTIONS - argued-through prose in a document that never became a
row. Measured 2026-08-17: 'Goodhart defence', 'attention scarcity' and the
examination-hall framing appear in four, two and three design files
respectively, and in ZERO ledger rows. The examination-hall framing is the
user's own ruling, written into the goal document as roughly 140 lines of §5a,
and it produced no work at all.

So a section is a first-class member of the sweep, and `prose` - meaning "this
section implies no module" - is a legitimate resolution that a human WRITES.
It is never inferred, because inferring it is the failure being fixed.

Matching is on a word boundary, not a substring. `FE-01` must not be satisfied
by a row citing `FE-014`; an over-matching sweep reports coverage it does not
have, which is the same lie as an under-swept one and harder to notice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from plan.master_plan import PlanSlice
from statuswall.catalogue import read_catalogue

OUTCOMES = frozenset({"assigned", "declined", "blocked", "prose", "unassigned"})

# Only these three may be WRITTEN. `assigned` is measured from the spine and
# `unassigned` is what is left over - typing either would be asserting a
# reconciliation instead of performing one.
_WRITABLE = frozenset({"declined", "blocked", "prose"})

_HEADING = re.compile(r"^(?P<hashes>#{2,3})\s+(?P<title>.+?)\s*$")
_LEDGER_ID = re.compile(r"^\|\s*(?P<id>[A-Z]{2,3}-\d+)\s*\|")
_TABLE_ROW = re.compile(r"^\|(?!\s*[-: ]+\|)(?P<first>[^|]+)\|")
_SEPARATOR_ROW = re.compile(r"^\|[\s:-]*\|")
_IDENTIFIER = re.compile(r"^[A-Z]{2,3}-\d+$")


@dataclass(frozen=True)
class SourceMember:
    population: str
    identifier: str
    title: str
    origin: str


@dataclass(frozen=True)
class Resolution:
    member: SourceMember
    outcome: str
    detail: str


def read_ledger_rows(root: Path) -> list[SourceMember]:
    """Every row across the merged ledger slices, identified or not.

    An earlier version of this reader matched only rows carrying an `XX-000`
    identifier and found 1,067 of the ledger's 1,491. The missing 424 live in
    the two `PARTIAL-*` slices, whose tables carry no ID column at all - so a
    sweep built to stop things being silently dropped was silently dropping
    them. Un-identified rows now get a synthesised identifier from their file
    and their first cell, which is stable enough to cite from a plan row and
    honest about being derived.
    """
    members: list[SourceMember] = []
    for path in sorted(Path(root).glob("*.md")):
        lines = path.read_text(errors="ignore").splitlines()
        for index, line in enumerate(lines):
            row = _TABLE_ROW.match(line)
            if not row:
                continue
            # A header row is the one followed by the `|---|` separator. Naming
            # headers by their text would need a growing list of column names
            # and would drop a real row the day one is called "Requirement".
            if index + 1 < len(lines) and _SEPARATOR_ROW.match(lines[index + 1]):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            found = _LEDGER_ID.match(line)
            if found:
                identifier = found["id"]
                title = cells[1] if len(cells) > 1 else ""
            else:
                title = re.sub(r"[*`]", "", cells[0]).strip()
                if not title or title.lower() in {
                        "feature", "idea", "capability", "item", "row", "name"}:
                    continue
                identifier = f"{path.stem}:{title[:80]}"
            members.append(SourceMember(
                population="ledger-row",
                identifier=identifier,
                title=title,
                origin=path.name,
            ))
    return members


def read_catalogue_rows(path: Path) -> list[SourceMember]:
    """Every feature in FEATURES.md, via the catalogue reader the wall uses.

    Deliberately NOT a second parser. `statuswall.catalogue.read_catalogue` is
    already tested, already knows which tables in that document are feature
    tables and which are not, and is the reader every probe result is keyed on -
    so a private copy here would eventually disagree with the wall about what a
    feature is, and the disagreement would show up as phantom coverage.

    A hand-rolled version of this counted 201 rows against the catalogue's 167.
    """
    return [SourceMember(
        population="catalogue-row",
        identifier=f"FEATURES §{feature.section_idx} {feature.name}",
        title=feature.name,
        origin=Path(path).name,
    ) for feature in read_catalogue(Path(path))]


def read_design_sections(roots: list[Path]) -> list[SourceMember]:
    """Every `##` and `###` heading across the design corpus.

    This is the population nothing swept before. A heading is the smallest unit
    of argued design in this corpus - §5a, §1a.2, the Goodhart section - and it
    is the unit at which an idea stops being followed through.
    """
    members: list[SourceMember] = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.md")):
            for line in path.read_text(errors="ignore").splitlines():
                heading = _HEADING.match(line)
                if not heading:
                    continue
                title = heading["title"].strip()
                members.append(SourceMember(
                    population="design-section",
                    identifier=f"{path.name}#{title}",
                    title=title,
                    origin=path.name,
                ))
    return members


def _is_named_by(identifier: str, sources: set[str]) -> bool:
    """Does any plan row's `sources` entry name this member?

    Exact match first. Then, for id-shaped members only, a word-boundary search
    inside a longer phrase - so `ledger FE-014, FE-021` names FE-014 and FE-021
    and does NOT name FE-01.
    """
    if identifier in sources:
        return True
    if _IDENTIFIER.match(identifier):
        pattern = re.compile(rf"(?<![\w-]){re.escape(identifier)}(?![\w-])")
        return any(pattern.search(source) for source in sources)
    return False


def reconcile(members: list[SourceMember],
              slices: list[PlanSlice],
              decisions: dict[str, tuple[str, str]]) -> list[Resolution]:
    """Classify every member. A member with no home is unassigned, loudly."""
    named: set[str] = {source
                       for plan_slice in slices
                       for row in plan_slice.rows
                       for source in row.sources}

    resolutions: list[Resolution] = []
    for member in members:
        decided = decisions.get(member.identifier)
        if decided is not None:
            outcome, detail = decided
            if outcome not in _WRITABLE:
                raise ValueError(
                    f"{member.identifier}: {outcome!r} is invented; a written "
                    f"decision may only be declined, blocked or prose - "
                    f"assigned is measured from the spine and unassigned is "
                    f"what is left")
            resolutions.append(Resolution(member, outcome, detail))
            continue

        if _is_named_by(member.identifier, named):
            resolutions.append(Resolution(member, "assigned", ""))
            continue

        resolutions.append(Resolution(member, "unassigned", ""))
    return resolutions


def summarise(resolutions: list[Resolution]) -> dict[str, dict[str, int]]:
    """Counts per population per outcome. Every member counted exactly once."""
    counts: dict[str, dict[str, int]] = {}
    for resolution in resolutions:
        population = counts.setdefault(resolution.member.population, {})
        population[resolution.outcome] = population.get(resolution.outcome, 0) + 1
    return counts
