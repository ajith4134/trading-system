"""The written spine of AJIT MASTER PLAN - order and decisions, never state.

This parser exists to defend one property. The plan document is committed and
holds what the user settled; the board is generated and holds what probes
measured. If a state value is ever typed into the document the two collapse
back into one artefact that asserts its own health - which is what
`2026-08-09-full-build-master-plan.md` did for eight days while paper trading
ran in contradiction of its own phase J, and what `DECISIONS.md` §13 records
about itself for six days before that.

So `state:` may say `measured by <probe>` and nothing else, and a row missing
any of its eight fields is refused rather than accepted half-specified: an
unwritten decision must be visible as a blank field before code is written,
not as a wrong module afterwards.

`BLOCKED` and `DECLINED` are the exception, and they are not measurements -
they are the user's decisions about a row, so they belong in the document.
Declining is allowed; forgetting is not.

The format, which is markdown a person can read and a parser can trust:

    ## SLICE spot-bot — SPOT BOT

    ### SP-04
      slice:      spot-bot
      does:       compute spot-only microstructure features per symbol
      satisfies:  RL-006 RL-009
      sources:    FEATURES §2.4 · ledger FE-014
      depends on: SP-02
      probe:      probe_spot_features
      accepts:    every value carries a staleness stamp and no value is
                  readable before its availability time
      state:      measured by probe_spot_features
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROW_FIELDS = ("does", "satisfies", "sources", "depends on", "probe", "accepts", "state")
DECIDED_STATES = frozenset({"BLOCKED", "DECLINED"})

# The status wall's vocabulary. Any of these in a `state:` field means somebody
# typed a measurement into the record, which is the defect this parser exists
# to catch - so they are matched by name rather than by a loose heuristic.
_MEASURED_WORDS = frozenset({
    "OK", "BUILT", "NOT BUILT", "PARTIAL", "DEGRADED", "FAILING", "STOPPED",
    "NOT MEASURED",
})

_SLICE = re.compile(r"^##\s+SLICE\s+(?P<key>[\w-]+)\s+—\s+(?P<title>.+?)\s*$")
_ROW_ID = re.compile(r"^###\s+(?P<id>[A-Z]{2,3}-\d+)\s*$")
_FIELD = re.compile(
    r"^\s{2}(?P<name>does|satisfies|sources|depends on|probe|accepts|state|slice)"
    r":\s*(?P<value>.*)$")
_SEPARATORS = re.compile(r"\s*·\s*|\s{2,}")


class MalformedRow(Exception):
    """A plan row that is not a complete contract, named by what it lacks."""


class TypedState(Exception):
    """A state value typed into the spine instead of measured by a probe."""


@dataclass(frozen=True)
class PlanRow:
    id: str
    slice_key: str
    does: str
    satisfies: tuple[str, ...]
    sources: tuple[str, ...]
    depends_on: tuple[str, ...]
    probe: str | None
    accepts: str
    decided: str | None


@dataclass(frozen=True)
class PlanSlice:
    key: str
    title: str
    rows: tuple[PlanRow, ...]


def _split(value: str) -> tuple[str, ...]:
    """Split a list field on the separators the document actually uses."""
    parts = [p.strip() for p in _SEPARATORS.split(value)]
    out: list[str] = []
    for part in parts:
        if not part or part.lower() == "none":
            continue
        # `satisfies` and `depends on` are space-separated ids; `sources` are
        # phrases. Only split on spaces when every piece looks like an id.
        pieces = part.split()
        if len(pieces) > 1 and all(re.fullmatch(r"[A-Z]{2,3}-\d+", p) for p in pieces):
            out.extend(pieces)
        else:
            out.append(part)
    return tuple(out)


def _check_state(row_id: str, state: str) -> str | None:
    """Return the decided state, or None. Raise if a measurement was typed."""
    bare = state.strip()
    first = bare.split()[0].rstrip("-,:").upper() if bare else ""
    if first in DECIDED_STATES:
        return first
    if not bare.lower().startswith("measured by "):
        raise TypedState(
            f"{row_id}: state is {bare!r}; the spine may only say "
            f"'measured by <probe>', BLOCKED or DECLINED")
    named = bare[len("measured by "):].strip().upper()
    if named in _MEASURED_WORDS:
        raise TypedState(f"{row_id}: {named} is a measurement, not a probe name")
    return None


def read_master_plan(path: Path) -> list[PlanSlice]:
    """Parse the spine into slices of eight-field rows."""
    slices: list[PlanSlice] = []
    key: str | None = None
    title: str | None = None
    rows: list[PlanRow] = []
    pending: dict[str, str] = {}
    row_id: str | None = None
    seen: set[str] = set()
    last_field: str | None = None

    def flush() -> None:
        nonlocal pending, row_id, last_field
        if row_id is None:
            return
        missing = [f for f in ROW_FIELDS if f not in pending]
        if missing:
            raise MalformedRow(f"{row_id} is missing: {', '.join(missing)}")
        if row_id in seen:
            raise MalformedRow(f"{row_id} appears more than once")
        seen.add(row_id)
        probe = pending["probe"].strip()
        decided = _check_state(row_id, pending["state"])
        rows.append(PlanRow(
            id=row_id,
            slice_key=pending.get("slice", key or "").strip(),
            does=pending["does"].strip(),
            satisfies=_split(pending["satisfies"]),
            sources=_split(pending["sources"]),
            depends_on=_split(pending["depends on"]),
            probe=None if probe.lower() in {"none", "null", ""} else probe,
            accepts=pending["accepts"].strip(),
            decided=decided,
        ))
        pending, row_id, last_field = {}, None, None

    for line in Path(path).read_text().splitlines():
        heading = _SLICE.match(line)
        if heading:
            flush()
            if key is not None:
                slices.append(PlanSlice(key, title or "", tuple(rows)))
            key, title, rows = heading["key"], heading["title"], []
            continue

        identifier = _ROW_ID.match(line)
        if identifier:
            flush()
            row_id = identifier["id"]
            continue

        field = _FIELD.match(line)
        if field:
            last_field = field["name"]
            pending[last_field] = field["value"]
            continue

        # A continuation line: `accepts:` and `does:` wrap, and the wrapped
        # remainder belongs to the field above it rather than to nothing.
        if row_id and last_field and line.startswith("    ") and line.strip():
            pending[last_field] = pending[last_field] + " " + line.strip()

    flush()
    if key is not None:
        slices.append(PlanSlice(key, title or "", tuple(rows)))
    return slices
