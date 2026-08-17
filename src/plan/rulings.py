"""The rulings register - what the user settled, in the user's own words.

A ruling is not a feature and not a plan. It is a decision the build is not
free to re-derive, and the reason this module exists is that for two weeks
RL-011 existed only in a conversation transcript while the build went another
way: three independent bots with their own data, features and architecture,
ruled on 2026-08-03, measured at 2 of 22 built on 2026-08-17.

No row carries a status. Each names a probe, and the board renders what that
probe measured; a typed status would be exactly the fiction the board exists
to prevent (Rule 8). A `probe` of null is a legitimate answer - it renders NOT
MEASURED - but the FIELD must be present, because an absent field is
indistinguishable from an oversight while an explicit null is a decision.

Verbatim text is reproduced as the user typed it, including typos. A cleaned-up
paraphrase reads better and is a different sentence, and the difference between
those two sentences is where a design quietly becomes something else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The four levels a decision can bind at. Closed on purpose: an invented fifth
# scope would have no arithmetic in scope_coverage, so it would silently count
# as resolved the moment one row existed anywhere.
SCOPES = frozenset({"per-segment", "per-brain", "shared", "system"})

_REQUIRED = ("id", "date", "verbatim", "means", "probe", "scope")


class MalformedRuling(Exception):
    """A ruling row the register cannot trust, named by what is wrong with it."""


@dataclass(frozen=True)
class Ruling:
    id: str
    date: str
    session: str
    verbatim: str
    means: str
    recorded_in: tuple[str, ...]
    probe: str | None
    scope: str
    superseded_by: str | None = None
    clarified_as: str | None = None


def read_rulings(path: Path) -> list[Ruling]:
    """Load the register, refusing anything ambiguous rather than guessing."""
    document = json.loads(Path(path).read_text())
    rows = document["rulings"]

    rulings: list[Ruling] = []
    seen: set[str] = set()
    for row in rows:
        identifier = row.get("id", "<no id>")
        for field in _REQUIRED:
            if field not in row:
                raise MalformedRuling(
                    f"{identifier} is missing the {field!r} field")
        if "status" in row:
            raise MalformedRuling(
                f"{identifier} carries a 'status' field; status is measured by "
                f"a probe and never typed into the register")
        if row["scope"] not in SCOPES:
            raise MalformedRuling(
                f"{identifier} has scope {row['scope']!r}, which is not one of "
                f"{sorted(SCOPES)}")
        if identifier in seen:
            raise MalformedRuling(f"{identifier} appears more than once")
        seen.add(identifier)

        rulings.append(Ruling(
            id=row["id"],
            date=row["date"],
            session=row.get("session", ""),
            verbatim=row["verbatim"],
            means=row["means"],
            recorded_in=tuple(row.get("recorded_in", ())),
            probe=row["probe"],
            scope=row["scope"],
            superseded_by=row.get("superseded_by"),
            clarified_as=row.get("clarified_as"),
        ))
    return rulings
