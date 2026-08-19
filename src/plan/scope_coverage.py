"""Per-scope coverage - the arithmetic that stops 'assigned' meaning 'done'.

A cross-cutting capability assigned to one slice reads as fully resolved while
three of four bots silently never receive it. That is the failure this whole
plan exists to prevent, reproduced inside the plan, and it is not
hypothetical: the user's examination-hall ruling of 2026-08-02 became roughly
140 lines of design in the goal document and zero rows of work, and a sweep
that only asked "is it assigned anywhere" would have called it covered.

So the denominator comes from the scope, not from the sweep. Coverage counts
DISTINCT slice keys, not rows - two rows for the spot bot is still one bot -
and it counts what a row CLAIMS to satisfy, never what it happens to sit near.

Coverage is not the same question as state, and the board shows both. Coverage
asks whether the plan has rows for a ruling; state asks whether the running
system does it. A ruling can be fully covered and still fail its probe. That is
not a contradiction, it is the difference between planned and true.
"""
from __future__ import annotations

from dataclasses import dataclass

from plan.master_plan import PlanSlice
from plan.rulings import Ruling

# The bot boundary is the segment (goal document §3b, RL-019). Four bots.
SEGMENTS = ("spot-bot", "perp-bot", "dated-bot", "options-bot")

# Three brains inside each (RL-011). The two axes compose rather than multiply
# into twelve separate bots: the user chose the nested reading on 2026-08-17.
DIRECTIONS = ("BULL", "BEAR", "PROFIT-TAIL")
BRAINS = tuple(f"{segment}/{direction}"
               for segment in SEGMENTS for direction in DIRECTIONS)


@dataclass(frozen=True)
class Coverage:
    subject: str
    scope: str
    have: int
    need: int
    missing: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return self.have >= self.need


def _keys_claiming(subject: str, slices: list[PlanSlice]) -> set[str]:
    """Distinct slice keys holding a row that claims to satisfy `subject`."""
    return {row.slice_key
            for plan_slice in slices
            for row in plan_slice.rows
            if subject in row.satisfies}


def _brains_of(slice_keys: set[str]) -> set[str]:
    """The brains a set of slice keys accounts for (RL-031, 2026-08-19).

    A row in a SEGMENT slice covers that segment's three brains together,
    because that is how the bots are actually built: a capability lands in a
    segment bot and BULL, BEAR and PROFIT-TAIL receive it at once. Rows do not
    name a brain, and requiring them to would make it a ninth mandatory field on
    every row that satisfies a per-brain ruling.

    **A shared or cross-cutting slice accounts for NO brain**, and that is the
    doctrine rather than an omission. A capability built once is not thereby
    delivered to four bots - letting `bot-framework` or `learned-brains` count
    for all twelve would reproduce inside the plan the exact failure the scope
    arithmetic exists to catch.

    Before this, the numerator compared slice keys against brain names directly.
    A slice key is `spot-bot`, never `spot-bot/BULL`, so the comparison could
    never match: measured 2026-08-19, all nine per-brain rulings read 0/12 and
    always would have, RL-026 and RL-030 among them. A fraction that can only be
    zero is not a measurement.
    """
    return {brain for brain in BRAINS
            if brain.split("/", 1)[0] in slice_keys}


def cover_ruling(ruling: Ruling, slices: list[PlanSlice]) -> Coverage:
    """How much of a ruling the plan's rows actually cover, per its scope."""
    claimed = _keys_claiming(ruling.id, slices)

    if ruling.scope == "per-segment":
        missing = tuple(s for s in SEGMENTS if s not in claimed)
        return Coverage(ruling.id, ruling.scope,
                        len(SEGMENTS) - len(missing), len(SEGMENTS), missing)

    if ruling.scope == "per-brain":
        covered = _brains_of(claimed)
        missing = tuple(b for b in BRAINS if b not in covered)
        return Coverage(ruling.id, ruling.scope,
                        len(BRAINS) - len(missing), len(BRAINS), missing)

    # shared and system: one row anywhere is the whole requirement.
    have = 1 if claimed else 0
    return Coverage(ruling.id, ruling.scope, have, 1,
                    () if have else ("no row anywhere",))
