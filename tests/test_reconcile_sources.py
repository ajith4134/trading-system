"""Three populations, one sweep, and nothing gets a fourth option.

The design-sections population is the one that matters. The ledger has
reconciled ROWS since 2026-08-08 and it works; what nothing reconciled was
argued-through prose that never became a row. Measured 2026-08-17: 'Goodhart
defence', 'attention scarcity' and the examination-hall framing appear in
four, two and three design files and in ZERO ledger rows.

'prose, no work implied' is a legitimate answer for a section like the prime
directive - but it has to be WRITTEN, because inferring it silently is exactly
how §5a became zero rows of work.
"""
from pathlib import Path

import pytest

from plan.master_plan import PlanRow, PlanSlice
from plan.reconcile_sources import (
    OUTCOMES,
    Resolution,
    SourceMember,
    read_catalogue_rows,
    read_design_sections,
    read_ledger_rows,
    reconcile,
    summarise,
)


def _member(identifier: str, population: str = "design-section") -> SourceMember:
    return SourceMember(population=population, identifier=identifier,
                        title=identifier, origin="test.md")


def _slice_with(sources: tuple[str, ...]) -> list[PlanSlice]:
    row = PlanRow(id="SL-01", slice_key="slice-0", does="a thing",
                  satisfies=("RL-021",), sources=sources, depends_on=(),
                  probe="probe_thing", accepts="it works", decided=None)
    return [PlanSlice("slice-0", "SLICE 0", (row,))]


# --- classification -------------------------------------------------------

def test_a_member_named_by_a_plan_row_is_assigned():
    resolutions = reconcile([_member("goal.md#5a")], _slice_with(("goal.md#5a",)), {})
    assert resolutions[0].outcome == "assigned"


def test_a_member_no_row_names_and_no_decision_covers_is_unassigned():
    resolutions = reconcile([_member("goal.md#5a")], _slice_with(("other",)), {})
    assert resolutions[0].outcome == "unassigned"


def test_prose_is_a_written_decision_and_never_inferred():
    member = _member("goal.md#0. Prime directive")
    inferred = reconcile([member], _slice_with(("other",)), {})
    assert inferred[0].outcome == "unassigned", (
        "a section with no row must not quietly become prose")

    written = reconcile([member], _slice_with(("other",)),
                        {member.identifier: ("prose", "a directive implies no module")})
    assert written[0].outcome == "prose"
    assert "directive" in written[0].detail


def test_declined_and_blocked_carry_their_reason():
    member = _member("on-chain DEX data")
    resolutions = reconcile([member], _slice_with(()),
                            {member.identifier: ("blocked", "no DEX venue is decided")})
    assert resolutions[0].outcome == "blocked"
    assert resolutions[0].detail == "no DEX venue is decided"


def test_an_unknown_outcome_is_refused():
    member = _member("something")
    with pytest.raises(ValueError, match="invented"):
        reconcile([member], _slice_with(()), {member.identifier: ("invented", "no")})


def test_a_written_decision_may_not_claim_assigned_which_is_measured():
    member = _member("something")
    with pytest.raises(ValueError, match="invented"):
        reconcile([member], _slice_with(()), {member.identifier: ("assigned", "no")})


def test_every_outcome_used_is_from_the_closed_set():
    assert OUTCOMES == {"assigned", "declined", "blocked", "prose", "unassigned"}


# --- matching -------------------------------------------------------------

def test_an_id_matches_on_a_word_boundary_not_a_prefix():
    """FE-01 must not be satisfied by a row citing FE-014."""
    resolutions = reconcile([_member("FE-01", population="ledger-row")],
                            _slice_with(("ledger FE-014",)), {})
    assert resolutions[0].outcome == "unassigned"


def test_an_id_inside_a_longer_sources_phrase_still_matches():
    resolutions = reconcile([_member("FE-014", population="ledger-row")],
                            _slice_with(("FEATURES §2.4 · ledger FE-014, FE-021",)), {})
    assert resolutions[0].outcome == "assigned"


# --- counting -------------------------------------------------------------

def test_summarise_counts_per_population_and_totals_match_input():
    members = [_member("a"), _member("b"), _member("c", population="ledger-row")]
    resolutions = reconcile(members, _slice_with(("a",)), {})
    counts = summarise(resolutions)
    assert counts["design-section"]["assigned"] == 1
    assert counts["design-section"]["unassigned"] == 1
    assert counts["ledger-row"]["unassigned"] == 1
    assert sum(sum(v.values()) for v in counts.values()) == len(members)


def test_every_member_is_resolved_exactly_once():
    members = [_member(f"m{i}") for i in range(25)]
    resolutions = reconcile(members, _slice_with(()), {})
    assert len(resolutions) == len(members)
    assert len({r.member.identifier for r in resolutions}) == len(members)


# --- the real populations, read from disk ---------------------------------

def test_the_real_populations_are_the_sizes_the_spec_measured():
    ledger = read_ledger_rows(Path.home() / "research" / "ledger" / "merged")
    catalogue = read_catalogue_rows(Path.home() / "research" / "FEATURES.md")
    sections = read_design_sections([
        Path.home() / "research",
        Path.home() / "trading-system" / "docs" / "superpowers",
    ])
    assert len(ledger) > 1000, f"ledger read {len(ledger)} rows, expected >1000"
    assert len(catalogue) > 120, f"catalogue read {len(catalogue)} rows, expected >120"
    assert len(sections) > 700, f"sections read {len(sections)}, expected >700"


def test_design_sections_carry_the_file_they_came_from():
    sections = read_design_sections([Path.home() / "research"])
    assert all(s.origin.endswith(".md") for s in sections)
    assert all(s.population == "design-section" for s in sections)


def test_the_examination_hall_section_is_in_the_swept_population():
    """The user's own example must be a member, or the sweep cannot find it."""
    sections = read_design_sections([
        Path.home() / "trading-system" / "docs" / "superpowers"])
    titles = " ".join(s.title.lower() for s in sections)
    assert "universe-wide scanning" in titles, (
        "goal §5a must appear as a design section the sweep can resolve")
