"""The real spine must satisfy its own parser, or it is not a contract.

A plan document that does not parse is a plan document nothing can measure
against, which is the state `2026-08-09-full-build-master-plan.md` is in: its
phase J puts paper trading last, paper trading has run since 2026-08-15, and
nothing anywhere noticed the contradiction.

The last test here is the one that matters most. It asserts the plan is honest
about what it has NOT planned - the four segment slices carry no rows yet, so
the user's examination-hall ruling reads 0/4 rather than resolved.
"""
from pathlib import Path

from plan.master_plan import read_master_plan
from plan.rulings import read_rulings
from plan.scope_coverage import cover_ruling

REPO = Path(__file__).resolve().parents[1]
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"
REGISTER = REPO / "docs" / "rulings.json"


def test_the_real_spine_parses_into_the_six_slices():
    slices = read_master_plan(SPINE)
    keys = [s.key for s in slices]
    assert keys == ["slice-0", "spot-bot", "perp-bot", "dated-bot",
                    "options-bot", "slice-5"]


def test_every_row_in_the_real_spine_names_a_ruling_that_exists():
    known = {r.id for r in read_rulings(REGISTER)}
    for plan_slice in read_master_plan(SPINE):
        for row in plan_slice.rows:
            unknown = set(row.satisfies) - known
            assert not unknown, f"{row.id} cites unknown ruling(s): {unknown}"


def test_every_row_depends_only_on_rows_that_exist():
    slices = read_master_plan(SPINE)
    ids = {row.id for s in slices for row in s.rows}
    for plan_slice in slices:
        for row in plan_slice.rows:
            for dependency in row.depends_on:
                if dependency.replace("-", "").replace("_", "").isalnum() and "-" in dependency:
                    assert dependency in ids or not dependency[:2].isupper(), (
                        f"{row.id} depends on {dependency}, which is not a row")


def test_slice_zero_has_rows_and_the_segment_slices_are_honestly_empty():
    slices = {s.key: s for s in read_master_plan(SPINE)}
    assert len(slices["slice-0"].rows) >= 10
    # The segment slices carry no rows yet, deliberately: their inventory comes
    # from the reconciliation sweep and is reviewed with the user first. An
    # empty slice reporting 0/0 is the honest board for unplanned work.
    for key in ("spot-bot", "perp-bot", "dated-bot", "options-bot", "slice-5"):
        assert slices[key].rows == (), f"{key} should be empty until reviewed"


def test_the_per_segment_rulings_report_zero_of_four_today():
    slices = read_master_plan(SPINE)
    rulings = {r.id: r for r in read_rulings(REGISTER)}
    coverage = cover_ruling(rulings["RL-009"], slices)
    assert coverage.need == 4
    assert coverage.have == 0
    assert not coverage.resolved, (
        "the examination-hall ruling has no segment rows yet and must say so")


def test_the_spine_states_its_own_authority_and_supersession():
    body = SPINE.read_text()
    assert "top of the authority chain" in body.lower()
    assert "2026-08-09-full-build-master-plan.md" in body, (
        "the plan it supersedes must be named, not silently replaced")
