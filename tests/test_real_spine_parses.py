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


def test_the_real_spine_parses_into_the_seven_slices():
    """`bot-framework` was added 2026-08-18 between governance and the segments.

    slice-0 deliberately left the shared bot framework out, to be planned when a
    segment slice needed it. All four needed it at once - RL-024 moved every bot's
    market data path to a live feed and RL-023 fixed the brain count at three - so
    the shared harness became a slice rather than being smuggled into whichever
    segment happened to be built first.
    """
    slices = read_master_plan(SPINE)
    keys = [s.key for s in slices]
    assert keys == ["slice-0", "bot-framework", "spot-bot", "perp-bot", "dated-bot",
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


def test_only_the_reviewed_segment_slices_carry_rows():
    """A segment slice carries rows only once its work was asked for, never before.

    **The boundary MOVED on 2026-08-18 and this test moved with it, deliberately.**
    Until then only perp had rows, reviewed with the user 2026-08-17 under RL-022.
    On 2026-08-18 the user asked for all four segment bots trading live, which is
    the authorisation the other three slices were waiting for, and RL-023/RL-024/
    RL-025 settled their shape. So spot, dated and options now carry the rows that
    request implies - a live universe row and a three-brain row each.

    **What they do NOT carry is a full inventory,** and the distinction is the point.
    Each of those slices still says in its own text that its remaining rows are
    pending review. `slice-5` has none at all and stays that way: nothing has anything
    to allocate between until two bots reach BUILT, and an empty slice reporting 0/0
    is still the honest board for unplanned work.

    The test therefore guards two things rather than one: that the slices asked for
    have rows, and that the one nobody has asked for does not quietly acquire them.
    """
    slices = {s.key: s for s in read_master_plan(SPINE)}
    assert len(slices["slice-0"].rows) >= 10
    assert len(slices["perp-bot"].rows) >= 10, "the reviewed slice lost its rows"
    assert len(slices["bot-framework"].rows) >= 8, (
        "the shared live framework slice lost its rows")
    for key in ("spot-bot", "dated-bot", "options-bot"):
        rows = slices[key].rows
        assert rows, f"{key} was asked for on 2026-08-18 and must carry its rows"
        # Asked-for, not fully inventoried. A slice that quietly grew to perp's size
        # would mean an inventory happened that the user was never part of.
        assert len(rows) < 10, (
            f"{key} has {len(rows)} rows; its full inventory is still pending review "
            f"with the user and must not appear without one")
    assert slices["slice-5"].rows == (), (
        "slice-5 should be empty until two bots reach BUILT")


def test_the_per_segment_rulings_now_report_four_of_four():
    """RL-009 reached all four segment slices on 2026-08-18, and that is the point.

    This test asserted 1/4 while only perp had rows. The arithmetic it defends is
    unchanged - a per-segment ruling needs a row in every segment slice and one
    slice discharges a quarter - but the measurement moved because the other three
    slices now carry rows, so the fraction it reports moved with it.

    **Coverage is not delivery, and nothing here claims it is.** Four assigned rows
    mean the ruling is no longer unaddressed in three segments; whether each bot
    actually scans its whole universe is what `probe_*_universe_measured` measures,
    and those probes report counts with denominators for exactly this reason. The
    2026-08-02 failure was a ruling reading as resolved in slices that had never
    received it - which is what this test still refuses.
    """
    slices = read_master_plan(SPINE)
    rulings = {r.id: r for r in read_rulings(REGISTER)}
    coverage = cover_ruling(rulings["RL-009"], slices)
    assert coverage.need == 4
    assert coverage.have == 4
    assert set(coverage.missing) == set()
    # ASSIGNED in all four, which is what `resolved` means here: every scope the
    # ruling binds at has a row that cites it. It does NOT mean the universe-wide
    # scan is delivered - `probe_perp_universe_measured` and its three siblings
    # measure that, and each publishes an admitted count against its denominator.
    # The distinction is the whole reason coverage is a fraction rather than a flag.
    assert coverage.resolved, (
        "every segment slice now carries a row citing RL-009; if this fails a "
        "segment slice lost its rows")


def test_the_spine_states_its_own_authority_and_supersession():
    body = SPINE.read_text()
    assert "top of the authority chain" in body.lower()
    assert "2026-08-09-full-build-master-plan.md" in body, (
        "the plan it supersedes must be named, not silently replaced")
