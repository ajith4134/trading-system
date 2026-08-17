"""The board answers 'how much is done and what is next' without being asked.

The user asked that question six times between 2026-08-08 and 2026-08-17. The
board's job is to make it answer itself: counts per slice, the active slice, the
single next row, and the unassigned totals - all measured, no estimates.

The two properties worth defending are the ones that would flatter. An empty
slice must render 0 / 0 rather than complete, because a slice with no rows is
unplanned work and not finished work. And the unassigned count must appear on
the page, because it is the number that has been invisible for fifteen days.
"""
import re

from plan.master_plan import PlanRow, PlanSlice
from plan.reconcile_sources import Resolution, SourceMember
from statuswall.evidence import NOT_BUILT, OK, ProbeResult
from statuswall.master_plan_board import (
    active_slice, next_row, render_master_plan_page,
)


def _row(rid, slice_key="slice-0") -> PlanRow:
    return PlanRow(id=rid, slice_key=slice_key, does=f"{rid} does a thing",
                   satisfies=("RL-021",), sources=("spec §1",), depends_on=(),
                   probe="probe_x", accepts="it works", decided=None)


def _slices() -> list[PlanSlice]:
    return [PlanSlice("slice-0", "FOUNDATION", (_row("SL-01"), _row("SL-02"))),
            PlanSlice("spot-bot", "SPOT BOT", ())]


def test_the_active_slice_is_the_first_one_not_fully_built():
    results = {"SL-01": ProbeResult(OK, "d", "p"),
               "SL-02": ProbeResult(NOT_BUILT, "d", "p")}
    assert active_slice(_slices(), results) == "slice-0"


def test_the_next_row_is_the_first_unbuilt_row_of_the_active_slice():
    results = {"SL-01": ProbeResult(OK, "d", "p"),
               "SL-02": ProbeResult(NOT_BUILT, "d", "p")}
    assert next_row(_slices(), results).id == "SL-02"


def test_an_empty_slice_is_active_once_the_slice_above_it_is_built():
    """Empty is unplanned, not finished - so it becomes the active slice."""
    results = {"SL-01": ProbeResult(OK, "d", "p"), "SL-02": ProbeResult(OK, "d", "p")}
    assert active_slice(_slices(), results) == "spot-bot"
    assert next_row(_slices(), results) is None


def test_an_empty_slice_reports_zero_of_zero_rather_than_complete():
    results = {"SL-01": ProbeResult(OK, "d", "p"), "SL-02": ProbeResult(OK, "d", "p")}
    page = render_master_plan_page(_slices(), results, [], generated_at_epoch_s=1)
    assert "0 / 0" in page, "an empty slice must not render as finished"


def test_the_page_publishes_the_unassigned_count_rather_than_hiding_it():
    member = SourceMember("design-section", "x.md#Something", "Something", "x.md")
    resolutions = [Resolution(member, "unassigned", "")]
    page = render_master_plan_page(_slices(), {}, resolutions, generated_at_epoch_s=1)
    assert "UNASSIGNED" in page.upper()


def test_a_row_with_no_probe_result_renders_not_measured():
    page = render_master_plan_page(_slices(), {}, [], generated_at_epoch_s=1)
    assert "NOT MEASURED" in page.upper()


def test_the_page_carries_no_date_estimate_or_percentage_complete():
    """Measured counts only. An estimate here would be a guess wearing a number."""
    results = {"SL-01": ProbeResult(OK, "d", "p")}
    page = render_master_plan_page(_slices(), results, [], generated_at_epoch_s=1)
    lowered = page.lower()
    # Word boundaries, not substrings: "eta" lives inside "<meta charset>".
    for banned in (r"\beta\b", r"\bestimated\b", r"%\s*complete",
                   r"\bdays remaining\b", r"\bweeks?\b"):
        assert not re.search(banned, lowered), (
            f"the board must not carry {banned!r}")


def test_the_next_row_names_what_it_does_so_the_board_is_actionable():
    results = {"SL-01": ProbeResult(NOT_BUILT, "d", "p")}
    page = render_master_plan_page(_slices(), results, [], generated_at_epoch_s=1)
    assert "SL-01 does a thing" in page
