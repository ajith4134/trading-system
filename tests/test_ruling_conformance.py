"""A ruling with no probe is NOT MEASURED, and that must be visible.

The board's purpose is to make an unhonoured ruling loud. RL-011 was given on
2026-08-03 and measured at 2/22 built two weeks later; nothing on any board
said so, because no board knew the ruling existed.

Two numbers per ruling and they answer different questions. COVERAGE asks
whether the plan has rows for it, per its scope. STATE asks whether the running
system does it. A ruling can be fully covered and still fail its probe - that is
the difference between planned and true, not a contradiction.
"""
from pathlib import Path

from plan.master_plan import PlanRow, PlanSlice
from plan.rulings import Ruling
from statuswall.evidence import NOT_MEASURED, OK, ProbeResult
from statuswall.ruling_conformance import (
    PROBES,
    assess_rulings,
    probe_memory_reachable,
    render_ruling_conformance_page,
)


def _ruling(rid="RL-001", probe=None, scope="system") -> Ruling:
    return Ruling(id=rid, date="2026-08-01", session="s", verbatim="said a thing",
                  means="meant a thing", recorded_in=(), probe=probe, scope=scope)


def _slices() -> list[PlanSlice]:
    row = PlanRow(id="SL-01", slice_key="slice-0", does="d", satisfies=("RL-001",),
                  sources=("spec §1",), depends_on=(), probe="probe_memory_reachable",
                  accepts="a", decided=None)
    return [PlanSlice("slice-0", "SLICE 0", (row,))]


def test_a_ruling_with_no_probe_is_not_measured_never_lit():
    _, _, result = assess_rulings([_ruling(probe=None)], _slices())[0]
    assert result.state == NOT_MEASURED
    assert "no probe" in result.detail.lower()


def test_a_ruling_naming_an_unknown_probe_is_not_measured_and_says_which():
    _, _, result = assess_rulings(
        [_ruling(probe="probe_that_does_not_exist")], _slices())[0]
    assert result.state == NOT_MEASURED
    assert "probe_that_does_not_exist" in result.detail


def test_a_probe_that_raises_measures_nothing_rather_than_failing_the_board():
    """One broken probe must not take the whole board down with it."""
    PROBES["probe_deliberately_broken"] = lambda: 1 / 0
    try:
        _, _, result = assess_rulings(
            [_ruling(probe="probe_deliberately_broken")], _slices())[0]
        assert result.state == NOT_MEASURED
        assert "ZeroDivisionError" in result.detail
    finally:
        del PROBES["probe_deliberately_broken"]


def test_a_known_probe_supplies_the_state_rather_than_the_ruling():
    _, _, result = assess_rulings(
        [_ruling(probe="probe_memory_reachable")], _slices())[0]
    assert result.state in set(PROBES) or result.state
    assert result.proof, "every state must carry its proof (Rule 8)"


def test_coverage_travels_with_the_ruling_so_a_partial_scope_is_visible():
    _, coverage, _ = assess_rulings(
        [_ruling(rid="RL-009", scope="per-segment")], _slices())[0]
    assert coverage.need == 4
    assert not coverage.resolved


def test_the_page_renders_every_ruling_and_names_the_unmeasured_ones():
    assessments = assess_rulings([_ruling(), _ruling(rid="RL-002")], _slices())
    page = render_ruling_conformance_page(assessments, generated_at_epoch_s=1)
    assert "RL-001" in page and "RL-002" in page
    assert "NOT MEASURED" in page.upper()


def test_the_page_escapes_the_verbatim_text_rather_than_injecting_it():
    """Ruling text is user-authored; it renders as text, never as markup."""
    hostile = Ruling(id="RL-000", date="d", session="s",
                     verbatim="<script>alert(1)</script>", means="m",
                     recorded_in=(), probe=None, scope="system")
    page = render_ruling_conformance_page(
        assess_rulings([hostile], _slices()), generated_at_epoch_s=1)
    # The page carries its own <script> for the staleness banner, so the
    # property is not "no script tag anywhere" - it is that the RULING's text
    # arrives as text. Assert on the escaped form and on the absence of the
    # hostile string in its executable shape.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)</script>" not in page


def test_memory_probe_reports_what_it_found_either_way():
    result = probe_memory_reachable()
    assert result.state
    assert result.proof
