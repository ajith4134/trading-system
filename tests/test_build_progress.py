"""The build-progress board only says what was measured.

The board's one failure mode worth testing for is silent under-counting: a
catalogue section no phase claims, whose features then vanish from every
phase row while the page still renders. That must refuse loudly. The rest is
counting, and the counts must reconcile exactly with the catalogue.
"""
from pathlib import Path

import pytest

from statuswall.build_progress import (
    PLAN_PHASES,
    UnmappedSection,
    count_unresolved_ledger_rows,
    probe_forward_paper,
    render_build_progress_page,
    summarise_build_progress,
)
from statuswall.catalogue import Feature, read_catalogue
from statuswall.evidence import BUILT, FAILING, NOT_BUILT, OK, ProbeResult


def _feature(name: str, section: str) -> Feature:
    return Feature(section_idx=section, section_title=f"Section {section}",
                   name=name, phase="P0", note="", missed=False)


def _results(features, state=NOT_BUILT):
    return {f.key: ProbeResult(state, "detail", "proof") for f in features}


# --- the mapping against the real catalogue -------------------------------

def test_real_catalogue_is_fully_mapped_and_counts_reconcile():
    features = read_catalogue(Path.home() / "research" / "FEATURES.md")
    phases = summarise_build_progress(features, _results(features))
    assert sum(p.total for p in phases) == len(features), (
        "every catalogue feature must land in exactly one phase")


def test_unmapped_section_refuses_instead_of_undercounting():
    features = [_feature("a real thing", "1"), _feature("a new thing", "99")]
    with pytest.raises(UnmappedSection, match="99"):
        summarise_build_progress(features, _results(features))


def test_no_section_is_claimed_twice():
    seen = set()
    for _key, _title, sections in PLAN_PHASES:
        for section in sections:
            assert section not in seen, f"section {section} mapped twice"
            seen.add(section)


# --- counting -------------------------------------------------------------

def test_lit_counts_everything_except_not_built():
    features = [_feature("one", "1"), _feature("two", "1"), _feature("three", "1")]
    results = {
        features[0].key: ProbeResult(OK, "d", "p"),
        features[1].key: ProbeResult(FAILING, "d", "p"),
        features[2].key: ProbeResult(NOT_BUILT, "d", "p"),
    }
    phase_a = summarise_build_progress(features, results)[0]
    assert phase_a.total == 3
    assert phase_a.lit == 2, "FAILING is measured, and measured means lit"


# --- the ledger probe -----------------------------------------------------

def test_unresolved_rows_counted_only_from_status_cells(tmp_path):
    (tmp_path / "slice.md").write_text(
        "prose mentioning UNRESOLVED does not count\n"
        "| Requirement | Status | Notes |\n"
        "|---|---|---|\n"
        "| thing one | UNRESOLVED | note |\n"
        "| thing two | PLANNED | note |\n"
        "| thing three | **UNRESOLVED** | bold status still counts |\n"
        "| thing four | UNRESOLVED (per original audit) | suffixed status counts |\n"
        "\n"
        "| UNRESOLVED | 37 |\n",
        encoding="utf-8")
    assert count_unresolved_ledger_rows(tmp_path) == 3, (
        "the two-column totals table must not be counted as a requirement row - "
        "that over-count is exactly how the first draft reported 109 against "
        "the index's 108")


def test_missing_ledger_reports_none_not_zero(tmp_path):
    assert count_unresolved_ledger_rows(tmp_path / "absent") is None, (
        "an unmeasured ledger must never render as a clean one")


# --- the paper probe ------------------------------------------------------

def test_paper_not_running_when_no_journal(tmp_path):
    running, evidence = probe_forward_paper(tmp_path)
    assert not running
    assert "no journal directory" in evidence


def _journal_with_heartbeat(tmp_path, written_at_ns, **overrides):
    from paper.forward_journal import ForwardJournal
    journal_dir = tmp_path / "paper" / "forward"
    journal_dir.mkdir(parents=True)
    fields = dict(strategy="plumbing-momentum", makes_edge_claim=False,
                  events_fed=12, orders_submitted=3, orders_rejected=0, fills=1,
                  open_orders=2, last_event_time_ns=99, detail="")
    fields.update(overrides)
    ForwardJournal(journal_dir).record_heartbeat(now_ns=written_at_ns, **fields)
    return journal_dir


def test_a_supervisor_restart_log_is_not_a_journal(tmp_path):
    """The regression this probe was rewritten for. It used to glob *.ndjson and
    would report RUNNING off `restarts.ndjson` — the supervisor's own log,
    written once at startup and never again. A tile that goes green because a
    process started once, and stays green after it dies, is the exact Rule 8
    failure this board exists to prevent."""
    journal_dir = tmp_path / "paper" / "forward"
    journal_dir.mkdir(parents=True)
    (journal_dir / "restarts.ndjson").write_text(
        '{"event": "supervisor_started"}\n', encoding="utf-8")
    running, evidence = probe_forward_paper(tmp_path)
    assert not running
    assert "heartbeat" in evidence


def test_paper_running_when_the_heartbeat_is_fresh(tmp_path):
    now = 1_000_000_000_000_000_000
    _journal_with_heartbeat(tmp_path, now - 10_000_000_000)
    running, evidence = probe_forward_paper(tmp_path, now_ns=now)
    assert running
    assert "plumbing-momentum" in evidence


def test_a_running_engine_that_claims_no_edge_says_so_on_the_tile(tmp_path):
    now = 1_000_000_000_000_000_000
    _journal_with_heartbeat(tmp_path, now - 10_000_000_000)
    _, evidence = probe_forward_paper(tmp_path, now_ns=now)
    assert "NO edge claim" in evidence


def test_an_engine_that_died_reads_stale_not_running(tmp_path):
    """The failure that motivated all of this: the box was off 2026-08-10 to
    2026-08-15 and the board went on looking healthy the whole time."""
    now = 1_000_000_000_000_000_000
    _journal_with_heartbeat(tmp_path, now - 5 * 86_400 * 1_000_000_000)
    running, evidence = probe_forward_paper(tmp_path, now_ns=now)
    assert not running
    assert "STALE" in evidence and "120.0h" in evidence


# --- the page -------------------------------------------------------------

def test_page_renders_dark_phases_and_not_running_paper():
    features = [_feature("one", "1")]
    page = render_build_progress_page(
        phases=summarise_build_progress(features, _results(features)),
        unresolved_rows=108,
        paper_running=False,
        paper_evidence="no journal directory at /x",
        generated_at="2026-08-09 15:00 UTC")
    assert "NOT RUNNING" in page, "a failing state must be reachable (Rule 8)"
    assert "108 UNRESOLVED" in page
    assert "0 / 1 lit" in page


def test_page_never_claims_an_unmeasured_ledger():
    page = render_build_progress_page(
        phases=[], unresolved_rows=None, paper_running=False,
        paper_evidence="no journal directory at /x",
        generated_at="2026-08-09 15:00 UTC")
    assert "NOT MEASURED" in page
