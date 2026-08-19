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
from statuswall import ruling_conformance
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


def test_the_archive_walk_returns_a_lower_bound_rather_than_never_returning(tmp_path):
    """Measured 2026-08-19: an unbudgeted walk of the bars store did not finish in
    120 seconds, and the plan board called it once per row - so the board froze for
    43 hours while its own loop reported nothing wrong. A partial count rendered as
    `N+` is a measurement; a walk that never returns is not."""
    dataset = tmp_path / "bars"
    for i in range(600):
        part = dataset / f"hour={i // 50}" / f"symbol=S{i}"
        part.mkdir(parents=True, exist_ok=True)
        (part / "part.parquet").write_bytes(b"")

    seen, complete = ruling_conformance.count_store_fragments(dataset, budget_s=0.0)

    assert not complete, "a walk past its budget must say it did not finish"
    assert 0 < seen <= 600


def test_the_archive_is_walked_once_per_process_not_once_per_row(tmp_path):
    dataset = tmp_path / "bars"
    (dataset / "hour=1").mkdir(parents=True)
    (dataset / "hour=1" / "a.parquet").write_bytes(b"")

    first = ruling_conformance.count_store_fragments(dataset)
    (dataset / "hour=1" / "b.parquet").write_bytes(b"")
    second = ruling_conformance.count_store_fragments(dataset)

    assert first == (1, True)
    assert second == first, "a second call in the same pass must not re-walk"


def test_a_dataset_that_does_not_exist_counts_zero_and_says_the_walk_finished(tmp_path):
    assert ruling_conformance.count_store_fragments(tmp_path / "absent") == (0, True)


# --- probes repointed off a retired engine (2026-08-19) --------------------

def _segment_root(tmp_path, segments=("perp", "spot", "dated", "options"),
                  age_s=1.0, recovered=True):
    import json as _json
    import time as _time
    root = tmp_path / "segment"
    for segment in segments:
        directory = root / segment
        directory.mkdir(parents=True)
        (directory / "heartbeat.json").write_text(_json.dumps(
            {"written_at_ns": _time.time_ns() - int(age_s * 1e9),
             "segment": segment}))
        (directory / "engine.log").write_text(
            "recovered 3 open position(s) from the journal\n" if recovered
            else "poll 1: {}\n")
    return root


def test_the_paper_engine_probe_measures_the_bots_that_are_running(tmp_path):
    """It measured `plumbing-momentum`, retired under RL-025 on 2026-08-18, and
    reported DEGRADED on a heartbeat 38 hours old and never coming back. A tile
    permanently red about something switched off on purpose is one everybody
    learns to skip."""
    from statuswall.evidence import OK
    result = ruling_conformance.probe_paper_engine_running(
        state_root=_segment_root(tmp_path))

    assert result.state == OK
    assert "4/4 segment paper engines polling" in result.detail
    assert "4 journalled a position recovery" in result.detail


def test_two_bots_polling_of_four_reads_as_partial_and_names_the_others(tmp_path):
    from statuswall.evidence import PARTIAL
    root = _segment_root(tmp_path, segments=("perp", "spot"))

    result = ruling_conformance.probe_paper_engine_running(state_root=root)

    assert result.state == PARTIAL
    assert "dated: no heartbeat" in result.detail


def test_a_stopped_bot_is_not_counted_as_a_running_paper_engine(tmp_path):
    from statuswall.evidence import NOT_MEASURED
    root = _segment_root(tmp_path, age_s=4000)

    result = ruling_conformance.probe_paper_engine_running(state_root=root)

    assert result.state == NOT_MEASURED
    assert "min old" in result.detail


def test_an_unpartitioned_store_reads_as_degraded_because_every_read_walks_it_all(
        tmp_path):
    from statuswall.evidence import DEGRADED
    dataset = tmp_path / "bars"
    (dataset / "symbol=BTCUSDT").mkdir(parents=True)
    (dataset / "symbol=BTCUSDT" / "a.parquet").write_bytes(b"")

    result = ruling_conformance.probe_poll_scan_cost(
        dataset=dataset, learn_root=tmp_path / "learn")

    assert result.state == DEGRADED
    assert "not partitioned by availability hour" in result.detail


def test_the_scan_probe_never_compares_an_exact_count_to_a_lower_bound(tmp_path,
                                                                      monkeypatch):
    """First run of this probe reported a healthy store as DEGRADED at
    `4288/2000+`: the numerator was exact and the denominator was whatever the
    budgeted walk had reached."""
    from statuswall.evidence import PARTIAL
    dataset = tmp_path / "bars"
    for hour in ("2026-08-19T05", "2026-08-19T06"):
        directory = dataset / f"availability_hour={hour}" / "symbol=BTCUSDT"
        directory.mkdir(parents=True)
        (directory / "part.parquet").write_bytes(b"")
    monkeypatch.setattr(ruling_conformance, "count_store_fragments",
                        lambda *a, **k: (1, False))

    result = ruling_conformance.probe_poll_scan_cost(
        dataset=dataset, learn_root=tmp_path / "learn")

    assert result.state == PARTIAL
    assert "lower bound" in result.detail


# --- SL-05: a fraction that does not name its gap does not close it ---------
#
# SL-05 accepts "a per-segment ruling with rows in three of four slices reports
# 3/4 and is unresolved, NAMING THE MISSING SEGMENT". It reported the fraction
# and stopped there, so the board said RL-006 2/4 without ever saying which two
# bots were short - and a gap nobody names is a gap nobody closes.

def _spine_with(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """A spine holding one row per (id, slice key, satisfied ruling)."""
    body = ["# AJIT MASTER PLAN", ""]
    for slice_key in dict.fromkeys(key for _, key, _ in rows):
        body += [f"## SLICE {slice_key} — {slice_key.upper()}", ""]
        for row_id, key, satisfies in rows:
            if key != slice_key:
                continue
            body += [
                f"### {row_id}",
                f"  slice:      {key}",
                "  does:       does a thing",
                f"  satisfies:  {satisfies}",
                "  sources:    spec.md#1",
                "  depends on: none",
                "  probe:      probe_thing",
                "  accepts:    it works",
                "  state:      measured by probe_thing",
                "",
            ]
    path = tmp_path / "spine.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _register_with(tmp_path: Path, scope: str, rid: str = "RL-006") -> Path:
    import json
    path = tmp_path / "rulings.json"
    path.write_text(json.dumps({"rulings": [{
        "id": rid, "date": "2026-08-01", "verbatim": "said a thing",
        "means": "meant a thing", "probe": None, "scope": scope}]}),
        encoding="utf-8")
    return path


def test_a_per_segment_ruling_short_of_four_names_the_missing_segments(tmp_path):
    spine = _spine_with(tmp_path, [("SP-01", "spot-bot", "RL-006"),
                                   ("PE-01", "perp-bot", "RL-006")])
    result = ruling_conformance.probe_scope_arithmetic(
        spine=spine, register=_register_with(tmp_path, "per-segment"))

    assert "2/4" in result.detail
    assert "dated-bot" in result.detail and "options-bot" in result.detail
    assert "spot-bot" not in result.detail.split("short:")[-1].replace(
        "spot-bot/", ""), "a covered segment must not be listed as missing"


def test_a_per_brain_ruling_names_the_brains_it_is_missing(tmp_path):
    spine = _spine_with(tmp_path, [("SP-01", "spot-bot", "RL-006")])
    result = ruling_conformance.probe_scope_arithmetic(
        spine=spine, register=_register_with(tmp_path, "per-brain"))

    assert "3/12" in result.detail
    assert "perp-bot/BULL" in result.detail


def test_a_fully_covered_register_reads_ok_and_names_nothing_missing(tmp_path):
    # Row ids follow the spine's own pattern - `[A-Z]{2,3}-\d+`. A fixture that
    # ignores it parses into slices holding no rows, which reads as a coverage
    # failure rather than as a malformed test.
    rows = [(f"{prefix}-0{i}", key, "RL-006")
            for i, (prefix, key) in enumerate(
                (("SP", "spot-bot"), ("PE", "perp-bot"),
                 ("DA", "dated-bot"), ("OP", "options-bot")))]
    result = ruling_conformance.probe_scope_arithmetic(
        spine=_spine_with(tmp_path, rows),
        register=_register_with(tmp_path, "per-segment"))

    assert result.state == OK
    assert "4/4" in result.detail or "1/1 rulings covered" in result.detail


def test_a_shared_ruling_with_no_row_says_so_rather_than_naming_a_subject(tmp_path):
    """"One row anywhere" has no subject to name, so it carries a sentence."""
    spine = _spine_with(tmp_path, [("SP-01", "spot-bot", "RL-018")])
    result = ruling_conformance.probe_scope_arithmetic(
        spine=spine, register=_register_with(tmp_path, "shared"))

    assert "RL-006 0/1 (no row anywhere)" in result.detail
    assert "missing no row" not in result.detail
