"""The wall's job is to be true, so these tests attack its truthfulness.

The failure worth preventing is not a crash - it is a board that renders a
feature as fine when nothing measured it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from statuswall.catalogue import Feature, normalise_key, read_catalogue
from statuswall.evidence import (
    FAILING, NOT_BUILT, OK, PROBES, STOPPED, ProbeResult, SystemFacts,
    UnknownProbeTarget, assess, measure_system, verify_probe_coverage,
)
from statuswall.wall_page import STATE_STYLE, WallInput, render_wall

NOW = dt.datetime(2026, 8, 3, 12, 0, tzinfo=dt.timezone.utc)


def _write_catalogue(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "FEATURES.md"
    path.write_text(body, encoding="utf-8")
    return path


def _facts(**overrides) -> SystemFacts:
    base = dict(
        measured_at="2026-08-03T12:00:00Z",
        capture_root=Path("/nonexistent"),
        repo_root=Path("/nonexistent"),
        capture_running=False,
        capture_pids=[],
        latest_capture_date=None,
        hours_since_capture=None,
        venues=[],
        reports={},
        free_bytes=0,
        daily_bytes=0.0,
        runway_days=0.0,
        runway_status="ok",
        restart_counts={},
    )
    base.update(overrides)
    return SystemFacts(**base)


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------

def test_zomma_table_is_not_mistaken_for_features(tmp_path):
    """FEATURES.md carries a worked Zomma example whose columns are numbers.

    Reading every table indiscriminately turns "0.636" into a phase and a Greek
    into a feature, which then occupies the wall as a permanently unmeasurable tile.
    """
    body = (
        "## 1. Market data\n\n"
        "| Feature | Phase | Notes |\n|---|---|---|\n"
        "| Trade tape | P0 | real |\n\n"
        "| Implied vol | Delta | Zomma |\n|---|---|---|\n"
        "| 15.9% | 0.636 | -0.749 |\n"
    )
    features = read_catalogue(_write_catalogue(tmp_path, body))
    assert [f.name for f in features] == ["Trade tape"]


def test_strategy_family_tables_are_read(tmp_path):
    """Section 4 names its columns Family/Phase/Verdict and is still features."""
    body = (
        "## 4. Strategy families\n\n"
        "| Family | Phase | Verdict |\n|---|---|---|\n"
        "| Funding-rate carry | P1 | Start here |\n"
    )
    features = read_catalogue(_write_catalogue(tmp_path, body))
    assert [f.name for f in features] == ["Funding-rate carry"]
    assert features[0].section_idx == "4"


def test_missed_marker_is_extracted_not_left_in_the_note(tmp_path):
    body = (
        "## 6. Risk\n\n"
        "| Feature | Phase | Notes |\n|---|---|---|\n"
        "| Liquidation-distance monitor | P0 | **[MISSED]** — survival metric |\n"
    )
    feature = read_catalogue(_write_catalogue(tmp_path, body))[0]
    assert feature.missed is True
    assert "MISSED" not in feature.note
    assert feature.note == "survival metric"


def test_key_ignores_markdown_emphasis(tmp_path):
    """Bolding a row in FEATURES.md must not detach it from its probe."""
    body = (
        "## 3. Models\n\n"
        "| Feature | Phase | Notes |\n|---|---|---|\n"
        "| **Linear / naive baseline** | P0 | x |\n"
    )
    feature = read_catalogue(_write_catalogue(tmp_path, body))[0]
    assert feature.key == normalise_key("Linear / naive baseline")


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def _feature(name: str) -> Feature:
    return Feature(section_idx="1", section_title="S", name=name, phase="P0",
                   note="", missed=False)


def test_unprobed_features_are_not_built_never_ok():
    results = assess([_feature("Something nobody built")], _facts())
    only = next(iter(results.values()))
    assert only.state == NOT_BUILT
    assert only.state != OK


def test_orphaned_probe_raises_rather_than_silently_dropping():
    """A probe whose feature was reworded must fail loudly.

    Ignoring it means the wall quietly stops reporting a real measurement, and
    nothing on the board looks wrong - the exact failure Rule 8 exists to stop.
    """
    with pytest.raises(UnknownProbeTarget):
        verify_probe_coverage([_feature("A feature no probe targets")])


def test_every_probe_target_exists_in_the_real_catalogue():
    """Guards the shipped map against drift in the actual FEATURES.md."""
    path = Path.home() / "research" / "FEATURES.md"
    if not path.is_file():
        pytest.skip("catalogue not present on this machine")
    keys = {f.key for f in read_catalogue(path)}
    assert set(PROBES) <= keys


def test_subscribed_but_silent_stream_reads_failing_not_absent():
    """A wired, dead stream is a failure; an unsubscribed one is merely unbuilt."""
    facts = _facts(
        venues=["binance"],
        reports={"binance": {"silent_stream_names": ["forceOrder"], "raw_bytes_by_stream": {}}},
    )
    result = PROBES["liquidation feed"](facts)
    assert result.state == FAILING
    assert "never delivered" in result.detail


def test_capture_not_running_reads_stopped_not_ok():
    facts = _facts(
        capture_running=False,
        latest_capture_date="2026-08-02",
        hours_since_capture=12.0,
        venues=["binance"],
        reports={"binance": {"raw_bytes_by_stream": {"depth_BTCUSDT": 10}, "silent_stream_names": []}},
    )
    assert PROBES["l2 order book depth 20 50 levels"](facts).state == STOPPED


def test_measure_system_on_an_empty_machine_reports_nothing_rather_than_healthy(tmp_path):
    facts = measure_system(capture_root=tmp_path, repo_root=tmp_path, now=NOW)
    assert facts.venues == []
    assert facts.latest_capture_date is None
    assert facts.hours_since_capture is None


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_every_state_has_a_style_so_no_state_can_render_unstyled():
    from statuswall.evidence import SEVERITY_ORDER
    assert set(SEVERITY_ORDER) == set(STATE_STYLE)


def test_page_renders_a_failure_visibly():
    """A board with no way to show red has not been tested against a failure."""
    features = [_feature("Liquidation feed")]
    results = {features[0].key: ProbeResult(FAILING, "wired and dead", "capture_health")}
    page = render_wall(WallInput(features, results, _facts()))
    assert "FAILING" in page
    assert "wired and dead" in page
    assert "Needs attention" in page


def test_feature_names_are_escaped_into_the_page():
    features = [_feature("Depth <20 levels> & more")]
    results = {features[0].key: ProbeResult(NOT_BUILT, "x", "y")}
    page = render_wall(WallInput(features, results, _facts()))
    assert "&lt;20 levels&gt; &amp; more" in page
    assert "<20 levels>" not in page
