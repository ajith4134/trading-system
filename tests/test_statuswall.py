"""The wall's job is to be true, so these tests attack its truthfulness.

The failure worth preventing is not a crash - it is a board that renders a
feature as fine when nothing measured it.
"""
from __future__ import annotations

import datetime as dt
import time
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


def test_the_trade_tape_tile_does_not_assert_bar_building_is_missing():
    """Rule 8: a tile shows measured state, never asserted state.

    The trade-tape detail carried a hand-typed "OHLCV bar building not
    implemented" long after the store held bars and the adjacent tile measured
    them. An asserted clause cannot go stale loudly - it simply keeps reading as
    true, which is the exact failure a measured board exists to prevent.
    """
    facts = _facts(
        capture_running=True,
        venues=["binance"],
        reports={"binance": {"raw_bytes_by_stream": {"trade_BTCUSDT": 10},
                             "silent_stream_names": []}},
    )
    detail = PROBES["spot ohlcv trade tape multi venue"](facts).detail
    assert "not implemented" not in detail, detail


def test_one_raising_probe_fails_its_own_tile_and_leaves_the_board_standing():
    """A board that renders nothing is useless exactly when something is wrong.

    `assess` called every probe unguarded, so one probe raising - a rotted
    Parquet part reaching `probe_bitemporal_store`, say - took down the whole
    wall and no feature rendered at all. The degraded outcome is one tile
    reading FAILING and naming the error; the tile must not read OK, and the
    error must not be swallowed into a vague message that hides a programming
    bug.
    """
    from statuswall.evidence import FAILING, PROBES as REAL_PROBES

    exploding = _feature("Exploding feature")
    healthy = _feature("Strategy health board")

    def raise_on_probe(facts):
        raise ZeroDivisionError("a Parquet part rotted under the probe")

    patched = dict(REAL_PROBES)
    patched[exploding.key] = raise_on_probe
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("statuswall.evidence.PROBES", patched)
        results = assess([exploding, healthy], _facts())

    assert set(results) == {exploding.key, healthy.key}, "the board lost a tile"
    assert results[healthy.key].state != FAILING, "an unrelated tile was damaged"
    broken = results[exploding.key]
    assert broken.state == FAILING
    assert "ZeroDivisionError" in broken.detail
    assert "a Parquet part rotted under the probe" in broken.detail
    assert "raise_on_probe" in broken.proof


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


def test_store_probe_reports_not_built_when_no_store_exists(tmp_path):
    """Before the first build there is no store, and the wall must say so."""
    from statuswall.evidence import NOT_BUILT, probe_bitemporal_store
    facts = _facts(capture_root=tmp_path)
    assert probe_bitemporal_store(facts).state == NOT_BUILT


def test_store_probe_reports_ok_once_bars_are_readable(tmp_path):
    """The state must come from reading the store, not from the module existing."""
    import pandas as pd
    from statuswall.evidence import OK, probe_bitemporal_store
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050], AVAILABILITY_TIME: [1_100], "close": [63113.2],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, "snap1")

    result = probe_bitemporal_store(_facts(capture_root=tmp_path))
    assert result.state == OK
    # The row count explicitly, not `"1" in detail`: the detail reads "1 rows
    # across 1 append-only part(s) in 1 dataset(s)", so a bare digit search is
    # answered by the dataset count and a probe reporting ZERO rows passes it.
    assert result.detail.startswith("1 rows "), result.detail


def test_clock_gate_probe_reports_not_built_when_no_store_exists(tmp_path):
    """No store means nothing to gate, so the wall must not claim a gate exists."""
    from statuswall.evidence import NOT_BUILT, probe_clock_gated_access
    facts = _facts(capture_root=tmp_path)
    assert probe_clock_gated_access(facts).state == NOT_BUILT


def test_clock_gate_probe_reports_ok_when_gate_hides_rows_before_availability(tmp_path):
    """The gate must be exercised live: read one ns before the earliest
    availability time and confirm nothing comes back."""
    import pandas as pd
    from statuswall.evidence import OK, probe_clock_gated_access
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050], AVAILABILITY_TIME: [1_100], "close": [63113.2],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, "snap1")

    result = probe_clock_gated_access(_facts(capture_root=tmp_path))
    assert result.state == OK
    assert "1100" in result.detail


def test_clock_gate_probe_reports_failing_when_a_row_leaks_before_availability(tmp_path, monkeypatch):
    """A gate that lets a row through before its availability time is a FAILING
    core-guarantee break, not a degraded metric - proven by deliberately
    inverting the gate rather than by asserting on the healthy path alone."""
    import pandas as pd
    from statuswall.evidence import FAILING, probe_clock_gated_access
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050], AVAILABILITY_TIME: [1_100], "close": [63113.2],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, "snap1")

    class LeakyReader:
        """Stands in for a ClockGatedReader whose gate has been inverted."""

        def __init__(self, store_root, dataset):
            self._store_root = store_root
            self._dataset = dataset

        def read_as_of(self, sim_clock_ns, symbols=None):
            from store.parquet_partition import read_dataset
            return read_dataset(self._store_root, self._dataset)

    import store.clock_gated_reader as cgr
    monkeypatch.setattr(cgr, "ClockGatedReader", LeakyReader)

    result = probe_clock_gated_access(_facts(capture_root=tmp_path))
    assert result.state == FAILING
    assert "visible before their availability time" in result.detail


def test_clock_gate_probe_stays_ok_when_the_store_holds_a_correction(tmp_path):
    """A corrected bar must not read as a leak.

    Both versions of the same key are stored: the original available at the bar
    boundary, the correction available when the fix ran. `read_as_of` serves only
    the correction, so the corrected view's earliest availability is the FIX time
    - and reading one ns before that legitimately returns the original. Deriving
    the earliest from the corrected view turned that into FAILING on the wall's
    core-guarantee tile for a gate that was working.
    """
    import pandas as pd
    from statuswall.evidence import OK, probe_clock_gated_access
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    def _row(availability, low):
        return pd.DataFrame({
            SYMBOL: ["BTCUSDT"], VENUE: ["binance"], EVENT_TIME: [1_000],
            INGESTION_TIME: [1_050], AVAILABILITY_TIME: [availability],
            "low": [low], "close": [63113.2],
        }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64",
                   AVAILABILITY_TIME: "int64"})

    append_partition(tmp_path / "store", "bars_60000000000ns", _row(1_100, 0.0), "snap1")
    append_partition(tmp_path / "store", "bars_60000000000ns", _row(9_000, 63113.1), "fix1")

    result = probe_clock_gated_access(_facts(capture_root=tmp_path))
    assert result.state == OK, result.detail
    # 1100, the original's availability - not 9000, the correction's.
    assert "nothing visible before 1100" in result.detail


def test_clock_gate_probe_fails_a_gate_that_serves_rows_ahead_of_the_clock(tmp_path, monkeypatch):
    """The negative check alone passes against a gate that hides everything until
    some late cutoff and then serves the lot. This is the invariant itself: a read
    at T may not contain a row that became available after T."""
    import pandas as pd
    from statuswall.evidence import FAILING, probe_clock_gated_access
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: ["BTCUSDT"] * 3, VENUE: ["binance"] * 3,
        EVENT_TIME: [1_000, 2_000, 3_000], INGESTION_TIME: [1_050, 2_050, 3_050],
        AVAILABILITY_TIME: [1_100, 2_100, 3_100], "close": [1.0, 2.0, 3.0],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, "snap1")

    class LateOpeningGate:
        """Hides everything below the earliest availability, then serves the lot."""

        def __init__(self, store_root, dataset):
            self._store_root, self._dataset = store_root, dataset

        def read_as_of(self, sim_clock_ns, symbols=None):
            from store.parquet_partition import read_dataset
            everything = read_dataset(self._store_root, self._dataset)
            if int(sim_clock_ns) < int(everything[AVAILABILITY_TIME].min()):
                return everything.iloc[0:0]
            return everything

    import store.clock_gated_reader as cgr
    monkeypatch.setattr(cgr, "ClockGatedReader", LateOpeningGate)

    result = probe_clock_gated_access(_facts(capture_root=tmp_path))
    assert result.state == FAILING
    assert "carry a later availability time" in result.detail


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


def test_a_polled_feed_answers_its_tile_while_the_withheld_stream_stays_silent():
    """Rule 8, in the case that made it concrete.

    Binance withheld `markPrice` from this host (measured 2026-08-03) and the
    feed was recovered by polling `premiumIndex`. Both names satisfy the same
    feature, so the tile has to read from the route that is working - while
    still refusing to count a route that is merely leaving old bytes behind.

    The assertion is on the capture half rather than on the whole tile because
    that tile gained a second half on 2026-08-10: the catalogue row names
    reconciliation as well as capture, and this fixture has no store to
    reconcile from, so the composed result is correctly PARTIAL. What is being
    defended here is unchanged - the polled route answers where the withheld
    stream is silent, and a silent `markPrice` does not fail the feature.
    """
    facts = _facts(
        capture_running=True,
        venues=["binance"],
        reports={"binance": {"silent_stream_names": ["markPrice"],
                             "raw_bytes_by_stream": {"premiumIndex_BTCUSDT": 4096}}},
    )
    result = PROBES["mark price vs index vs oracle price per venue"](facts)
    assert result.state not in (FAILING, NOT_BUILT), result.detail
    assert "streaming" in result.detail


def test_a_dead_stream_that_left_bytes_behind_still_reads_failing():
    """Bytes on disk are not liveness: a stream that filled a file and then died
    leaves exactly what a running one leaves. Judged per (venue, stream), so
    this must not be rescued by the sibling route being healthy elsewhere."""
    facts = _facts(
        capture_running=True,
        venues=["binance"],
        reports={"binance": {"silent_stream_names": ["premiumIndex"],
                             "raw_bytes_by_stream": {"premiumIndex_BTCUSDT": 4096}}},
    )
    result = PROBES["mark price vs index vs oracle price per venue"](facts)
    assert result.state == FAILING, result.detail


# --- the cost engine tile ----------------------------------------------------

def _facts_with_store(tmp_path, datasets):
    from statuswall.evidence import SystemFacts
    for name in datasets:
        (tmp_path / "store" / name).mkdir(parents=True, exist_ok=True)
    return SystemFacts(measured_at="2026-08-08T00:00:00Z", capture_root=tmp_path,
                       repo_root=tmp_path, capture_running=True, capture_pids=[1],
                       latest_capture_date="2026-08-08", hours_since_capture=0.0,
                       venues=["binance"], reports={},
                       free_bytes=10**11, daily_bytes=10**9, runway_days=100.0,
                       runway_status="ok", restart_counts={})


def test_the_cost_tile_never_reads_ok_while_a_fee_is_only_declared(tmp_path):
    """Rule 8 on the one gate every signal passes. Binance will not serve its
    schedule without an API key, so its rates are a human's reading of a fee
    page - and ARCHITECTURE.md puts fees at 5-10x slippage in deciding
    breakeven. Green here would be the display asserting what nobody measured."""
    from statuswall.evidence import OK, probe_cost_engine

    result = probe_cost_engine(_facts_with_store(tmp_path, ["funding", "book"]))
    assert result.state != OK
    assert "declared" in result.detail


def test_a_missing_dataset_is_named_not_hidden(tmp_path):
    """spread and impact charge zero without a book dataset, which understates
    cost - the dangerous direction. A tile that stayed quiet about it would be
    reporting a cheaper world than the real one."""
    from statuswall.evidence import probe_cost_engine

    result = probe_cost_engine(_facts_with_store(tmp_path, ["funding"]))
    assert "book" in result.detail
    assert "zero" in result.detail


def test_no_datasets_at_all_reads_not_built(tmp_path):
    from statuswall.evidence import NOT_BUILT, probe_cost_engine

    assert probe_cost_engine(_facts_with_store(tmp_path, [])).state == NOT_BUILT


def _write_receipt(tmp_path, venues, age_hours=0.0):
    import json, time
    d = tmp_path / "fee-verification"; d.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    (d / "latest.json").write_text(json.dumps({
        "measured_at_ns": now - int(age_hours * 3_600 * 1e9),
        "venues": {v: {"verified": ok, "maker_bps": "2", "taker_bps": "5",
                       "tier": "account", "source": "venue_api",
                       "fetched_at_ns": now - int(age_hours * 3_600 * 1e9)}
                   for v, ok in venues.items()}}), encoding="utf-8")


def test_the_tile_reads_ok_once_a_fetch_is_recorded_and_fresh(tmp_path):
    """The tile was stale in the pessimistic direction: it measured the static
    declared table, so it kept saying "declared rather than fetched" after a
    live signed fetch had already succeeded. Wrong safely is still wrong."""
    from statuswall.evidence import OK, probe_cost_engine

    _write_receipt(tmp_path, {"binance:perp": True, "hyperliquid:perp": True})
    r = probe_cost_engine(_facts_with_store(tmp_path, ["funding", "book"]))
    assert r.state == OK, r.detail
    assert "binance" in r.detail


def test_an_old_verification_does_not_still_read_as_verified(tmp_path):
    """Fee tiers move with 30-day volume. A fetch from days ago is a historical
    fact, not a current one - so it caps the tile rather than passing it."""
    from statuswall.evidence import OK, probe_cost_engine

    _write_receipt(tmp_path, {"binance:perp": True}, age_hours=72)
    r = probe_cost_engine(_facts_with_store(tmp_path, ["funding", "book"]))
    assert r.state != OK
    assert "stale" in r.detail.lower() or "old" in r.detail.lower()


def test_a_refused_venue_is_named_in_the_tile(tmp_path):
    from statuswall.evidence import OK, probe_cost_engine

    _write_receipt(tmp_path, {"binance:perp": False, "hyperliquid:perp": True})
    r = probe_cost_engine(_facts_with_store(tmp_path, ["funding", "book"]))
    assert r.state != OK
    assert "binance" in r.detail


def test_no_receipt_at_all_still_reports_declared(tmp_path):
    """Absence of a verification is its own state, not a pass."""
    from statuswall.evidence import OK, probe_cost_engine

    r = probe_cost_engine(_facts_with_store(tmp_path, ["funding", "book"]))
    assert r.state != OK
    assert "never" in r.detail.lower() or "declared" in r.detail.lower()


def _bars_partition(tmp_path, snapshot: str, symbol: str = "BTCUSDT",
                    low: float = 63000.0, available_at: int = 1_100):
    """One readable bar partition, so the probe has something real to read."""
    import pandas as pd
    from store.parquet_partition import append_partition
    from store.temporal_schema import (
        AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE)

    frame = pd.DataFrame({
        SYMBOL: [symbol], VENUE: ["binance"], EVENT_TIME: [1_000],
        INGESTION_TIME: [1_050], AVAILABILITY_TIME: [available_at],
        "open": [63100.0], "high": [63200.0], "low": [low], "close": [63113.2],
    }).astype({EVENT_TIME: "int64", INGESTION_TIME: "int64", AVAILABILITY_TIME: "int64"})
    append_partition(tmp_path / "store", "bars_60000000000ns", frame, snapshot)


def _age_the_store(tmp_path, hours: float):
    import os
    import time
    when = time.time() - hours * 3600
    for part in (tmp_path / "store").rglob("*.parquet"):
        os.utime(part, (when, when))


def _captured_tape(tmp_path, date: str, symbols, venue: str = "binance"):
    folder = tmp_path / "raw" / venue / date
    folder.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        (folder / f"trade_{symbol}_{date}T00.ndjson.zst").write_bytes(b"x")


def test_a_store_that_stopped_being_written_is_not_ok(tmp_path):
    """A row count cannot tell a live store from an abandoned one.

    Measured on the real archive 2026-08-08: the wall reported
    `bitemporal store | ok | 1671 rows across 12 append-only part(s)` while the
    newest partition was five days old and covered 6 of 2,109 captured symbols.
    Green, for a pipeline that had stopped producing. Rule 8 names this exact
    failure - "if the source stopped updating, the board says stopped" - and the
    probe was counting rows, which a dead store keeps forever.

    Bars legitimately trail the tape by up to a day, because they build closed days
    only, so the threshold has to sit above that rather than at zero.
    """
    from statuswall.evidence import STOPPED, probe_bitemporal_store

    _bars_partition(tmp_path, "snap1")
    _age_the_store(tmp_path, hours=120)

    result = probe_bitemporal_store(_facts(capture_root=tmp_path))
    assert result.state == STOPPED
    assert "120h" in result.detail or "5.0 day" in result.detail, result.detail


def test_a_store_covering_a_fraction_of_the_captured_tape_is_degraded(tmp_path):
    """Six symbols out of 2,109 must not read the same as six out of six.

    The number that was wrong for five days was breadth, and nothing on the board
    was measuring it - so a store holding 0.3% of the captured universe presented
    as a healthy store.
    """
    from statuswall.evidence import DEGRADED, probe_bitemporal_store

    _bars_partition(tmp_path, "snap1", symbol="BTCUSDT")
    _captured_tape(tmp_path, "2026-08-03", [f"SYM{i}USDT" for i in range(50)])

    result = probe_bitemporal_store(
        _facts(capture_root=tmp_path, venues=["binance"], latest_capture_date="2026-08-03"))
    assert result.state == DEGRADED
    assert "1 of 50" in result.detail, result.detail


def test_coverage_that_cannot_be_measured_is_named_not_assumed(tmp_path):
    """No captured tape to compare against is not the same as full coverage.

    It must not degrade the tile - there is no evidence of a shortfall - and it
    must not silently read as complete either. So the detail says coverage was not
    measured, which is Rule 8's "absence of evidence renders as its own state".
    """
    from statuswall.evidence import OK, probe_bitemporal_store

    _bars_partition(tmp_path, "snap1")
    result = probe_bitemporal_store(_facts(capture_root=tmp_path))

    assert result.state == OK
    assert "coverage not measured" in result.detail, result.detail


def test_price_validity_probe_fails_on_a_stored_bar_whose_price_is_zero(tmp_path):
    """A price of zero is not a price, and nothing on the wall was checking.

    The store tile counts rows, then freshness, then coverage - and every one of
    those was green while 746 of 1,671 bars carried `low <= 0`. Binance emits
    placeholder frames on its trade stream (`p` "0", `q` "0", `X` "NA") and one of
    them reaching `low=("price", "min")` ruins a bar that otherwise looks perfect:
    right open, right high, hundreds of trades. It was found by a paper-plumbing run
    refusing to divide by zero, which is not a monitoring strategy.

    FAILING rather than DEGRADED: a bar with an impossible price is not a smaller
    truth, it is a wrong one, and anything that reads it computes a wrong number.
    """
    from statuswall.evidence import FAILING, probe_bar_price_validity

    _bars_partition(tmp_path, "good", symbol="BTCUSDT")
    _bars_partition(tmp_path, "bad", symbol="ETHUSDT", low=0.0)

    result = probe_bar_price_validity(_facts(capture_root=tmp_path))
    assert result.state == FAILING
    assert "1 of 2" in result.detail, result.detail
    assert "ETHUSDT" in result.detail, "the tile must name what to go and look at"


def test_price_validity_probe_is_ok_when_every_served_price_is_positive(tmp_path):
    from statuswall.evidence import OK, probe_bar_price_validity

    _bars_partition(tmp_path, "good", symbol="BTCUSDT")
    result = probe_bar_price_validity(_facts(capture_root=tmp_path))

    assert result.state == OK
    assert "1 bar(s)" in result.detail, result.detail


def test_price_validity_probe_reports_not_built_before_any_store_exists(tmp_path):
    from statuswall.evidence import NOT_BUILT, probe_bar_price_validity
    assert probe_bar_price_validity(_facts(capture_root=tmp_path)).state == NOT_BUILT


def test_price_validity_probe_judges_what_the_reader_serves_not_the_files(tmp_path):
    """A corrupt row already superseded by a correction is not a live defect.

    The store is append-only and corrections are new rows, so the poisoned bars are
    still on disk after remediation - permanently. A probe reading the parquet files
    would report FAILING forever and could never be cleared, which trains everyone
    to ignore it.
    """
    from statuswall.evidence import OK, probe_bar_price_validity

    _bars_partition(tmp_path, "bad", symbol="BTCUSDT", low=0.0, available_at=1_100)
    _bars_partition(tmp_path, "fixed", symbol="BTCUSDT", low=63870.0, available_at=9_999)

    assert probe_bar_price_validity(_facts(capture_root=tmp_path)).state == OK


# --------------------------------------------------------------------------
# the descriptor pool tile
#
# The failure it watches for is quiet by nature: a pool evicting steadily loses
# nothing and breaks nothing, it just reopens hours and compresses worse, and it
# looks identical on the board to a pool doing nothing. The only difference is a
# counter, so a tile that cannot tell them apart is not worth having.
# --------------------------------------------------------------------------

def _pool_report(evicted=0, peak=100, budget=32640, ts_ns=None, **extra):
    import time
    return {"writer_pool": {"evicted": evicted, "peak_open_hours": peak,
                            "budget": budget, "open_hours": peak},
            "writer_pool_ts_ns": time.time_ns() if ts_ns is None else ts_ns,
            **extra}


def test_pool_tile_is_not_measured_when_no_recorder_ever_reported_one():
    """Rule 8, and the whole reason this state exists. A recorder predating the
    report, or one that has not run, must not read as a healthy pool - "nobody
    looked" and "nothing was evicted" are different facts."""
    from statuswall.evidence import NOT_MEASURED, probe_writer_descriptor_pool
    facts = _facts(reports={"binance": {"writer_pool": None, "writer_pool_ts_ns": None}})

    result = probe_writer_descriptor_pool(facts)

    assert result.state == NOT_MEASURED
    assert "binance" in result.detail


def test_pool_tile_is_ok_only_when_a_report_says_nothing_was_evicted():
    from statuswall.evidence import OK, probe_writer_descriptor_pool
    facts = _facts(reports={"binance": _pool_report(evicted=0, peak=1200)})

    result = probe_writer_descriptor_pool(facts)

    assert result.state == OK
    assert "nothing evicted" in result.detail
    # The numbers, not just the verdict: a tile whose proof cannot be checked
    # against the ledger is an assertion.
    assert "1200 of 32640" in result.detail


def test_pool_tile_degrades_as_soon_as_anything_is_evicted():
    """Degraded rather than failing - no frame is lost. But a recorder evicting
    at all means the descriptor budget has become the binding constraint rather
    than the safety net it is meant to be."""
    from statuswall.evidence import DEGRADED, probe_writer_descriptor_pool
    facts = _facts(reports={"binance": _pool_report(evicted=5216, peak=384, budget=384)})

    result = probe_writer_descriptor_pool(facts)

    assert result.state == DEGRADED
    assert "5216 evicted" in result.detail


def test_pool_tile_warns_before_eviction_starts_rather_than_after():
    """A pool at 80% of budget is one burst of new listings away from evicting.
    Reporting only after the fact makes the tile a historian."""
    from statuswall.evidence import DEGRADED, OK, probe_writer_descriptor_pool

    tight = probe_writer_descriptor_pool(
        _facts(reports={"binance": _pool_report(evicted=0, peak=90, budget=100)}))
    roomy = probe_writer_descriptor_pool(
        _facts(reports={"binance": _pool_report(evicted=0, peak=50, budget=100)}))

    assert tight.state == DEGRADED
    assert roomy.state == OK


def test_pool_tile_says_so_when_the_newest_report_is_stale():
    """A healthy report from a recorder that died three hours ago is a fact about
    three hours ago. Rule 8: staleness has to be loud, not implied."""
    import time
    from statuswall.evidence import PARTIAL, probe_writer_descriptor_pool
    old = time.time_ns() - 3 * 3600 * 1_000_000_000
    facts = _facts(reports={"binance": _pool_report(ts_ns=old)})

    result = probe_writer_descriptor_pool(facts)

    assert result.state == PARTIAL
    assert "minutes old" in result.detail


def test_pool_tile_names_the_venues_that_reported_nothing():
    """One venue reporting is not every venue reporting, and the difference is
    exactly where a blind spot would hide."""
    from statuswall.evidence import PARTIAL, probe_writer_descriptor_pool
    facts = _facts(reports={"binance": _pool_report(),
                            "hyperliquid": {"writer_pool": None, "writer_pool_ts_ns": None}})

    result = probe_writer_descriptor_pool(facts)

    assert result.state == PARTIAL
    assert "Not reported by hyperliquid" in result.detail


def test_an_unmeasured_tile_reaches_the_attention_panel_and_renders():
    """Rule 8 again: a board with no way to show a state has not been tested
    against it. Unmeasured must be visible, not merely defined."""
    from statuswall.evidence import NOT_MEASURED
    features = [_feature("Bounded writer descriptor pool")]
    results = {features[0].key: ProbeResult(NOT_MEASURED, "nobody reported one", "ledger")}

    page = render_wall(WallInput(features, results, _facts()))

    assert "NOT MEASURED" in page
    assert "nobody reported one" in page
    assert "Needs attention" in page


def test_not_measured_is_not_on_the_health_colour_axis():
    """It is not a degree of health, so it must not borrow a hue that reads as
    one. Sharing green's or amber's colour is how "nobody looked" starts looking
    like "nearly fine"."""
    from statuswall.evidence import DEGRADED, NOT_MEASURED, OK
    assert STATE_STYLE[NOT_MEASURED][0] not in {STATE_STYLE[OK][0], STATE_STYLE[DEGRADED][0]}


# --------------------------------------------------------------------------
# the venue-health tile
#
# It read "health monitoring live and auto-halt armed" until 2026-08-09 and the
# second half was measured by nothing. `observe()` and `assess_venue()` have no
# callers anywhere in src/, so no degradation can halt a venue - and the stored
# state the tile rendered as current had been written once, 19 hours earlier, by
# an ad-hoc run. A green claim about a safety mechanism is worse than no claim:
# it is what stops anyone checking whether the mechanism exists.
# --------------------------------------------------------------------------

def _health_facts(tmp_path, **report):
    base = {"silent_streams": 0, "corrupting_non_gap": 0}
    base.update(report)
    return _facts(capture_root=tmp_path, venues=["binance"], reports={"binance": base})


def test_the_tile_never_claims_the_halt_is_armed_when_nothing_feeds_it():
    """The exact false claim. No registry state at all is the honest zero case."""
    from statuswall.evidence import DEGRADED, probe_venue_health
    result = probe_venue_health(_health_facts(Path("/nonexistent")))

    assert result.state == DEGRADED
    assert "NOT ARMED" in result.detail
    assert "armed (" not in result.detail, "still asserting the halt is armed"


def test_a_stale_halt_verdict_is_not_reported_as_current(tmp_path):
    """The state on disk on 2026-08-09: healthy, and 19 hours old. Rendering it
    as "3 venue(s) tradeable" is a statement about yesterday."""
    from ops.venue_halt import VenueHaltRegistry
    from statuswall.evidence import DEGRADED, probe_venue_health

    day_ago = time.time_ns() - 19 * 3600 * 1_000_000_000
    VenueHaltRegistry(tmp_path / "ops", clock_ns=lambda: day_ago).observe(
        "binance", {"silent_streams": 0, "corrupting_non_gap": 0})

    result = probe_venue_health(_health_facts(tmp_path))

    assert result.state == DEGRADED
    assert "NOT ARMED" in result.detail
    assert "19h" in result.detail, result.detail


def test_a_freshly_fed_registry_is_what_earns_the_ok(tmp_path):
    """The state this tile is allowed to be green in - and the one nothing in
    the system currently produces, because no caller feeds the registry."""
    from ops.venue_halt import VenueHaltRegistry
    from statuswall.evidence import OK, probe_venue_health

    VenueHaltRegistry(tmp_path / "ops", clock_ns=time.time_ns).observe(
        "binance", {"silent_streams": 0, "corrupting_non_gap": 0})

    result = probe_venue_health(_health_facts(tmp_path))

    assert result.state == OK
    assert "1 of 1 venue(s) tradeable" in result.detail


def test_the_tradeable_count_comes_from_the_registry_not_the_venue_list(tmp_path):
    """It was `len(facts.reports)` - the number of venues capturing, which is not
    a statement about whether any of them may be traded, and which can never
    disagree with itself however halted everything is."""
    from ops.venue_halt import VenueHaltRegistry
    from statuswall.evidence import probe_venue_health

    now = time.time_ns()
    registry = VenueHaltRegistry(tmp_path / "ops", clock_ns=lambda: now)
    registry.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 0})
    registry.observe("hyperliquid", {"silent_streams": 0, "corrupting_non_gap": 0})

    facts = _facts(capture_root=tmp_path, venues=["binance", "hyperliquid"],
                   reports={"binance": {"silent_streams": 0, "corrupting_non_gap": 0},
                            "hyperliquid": {"silent_streams": 0, "corrupting_non_gap": 0},
                            "binance-spot": {"silent_streams": 0, "corrupting_non_gap": 0}})
    result = probe_venue_health(facts)

    # binance-spot was never observed, so it is not tradeable and the tile is not
    # allowed to count it.
    assert "NOT ARMED" in result.detail
    assert "binance-spot" in result.detail


def test_a_recorded_halt_still_outranks_everything_on_the_tile(tmp_path):
    """Whatever else is wrong, a venue someone marked untradeable has to show."""
    from ops.venue_halt import VenueHaltRegistry
    from statuswall.evidence import FAILING, probe_venue_health

    VenueHaltRegistry(tmp_path / "ops", clock_ns=time.time_ns).observe(
        "binance", {"silent_streams": 0, "corrupting_non_gap": 9})

    result = probe_venue_health(_health_facts(tmp_path, corrupting_non_gap=9))

    assert result.state == FAILING
    assert "HALTED: binance" in result.detail


# --------------------------------------------------------------------------
# the reachability tile - the board's own blind spot, put on the board
# --------------------------------------------------------------------------

def test_the_reachability_tile_is_not_measured_without_a_ledger():
    """"No unsupported claims" and "no claims examined" are different answers,
    and a pass given no ledger produced the second one."""
    from statuswall.evidence import NOT_MEASURED, probe_unsupported_claims
    result = probe_unsupported_claims(_facts(ledger_root=None))

    assert result.state == NOT_MEASURED
    assert "no requirements ledger" in result.detail


def test_the_reachability_tile_degrades_on_a_row_claiming_dead_code(tmp_path):
    from statuswall.evidence import DEGRADED, probe_unsupported_claims

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "alive.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "orphan.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("python -m alive\n", encoding="utf-8")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "slice.md").write_text(
        "| # | Requirement | Category | Status | Phase | Evidence | Sources | Notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| XX-001 | A thing | ops | BUILT | P0 | `src/orphan.py` | s | n |\n",
        encoding="utf-8")

    result = probe_unsupported_claims(
        _facts(repo_root=tmp_path, ledger_root=ledger))

    assert result.state == DEGRADED
    assert "XX-001" in result.detail
    assert "1 ledger row(s) claim BUILT" in result.detail


def test_the_reachability_tile_is_ok_when_every_claim_is_reachable(tmp_path):
    from statuswall.evidence import OK, probe_unsupported_claims

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "alive.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("python -m alive\n", encoding="utf-8")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "slice.md").write_text(
        "| # | Requirement | Category | Status | Phase | Evidence | Sources | Notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| XX-001 | A thing | ops | BUILT | P0 | `src/alive.py` | s | n |\n",
        encoding="utf-8")

    result = probe_unsupported_claims(_facts(repo_root=tmp_path, ledger_root=ledger))

    assert result.state == OK
    assert "0 unreachable module(s) of 1" in result.detail


def test_the_trade_tape_counts_venues_that_have_a_tape_not_venues_with_a_folder():
    """It read `len(facts.venues)` until 2026-08-10 - every venue with an
    archive directory, including two that carry no trades at all: bybit is
    polled for funding and bybit-liq records liquidations. The tile therefore
    claimed wider coverage than it had, and it was doing so before coinbase
    existed to make the gap visible.

    `matches` is in the count because that is coinbase's name for the tape. An
    exact stream-name match means a venue whose word for a feed is missing from
    the tuple is silently absent from the tile that exists to count it.
    """
    facts = _facts(
        capture_running=True,
        venues=["binance", "coinbase", "bybit", "bybit-liq"],
        reports={
            "binance": {"raw_bytes_by_stream": {"trade_BTCUSDT": 4096}},
            "coinbase": {"raw_bytes_by_stream": {"matches_BTC-USD": 2048}},
            "bybit": {"raw_bytes_by_stream": {"linearTickers_ALL": 9999}},
            "bybit-liq": {"raw_bytes_by_stream": {"allLiquidation_ALL": 9999}},
        },
    )

    result = PROBES["spot ohlcv trade tape multi venue"](facts)

    assert "captured on 2 venues" in result.detail, result.detail


def test_every_venues_word_for_depth_reaches_the_depth_tile():
    """Four names for one feed: binance `depth`, hyperliquid `l2Book`,
    coinbase `level2`, and the periodic REST book as `depthSnapshot` on both
    binance-spot and coinbase. The snapshot streams were absent from this tile
    before 2026-08-10, which is the stream the book dataset is built from."""
    facts = _facts(
        capture_running=True,
        venues=["binance", "hyperliquid", "coinbase"],
        reports={
            "binance": {"raw_bytes_by_stream": {"depth_BTCUSDT": 1,
                                                "depthSnapshot_BTCUSDT": 1}},
            "hyperliquid": {"raw_bytes_by_stream": {"l2Book_BTC": 1}},
            "coinbase": {"raw_bytes_by_stream": {"level2_BTC-USD": 1,
                                                 "level2Snapshot_BTC-USD": 1,
                                                 "depthSnapshot_BTC-USD": 1}},
        },
    )

    result = PROBES["l2 order book depth 20 50 levels"](facts)

    assert "6 depth streams" in result.detail, result.detail


# --------------------------------------------------------------------------
# Phase B feature tiles - and the guard against a tile certifying itself
# --------------------------------------------------------------------------

def _fake_facts(tmp_path):
    """Only the two fields `_probe_computed_feature` reads."""
    from types import SimpleNamespace
    return SimpleNamespace(capture_root=tmp_path, repo_root=Path.cwd())


class _Table:
    def __init__(self, rows, refused):
        import pandas as pd
        self.rows = pd.DataFrame({"v": list(range(rows))})
        self.refused = refused


def test_the_board_measuring_a_feature_does_not_count_as_consuming_it(tmp_path):
    """The probe has to call the feature, and that call is in the source tree.

    On the first run of these tiles, five of six graded OK with the detail "read
    by statuswall.evidence" - the board had found its own probe call and read it
    as a consumer. A tile that counts its own measurement as the thing being
    used certifies itself, which is Rule 8's failure wearing the costume of the
    fix for it.
    """
    from statuswall.evidence import _consumers_of

    consumers = _consumers_of(Path.cwd(), "features.microprice", "compute_microprice")
    assert not any(name.startswith("statuswall.") for name in consumers), (
        "the board is an observer of a feature, never a consumer of it")


def test_a_feature_nothing_reads_is_partial_rather_than_ok(tmp_path):
    """§1a.5: a value no one consumes cannot change what the system does when it
    is wrong, so producing rows is not the same as being finished."""
    from statuswall.evidence import PARTIAL, _probe_computed_feature

    result = _probe_computed_feature(
        _fake_facts(tmp_path), module="features.nothing_reads_this",
        entry_point="compute_nothing_reads_this",
        compute=lambda root, now: _Table(rows=5, refused={}), unit="values")

    assert result.state == PARTIAL
    assert "Nothing consumes it" in result.detail


def test_a_feature_that_refuses_everything_is_degraded_not_unbuilt(tmp_path):
    """The state that matters most, and the one a naive tile gets wrong twice.

    `realized_volatility` refuses every window while bars are hours stale. That
    must not render as NOT BUILT - the module exists and is behaving exactly as
    designed - and must not render healthy either, because nothing was measured.
    """
    from statuswall.evidence import DEGRADED, _probe_computed_feature

    result = _probe_computed_feature(
        _fake_facts(tmp_path), module="features.refuses_everything",
        entry_point="compute_refuses_everything",
        compute=lambda root, now: _Table(rows=0, refused={"too_few_observations": 10600}),
        unit="values")

    assert result.state == DEGRADED
    assert "too_few_observations" in result.detail
    assert "10600" in result.detail


def test_a_feature_that_raises_is_reported_not_swallowed(tmp_path):
    """A tile that caught the exception would render a broken feature as quiet."""
    from statuswall.evidence import NOT_MEASURED, _probe_computed_feature

    def explode(root, now):
        raise ValueError("store is gone")

    result = _probe_computed_feature(
        _fake_facts(tmp_path), module="features.explodes",
        entry_point="compute_explodes", compute=explode, unit="values")

    assert result.state == NOT_MEASURED
    assert "store is gone" in result.detail


def test_every_phase_b_module_built_so_far_has_a_tile():
    """A feature module with no probe is invisible: the board reports NOT BUILT
    for code that exists, which is how six modules landed and moved nothing."""
    from statuswall.evidence import PROBES

    for catalogue_key in ("realized volatility multi horizon", "microprice",
                          "depth weighted order flow imbalance",
                          "absorption detection delta vs price hold",
                          "kyle s lambda", "fractional differentiation"):
        assert catalogue_key in PROBES, f"{catalogue_key} has no probe"
