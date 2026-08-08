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
    """
    facts = _facts(
        capture_running=True,
        venues=["binance"],
        reports={"binance": {"silent_stream_names": ["markPrice"],
                             "raw_bytes_by_stream": {"premiumIndex_BTCUSDT": 4096}}},
    )
    result = PROBES["mark price vs index vs oracle price per venue"](facts)
    assert result.state == OK, result.detail


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
