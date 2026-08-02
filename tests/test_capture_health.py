import json
import os
import time
from pathlib import Path

import pytest

from capture.capture_health import (
    compute_runway_days, classify_runway, measure_daily_bytes,
    build_report, write_alerts,
)
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO,
    SEVERITY_OBSERVATION_LOSS,
)
from capture.raw_writer import paths_for, writing_marker_path

TS = 1785648600_000_000_000
DATE = "2026-08-02"
BIG = 100_000_000_000                    # free bytes that leave runway "ok"
DAILY = 2_000_000_000


def _write_capture_file(root: Path, venue: str, date: str, size: int,
                        symbol: str = "BTCUSDT", age_days: float = 0.0) -> Path:
    """Put a raw capture file exactly where RawWriter would have put it."""
    raw_path, _ = paths_for(root, venue, "depth", symbol, f"{date}T00")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"x" * size)
    if age_days:
        when = time.time() - age_days * 86400
        os.utime(raw_path, (when, when))
    return raw_path


def _alert_lines(root: Path) -> list[dict]:
    """Every alert record on disk, skipping any line a test planted broken."""
    path = root / "health" / "alerts.ndjson"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _reasons(root: Path) -> list[str]:
    return [alert["reason"] for alert in _alert_lines(root)]


def test_runway_is_free_over_daily():
    assert compute_runway_days(100_000_000_000, 2_000_000_000) == 50.0


def test_runway_is_infinite_when_nothing_written():
    assert compute_runway_days(100, 0) == float("inf")


def test_classify_thresholds():
    assert classify_runway(90) == "ok"
    assert classify_runway(25) == "warn"
    assert classify_runway(10) == "alert"
    assert classify_runway(5) == "decision_point"


def test_report_counts_gaps_by_severity(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap",
                              SEVERITY_CORRUPTING, {"symbol": "BTCUSDT"}))
    ledger.close()

    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=100_000_000_000, daily_bytes=2_000_000_000)
    assert report["gaps"]["corrupting"] == 1
    assert report["runway_days"] == 50.0
    assert report["runway_status"] == "ok"


def test_write_alerts_emits_lines_for_bad_states(tmp_path: Path):
    report = build_report(tmp_path, "binance", "2026-08-02",
                          free_bytes=4_000_000_000, daily_bytes=2_000_000_000)
    count = write_alerts(tmp_path, report)
    assert count == 1
    lines = (tmp_path / "health" / "alerts.ndjson").read_text().splitlines()
    assert json.loads(lines[0])["reason"] == "runway_decision_point"


# --- thresholds -------------------------------------------------------------

def test_each_threshold_is_inclusive_of_its_own_boundary():
    """30/14/7 are the escalation points, so landing exactly on one escalates."""
    assert classify_runway(30.0) == "warn"
    assert classify_runway(30.001) == "ok"
    assert classify_runway(14.0) == "alert"
    assert classify_runway(14.001) == "warn"
    assert classify_runway(7.0) == "decision_point"
    assert classify_runway(7.001) == "alert"


def test_a_full_disk_is_a_decision_point_not_an_ok():
    assert compute_runway_days(0, DAILY) == 0.0
    assert classify_runway(0.0) == "decision_point"
    assert classify_runway(float("-inf")) == "decision_point"


def test_nothing_written_is_never_an_alert():
    assert classify_runway(compute_runway_days(1, 0)) == "ok"


def test_a_nonsense_write_rate_is_refused_rather_than_called_ok():
    """Both of these classify as "ok" if they are allowed through, and a
    silently healthy answer is the failure this module exists to prevent."""
    with pytest.raises(ValueError):
        compute_runway_days(BIG, float("nan"))
    with pytest.raises(ValueError):
        compute_runway_days(BIG, -DAILY)


# --- measured write rate ----------------------------------------------------

def test_measure_daily_bytes_of_a_root_that_never_captured_is_zero(tmp_path: Path):
    assert measure_daily_bytes(tmp_path) == 0.0


def test_measure_daily_bytes_ignores_files_older_than_the_window(tmp_path: Path):
    _write_capture_file(tmp_path, "binance", "2026-07-01", 7_000, age_days=30)
    assert measure_daily_bytes(tmp_path, days=7) == 0.0


def test_measure_daily_bytes_divides_by_days_observed_not_by_the_window(tmp_path: Path):
    """A young archive must not look like a slow one.

    Two days of data divided by a 7-day window reports 2/7ths of the real
    write rate, which multiplies the runway estimate by 3.5 - reporting
    healthy for a disk that is not.
    """
    _write_capture_file(tmp_path, "binance", "2026-08-01", 1_000_000, age_days=2)
    _write_capture_file(tmp_path, "binance", DATE, 1_000_000, age_days=0)
    assert measure_daily_bytes(tmp_path, days=7) == pytest.approx(1_000_000, rel=0.02)


def test_measure_daily_bytes_never_divides_by_less_than_a_day(tmp_path: Path):
    """An hour-old archive would otherwise report a 24x inflated rate."""
    _write_capture_file(tmp_path, "binance", DATE, 1_000_000, age_days=1 / 24)
    assert measure_daily_bytes(tmp_path, days=7) == pytest.approx(1_000_000, rel=0.02)


def test_measure_daily_bytes_counts_index_files_because_they_use_the_disk_too(tmp_path: Path):
    raw_path, idx_path = paths_for(tmp_path, "binance", "depth", "BTCUSDT", f"{DATE}T00")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"x" * 600_000)
    idx_path.write_bytes(b"y" * 400_000)
    assert measure_daily_bytes(tmp_path, days=1) == pytest.approx(1_000_000, rel=0.02)


def test_measure_daily_bytes_survives_a_file_that_cannot_be_stat_ed(tmp_path: Path):
    """Files rotate, quarantine and vanish under a live capture; a scan that
    dies on one of them takes the whole health check down with it."""
    _write_capture_file(tmp_path, "binance", DATE, 1_000_000)
    folder = paths_for(tmp_path, "binance", "depth", "BTCUSDT", f"{DATE}T00")[0].parent
    (folder / "vanished.ndjson.zst").symlink_to(folder / "no_such_file.ndjson.zst")
    assert measure_daily_bytes(tmp_path, days=1) == pytest.approx(1_000_000, rel=0.02)


def test_measure_daily_bytes_refuses_a_window_of_zero_days(tmp_path: Path):
    with pytest.raises(ValueError):
        measure_daily_bytes(tmp_path, days=0)


# --- absence, which must never read as health -------------------------------

def test_a_venue_day_that_captured_nothing_is_absent_not_healthy(tmp_path: Path):
    _write_capture_file(tmp_path, "hyperliquid", DATE, 5_000)      # another venue did write
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["capture_status"] == "absent"
    assert report["raw_data_bytes"] == 0


def test_a_venue_day_that_captured_data_is_present(tmp_path: Path):
    _write_capture_file(tmp_path, "binance", DATE, 4_096)
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["capture_status"] == "present"
    assert report["raw_data_bytes"] == 4_096


def test_a_writing_marker_alone_is_not_captured_data(tmp_path: Path):
    """A marker is a sidecar with a pid in it. Counting bytes indiscriminately
    would let an hour that opened and captured nothing report as present."""
    _write_capture_file(tmp_path, "hyperliquid", DATE, 5_000)
    raw_path, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", f"{DATE}T00")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    writing_marker_path(raw_path).write_text("12345")
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["capture_status"] == "absent"


def test_a_root_that_has_never_captured_anything_is_not_called_absent(tmp_path: Path):
    """Absence means "this stream stopped", which cannot be claimed before the
    system has ever run. Alerting here would fire on every fresh install."""
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["capture_status"] == "uninitialised"
    assert write_alerts(tmp_path, report) == 0


def test_absent_capture_raises_an_alert(tmp_path: Path):
    _write_capture_file(tmp_path, "hyperliquid", DATE, 5_000)
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1
    assert _reasons(tmp_path) == ["capture_absent"]


def test_a_date_that_is_not_a_date_is_refused(tmp_path: Path):
    """A mistyped date reads an empty ledger and would report a clean day."""
    with pytest.raises(ValueError):
        build_report(tmp_path, "binance", "2026-8-2", free_bytes=BIG, daily_bytes=DAILY)


# --- what the ledger says ---------------------------------------------------

def test_damaged_ledger_lines_are_reported_and_alerted(tmp_path: Path):
    """An unreadable ledger is not a quiet ledger. Counting only the events
    that parsed reports a clean day from a file that lost its evidence."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap",
                              SEVERITY_OBSERVATION_LOSS, {}))
    ledger.close()
    path = tmp_path / "ledger" / "binance" / DATE / "events.ndjson"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"ts_ns": 1785648600000000000, "venue": "bin')

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["damaged_ledger_lines"] == 1
    assert report["events_total"] == 1
    write_alerts(tmp_path, report)
    assert "ledger_damaged" in _reasons(tmp_path)


def test_a_mistyped_severity_does_not_take_the_whole_report_down(tmp_path: Path):
    """The line is valid JSON, so `read_all` returns it as an event. A report
    that raises here leaves the venue-day unreported, which downstream cannot
    tell apart from a healthy one."""
    path = tmp_path / "ledger" / "binance" / DATE / "events.ndjson"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ts_ns": TS, "venue": "binance", "stream": "depth",
                                "kind": "gap", "severity": ["corrupting"],
                                "detail": {}}) + "\n")

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["events_total"] == 1
    assert report["gaps"]["unknown"] == 1
    assert report["gaps"]["corrupting"] == 0


def test_corrupting_events_that_are_not_gaps_are_still_counted(tmp_path: Path):
    """`unwritable_stream_total` means a stream is being dropped entirely - the
    worst thing in the ledger, and not a gap."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "unwritable_stream_total",
                              SEVERITY_CORRUPTING, {"frames": 900}))
    ledger.record(LedgerEvent(TS, "binance", "trades", "gap",
                              SEVERITY_CORRUPTING, {}))
    ledger.record(LedgerEvent(TS, "binance", "trades", "malformed",
                              SEVERITY_INFO, {}))
    ledger.record(LedgerEvent(TS, "binance", "book", "gap",
                              SEVERITY_OBSERVATION_LOSS, {}))
    ledger.close()

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["gaps"] == {"corrupting": 1, "observation_loss": 1, "info": 0}
    assert report["corrupting_non_gap"] == 1
    assert report["events_total"] == 4
    write_alerts(tmp_path, report)
    assert "corrupting_non_gap_events" in _reasons(tmp_path)


# --- alerts -----------------------------------------------------------------

def test_a_healthy_report_writes_no_alert_file_at_all(tmp_path: Path):
    _write_capture_file(tmp_path, "binance", DATE, 4_096)
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 0
    assert not (tmp_path / "health" / "alerts.ndjson").exists()


def test_the_worst_condition_is_the_first_line(tmp_path: Path):
    """Whoever opens this file reads the top of it."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap", SEVERITY_CORRUPTING, {}))
    ledger.close()
    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 2
    assert _reasons(tmp_path) == ["runway_decision_point", "corrupting_gaps"]


def test_repeating_the_same_alert_does_not_repeat_the_line(tmp_path: Path):
    """A report run hourly must not write 24 identical lines a day. A file that
    floods trains its reader to stop reading it."""
    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1
    assert write_alerts(tmp_path, report) == 0
    assert write_alerts(tmp_path, report) == 0
    assert _reasons(tmp_path) == ["runway_decision_point"]


def test_a_worsening_runway_is_a_new_alert(tmp_path: Path):
    warned = build_report(tmp_path, "binance", DATE,
                          free_bytes=40_000_000_000, daily_bytes=DAILY)   # 20 days
    assert write_alerts(tmp_path, warned) == 1
    worse = build_report(tmp_path, "binance", DATE,
                         free_bytes=20_000_000_000, daily_bytes=DAILY)    # 10 days
    assert write_alerts(tmp_path, worse) == 1
    assert _reasons(tmp_path) == ["runway_warn", "runway_alert"]


def test_a_tenfold_worsening_re_alerts_but_drift_does_not(tmp_path: Path):
    """Collapsing repeats must not also collapse an order-of-magnitude jump."""
    def report_with_gaps(count: int) -> dict:
        report = build_report(tmp_path, "binance", DATE,
                              free_bytes=BIG, daily_bytes=DAILY)
        report["gaps"]["corrupting"] = count
        return report

    assert write_alerts(tmp_path, report_with_gaps(1)) == 1
    assert write_alerts(tmp_path, report_with_gaps(4)) == 0        # same decade
    assert write_alerts(tmp_path, report_with_gaps(40)) == 1       # 10x worse
    assert write_alerts(tmp_path, report_with_gaps(90)) == 0
    assert [alert["count"] for alert in _alert_lines(tmp_path)] == [1, 40]


def test_the_same_reason_for_a_different_day_is_a_different_alert(tmp_path: Path):
    for date in (DATE, "2026-08-03"):
        report = build_report(tmp_path, "binance", date,
                              free_bytes=4_000_000_000, daily_bytes=DAILY)
        assert write_alerts(tmp_path, report) == 1
    assert [alert["date"] for alert in _alert_lines(tmp_path)] == [DATE, "2026-08-03"]


def test_an_unparseable_alert_line_does_not_silence_the_next_alert(tmp_path: Path):
    """One bad line must not make every future alert look already-recorded."""
    folder = tmp_path / "health"
    folder.mkdir()
    (folder / "alerts.ndjson").write_text("not json at all\n")

    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1
    assert "runway_decision_point" in _reasons(tmp_path)[-1:]


def test_a_torn_last_line_is_not_glued_to_the_next_alert(tmp_path: Path):
    """The alert file is appended to; its last line is what a crash tears.
    Appending onto that line would destroy a second record as well."""
    folder = tmp_path / "health"
    folder.mkdir()
    (folder / "alerts.ndjson").write_text('{"reason":"runway_al')      # no newline

    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1
    lines = (folder / "alerts.ndjson").read_text().splitlines()
    assert lines[0] == '{"reason":"runway_al'
    assert json.loads(lines[-1])["reason"] == "runway_decision_point"


def test_every_alert_carries_what_it_is_about(tmp_path: Path):
    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    write_alerts(tmp_path, report)
    alert = _alert_lines(tmp_path)[0]
    assert alert["venue"] == "binance"
    assert alert["date"] == DATE
    assert alert["runway_days"] == 2.0
    assert alert["ts"].endswith("Z")


def test_the_count_returned_is_what_reached_the_disk(tmp_path: Path):
    """A batch where some alerts are new and some are repeats must report only
    the new ones - a caller told '2' would believe two lines exist."""
    report = build_report(tmp_path, "binance", DATE,
                          free_bytes=4_000_000_000, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1                    # runway only

    with_gaps = dict(report, gaps=dict(report["gaps"], corrupting=3))
    assert write_alerts(tmp_path, with_gaps) == 1                 # runway repeats
    assert _reasons(tmp_path) == ["runway_decision_point", "corrupting_gaps"]


def test_an_alert_is_on_disk_before_it_is_reported_as_written(tmp_path: Path):
    """The count is a promise that the line survives a power cut. A buffered
    write that is lost would leave a caller believing it raised an alarm."""
    import capture.capture_health as module

    fsynced: list[str] = []
    real_fsync = os.fsync

    def trace_fsync(fd):
        fsynced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real_fsync(fd)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.os, "fsync", trace_fsync)
    try:
        report = build_report(tmp_path, "binance", DATE,
                              free_bytes=4_000_000_000, daily_bytes=DAILY)
        assert write_alerts(tmp_path, report) == 1
    finally:
        monkeypatch.undo()

    assert str(tmp_path / "health" / "alerts.ndjson") in fsynced, \
        "the alert was never fsynced"


def test_alerts_are_appended_not_rewritten(tmp_path: Path):
    _write_capture_file(tmp_path, "hyperliquid", DATE, 5_000)
    for venue in ("binance", "bybit"):
        report = build_report(tmp_path, venue, DATE, free_bytes=BIG, daily_bytes=DAILY)
        write_alerts(tmp_path, report)
    assert [alert["venue"] for alert in _alert_lines(tmp_path)] == ["binance", "bybit"]


def test_silent_streams_are_counted_and_named_in_the_report(tmp_path: Path):
    """A subscribed stream producing nothing is the failure that started this:
    the archive holds order book and nothing else, and every check that reads
    only file sizes and gap counts calls that healthy. It has to be in the
    report an operator actually opens, not only in the ledger."""
    ledger = CaptureLedger(tmp_path, "binance")
    for stream, symbol in (("markPrice", "BTCUSDT"), ("forceOrder", "BTCUSDT"),
                           ("markPrice", "ETHUSDT")):
        ledger.record(LedgerEvent(TS, "binance", stream, "silent_stream",
                                  SEVERITY_OBSERVATION_LOSS,
                                  {"symbol": symbol, "frames_received": 0}))
    ledger.record(LedgerEvent(TS, "binance", "depth", "gap",
                              SEVERITY_OBSERVATION_LOSS, {}))
    ledger.close()
    _write_capture_file(tmp_path, "binance", DATE, size=1000)

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)

    assert report["silent_streams"] == 3
    assert report["silent_stream_names"] == ["forceOrder", "markPrice"]
    # a silent stream is not a gap, and must not be counted as one
    assert report["gaps"] == {"corrupting": 0, "observation_loss": 1, "info": 0}
    assert report["corrupting_non_gap"] == 0


def test_a_silent_stream_raises_an_alert(tmp_path: Path):
    """Unlike a routine observation_loss gap, this one changes what the operator
    does: a stream is delivering nothing at all."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "markPrice", "silent_stream",
                              SEVERITY_OBSERVATION_LOSS, {"symbol": "BTCUSDT"}))
    ledger.close()
    _write_capture_file(tmp_path, "binance", DATE, size=1000)

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1

    alert = next(a for a in _alert_lines(tmp_path) if a["reason"] == "silent_streams")
    assert alert["count"] == 1
    assert alert["streams"] == ["markPrice"]


def test_the_same_silent_streams_do_not_alert_twice(tmp_path: Path):
    """A health report runs on a schedule and the streams stay silent. One line
    per run is how the alert file becomes unread."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(TS, "binance", "markPrice", "silent_stream",
                              SEVERITY_OBSERVATION_LOSS, {"symbol": "BTCUSDT"}))
    ledger.close()
    _write_capture_file(tmp_path, "binance", DATE, size=1000)

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert write_alerts(tmp_path, report) == 1
    assert write_alerts(tmp_path, report) == 0
    assert _reasons(tmp_path) == ["silent_streams"]


def test_ten_times_as_many_silent_streams_is_a_new_alert(tmp_path: Path):
    """Repeats collapse, a tenfold worsening does not - the same rule the other
    counted alerts use."""
    _write_capture_file(tmp_path, "binance", DATE, size=1000)
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)

    write_alerts(tmp_path, dict(report, silent_streams=1, silent_stream_names=["a"]))
    write_alerts(tmp_path, dict(report, silent_streams=12, silent_stream_names=["a"]))

    assert _reasons(tmp_path) == ["silent_streams", "silent_streams"]


# --------------------------------------------------------------------------
# CRITICAL - one live stream must not report presence on behalf of its siblings
# --------------------------------------------------------------------------

def test_raw_bytes_are_reported_per_stream_not_only_as_a_venue_total(tmp_path: Path):
    """`raw_data_bytes` sums the whole venue-day folder, so one stream that is
    still flowing makes the venue-day look present while every sibling is dead.
    That is the archive-of-order-book-and-nothing-else failure, and the total
    alone cannot show it. The per-stream breakdown can.
    """
    depth, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", f"{DATE}T00")
    depth.parent.mkdir(parents=True, exist_ok=True)
    depth.write_bytes(b"x" * 4_096)
    trade, _ = paths_for(tmp_path, "binance", "trade", "BTCUSDT", f"{DATE}T00")
    trade.write_bytes(b"y" * 512)

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)

    assert report["raw_data_bytes"] == 4_608
    assert report["raw_bytes_by_stream"] == {"depth_BTCUSDT": 4_096,
                                             "trade_BTCUSDT": 512}


def test_bytes_of_one_stream_are_summed_across_its_hours(tmp_path: Path):
    for hour in ("00", "01", "02"):
        raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", f"{DATE}T{hour}")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"x" * 100)

    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert report["raw_bytes_by_stream"] == {"depth_BTCUSDT": 300}


def test_a_stream_that_wrote_nothing_that_day_is_absent_from_the_breakdown(
        tmp_path: Path):
    """A stream with no bytes has no key, which is what makes the difference
    between "quiet market" and "dead" answerable against the subscribed set."""
    _write_capture_file(tmp_path, "binance", DATE, 4_096)
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert list(report["raw_bytes_by_stream"]) == ["depth_BTCUSDT"]


def test_a_writing_marker_is_not_counted_as_a_streams_bytes(tmp_path: Path):
    _write_capture_file(tmp_path, "binance", DATE, 4_096)
    raw, _ = paths_for(tmp_path, "binance", "trade", "BTCUSDT", f"{DATE}T00")
    writing_marker_path(raw).write_text("12345")
    report = build_report(tmp_path, "binance", DATE, free_bytes=BIG, daily_bytes=DAILY)
    assert list(report["raw_bytes_by_stream"]) == ["depth_BTCUSDT"]
