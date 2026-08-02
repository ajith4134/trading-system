import os
from pathlib import Path
from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, read_all,
    SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS,
)


def _record_events(root: Path, venue: str, count: int) -> None:
    ledger = CaptureLedger(root, venue)
    for i in range(count):
        ledger.record(LedgerEvent(
            ts_ns=1785648600_000_000_000 + i, venue=venue, stream="depth",
            kind=f"event{i}", severity=SEVERITY_CORRUPTING, detail={"i": i},
        ))
    ledger.close()


def test_torn_final_line_does_not_lose_the_intact_events(tmp_path: Path):
    """The ledger is the anomaly record, read during an incident.

    A crash truncates the last line. Parsing the file as one list comprehension
    made that single torn line fatal for the whole day, so every intact event
    before it became unreadable at the moment it was needed most.
    """
    _record_events(tmp_path, "binance", 5)
    path = tmp_path / "ledger" / "binance" / "2026-08-02" / "events.ndjson"
    path.write_bytes(path.read_bytes()[:-10])

    result = read_all(tmp_path, "binance", "2026-08-02")

    assert [e.kind for e in result] == ["event0", "event1", "event2", "event3"]
    assert len(result.damaged) == 1
    assert result.damaged[0].line_number == 5
    assert "event4" in result.damaged[0].text        # the damage is surfaced, not dropped
    assert result.damaged[0].error


def test_a_torn_line_in_the_middle_keeps_the_events_around_it(tmp_path: Path):
    _record_events(tmp_path, "binance", 4)
    path = tmp_path / "ledger" / "binance" / "2026-08-02" / "events.ndjson"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1][: len(lines[1]) // 2]        # torn line, not last
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = read_all(tmp_path, "binance", "2026-08-02")
    assert [e.kind for e in result] == ["event0", "event2", "event3"]
    assert [d.line_number for d in result.damaged] == [2]


def test_a_line_that_is_json_but_not_an_event_is_reported_damaged(tmp_path: Path):
    _record_events(tmp_path, "binance", 1)
    path = tmp_path / "ledger" / "binance" / "2026-08-02" / "events.ndjson"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"not":"an event"}\n')

    result = read_all(tmp_path, "binance", "2026-08-02")
    assert len(result) == 1
    assert [d.line_number for d in result.damaged] == [2]


def test_an_intact_ledger_reports_no_damage(tmp_path: Path):
    _record_events(tmp_path, "binance", 3)
    result = read_all(tmp_path, "binance", "2026-08-02")
    assert len(result) == 3
    assert result.damaged == []
    assert result.events == list(result)


def test_records_and_reads_back(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="gap", severity=SEVERITY_CORRUPTING,
        detail={"symbol": "BTCUSDT", "expected_pu": 5, "got_pu": 9},
    ))
    ledger.close()

    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len(events) == 1
    assert events[0].kind == "gap"
    assert events[0].severity == SEVERITY_CORRUPTING
    assert events[0].detail["expected_pu"] == 5


def test_appends_across_multiple_records(tmp_path: Path):
    ledger = CaptureLedger(tmp_path, "hyperliquid")
    for i in range(3):
        ledger.record(LedgerEvent(
            ts_ns=1785648600_000_000_000 + i, venue="hyperliquid",
            stream="l2Book", kind="stale", severity=SEVERITY_OBSERVATION_LOSS,
            detail={"symbol": "BTC"},
        ))
    ledger.close()
    assert len(read_all(tmp_path, "hyperliquid", "2026-08-02")) == 3


def test_handles_non_serializable_types_in_detail(tmp_path: Path):
    """Bytes and other non-JSON types degrade to string, not lost."""
    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="malformed", severity=SEVERITY_CORRUPTING,
        detail={"payload": b"corrupted_bytes_data", "symbol": "BTCUSDT"},
    ))
    ledger.close()

    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len(events) == 1
    # The bytes should be stringified, not lost
    assert "payload" in events[0].detail
    assert isinstance(events[0].detail["payload"], str)
    assert "corrupted_bytes_data" in events[0].detail["payload"]


def test_midnight_rollover_routes_to_correct_date(tmp_path: Path):
    """Events straddling UTC midnight land in their respective date files."""
    ledger = CaptureLedger(tmp_path, "kraken")
    # Reference: 1785648600_000_000_000 ns = 2026-08-02T05:30:00Z
    # Last moment of 2026-08-02: reference + (18.5 hours - 1 second) = 2026-08-02T23:59:59Z
    ts_aug_2_last = 1785648600_000_000_000 + 66599_000_000_000
    # First moment of 2026-08-03: reference + 24 hours = 2026-08-03T05:30:00Z
    ts_aug_3_first = 1785648600_000_000_000 + 86400_000_000_000

    ledger.record(LedgerEvent(
        ts_ns=ts_aug_2_last, venue="kraken", stream="trades",
        kind="late", severity=SEVERITY_OBSERVATION_LOSS,
        detail={"symbol": "BTC"},
    ))
    ledger.record(LedgerEvent(
        ts_ns=ts_aug_3_first, venue="kraken", stream="trades",
        kind="early", severity=SEVERITY_OBSERVATION_LOSS,
        detail={"symbol": "BTC"},
    ))
    ledger.close()

    events_aug_2 = read_all(tmp_path, "kraken", "2026-08-02")
    events_aug_3 = read_all(tmp_path, "kraken", "2026-08-03")

    assert len(events_aug_2) == 1
    assert len(events_aug_3) == 1
    assert events_aug_2[0].kind == "late"
    assert events_aug_3[0].kind == "early"


def test_the_last_nanosecond_of_a_day_is_filed_under_that_day(tmp_path: Path):
    """`ts_ns / 1e9` rounds, and the rounding crosses midnight.

    A present-day nanosecond timestamp is ~1.8e18, far past 2^53 where a float
    stops holding every integer. One nanosecond before UTC midnight divides to
    exactly midnight, so the event was filed under the NEXT day - and a health
    report for the day it belongs to never sees it.
    """
    ts_last_ns_of_aug_2 = 1785715199_999_999_999      # 2026-08-02T23:59:59.999999999Z

    ledger = CaptureLedger(tmp_path, "kraken")
    ledger.record(LedgerEvent(
        ts_ns=ts_last_ns_of_aug_2, venue="kraken", stream="trades",
        kind="last_nanosecond", severity=SEVERITY_OBSERVATION_LOSS, detail={}))
    ledger.close()

    assert [e.kind for e in read_all(tmp_path, "kraken", "2026-08-02")] == ["last_nanosecond"]
    assert read_all(tmp_path, "kraken", "2026-08-03") == []


def test_every_recorded_event_is_fsynced_not_merely_flushed(tmp_path: Path,
                                                            monkeypatch):
    """The ledger is the incident record and the sole evidence of every anomaly.

    `fh.flush()` reaches the page cache, which survives `kill -9` and not a
    power loss or a hypervisor reset. Losing this file's tail is losing exactly
    the events that describe the crash that lost them.

    Power loss cannot be simulated in-process, so this constrains the mechanism;
    the reasoning and the measured cost are on `CaptureLedger.record`.
    """
    synced_fds = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced_fds.append(fd)
        return real_fsync(fd)

    ledger = CaptureLedger(tmp_path, "kraken")
    monkeypatch.setattr(os, "fsync", recording_fsync)
    for i in range(3):
        ledger.record(LedgerEvent(
            ts_ns=1785648600_000_000_000 + i, venue="kraken", stream="trades",
            kind="gap", severity=SEVERITY_OBSERVATION_LOSS, detail={}))
    ledger_fd = ledger._fh.fileno()
    monkeypatch.undo()
    ledger.close()

    assert synced_fds == [ledger_fd] * 3, (
        "an event returned from record() while still only in the page cache")
