import json
import os
import time
from pathlib import Path

import pytest

from capture.universe_tracker import (
    ImplausibleUniverseSnapshot,
    OutOfOrderSnapshot,
    UniverseTracker,
    UnreadableUniverseState,
    diff_universe,
)

TS = 1785648600_000_000_000          # 2026-08-02T05:30:00Z
DATE = "2026-08-02"


def _instruments_path(root: Path, venue: str = "binance", date: str = DATE) -> Path:
    return root / "universe" / venue / date / "instruments.ndjson"


def _read_records(root: Path, venue: str = "binance", date: str = DATE) -> list[dict]:
    return [json.loads(line)
            for line in _instruments_path(root, venue, date).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _many_symbols(count: int, start: int = 0) -> list[str]:
    return [f"SYM{i:03d}USDT" for i in range(start, start + count)]


def test_diff_detects_listing_and_delisting():
    events = diff_universe(["BTCUSDT", "OLDUSDT"], ["BTCUSDT", "NEWUSDT"],
                           "binance", TS)
    kinds = {(e.symbol, e.kind) for e in events}
    assert ("NEWUSDT", "listed") in kinds
    assert ("OLDUSDT", "delisted") in kinds
    assert ("BTCUSDT", "listed") not in kinds


def test_first_snapshot_lists_everything_as_listed(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    events = tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    assert {e.kind for e in events} == {"listed"}
    assert len(events) == 2


def test_second_snapshot_only_reports_changes(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)
    kinds = {(e.symbol, e.kind) for e in events}
    assert kinds == {("SOLUSDT", "listed"), ("ETHUSDT", "delisted")}


def test_snapshot_is_persisted_and_reloadable(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    reloaded = UniverseTracker(tmp_path, "binance")
    assert reloaded.load_last(TS) == ["BTCUSDT"]


def test_diff_orders_listings_before_delistings_and_sorts_each():
    events = diff_universe(["OLD_B", "OLD_A", "KEEP"], ["KEEP", "NEW_B", "NEW_A"],
                           "binance", TS)
    assert [(e.symbol, e.kind) for e in events] == [
        ("NEW_A", "listed"), ("NEW_B", "listed"),
        ("OLD_A", "delisted"), ("OLD_B", "delisted"),
    ]


def test_diff_carries_the_venue_and_timestamp_onto_every_event():
    events = diff_universe(["OLD"], ["NEW"], "hyperliquid", TS)
    assert {e.venue for e in events} == {"hyperliquid"}
    assert {e.ts_ns for e in events} == {TS}


# --- the empty / implausible parse hazard -----------------------------------
# parse_instruments returns [] both for a malformed payload and for a genuinely
# empty universe. Recording [] as truth writes a mass delisting that is
# indistinguishable from a real one forever after.


def test_empty_snapshot_is_refused_outright(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot([], TS)


def test_refused_empty_snapshot_writes_nothing_at_all(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    before = _instruments_path(tmp_path).read_text(encoding="utf-8")

    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot([], TS + 1)

    assert _instruments_path(tmp_path).read_text(encoding="utf-8") == before
    assert tracker.load_last(TS + 1) == ["BTCUSDT", "ETHUSDT"]


def test_a_refused_snapshot_leaves_the_next_diff_correct(tmp_path: Path):
    """The failed poll must not later surface as a mass listing."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot([], TS + 1)

    events = tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS + 2)
    assert events == []


def test_symbols_that_are_not_strings_are_refused(tmp_path: Path):
    """A malformed parse must not poison the state file it would be written to."""
    tracker = UniverseTracker(tmp_path, "binance")
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot(["BTCUSDT", None], TS)
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot(["BTCUSDT", ""], TS)


def test_mass_delisting_is_refused(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(_many_symbols(20), TS)
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot(_many_symbols(5), TS + 1)
    assert tracker.load_last(TS + 1) == _many_symbols(20)


def test_delisting_below_the_threshold_is_recorded(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(_many_symbols(20), TS)
    events = tracker.record_snapshot(_many_symbols(15), TS + 1)
    assert {e.kind for e in events} == {"delisted"}
    assert len(events) == 5


def test_operator_can_widen_the_mass_delisting_threshold(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance", max_delisted_fraction=1.0)
    tracker.record_snapshot(_many_symbols(20), TS)
    events = tracker.record_snapshot(_many_symbols(1), TS + 1)
    assert len(events) == 19


def test_a_small_universe_is_exempt_from_the_fraction_guard(tmp_path: Path):
    """On a 3-symbol universe the fraction carries no signal."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["A", "B", "C"], TS)
    events = tracker.record_snapshot(["A"], TS + 1)
    assert {(e.symbol, e.kind) for e in events} == {("B", "delisted"), ("C", "delisted")}


def test_a_large_first_snapshot_is_not_refused(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    events = tracker.record_snapshot(_many_symbols(300), TS)
    assert len(events) == 300


def test_mass_listing_is_never_refused(tmp_path: Path):
    """Only delistings are the dangerous direction."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(_many_symbols(20), TS)
    events = tracker.record_snapshot(_many_symbols(200), TS + 1)
    assert len(events) == 180


# --- state file integrity ---------------------------------------------------


def test_corrupt_state_is_not_silently_treated_as_an_empty_universe(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    state = tmp_path / "universe" / "binance" / "last_snapshot.json"
    state.write_text('{"ts_ns": 1, "symbols":', encoding="utf-8")
    with pytest.raises(UnreadableUniverseState):
        tracker.load_last(TS)
    with pytest.raises(UnreadableUniverseState):
        tracker.record_snapshot(["BTCUSDT"], TS + 1)


@pytest.mark.parametrize("payload", [
    '{"ts_ns": 1, "symbols": "BTCUSDT"}',      # a string is iterable - set() would
                                              # split it into single characters
    '{"ts_ns": 1}',
    '{"symbols": ["BTCUSDT"]}',
    '["BTCUSDT"]',
    '{"ts_ns": "1", "symbols": ["BTCUSDT"]}',
    '{"ts_ns": 1, "symbols": [1, 2]}',
])
def test_structurally_wrong_state_is_refused(tmp_path: Path, payload: str):
    state = tmp_path / "universe" / "binance" / "last_snapshot.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(payload, encoding="utf-8")
    tracker = UniverseTracker(tmp_path, "binance")
    with pytest.raises(UnreadableUniverseState):
        tracker.load_last(TS)


def test_state_write_is_atomic_so_a_crash_keeps_the_previous_universe(tmp_path: Path, monkeypatch):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)

    import capture.universe_tracker as module

    def crash_instead_of_replacing(src, dst):
        raise OSError("simulated crash during the state write")

    monkeypatch.setattr(module.os, "replace", crash_instead_of_replacing)
    with pytest.raises(OSError):
        tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)
    monkeypatch.undo()

    assert tracker.load_last(TS + 1) == ["BTCUSDT", "ETHUSDT"]
    leftovers = sorted(p.name for p in (tmp_path / "universe" / "binance").iterdir()
                       if p.is_file())
    assert leftovers == ["last_snapshot.json"]


def test_the_event_record_is_durable_before_the_state_advances(tmp_path: Path, monkeypatch):
    """State must never claim a universe whose events were not written first.

    If the state advanced first and the append were then lost, the transition
    would be gone from the record and no later diff would ever re-emit it.
    """
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)

    import capture.universe_tracker as module
    trace: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def trace_fsync(fd):
        trace.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real_fsync(fd)

    def trace_replace(src, dst):
        trace.append("replace-state")
        return real_replace(src, dst)

    monkeypatch.setattr(module.os, "fsync", trace_fsync)
    monkeypatch.setattr(module.os, "replace", trace_replace)
    tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)
    monkeypatch.undo()

    instruments = str(_instruments_path(tmp_path))
    assert instruments in trace, "the event record was never fsynced"
    assert "replace-state" in trace
    assert trace.index(instruments) < trace.index("replace-state")


def test_a_failed_event_write_does_not_advance_the_state(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)

    # A plain file where the next day's directory must go: mkdir will fail.
    next_day = tmp_path / "universe" / "binance" / "2026-08-03"
    next_day.write_text("in the way", encoding="utf-8")
    later = TS + 86_400_000_000_000

    with pytest.raises(OSError):
        tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], later)
    assert tracker.load_last(later) == ["BTCUSDT", "ETHUSDT"]


# --- point-in-time semantics ------------------------------------------------


def test_load_last_does_not_return_a_snapshot_from_the_future(tmp_path: Path):
    """Membership known at T cannot include an observation made after T."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    assert tracker.load_last(TS - 1) == []
    assert tracker.load_last(TS) == ["BTCUSDT"]


def test_a_snapshot_older_than_the_recorded_state_is_refused(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    with pytest.raises(OutOfOrderSnapshot):
        tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS - 1)
    assert tracker.load_last(TS) == ["BTCUSDT", "ETHUSDT"]


def test_an_unchanged_universe_still_records_a_snapshot(tmp_path: Path):
    """Membership at every observation is the record; events alone are not."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT"], TS + 1)
    assert events == []

    snapshots = [r for r in _read_records(tmp_path) if r["kind"] == "snapshot"]
    assert [s["ts_ns"] for s in snapshots] == [TS, TS + 1]
    assert snapshots[1]["symbols"] == ["BTCUSDT"]


def test_duplicate_symbols_do_not_produce_duplicate_events(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    events = tracker.record_snapshot(["BTCUSDT", "BTCUSDT", "ETHUSDT"], TS)
    assert len(events) == 2
    assert tracker.load_last(TS) == ["BTCUSDT", "ETHUSDT"]


def test_persisted_event_lines_match_the_returned_events(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT", "SOLUSDT"], TS + 1)

    persisted = [r for r in _read_records(tmp_path) if r["kind"] in ("listed", "delisted")]
    assert [(r["symbol"], r["kind"], r["ts_ns"], r["venue"], r["detail"]) for r in persisted] == [
        ("BTCUSDT", "listed", TS, "binance", {}),
        ("ETHUSDT", "listed", TS, "binance", {}),
        ("SOLUSDT", "listed", TS + 1, "binance", {}),
        ("ETHUSDT", "delisted", TS + 1, "binance", {}),
    ]
    # what the caller was handed is exactly what went on disk
    assert [(e.symbol, e.kind, e.ts_ns, e.venue, e.detail) for e in events] == [
        (r["symbol"], r["kind"], r["ts_ns"], r["venue"], r["detail"]) for r in persisted[2:]
    ]


def test_each_venue_keeps_its_own_universe(tmp_path: Path):
    binance = UniverseTracker(tmp_path, "binance")
    hyperliquid = UniverseTracker(tmp_path, "hyperliquid")
    binance.record_snapshot(["BTCUSDT"], TS)
    events = hyperliquid.record_snapshot(["BTC"], TS)
    assert [(e.symbol, e.kind, e.venue) for e in events] == [("BTC", "listed", "hyperliquid")]
    assert binance.load_last(TS) == ["BTCUSDT"]


def test_the_date_partition_uses_utc_not_local_time(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TZ", "America/New_York")      # UTC-4 in August
    time.tzset()
    try:
        tracker = UniverseTracker(tmp_path, "binance")
        tracker.record_snapshot(["BTCUSDT"], 1785630600_000_000_000)   # 2026-08-02T00:30Z
        assert _instruments_path(tmp_path, date="2026-08-02").exists()
    finally:
        monkeypatch.undo()
        time.tzset()


def test_the_fraction_guard_applies_at_exactly_the_minimum_universe_size(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(_many_symbols(10), TS)
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot(_many_symbols(1), TS + 1)


def test_replaying_the_same_snapshot_is_accepted_and_changes_nothing(tmp_path: Path):
    """Re-running a poll is how a crash between the two writes is recovered."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    events = tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], TS)
    assert events == []
    assert tracker.load_last(TS) == ["BTCUSDT", "ETHUSDT"]


def test_delisting_exactly_at_the_threshold_is_still_recorded(tmp_path: Path):
    """max_delisted_fraction is the largest fraction allowed, not the smallest refused."""
    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(_many_symbols(20), TS)
    events = tracker.record_snapshot(_many_symbols(10), TS + 1)
    assert len(events) == 10
    assert {e.kind for e in events} == {"delisted"}


def test_a_malformed_venue_payload_cannot_reach_the_record(tmp_path: Path):
    """The hazard end to end: parse_instruments cannot tell [] from [].

    A Binance error document parses to exactly the same empty list a genuinely
    empty venue would, so the tracker - not the parser - has to be the thing
    that refuses.
    """
    from capture.venues.binance import BinanceVenue

    venue = BinanceVenue()
    tracker = UniverseTracker(tmp_path, venue.name)
    healthy = {"symbols": [{"symbol": s, "contractType": "PERPETUAL", "status": "TRADING"}
                           for s in _many_symbols(30)]}
    tracker.record_snapshot(venue.parse_instruments(healthy), TS)

    error_document = {"code": -1121, "msg": "Invalid symbol."}
    assert venue.parse_instruments(error_document) == []
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot(venue.parse_instruments(error_document), TS + 1)
    assert tracker.load_last(TS + 1) == _many_symbols(30)


def test_a_bare_string_is_not_mistaken_for_a_universe(tmp_path: Path):
    """Every character of "BTCUSDT" is a non-empty string; the list itself must be checked."""
    tracker = UniverseTracker(tmp_path, "binance")
    with pytest.raises(ImplausibleUniverseSnapshot):
        tracker.record_snapshot("BTCUSDT", TS)
    assert tracker.load_last(TS) == []


def test_a_non_integer_timestamp_cannot_poison_the_state_file(tmp_path: Path):
    tracker = UniverseTracker(tmp_path, "binance")
    with pytest.raises(TypeError):
        tracker.record_snapshot(["BTCUSDT"], TS / 1e9)
    assert not (tmp_path / "universe" / "binance" / "last_snapshot.json").exists()


def test_the_last_nanosecond_of_a_day_is_filed_under_that_day(tmp_path: Path):
    """The universe record is permanent, so a wrong date in it is permanent too.

    `ts_ns / 1e9` rounds: a present-day nanosecond timestamp is ~1.8e18, far past
    the 2^53 a float can hold exactly, so one nanosecond before UTC midnight
    divides to exactly midnight and the snapshot was filed under the NEXT day.
    A backtest reading membership by date then sees the change a day early -
    which is lookahead, the single thing this module exists to prevent.
    """
    ts_last_ns_of_aug_2 = 1785715199_999_999_999      # 2026-08-02T23:59:59.999999999Z

    tracker = UniverseTracker(tmp_path, "binance")
    tracker.record_snapshot(["BTCUSDT", "ETHUSDT"], ts_last_ns_of_aug_2)

    assert _instruments_path(tmp_path, date="2026-08-02").exists()
    assert not _instruments_path(tmp_path, date="2026-08-03").exists()
