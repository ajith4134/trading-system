"""The driver that makes the halt registry a mechanism rather than a library.

`VenueHaltRegistry` was complete and uncalled from the day it was written -
`observe()` and `assess_venue()` had zero callers anywhere in src/ - so no
degradation could halt a venue, and the status wall rendered "auto-halt armed"
over the top of it. Found 2026-08-09 by asking what called it, not by anything
failing.
"""
import json
from pathlib import Path

import pytest

from capture.capture_ledger import CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING
from ops.venue_halt import HALT_CORRUPTING, VenueHaltRegistry
from ops.venue_health_watch import capture_dates_by_venue, observe_all_venues

DATE = "2026-08-09"
NOON_NS = 1786276800_000_000_000   # 2026-08-09T12:00:00Z, the same UTC day as DATE


def _captured(root: Path, venue: str, date: str = DATE) -> None:
    """A venue-day that exists on disk, with one trade file so it is not empty."""
    folder = root / "raw" / venue / date
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"trade_BTCUSDT_{date}T00.ndjson.zst").write_bytes(b"\x28\xb5\x2f\xfd\x00\x00")


def _corrupting_event(root: Path, venue: str, date: str = DATE) -> None:
    ledger = CaptureLedger(root, venue)
    ledger.record(LedgerEvent(
        ts_ns=NOON_NS, venue=venue, stream="trade", kind="unwritable_stream_total",
        severity=SEVERITY_CORRUPTING, detail={"symbol": "BTCUSDT"}))
    ledger.close()


def test_every_captured_venue_is_observed(tmp_path):
    """The whole point. Before this, the registry had never been written to by
    anything running."""
    for venue in ("binance", "hyperliquid"):
        _captured(tmp_path, venue)
    registry = VenueHaltRegistry(tmp_path / "ops", clock_ns=lambda: NOON_NS)

    result = observe_all_venues(tmp_path, registry)

    assert sorted(result["observed"]) == ["binance", "hyperliquid"]
    assert registry.last_observed_ns("binance") == NOON_NS
    assert registry.last_observed_ns("hyperliquid") == NOON_NS


def test_the_observation_survives_into_a_fresh_registry(tmp_path):
    """The wall reads this through its own `VenueHaltRegistry`, in another
    process. An observation that lived only in memory would leave the tile
    reading NOT ARMED while the loop ran happily."""
    _captured(tmp_path, "binance")
    observe_all_venues(tmp_path, VenueHaltRegistry(tmp_path / "ops",
                                                   clock_ns=lambda: NOON_NS))

    reader = VenueHaltRegistry(tmp_path / "ops")
    assert reader.last_observed_ns("binance") == NOON_NS
    assert reader.is_tradeable("binance")


def test_a_corrupting_venue_is_halted(tmp_path):
    """The mechanism firing, end to end from the ledger rather than from a
    hand-made report - if `build_report` and `assess_venue` disagree about the
    field name, a report built by hand in a test would not notice."""
    _captured(tmp_path, "binance")
    _corrupting_event(tmp_path, "binance")
    registry = VenueHaltRegistry(tmp_path / "ops", clock_ns=lambda: NOON_NS)

    result = observe_all_venues(tmp_path, registry)

    assert result["halted"] == ["binance"]
    assert not registry.is_tradeable("binance")
    assert registry.halt_reason("binance") == HALT_CORRUPTING


def test_silence_alone_does_not_halt_a_venue(tmp_path):
    """Measured on this host the day the loop was written: 0 corrupting events
    against 6,987 silence events on binance-spot. Halting on silence would have
    stopped all three venues on their first observed pass, and the count is
    inflated by thin tail symbols and by a feed the venue withholds."""
    _captured(tmp_path, "binance-spot")
    ledger = CaptureLedger(tmp_path, "binance-spot")
    for i in range(50):
        ledger.record(LedgerEvent(
            ts_ns=NOON_NS + i, venue="binance-spot", stream=f"trade{i}",
            kind="silent_stream", severity="observation_loss", detail={}))
    ledger.close()

    result = observe_all_venues(tmp_path, VenueHaltRegistry(tmp_path / "ops",
                                                            clock_ns=lambda: NOON_NS))

    assert result["halted"] == []
    assert result["observed"]["binance-spot"]["silent_streams"] > 0


def test_the_newest_day_is_the_one_observed(tmp_path):
    """A halt decided from a week-old report is a halt about last week."""
    _captured(tmp_path, "binance", "2026-08-01")
    _captured(tmp_path, "binance", DATE)

    result = observe_all_venues(tmp_path, VenueHaltRegistry(tmp_path / "ops",
                                                            clock_ns=lambda: NOON_NS))

    assert result["observed"]["binance"]["date"] == DATE


def test_a_venue_whose_report_will_not_build_is_named_not_dropped(tmp_path, monkeypatch):
    """Skipping it silently leaves the registry serving its last verdict with
    nothing anywhere saying the input stopped arriving - which is the same
    failure as never having called it."""
    _captured(tmp_path, "binance")
    _captured(tmp_path, "hyperliquid")

    import ops.venue_health_watch as watch
    real = watch.build_report

    def explode(root, venue, date, free_bytes, daily_bytes):
        if venue == "binance":
            raise ValueError("a rotted ledger line")
        return real(root, venue, date, free_bytes, daily_bytes)

    monkeypatch.setattr(watch, "build_report", explode)
    result = observe_all_venues(tmp_path, VenueHaltRegistry(tmp_path / "ops",
                                                            clock_ns=lambda: NOON_NS))

    assert "binance" in result["skipped"]
    assert "ValueError" in result["skipped"]["binance"]
    # And the other venue is still observed - one bad report must not stop the pass.
    assert "hyperliquid" in result["observed"]


def test_an_empty_capture_root_observes_nothing_rather_than_raising(tmp_path):
    """First boot. Nothing captured is not an error, and it must not mark venues
    tradeable either - there are none."""
    assert observe_all_venues(tmp_path) == {"observed": {}, "skipped": {}, "halted": []}


def test_dates_are_read_newest_last(tmp_path):
    _captured(tmp_path, "binance", "2026-08-01")
    _captured(tmp_path, "binance", "2026-08-09")
    assert capture_dates_by_venue(tmp_path)["binance"] == ["2026-08-01", "2026-08-09"]


def test_a_failed_pass_exits_nonzero_so_the_supervisor_records_it(tmp_path, monkeypatch, capsys):
    """A venue that could not be reported means the registry went an interval
    with no input, and only the exit code says so - the JSON goes to a log
    nobody watches minute to minute."""
    _captured(tmp_path, "binance")
    import ops.venue_health_watch as watch
    monkeypatch.setattr(watch, "build_report",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))

    code = watch.main(["--capture-root", str(tmp_path)])

    assert code == 1
    assert "binance" in json.loads(capsys.readouterr().out)["skipped"]


def test_a_healthy_pass_exits_zero(tmp_path, capsys):
    """A halted venue is the mechanism working, not this process failing."""
    _captured(tmp_path, "binance")
    _corrupting_event(tmp_path, "binance")

    import ops.venue_health_watch as watch
    code = watch.main(["--capture-root", str(tmp_path)])

    assert code == 0, "a halt was treated as a crash"
    assert json.loads(capsys.readouterr().out)["halted"] == ["binance"]
