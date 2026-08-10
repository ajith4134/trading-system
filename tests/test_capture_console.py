"""The operator's view of whether capture is alive, and it had no test at all.

Its axis verdict failed depth on 2026-08-09 for exactly that: nothing
demonstrated it renders correctly under any condition, and an operator view that
is wrong when the system is wrong is Rule 8 in miniature. During a wait for
observed history this is the tool someone reaches for to check the wait is still
accumulating, so "probably fine" is not good enough.

The tests below are about the failing states. A console that cannot render a
dead stream red has not been tested against the thing it exists for.
"""
import time
from pathlib import Path

import pytest

from capture.capture_console import (
    STALE_AFTER_SECONDS, _utc_today, collect_capture_status,
    collect_stream_freshness, find_quarantined_hours, measure_archive_age_seconds,
    read_recent_restarts, render_status_html, render_status_text,
    write_console_page,
)
from capture.raw_writer import RAW_SUFFIX

# The console only ever looks at the UTC day it is running on, so a fixture
# written under any other day is invisible to it. This was hardcoded to
# "2026-08-09" when these tests were written, and four of them passed for
# exactly one day: on 2026-08-10 the console correctly reported "No files
# today" about a fixture filed under yesterday, and the tests read that as the
# renderer being broken. Reading the date off the same clock the console reads
# is the invariant those tests meant to assert.
DATE = _utc_today()


def _hour_file(root: Path, venue: str, stream: str, symbol: str,
               age_seconds: float = 0.0, date: str = DATE) -> Path:
    folder = root / "raw" / venue / date
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stream}_{symbol}_{date}T00{RAW_SUFFIX}"
    path.write_bytes(b"x")
    stamp = time.time() - age_seconds
    import os
    os.utime(path, (stamp, stamp))
    return path


# --------------------------------------------------------------------------
# freshness - the one question the console exists to answer
# --------------------------------------------------------------------------

def test_a_stream_that_stopped_is_marked_stale(tmp_path):
    """The condition the console exists for. Freshness comes from file mtimes
    rather than the ledger, deliberately: a stream silently doing nothing writes
    no events, so the ledger cannot show it."""
    _hour_file(tmp_path, "binance", "trade", "DEAD", age_seconds=STALE_AFTER_SECONDS + 60)
    _hour_file(tmp_path, "binance", "trade", "LIVE", age_seconds=1)

    rows = {r["symbol"]: r for r in collect_stream_freshness(tmp_path, "binance", DATE)}

    assert rows["DEAD"]["is_stale"] is True
    assert rows["LIVE"]["is_stale"] is False


def test_a_stale_stream_is_visible_in_the_rendered_text(tmp_path):
    """A row that is stale in the data and not on the screen is worse than no
    console - it reassures exactly when attention is required."""
    _hour_file(tmp_path, "binance", "trade", "DEAD", age_seconds=STALE_AFTER_SECONDS + 60)
    status = collect_capture_status(tmp_path, ["binance"])

    text = render_status_text(status)

    assert "STALE" in text
    assert "DEAD" in text


def test_a_stale_stream_is_visible_in_the_rendered_html(tmp_path):
    _hour_file(tmp_path, "binance", "trade", "DEAD", age_seconds=STALE_AFTER_SECONDS + 60)
    status = collect_capture_status(tmp_path, ["binance"])

    page = render_status_html(status)

    assert "DEAD" in page
    assert "stale" in page.lower()


def test_a_name_that_could_break_the_page_is_escaped(tmp_path):
    """Symbols come off the wire. The recorder path-encodes them before they
    reach a filename, but the console renders whatever the archive hands it and
    must not build markup out of it.

    Driven through the real collector rather than a hand-built status dict, so
    the escaping is tested on the path that actually runs.

    The name is filename-legal but markup-dangerous, which is precisely the
    residual case: `_safe_path_token` already refuses anything carrying a path
    separator, so a `</script>` can never reach a filename - but `<b` and `&`
    can, and the console must still not build markup out of them.
    """
    _hour_file(tmp_path, "binance", "trade", "A<b&C", age_seconds=1)

    page = render_status_html(collect_capture_status(tmp_path, ["binance"]))

    assert "A<b&C" not in page, "rendered a symbol as markup"
    assert "A&lt;b&amp;C" in page


# --------------------------------------------------------------------------
# the other failing states
# --------------------------------------------------------------------------

def test_a_quarantined_hour_is_reported(tmp_path):
    """An hour the recorder could not write is the loudest thing on the box and
    must not need a log grep to find."""
    from capture.capture_ledger import CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING

    # The ledger partitions by the event's own timestamp, so the event has to
    # fall on the day being read - a mismatch reads as "no quarantine", which is
    # the reassuring answer and the wrong one.
    import datetime as _dt
    noon = int(_dt.datetime.fromisoformat(f"{DATE}T12:00:00+00:00").timestamp()) * 10**9

    ledger = CaptureLedger(tmp_path, "binance")
    ledger.record(LedgerEvent(
        ts_ns=noon, venue="binance", stream="trade",
        kind="unwritable_stream", severity=SEVERITY_CORRUPTING,
        detail={"symbol": "BROKEN", "hour": f"{DATE}T00"}))
    ledger.close()

    found = find_quarantined_hours(tmp_path, "binance", DATE)
    assert any("BROKEN" in str(entry) for entry in found), found


def test_recent_restarts_are_surfaced(tmp_path):
    """A recorder restarting every two minutes looks identical to a healthy one
    in a snapshot of file ages."""
    state = tmp_path / "supervisor"
    state.mkdir(parents=True)
    (state / "binance.restarts.ndjson").write_text(
        "\n".join('{"event":"recorder_exited","exit_code":1}' for _ in range(9)) + "\n",
        encoding="utf-8")

    lines = read_recent_restarts(tmp_path, "binance", limit=5)
    assert len(lines) == 5


def test_an_empty_archive_renders_rather_than_raising(tmp_path):
    """First boot, and the console is exactly what an operator opens then."""
    status = collect_capture_status(tmp_path, ["binance"])

    assert render_status_text(status)
    assert render_status_html(status)
    assert measure_archive_age_seconds(tmp_path) == 0.0


def test_the_page_is_written_whole(tmp_path):
    """A reader either gets the previous page or the new one - a half-written
    console during an incident is the worst possible moment for one."""
    _hour_file(tmp_path, "binance", "trade", "BTCUSDT", age_seconds=1)
    destination = tmp_path / "console" / "capture.html"

    write_console_page(tmp_path, ["binance"], destination)

    assert destination.exists()
    assert "BTCUSDT" in destination.read_text(encoding="utf-8")


def test_the_page_refreshes_itself_without_a_server(tmp_path):
    """The rendering choice this module argues for: a plain file with a
    meta-refresh needs no server process that could itself die and take the view
    with it."""
    status = collect_capture_status(tmp_path, ["binance"])
    assert "http-equiv" in render_status_html(status).lower()
