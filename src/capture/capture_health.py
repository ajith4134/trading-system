"""Answers 'is capture healthy?' with evidence rather than assumption.

Runway is measured in days remaining, not percent used, because percent
thresholds mean nothing when the write rate changes. Local-only capture never
prunes - GCS offload is unproven and gated - so the archive grows
monotonically and the disk ladder is the only defence. Measured 2026-08-02:
96 GB total, ~91 GB free, growable to a hard ceiling of 400 GB, against an
estimated 1-2 GB/day compressed. That is roughly 45-90 days today and 200-400
days at the ceiling, so the escalation points (30 / 14 / 7 days) are computed
from the rate actually observed on disk rather than from a fixed constant.

**Alerts here are pull-only, and nobody is notified.** There is deliberately no
email, Slack or webhook in this module: an outbound channel carries credentials
and belongs to a later operations sub-project. The consequence has to be stated
plainly rather than discovered: until that channel exists, a silent capture
outage produces alerts that nobody reads. `health/alerts.ndjson` is only as
useful as the habit of opening it.

The second failure mode this module guards against is its own noise. A report
that floods is worse than one that is quiet, because it teaches its reader to
ignore the file - and this is not hypothetical here, the gap detector's alarm
rate on bursty streams is still under watch. So an alert that is already
recorded for the same venue, date and severity band is not repeated (see
`write_alerts`), and routine `observation_loss` gaps are counted in the report
but never alerted on. Only conditions that change the operator's decision get
a line.

A `silent_stream` event carries the same `observation_loss` severity and is
nevertheless alerted on, because it is not a gap in a working stream - it is a
stream that delivered nothing at all. That is the failure which produced an
archive holding order book and no trades, funding or liquidations, and it is
raised at most once per magnitude of the count, like every other counted alert.

Absence is treated as a failure, not as health. A venue-day that captured
nothing at all reports `capture_status="absent"` and alerts, because "no gaps
recorded" is exactly what a stream that never connected also looks like.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from math import isnan
from pathlib import Path
from stat import S_ISREG

from capture.capture_ledger import SEVERITY_CORRUPTING, read_all
from capture.raw_writer import RAW_SUFFIX, paths_for

WARN_DAYS = 30.0
ALERT_DAYS = 14.0
DECISION_DAYS = 7.0

STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"
STATUS_UNINITIALISED = "uninitialised"

ALERTS_FILENAME = "alerts.ndjson"

_SECONDS_PER_DAY = 86400


def compute_runway_days(free_bytes: int, daily_bytes: float) -> float:
    """Days of disk left at the current write rate.

    A rate of zero is infinite runway - nothing is being written, which the
    absence checks are responsible for, not this one. A negative or NaN rate is
    refused instead: both would flow straight through `classify_runway` as
    "ok", and a silent "ok" is the one answer this module must never invent.
    """
    if isnan(free_bytes) or isnan(daily_bytes):
        raise ValueError(f"runway is undefined for free_bytes={free_bytes!r}, "
                         f"daily_bytes={daily_bytes!r}")
    if daily_bytes < 0:
        raise ValueError(f"daily_bytes cannot be negative, got {daily_bytes!r}")
    if daily_bytes == 0:
        return float("inf")
    return free_bytes / daily_bytes


def classify_runway(days: float) -> str:
    """Which escalation band a runway falls in: ok/warn/alert/decision_point.

    Each boundary belongs to the worse band: landing exactly on 7 days is the
    decision point, not the band above it.
    """
    if days <= DECISION_DAYS:
        return "decision_point"
    if days <= ALERT_DAYS:
        return "alert"
    if days <= WARN_DAYS:
        return "warn"
    return "ok"


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_unless_utc_date(date: str) -> None:
    """A mistyped date reads an empty ledger and would report a clean day.

    The exact `YYYY-MM-DD` form is required, not merely something ISO can
    parse: `20260802` round-trips through `fromisoformat` but names no
    directory that capture ever writes.
    """
    if not isinstance(date, str) or dt.date.fromisoformat(date).isoformat() != date:
        raise ValueError(f"date must be a UTC YYYY-MM-DD date, got {date!r}")


def _venue_day_folder(root: Path, venue: str, date: str) -> Path:
    """The folder `RawWriter` writes this venue-day into."""
    raw_path, _ = paths_for(Path(root), venue, "any_stream", "any_symbol", f"{date}T00")
    return raw_path.parent


def _raw_root(root: Path) -> Path:
    """The folder every venue's capture lives under.

    Derived from `raw_writer.paths_for` rather than restated, because getting
    this wrong fails silently and permanently: this module would walk an empty
    tree, call every stream "uninitialised" and never raise an alert again.
    """
    return _venue_day_folder(root, "any_venue", "1970-01-01").parents[1]


def _has_captured_anything(root: Path) -> bool:
    """True once any venue has written a raw file under `root`.

    Cheap and lazy on purpose - it stops at the first file rather than walking
    the archive.
    """
    raw_root = _raw_root(root)
    if not raw_root.is_dir():
        return False
    return next(raw_root.rglob(f"*{RAW_SUFFIX}"), None) is not None


def _measure_raw_bytes_by_stream(folder: Path) -> dict[str, int]:
    """Bytes of captured data in one venue-day folder, split per stream-symbol.

    Only raw capture files count. Index files are derived, and a `.writing`
    marker is a sidecar holding a pid - counting bytes indiscriminately would
    let an hour that opened a stream and captured nothing report as present.

    Split rather than summed because a venue-day total answers the wrong
    question. One stream that is still flowing carries the whole folder over
    zero, so `capture_status` reads "present" while every sibling has been dead
    for days - the archive-of-order-book-with-no-trades failure, invisible in the
    one number that was reported. Verified 2026-08-02 over four days with three
    of four streams dead.

    `RawWriter` names files `{stream}_{symbol}_{hour}` and both stream and symbol
    may themselves contain underscores, so only the trailing hour can be split
    off unambiguously. The key is therefore the `stream_symbol` prefix exactly as
    it appears on disk, which is also how an operator finds the file.
    """
    if not folder.is_dir():
        return {}
    bytes_by_stream: dict[str, int] = {}
    for path in folder.glob(f"*{RAW_SUFFIX}"):
        try:
            size = path.stat().st_size
        except OSError:
            continue          # rotated or removed mid-scan
        stream_symbol = path.name[: -len(RAW_SUFFIX)].rsplit("_", 1)[0]
        bytes_by_stream[stream_symbol] = bytes_by_stream.get(stream_symbol, 0) + size
    return bytes_by_stream


def measure_daily_bytes(root: Path, days: int = 7) -> float:
    """Bytes written per day recently, measured from the archive itself.

    Everything under `raw/` counts, including index and quarantined files: they
    occupy the disk whose runway is being estimated.

    The divisor is the number of days actually observed, not the width of the
    window. Two days of history divided by a 7-day window reports 2/7ths of the
    true write rate and multiplies the runway estimate by 3.5 - reporting
    healthy for a disk that is not. It is floored at one day in the other
    direction, so an archive an hour old cannot report a 24x inflated rate and
    alarm on its first hour of life.
    """
    if days <= 0:
        raise ValueError(f"days must be a positive number of days, got {days!r}")
    raw_root = _raw_root(root)
    if not raw_root.is_dir():
        return 0.0

    now = time.time()
    cutoff = now - days * _SECONDS_PER_DAY
    total = 0
    earliest = now
    for path in raw_root.rglob("*"):
        try:
            info = path.stat()
        except OSError:
            # Files rotate, quarantine and vanish under a live capture. A scan
            # that dies on one of them takes the whole health check down.
            continue
        if not S_ISREG(info.st_mode) or info.st_mtime < cutoff:
            continue
        total += info.st_size
        earliest = min(earliest, info.st_mtime)

    if total == 0:
        return 0.0
    observed_days = min(float(days), max(1.0, (now - earliest) / _SECONDS_PER_DAY))
    return total / observed_days


def build_report(root: Path, venue: str, date: str,
                 free_bytes: int, daily_bytes: float) -> dict:
    """One venue-day's capture health: what the ledger recorded, whether
    anything was captured at all, and how many days of disk are left.

    Reads only - the report is written nowhere. `write_alerts` is the only
    function here that touches the filesystem.
    """
    _refuse_unless_utc_date(date)
    events = read_all(root, venue, date)

    gaps = {"corrupting": 0, "observation_loss": 0, "info": 0}
    corrupting_non_gap = 0
    # Streams the recorder subscribed to and heard nothing from. Counted apart
    # from gaps: a gap is a hole in a stream that is working, this is a stream
    # that is not. Deliberately not folded into `gaps[observation_loss]`, which
    # is the routine bucket that is never alerted on.
    silent_streams = 0
    silent_stream_names: set[str] = set()
    for event in events:
        # A ledger line can be valid JSON and still carry a mistyped severity,
        # which `read_all` has no reason to reject. Bucketing it keeps one odd
        # line from raising and leaving the whole venue-day unreported - which
        # is indistinguishable from healthy to anything downstream.
        severity = event.severity if isinstance(event.severity, str) else "unknown"
        if event.kind == "gap":
            gaps[severity] = gaps.get(severity, 0) + 1
        elif event.kind == "silent_stream":
            silent_streams += 1
            if isinstance(event.stream, str):
                silent_stream_names.add(event.stream)
        elif severity == SEVERITY_CORRUPTING:
            # `unwritable_stream_total` is the worst thing in the ledger - a
            # stream being dropped entirely - and it is not a gap.
            corrupting_non_gap += 1

    raw_bytes_by_stream = _measure_raw_bytes_by_stream(
        _venue_day_folder(root, venue, date))
    raw_data_bytes = sum(raw_bytes_by_stream.values())
    if raw_data_bytes > 0:
        capture_status = STATUS_PRESENT
    elif _has_captured_anything(root):
        capture_status = STATUS_ABSENT
    else:
        # Nothing anywhere has ever been captured. "This stream stopped" cannot
        # be claimed before the system has run at all, and claiming it would
        # alert on every fresh install. The cost of this choice is honest: the
        # very first day of capture cannot raise an absence alert.
        capture_status = STATUS_UNINITIALISED

    runway_days = compute_runway_days(free_bytes, daily_bytes)
    return {
        "venue": venue,
        "date": date,
        "generated_at": _now_utc_iso(),
        "events_total": len(events),
        "damaged_ledger_lines": len(events.damaged),
        "gaps": gaps,
        "corrupting_non_gap": corrupting_non_gap,
        "silent_streams": silent_streams,
        "silent_stream_names": sorted(silent_stream_names),
        "raw_data_bytes": raw_data_bytes,
        # Per stream-symbol, so a live stream cannot report presence on behalf of
        # a dead sibling - see `_measure_raw_bytes_by_stream`. A stream that
        # wrote nothing that day has no key at all.
        "raw_bytes_by_stream": dict(sorted(raw_bytes_by_stream.items())),
        "capture_status": capture_status,
        "free_bytes": free_bytes,
        "daily_bytes": daily_bytes,
        "runway_days": runway_days,
        "runway_status": classify_runway(runway_days),
    }


def _order_of_magnitude(count: int) -> int:
    """Decade band of a count: 1-9 -> 0, 10-99 -> 1, 100-999 -> 2.

    Repeats collapse, but a tenfold worsening is a different situation and has
    to reach the operator.
    """
    return len(str(max(int(count), 1))) - 1


def _build_alerts(report: dict) -> list[dict]:
    """The alert records a report justifies, worst-first.

    Routine `observation_loss` gaps are deliberately absent: they are the
    expected case on a bursty stream, and alerting on them is how a health file
    becomes unread.
    """
    alerts: list[dict] = []

    def add(reason: str, band: str, **detail: object) -> None:
        alerts.append({
            "reason": reason,
            "venue": report["venue"],
            "date": report["date"],
            "ts": _now_utc_iso(),
            # Written into the record so the file itself is the dedup state -
            # there is nothing else to keep in sync with it.
            "dedup_key": f"{report['venue']}|{report['date']}|{reason}|{band}",
            **detail,
        })

    if report["runway_status"] != "ok":
        # The band is already in the reason, so an escalation is a new alert.
        add(f"runway_{report['runway_status']}", "",
            runway_days=report["runway_days"], free_bytes=report["free_bytes"],
            daily_bytes=report["daily_bytes"])
    if report["capture_status"] == STATUS_ABSENT:
        add("capture_absent", "")
    if report["gaps"].get("corrupting", 0) > 0:
        count = report["gaps"]["corrupting"]
        add("corrupting_gaps", str(_order_of_magnitude(count)), count=count)
    if report.get("corrupting_non_gap", 0) > 0:
        count = report["corrupting_non_gap"]
        add("corrupting_non_gap_events", str(_order_of_magnitude(count)), count=count)
    if report.get("silent_streams", 0) > 0:
        # Not a routine observation_loss: a subscribed stream delivering nothing
        # changes what the operator does, and it is the failure that produced an
        # archive of order book with no trades, funding or liquidations in it.
        # The names travel with the alert so the file answers "which ones?"
        # without anyone opening the ledger.
        count = report["silent_streams"]
        add("silent_streams", str(_order_of_magnitude(count)), count=count,
            streams=report.get("silent_stream_names", []))
    if report.get("damaged_ledger_lines", 0) > 0:
        count = report["damaged_ledger_lines"]
        add("ledger_damaged", str(_order_of_magnitude(count)), count=count)
    return alerts


def _read_recorded_dedup_keys(path: Path) -> set[str]:
    """Dedup keys already in the alert file.

    A line that cannot be parsed is skipped rather than fatal, exactly as in
    the ledger: the worst case is one alert repeating, and the alternative is a
    torn line silencing every future alert.
    """
    keys: set[str] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            key = record.get("dedup_key") if isinstance(record, dict) else None
            if isinstance(key, str):
                keys.add(key)
    return keys


def _is_last_line_torn(path: Path) -> bool:
    """True when the file ends mid-line, as an interrupted append leaves it."""
    try:
        with open(path, "rb") as fh:
            if fh.seek(0, os.SEEK_END) == 0:
                return False
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except OSError:
        return False


def write_alerts(root: Path, report: dict) -> int:
    """Append this report's alerts to `health/alerts.ndjson`, returning how
    many were newly recorded.

    An alert already recorded for the same venue, date, reason and severity
    band is not repeated. A health report is expected to run on a schedule, and
    without this an unresolved condition writes one identical line per run
    until the file is worthless. What is deliberately not collapsed: a worse
    runway band, and a count that has grown by an order of magnitude - both are
    a different situation, not the same one restated.

    The count returned is what reached the disk, never what the report merely
    justified, so a caller cannot be told about an alert that was suppressed or
    about one whose write failed. Records go down in a single fsynced write; if
    that write is interrupted the file may end mid-line, and the next call
    starts a fresh line rather than gluing a good record onto a torn one.

    Dedup state lives in the file itself and is read, not locked, so two
    processes reporting at the same instant can both decide an alert is new and
    write it twice. That is the failure worth having: the alternative, a lock,
    would let one stuck process stop the other from raising an alarm at all.
    """
    alerts = _build_alerts(report)
    if not alerts:
        return 0

    path = Path(root) / "health" / ALERTS_FILENAME
    recorded = _read_recorded_dedup_keys(path)
    fresh = []
    for alert in alerts:
        if alert["dedup_key"] in recorded:
            continue
        recorded.add(alert["dedup_key"])          # never twice in one batch either
        fresh.append(alert)
    if not fresh:
        return 0

    blob = "".join(json.dumps(alert, separators=(",", ":"), sort_keys=True) + "\n"
                   for alert in fresh)
    if _is_last_line_torn(path):
        blob = "\n" + blob

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(blob)          # one write call: no interleaving with a concurrent appender
        fh.flush()
        os.fsync(fh.fileno())
    return len(fresh)
