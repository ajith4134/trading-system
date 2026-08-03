"""Measures what is actually true on this machine, feature by feature.

Rule 8: a display shows measured state, never asserted state. Everything here
returns what a probe found, and a feature with no probe is NOT_BUILT by
construction rather than by omission - there is no code path that can render a
feature healthy without a measurement behind it.

`ProbeResult.proof` carries where the answer came from, and the renderer shows
it. A status whose provenance cannot be named is not a status.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from capture.capture_health import (
    build_report, classify_runway, compute_runway_days, measure_daily_bytes,
)
from statuswall.catalogue import Feature, normalise_key

# Ordered worst-first: this is the order the wall sorts by, so what is broken
# arrives at the top of the board without anyone scrolling.
FAILING = "failing"
DEGRADED = "degraded"
STOPPED = "stopped"
PARTIAL = "partial"
OK = "ok"
BUILT = "built"
NOT_BUILT = "not_built"

SEVERITY_ORDER = [FAILING, DEGRADED, STOPPED, PARTIAL, OK, BUILT, NOT_BUILT]

STATE_LABEL = {
    FAILING: "FAILING",
    DEGRADED: "DEGRADED",
    STOPPED: "STOPPED",
    PARTIAL: "PARTIAL",
    OK: "OK",
    BUILT: "BUILT",
    NOT_BUILT: "NOT BUILT",
}


@dataclass(frozen=True)
class ProbeResult:
    state: str
    detail: str
    proof: str


@dataclass(frozen=True)
class SystemFacts:
    """One measurement pass over the machine, shared by every probe.

    Taken once so that every tile on a board describes the same instant. Probing
    per feature would let the top of the wall disagree with the bottom about
    whether capture was running.
    """
    measured_at: str
    capture_root: Path
    repo_root: Path
    capture_running: bool
    capture_pids: list[int]
    latest_capture_date: str | None
    hours_since_capture: float | None
    venues: list[str]
    reports: dict[str, dict]
    free_bytes: int
    daily_bytes: float
    runway_days: float
    runway_status: str
    restart_counts: dict[str, int]


def _read_capture_pids() -> list[int]:
    """PIDs of running capture processes, empty when none are.

    `pgrep -f` matches the full command line because the capture runs as
    `python -m capture.cli`, whose process name is just `python`.
    """
    try:
        done = subprocess.run(
            ["pgrep", "-f", "capture.cli"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in done.stdout.split() if line.strip().isdigit()]


def _capture_dates(capture_root: Path) -> dict[str, list[str]]:
    """UTC dates present per venue under raw/, newest last."""
    raw = capture_root / "raw"
    if not raw.is_dir():
        return {}
    found: dict[str, list[str]] = {}
    for venue_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        dates = sorted(p.name for p in venue_dir.iterdir() if p.is_dir())
        if dates:
            found[venue_dir.name] = dates
    return found


def _count_restarts(capture_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    supervisor = capture_root / "supervisor"
    if not supervisor.is_dir():
        return counts
    for path in sorted(supervisor.glob("*.restarts.ndjson")):
        venue = path.name.split(".", 1)[0]
        try:
            counts[venue] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            continue
    return counts


def measure_system(capture_root: Path, repo_root: Path, now: dt.datetime) -> SystemFacts:
    """Take one pass over the machine. Every number below was read, not assumed."""
    dates_by_venue = _capture_dates(capture_root)
    venues = sorted(dates_by_venue)
    latest = max((d[-1] for d in dates_by_venue.values()), default=None)

    hours_since = None
    if latest is not None:
        # Dated folders are UTC days, so the youngest possible data in the newest
        # folder is its final second. Ageing from the day's end rather than its
        # start avoids reporting a stream that stopped minutes ago as a day stale.
        end_of_day = dt.datetime.fromisoformat(latest).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
        hours_since = max(0.0, (now - end_of_day).total_seconds() / 3600.0)

    free_bytes = shutil.disk_usage(capture_root).free if capture_root.exists() else 0
    daily_bytes = measure_daily_bytes(capture_root) if capture_root.exists() else 0.0
    runway = compute_runway_days(free_bytes, daily_bytes)

    reports: dict[str, dict] = {}
    for venue, dates in dates_by_venue.items():
        reports[venue] = build_report(capture_root, venue, dates[-1], free_bytes, daily_bytes)

    pids = _read_capture_pids()
    return SystemFacts(
        measured_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        capture_root=capture_root,
        repo_root=repo_root,
        capture_running=bool(pids),
        capture_pids=pids,
        latest_capture_date=latest,
        hours_since_capture=hours_since,
        venues=venues,
        reports=reports,
        free_bytes=free_bytes,
        daily_bytes=daily_bytes,
        runway_days=runway,
        runway_status=classify_runway(runway),
        restart_counts=_count_restarts(capture_root),
    )


# ---------------------------------------------------------------------------
# Probes. Each returns what it measured and names where the answer came from.
# ---------------------------------------------------------------------------

def _stream_bytes(facts: SystemFacts, prefixes: tuple[str, ...]) -> dict[str, int]:
    total: dict[str, int] = {}
    for venue, report in facts.reports.items():
        for stream_symbol, size in report.get("raw_bytes_by_stream", {}).items():
            if stream_symbol.split("_", 1)[0] in prefixes:
                total[f"{venue}/{stream_symbol}"] = size
    return total


def _capture_liveness(facts: SystemFacts) -> tuple[str, str]:
    """How a capture-backed feature should read given whether capture is running."""
    if facts.capture_running:
        return OK, "capture process live"
    if facts.hours_since_capture is None:
        return NOT_BUILT, "no capture data on disk"
    return STOPPED, f"no capture process; newest data {facts.hours_since_capture:.0f}h old"


def probe_trade_tape(facts: SystemFacts) -> ProbeResult:
    sizes = _stream_bytes(facts, ("trade", "trades"))
    if not sizes:
        return ProbeResult(NOT_BUILT, "no trade streams on disk", "capture/raw")
    state, why = _capture_liveness(facts)
    total_mb = sum(sizes.values()) / 1e6
    return ProbeResult(
        PARTIAL if state == OK else state,
        # Only what was measured. The clause that used to follow - "OHLCV bar
        # building not implemented" - was hand-typed, and it went on reading as
        # true after the store held bars and the adjacent tile measured them. An
        # asserted claim cannot go stale loudly, which is the whole reason Rule 8
        # allows a tile to show nothing but measured state.
        f"trade tape captured on {len(facts.venues)} venues ({total_mb:.0f} MB). {why}",
        f"raw_bytes_by_stream over {len(sizes)} trade streams",
    )


def probe_l2_depth(facts: SystemFacts) -> ProbeResult:
    sizes = _stream_bytes(facts, ("depth", "l2Book"))
    if not sizes:
        return ProbeResult(NOT_BUILT, "no depth streams on disk", "capture/raw")
    state, why = _capture_liveness(facts)
    total_mb = sum(sizes.values()) / 1e6
    return ProbeResult(state, f"{len(sizes)} depth streams, {total_mb:.0f} MB. {why}",
                       "raw_bytes_by_stream over depth streams")


def _silent_stream_probe(facts: SystemFacts, stream: str, label: str) -> ProbeResult:
    """A subscribed stream that never spoke is a failure, not an absence.

    The distinction matters: a stream nobody subscribed to is simply not built,
    while a stream that was subscribed and delivered nothing is wired and dead -
    and it looks identical on disk to a quiet market unless the ledger is read.
    """
    silent_in: list[str] = []
    for venue, report in facts.reports.items():
        if stream in report.get("silent_stream_names", []):
            silent_in.append(venue)
    if silent_in:
        return ProbeResult(
            FAILING,
            f"'{stream}' subscribed on {', '.join(silent_in)} and never delivered a frame",
            "capture_health silent_stream_names",
        )
    if _stream_bytes(facts, (stream,)):
        state, why = _capture_liveness(facts)
        return ProbeResult(state, f"{label} streaming. {why}", "raw_bytes_by_stream")
    return ProbeResult(NOT_BUILT, f"{label} not subscribed on any venue", "capture_health")


def probe_liquidation_feed(facts: SystemFacts) -> ProbeResult:
    return _silent_stream_probe(facts, "forceOrder", "liquidation feed")


def probe_mark_price(facts: SystemFacts) -> ProbeResult:
    return _silent_stream_probe(facts, "markPrice", "mark price")


def probe_gap_detection(facts: SystemFacts) -> ProbeResult:
    observed = sum(r.get("gaps", {}).get("observation_loss", 0) for r in facts.reports.values())
    corrupting = sum(r.get("gaps", {}).get("corrupting", 0) for r in facts.reports.values())
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no capture to detect gaps in", "capture/ledger")
    return ProbeResult(
        PARTIAL,
        f"detection live: {observed} observation-loss and {corrupting} corrupting gaps recorded. "
        f"Provenance-flagged backfill not implemented",
        "capture_health gaps + src/capture/sequencing.py",
    )


def probe_sequence_gap_detection(facts: SystemFacts) -> ProbeResult:
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no capture on disk", "capture/ledger")
    total = sum(r.get("gaps", {}).get("observation_loss", 0) for r in facts.reports.values())
    state, why = _capture_liveness(facts)
    return ProbeResult(
        DEGRADED if state == OK and total else state,
        f"{total} sequence gaps recorded across {len(facts.reports)} venues. {why}",
        "src/capture/sequencing.py, capture_health gaps",
    )


def probe_resource_watchdog(facts: SystemFacts) -> ProbeResult:
    free_gb = facts.free_bytes / 1e9
    daily_mb = facts.daily_bytes / 1e6
    state = OK if facts.runway_status == "ok" else DEGRADED
    return ProbeResult(
        state,
        f"disk runway {facts.runway_days:.0f} days ({free_gb:.0f} GB free, "
        f"{daily_mb:.0f} MB/day). Memory and CPU watchdogs not implemented",
        "capture_health.compute_runway_days",
    )


def probe_venue_health(facts: SystemFacts) -> ProbeResult:
    alarms = []
    for venue, report in facts.reports.items():
        if report.get("silent_streams"):
            alarms.append(f"{venue}: {report['silent_streams']} silence events")
        if report.get("corrupting_non_gap"):
            alarms.append(f"{venue}: {report['corrupting_non_gap']} corrupting events")
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no venue data", "capture_health")
    detail = "; ".join(alarms) if alarms else "no venue alarms"
    return ProbeResult(
        DEGRADED if alarms else PARTIAL,
        f"health monitoring live ({detail}). Auto-halt on degradation not implemented",
        "src/capture/capture_health.py build_report",
    )


def probe_data_quality_score(facts: SystemFacts) -> ProbeResult:
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no feeds measured", "capture_health")
    return ProbeResult(
        PARTIAL,
        f"per-stream byte volume and silence tracked for {len(facts.reports)} venues; "
        f"no composite quality score computed",
        "capture_health raw_bytes_by_stream, silent_stream_names",
    )


def probe_capture_supervisor(facts: SystemFacts) -> ProbeResult:
    if not facts.restart_counts:
        return ProbeResult(NOT_BUILT, "no supervisor restart records", "capture/supervisor")
    total = sum(facts.restart_counts.values())
    breakdown = ", ".join(f"{v}: {n}" for v, n in sorted(facts.restart_counts.items()))
    state, why = _capture_liveness(facts)
    return ProbeResult(state, f"{total} recorded restarts ({breakdown}). {why}",
                       "capture/supervisor/*.restarts.ndjson")


def _store_root(facts: SystemFacts) -> Path:
    return facts.capture_root / "store"


def _bar_datasets(facts: SystemFacts) -> list[Path]:
    root = _store_root(facts)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("bars_"))


def probe_bitemporal_store(facts: SystemFacts) -> ProbeResult:
    """Reports on the store by reading it, not by checking a path exists.

    Row and part counts come from `ClockGatedReader.read_as_of`, the same call
    every consumer makes - a probe that stat()ed the directory instead would go
    on reporting OK against a store whose Parquet files had rotted or emptied.
    """
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store built from the archive yet",
                           "capture/store")
    parts = sum(1 for dataset in datasets for _ in dataset.rglob("*.parquet"))
    from store.clock_gated_reader import ClockGatedReader
    rows = len(ClockGatedReader(_store_root(facts), datasets[0].name).read_as_of(2**62))
    return ProbeResult(
        OK,
        f"{rows} rows across {parts} append-only part(s) in {len(datasets)} dataset(s)",
        f"capture/store/{datasets[0].name}",
    )


def probe_clock_gated_access(facts: SystemFacts) -> ProbeResult:
    """Reports on the gate by exercising it, not by checking the file exists.

    The check that matters is negative: read one nanosecond before the
    earliest availability time and confirm nothing comes back. A probe that
    only confirmed `src/store/clock_gated_reader.py` exists on disk would pass
    just as happily against a gate that leaks everything.
    """
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store to gate", "src/store/clock_gated_reader.py")
    from store.clock_gated_reader import ClockGatedReader
    from store.temporal_schema import AVAILABILITY_TIME

    reader = ClockGatedReader(_store_root(facts), datasets[0].name)
    everything = reader.read_as_of(2**62)
    if everything.empty:
        return ProbeResult(DEGRADED, "store exists but reads empty", "ClockGatedReader.read_as_of")

    earliest = int(everything[AVAILABILITY_TIME].min())
    hidden = reader.read_as_of(earliest - 1)
    if not hidden.empty:
        # The gate is the whole layer. If it lets anything through early, that is
        # a failure of the system's core guarantee, not a degraded metric.
        return ProbeResult(FAILING, f"{len(hidden)} row(s) visible before their availability time",
                           "ClockGatedReader.read_as_of")
    return ProbeResult(OK, f"gate holds: nothing visible before {earliest}",
                       "ClockGatedReader.read_as_of, exercised live")


def probe_status_wall(facts: SystemFacts) -> ProbeResult:
    """This board, reporting on itself. It exists, so it says so."""
    return ProbeResult(
        BUILT,
        "this board; generated from probes, never hand-written",
        "src/statuswall/",
    )


# Feature key -> probe. A feature absent from this map has no measurement and is
# therefore NOT_BUILT. Adding a row here is a claim that something is real, and
# the probe is what has to defend it.
PROBES = {
    "spot ohlcv trade tape multi venue": probe_trade_tape,
    "l2 order book depth 20 50 levels": probe_l2_depth,
    "liquidation feed": probe_liquidation_feed,
    "mark price vs index vs oracle price per venue": probe_mark_price,
    "gap detection provenance flagged backfill": probe_gap_detection,
    "per feed data quality score": probe_data_quality_score,
    "sequence gap detection on book streams": probe_sequence_gap_detection,
    "disk memory resource watchdog": probe_resource_watchdog,
    "venue health monitor auto halt": probe_venue_health,
    "reconnect with full jitter backoff honour retry after": probe_capture_supervisor,
    "strategy health board": probe_status_wall,
    "bitemporal store": probe_bitemporal_store,
    "clock gated access api": probe_clock_gated_access,
}


class UnknownProbeTarget(KeyError):
    """A probe is mapped to a feature the catalogue no longer contains.

    Raised rather than ignored: a silently orphaned probe means the wall stops
    reporting a real measurement the moment someone rewords a row in
    FEATURES.md, and nothing on the board would look wrong.
    """


def verify_probe_coverage(features: list[Feature]) -> None:
    """Refuse to build a wall whose probes have drifted off the catalogue.

    Kept separate from `assess` so that assessing a subset of features stays
    possible; this is a whole-catalogue check and the CLI is the only caller
    that holds the whole catalogue.
    """
    known = {normalise_key(f.name) for f in features}
    orphans = sorted(set(PROBES) - known)
    if orphans:
        raise UnknownProbeTarget(
            f"{len(orphans)} probe(s) target features not in the catalogue: {orphans}")


def assess(features: list[Feature], facts: SystemFacts) -> dict[str, ProbeResult]:
    """Status for every catalogued feature. Unprobed features are NOT_BUILT."""
    results: dict[str, ProbeResult] = {}
    for feature in features:
        probe = PROBES.get(feature.key)
        results[feature.key] = probe(facts) if probe else ProbeResult(
            NOT_BUILT, "designed only; no code and no measurement", "—")
    return results
