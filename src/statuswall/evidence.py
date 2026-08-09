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
import time
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


def _silent_stream_probe(facts: SystemFacts, streams: tuple[str, ...],
                         label: str) -> ProbeResult:
    """A subscribed stream that never spoke is a failure, not an absence.

    The distinction matters: a stream nobody subscribed to is simply not built,
    while a stream that was subscribed and delivered nothing is wired and dead -
    and it looks identical on disk to a quiet market unless the ledger is read.

    Several stream names may satisfy one feature, because a feed the venue
    refuses to push can still be recorded by polling it. Data on any of them
    answers the feature; the tile only fails when every route is silent.
    """
    silent_routes: set[tuple[str, str]] = set()
    for venue, report in facts.reports.items():
        for stream in streams:
            if stream in report.get("silent_stream_names", []):
                silent_routes.add((venue, stream))
    silent_in = sorted({venue for venue, _ in silent_routes})

    # Bytes alone do not make a feed healthy: a stream that filled a file for a
    # week and then died leaves exactly the same bytes behind as one still
    # running. A route counts only where it has data AND is not reported silent
    # - judged per (venue, stream), not per venue, so a live poll answers the
    # tile even while the withheld websocket stream it replaced is silent on
    # that same venue for that same feature.
    live_bytes = {}
    for name, size in _stream_bytes(facts, streams).items():
        venue, _, stream_symbol = name.partition("/")
        if (venue, stream_symbol.split("_", 1)[0]) not in silent_routes:
            live_bytes[name] = size
    if live_bytes:
        state, why = _capture_liveness(facts)
        return ProbeResult(state, f"{label} streaming. {why}", "raw_bytes_by_stream")

    if silent_in:
        return ProbeResult(
            FAILING,
            f"'{'/'.join(streams)}' subscribed on {', '.join(silent_in)} "
            "and never delivered a frame",
            "capture_health silent_stream_names",
        )
    return ProbeResult(NOT_BUILT, f"{label} not subscribed on any venue", "capture_health")


def probe_liquidation_feed(facts: SystemFacts) -> ProbeResult:
    return _silent_stream_probe(facts, ("forceOrder",), "liquidation feed")


def probe_mark_price(facts: SystemFacts) -> ProbeResult:
    # `premiumIndex` is the REST poll that replaced the withheld `markPrice`
    # websocket stream on 2026-08-03; it carries mark, index and settlement
    # price in one body. `markPrice` stays listed so an archive written before
    # that date still answers this tile.
    return _silent_stream_probe(facts, ("premiumIndex", "markPrice"), "mark price")


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
    # The halt registry is read, never driven from here: a display must not
    # decide whether a venue may be traded. It reports what the registry
    # recorded, so a halt shows on the wall with the reason that caused it.
    from ops.venue_halt import VenueHaltRegistry

    registry = VenueHaltRegistry(Path(facts.capture_root) / "ops")
    halted = {v: registry.halt_reason(v) for v in facts.reports
              if not registry.is_tradeable(v)}

    detail = "; ".join(alarms) if alarms else "no venue alarms"
    if halted:
        return ProbeResult(
            FAILING,
            f"HALTED: " + ", ".join(f"{v} ({r})" for v, r in sorted(halted.items()))
            + f". {detail}",
            "src/ops/venue_halt.py + capture_health")
    return ProbeResult(
        DEGRADED if alarms else OK,
        f"health monitoring live and auto-halt armed ({detail}); "
        f"{len(facts.reports)} venue(s) tradeable",
        "src/ops/venue_halt.py + capture_health",
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


# Bars build closed days only, so the newest partition legitimately trails the tape
# by up to a day plus however long the build takes. The threshold sits above that,
# so a normal lag is not reported as a stopped pipeline - and a genuinely abandoned
# store still surfaces within a day and a half rather than never.
_STORE_STALE_HOURS = 36.0
# Below this share of the captured symbols the store is not a smaller store, it is
# a different one from the archive beside it.
_STORE_COVERAGE_FLOOR = 0.9


def _store_symbols(datasets: list[Path]) -> int:
    """Symbols the store holds bars for, counted from its own partition layout."""
    return len({part.name for dataset in datasets for part in dataset.iterdir()
                if part.is_dir() and part.name.startswith("symbol=")})


def _captured_symbols_on(facts: SystemFacts) -> int | None:
    """Symbols with a trade tape on the newest captured day, or None if unknowable.

    None rather than zero, because "no archive to compare against" and "an archive
    holding nothing" are different facts and only one of them is a shortfall. Reuses
    `store.cli.captured_symbols` rather than re-deriving the filename convention, so
    the board cannot disagree with the builder about what was captured.
    """
    if not facts.latest_capture_date or not facts.venues:
        return None
    from store.cli import captured_symbols

    total = 0
    measured_any = False
    for venue in facts.venues:
        try:
            total += len(captured_symbols(facts.capture_root, venue, facts.latest_capture_date))
            measured_any = True
        except Exception:
            # A venue with no trade stream registered, or a day it never captured.
            # Skipped rather than counted as zero - see the None contract above.
            continue
    return total if measured_any else None


def probe_bitemporal_store(facts: SystemFacts) -> ProbeResult:
    """Reports on the store by reading it, and on whether it is still being written.

    Row and part counts come from `ClockGatedReader.read_as_of`, the same call
    every consumer makes - a probe that stat()ed the directory instead would go
    on reporting OK against a store whose Parquet files had rotted or emptied.

    Rows alone were not enough, and the failure was live rather than hypothetical.
    On 2026-08-08 this tile read `ok | 1671 rows across 12 append-only part(s)`
    while the newest partition was five days old and the store covered 6 of the
    2,109 symbols being captured. A dead store keeps its rows forever, so a row
    count cannot tell a working pipeline from an abandoned one - and breadth, the
    number that had been wrong for days, was not measured at all.

    So the tile now carries both: how long since anything was written, and how many
    of the captured symbols made it in. Freshness decides first, because a stopped
    pipeline makes the coverage number meaningless.
    """
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store built from the archive yet",
                           "capture/store")
    parts = [part for dataset in datasets for part in dataset.rglob("*.parquet")]
    from store.clock_gated_reader import ClockGatedReader
    rows = len(ClockGatedReader(_store_root(facts), datasets[0].name).read_as_of(2**62))

    evidence = f"capture/store/{datasets[0].name}"
    counts = f"{rows} rows across {len(parts)} append-only part(s) in {len(datasets)} dataset(s)"
    if not parts or rows == 0:
        return ProbeResult(DEGRADED, f"store exists and reads empty: {counts}", evidence)

    newest_ns = max(part.stat().st_mtime for part in parts)
    age_hours = (time.time() - newest_ns) / 3600

    in_store = _store_symbols(datasets)
    captured = _captured_symbols_on(facts)
    if captured is None:
        coverage = "coverage not measured (no captured trade tape to compare against)"
    else:
        coverage = f"{in_store} of {captured} captured symbol(s) built"

    if age_hours > _STORE_STALE_HOURS:
        return ProbeResult(
            STOPPED,
            f"nothing written for {age_hours:.0f}h ({age_hours / 24:.1f} days); "
            f"{counts}; {coverage}",
            evidence)
    if captured is not None and in_store < _STORE_COVERAGE_FLOOR * captured:
        return ProbeResult(
            DEGRADED,
            f"only {coverage}; newest part {age_hours:.0f}h old; {counts}",
            evidence)
    return ProbeResult(OK, f"{counts}; {coverage}; newest part {age_hours:.0f}h old",
                       evidence)


def probe_bar_price_validity(facts: SystemFacts) -> ProbeResult:
    """Are the prices in the store prices at all?

    Nothing was asking. The store tile reports rows, then freshness, then coverage,
    and all three were green on 2026-08-08 while 746 of 1,671 bars carried a
    non-positive price. Binance emits placeholder frames on its `@trade` stream -
    `p` "0", `q` "0", `X` "NA" - and `_extract_binance` turned each into a trade at
    0.0, where `low=("price", "min")` needs exactly one to ruin a bar. The ruined
    bars looked perfect otherwise: correct open, correct high, hundreds of trades.
    Found by a paper-plumbing run refusing to divide by a zero price, which is luck,
    not monitoring.

    FAILING, not DEGRADED: an impossible price is not a smaller truth, it is a wrong
    one, and every number computed from it is wrong without saying so.

    Judged on what `ClockGatedReader` SERVES, never on the parquet files. The store
    is append-only and corrections are new rows, so the poisoned bars remain on disk
    permanently after remediation. A probe reading the files would report FAILING
    forever with no way to clear it, and a permanently red tile is one everybody
    learns to skip - which is how the next real failure gets missed.
    """
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store to check prices in", "capture/store")

    from store.clock_gated_reader import ClockGatedReader
    served = ClockGatedReader(_store_root(facts), datasets[0].name).read_as_of(2**62)
    evidence = f"capture/store/{datasets[0].name}"
    if served.empty:
        return ProbeResult(DEGRADED, "store reads empty, so no price can be checked",
                           evidence)

    columns = [c for c in ("open", "high", "low", "close") if c in served.columns]
    if not columns:
        return ProbeResult(
            DEGRADED,
            f"no OHLC columns in {datasets[0].name}; nothing here knows what a price is",
            evidence)

    impossible = served[(served[columns] <= 0).any(axis=1)]
    if impossible.empty:
        return ProbeResult(
            OK,
            f"every price positive across {len(served)} bar(s) and {len(columns)} "
            f"OHLC column(s)",
            evidence)

    # Named, not just counted. A count tells a reader something is wrong; the symbols
    # and the worst column tell them where to go and look.
    from store.temporal_schema import SYMBOL
    symbols = sorted(impossible[SYMBOL].unique().tolist())
    worst = {c: int((served[c] <= 0).sum()) for c in columns if (served[c] <= 0).any()}
    return ProbeResult(
        FAILING,
        f"{len(impossible)} of {len(served)} served bar(s) carry a non-positive price "
        f"- {', '.join(f'{c}:{n}' for c, n in sorted(worst.items()))} - "
        f"in {', '.join(symbols[:6])}{' ...' if len(symbols) > 6 else ''}",
        evidence)


def probe_clock_gated_access(facts: SystemFacts) -> ProbeResult:
    """Reports on the gate by exercising it, not by checking the file exists.

    The check that matters is negative: read one nanosecond before the
    earliest availability time and confirm nothing comes back. A probe that
    only confirmed `src/store/clock_gated_reader.py` exists on disk would pass
    just as happily against a gate that leaks everything.

    "Earliest" is taken over every stored version, read straight off the
    partitions, and NOT over what `read_as_of` returns. A corrected bar is two
    rows for the same key: the original, available at the bar boundary, and the
    correction, available whenever the fix ran. The corrected view carries only
    the second, so its minimum availability sits AFTER the moment the original
    first became visible - and reading one nanosecond before that minimum
    legitimately returns the pre-correction rows. This probe read that as the
    gate leaking and put the wall's core-guarantee tile into FAILING for a store
    whose gate was correct; the first correction ever written was what broke it.
    """
    datasets = _bar_datasets(facts)
    if not datasets:
        return ProbeResult(NOT_BUILT, "no store to gate", "src/store/clock_gated_reader.py")
    from store.clock_gated_reader import ClockGatedReader
    from store.parquet_partition import read_dataset
    from store.temporal_schema import AVAILABILITY_TIME

    store_root, dataset = _store_root(facts), datasets[0].name
    reader = ClockGatedReader(store_root, dataset)
    stored = read_dataset(store_root, dataset)
    if stored.empty or reader.read_as_of(2**62).empty:
        return ProbeResult(DEGRADED, "store exists but reads empty", "ClockGatedReader.read_as_of")

    earliest = int(stored[AVAILABILITY_TIME].min())
    hidden = reader.read_as_of(earliest - 1)
    if not hidden.empty:
        # The gate is the whole layer. If it lets anything through early, that is
        # a failure of the system's core guarantee, not a degraded metric.
        return ProbeResult(FAILING, f"{len(hidden)} row(s) visible before their availability time",
                           "ClockGatedReader.read_as_of")

    # The negative check alone passes against a gate that serves nothing at all
    # before some late cutoff and then everything at once. This second read sits
    # inside the data and asserts the invariant on what actually came back, which
    # is the guarantee stated rather than a proxy for it.
    midpoint = int(stored[AVAILABILITY_TIME].median())
    served = reader.read_as_of(midpoint)
    ahead = served[served[AVAILABILITY_TIME] > midpoint]
    if not ahead.empty:
        return ProbeResult(FAILING,
                           f"{len(ahead)} of {len(served)} row(s) served at {midpoint} "
                           f"carry a later availability time",
                           "ClockGatedReader.read_as_of")
    return ProbeResult(OK,
                       f"gate holds: nothing visible before {earliest}, and all "
                       f"{len(served)} row(s) served mid-history were already available",
                       "ClockGatedReader.read_as_of, exercised live")


# How long a fetched fee schedule stays current. Tiers move with 30-day rolling
# volume, so a fetch from days ago describes a rate that may no longer apply -
# and a tile still reading "verified" would be asserting a stale measurement.
_FEE_VERIFICATION_MAX_AGE_HOURS = 24.0


def probe_cost_engine(facts: SystemFacts) -> ProbeResult:
    """What the cost engine can actually price right now, and on what evidence.

    Rule 8 applied to the one gate every signal passes: a cost engine quoting
    from fees nobody fetched must not render green. Being wrong here is not a
    display bug - `ARCHITECTURE.md` Layer 1 puts fees at 5-10x slippage in
    deciding breakeven, so an unverified fee silently trusted is the difference
    between an edge and a loss.

    Three things are measured, none asserted: whether any venue's schedule was
    actually fetched, and whether the funding and book datasets exist for the
    spread, impact and funding components to read. A component with no dataset
    is charged as zero, which understates cost - the dangerous direction - so
    it caps this tile below OK however healthy everything else looks.
    """
    import json
    import time

    from cost.fee_schedule import DECLARED_SCHEDULES

    store = Path(facts.capture_root) / "store"
    missing = [name for name in ("funding", "book")
               if not (store / name).is_dir()]

    # The receipt left by scripts/verify_fee_schedules.sh. Read rather than
    # re-fetched: a display that authenticates on every render puts credentials
    # in the render path and spends rate limit to draw a tile. Verification is a
    # separate act; this reports what it recorded, and how long ago.
    receipt_path = Path(facts.capture_root) / "fee-verification" / "latest.json"
    verified, refused, age_hours = [], [], None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        receipt = None
    else:
        measured = receipt.get("measured_at_ns")
        if isinstance(measured, int):
            age_hours = (time.time_ns() - measured) / 3.6e12
        for venue, entry in (receipt.get("venues") or {}).items():
            (verified if entry.get("verified") else refused).append(venue)

    declared = sorted({v for (v, _), sched in DECLARED_SCHEDULES.items()
                       if not sched.is_verified})
    proof = "capture/fee-verification + capture/store"

    if receipt is None:
        detail = (f"no fee schedule has ever been verified - running on the "
                  f"declared table ({', '.join(declared)}). Run "
                  f"scripts/verify_fee_schedules.sh")
        return ProbeResult(NOT_BUILT if missing else PARTIAL,
                           detail + (f"; missing dataset(s): {', '.join(missing)}"
                                     f" - spread and impact charge zero"
                                     if missing else ""), proof)

    if missing:
        return ProbeResult(
            PARTIAL,
            f"verified {', '.join(verified) or 'nothing'}, but "
            f"{', '.join(missing)} dataset(s) absent so those components charge "
            f"zero, which understates cost", proof)
    if refused:
        return ProbeResult(
            PARTIAL,
            f"{', '.join(refused)} would not serve a schedule, so quotes there "
            f"rest on the declared table; verified {', '.join(verified) or 'none'}",
            proof)
    if age_hours is not None and age_hours > _FEE_VERIFICATION_MAX_AGE_HOURS:
        return ProbeResult(
            PARTIAL,
            f"every schedule verified, but the last fetch is stale - "
            f"{age_hours:.0f}h old against a {_FEE_VERIFICATION_MAX_AGE_HOURS:.0f}h "
            f"limit. Fee tiers move with 30-day volume, so an old fetch is a "
            f"historical fact rather than a current one", proof)
    return ProbeResult(
        OK,
        f"every schedule fetched and verified ({', '.join(sorted(verified))}), "
        f"newest {age_hours:.1f}h old; funding and book datasets present", proof)


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
    "stored bar price validity gate": probe_bar_price_validity,
    "clock gated access api": probe_clock_gated_access,
    "cost engine round trip breakeven gate": probe_cost_engine,
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
    """Status for every catalogued feature. Unprobed features are NOT_BUILT.

    A probe that raises loses its own tile and nothing else. Calling them
    unguarded meant one exception - a rotted Parquet part reaching
    `probe_bitemporal_store`, a venue report missing a key - took down the entire
    board and no feature rendered at all, precisely at the moment something was
    wrong and the board was the thing worth reading.

    The failure is caught, never hidden: the tile reads FAILING, carries the
    exception type and message, and names the probe in its proof, so a
    programming error arrives as a visible red tile rather than as a quiet OK.
    Losing one tile to a bug is degraded; losing the board is useless.
    """
    results: dict[str, ProbeResult] = {}
    for feature in features:
        probe = PROBES.get(feature.key)
        if probe is None:
            results[feature.key] = ProbeResult(
                NOT_BUILT, "designed only; no code and no measurement", "—")
            continue
        try:
            results[feature.key] = probe(facts)
        except Exception as exc:
            results[feature.key] = ProbeResult(
                FAILING,
                f"probe raised {type(exc).__name__}: {exc}",
                f"{getattr(probe, '__name__', probe)} raised; this tile measured nothing",
            )
    return results
