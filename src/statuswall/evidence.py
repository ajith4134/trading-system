"""Measures what is actually true on this machine, feature by feature.

Rule 8: a display shows measured state, never asserted state. Everything here
returns what a probe found, and a feature with no probe is NOT_BUILT by
construction rather than by omission - there is no code path that can render a
feature healthy without a measurement behind it.

`ProbeResult.proof` carries where the answer came from, and the renderer shows
it. A status whose provenance cannot be named is not a status.
"""
from __future__ import annotations

import json
import datetime as dt
import functools
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
# Built, and nothing measured it. Distinct from NOT_BUILT, which says the thing
# does not exist, and distinct from OK, which is a measurement that came back
# clean. Rule 8: absence of evidence renders as its own state, never as green and
# never as blank. It sorts above PARTIAL because being blind about something that
# exists is worse than knowing it is half-done.
NOT_MEASURED = "not_measured"
PARTIAL = "partial"
OK = "ok"
BUILT = "built"
NOT_BUILT = "not_built"

SEVERITY_ORDER = [FAILING, DEGRADED, STOPPED, NOT_MEASURED, PARTIAL, OK, BUILT, NOT_BUILT]

STATE_LABEL = {
    FAILING: "FAILING",
    DEGRADED: "DEGRADED",
    STOPPED: "STOPPED",
    NOT_MEASURED: "NOT MEASURED",
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
    # Where the requirements ledger lives, or None when this pass was not given
    # one. Optional rather than assumed: a probe that fell back to a path under
    # `$HOME` would report a clean bill of health on any machine where the ledger
    # is somewhere else, which is the failure it exists to catch.
    ledger_root: Path | None = None


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


def measure_system(capture_root: Path, repo_root: Path, now: dt.datetime,
                   ledger_root: Path | None = None) -> SystemFacts:
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
        ledger_root=ledger_root,
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
    # `matches` is coinbase's word for the trade tape, added 2026-08-10. The
    # match here is exact on the stream name rather than a substring, so a venue
    # whose name for a feed is not in this tuple is invisible to the tile that
    # exists to count that feed - it reads "captured on 3 venues" while four are
    # writing, and nothing says which one was dropped.
    sizes = _stream_bytes(facts, ("trade", "trades", "matches"))
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
        # Counted from the streams that carry a trade, not from the archive
        # directories. `facts.venues` is every venue with a folder, and two of
        # them have no tape at all - bybit is polled for funding and bybit-liq
        # records liquidations - so this read "captured on 6 venues" while four
        # were writing trades. A tile that overstates its own coverage is the
        # failure Rule 8 is about, and it was overstating before coinbase
        # existed to notice it.
        f"trade tape captured on {len({key.split('/', 1)[0] for key in sizes})} "
        f"venues ({total_mb:.0f} MB). {why}",
        f"raw_bytes_by_stream over {len(sizes)} trade streams",
    )


def probe_l2_depth(facts: SystemFacts) -> ProbeResult:
    # Four names for one feed across four venues: binance pushes `depth`,
    # hyperliquid `l2Book`, coinbase `level2` with a `level2Snapshot` at
    # subscribe, and the periodic REST book lands as `depthSnapshot` on both
    # binance-spot and coinbase. The snapshot streams were missing from this
    # tuple before 2026-08-10 and so were absent from the tile - the polled book
    # is the one this dataset is actually built from.
    sizes = _stream_bytes(facts, ("depth", "l2Book", "level2", "level2Snapshot",
                                  "depthSnapshot"))
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
    # running. A route counts only where it has data AND is not reported silent.
    #
    # Silence is judged per SYMBOL where the report can say (newer reports
    # carry `silent_stream_symbols`), and per stream name only as the fallback
    # for reports written before that key existed. The granularity is the whole
    # question on a market-wide event stream: bybit-liq's first day had 700+
    # liquidation symbols correctly quiet and five delivering, and name-level
    # judgement read the five as part of a dead feed. Per symbol, a quiet
    # symbol is the market being calm; a dead FEED is every symbol silent,
    # which then leaves no live route and the tile fails as it should.
    live_bytes = {}
    for name, size in _stream_bytes(facts, streams).items():
        venue, _, stream_symbol = name.partition("/")
        stream = stream_symbol.split("_", 1)[0]
        report = facts.reports.get(venue, {})
        if "silent_stream_symbols" in report:
            silent_symbols = report["silent_stream_symbols"]
            # A market-wide stream files its silence under the ALL-MARKET
            # sentinel; that entry condemns every symbol's bytes at once,
            # because on such a stream the whole feed is the unit that dies.
            if stream_symbol in silent_symbols or f"{stream}_ALL-MARKET" in silent_symbols:
                continue
        elif (venue, stream) in silent_routes:
            continue
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
    # `allLiquidation` is bybit-liq's market-wide stream, added 2026-08-09 as
    # the second venue DECISIONS §12.6 said would end this tile's wait - the
    # probe measured 16 frames in 90s from this host before the venue module
    # was written. `forceOrder` stays listed: binance's subscription is kept
    # deliberately (a recovery would be noticed), and an archive written before
    # today still answers this tile through it.
    return _silent_stream_probe(facts, ("forceOrder", "allLiquidation"),
                                "liquidation feed")


def probe_wash_trading_discount(facts: SystemFacts) -> ProbeResult:
    # Computes the discount rather than checking a file exists. PARTIAL, not
    # OK, and the reason is on the tile: nothing sizes off the discounted
    # figure yet because no sizer exists, and the measure is an upper bound on
    # uncorroborated volume rather than a wash-trading verdict.
    try:
        from features.volume_quality import measure_volume_quality
        table = measure_volume_quality(facts.capture_root / "store",
                                       int(time.time() * 1e9))
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"volume quality failed: {error}",
                           "features/volume_quality.py")
    if table.rows.empty:
        return ProbeResult(NOT_BUILT, "no symbol has enough bars to corroborate",
                           "capture/store/bars")
    reported = float(table.rows["reported_volume"].sum())
    discounted = float(table.rows["discounted_volume"].sum())
    share = 0.0 if reported <= 0 else (reported - discounted) / reported
    median = float(table.rows["no_impact_fraction"].median())
    return ProbeResult(
        PARTIAL,
        f"{len(table.rows)} (venue, symbol) pairs corroborated: {share:.1%} of all "
        f"reported volume moved no price, median symbol {median:.1%}. Upper bound "
        f"on uncorroborated volume, not a wash verdict; nothing sizes off it yet",
        "features/volume_quality.py + capture/store/bars")


def probe_consolidated_price(facts: SystemFacts) -> ProbeResult:
    # Asked at the newest clock the book dataset supports, not at wall-clock
    # now, and the gap between those two is reported rather than hidden: the
    # store builds closed hours, so Layer 1's depth trails live by up to an
    # hour and a half. Measured 2026-08-09: 82 minutes. A consolidated price
    # is a live-pricing input, so that lag is the honest headline about this
    # feature - asking at `now` correctly returns nothing at all, every venue
    # excluded as stale.
    try:
        from features.consolidated_price import consolidate_prices
        from store.clock_gated_reader import ClockGatedReader
        book = ClockGatedReader(facts.capture_root / "store", "book").read_as_of(2**62)
        if book.empty:
            return ProbeResult(NOT_BUILT, "no book dataset to consolidate from",
                               "capture/store/book")
        newest_ns = int(book["event_time_ns"].max())
        lag_minutes = (time.time() * 1e9 - newest_ns) / 6e10
        table = consolidate_prices(facts.capture_root / "store", newest_ns + 1)
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"consolidation failed: {error}",
                           "features/consolidated_price.py")
    if table.rows.empty:
        return ProbeResult(DEGRADED,
                           f"no symbol could be consolidated; excluded {table.excluded}",
                           "features/consolidated_price.py")
    multi = int((table.rows["venues_used"] > 1).sum())
    worst = float(table.rows["disagreement_bps"].max())
    return ProbeResult(
        PARTIAL,
        f"{len(table.rows)} symbol(s) consolidated, {multi} from more than one venue, "
        f"worst venue disagreement {worst:.1f} bps; depth exists for the 3 core "
        f"symbols only, and Layer 1's book trails live by {lag_minutes:.0f} min",
        "features/consolidated_price.py + capture/store/book")


def probe_peg_monitor(facts: SystemFacts) -> ProbeResult:
    # Runs the monitor rather than checking that a file exists. Caps at
    # PARTIAL and says why: nothing consumes a breach yet, because position
    # sizing does not exist - the module's own axis verdict fails DEPTH for
    # the same reason. A green tile over an unwired monitor is the Rule 8
    # failure this board was built to prevent.
    try:
        from features.peg_monitor import monitor_pegs
        table = monitor_pegs(facts.capture_root / "store", int(time.time() * 1e9))
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"peg monitor failed: {error}",
                           "features/peg_monitor.py")
    if table.rows.empty:
        return ProbeResult(NOT_BUILT, "no asset has enough history to judge",
                           "capture/store/bars")
    breached = int((table.rows["verdict"] == "BREACHED").sum())
    moved = int(table.rows["level_moved"].sum())
    unfloored = int((~table.rows["threshold_floored"]).sum())
    return ProbeResult(
        PARTIAL,
        f"{len(table.rows)} pegged (venue, asset) pairs judged: {breached} breached, "
        f"{moved} at a moved level, {unfloored} on thresholds no verified fee "
        f"schedule could floor; {table.skipped['not_pegged']} assets measured "
        f"not-pegged. Nothing consumes a breach - no sizer exists",
        "features/peg_monitor.py + capture/store/bars")


def probe_spot_perp_basis(facts: SystemFacts) -> ProbeResult:
    # Computes both actual numbers rather than checking that files exist: a
    # basis and a curve are derivations, and the only proof a derivation works
    # is running it. The catalogue row names two things - the perpetual basis
    # and the term structure - so this tile stays PARTIAL while either half is
    # empty, however healthy the other one looks.
    now_ns = int(time.time() * 1e9)
    try:
        from features.spot_perp_basis import compute_spot_perp_basis
        table = compute_spot_perp_basis(facts.capture_root / "store", now_ns)
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"basis computation failed: {error}",
                           "features/spot_perp_basis.py")
    if table.rows.empty:
        return ProbeResult(NOT_BUILT, "no funding rows to price a basis from",
                           "capture/store/funding")
    venues = sorted(table.rows["venue"].unique())
    refused = sum(table.refused.values())
    basis_line = (f"basis computed live: {len(table.rows)} (venue, symbol) pairs "
                  f"across {', '.join(venues)}, {refused} refused")

    try:
        from features.term_structure import compute_term_structure, summarise_curves
        curve = compute_term_structure(facts.capture_root / "store", now_ns)
    except Exception as error:
        return ProbeResult(PARTIAL, f"{basis_line}; term structure failed: {error}",
                           "features/term_structure.py")
    if curve.rows.empty:
        return ProbeResult(
            PARTIAL,
            f"{basis_line}; term structure built but empty - "
            f"refused {curve.refused}",
            "features/term_structure.py + capture/store/dated_futures")

    curves = summarise_curves(curve)
    multi = int((curves["tenors"] >= 2).sum())
    unannualisable = curve.refused["too_near_expiry_to_annualise"]
    return ProbeResult(
        OK,
        f"{basis_line}. Term structure: {len(curve.rows)} dated contract(s) on "
        f"{len(curves)} underlying(s), {multi} carrying two or more tenors; "
        f"{unannualisable} too near expiry to annualise",
        "features/spot_perp_basis.py + features/term_structure.py "
        "+ capture/store/{funding,dated_futures}")


def probe_funding_rates(facts: SystemFacts) -> ProbeResult:
    # Four carriers, all verified on stored frames: binance's `premiumIndex`
    # (lastFundingRate + nextFundingTime - rate AND schedule in one body),
    # bybit's `linearTickers` (fundingRate, nextFundingTime, per-symbol
    # interval), hyperliquid's `assetCtx` (funding, hourly on oracle), and the
    # settled-history `fundingRate` backfill. This tile read NOT BUILT for a
    # feed captured on three venues because nothing had wired a probe - the
    # exact gap `verify_probe_coverage` cannot catch, since it only checks
    # probes against features, not features against capture.
    return _silent_stream_probe(
        facts, ("premiumIndex", "linearTickers", "assetCtx", "fundingRate"),
        "funding rate + schedule")


def probe_open_interest(facts: SystemFacts) -> ProbeResult:
    # Three carriers, verified on stored frames 2026-08-09: binance's
    # per-symbol `openInterest` poll, and the `openInterest` field riding
    # bybit's `linearTickers` and hyperliquid's `assetCtx` funding polls.
    return _silent_stream_probe(
        facts, ("openInterest", "linearTickers", "assetCtx"), "open interest")


def probe_mark_price(facts: SystemFacts) -> ProbeResult:
    # The catalogue row names two things - capturing the three prices, and
    # RECONCILING them - so this tile runs both and reports the weaker.
    #
    # Capture: `premiumIndex` is the REST poll that replaced the withheld
    # `markPrice` websocket stream on 2026-08-03; it carries binance's mark and
    # index in one body. `markPrice` stays listed so an archive written before
    # that date still answers this tile. The other carriers, verified on stored
    # frames 2026-08-09: bybit's `linearTickers` (markPrice, indexPrice) and
    # hyperliquid's `assetCtx` (markPx, oraclePx, midPx) - the oracle-vs-mark
    # distinction the catalogue flags as commonly missed is exactly the
    # difference between those two bodies, and both are on disk per minute.
    captured = _silent_stream_probe(
        facts, ("premiumIndex", "markPrice", "linearTickers", "assetCtx"),
        "mark/index/oracle price")
    if captured.state in (NOT_BUILT, FAILING):
        return captured

    try:
        from features.price_divergence import reconcile_prices
        table = reconcile_prices(facts.capture_root / "store",
                                 int(time.time() * 1e9))
    except Exception as error:
        return ProbeResult(PARTIAL,
                           f"{captured.detail} Reconciliation failed: {error}",
                           "features/price_divergence.py")
    if table.rows.empty:
        return ProbeResult(PARTIAL,
                           f"{captured.detail} Reconciliation ran and had "
                           f"nothing in its window to judge",
                           "features/price_divergence.py")

    counts = table.by_verdict()
    judged = sum(v for k, v in counts.items() if not k.startswith("UNJUDGED"))
    flagged = counts.get("EXTREME", 0) + counts.get("STALE_MARK", 0)
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    detail = (f"{captured.detail} Reconciled {len(table.rows)} instrument(s) "
              f"over {table.window_ns // 3_600_000_000_000}h: {summary}")
    if not judged:
        # Every instrument unjudged is not a clean bill of health - it is the
        # window holding too little to measure a scale from, and it renders as
        # its own state rather than as green.
        return ProbeResult(PARTIAL, f"{detail} - nothing judged",
                           "features/price_divergence.py")
    if flagged:
        return ProbeResult(DEGRADED, detail, "features/price_divergence.py")
    return ProbeResult(captured.state, detail,
                       "features/price_divergence.py + capture/store/funding")


def probe_gap_detection(facts: SystemFacts) -> ProbeResult:
    # The catalogue row is "gap detection + provenance-flagged backfill", so
    # both halves are measured. The backfill half is measured by reading the
    # dataset rather than by checking the module exists: a labelling scheme
    # with nothing behind it is the thing this row was written about.
    observed = sum(r.get("gaps", {}).get("observation_loss", 0) for r in facts.reports.values())
    corrupting = sum(r.get("gaps", {}).get("corrupting", 0) for r in facts.reports.values())
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no capture to detect gaps in", "capture/ledger")
    detection = (f"detection live: {observed} observation-loss and "
                 f"{corrupting} corrupting gaps recorded")

    try:
        from store.bar_backfill import compare_reconstructed_to_observed
        agreement = compare_reconstructed_to_observed(
            facts.capture_root / "store", int(time.time() * 1e9))
    except Exception as error:
        return ProbeResult(PARTIAL, f"{detection}. Backfill audit failed: {error}",
                           "src/store/bar_backfill.py")
    if not agreement["reconstructed_bars"]:
        return ProbeResult(
            PARTIAL,
            f"{detection}. Backfill labelling built and nothing backfilled yet - "
            f"0 reconstructed bars stored",
            "src/capture/sequencing.py + store/bars_reconstructed")

    leaked = agreement["reconstructed_rows_in_observed"]
    if leaked:
        # The one thing this design is supposed to make impossible. Counted
        # rather than trusted, and it fails the tile outright.
        return ProbeResult(
            FAILING,
            f"{detection}. {leaked} reconstructed row(s) found INSIDE the "
            f"observed bars dataset - the two are blended",
            "src/store/bar_backfill.py")

    overlapping = agreement["overlapping"]
    provenance = (f"{agreement['reconstructed_bars']} reconstructed bar(s), every "
                  f"one labelled, 0 of them inside the observed dataset")
    if not overlapping:
        # Never compared is not agreement, and it is the state this reports
        # rather than a clean bill of health.
        return ProbeResult(PARTIAL, f"{detection}. {provenance}; never compared "
                                    f"against an observed bar",
                           "src/store/bar_backfill.py")
    return ProbeResult(
        OK,
        f"{detection}. {provenance}. Against {overlapping} overlapping observed "
        f"bar(s): {agreement['close_identical']} closes identical, median "
        f"{agreement['close_agreement_bps_median']:.4f} bps apart, p99 "
        f"{agreement['close_agreement_bps_p99']:.4f} bps",
        "src/store/bar_backfill.py + capture_health gaps",
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


# A halt registry nothing has fed for this long is not guarding anything, however
# healthy its stored verdict looks. Generous by design - the point is to catch a
# registry with no driver at all, not to grade a slow cycle.
_HALT_OBSERVATION_MAX_AGE_SECONDS = 3600


def probe_venue_health(facts: SystemFacts) -> ProbeResult:
    """Health monitoring and the auto-halt, reported separately because only one
    of them is running.

    This tile read "health monitoring live and auto-halt armed" until 2026-08-09,
    and the second half was never measured by anything. `VenueHaltRegistry` is
    imported here and nowhere else in `src/`; `observe()` and `assess_venue()`
    have no callers, so nothing can record a halt. The stored state had been
    written once, 19 hours earlier, by an ad-hoc run - and the tile rendered its
    "3 venue(s) tradeable" as a current fact.

    Rule 8's exact failure, inside the board built to prevent it, and worse than
    a missing tile: a green claim about a safety mechanism is what stops anyone
    checking whether the mechanism exists.

    Both halves are now measured. Health monitoring comes from `capture_health`,
    which is real. The halt comes from `last_observed_ns` on the registry's own
    state - when nothing has fed it, the tile says so instead of inferring that
    silence means safety.
    """
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no venue data", "capture_health")

    alarms = []
    for venue, report in facts.reports.items():
        if report.get("silent_streams"):
            alarms.append(f"{venue}: {report['silent_streams']} silence events")
        if report.get("corrupting_non_gap"):
            alarms.append(f"{venue}: {report['corrupting_non_gap']} corrupting events")
    monitoring = "; ".join(alarms) if alarms else "no venue alarms"

    # The registry is read, never driven from here: a display must not decide
    # whether a venue may be traded.
    from ops.venue_halt import VenueHaltRegistry

    registry = VenueHaltRegistry(Path(facts.capture_root) / "ops")
    proof = "src/ops/venue_halt.py last_observed_ns + capture_health"

    halted = {v: registry.halt_reason(v) for v in facts.reports
              if not registry.is_tradeable(v) and registry.last_observed_ns(v) is not None}
    if halted:
        # A recorded halt outranks everything else here, stale or not: a venue
        # someone marked untradeable stays untradeable until something says
        # otherwise.
        return ProbeResult(
            FAILING,
            "HALTED: " + ", ".join(f"{v} ({r})" for v, r in sorted(halted.items()))
            + f". {monitoring}", proof)

    ages = {venue: _seconds_since(registry.last_observed_ns(venue))
            for venue in facts.reports}
    unfed = sorted(v for v, age in ages.items()
                   if age > _HALT_OBSERVATION_MAX_AGE_SECONDS)
    if unfed:
        freshest = min((ages[v] for v in unfed), default=float("inf"))
        # An unstamped state file cannot prove "never" - only that it carries no
        # observation this code can date. Saying "never" there would be a smaller
        # version of the over-claim this probe was rewritten to remove.
        when = ("never, or not since the registry began stamping"
                if freshest == float("inf") else f"not for {freshest / 3600:.0f}h")
        return ProbeResult(
            DEGRADED,
            f"monitoring live, AUTO-HALT NOT ARMED - nothing has fed the halt "
            f"registry for {', '.join(unfed)} ({when}), so no degradation can "
            f"halt a venue. {monitoring}", proof)

    tradeable = [v for v in facts.reports if registry.is_tradeable(v)]
    return ProbeResult(
        DEGRADED if alarms else OK,
        f"monitoring live and auto-halt fed within the hour; "
        f"{len(tradeable)} of {len(facts.reports)} venue(s) tradeable. {monitoring}",
        proof)


def probe_data_quality_score(facts: SystemFacts) -> ProbeResult:
    """The composite score, computed from the same pass every other tile uses."""
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no feeds measured", "capture_health")
    from features.feed_quality import score_feeds

    scores = score_feeds(facts.reports)
    if not scores:
        return ProbeResult(NOT_BUILT, "no feed has any captured data to score",
                           "capture_health raw_bytes_by_stream")
    worst = min(scores, key=lambda s: s.score)
    healthy = sum(1 for s in scores if s.score >= 0.99)
    state = OK if worst.score >= 0.99 else DEGRADED
    return ProbeResult(
        state,
        f"{len(scores)} feeds scored, {healthy} at 0.99+; worst "
        f"{worst.venue}/{worst.stream} at {worst.score:.3f} on {worst.worst_component}. "
        f"Score is the MINIMUM of delivery, integrity, continuity and freshness, "
        f"never their mean",
        "features/feed_quality.py over capture_health reports",
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


# A pool this full is one burst of new symbols away from evicting, and eviction
# is the thing this tile exists to make visible before it becomes routine.
_POOL_HEADROOM_WARNING = 0.80
# Older than this and the newest report describes a recorder that is no longer
# running - twice the reporting cadence, so one missed interval is not an alarm.
_POOL_REPORT_MAX_AGE_SECONDS = 600


def _seconds_since(ts_ns: object) -> float:
    """Age of a nanosecond timestamp, with "no timestamp" reading as infinitely old.

    Infinity rather than zero on purpose: a missing timestamp must not make a
    report look freshly taken. Everything here grades staleness upward, so the
    unknown case lands on the cautious side by construction.
    """
    if not isinstance(ts_ns, int):
        return float("inf")
    return (time.time_ns() - ts_ns) / 1e9


def probe_writer_descriptor_pool(facts: SystemFacts) -> ProbeResult:
    """What the open-hour pool is doing, per venue, as the recorder last reported it.

    The failure this watches for is not dramatic and that is exactly why it needs
    a tile. A pool evicting steadily loses nothing and breaks nothing - it just
    reopens hours over and over, writing shorter zstd frames and compressing
    worse, and it looks identical on the board to a pool doing nothing. The only
    difference is a counter.

    Read from the ledger rather than from a live process, because the wall must
    keep reporting after the recorder that produced the number has gone. That is
    also why staleness is graded: a healthy report from a recorder which died
    three hours ago is a fact about three hours ago.
    """
    proof = "capture_health writer_pool, recorded by VenueRecorder"
    if not facts.reports:
        return ProbeResult(NOT_BUILT, "no venue data", proof)

    reported = {venue: report["writer_pool"] for venue, report in facts.reports.items()
                if report.get("writer_pool")}
    unreported = sorted(set(facts.reports) - set(reported))
    if not reported:
        # The honest answer for a recorder predating the report, or one that has
        # not run since. Not OK, and certainly not zero evictions.
        return ProbeResult(
            NOT_MEASURED,
            f"no venue has reported its descriptor pool ({', '.join(sorted(facts.reports))})",
            proof)

    evicting, tight, lines = [], [], []
    for venue, pool in sorted(reported.items()):
        evicted = int(pool.get("evicted", 0))
        peak, budget = int(pool.get("peak_open_hours", 0)), int(pool.get("budget", 0))
        share = peak / budget if budget else 0.0
        lines.append(f"{venue}: peak {peak} of {budget} open hours "
                     f"({share * 100:.0f}%), {evicted} evicted")
        if evicted:
            evicting.append(venue)
        elif share >= _POOL_HEADROOM_WARNING:
            tight.append(venue)

    stale = sorted(
        venue for venue, report in facts.reports.items()
        if venue in reported and _seconds_since(report.get("writer_pool_ts_ns"))
        > _POOL_REPORT_MAX_AGE_SECONDS)

    detail = "; ".join(lines)
    if unreported:
        detail += f". Not reported by {', '.join(unreported)}"
    if stale:
        detail += (f". Newest report from {', '.join(stale)} is over "
                   f"{_POOL_REPORT_MAX_AGE_SECONDS // 60} minutes old")

    if evicting:
        # Degraded, not failing: nothing is lost. Every evicted hour was finished
        # properly and reopened on the next frame. What it costs is compression,
        # and a recorder evicting at all means the descriptor budget is now the
        # binding constraint rather than a safety net.
        return ProbeResult(DEGRADED, f"evicting open hours - {detail}", proof)
    if stale or unreported:
        return ProbeResult(PARTIAL, detail, proof)
    if tight:
        return ProbeResult(
            DEGRADED,
            f"within {100 - _POOL_HEADROOM_WARNING * 100:.0f}% of the budget - {detail}",
            proof)
    return ProbeResult(OK, f"bounded, nothing evicted - {detail}", proof)


def probe_unsupported_claims(facts: SystemFacts) -> ProbeResult:
    """Ledger rows claiming BUILT whose named module nothing in `src/` can reach.

    The board's own blind spot, put on the board. Every other tile here asks
    whether a thing works; this one asks whether the thing the ledger says exists
    is connected to anything. Four defects in this project have had that shape,
    all failing in the flattering direction, and two of them were found on
    2026-08-09 only because somebody went looking by hand.

    DEGRADED rather than FAILING when rows are unsupported: nothing is broken at
    runtime. What is broken is the record, and a wrong record is what makes the
    next decision wrong. Nine of today's eleven are `validation/`, which is the
    honest shape of a phase whose consumer does not exist yet.
    """
    proof = "integrity.unsupported_claims over src/ + the requirements ledger"
    if facts.ledger_root is None:
        # Not OK. A pass given no ledger checked nothing, and saying so is the
        # difference between "no unsupported claims" and "no claims examined".
        return ProbeResult(NOT_MEASURED, "no requirements ledger given to check against",
                           proof)

    from integrity.unsupported_claims import (
        UNREACHABLE, audit_claims, classify_modules, invoked_modules, read_source_tree,
    )

    verdict = classify_modules(read_source_tree(Path(facts.repo_root) / "src"),
                               invoked_modules(Path(facts.repo_root)))
    dead = sorted(name for name, state in verdict.items() if state == UNREACHABLE)
    audits = audit_claims(Path(facts.repo_root), Path(facts.ledger_root))
    unsupported = [a for a in audits if a.is_unsupported]

    census = (f"{len(dead)} unreachable module(s) of {len(verdict)}")
    if not unsupported:
        return ProbeResult(
            OK, f"no ledger row claims BUILT for unreachable code; {census}", proof)

    named = ", ".join(a.row_id for a in unsupported[:6])
    more = f" and {len(unsupported) - 6} more" if len(unsupported) > 6 else ""
    return ProbeResult(
        DEGRADED,
        f"{len(unsupported)} ledger row(s) claim BUILT for code nothing reaches "
        f"({named}{more}); {census}", proof)


# Below this share of modules judged, the record is a sample rather than a
# standard. Not a target - §1a.6 says NO module ships without a verdict, so the
# only passing number is all of them. This is where the tile stops calling a
# partial record partial and starts calling it absent.
_AXIS_COVERAGE_FLOOR = 0.50


def probe_axis_verdicts(facts: SystemFacts) -> ProbeResult:
    """§1a.6: which modules carry a learning / reasoning / depth verdict.

    The verdict is a judgement and this tile does not render it as a measurement.
    What it renders is coverage - counted from `src/`, so a module added without
    a verdict lowers the number rather than going unnoticed - and whether every
    named artifact is on disk.

    A verdict citing a test nobody wrote is FAILING rather than partial. It is
    the same defect as a ledger row citing a module nobody calls, and this board
    already refuses that one.
    """
    proof = "integrity.axis_verdicts over src/ + docs/axis-verdicts.json"
    from integrity.axis_verdicts import assess

    coverage = assess(Path(facts.repo_root))
    if coverage.total == 0:
        return ProbeResult(NOT_MEASURED, "no modules found to judge", proof)

    broken = []
    if coverage.incomplete:
        broken.append(f"{len(coverage.incomplete)} with an axis left blank")
    if coverage.invalid:
        broken.append(f"{len(coverage.invalid)} with a word that is not a verdict")
    if coverage.missing_evidence:
        broken.append(f"{len(coverage.missing_evidence)} citing evidence not on disk")
    if coverage.orphaned:
        broken.append(f"{len(coverage.orphaned)} judging a module that no longer exists")

    census = (f"{len(coverage.verdicted)} of {coverage.total} modules judged "
              f"({coverage.share:.0%})")
    if broken:
        return ProbeResult(FAILING, f"{census}; " + ", ".join(broken), proof)
    if coverage.failing:
        return ProbeResult(
            DEGRADED,
            f"{census}; {len(coverage.failing)} module(s) carry an explicit FAIL: "
            f"{', '.join(sorted(coverage.failing))}", proof)
    if coverage.share < _AXIS_COVERAGE_FLOOR:
        # Not green, and not PARTIAL either. §1a.6 says no module ships without
        # a verdict, so most of this repo is unjudged rather than half-judged -
        # and absence of evidence renders as its own state.
        return ProbeResult(
            NOT_MEASURED,
            f"{census}; the rest are unjudged, not passing", proof)
    return ProbeResult(
        PARTIAL if coverage.unverdicted else OK,
        census + (f"; {len(coverage.unverdicted)} still unjudged"
                  if coverage.unverdicted else "; every module judged"), proof)


def probe_feature_staleness(facts: SystemFacts) -> ProbeResult:
    """Does every feature value carry the age of what it was computed from?

    Measured by RUNNING each feature and reading its columns, not by checking
    that `features/staleness.py` exists. A contract nothing was checked against
    is a docstring, and this tile is the check.

    The freshness verdicts are reported as well as the coverage, because that
    is the number this exists to surface: a feature computing happily off
    inputs that stopped arriving hours ago is the "confident staleness" the
    goal spec names as the failure the whole intelligence standard is designed
    against.
    """
    proof = "features.staleness columns on each feature's live output"
    from features.staleness import COLUMNS, STALE

    now_ns = int(time.time() * 1e9)
    store_root = facts.capture_root / "store"
    features = {}
    try:
        from features.spot_perp_basis import compute_spot_perp_basis
        from features.term_structure import compute_term_structure
        features["spot_perp_basis"] = compute_spot_perp_basis(store_root, now_ns).rows
        features["term_structure"] = compute_term_structure(store_root, now_ns).rows
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"a feature failed to compute: {error}",
                           proof)

    unstamped = sorted(name for name, rows in features.items()
                       if not set(COLUMNS) <= set(rows.columns))
    if unstamped:
        return ProbeResult(
            FAILING,
            f"{len(unstamped)} feature(s) emit values with no staleness stamp: "
            f"{', '.join(unstamped)}",
            proof)

    counts: dict[str, int] = {}
    for rows in features.values():
        if rows.empty:
            continue
        for verdict, count in rows["freshness"].value_counts().items():
            counts[str(verdict)] = counts.get(str(verdict), 0) + int(count)
    if not counts:
        return ProbeResult(
            PARTIAL,
            f"{len(features)} feature(s) carry the stamp; none produced a value "
            f"to stamp at this clock",
            proof)

    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    stale = counts.get(STALE, 0)
    total = sum(counts.values())
    detail = (f"{len(features)} feature(s) stamped, {total} value(s): {summary}")
    if stale:
        return ProbeResult(DEGRADED,
                           f"{detail} - {stale} computed from inputs older than "
                           f"their own cadence allows", proof)
    return ProbeResult(OK, detail, proof)


def probe_exchange_reserves(facts: SystemFacts) -> ProbeResult:
    """What the venues we trade on are holding, and how much of it they printed.

    Reads the stored polls rather than fetching: a tile that made a network
    call would report the source's health, not this system's.

    OK requires the venues this system actually trades on to be in the reading.
    A dataset covering 78 exchanges while missing binance would be a full board
    about somebody else's risk.
    """
    proof = "store/exchange_reserves (defillama CEX transparency)"
    import pandas as pd

    from store.clock_gated_reader import ClockGatedReader
    from store.exchange_reserves import DATASET
    from store.temporal_schema import AVAILABILITY_TIME

    store_root = Path(facts.capture_root) / "store"
    if not (store_root / DATASET).is_dir():
        return ProbeResult(NOT_BUILT, "no exchange-reserve poll stored", proof)

    frame = ClockGatedReader(store_root, DATASET).read_as_of(int(time.time() * 1e9))
    if frame.empty:
        return ProbeResult(NOT_BUILT, "reserve dataset exists and is empty", proof)

    newest_ns = int(frame[AVAILABILITY_TIME].max())
    latest = frame[frame[AVAILABILITY_TIME] == newest_ns]
    age_hours = (time.time() * 1e9 - newest_ns) / 3_600_000_000_000

    traded_here = latest[latest["exchange"].isin(["Binance", "Bybit"])]
    if traded_here.empty:
        return ProbeResult(
            PARTIAL,
            f"{len(latest)} exchange(s) measured but none of the venues this "
            f"system trades on",
            proof)

    lines = []
    for row in traded_here.sort_values("exchange").itertuples(index=False):
        share = getattr(row, "own_token_share")
        share_text = ("own-token share unmeasured" if share is None or pd.isna(share)
                      else f"{share:.1%} of it its own token")
        lines.append(f"{row.exchange} ${row.total_reserve_usd / 1e9:.1f}bn, "
                     f"{share_text}, 24h netflow "
                     f"${row.netflow_24h_usd / 1e6:+.1f}m")
    detail = (f"{len(latest)} exchange(s) in the newest poll, "
              f"{age_hours:.1f}h old. " + "; ".join(lines))

    # A reading nobody refreshed is a snapshot presented as live. The source
    # updates daily, so a day and a half without one is stale rather than slow.
    if age_hours > 36:
        return ProbeResult(DEGRADED, f"{detail} - STALE, no poll in over 36h", proof)
    return ProbeResult(OK, detail, proof)


def probe_promotion_readiness(facts: SystemFacts) -> ProbeResult:
    """How far the observed record is from being able to support a promotion.

    The decision of 2026-08-09 was to wait for observed history rather than
    build Phase 5. A wait with no measured end is indefinite by construction, so
    this is the end, measured: days accumulated against the days MinBTL will
    demand at the current trial count.

    Two things about the number will surprise a reader who does not know them,
    so the tile says both. It RECEDES as the Trial Registry grows, because N only
    rises and MinBTL rises with it. And it will DROP sharply around day 30, when
    the observed window first supports an effective-breadth estimate and the gate
    stops failing closed to calendar days - the projection before then is the
    conservative one, not the expected one.
    """
    proof = "validation.promotion_pipeline.readiness over the observed funding dataset"
    from store.clock_gated_reader import ClockGatedReader
    from validation.promotion_pipeline import (
        MIN_DAYS_FOR_BREADTH, OBSERVED_FUNDING, readiness)
    from validation.trial_registry import TrialRegistry

    store_root = Path(facts.capture_root) / "store"
    if not (store_root / OBSERVED_FUNDING).is_dir():
        return ProbeResult(NOT_MEASURED, "no observed funding dataset to measure", proof)

    frame = ClockGatedReader(store_root, OBSERVED_FUNDING).read_as_of(2**62)
    if frame.empty:
        return ProbeResult(NOT_MEASURED, "observed funding dataset is empty", proof)

    state = readiness(frame, TrialRegistry(Path(facts.capture_root) / "trials"))
    census = (f"{state['days_observed']} observed day(s) of "
              f"{state['days_required']:.0f} required for a "
              f"Sharpe-{state['target_sharpe']:.1f} claim at N={state['n_trials']}")

    if state["ready"]:
        return ProbeResult(OK, f"{census} - the record can support a promotion", proof)
    if state["days_observed"] < MIN_DAYS_FOR_BREADTH:
        # The conservative projection, and it must not read as the expectation.
        return ProbeResult(
            PARTIAL,
            f"{census}. Under {MIN_DAYS_FOR_BREADTH} days the effective-breadth "
            f"estimate fails closed to calendar days, so this figure is the "
            f"CONSERVATIVE bound and will fall sharply once breadth is "
            f"measurable", proof)
    return ProbeResult(
        PARTIAL,
        f"{census}; {state['days_remaining']:.0f} to go at effective breadth "
        f"{state['effective_breadth']:.0f}. Recedes as N grows", proof)


@functools.lru_cache(maxsize=4)
def _module_references(repo_root: Path) -> dict[str, frozenset[str]]:
    """Every module's referenced names, read once per wall build.

    Cached because six phase-B tiles ask the same question of the same tree, and
    parsing `src/` once per tile would make the board's cost scale with how many
    features it reports on.
    """
    from integrity.unsupported_claims import read_source_tree
    return {name: facts.references
            for name, facts in read_source_tree(Path(repo_root) / "src").items()}


def _consumers_of(repo_root: Path, module: str, entry_point: str) -> list[str]:
    """Which modules besides its own reach this feature's entry point.

    MEASURED, not declared. Every phase-B feature shipped with an axis verdict
    of `depth: fail` reasoning that nothing reads it - but a verdict is a
    sentence somebody wrote, and it goes stale silently the day a consumer
    appears. This asks the source tree instead, so the tile changes on the day
    the code does rather than on the day someone remembers this file.

    **`statuswall` is not a consumer, and excluding it is the whole point.** The
    probe below has to call the feature in order to measure it, so its own call
    lands in the source tree as a reference - and on the first run, five of six
    features reported "read by statuswall.evidence" and graded OK. A board that
    measures a thing and counts its own measurement as the thing being used is
    a tile that certifies itself, which is the failure Rule 8 exists to prevent
    wearing the costume of the fix for it.
    """
    return sorted(name for name, references in _module_references(repo_root).items()
                  if name != module
                  and not name.startswith("statuswall.")
                  and entry_point in references)


def _probe_computed_feature(facts: SystemFacts, *, module: str, entry_point: str,
                            compute, unit: str) -> ProbeResult:
    """Run one phase-B feature against the live store and report what it did.

    Shared by six tiles rather than copied into each. `what_makes_code_deep.md`'s
    D4 names many near-identical shallow modules as the failure pattern
    generated code falls into, and six probes differing only in which function
    they call would be exactly that.

    The grading, and why it is not simply "did it return rows":

    * **Rows, and something reads them** -> OK. The only green here.
    * **Rows, and nothing reads them** -> PARTIAL. A value computed for no
      consumer cannot change what the system does when it is wrong, which is
      §1a.5's master test, so it is not finished merely because it runs. Same
      posture the `stablecoin peg monitor` tile already takes.
    * **No rows, but counted refusals** -> DEGRADED, naming the biggest reason.
      This is the state that matters most: the module is working exactly as
      designed and there is nothing for it to work on. `realized_volatility`
      refusing every window because bars are hours stale must not look like a
      module that was never built, and must not look healthy either.
    * **No rows and no refusals** -> NOT_MEASURED. Nothing reached it at all.
    * **Raised** -> NOT_MEASURED, carrying the error. A tile that swallowed the
      exception would render a broken feature as merely quiet.
    """
    proof = f"{module}.{entry_point} run against the live store"
    try:
        table = compute(facts.capture_root / "store", int(time.time() * 1e9))
    except Exception as error:
        return ProbeResult(NOT_MEASURED, f"{entry_point} raised: {error}", proof)

    rows = len(table.rows)
    refused = {reason: count for reason, count in table.refused.items() if count}
    refusal_note = (", ".join(f"{reason} {count}" for reason, count in
                              sorted(refused.items(), key=lambda kv: -kv[1])))

    if rows == 0 and not refused:
        return ProbeResult(NOT_MEASURED,
                           f"no input reached {module} at this clock", proof)
    if rows == 0:
        biggest = max(refused.items(), key=lambda kv: kv[1])
        return ProbeResult(
            DEGRADED,
            f"computed nothing: every candidate refused ({refusal_note}). "
            f"Dominant reason {biggest[0]} on {biggest[1]} - the module is "
            f"working and its inputs are not", proof)

    consumers = _consumers_of(facts.repo_root, module, entry_point)
    detail = f"{rows} {unit}"
    if refused:
        detail += f"; refused {refusal_note}"
    if not consumers:
        return ProbeResult(
            PARTIAL,
            f"{detail}. Nothing consumes it - no module outside {module} calls "
            f"{entry_point}, so it cannot yet change what the system does when "
            f"it is wrong", proof)
    return ProbeResult(OK, f"{detail}; read by {', '.join(consumers)}", proof)


def probe_realized_volatility(facts: SystemFacts) -> ProbeResult:
    from features.realized_volatility import compute_realized_volatility
    return _probe_computed_feature(
        facts, module="features.realized_volatility",
        entry_point="compute_realized_volatility",
        compute=compute_realized_volatility,
        unit="(venue, symbol, horizon) volatilities")


def probe_microprice(facts: SystemFacts) -> ProbeResult:
    from features.microprice import compute_microprice
    return _probe_computed_feature(
        facts, module="features.microprice", entry_point="compute_microprice",
        compute=compute_microprice, unit="depth-weighted microprices")


def probe_order_flow_imbalance(facts: SystemFacts) -> ProbeResult:
    from features.order_flow_imbalance import compute_order_flow_imbalance
    return _probe_computed_feature(
        facts, module="features.order_flow_imbalance",
        entry_point="compute_order_flow_imbalance",
        compute=compute_order_flow_imbalance,
        unit="depth-weighted book imbalances")


def probe_absorption(facts: SystemFacts) -> ProbeResult:
    from features.absorption import compute_absorption
    return _probe_computed_feature(
        facts, module="features.absorption", entry_point="compute_absorption",
        compute=compute_absorption, unit="bars judged for absorption")


def probe_kyle_lambda(facts: SystemFacts) -> ProbeResult:
    from features.kyle_lambda import compute_kyle_lambda
    return _probe_computed_feature(
        facts, module="features.kyle_lambda", entry_point="compute_kyle_lambda",
        compute=compute_kyle_lambda, unit="fitted price-impact slopes")


def probe_fractional_differentiation(facts: SystemFacts) -> ProbeResult:
    from features.fractional_differentiation import compute_fractional_differentiation
    return _probe_computed_feature(
        facts, module="features.fractional_differentiation",
        entry_point="compute_fractional_differentiation",
        compute=compute_fractional_differentiation,
        unit="series differenced at a searched d")


def probe_status_wall(facts: SystemFacts) -> ProbeResult:
    """This board, reporting on itself. It exists, so it says so."""
    return ProbeResult(
        BUILT,
        "this board; generated from probes, never hand-written",
        "src/statuswall/",
    )


# The supervisor polls every 60s, so five minutes is four missed polls: long
# enough not to flap on a slow universe-wide read, short enough that a dead engine
# is visible within one coffee. Shared with build_progress's Phase J row so the
# two boards cannot disagree about whether the engine is alive.
_PAPER_HEARTBEAT_STALE_HOURS = 300 / 3600
# Where a participation calibration receipt lands, mirroring the fee receipt at
# ~/capture/fee-verification/latest.json. Declared here rather than derived, so a
# probe cannot report a clean bill of health by looking somewhere nothing writes.
PARTICIPATION_RECEIPT_DIR = "participation-calibration"


def probe_paper_engine(facts: SystemFacts) -> ProbeResult:
    """Is the forward paper engine running, and what has it actually done?

    Graded on the engine's own heartbeat and its AGE — never on the presence of
    the code, and never on the presence of a file. Rule 8's whole point: this
    tile must be able to say the engine died, and a tile that goes green because
    a module exists can never say that.

    The states are kept apart because a board that cannot distinguish them is
    worse than no board. An engine running with nothing to trade and an engine
    that stopped five days ago produce identical fill counts; only the heartbeat
    age separates them, and this box spent 2026-08-10 to 2026-08-15 proving it.
    """
    from paper.forward_journal import count_fills, read_heartbeat

    journal_dir = facts.capture_root / "paper" / "forward"
    proof = str(journal_dir)
    if not journal_dir.is_dir():
        return ProbeResult(NOT_BUILT, "no forward journal; the engine has never "
                                      "run on this machine", proof)

    beat = read_heartbeat(journal_dir)
    if beat is None:
        return ProbeResult(
            NOT_MEASURED,
            "journal directory exists but holds no readable heartbeat - the "
            "engine has not completed a poll, and nothing here says it works",
            proof)

    age_hours = beat.age_ns(time.time_ns()) / 3_600_000_000_000
    claim = "claims edge" if beat.makes_edge_claim else "makes NO edge claim"
    fills = count_fills(journal_dir)
    summary = (f"strategy {beat.strategy!r} ({claim}); {beat.events_fed} event(s) "
               f"fed, {beat.orders_submitted} submitted, {beat.open_orders} "
               f"resting, {fills} fill(s) journalled")

    if age_hours > _PAPER_HEARTBEAT_STALE_HOURS:
        return ProbeResult(
            STOPPED,
            f"last heartbeat {age_hours:.1f}h ago - the engine is not running. "
            f"{summary}", proof)
    if beat.makes_edge_claim is False:
        # Running, and deliberately not OK. The engine is doing its job, but the
        # only signal it can run is a plumbing signal, and a green tile over that
        # would be read six weeks from now as "paper trading is working" in the
        # sense that matters. PARTIAL is the honest state until a model from
        # Phase C is the thing being journalled.
        return ProbeResult(
            PARTIAL,
            f"running, but on a signal that claims no edge - its P&L is not a "
            f"result. {summary}", proof)
    return ProbeResult(OK, summary, proof)


def probe_participation_calibration(facts: SystemFacts) -> ProbeResult:
    """Has the fill model's participation rate been measured, or is it declared?

    NOT_MEASURED when no receipt exists, which is the state today. That is the
    difference between a fraction measured from resting size at the touch and a
    fraction somebody typed, and it is exactly the difference that decides whether
    a paper result may be promoted. A tile that could not tell them apart would
    let an assumed number be read as a measured one - which the design document
    names as the single largest lever in the engine.
    """
    receipt = facts.capture_root / PARTICIPATION_RECEIPT_DIR / "latest.json"
    proof = str(receipt)
    if not receipt.is_file():
        return ProbeResult(
            NOT_MEASURED,
            "no calibration receipt - participation is a declared number, every "
            "paper fill carries uncalibrated=true, and no tier-2 promotion may "
            "read one",
            proof)
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        symbols = payload["symbols"]
        measured_at_ns = int(payload["measured_at_ns"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ProbeResult(
            FAILING,
            f"calibration receipt present but unreadable ({type(exc).__name__}: "
            f"{exc}) - refusing to report a rate nobody can check",
            proof)

    observations = sum(int(s.get("n_observations", 0)) for s in symbols.values()
                       if isinstance(s, dict))
    age_hours = (time.time_ns() - measured_at_ns) / 3_600_000_000_000
    detail = (f"{len(symbols)} symbol(s), {observations} observation(s), measured "
              f"{age_hours:.1f}h ago")
    if observations == 0:
        return ProbeResult(
            NOT_MEASURED,
            f"receipt exists and rests on nothing: {detail}. A receipt with no "
            f"observations is a declared number wearing a measurement's clothes",
            proof)
    return ProbeResult(OK, detail, proof)


def probe_naive_baseline_gate(facts: SystemFacts) -> ProbeResult:
    """Does the mandatory naive-baseline gate actually refuse?

    This tile EXERCISES the control rather than asserting it exists. The
    difference is the whole of Rule 8 and this repo has been caught by it before:
    the wall once rendered "auto-halt armed" over a registry whose `observe()`
    had no caller. A gate is not a file, it is a refusal, and the only honest way
    to report one is to ask it to refuse.

    So: hand `_baseline_gate` no comparison and check it says no. If that ever
    returns a pass, MD-001's "mandatory" has quietly become optional and this
    tile goes FAILING - loudly, and before a model is promoted on a check nobody
    ran.
    """
    from validation.promotion_gate import _baseline_gate

    absent = _baseline_gate(None)
    if absent.passed:
        return ProbeResult(
            FAILING,
            "the naive-baseline gate PASSED with no comparison supplied - "
            "MD-001's mandatory check has become optional, and a model can now "
            "be promoted on an overfitting check nobody ran",
            "src/validation/promotion_gate.py:_baseline_gate")
    return ProbeResult(
        OK,
        f"mandatory and armed: with no comparison supplied the gate refuses "
        f"(measured p={absent.measured}, threshold {absent.threshold}). Exercised "
        f"on this pass, not inferred from the file existing",
        "src/models/naive_baseline.py")


def probe_outage_detection(facts: SystemFacts) -> ProbeResult:
    """Is the system recording its own liveness, and what gaps has it found?

    The tile reports the outages rather than hiding them once they end. A gap in
    the tape is a permanent fact about every model later trained on it, and the
    five days this box spent off in August 2026 are exactly the kind of thing
    that gets rediscovered months later as an unexplained hole.

    DEGRADED while outages are on record, not FAILING and not OK. They are real
    and they are over: OK would erase them, and FAILING would imply something is
    wrong right now when what is wrong already happened.
    """
    from ops.liveness_ledger import last_seen_ns, read_outages

    root = facts.capture_root / "liveness"
    proof = str(root)
    stamp = last_seen_ns(root)
    if stamp is None:
        return ProbeResult(
            NOT_MEASURED,
            "no liveness stamp - nothing is recording whether this system is "
            "running, so an outage would leave no trace but a hole in the tape",
            proof)

    age_hours = (time.time_ns() - stamp) / 3_600_000_000_000
    outages = read_outages(root)
    if age_hours > 1.0:
        return ProbeResult(
            STOPPED,
            f"the liveness stamp is {age_hours:.1f}h old - whatever writes it is "
            f"not running, which is the one failure this feature exists to catch",
            proof)
    if outages:
        worst = max(outages, key=lambda o: o.duration_ns)
        observed = sum(1 for o in outages if not o.is_reconstructed)
        reconstructed = len(outages) - observed
        # Counted apart on the tile for the same reason bars_reconstructed_* is a
        # separate dataset: a gap this system watched happen and one worked out
        # afterwards from boot logs are different claims, and a single number
        # would let the weaker one be read as the stronger.
        breakdown = (f"{observed} observed, {reconstructed} reconstructed"
                     if reconstructed else f"{observed} observed")
        return ProbeResult(
            DEGRADED,
            f"{len(outages)} outage(s) on record ({breakdown}), longest "
            f"{worst.duration_hours:.1f}h - the tape has holes and every model "
            f"trained on it inherits them",
            proof)
    return ProbeResult(
        OK, f"liveness stamped {age_hours * 60:.0f} min ago; no outages recorded",
        proof)


# Feature key -> probe. A feature absent from this map has no measurement and is
# therefore NOT_BUILT. Adding a row here is a claim that something is real, and
# the probe is what has to defend it.
PROBES = {
    "linear naive baseline mandatory": probe_naive_baseline_gate,
    "outage detection the system s record of its own absence":
        probe_outage_detection,
    "paper execution engine forward journal both accountings": probe_paper_engine,
    "participation rate calibrated from the depth archive":
        probe_participation_calibration,
    "exchange reserve netflow": probe_exchange_reserves,
    "feature staleness timestamp on every value": probe_feature_staleness,
    "spot ohlcv trade tape multi venue": probe_trade_tape,
    "l2 order book depth 20 50 levels": probe_l2_depth,
    "liquidation feed": probe_liquidation_feed,
    "open interest": probe_open_interest,
    "perpetual funding rate history schedule": probe_funding_rates,
    "spot perp basis term structure": probe_spot_perp_basis,
    "stablecoin peg monitor": probe_peg_monitor,
    "cross venue consolidated price liquidity weighted": probe_consolidated_price,
    "wash trading discount on reported volume": probe_wash_trading_discount,
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
    "bounded writer descriptor pool": probe_writer_descriptor_pool,
    "reachability audit of every built claim": probe_unsupported_claims,
    "learning reasoning depth verdict per module": probe_axis_verdicts,
    "observed history sufficient for a promotion": probe_promotion_readiness,
    # Phase B. Each runs its feature against the live store; none of them is
    # green today, because nothing reads what they compute - see
    # `_probe_computed_feature` for why that is PARTIAL rather than OK.
    "realized volatility multi horizon": probe_realized_volatility,
    "microprice": probe_microprice,
    "depth weighted order flow imbalance": probe_order_flow_imbalance,
    "absorption detection delta vs price hold": probe_absorption,
    "kyle s lambda": probe_kyle_lambda,
    "fractional differentiation": probe_fractional_differentiation,
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
