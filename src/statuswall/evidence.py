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
from store.hourly_migration import BUILDING_SUFFIX, LEGACY_SUFFIX

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


# --- measurements too expensive to repeat every five minutes ---------------
#
# **Measured 2026-08-19.** Five probes on this wall each call
# `ClockGatedReader.read_as_of(2**62)`, which materialises a WHOLE dataset -
# bars, book, funding - and the boards supervisor runs the wall every five
# minutes. One pass was still reading `store/funding/symbol=.../part-*.parquet`
# after 25 minutes at 2.5 GB and climbing; earlier passes reached 11.7 GB on a
# 30 GB box and were OOM-killed three times in one morning. The board that
# reports whether the system is healthy was the thing taking it down.
#
# The numbers those probes produce move on the order of a day - observed days of
# history, symbols in the store, bar validity across the archive. Re-deriving
# them every five minutes buys nothing and costs the board.
#
# So an expensive measurement is taken on its own cadence and the tile SAYS WHEN
# IT WAS TAKEN. That is the Rule 8 line: a cached measurement is still a
# measurement as long as its age is on the face of it, and this appends the age
# to the detail rather than presenting an old number as a new one. A cache that
# hid its age would be an assertion.
PROBE_CACHE_DIR = Path.home() / "capture" / "boards" / "probe-cache"
# Six hours. Longer than any board pass, far shorter than the day these numbers
# actually move on.
EXPENSIVE_TTL_S = 6 * 3600.0


def measured_periodically(key: str, compute, ttl_s: float = EXPENSIVE_TTL_S,
                          cache_dir: Path | None = None) -> ProbeResult:
    """Run `compute` at most once per `ttl_s`, and age the answer on its face.

    A compute that raises is NOT swallowed - it belongs to `assess`, which turns
    it into a visible FAILING tile. Only a successful measurement is stored, so
    a cache can never hold a result nothing measured.
    """
    directory = cache_dir or PROBE_CACHE_DIR
    path = directory / f"{key}.json"
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
        age_s = time.time() - float(held["measured_at_s"])
        if 0 <= age_s < ttl_s:
            hours = age_s / 3600.0
            stamp = (f"measured {age_s / 60:.0f} min ago" if hours < 1
                     else f"measured {hours:.1f}h ago")
            return ProbeResult(held["state"], f"{held['detail']} ({stamp})",
                               held["proof"])
    except (OSError, ValueError, KeyError, TypeError):
        pass                      # no usable cache is not an error, it is a miss

    result = compute()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "state": result.state, "detail": result.detail, "proof": result.proof,
            "measured_at_s": time.time()}), encoding="utf-8")
    except OSError:
        pass                      # an unwritable cache costs speed, never truth
    return result


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


def _probe_consolidated_price_now(facts: SystemFacts) -> ProbeResult:
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
    # The migration of SL-15 leaves two more directories whose names begin
    # "bars_": the half-built copy and the retired layout kept beside it. Both
    # hold real parquet, so a prefix match alone would count the same symbols
    # twice and grade the wall on a dataset nothing writes to any more - a tile
    # going stale because the store it names was deliberately retired.
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("bars_")
                  and not p.name.endswith((BUILDING_SUFFIX, LEGACY_SUFFIX)))


# Bars build closed days only, so the newest partition legitimately trails the tape
# by up to a day plus however long the build takes. The threshold sits above that,
# so a normal lag is not reported as a stopped pipeline - and a genuinely abandoned
# store still surfaces within a day and a half rather than never.
_STORE_STALE_HOURS = 36.0
# Below this share of the captured symbols the store is not a smaller store, it is
# a different one from the archive beside it.
_STORE_COVERAGE_FLOOR = 0.9


def _store_symbols(datasets: list[Path]) -> int:
    """Symbols the store holds bars for, counted from its own partition layout.

    `symbol=` sits one level below the availability hour since SL-15, so the
    count is over directory names at any depth rather than at the top: a
    top-level count would have silently become zero the moment the layout
    changed, and a store tile reading "0 symbols" is the sort of wrong that
    looks like a data loss.
    """
    return len({part.name for dataset in datasets for part in dataset.glob("*/symbol=*")
                if part.is_dir()})


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


def _probe_bitemporal_store_now(facts: SystemFacts) -> ProbeResult:
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


def _probe_bar_price_validity_now(facts: SystemFacts) -> ProbeResult:
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


def _probe_clock_gated_access_now(facts: SystemFacts) -> ProbeResult:
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


def _probe_promotion_readiness_now(facts: SystemFacts) -> ProbeResult:
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


def _exercise_exit_policy():
    """Run the four rails and the ratchet once. Shared by two tiles.

    Two catalogue rows are satisfied by one module - the deterministic exit
    policy and the ratchet inside it - and running the same exercise twice would
    be two probes disagreeing about one measurement the first time either drifted.
    """
    from decimal import Decimal

    from strategy.deterministic_exit import (
        HARD_STOP, LONG, PROFIT_LOCK, PROFIT_TARGET, SHORT, VERTICAL_BARRIER,
        Bar, ExitPolicy, ratchet_profit_lock, run_exit_policy,
    )

    def bar(high, low, close):
        return Bar(high=Decimal(str(high)), low=Decimal(str(low)),
                   close=Decimal(str(close)))

    entry, atr = Decimal(100), Decimal(1)
    results = {
        "hard_stop": run_exit_policy(LONG, entry, [bar(101, 96, 97)], atr),
        "target": run_exit_policy(LONG, entry, [bar(105, 100, 104)], atr),
        "lock": run_exit_policy(
            LONG, entry,
            [bar(103, 100, 102.5), bar(103, 101, 101.5), bar(102, 100, 100.5)],
            atr),
        "vertical": run_exit_policy(
            LONG, entry, [bar(100, 100, 100)] * 8, atr,
            ExitPolicy(max_holding_bars=4)),
        "ambiguous": run_exit_policy(LONG, entry, [bar(105, 96, 100)], atr),
    }
    # The monotone property, exercised rather than trusted: a helpful branch
    # added later is how "it never widens" dies.
    lock, widened = None, False
    for candidate in (Decimal(90), Decimal(95), Decimal(80), Decimal(97),
                      Decimal(10)):
        new = ratchet_profit_lock(LONG, lock, candidate)
        widened = widened or (lock is not None and new < lock)
        lock = new
    short_lock, short_widened = None, False
    for candidate in (Decimal(110), Decimal(105), Decimal(120)):
        new = ratchet_profit_lock(SHORT, short_lock, candidate)
        short_widened = short_widened or (short_lock is not None
                                          and new > short_lock)
        short_lock = new
    return results, (widened or short_widened), lock


def probe_ratchet_profit_lock(facts: SystemFacts) -> ProbeResult:
    """Is the lock still monotone?

    MD-017, and the one property the spec states absolutely: *"It never widens,
    under any condition, for any model output. A lock that can loosen is not a
    lock."* Exercised on every board pass rather than asserted, because the way
    this dies is a helpful branch somebody adds later - and a lock that widened
    once, on one path, leaves no trace in any output.
    """
    proof = "src/strategy/deterministic_exit.py"
    try:
        _results, widened, final_lock = _exercise_exit_policy()
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the exit policy raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    if widened:
        return ProbeResult(
            FAILING,
            "THE PROFIT LOCK WIDENED. It moved away from the position on at "
            "least one path, which the spec forbids under any condition - a lock "
            "that can loosen is not a lock, and the loss it fails to prevent "
            "leaves no trace in any output", proof)
    return ProbeResult(
        PARTIAL,
        f"monotone across a falling-then-rising candidate sequence on both "
        f"sides; the long lock ends at {final_lock}, the highest candidate ever "
        f"seen, and never moved down. Volatility-scaled by construction - every "
        f"distance is a multiple of the ATR measured at entry, never a percent. "
        f"Exercised on this pass. No venue-side mirror: MD-018 needs an order "
        f"path this system does not have, so a lock held only in memory is what "
        f"exists", proof)


def probe_deterministic_exit(facts: SystemFacts) -> ProbeResult:
    """Are all four rails still reachable, and does the adverse one still win?

    A rail nothing can trigger is a rail that is not in the policy, so the probe
    drives each of the four and fails if any stops firing. And a bar that touched
    both the target and the hard stop is resolved AGAINST the position - assuming
    the favourable one is worth between a few basis points and the whole trade,
    on exactly the bars that matter most, and it would show up as a better
    result.
    """
    from strategy.deterministic_exit import (
        HARD_STOP, PROFIT_LOCK, PROFIT_TARGET, VERTICAL_BARRIER,
    )

    proof = "src/strategy/deterministic_exit.py"
    try:
        results, _widened, _lock = _exercise_exit_policy()
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the exit policy raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    expected = {"hard_stop": HARD_STOP, "target": PROFIT_TARGET,
                "lock": PROFIT_LOCK, "vertical": VERTICAL_BARRIER}
    broken = [f"{name} exited on {results[name].exit_reason} rather than "
              f"{reason}" for name, reason in expected.items()
              if results[name].exit_reason != reason]
    ambiguous = results["ambiguous"]
    if ambiguous.exit_reason != HARD_STOP:
        broken.append(
            f"a bar touching BOTH rails exited on {ambiguous.exit_reason} rather "
            f"than the hard stop - the ambiguity is being resolved in the "
            f"position's favour, which reads as a better result")
    if not ambiguous.is_ambiguous:
        broken.append("a bar touching both rails was not marked ambiguous")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    lock_record = results["lock"]
    return ProbeResult(
        PARTIAL,
        f"all four rails reachable and the ambiguous bar resolved against the "
        f"position; the lock case kept {lock_record.pnl_per_unit} per unit and "
        f"recorded MFE {lock_record.max_favourable_excursion} against MAE "
        f"{lock_record.max_adverse_excursion}, which is the path shape "
        f"PROFIT-TAIL trains on. Exercised on this pass - nothing routes live "
        f"positions through it yet, so no exit record has been generated from "
        f"the market", proof)


def probe_pre_trade_gate(facts: SystemFacts) -> ProbeResult:
    """Are the limits on disk, and does the gate still refuse and still allow?

    Two failures, opposite and both silent. A gate that blocks nothing is the
    state this system was in until 2026-08-16 - 363 open positions, nothing
    saying no. A gate that blocks everything looks like caution and is an outage.
    Both are exercised on every pass.

    The third check is the one found by a test: a position already over its cap
    must still be reducible. Blocking a reducing order traps the book at its own
    limit, in the one state where an exit matters most.
    """
    from decimal import Decimal

    from risk.pre_trade_gate import (
        LimitsNotSet, PreTradeGate, RejectionReason, read_gate_limits,
    )

    risk_root = facts.capture_root / "risk"
    proof = str(risk_root / "paper-gate.json")
    try:
        limits, nav = read_gate_limits(risk_root)
    except LimitsNotSet as error:
        return ProbeResult(
            NOT_MEASURED,
            f"no gate limits on disk, so the engine runs UNGATED: {error}",
            proof)

    gate = PreTradeGate(limits)
    price = Decimal("100")
    common = dict(venue="binance", symbol="PROBEUSDT",
                  reference_price=price, limit_price=None, nav=nav,
                  now_ns=int(time.time() * 1e9))
    try:
        allowed = gate.evaluate(side="BUY", quantity=Decimal("0.001"),
                                open_positions={}, **common)
        blocked = gate.evaluate(
            side="BUY", quantity=limits.max_order_notional / price * 10,
            open_positions={}, **common)
        over_cap = limits.max_position_notional / price * 10
        reducing = gate.evaluate(
            side="SELL", quantity=Decimal("0.001"),
            open_positions={("binance", "PROBEUSDT"): over_cap}, **common)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the gate raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if not allowed.approved:
        broken.append(
            f"a tiny order inside every limit was REFUSED "
            f"({', '.join(r.value for r in allowed.reasons)}) - a gate that "
            f"blocks everything looks like caution and is an outage")
    if blocked.approved:
        broken.append(
            "an order ten times the notional cap was APPROVED - the gate is "
            "blocking nothing, which is the state that produced 363 unbounded "
            "positions")
    if RejectionReason.POSITION_CAP_EXCEEDED in reducing.reasons:
        broken.append(
            "a REDUCING order against an over-cap position was blocked by the "
            "position cap - the book is trapped at its own limit, in the one "
            "state where an exit matters most")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"limits on disk (NAV {nav}, max {limits.max_open_instruments} "
        f"instruments, {limits.max_order_notional} per order, "
        f"{limits.max_orders_per_window} orders per "
        f"{limits.order_rate_window_ns // 1_000_000_000}s): a compliant order "
        f"passes, one ten times the cap is refused, and an over-cap position can "
        f"still be reduced. The numbers are a SEEDED PROPOSAL the user has not "
        f"confirmed, and daily-loss and drawdown kills are not here - they are "
        f"risk/tail-cap.json", proof)


def probe_paper_blotter(facts: SystemFacts) -> ProbeResult:
    """Open positions and closed round trips, as the journal actually has them.

    This tile exists to answer a question directly: what has paper trading
    actually done. So it reports the counts rather than a health verdict, and
    the state reflects what the numbers mean.

    DEGRADED when the journal holds only one side. That is not a broken engine -
    `plumbing-momentum` rests a bid and never sells - but a blotter showing zero
    closed trades against thousands of fills is a system that cannot yet
    demonstrate a single completed trade, and a green tile would say the opposite.
    """
    from paper.blotter import OPTIMISTIC, PESSIMISTIC, read_blotter

    proof = str(facts.capture_root / "paper" / "forward")
    try:
        view = read_blotter(facts.capture_root)
    except Exception as error:                     # noqa: BLE001 - reported, not hidden
        return ProbeResult(NOT_MEASURED, f"read_blotter raised: {error}", proof)

    if view.fills_read == 0:
        return ProbeResult(
            NOT_MEASURED,
            "no fills journalled - the paper engine has not traded, so there is "
            "no blotter to show", proof)

    uncalibrated = ""
    if view.uncalibrated_fills:
        uncalibrated = (f" All {view.uncalibrated_fills} fill(s) carry "
                        f"uncalibrated=true, so the fill prices rest on a "
                        f"declared participation rather than a measured one.")

    if not view.closed_trades_possible:
        only = view.sides_seen[0] if view.sides_seen else "one side"
        return ProbeResult(
            DEGRADED,
            f"{view.fills_read} fill(s) and {len(view.open_positions)} open "
            f"position(s), but ZERO closed round trips - every fill is a {only}, "
            f"so the system has never completed a trade. Not a broken engine: "
            f"the running strategy ({', '.join(view.strategies)}) rests a bid "
            f"and never exits.{uncalibrated}", proof)

    return ProbeResult(
        PARTIAL,
        f"{view.fills_read} fill(s), {len(view.open_positions)} open, "
        f"{len(view.closed_trades)} closed; realised "
        f"{view.realised_pnl(OPTIMISTIC):.6f} optimistic against "
        f"{view.realised_pnl(PESSIMISTIC):.6f} pessimistic, and the gap between "
        f"those two is how much of it is an execution assumption. "
        f"{view.unmarked_positions} open position(s) have no mark, so their "
        f"unrealised P&L is unknown rather than zero.{uncalibrated}", proof)


def probe_funding_carry(facts: SystemFacts) -> ProbeResult:
    """Does the carry setup still refuse what it cannot hedge?

    Run against the live store and the live dollar-quoted universe on every board
    pass. The check that matters is `no_spot_leg`: without the universe filter
    this setup's top proposals were microcap perps at 79-137% annualised with no
    spot market to hedge against, because that is precisely what the funding was
    paying for. A run where nothing is declined for a missing hedge leg means the
    filter has stopped working, and the output would look better, not worse.

    A stand-aside is a healthy result, not a failure - the default action is to
    stand aside and the tile says so rather than reading red on a quiet market.
    """
    import tempfile

    from store.quote_currency import QuoteAssetsNotRecorded, dollar_quoted_symbols
    from strategy.funding_carry import HEDGE_VENUES, select
    from validation.trial_registry import TrialRegistry

    proof = "src/strategy/funding_carry.py"
    now = int(time.time() * 1e9)
    venues = sorted({v for perp, (spot, _p, _s) in HEDGE_VENUES.items()
                     for v in (perp, spot)})
    try:
        universe = {venue: frozenset(
            dollar_quoted_symbols(facts.capture_root, venue, now).dollar)
            for venue in venues}
    except QuoteAssetsNotRecorded as error:
        return ProbeResult(
            DEGRADED,
            f"no dollar-quoted universe to select from: {error}. The setup "
            f"refuses rather than treating 'unknown' as 'everything', which is "
            f"correct and means nothing can be proposed", proof)

    try:
        with tempfile.TemporaryDirectory() as scratch:
            selection = select(
                facts.capture_root / "store", now, capacity=10,
                registry=TrialRegistry(Path(scratch)),
                trial_name="board-probe", dollar_quoted=universe)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the carry setup raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    declined = selection.declined
    if selection.candidates and declined["no_spot_leg"] == 0:
        return ProbeResult(
            FAILING,
            f"{selection.candidates} candidate(s) and NOT ONE was declined for a "
            f"missing hedge leg - the dollar-quoted filter has stopped working, "
            f"and the proposals now include perps with no spot market to hedge "
            f"against. That reads as a better result, which is why it is checked",
            proof)

    note = ", ".join(f"{reason} {count}" for reason, count
                     in sorted(declined.items(), key=lambda kv: -kv[1]) if count)
    return ProbeResult(
        PARTIAL,
        f"{selection.describe()}; declined {note}. Proposals only - nothing "
        f"sizes or orders from them, because the arbiter and the risk gate do "
        f"not exist yet", proof)


def probe_universe_coverage(facts: SystemFacts) -> ProbeResult:
    """The roll-call: every symbol, every segment, including the quiet ones.

    Graded on whether the watch list can still tell a lagging build from a dead
    feed, because without that distinction it is thousands of rows all reading
    STALE - which is what its first live run produced, correctly, while the feeds
    were fine.

    DEGRADED when a segment is entirely stale AND nothing is quiet beyond the
    build: that is the pipeline being behind, which is a real problem with a
    known home (FE-001) and is not the venues' fault. FAILING only if the module
    cannot classify at all.
    """
    from features.universe_coverage import compute_universe_coverage

    proof = "src/features/universe_coverage.py"
    try:
        coverage = compute_universe_coverage(facts.capture_root / "store",
                                             int(time.time() * 1e9))
    except Exception as error:                     # noqa: BLE001 - reported, not hidden
        return ProbeResult(NOT_MEASURED,
                           f"compute_universe_coverage raised: {error}", proof)

    if not coverage.segments:
        return ProbeResult(NOT_MEASURED,
                           "no symbols visible in any segment at this clock", proof)

    missing = [segment for segment in ("perpetual", "dated-future", "spot")
               if segment not in {s.segment for s in coverage.segments}]
    worst_lag = max(s.build_lag_ns for s in coverage.segments)
    lag_minutes = worst_lag / 60_000_000_000
    fresh = sum(s.fresh for s in coverage.segments)
    quiet = sum(s.behind_build for s in coverage.segments)
    detail = coverage.describe()

    if missing:
        return ProbeResult(
            DEGRADED,
            f"{coverage.total_symbols} symbol(s) watched but {', '.join(missing)} "
            f"has no rows at all - a segment missing from the roll-call is the "
            f"one thing this feature exists to make impossible. {detail}", proof)
    if fresh == 0:
        return ProbeResult(
            DEGRADED,
            f"NOTHING is fresh anywhere: the store build is {lag_minutes:.0f} min "
            f"behind the capture feeds, which makes a one-minute bar series stale "
            f"across the whole universe at once. That is FE-001's recorded defect "
            f"- the store supervisor serialising polled builds behind a "
            f"universe-wide bars build - not a venue outage. {detail}", proof)
    return ProbeResult(
        PARTIAL,
        f"{detail}. Nothing consumes it - no module outside "
        f"features.universe_coverage calls compute_universe_coverage, so a "
        f"symbol going quiet cannot yet change what the system does", proof)


def probe_paper_tail_cap(facts: SystemFacts) -> ProbeResult:
    """Can the adaptive paper cap still refuse to widen without limit?

    The user asked for a cap that adapts on paper. The naive version of that
    request is how accounts die - a limit derived from recent realised risk rises
    exactly when risk rises - so the probe feeds it a violent series and a calm
    one on every pass and checks both bounds hold.

    It also re-checks that the LIVE ceiling module still refuses to be
    overwritten. That invariant is §6 and it lives in a different module, which
    is exactly why a regression in it would not show up in any test of this one.
    """
    import tempfile
    from decimal import Decimal

    import numpy as np

    from risk.paper_tail_cap import (
        MAX_WIDENING_MULTIPLE, MIN_TIGHTENING_MULTIPLE, derive_paper_cap,
    )
    from risk.tail_cap import CeilingNotSet, TailCap, read_ceiling, seed_ceiling

    proof = "src/risk/paper_tail_cap.py"
    live_root = facts.capture_root / "risk"
    try:
        live = read_ceiling(live_root)
    except CeilingNotSet as error:
        return ProbeResult(
            NOT_MEASURED,
            f"no live ceiling to bound the paper cap against: {error}", proof)

    rng = np.random.default_rng(0)
    try:
        violent = derive_paper_cap(live, list(rng.normal(0, 0.05, 2000)),
                                   n_samples=200)
        calm = derive_paper_cap(live, list(rng.normal(0, 1e-6, 2000)),
                                n_samples=200)
        thin = derive_paper_cap(live, list(rng.normal(0, 0.01, 10)))
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            seed_ceiling(root, Decimal("0.01"), Decimal("0.03"))
            seed_ceiling(root, Decimal("0.50"), Decimal("0.90"))
            unwidened = read_ceiling(root) == TailCap(Decimal("0.01"),
                                                      Decimal("0.03"))
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the paper cap raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if violent.cap.drawdown_fraction > live.drawdown_fraction * MAX_WIDENING_MULTIPLE:
        broken.append(
            f"a violent series widened the paper cap to "
            f"{violent.cap.drawdown_fraction} against a bound of "
            f"{live.drawdown_fraction * MAX_WIDENING_MULTIPLE} - an adaptive "
            f"limit that rises without a bound stops binding at the moment it "
            f"was for")
    if calm.cap.drawdown_fraction < live.drawdown_fraction * MIN_TIGHTENING_MULTIPLE:
        broken.append("a calm series collapsed the paper cap below its floor")
    if thin.is_derived:
        broken.append(
            "a cap was derived from 10 observations - that is random rather than "
            "adaptive, and the tidy percentile it produces is what makes it "
            "convincing")
    if not unwidened:
        broken.append(
            "the LIVE ceiling was overwritten by a second seed - §6 says no code "
            "in this system raises it, and that invariant is the reason the "
            "adaptive path is a separate module")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"live ceiling {live.describe()}; a violent series is capped at "
        f"{MAX_WIDENING_MULTIPLE}x and a calm one floored at "
        f"{MIN_TIGHTENING_MULTIPLE}x, 10 observations are refused, and the live "
        f"ceiling still refuses to be overwritten. Exercised on this pass, on "
        f"SYNTHETIC returns - no paper run feeds it yet, so no cap has been "
        f"derived from this system's own results", proof)


def probe_stacked_ensemble(facts: SystemFacts) -> ProbeResult:
    """Is the meta-learner still trained on out-of-fold base predictions?

    The defect this module closes structurally is the one that makes stacking
    look brilliant: train the stacker on the base models' IN-SAMPLE predictions
    and it sees a column that is nearly the answer. Because `stack()` takes
    fitters rather than predictions, the probe can check the property directly -
    it records every call the module makes and asserts no base learner was ever
    asked to predict a row it had just trained on.

    It also runs a REDUNDANT pair, because a module that always reports an
    improvement is reporting the fit rather than the finding.
    """
    import tempfile

    import numpy as np

    from models.stacked_ensemble import BaseLearner, _sigmoid, stack
    from validation.trial_registry import TrialRegistry

    proof = "src/models/stacked_ensemble.py"
    rng = np.random.default_rng(0)
    n = 600
    features = rng.normal(size=(n, 2))
    labels = np.where(features[:, 0] + features[:, 1] > 0, 1, -1)
    calls: list[tuple] = []

    def column(index):
        def fit_predict(train_rows, test_rows):
            calls.append((np.asarray(train_rows), np.asarray(test_rows)))
            return _sigmoid(3.0 * features[test_rows, index])
        return fit_predict

    try:
        with tempfile.TemporaryDirectory() as scratch:
            registry = TrialRegistry(Path(scratch))
            complementary = stack(
                labels, [BaseLearner("a", column(0)), BaseLearner("b", column(1))],
                family="microstructure", registry=registry,
                trial_name="probe-complementary", n_groups=4, k_test=2)
            redundant = stack(
                labels,
                [BaseLearner("a", column(0)),
                 BaseLearner("b", lambda _t, rows:
                             _sigmoid(2.9 * features[rows, 0]))],
                family="microstructure", registry=registry,
                trial_name="probe-redundant", n_groups=4, k_test=2)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the stacker raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    leaked = [i for i, (train_rows, test_rows) in enumerate(calls)
              if set(train_rows.tolist()) & set(test_rows.tolist())]
    broken = []
    if leaked:
        broken.append(
            f"{len(leaked)} base-learner call(s) were asked to predict rows they "
            f"had just trained on - the meta-learner is being fed in-sample base "
            f"predictions, which is the defect that makes a stack look brilliant")
    if not complementary.beats_best_base:
        broken.append(
            f"two complementary base models did not beat the better of them "
            f"({complementary.stack_accuracy:.3f} against "
            f"{complementary.base_accuracies[complementary.best_base]:.3f})")
    if redundant.beats_best_base:
        broken.append(
            "two REDUNDANT base models were reported as worth combining - the "
            "stacker is reporting the fit rather than the finding")
    if not complementary.meta_learner.converged:
        broken.append("the meta-learner did not converge")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"no base learner saw a row it was asked to predict across "
        f"{len(calls)} call(s); complementary pair "
        f"{complementary.stack_accuracy:.3f} beats its best base "
        f"{complementary.base_accuracies[complementary.best_base]:.3f}, and a "
        f"redundant pair is correctly not worth combining. Exercised on this "
        f"pass, on SYNTHETIC base models - no two real models exist to stack",
        proof)


def probe_ledger_meta_model(facts: SystemFacts) -> ProbeResult:
    """Does it still refuse to model a ledger too small to model?

    The row's honest state today is a refusal, and the probe reports the actual
    trial count from the live registry against the floor - so the tile shows
    progress toward being able to answer rather than a flat "not built".

    A refusal that stopped firing is the failure that matters: a meta-model over
    thirty trials produces coefficients, and coefficients read as knowledge.
    """
    from models.ledger_meta_model import (
        MIN_TRIALS_TO_FIT, FitRefused, TrialOutcome, fit, summarise,
    )
    from validation.trial_registry import TrialRegistry

    proof = "src/models/ledger_meta_model.py"
    registry_root = facts.capture_root / "trials"
    try:
        live_trials = (TrialRegistry(registry_root).cumulative_count()
                       if registry_root.is_dir() else 0)
        # The refusal, checked on a ledger deliberately below the floor.
        small = [TrialOutcome(i, "carry", "calm", i % 2 == 0, 1_000 + i)
                 for i in range(MIN_TRIALS_TO_FIT - 10)]
        refused = fit(small)
        summary = summarise(small, abandoned=0)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the ledger meta-model raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    if not isinstance(refused, FitRefused):
        return ProbeResult(
            FAILING,
            f"a meta-model was fitted on {len(small)} trial(s), below the floor "
            f"of {MIN_TRIALS_TO_FIT} - coefficients from a ledger that size read "
            f"as knowledge and describe the sample", proof)

    return ProbeResult(
        PARTIAL,
        f"the live trial registry holds {live_trials} trial(s) against a floor "
        f"of {MIN_TRIALS_TO_FIT}, so the model is correctly refused and the "
        f"descriptive half is what is available: {summary.describe()}. The "
        f"summary here is over a scratch ledger - no trial carries a regime "
        f"label yet, because nothing joins a trial's window to "
        f"features.volatility_regime", proof)


def probe_champion_challenger(facts: SystemFacts) -> ProbeResult:
    """Does it still keep pending decisions out of the score?

    The defect this module exists to prevent is silent and one-sided: scoring a
    pending decision as a loss punishes whichever model made more recent
    predictions, which is always the challenger. So the probe plants a pending
    decision and a matured-but-unsettled one alongside a settled population, and
    checks all three land in different counts.

    It also checks the floor still refuses a verdict on too few decisions, which
    is the other way this goes quiet: a swap recommended off eleven observations
    reads exactly like one recommended off a thousand.
    """
    import tempfile

    from models.champion_challenger import (
        MIN_MATURED_DECISIONS, NoVerdict, ShadowLedger,
    )

    proof = "src/models/champion_challenger.py"
    hour = 3_600_000_000_000
    start = 1_700_000_000_000_000_000
    now = start + 1_000 * hour

    try:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = ShadowLedger(Path(scratch) / "shadow")
            settled = MIN_MATURED_DECISIONS + 20
            for i in range(settled):
                made = start + i * 60_000_000_000
                ledger.record_decision(
                    f"p{i:04d}", made_at_ns=made, matures_at_ns=made + hour,
                    champion_prediction=1 if i % 3 else 0,
                    challenger_prediction=1)
                ledger.settle(f"p{i:04d}", 1, settled_at_ns=made + 2 * hour)
            # One matured with nobody writing the outcome, one still pending.
            ledger.record_decision("stale", made_at_ns=start,
                                   matures_at_ns=start + hour,
                                   champion_prediction=0,
                                   challenger_prediction=1)
            ledger.record_decision("pending", made_at_ns=now,
                                   matures_at_ns=now + hour,
                                   champion_prediction=0,
                                   challenger_prediction=1)
            verdict = ledger.compare(now_ns=now)

            thin = ShadowLedger(Path(scratch) / "thin")
            for i in range(MIN_MATURED_DECISIONS - 5):
                made = start + i * 60_000_000_000
                thin.record_decision(f"t{i:04d}", made_at_ns=made,
                                     matures_at_ns=made + hour,
                                     champion_prediction=0,
                                     challenger_prediction=1)
                thin.settle(f"t{i:04d}", 1, settled_at_ns=made + 2 * hour)
            thin_verdict = thin.compare(now_ns=now)
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the shadow ledger raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if getattr(verdict, "matured_decisions", None) != settled:
        broken.append(
            f"{getattr(verdict, 'matured_decisions', '?')} decision(s) scored "
            f"from {settled} settled - a pending or unsettled decision is "
            f"reaching the comparison, which punishes the challenger by "
            f"construction")
    if getattr(verdict, "pending_decisions", None) != 1:
        broken.append("the pending decision is not being counted apart")
    if getattr(verdict, "unsettled_past_maturity", None) != 1:
        broken.append(
            "a matured decision nobody settled is not counted apart - that is an "
            "operational fault and it looks identical to pending unless split")
    if not isinstance(thin_verdict, NoVerdict):
        broken.append(
            f"a verdict was issued on {MIN_MATURED_DECISIONS - 5} decision(s), "
            f"below the floor of {MIN_MATURED_DECISIONS}")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"{verdict.matured_decisions} matured decision(s) scored, 1 pending and "
        f"1 matured-but-unsettled held out; a {MIN_MATURED_DECISIONS - 5}-decision "
        f"ledger is refused a verdict rather than given a weak one. Exercised on "
        f"this pass, on a scratch ledger - no challenger is shadowing anything "
        f"yet, because no model is live to be the champion", proof)


def probe_walk_forward(facts: SystemFacts) -> ProbeResult:
    """Does it still see a decay that a pooled score hides?

    Run on every board pass against a dataset with signal in its first half and
    noise in its second - pooled accuracy well above the base rate, per window
    alive then dead. That is the shape a decayed strategy has, and reporting the
    pooled number is how one survives a review.

    A regression here is silent in the worst way: the windows still come back,
    the pooled accuracy is still right, and only the decay verdict stops firing.
    So the probe checks the verdict, not the plumbing.
    """
    import tempfile

    import numpy as np

    from models.walk_forward import WindowScheme, walk_forward
    from validation.trial_registry import TrialRegistry

    proof = "src/models/walk_forward.py"
    rng = np.random.default_rng(0)
    n = 1200
    column = rng.normal(size=n)
    labels = np.where(np.arange(n) < n // 2, np.sign(column),
                      rng.integers(0, 2, n) * 2 - 1).astype(int)

    def fit_predict(_train_rows, test_rows):
        return (column[test_rows] > 0).astype(float)

    try:
        with tempfile.TemporaryDirectory() as scratch:
            registry = TrialRegistry(Path(scratch))
            result = walk_forward(
                labels, fit_predict, scheme=WindowScheme.EXPANDING,
                test_rows=100, initial_train_rows=200, family="microstructure",
                registry=registry, trial_name="probe-decay")
            counted = registry.cumulative_count()
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the walk-forward raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if result.decay is None or not result.decay.is_significant:
        broken.append(
            "a dataset that is signal then noise was NOT reported as decayed - "
            "the per-window series is the only thing that shows an edge dying, "
            "and a pooled score hides it by averaging the good half with the bad")
    if result.pooled_accuracy <= result.pooled_base_rate:
        broken.append(
            f"pooled accuracy {result.pooled_accuracy:.3f} did not exceed the "
            f"{result.pooled_base_rate:.3f} base rate, so the fixture no longer "
            f"demonstrates the trap it exists for")
    if counted != 1:
        broken.append(
            f"{counted} trial(s) registered for one walk-forward - ten windows "
            f"answer one question, and registering ten would inflate N tenfold")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"{len(result.windows)} windows; pooled {result.pooled_accuracy:.3f} "
        f"against a {result.pooled_base_rate:.3f} base rate reads as an edge, "
        f"and the windows report {result.decay.early_accuracy:.3f} early against "
        f"{result.decay.late_accuracy:.3f} late (p={result.decay.p_value:.3f}). "
        f"Exercised on this pass, on a SYNTHETIC decaying series - no strategy is "
        f"walk-forwarded yet, so this is the detector being checked", proof)


def probe_model_registry(facts: SystemFacts) -> ProbeResult:
    """Does an alias still move without losing where it pointed?

    The registry's one irreplaceable property is that `alias-history.ndjson`
    answers *what was live when this trade happened*. A pointer file alone knows
    only now, and the regression that loses the history leaves every other
    behaviour intact.

    Also checks the artefact verification, which fails in the direction that
    loads: a registry that stopped re-hashing would serve a corrupted or
    substituted file without complaint.
    """
    import tempfile

    from models.model_registry import ArtifactCorrupt, ModelRegistry

    proof = "src/models/model_registry.py"
    try:
        with tempfile.TemporaryDirectory() as scratch:
            registry = ModelRegistry(Path(scratch) / "registry")
            first = registry.register(b"probe-v1", trial_id=1, family="carry",
                                      name="probe")
            second = registry.register(b"probe-v2", trial_id=2, family="carry",
                                       name="probe")
            registry.assign_alias("production", first.version_id,
                                  reason="probe")
            registry.assign_alias("production", second.version_id,
                                  reason="probe")
            history = registry.alias_history("production")
            live_now = registry.resolve("production").version_id

            artefact = (Path(scratch) / "registry" / "models"
                        / f"{first.version_id}.bin")
            artefact.write_bytes(b"tampered")
            try:
                registry.load(first.version_id)
                verified = False
            except ArtifactCorrupt:
                verified = True
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the registry raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if len(history) != 2 or history[0].version_id != first.version_id:
        broken.append(
            "the alias history does not record where production pointed before "
            "it moved - which is the only question anyone asks after a bad fill")
    if live_now != second.version_id:
        broken.append("the alias did not move")
    if not verified:
        broken.append(
            "a tampered artefact loaded without complaint - the hash is no "
            "longer verified on read, and a corrupted file and a substituted one "
            "are indistinguishable at that moment")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        "an alias moved and both assignments are on the append-only history; a "
        "tampered artefact was refused on read. Exercised on this pass, in a "
        "scratch registry - no model is registered for real yet, because nothing "
        "has been trained on the market", proof)


def probe_gradient_boosted_trees(facts: SystemFacts) -> ProbeResult:
    """Can the trained-honestly pipeline still tell signal from noise?

    Trains two LightGBM models on synthetic data on every board pass - one with a
    real relationship in it, one on random labels - and checks the pipeline
    separates them. Exercised rather than asserted because every part of this is
    a behaviour: a purge that stopped purging, a weight vector that stopped being
    applied, and a bootstrap that lost its studentisation all leave a module that
    imports, trains, and returns a number.

    The noise case is the one that matters. If a test block's outcome leaks into
    the fit, a model trained on RANDOM labels scores above the base rate - which
    is the defect that makes most backtests in this literature look good, and it
    is invisible from the output of the signal case alone.

    Synthetic on purpose, and the tile says so: there is no labelled dataset in
    this store yet, because nothing joins the phase-B features to triple-barrier
    labels. What runs here is the control, not a model being trained on the
    market.
    """
    import tempfile

    import numpy as np

    from features.sample_uniqueness import LabelSpan
    from models.gradient_boosted_trees import train_gbt
    from validation.trial_registry import TrialRegistry

    proof = "src/models/gradient_boosted_trees.py"
    rng = np.random.default_rng(0)
    # 480, not 240. The first version of this probe used 240 and the tile went
    # red on its first live pass: with 4 groups and 2 in test each fold trains on
    # about 120 rows, `min_data_in_leaf` is 50, and LightGBM makes no split it
    # will accept - so the model predicts the majority class on every row and the
    # accuracy equals the base rate exactly. The board was right and the fixture
    # was wrong, which is the correct way round for that to have been found.
    n = 480
    features = rng.normal(size=(n, 4))
    signal = np.where(features[:, 0] + 0.3 * rng.normal(size=n) > 0, 1, -1)
    noise = rng.integers(0, 2, size=n) * 2 - 1
    spans = [LabelSpan(i, min(i + 4, n - 1)) for i in range(n)]

    try:
        with tempfile.TemporaryDirectory() as scratch:
            registry = TrialRegistry(Path(scratch))
            common = dict(family="microstructure", registry=registry,
                          boost_rounds=20, n_groups=4, k_test=2)
            learned = train_gbt(features, signal, spans,
                                trial_name="probe-signal", **common)
            unlearned = train_gbt(features, noise, spans,
                                  trial_name="probe-noise", **common)
            counted = registry.cumulative_count()
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the training pipeline raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if not learned.beats_majority_class:
        # Name the copy case explicitly. A red tile reading only "accuracy 0.525
        # against a 0.525 base rate" sends a reader to look for a broken model;
        # "it predicted one class on every row" points at the training set being
        # too small for `min_data_in_leaf`, which is what it was the first time.
        diagnosis = (" - it predicted ONE CLASS on every row, so the training "
                     "folds are too small for min_data_in_leaf to allow a split"
                     if learned.reproduced_majority_class else "")
        broken.append(
            f"a learnable relationship was not found: accuracy "
            f"{learned.accuracy:.3f} against a {learned.base_rate:.3f} base "
            f"rate, p={learned.p_value:.3f}{diagnosis}")
    if unlearned.beats_majority_class:
        broken.append(
            f"RANDOM LABELS beat the majority class at p={unlearned.p_value:.3f} "
            f"- a test block's outcome is leaking into the fit, which is the "
            f"defect that makes a backtest look good")
    if learned.n_out_of_fold != n:
        broken.append(
            f"{learned.n_out_of_fold} scored rows from {n} - CPCV folds are "
            f"being concatenated rather than averaged, so the bootstrap is "
            f"seeing every row {learned.paths_per_row:.0f} times")
    if counted != 2:
        broken.append(f"{counted} trial(s) registered for 2 fits - N is drifting "
                      f"below the true trial count")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        f"signal found ({learned.accuracy:.3f} vs {learned.base_rate:.3f} base "
        f"rate, p={learned.p_value:.3f}) and noise refused "
        f"(p={unlearned.p_value:.3f}); both fits registered, "
        f"{learned.rows_purged} training row(s) purged, effective sample "
        f"{learned.effective_sample_fraction:.2f} of nominal. Exercised on this "
        f"pass, on SYNTHETIC data - no labelled dataset exists in the store yet, "
        f"so this is the control being checked rather than a model trained on "
        f"the market", proof)


def probe_beta_to_btc(facts: SystemFacts) -> ProbeResult:
    from features.beta_to_btc import compute_beta_to_btc
    return _probe_computed_feature(
        facts, module="features.beta_to_btc", entry_point="compute_beta_to_btc",
        compute=compute_beta_to_btc,
        unit="(venue, symbol, horizon) betas to BTC")


def probe_meta_labelling(facts: SystemFacts) -> ProbeResult:
    """Does the grader still follow the SIDE rather than the price?

    Exercised on every pass, not asserted, because the module is one line -
    `meta_label = 1 if barrier_label == side else 0` - and that line is silent
    when it is wrong. A version reading the barrier label as the outcome
    regardless of side produces a perfectly well-formed dataset with a plausible
    base rate, and the only symptom is a live system that loses money in
    proportion to its confidence.

    Two checks, both in the flattering direction if they regress: a SHORT into a
    lower barrier must grade 1, and an unresolved event must be EXCLUDED rather
    than graded 0 - unresolved events cluster at the newest end of every dataset,
    so zeroing them teaches a secondary model to veto recent signals as a class.

    There is no live dataset behind this tile and the detail says so: no primary
    model exists to produce sides. What runs here is the control being checked.
    """
    from features.meta_labelling import make_meta_labels
    from features.triple_barrier import BarrierTouch

    proof = "src/features/meta_labelling.py"

    def touch(index, label):
        return BarrierTouch(
            event_index=index, label=label, reason="probe",
            touched_at_index=None if label is None else index + 1,
            entry_price=100.0, upper_barrier=102.0, lower_barrier=98.0,
            is_ambiguous=False)

    try:
        short_win = make_meta_labels([touch(0, -1)], [-1])
        long_loss = make_meta_labels([touch(0, -1)], [1])
        with_unresolved = make_meta_labels([touch(0, 1), touch(1, None)], [1, 1])
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the meta-labeller raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if short_win.events[0].meta_label != 1:
        broken.append("a SHORT into a lower barrier graded 0 - the grade is "
                      "following the price instead of the side, which inverts "
                      "every short the secondary model will ever see")
    if long_loss.events[0].meta_label != 0:
        broken.append("a LONG into a lower barrier graded 1")
    if with_unresolved.graded != 1 or with_unresolved.unresolved != 1:
        broken.append("an unresolved barrier was graded rather than excluded - "
                      "the newest end of every dataset would be labelled as the "
                      "primary having been wrong")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)

    return ProbeResult(
        PARTIAL,
        "grades the side, not the price: a SHORT into a lower barrier scores 1, "
        "and an unresolved barrier is excluded rather than zeroed. Exercised on "
        "this pass. No dataset is built - no primary model exists to produce "
        "sides, so what runs here is the control being checked, not a secondary "
        "model being trained", proof)


def probe_cross_sectional(facts: SystemFacts) -> ProbeResult:
    from features.cross_sectional import compute_cross_sectional
    return _probe_computed_feature(
        facts, module="features.cross_sectional",
        entry_point="compute_cross_sectional",
        compute=compute_cross_sectional,
        unit="(venue, symbol, horizon) cross-sectional ranks")


def probe_volatility_regime(facts: SystemFacts) -> ProbeResult:
    from features.volatility_regime import compute_volatility_regime
    return _probe_computed_feature(
        facts, module="features.volatility_regime",
        entry_point="compute_volatility_regime",
        compute=compute_volatility_regime,
        unit="(venue, symbol) volatility deciles")


def probe_calendar_effects(facts: SystemFacts) -> ProbeResult:
    """Is there a funding-hour effect on anything, and is it distinguishable?

    Like `probe_har_rv`, this one asks the harder question rather than routing
    through `_probe_computed_feature`, and for the same reason: the feature's
    claim is EMPIRICAL - that flow around settlement is predictably different -
    and returning rows does not test a claim like that. A tile going green on
    row count would be green on a calendar dummy that means nothing.

    A row whose p-value has not cleared its own schedule's floor is reported as
    an effect that is not distinguishable from alignment, which is the honest
    reading and the common one.
    """
    from features.calendar_effects import compute_calendar_effects

    module, entry_point = "features.calendar_effects", "compute_calendar_effects"
    proof = f"{module}.{entry_point} run against the live store"
    try:
        table = compute_calendar_effects(facts.capture_root / "store",
                                         int(time.time() * 1e9))
    except Exception as error:                     # noqa: BLE001 - reported, not hidden
        return ProbeResult(NOT_MEASURED, f"{entry_point} raised: {error}", proof)

    refused = {reason: count for reason, count in table.refused.items() if count}
    refusal_note = ", ".join(f"{reason} {count}" for reason, count in
                             sorted(refused.items(), key=lambda kv: -kv[1]))
    rows = table.rows
    if rows.empty and not refused:
        return ProbeResult(NOT_MEASURED,
                           f"no input reached {module} at this clock", proof)
    if rows.empty:
        biggest = max(refused.items(), key=lambda kv: kv[1])
        return ProbeResult(
            DEGRADED,
            f"measured nothing: every key refused ({refusal_note}). Dominant "
            f"reason {biggest[0]} on {biggest[1]}", proof)

    # A p-value at the floor is the strongest this test can say; one above it is
    # an effect the alignment shift reproduces, which is no effect at all.
    significant = rows[rows["p_value"] <= rows["min_achievable_p_value"]]
    detail = f"{len(rows)} (venue, symbol) contrast(s) measured"
    if refused:
        detail += f"; refused {refusal_note}"

    if significant.empty:
        return ProbeResult(
            PARTIAL,
            f"{detail}. NO funding-hour effect is distinguishable from a shifted "
            f"alignment on any of them - which is a result, and the common one. "
            f"The calendar position is emitted regardless; the claim that it "
            f"matters is not supported here", proof)

    loudest = significant.loc[
        (significant["funding_hour_variance_ratio"] - 1.0).abs().idxmax()]
    consumers = _consumers_of(facts.repo_root, module, entry_point)
    summary = (f"{detail}; {len(significant)} show a funding-hour effect at "
               f"their schedule's resolution floor, loudest "
               f"{loudest['venue']}/{loudest['symbol']} at "
               f"{loudest['funding_hour_variance_ratio']:.2f}x variance "
               f"(p={loudest['p_value']:.3f}, floor "
               f"{loudest['min_achievable_p_value']:.3f})")
    if not consumers:
        return ProbeResult(
            PARTIAL,
            f"{summary}. Nothing consumes it - no module outside {module} calls "
            f"{entry_point}, so a real effect cannot yet change what the system "
            f"does", proof)
    return ProbeResult(OK, f"{summary}; read by {', '.join(consumers)}", proof)


def probe_funding_basis(facts: SystemFacts) -> ProbeResult:
    from features.funding_basis import compute_funding_basis
    return _probe_computed_feature(
        facts, module="features.funding_basis",
        entry_point="compute_funding_basis", compute=compute_funding_basis,
        unit="(venue, symbol) funding/basis feature rows")


def probe_har_rv(facts: SystemFacts) -> ProbeResult:
    """Did the cascade beat the random walk on this pass, on anything?

    Not routed through `_probe_computed_feature` like its five phase-B siblings,
    and the difference is the point. Those five compute a value; there is no such
    thing as a wrong microprice on the board's terms, only a missing one. HAR-RV
    computes a FORECAST, and a forecast has a second way to be worthless that
    producing rows does not rule out: it can be beaten by "tomorrow looks like
    today". A tile reading OK because the module returned rows would be green on
    a model that lost to the cheapest benchmark in the literature.

    So a fitted symbol that did not beat the random walk holds the tile at
    PARTIAL and says so with its p-value. That is not a failure - the crypto HAR
    evidence `finml-feature-engineering.md` cites reports no universal winner, so
    losing on a given symbol is a result. It is simply not a reason for green.

    `reproduced_lag_one_trap` is surfaced separately and is the worse finding of
    the two: it means the coefficients collapsed onto the benchmark, so the
    cascade is not being outperformed, it is being imitated.
    """
    from features.har_rv import compute_har_rv

    module, entry_point = "features.har_rv", "compute_har_rv"
    proof = f"{module}.{entry_point} run against the live store"
    try:
        table = compute_har_rv(facts.capture_root / "store", int(time.time() * 1e9))
    except Exception as error:                     # noqa: BLE001 - reported, not hidden
        return ProbeResult(NOT_MEASURED, f"{entry_point} raised: {error}", proof)

    refused = {reason: count for reason, count in table.refused.items() if count}
    refusal_note = ", ".join(f"{reason} {count}" for reason, count in
                             sorted(refused.items(), key=lambda kv: -kv[1]))
    rows = table.rows
    if rows.empty and not refused:
        return ProbeResult(NOT_MEASURED,
                           f"no input reached {module} at this clock", proof)
    if rows.empty:
        biggest = max(refused.items(), key=lambda kv: kv[1])
        return ProbeResult(
            DEGRADED,
            f"fitted nothing: every candidate refused ({refusal_note}). "
            f"Dominant reason {biggest[0]} on {biggest[1]} - the cascade needs a "
            f"day of unbroken bars behind its first observation, and the tape is "
            f"not offering one", proof)

    beaten = rows[rows["beats_naive"].astype(bool)]
    copied = int(rows["reproduced_lag_one_trap"].astype(bool).sum())
    detail = f"{len(rows)} (venue, symbol) fit(s) walk-forward"
    if refused:
        detail += f"; refused {refusal_note}"

    if beaten.empty:
        note = (f", {copied} of which reproduced the lag-one trap"
                if copied else "")
        best = rows.loc[rows["p_value"].idxmin()]
        return ProbeResult(
            PARTIAL,
            f"{detail}. NONE beat the random walk on variance{note} - closest "
            f"was {best['venue']}/{best['symbol']} at p={best['p_value']:.3f}, "
            f"loss {best['loss_reduction_pct']:+.1f}% against the benchmark. The "
            f"model runs and does not yet earn its place", proof)

    winners = ", ".join(f"{r.venue}/{r.symbol} {r.loss_reduction_pct:+.1f}% "
                        f"p={r.p_value:.3f}"
                        for r in beaten.itertuples(index=False))
    consumers = _consumers_of(facts.repo_root, module, entry_point)
    if not consumers:
        return ProbeResult(
            PARTIAL,
            f"{detail}; beat the random walk on {len(beaten)} of them "
            f"({winners}). Nothing consumes it - no module outside {module} "
            f"calls {entry_point}, so a forecast that works cannot yet change "
            f"what the system does", proof)
    return ProbeResult(
        OK,
        f"{detail}; beat the random walk on {len(beaten)} of them ({winners}); "
        f"read by {', '.join(consumers)}", proof)


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


def probe_sample_uniqueness(facts: SystemFacts) -> ProbeResult:
    """Does the sequential bootstrap still beat uniform sampling?

    Exercised, not asserted. The claim this module makes is EMPIRICAL - that
    drawing proportional to remaining uniqueness produces a more independent
    sample than drawing uniformly - and an empirical claim is the one kind that
    absolutely cannot be reported from the presence of a file. A sequential
    bootstrap that stopped working would still import, still return the right
    number of draws, and still be wrong.

    Two overlapping labels are also checked to report less than full uniqueness,
    which catches the other way this breaks: a concurrency count that silently
    stopped counting would make every sample look perfectly independent, and that
    is the flattering direction - it inflates the effective sample size behind
    every significance test downstream.
    """
    import random as _random

    from features.sample_uniqueness import (
        LabelSpan, average_uniqueness, sequential_bootstrap)

    proof = "src/features/sample_uniqueness.py"
    try:
        pair = average_uniqueness(
            [LabelSpan(0, 3), LabelSpan(0, 3)], n_bars=4)
        spans = [LabelSpan(i * 2, i * 2 + 19) for i in range(40)]
        n_bars, size, runs = 200, 8, 12

        def mean_uniqueness(draw):
            drawn = [spans[i] for i in draw]
            return sum(average_uniqueness(drawn, n_bars=n_bars)) / len(drawn)

        sequential, uniform = [], []
        for seed in range(runs):
            sequential.append(mean_uniqueness(
                sequential_bootstrap(spans, n_bars=n_bars, size=size, seed=seed)))
            rng = _random.Random(seed)
            uniform.append(mean_uniqueness(
                [rng.randrange(len(spans)) for _ in range(size)]))
    except Exception as exc:                      # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the uniqueness machinery raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    if max(pair) >= 1.0:
        return ProbeResult(
            FAILING,
            f"two labels covering identical bars each scored {max(pair)} "
            f"uniqueness instead of 0.5 - concurrency has stopped counting, and "
            f"every overlapping sample now looks perfectly independent",
            proof)

    mean_sequential = sum(sequential) / runs
    mean_uniform = sum(uniform) / runs
    if mean_sequential <= mean_uniform:
        return ProbeResult(
            FAILING,
            f"the sequential bootstrap scored {mean_sequential:.4f} against "
            f"uniform sampling's {mean_uniform:.4f} over {runs} seeds - it is no "
            f"longer buying anything for its non-parallelisable cost",
            proof)
    lift = (mean_sequential / mean_uniform - 1) * 100
    return ProbeResult(
        OK,
        f"sequential bootstrap {mean_sequential:.4f} vs uniform "
        f"{mean_uniform:.4f} over {runs} seeds on heavily overlapping labels, "
        f"{lift:+.1f}% more effective samples. Measured on this pass, and it is "
        f"an improvement rather than a fix for non-IID data",
        proof)


def probe_triple_barrier_labelling(facts: SystemFacts) -> ProbeResult:
    """Does the labeller still refuse the two things it exists to refuse?

    Exercised on every board pass rather than asserted, for the reason
    `probe_naive_baseline_gate` is: a labelling rule is a behaviour, and the only
    honest way to report a behaviour is to run it.

    Two properties, both of which fail in the flattering direction if they
    regress:

      * a bar that closes UP having traded through the stop must label -1. The
        naive close-only version calls it a win, which is the exact defect
        `finml-feature-engineering.md` names triple-barrier labelling as fixing.
      * an event whose vertical barrier runs past the data must be UNRESOLVED,
        never 0. Zeroing it fills the newest stretch of every dataset - the part
        closest to live - with an outcome nobody observed.

    This is also the module's only caller today. The training consumer is MD-010
    and is not built, and that is stated in the axis verdict rather than dressed
    up: what runs here is the control being checked, not a model being trained.
    """
    from features.triple_barrier import Bars, label_triple_barrier

    proof = "src/features/triple_barrier.py"
    try:
        through_stop = label_triple_barrier(
            Bars(high=[100.0, 103.0], low=[100.0, 97.0], close=[100.0, 102.5]),
            event_indices=[0], volatility=[0.01], profit_take_multiple=2.0,
            stop_loss_multiple=2.0, max_holding_bars=5)[0]
        ran_out = label_triple_barrier(
            Bars(high=[100.0] * 4, low=[100.0] * 4, close=[100.0] * 4),
            event_indices=[2], volatility=[0.10], profit_take_multiple=2.0,
            stop_loss_multiple=2.0, max_holding_bars=5)[0]
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        return ProbeResult(
            FAILING,
            f"the labeller raised while being exercised "
            f"({type(exc).__name__}: {exc})", proof)

    broken = []
    if through_stop.label != -1:
        broken.append(
            f"a bar closing up through the stop labelled {through_stop.label} "
            f"instead of -1 - close-only labelling has come back, and it calls "
            f"stopped-out trades winners")
    if ran_out.label is not None:
        broken.append(
            f"an event with no data past its vertical barrier labelled "
            f"{ran_out.label} instead of being left unresolved - the newest "
            f"stretch of every dataset is now filled with an unobserved outcome")
    if broken:
        return ProbeResult(FAILING, "; ".join(broken), proof)
    return ProbeResult(
        OK,
        "path-dependent and honest about the tail: a bar closing up through the "
        "stop labels -1, and an event whose window runs past the data stays "
        "unresolved rather than 0. Exercised on this pass, not inferred",
        proof)


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

def _probe_cache_dir(facts: SystemFacts) -> Path:
    """Beside the archive the measurement was taken from, never a global path."""
    return Path(facts.capture_root) / "boards" / "probe-cache"


def probe_bitemporal_store(facts: SystemFacts) -> ProbeResult:
    """Reads a whole dataset, so it runs on its own cadence (see
    `measured_periodically`). The tile carries the age of the measurement.

    The cache lives UNDER the capture root it measured. Keyed by probe name
    alone it would serve one machine's answer for another's - and every test
    here points a probe at a fabricated store, which is exactly that case.
    """
    return measured_periodically("probe_bitemporal_store", lambda: _probe_bitemporal_store_now(facts),
                                cache_dir=_probe_cache_dir(facts))

def probe_bar_price_validity(facts: SystemFacts) -> ProbeResult:
    """Reads a whole dataset, so it runs on its own cadence (see
    `measured_periodically`). The tile carries the age of the measurement.

    The cache lives UNDER the capture root it measured. Keyed by probe name
    alone it would serve one machine's answer for another's - and every test
    here points a probe at a fabricated store, which is exactly that case.
    """
    return measured_periodically("probe_bar_price_validity", lambda: _probe_bar_price_validity_now(facts),
                                cache_dir=_probe_cache_dir(facts))

def probe_clock_gated_access(facts: SystemFacts) -> ProbeResult:
    """Reads a whole dataset, so it runs on its own cadence (see
    `measured_periodically`). The tile carries the age of the measurement.

    The cache lives UNDER the capture root it measured. Keyed by probe name
    alone it would serve one machine's answer for another's - and every test
    here points a probe at a fabricated store, which is exactly that case.
    """
    return measured_periodically("probe_clock_gated_access", lambda: _probe_clock_gated_access_now(facts),
                                cache_dir=_probe_cache_dir(facts))

def probe_promotion_readiness(facts: SystemFacts) -> ProbeResult:
    """Reads a whole dataset, so it runs on its own cadence (see
    `measured_periodically`). The tile carries the age of the measurement.

    The cache lives UNDER the capture root it measured. Keyed by probe name
    alone it would serve one machine's answer for another's - and every test
    here points a probe at a fabricated store, which is exactly that case.
    """
    return measured_periodically("probe_promotion_readiness", lambda: _probe_promotion_readiness_now(facts),
                                cache_dir=_probe_cache_dir(facts))

def probe_consolidated_price(facts: SystemFacts) -> ProbeResult:
    """Reads a whole dataset, so it runs on its own cadence (see
    `measured_periodically`). The tile carries the age of the measurement.

    The cache lives UNDER the capture root it measured. Keyed by probe name
    alone it would serve one machine's answer for another's - and every test
    here points a probe at a fabricated store, which is exactly that case.
    """
    return measured_periodically("probe_consolidated_price", lambda: _probe_consolidated_price_now(facts),
                                cache_dir=_probe_cache_dir(facts))

PROBES = {
    "sample uniqueness sequential bootstrap": probe_sample_uniqueness,
    "triple barrier labelling": probe_triple_barrier_labelling,
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
    "funding basis spread features": probe_funding_basis,
    "volatility regime decile": probe_volatility_regime,
    "cross sectional ranking across pairs": probe_cross_sectional,
    "correlation beta to btc": probe_beta_to_btc,
    # Phase C. Graded on whether the pipeline still separates signal from noise,
    # not on whether it trains - see `probe_gradient_boosted_trees`.
    "gradient boosted trees": probe_gradient_boosted_trees,
    "rolling walk forward retrain": probe_walk_forward,
    "model registry with aliases": probe_model_registry,
    "champion challenger with delayed label comparison":
        probe_champion_challenger,
    "stacked ensemble": probe_stacked_ensemble,
    # Added 2026-08-16 at the user's instruction. Neither is a §3 model; both
    # answer a direct request and carry a catalogue row of their own.
    "universe watch list across spot perp and dated futures segments":
        probe_universe_coverage,
    "adaptive paper tail cap bounded": probe_paper_tail_cap,
    # §4's first strategy family. Graded on whether it still refuses what it
    # cannot hedge - see `probe_funding_carry`.
    "funding rate carry": probe_funding_carry,
    "paper blotter open positions and closed round trips": probe_paper_blotter,
    "pre trade gate notional leverage position cap": probe_pre_trade_gate,
    # Two catalogue rows, one module: the exit policy and the ratchet inside it.
    "ratchetprofitlock monotone volatility scaled": probe_ratchet_profit_lock,
    "deterministic exit policy as p1 fallback and permanent rollback target":
        probe_deterministic_exit,
    "meta model over the experiment ledger": probe_ledger_meta_model,
    "meta labelling": probe_meta_labelling,
    # Also graded on the empirical claim rather than on row count - a calendar
    # dummy always returns rows. See `probe_calendar_effects`.
    "time of day day of week funding hour effects": probe_calendar_effects,
    # Graded on whether it beats the random walk, not on whether it returns
    # rows - see `probe_har_rv` for why a forecast needs the harder question.
    "har rv": probe_har_rv,
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
