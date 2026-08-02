"""The operator's view of whether capture is alive.

`capture_health.build_report` already computes everything worth knowing and
writes it nowhere. This module is the missing half: it supplies the two inputs
the report cannot produce for itself - free disk and the recent write rate -
renders the result, and answers the one question that matters at a glance,
which is whether frames are still arriving right now.

Two renderings, one status. The terminal one works over a bare SSH session with
no ports and no browser; the HTML one is a plain file with a meta-refresh, so it
needs no server process that could itself die and take the view with it. A
console that requires its own supervisor is a second thing to watch.

Freshness is measured from file modification times rather than from the ledger,
deliberately. The ledger records anomalies, so a stream that is silently doing
nothing writes nothing to it - the absence of events is exactly the condition
that must be visible, and only the archive itself carries that evidence.
"""
from __future__ import annotations

import html
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from capture.capture_health import build_report, measure_daily_bytes, STATUS_PRESENT
from capture.raw_writer import RAW_SUFFIX

# A stream quieter than this is called stale on the console. It is deliberately
# not the alerting threshold: `StalenessTracker` learns each stream's own cadence
# and is the authority on what counts as a real outage. This is a display hint
# generous enough that a slow-but-healthy stream is not painted red.
STALE_AFTER_SECONDS = 120.0

REFRESH_SECONDS = 30

# `measure_daily_bytes` floors its divisor at one day, so an archive minutes old
# reports minutes-worth of bytes as a daily rate - which is honest arithmetic and
# a runway of a hundred thousand days on screen. Below this much history the
# console shows "measuring" instead of a number, because a green figure derived
# from ten minutes is worse than no figure: it is the disk reporting healthy
# while nothing has been measured.
MINIMUM_HISTORY_FOR_RUNWAY_SECONDS = 6 * 3600


def measure_archive_age_seconds(root: Path) -> float:
    """How long the archive has been accumulating, from its own contents.

    Taken from the oldest raw file rather than a recorded start time: the
    archive is the only thing that survives every restart.
    """
    raw_root = Path(root) / "raw"
    if not raw_root.is_dir():
        return 0.0
    oldest = None
    for path in raw_root.rglob(f"*{RAW_SUFFIX}"):
        try:
            created = path.stat().st_mtime
        except OSError:
            continue
        if oldest is None or created < oldest:
            oldest = created
    if oldest is None:
        return 0.0
    return max(0.0, time.time() - oldest)


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def collect_stream_freshness(root: Path, venue: str, date: str) -> list[dict]:
    """Seconds since each of today's raw files was last appended to.

    One row per (stream, symbol). Index and quarantine files are excluded: they
    are not evidence that frames are arriving.
    """
    folder = Path(root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    now = time.time()
    rows = []
    for path in sorted(folder.glob(f"*{RAW_SUFFIX}")):
        stem = path.name[: -len(RAW_SUFFIX)]
        # "{stream}_{symbol}_{hour}" - the hour is fixed-width, the symbol has no
        # underscore, so splitting from the right is unambiguous even for stream
        # names that contain one.
        parts = stem.rsplit("_", 2)
        stream, symbol = (parts[0], parts[1]) if len(parts) == 3 else (stem, "-")
        age = now - path.stat().st_mtime
        rows.append({
            "stream": stream,
            "symbol": symbol,
            "bytes": path.stat().st_size,
            "age_seconds": age,
            "is_stale": age > STALE_AFTER_SECONDS,
        })
    return rows


def find_quarantined_hours(root: Path, venue: str, date: str) -> set[tuple[str, str, str]]:
    """The (stream, symbol, hour) triples the recorder is currently refusing.

    File freshness alone cannot see this, and gets it exactly backwards: running
    `reconcile_pair` rewrites the very files a quarantined stream is not writing
    to, so a repaired hour looks freshly written while the recorder is still
    refusing it. A console that reports a dark stream as live is worse than one
    that reports nothing.

    The ledger carries both halves - `unwritable_stream` when the refusal starts
    and `unwritable_stream_total` when the recorder finally rotates past it - so
    a refusal with no matching total is one still in force.
    """
    from capture.capture_ledger import read_all

    opened, closed = set(), set()
    for event in read_all(Path(root), venue, date):
        detail = event.detail or {}
        key = (event.stream, str(detail.get("symbol", "")), str(detail.get("hour", "")))
        if event.kind == "unwritable_stream":
            opened.add(key)
        elif event.kind == "unwritable_stream_total":
            closed.add(key)
    return opened - closed


def read_recent_restarts(root: Path, venue: str, limit: int = 5) -> list[str]:
    """The tail of the supervisor's own log.

    The recorder cannot record its own downtime, so restarts are the only
    evidence of a gap that leaves no trace in the archive.
    """
    path = Path(root) / "supervisor" / f"{venue}.restarts.ndjson"
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def collect_capture_status(root: Path, venues: list[str]) -> dict:
    """Everything the console shows, for every venue, in one pass."""
    root = Path(root)
    date = _utc_today()
    usage = shutil.disk_usage(root if root.exists() else root.parent)
    daily_bytes = measure_daily_bytes(root)

    venue_rows = []
    for venue in venues:
        report = build_report(root, venue, date, usage.free, daily_bytes)
        streams = collect_stream_freshness(root, venue, date)
        quarantined = find_quarantined_hours(root, venue, date)
        quarantined_streams = {(stream, symbol) for stream, symbol, _ in quarantined}
        for s in streams:
            s["is_quarantined"] = (s["stream"], s["symbol"]) in quarantined_streams
        venue_rows.append({
            "venue": venue,
            "report": report,
            "streams": streams,
            "live_streams": sum(
                1 for s in streams if not s["is_stale"] and not s["is_quarantined"]),
            "quarantined": sorted(quarantined),
            "restarts": read_recent_restarts(root, venue),
        })

    archive_age = measure_archive_age_seconds(root)
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "date": date,
        "root": str(root),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "daily_bytes": daily_bytes,
        "archive_age_seconds": archive_age,
        "runway_is_measurable": archive_age >= MINIMUM_HISTORY_FOR_RUNWAY_SECONDS,
        "venues": venue_rows,
    }


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _format_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def render_status_text(status: dict) -> str:
    """The same status over a bare SSH session, no browser, no ports."""
    measurable = status["runway_is_measurable"]
    rate_text = (f"writing {_format_bytes(status['daily_bytes'])}/day" if measurable
                 else f"write rate measuring "
                      f"({_format_age(status['archive_age_seconds'])} of history)")
    out = [
        f"CAPTURE  {status['generated_at']}   root={status['root']}",
        f"disk     {_format_bytes(status['free_bytes'])} free of "
        f"{_format_bytes(status['total_bytes'])}   {rate_text}",
    ]
    for row in status["venues"]:
        report, streams = row["report"], row["streams"]
        runway = report["runway_days"]
        if not measurable:
            runway_text = "measuring"
        elif runway == float("inf"):
            runway_text = "unmeasured"
        else:
            runway_text = f"{runway:.0f} days"
        out.append("")
        out.append(
            f"{row['venue'].upper():<13} {report['capture_status']}   "
            f"{row['live_streams']}/{len(streams)} streams live   "
            f"runway {runway_text} ({report['runway_status']})")
        out.append(
            f"{'':<13} {_format_bytes(report['raw_data_bytes'])} today   "
            f"gaps corrupting={report['gaps']['corrupting']} "
            f"loss={report['gaps']['observation_loss']}   "
            f"silent={report['silent_streams']}")
        for s in sorted(streams, key=lambda s: (-s["age_seconds"], s["stream"])):
            if s["is_quarantined"]:
                mark = "QUARANTINED"
            elif s["is_stale"]:
                mark = "STALE"
            else:
                mark = "ok"
            out.append(
                f"{'':<15}{s['stream']:<22} {s['symbol']:<10} "
                f"{_format_age(s['age_seconds']):>6} ago  "
                f"{_format_bytes(s['bytes']):>9}  {mark}")
    return "\n".join(out) + "\n"


def render_status_html(status: dict) -> str:
    """A single self-contained page. No server, no external requests.

    Meta-refresh rather than JavaScript polling: the page must keep working when
    it is opened from a file:// URL or copied to another machine.
    """
    e = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Capture</title><style>",
        """
        :root { color-scheme: light dark;
          --bg:#fbfbfa; --fg:#1a1a19; --dim:#6b6b66; --line:#e3e3df;
          --ok:#1f7a4d; --warn:#a86400; --bad:#b02020; --card:#fff; }
        @media (prefers-color-scheme: dark) { :root {
          --bg:#161615; --fg:#eceae4; --dim:#8f8d86; --line:#2c2c2a;
          --ok:#4ba97a; --warn:#d99b2b; --bad:#e06c6c; --card:#1f1f1d; } }
        * { box-sizing:border-box }
        body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
          font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
        .wrap { max-width:70rem; margin:0 auto }
        h1 { font-size:1.05rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase;
          margin:0 0 .2rem; }
        .sub { color:var(--dim); font-size:.82rem; margin-bottom:2rem }
        .tiles { display:grid; gap:.75rem; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
          margin-bottom:2rem }
        .tile { background:var(--card); border:1px solid var(--line); border-radius:.5rem;
          padding:.85rem 1rem }
        .tile .k { color:var(--dim); font-size:.7rem; letter-spacing:.1em; text-transform:uppercase }
        .tile .v { font-size:1.5rem; margin-top:.25rem; font-variant-numeric:tabular-nums }
        h2 { font-size:.95rem; margin:2rem 0 .6rem; display:flex; align-items:baseline; gap:.75rem }
        h2 .badge { font-size:.68rem; letter-spacing:.1em; text-transform:uppercase;
          border:1px solid currentColor; border-radius:.25rem; padding:.1rem .4rem }
        .scroll { overflow-x:auto }
        table { border-collapse:collapse; width:100%; font-size:.85rem; min-width:34rem }
        th { text-align:left; color:var(--dim); font-weight:500; font-size:.7rem;
          letter-spacing:.08em; text-transform:uppercase; padding:.4rem .7rem .4rem 0;
          border-bottom:1px solid var(--line) }
        td { padding:.35rem .7rem .35rem 0; border-bottom:1px solid var(--line);
          font-variant-numeric:tabular-nums }
        td.n { text-align:right; padding-right:1.4rem }
        .ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)} .dim{color:var(--dim)}
        pre { background:var(--card); border:1px solid var(--line); border-radius:.5rem;
          padding:.75rem 1rem; overflow-x:auto; font-size:.75rem; color:var(--dim); margin:.5rem 0 0 }
        """,
        "</style></head><body><div class='wrap'>",
        "<h1>Capture</h1>",
        f"<div class='sub'>{e(status['generated_at'])} &nbsp;·&nbsp; "
        f"refreshes every {REFRESH_SECONDS}s &nbsp;·&nbsp; {e(status['root'])}</div>",
    ]

    total_live = sum(v["live_streams"] for v in status["venues"])
    total_streams = sum(len(v["streams"]) for v in status["venues"])
    measurable = status["runway_is_measurable"]
    worst = min((v["report"]["runway_days"] for v in status["venues"]), default=float("inf"))
    if not measurable:
        runway_value, runway_class = "measuring", "dim"
    elif worst == float("inf"):
        runway_value, runway_class = "—", "dim"
    else:
        runway_value = f"{worst:.0f}d"
        runway_class = "ok" if worst > 30 else ("warn" if worst > 14 else "bad")

    parts.append("<div class='tiles'>")
    for key, value, cls in [
        ("streams live", f"{total_live}/{total_streams}",
         "ok" if total_live == total_streams and total_streams else "bad"),
        ("written today", _format_bytes(sum(
            v["report"]["raw_data_bytes"] for v in status["venues"])), ""),
        ("write rate",
         f"{_format_bytes(status['daily_bytes'])}/day" if measurable else "measuring",
         "" if measurable else "dim"),
        ("disk runway", runway_value, runway_class),
        ("disk free", _format_bytes(status["free_bytes"]), ""),
    ]:
        parts.append(
            f"<div class='tile'><div class='k'>{e(key)}</div>"
            f"<div class='v {cls}'>{e(value)}</div></div>")
    parts.append("</div>")

    for row in status["venues"]:
        report, streams = row["report"], row["streams"]
        present = report["capture_status"] == STATUS_PRESENT
        badge_class = "ok" if present else "bad"
        parts.append(
            f"<h2>{e(row['venue'])}"
            f"<span class='badge {badge_class}'>{e(report['capture_status'])}</span>"
            f"<span class='dim' style='font-size:.75rem;font-weight:400'>"
            f"gaps corrupting {report['gaps']['corrupting']} · "
            f"loss {report['gaps']['observation_loss']} · "
            f"silent {report['silent_streams']} · "
            f"ledger events {report['events_total']}</span></h2>")

        if not streams:
            parts.append("<p class='bad'>No files today. Nothing is being captured.</p>")
        else:
            parts.append("<div class='scroll'><table><thead><tr>"
                         "<th>stream</th><th>symbol</th><th class='n'>last frame</th>"
                         "<th class='n'>size</th><th></th></tr></thead><tbody>")
            for s in sorted(streams, key=lambda s: (-s["age_seconds"], s["stream"])):
                if s["is_quarantined"]:
                    cls, mark = "bad", "quarantined"
                elif s["is_stale"]:
                    cls, mark = "bad", "stale"
                else:
                    cls, mark = "ok", "live"
                parts.append(
                    f"<tr><td>{e(s['stream'])}</td><td>{e(s['symbol'])}</td>"
                    f"<td class='n'>{e(_format_age(s['age_seconds']))} ago</td>"
                    f"<td class='n dim'>{e(_format_bytes(s['bytes']))}</td>"
                    f"<td class='{cls}'>{mark}</td></tr>")
            parts.append("</tbody></table></div>")

        if row["restarts"]:
            parts.append("<pre>" + e("\n".join(row["restarts"])) + "</pre>")

    parts.append("</div></body></html>")
    return "".join(parts)


def write_console_page(root: Path, venues: list[str], destination: Path) -> dict:
    """Render the page to disk atomically and return the status it rendered.

    Atomic because the page is read by a browser on a timer: a half-written file
    is a blank screen at exactly the moment someone is checking whether capture
    is alive.
    """
    status = collect_capture_status(root, venues)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(render_status_html(status), encoding="utf-8")
    temporary.replace(destination)
    return status


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Show whether capture is alive, as text or as a page on disk.")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    parser.add_argument("--venues", default="binance,hyperliquid",
                        help="comma-separated")
    parser.add_argument("--html", default=None,
                        help="write the page here instead of printing")
    parser.add_argument("--watch", type=float, default=0.0,
                        help="seconds between refreshes; 0 renders once and exits")
    args = parser.parse_args(argv)

    root = Path(args.root)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]

    while True:
        if args.html:
            write_console_page(root, venues, Path(args.html))
        else:
            print(render_status_text(collect_capture_status(root, venues)))
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
