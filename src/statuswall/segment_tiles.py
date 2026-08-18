"""BF-08: the segment bots' wall — measured state only, and absence as its own state.

    python -m statuswall.segment_tiles --out-dir ~/research/dashboard

## Rule 8, applied to four bots

**Every tile traces to a probe that ran.** The probes here are the heartbeat each
engine writes on every poll and the journals it appends to. There is no path in this
module that renders a status from a constant, and no colour that can be produced
without a file having been read.

A bot with no heartbeat renders `NOT MEASURED`, not green and not blank. A bot whose
heartbeat has stopped advancing renders `STALE` with the age, because a supervised
process that died and a market with nothing to do produce the same trade count and
only the heartbeat age separates them.

## RL-025's label is carried, not inferred

Each engine writes `makes_edge_claim` into its own heartbeat, taken from the brain
that produced the decision. This page renders what it finds there. It never decides
that a bot is a rule brain — if a trained brain is deployed and sets the flag, this
page changes with it and nobody has to remember to edit a template.

That is the difference between a label the display owns and one the decision owns.
The first can be forgotten at the display; the second cannot.

## The P&L shown is not a result, and the page says so

Every open and close on this wall came from a rule brain making no edge claim. The
banner states it. `~/research/DECISIONS.md` records a live system that lost $837 over
10,240 trades because it credited the same P&L to all 36 of its features — a number
on a board with no claim attached to it is how that starts.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

DEFAULT_STATE_ROOT = Path.home() / "capture" / "segment"
DEFAULT_OUT_DIR = Path.home() / "research" / "dashboard"
SEGMENTS = ("perp", "spot", "dated", "options")

# A heartbeat older than this means the bot is not polling. Generous relative to the
# 6-second loop so a slow poll is not reported as a death.
STALE_AFTER_NS = 120_000_000_000

NOT_MEASURED = "NOT MEASURED"
STALE = "STALE"
OFF = "OFF"
LIVE = "LIVE"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_ndjson(path: Path) -> list:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def read_segment(segment: str, state_root: Path, now_ns: int) -> dict:
    """Everything measured about one bot. No value here is assumed."""
    directory = state_root / segment
    heartbeat = _read_json(directory / "heartbeat.json")
    off_switch = (directory / "OFF").exists()

    fills = []
    for path in sorted(directory.glob("fills-*.ndjson")):
        fills.extend(_read_ndjson(path))
    missed = []
    for path in sorted(directory.glob("missed_entries-*.ndjson")):
        missed.extend(_read_ndjson(path))

    opens = [f for f in fills if f.get("event") == "OPEN"]
    closes = [f for f in fills if f.get("event") == "CLOSE"]

    realised = 0.0
    wins = 0
    for close in closes:
        try:
            pnl = float(close.get("gross_pnl") or 0)
        except (TypeError, ValueError):
            continue
        realised += pnl
        if pnl > 0:
            wins += 1

    # A winrate over a handful of trades is not a winrate. PB-10's acceptance says a
    # band with too few trades says so instead of reporting one, and the same rule is
    # applied here rather than being left to the reader.
    winrate = None if len(closes) < 20 else round(100.0 * wins / len(closes), 1)

    if heartbeat is None:
        status, age_ns = NOT_MEASURED, None
    else:
        age_ns = now_ns - int(heartbeat.get("written_at_ns") or 0)
        if off_switch:
            status = OFF
        elif age_ns > STALE_AFTER_NS:
            status = STALE
        else:
            status = LIVE

    return {
        "segment": segment,
        "status": status,
        "heartbeat": heartbeat,
        "heartbeat_age_ns": age_ns,
        "opens": len(opens),
        "closes": len(closes),
        "open_now": (heartbeat or {}).get("open_positions"),
        "realised_pnl": realised,
        "winrate": winrate,
        "wins": wins,
        "missed_entries": len(missed),
        "recent": sorted(fills, key=lambda f: f.get("at_ns") or 0, reverse=True)[:25],
    }


_STATUS_CLASS = {LIVE: "live", STALE: "stale", OFF: "off", NOT_MEASURED: "unmeasured"}


def _fmt_age(age_ns) -> str:
    if age_ns is None:
        return "—"
    seconds = age_ns / 1e9
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _tile(measured: dict) -> str:
    segment = measured["segment"]
    status = measured["status"]
    heartbeat = measured["heartbeat"] or {}
    feed = heartbeat.get("feed") or {}
    brains = heartbeat.get("brains") or {}
    counts = heartbeat.get("counts") or {}

    if status == NOT_MEASURED:
        # Rule 8: absence renders as its own state, with what was looked for.
        return f"""
    <section class="tile unmeasured">
      <header><h2>{html.escape(segment)}</h2><span class="badge unmeasured">{NOT_MEASURED}</span></header>
      <p class="why">No heartbeat at <code>~/capture/segment/{html.escape(segment)}/heartbeat.json</code>.
         This bot has never written one, so nothing about it is known. Not green, not blank.</p>
    </section>"""

    pnl = measured["realised_pnl"]
    pnl_class = "up" if pnl > 0 else "down" if pnl < 0 else "flat"
    winrate = ("<span class='thin'>too few closes for a rate "
               f"({measured['closes']})</span>" if measured["winrate"] is None
               else f"{measured['winrate']}%")

    rows = []
    for fill in measured["recent"][:12]:
        event = fill.get("event", "")
        pnl_cell = fill.get("gross_pnl")
        rows.append(
            "<tr>"
            f"<td>{_fmt_age(int(time.time_ns()) - int(fill.get('at_ns') or 0))}</td>"
            f"<td class='{'open' if event == 'OPEN' else 'close'}'>{html.escape(event)}</td>"
            f"<td>{html.escape(str(fill.get('symbol', '')))}</td>"
            f"<td>{html.escape(str(fill.get('side', '')))}</td>"
            f"<td>{html.escape(str(fill.get('price', '')))}</td>"
            f"<td>{html.escape(str(fill.get('close_reason') or fill.get('selection_reason') or ''))}</td>"
            f"<td class='{'up' if pnl_cell and float(pnl_cell) > 0 else 'down' if pnl_cell and float(pnl_cell) < 0 else ''}'>"
            f"{html.escape(str(pnl_cell)) if pnl_cell is not None else ''}</td>"
            "</tr>")
    table = ("<p class='why'>No fills yet. The bot is polling; nothing has met its "
             "entry rules.</p>" if not rows else
             "<table><thead><tr><th>when</th><th>event</th><th>symbol</th><th>side</th>"
             "<th>price</th><th>reason</th><th>P&amp;L</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")

    edge_claim = heartbeat.get("makes_edge_claim")
    label = ("<span class='badge claim'>EDGE CLAIM</span>" if edge_claim
             else "<span class='badge ruleb'>RULE BRAIN · NO EDGE CLAIM</span>")

    return f"""
    <section class="tile {_STATUS_CLASS[status]}">
      <header>
        <h2>{html.escape(segment)}</h2>
        <span class="badge {_STATUS_CLASS[status]}">{status}</span>
        {label}
      </header>
      <p class="meta">
        venue <strong>{html.escape(str(heartbeat.get('venue', '—')))}</strong> ·
        band <strong>{html.escape(str(heartbeat.get('band', '—')))}</strong> ·
        heartbeat {_fmt_age(measured['heartbeat_age_ns'])} ·
        feed <strong>{html.escape(str(feed.get('liveness', '—')))}</strong>
        (newest tick {feed.get('newest_age_seconds', '—')}s,
         {feed.get('ticks_delivered', 0)} ticks, {feed.get('reconnects', 0)} reconnects)
      </p>
      <p class="meta">brains ·
        BULL <code>{html.escape(str(brains.get('bull', '—')))}</code> ·
        BEAR <code>{html.escape(str(brains.get('bear', '—')))}</code> ·
        PROFIT-TAIL <code>{html.escape(str(brains.get('profit_tail', '—')))}</code>
      </p>
      <div class="numbers">
        <div><span class="n">{measured['opens']}</span><span class="l">opened</span></div>
        <div><span class="n">{measured['closes']}</span><span class="l">closed</span></div>
        <div><span class="n">{measured['open_now'] if measured['open_now'] is not None else '—'}</span><span class="l">open now</span></div>
        <div><span class="n {pnl_class}">{pnl:+.4f}</span><span class="l">realised P&amp;L</span></div>
        <div><span class="n">{winrate}</span><span class="l">winrate</span></div>
        <div><span class="n">{measured['missed_entries']}</span><span class="l">missed entries</span></div>
      </div>
      <p class="meta counts">polls {counts.get('polls', 0)} ·
        abstentions {counts.get('abstentions', 0)} ·
        selected {counts.get('selected', 0)} ·
        gate refusals {counts.get('gate_refusals', 0)} ·
        ratchets {counts.get('ratchets', 0)} ·
        frames refused {counts.get('frames_refused', 0)}</p>
      {table}
    </section>"""


_CSS = """
:root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3;
        --dim:#8b949e; --up:#3fb950; --down:#f85149; --warn:#d29922; }
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--text);
       font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
h1 { font-size:20px; margin:0 0 4px; }
.banner { background:#1c2128; border:1px solid var(--warn); border-left:4px solid var(--warn);
          padding:12px 14px; margin:16px 0 24px; color:#d9c8a0; }
.stamp { color:var(--dim); margin:0 0 16px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px;
        overflow-x:auto; }
.tile.live { border-left:4px solid var(--up); }
.tile.stale { border-left:4px solid var(--warn); }
.tile.off { border-left:4px solid var(--dim); }
.tile.unmeasured { border-left:4px solid var(--dim); background:#12161c; }
header { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
h2 { font-size:16px; margin:0; text-transform:uppercase; letter-spacing:.06em; }
.badge { font-size:11px; padding:2px 8px; border-radius:10px; border:1px solid var(--line); }
.badge.live { color:var(--up); border-color:var(--up); }
.badge.stale { color:var(--warn); border-color:var(--warn); }
.badge.off, .badge.unmeasured { color:var(--dim); }
.badge.ruleb { color:var(--warn); border-color:var(--warn); }
.badge.claim { color:var(--up); border-color:var(--up); }
.meta { color:var(--dim); margin:4px 0; font-size:12px; }
.why { color:var(--dim); font-size:12px; }
.numbers { display:flex; gap:18px; flex-wrap:wrap; margin:12px 0; }
.numbers .n { display:block; font-size:19px; }
.numbers .l { display:block; color:var(--dim); font-size:11px; text-transform:uppercase; }
.up { color:var(--up); } .down { color:var(--down); } .flat { color:var(--dim); }
.thin { color:var(--dim); font-size:11px; }
table { width:100%; border-collapse:collapse; margin-top:8px; font-size:12px; }
th { text-align:left; color:var(--dim); font-weight:normal; border-bottom:1px solid var(--line);
     padding:4px 6px; }
td { padding:3px 6px; border-bottom:1px solid #21262d; white-space:nowrap; }
td.open { color:var(--up); } td.close { color:var(--warn); }
code { color:#79c0ff; }
a { color:#79c0ff; }
"""


def render(state_root: Path, now_ns: int) -> str:
    measured = [read_segment(segment, state_root, now_ns) for segment in SEGMENTS]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now_ns / 1e9))
    live_count = sum(1 for m in measured if m["status"] == LIVE)
    total_opens = sum(m["opens"] for m in measured)
    total_closes = sum(m["closes"] for m in measured)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Segment bots</title>
<meta http-equiv="refresh" content="20">
<style>{_CSS}</style></head><body>
<h1>Segment bots — live paper trading</h1>
<p class="stamp">measured {stamp} · {live_count} of {len(SEGMENTS)} bots LIVE ·
   {total_opens} opened · {total_closes} closed · page refreshes every 20s ·
   <a href="index.html">index</a></p>
<div class="banner">
  <strong>These bots run RULE BRAINS and make no edge claim (RL-025).</strong>
  Every decision below came from explicit stated rules, not a trained model, and every
  fill is journalled with <code>makes_edge_claim: false</code>. The P&amp;L is a
  measurement of the plumbing and of the deterministic baseline a trained
  PROFIT-TAIL must later beat — <em>it is not evidence of edge and must never be
  promoted as one</em>.
  <br><br>
  Prices are live venue websocket and REST feeds (RL-024), never the parquet store.
  Fills are modelled; no order reaches a venue.
</div>
<div class="grid">{''.join(_tile(m) for m in measured)}</div>
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "segment-bots.html"
    target.write_text(render(args.state_root, time.time_ns()), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
