"""The paper blotter as a page: what is open, what closed, and for how much.

Added 2026-08-16 because the user asked for a board link to see open and closed
paper trades. The status wall carries a one-line tile; a tile cannot show 363
positions, and "see the trades" means a table.

Generated from `paper.blotter`, which reads the engine's own journal. Nothing on
this page is typed and nothing is derived here — Rule 8: the generator is the
deliverable and the page is its output.

## Every number appears twice, and that is the point

Each position and each round trip is rendered under **both** accountings. A
blotter page showing one column is choosing which, and the column that gets
chosen is the flattering one. The optimistic and pessimistic figures sit side by
side so the gap between them is read rather than looked up.

## The empty case is the loud case

A journal holding only one side cannot produce a closed trade. The closed-trades
section says so in a banner rather than rendering an empty table, because an
empty table reads as "a quiet session" and the truth is "this system has never
completed a trade". That distinction is the reason the page exists at all.

## Caps, and saying what was cut

A universe of several hundred positions does not belong in one HTML table, so the
page renders the largest `MAX_ROWS` by notional and **states how many were not
shown**. A silently truncated table is a board that reads as complete, which is
the failure Rule 8 names.
"""
from __future__ import annotations

import html
from decimal import Decimal

from paper.blotter import OPTIMISTIC, PESSIMISTIC, BlotterView, ClosedTrade, OpenPosition
from statuswall.staleness_banner import render_staleness_banner

# Enough to read, few enough to load. The count NOT shown is printed beside the
# table, because a silently truncated board reads as a complete one.
MAX_ROWS = 60


def _decimal(value: Decimal | None, places: int = 6) -> str:
    if value is None:
        return "<span class='unknown'>UNKNOWN</span>"
    return f"{value:,.{places}f}"


def _position_row(position: OpenPosition) -> str:
    unrealised_optimistic = position.unrealised_pnl(OPTIMISTIC)
    unrealised_pessimistic = position.unrealised_pnl(PESSIMISTIC)
    return (
        "<tr>"
        f"<td class='sym'>{html.escape(position.symbol)}</td>"
        f"<td class='dim'>{html.escape(position.venue)}</td>"
        f"<td class='side {position.side.lower()}'>{html.escape(position.side)}</td>"
        f"<td class='n'>{_decimal(position.quantity, 6)}</td>"
        f"<td class='n'>{_decimal(position.average_optimistic, 8)}</td>"
        f"<td class='n'>{_decimal(position.average_pessimistic, 8)}</td>"
        f"<td class='n'>{_decimal(unrealised_optimistic)}</td>"
        f"<td class='n'>{_decimal(unrealised_pessimistic)}</td>"
        f"<td class='n dim'>{position.fills}</td>"
        "</tr>")


def _trade_row(trade: ClosedTrade) -> str:
    optimistic = trade.pnl(OPTIMISTIC)
    pessimistic = trade.pnl(PESSIMISTIC)
    held_hours = trade.held_ns / 3_600_000_000_000
    return (
        "<tr>"
        f"<td class='sym'>{html.escape(trade.symbol)}</td>"
        f"<td class='dim'>{html.escape(trade.venue)}</td>"
        f"<td class='side {trade.side.lower()}'>{html.escape(trade.side)}</td>"
        f"<td class='n'>{_decimal(trade.quantity, 6)}</td>"
        f"<td class='n'>{_decimal(trade.entry_optimistic, 8)}</td>"
        f"<td class='n'>{_decimal(trade.exit_optimistic, 8)}</td>"
        f"<td class='n {"win" if optimistic > 0 else "loss"}'>"
        f"{_decimal(optimistic)}</td>"
        f"<td class='n {"win" if pessimistic > 0 else "loss"}'>"
        f"{_decimal(pessimistic)}</td>"
        f"<td class='n dim'>{held_hours:.1f}h</td>"
        "</tr>")


def _closed_section(view: BlotterView) -> str:
    if not view.closed_trades_possible:
        only = html.escape(view.sides_seen[0]) if view.sides_seen else "one side"
        strategies = html.escape(", ".join(view.strategies) or "none")
        return (
            "<div class='banner fail'>"
            "<strong>ZERO closed round trips, and zero are possible.</strong> "
            f"Every one of the {view.fills_read:,} journalled fills is a "
            f"<code>{only}</code>, so nothing can close. This is not a broken "
            f"engine — the running strategy (<code>{strategies}</code>) rests a "
            f"bid and never exits. An empty table here would read as a quiet "
            f"session; the truth is that this system has never completed a trade."
            "</div>")

    shown = sorted(view.closed_trades, key=lambda t: t.closed_at_ns,
                   reverse=True)[:MAX_ROWS]
    hidden = len(view.closed_trades) - len(shown)
    note = (f"<p class='note'>Showing the {len(shown)} most recent of "
            f"{len(view.closed_trades):,}; {hidden:,} not shown.</p>"
            if hidden else "")
    return f"""
    <div class='totals'>
      <div><span class='k'>Realised, optimistic</span>
           <span class='v'>{_decimal(view.realised_pnl(OPTIMISTIC))}</span></div>
      <div><span class='k'>Realised, pessimistic</span>
           <span class='v'>{_decimal(view.realised_pnl(PESSIMISTIC))}</span></div>
    </div>
    {note}
    <table>
      <thead><tr>
        <th>Symbol</th><th>Venue</th><th>Side</th><th>Qty</th>
        <th>Entry (opt)</th><th>Exit (opt)</th>
        <th>P&amp;L optimistic</th><th>P&amp;L pessimistic</th><th>Held</th>
      </tr></thead>
      <tbody>{''.join(_trade_row(t) for t in shown)}</tbody>
    </table>"""


def _open_section(view: BlotterView) -> str:
    if not view.open_positions:
        return "<div class='banner'>No open positions.</div>"
    shown = sorted(view.open_positions,
                   key=lambda p: p.average_optimistic * p.quantity,
                   reverse=True)[:MAX_ROWS]
    hidden = len(view.open_positions) - len(shown)
    note = (f"<p class='note'>Showing the {len(shown)} largest by notional of "
            f"{len(view.open_positions):,}; {hidden:,} not shown.</p>"
            if hidden else "")
    unmarked = ""
    if view.unmarked_positions:
        unmarked = (f"<div class='banner warn'>{view.unmarked_positions:,} of "
                    f"{len(view.open_positions):,} position(s) have no mark "
                    f"price, so their unrealised P&amp;L is <strong>unknown, not "
                    f"zero</strong>. Marking against a stale price would report a "
                    f"number about a market that has moved.</div>")
    return f"""{unmarked}{note}
    <table>
      <thead><tr>
        <th>Symbol</th><th>Venue</th><th>Side</th><th>Qty</th>
        <th>Avg entry (opt)</th><th>Avg entry (pess)</th>
        <th>Unrealised optimistic</th><th>Unrealised pessimistic</th><th>Fills</th>
      </tr></thead>
      <tbody>{''.join(_position_row(p) for p in shown)}</tbody>
    </table>"""


def render_blotter_page(view: BlotterView, generated_at: str,
                        generated_at_epoch_s: int) -> str:
    """The whole page. Every figure arrived on `view`, measured from the journal."""
    claim = ""
    if not view.makes_edge_claim and view.fills_read:
        claim = ("<div class='banner warn'><strong>The running strategy makes no "
                 "edge claim.</strong> <code>makes_edge_claim</code> is false on "
                 "every fill, so nothing on this page is performance — it is "
                 "proof that the plumbing carries an order end to end.</div>")
    uncalibrated = ""
    if view.uncalibrated_fills:
        uncalibrated = (f"<div class='banner warn'>All "
                        f"{view.uncalibrated_fills:,} fill(s) carry "
                        f"<code>uncalibrated=true</code>: the fill prices rest on "
                        f"a <em>declared</em> participation rate, not a measured "
                        f"one. No tier-2 promotion may read them.</div>")
    if view.fills_read == 0:
        body = ("<div class='banner fail'>No fills journalled. The paper engine "
                "has not traded, so there is no blotter to show.</div>")
    else:
        body = f"""
    {claim}{uncalibrated}
    <h2>Open positions <span class='count'>{len(view.open_positions):,}</span></h2>
    {_open_section(view)}
    <h2>Closed round trips <span class='count'>{len(view.closed_trades):,}</span></h2>
    {_closed_section(view)}"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Paper blotter — open and closed trades</title>
<style>
  :root {{ color-scheme: dark;
    --ground:#080D11; --panel:#0F171D; --rule:#1E2C35;
    --ink:#E8F0F4; --ink-2:#93A7B3; --ink-3:#61737E;
    --win:#3FD68A; --loss:#F2707A; --warn:#E8B44F; --fail:#F2707A;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  @media (prefers-color-scheme: light) {{
    :root {{ --ground:#EDF1F4; --panel:#FFF; --rule:#D6DFE5;
      --ink:#0B1418; --ink-2:#465761; --ink-3:#6D808B;
      --win:#12764A; --loss:#B4232F; --warn:#8A6100; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--ink);
    font-family:var(--sans); line-height:1.5; padding:clamp(16px,4vw,40px); }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  .kicker {{ font-family:var(--mono); font-size:11px; letter-spacing:.18em;
    text-transform:uppercase; color:var(--ink-3); }}
  h1 {{ margin:6px 0 4px; font-size:clamp(22px,4vw,30px); letter-spacing:-.02em; }}
  h2 {{ margin:34px 0 10px; font-size:17px; letter-spacing:-.01em;
    display:flex; align-items:baseline; gap:10px; }}
  .count {{ font-family:var(--mono); font-size:13px; color:var(--ink-3); }}
  .sub {{ margin:0 0 8px; color:var(--ink-2); font-size:14px; }}
  .banner {{ background:var(--panel); border:1px solid var(--rule);
    border-left:3px solid var(--ink-3); border-radius:4px; padding:12px 15px;
    margin:12px 0; font-size:13.5px; color:var(--ink-2); }}
  .banner.warn {{ border-left-color:var(--warn); }}
  .banner.fail {{ border-left-color:var(--fail); }}
  .banner strong {{ color:var(--ink); }}
  .note {{ color:var(--ink-3); font-size:12.5px; margin:6px 0; }}
  .totals {{ display:flex; gap:26px; flex-wrap:wrap; margin:12px 0; }}
  .totals .k {{ display:block; font-family:var(--mono); font-size:10.5px;
    letter-spacing:.12em; text-transform:uppercase; color:var(--ink-3); }}
  .totals .v {{ font-family:var(--mono); font-size:18px; }}
  .scroll {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px;
    background:var(--panel); border:1px solid var(--rule); border-radius:4px; }}
  th, td {{ text-align:left; padding:7px 11px; border-bottom:1px solid var(--rule);
    white-space:nowrap; }}
  th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--ink-3); font-weight:500; }}
  tr:last-child td {{ border-bottom:none; }}
  td.n {{ font-family:var(--mono); text-align:right; }}
  td.sym {{ font-weight:600; }}
  td.dim, .dim {{ color:var(--ink-3); }}
  td.side {{ font-family:var(--mono); font-size:11.5px; }}
  td.side.buy {{ color:var(--win); }}
  td.side.sell {{ color:var(--loss); }}
  td.win {{ color:var(--win); }}
  td.loss {{ color:var(--loss); }}
  .unknown {{ font-family:var(--mono); font-size:11px; color:var(--warn); }}
  code {{ font-family:var(--mono); font-size:.92em; }}
  footer {{ margin-top:34px; color:var(--ink-3); font-size:12px; }}
</style></head>
<body><div class="wrap">
  <div class="kicker">Paper trading</div>
  <h1>Blotter — open and closed trades</h1>
  <p class="sub">{html.escape(view.describe())}</p>
  {render_staleness_banner(generated_at_epoch_s, generated_at)}
  <div class="scroll">{body}</div>
  <footer>
    Generated {html.escape(generated_at)} from the forward engine's own fill
    journal by <code>statuswall.blotter_page</code>. Every figure is read from
    <code>paper/forward/fills-*.ndjson</code>; nothing here is typed and nothing
    is derived. Round trips are paired FIFO per (venue, symbol) — a convention,
    stated because it changes the P&amp;L of every partially-closed position.
  </footer>
</div></body></html>
"""
