"""Renders the status wall as one self-contained HTML page.

Every value on the page arrives from `evidence.assess` and `evidence.measure_system`.
Nothing here invents a number, and there is deliberately no default that renders
a feature as healthy - `STATE_STYLE` has no fallback, so an unhandled state is a
crash rather than a green tile.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

from statuswall.catalogue import Feature
from statuswall.evidence import (
    BUILT, DEGRADED, FAILING, NOT_BUILT, NOT_MEASURED, OK, PARTIAL, SEVERITY_ORDER,
    STATE_LABEL, STOPPED, ProbeResult, SystemFacts,
)

# Semantic status colour is separate from any accent: these hues mean one thing
# each and are not reused for decoration anywhere on the page.
STATE_STYLE = {
    FAILING:   ("#FF4D4D", "#3A0F12", "#FF8A8A", "#C0121B", "#FFE3E3"),
    DEGRADED:  ("#FFB020", "#3A2A08", "#FFD27A", "#9A6300", "#FFF1D6"),
    STOPPED:   ("#C08BFF", "#2A1B3D", "#D9B8FF", "#6B3FA0", "#F0E4FF"),
    # Hazard yellow-white, deliberately not on the green-to-red axis: this state
    # is not a degree of health, it is the absence of a reading. Anything on that
    # axis invites being read as "nearly OK".
    NOT_MEASURED: ("#E8E06A", "#2E2C0F", "#F2ECA0", "#7A7100", "#FBF8DC"),
    PARTIAL:   ("#4FC3E8", "#0E2A34", "#9BDCF2", "#0F6382", "#DFF3FA"),
    OK:        ("#3FD68A", "#0D2C1E", "#8FE9BE", "#12764A", "#DDF6E9"),
    BUILT:     ("#7C96A8", "#17222A", "#A9BFCC", "#3C5A6C", "#E3EBF0"),
    NOT_BUILT: ("#4A5560", "#101820", "#66727D", "#6E7B85", "#E8ECEF"),
}

# An unmeasured feature belongs here for the same reason it has its own colour:
# a tile nobody probed is a question, and the panel is where questions go. Being
# absent from this list is how it would quietly become invisible.
_ATTENTION_STATES = (FAILING, DEGRADED, STOPPED, NOT_MEASURED)


@dataclass(frozen=True)
class WallInput:
    features: list[Feature]
    results: dict[str, ProbeResult]
    facts: SystemFacts


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _state_css() -> str:
    """Status colours as tokens, defined three times on purpose.

    Dark is the default because this board's home is a wall. The light values are
    emitted twice - once inside the media query for a viewer whose OS asks for
    light, and once under `[data-theme="light"]` so the in-page toggle wins over
    the OS in both directions. Components only ever read `--sc`, never a literal.
    """
    dark_rules = []
    light_media = []
    light_forced = []
    for state, (fg, bg, fg_soft, fg_light, bg_light) in STATE_STYLE.items():
        dark_rules.append(
            f'.s-{state} {{ --sc: {fg}; --sc-bg: {bg}; --sc-soft: {fg_soft}; }}')
        light_media.append(
            f'  :root:not([data-theme="dark"]) .s-{state} '
            f'{{ --sc: {fg_light}; --sc-bg: {bg_light}; --sc-soft: {fg_light}; }}')
        light_forced.append(
            f':root[data-theme="light"] .s-{state} '
            f'{{ --sc: {fg_light}; --sc-bg: {bg_light}; --sc-soft: {fg_light}; }}')
    return "\n".join([
        *dark_rules,
        "@media (prefers-color-scheme: light) {",
        *light_media,
        "}",
        *light_forced,
    ])


def _system_banner(facts: SystemFacts) -> str:
    if facts.capture_running:
        state, headline = OK, f"CAPTURE LIVE — {len(facts.capture_pids)} process(es)"
    elif facts.latest_capture_date is None:
        state, headline = FAILING, "NO CAPTURE DATA ON DISK"
    else:
        hours = facts.hours_since_capture or 0.0
        state = FAILING if hours >= 24 else STOPPED
        headline = f"CAPTURE STOPPED — newest data {hours:.0f}h old"
    return (
        f'<div class="banner s-{state}">'
        f'<span class="banner-dot"></span>'
        f'<span class="banner-text">{_esc(headline)}</span>'
        f'<span class="banner-meta">measured {_esc(facts.measured_at)}</span>'
        f"</div>"
    )


def _system_tiles(facts: SystemFacts) -> str:
    venues = ", ".join(facts.venues) or "none"
    restarts = sum(facts.restart_counts.values())
    silence = sum(r.get("silent_streams", 0) for r in facts.reports.values())
    corrupting = sum(r.get("corrupting_non_gap", 0) for r in facts.reports.values())
    gaps = sum(r.get("gaps", {}).get("observation_loss", 0) for r in facts.reports.values())

    tiles = [
        ("Venues capturing", venues, f"{len(facts.venues)} venue(s) with data on disk",
         OK if facts.venues else FAILING),
        ("Newest data", facts.latest_capture_date or "none",
         f"{facts.hours_since_capture:.0f}h old" if facts.hours_since_capture is not None else "no data",
         OK if (facts.hours_since_capture or 999) < 2 else STOPPED),
        ("Disk runway", f"{facts.runway_days:.0f}d",
         f"{facts.free_bytes / 1e9:.0f} GB free · {facts.daily_bytes / 1e6:.0f} MB/day",
         OK if facts.runway_status == "ok" else DEGRADED),
        ("Silence events", f"{silence}", "subscribed streams that delivered nothing",
         DEGRADED if silence else OK),
        ("Corrupting events", f"{corrupting}", "non-gap corruption in the ledger",
         FAILING if corrupting else OK),
        ("Observation gaps", f"{gaps}", "sequence gaps detected and recorded",
         DEGRADED if gaps else OK),
        ("Supervisor restarts", f"{restarts}", "recorded process restarts",
         DEGRADED if restarts else OK),
    ]
    cells = "".join(
        f'<div class="systile s-{state}">'
        f'<div class="systile-k">{_esc(label)}</div>'
        f'<div class="systile-v">{_esc(value)}</div>'
        f'<div class="systile-s">{_esc(sub)}</div>'
        f"</div>"
        for label, value, sub, state in tiles
    )
    return f'<div class="systiles">{cells}</div>'


def _attention_panel(data: WallInput) -> str:
    rows = []
    for state in _ATTENTION_STATES:
        for feature in data.features:
            result = data.results[feature.key]
            if result.state != state:
                continue
            rows.append(
                f'<li class="att s-{state}">'
                f'<span class="att-chip">{_esc(STATE_LABEL[state])}</span>'
                f'<span class="att-name">{_esc(_plain(feature.name))}</span>'
                f'<span class="att-detail">{_esc(result.detail)}</span>'
                f'<span class="att-proof">{_esc(result.proof)}</span>'
                f"</li>"
            )
    if not rows:
        return ('<div class="panel"><h2>Needs attention</h2>'
                '<p class="none">Nothing measured is failing, degraded or stopped.</p></div>')
    return (f'<div class="panel"><h2>Needs attention <span class="pill">{len(rows)}</span></h2>'
            f'<ul class="attlist">{"".join(rows)}</ul></div>')


def _coverage_bar(data: WallInput) -> str:
    counts = {state: 0 for state in SEVERITY_ORDER}
    for result in data.results.values():
        counts[result.state] += 1
    total = sum(counts.values()) or 1
    segments = "".join(
        f'<div class="seg s-{state}" style="flex-grow:{counts[state]}" '
        f'title="{_esc(STATE_LABEL[state])}: {counts[state]}"></div>'
        for state in SEVERITY_ORDER if counts[state]
    )
    legend = "".join(
        f'<span class="leg s-{state}"><i></i>{_esc(STATE_LABEL[state])} '
        f'<b>{counts[state]}</b></span>'
        for state in SEVERITY_ORDER if counts[state]
    )
    measured = total - counts[NOT_BUILT]
    return (
        f'<div class="panel"><h2>Coverage <span class="pill">{measured} of {total} measured</span></h2>'
        f'<div class="bar">{segments}</div><div class="legend">{legend}</div>'
        f'<p class="foot">A feature with no probe behind it renders NOT BUILT. '
        f'That is the honest default, not a gap in the board.</p></div>'
    )


def _plain(name: str) -> str:
    return name.replace("**", "").replace("`", "").replace("*", "")


def _wall(data: WallInput) -> str:
    by_section: dict[tuple[str, str], list[Feature]] = {}
    for feature in data.features:
        by_section.setdefault((feature.section_idx, feature.section_title), []).append(feature)

    blocks = []
    for (idx, title), features in by_section.items():
        tiles = []
        for feature in sorted(features, key=lambda f: SEVERITY_ORDER.index(data.results[f.key].state)):
            result = data.results[feature.key]
            tiles.append(
                f'<div class="tile s-{result.state}" data-state="{result.state}" '
                f'data-phase="{_esc(feature.phase)}">'
                f'<div class="tile-top">'
                f'<span class="tile-phase">{_esc(feature.phase)}</span>'
                f'<span class="tile-state">{_esc(STATE_LABEL[result.state])}</span>'
                f"</div>"
                f'<div class="tile-name">{_esc(_plain(feature.name))}</div>'
                f'<div class="tile-detail">{_esc(result.detail)}</div>'
                f'<div class="tile-proof">{_esc(result.proof)}</div>'
                f"</div>"
            )
        blocks.append(
            f'<section class="block"><h3><span class="bidx">{_esc(idx)}</span>'
            f'{_esc(title)}<span class="bcount">{len(features)}</span></h3>'
            f'<div class="grid">{"".join(tiles)}</div></section>'
        )
    return "".join(blocks)


def render_wall(data: WallInput) -> str:
    """The whole page. Self-contained: no external fonts, scripts or styles."""
    return f"""<title>Status Wall — Autonomous Crypto Trading System</title>
<style>
:root {{
  color-scheme: dark;
  --ground:#080D11; --panel:#0F171D; --panel-2:#131F26; --rule:#1E2C35;
  --ink:#E8F0F4; --ink-2:#93A7B3; --ink-3:#61737E;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --ground:#EDF1F4; --panel:#FFFFFF; --panel-2:#F4F7F9; --rule:#D6DFE5;
  --ink:#0B1418; --ink-2:#465761; --ink-3:#6D808B;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    color-scheme: light;
    --ground:#EDF1F4; --panel:#FFFFFF; --panel-2:#F4F7F9; --rule:#D6DFE5;
    --ink:#0B1418; --ink-2:#465761; --ink-3:#6D808B;
  }}
}}
{_state_css()}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
       font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1500px; margin:0 auto; padding:0 clamp(12px,3vw,32px) 80px; }}

header.top {{ padding:clamp(20px,4vw,40px) 0 0; }}
.kicker {{ font-family:var(--mono); font-size:11px; letter-spacing:.18em;
           text-transform:uppercase; color:var(--ink-3); }}
h1 {{ margin:6px 0 0; font-size:clamp(24px,3.6vw,40px); letter-spacing:-.025em;
      font-weight:660; line-height:1.05; text-wrap:balance; }}
.sub {{ margin:8px 0 0; color:var(--ink-2); max-width:70ch; font-size:14px; }}

.banner {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap;
           margin:22px 0 0; padding:16px 20px; border-radius:4px;
           background:var(--sc-bg); border:1px solid var(--sc); }}
.banner-dot {{ width:12px; height:12px; border-radius:50%; background:var(--sc);
               box-shadow:0 0 0 4px color-mix(in srgb, var(--sc) 25%, transparent); flex:none; }}
.banner-text {{ font-family:var(--mono); font-size:clamp(15px,2.2vw,22px);
                font-weight:600; letter-spacing:.02em; color:var(--sc); }}
.banner-meta {{ font-family:var(--mono); font-size:11px; color:var(--ink-2); margin-left:auto; }}

.systiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
             gap:10px; margin:14px 0 0; }}
.systile {{ background:var(--panel); border:1px solid var(--rule);
            border-left:3px solid var(--sc); border-radius:3px; padding:12px 14px; }}
.systile-k {{ font-family:var(--mono); font-size:10px; letter-spacing:.12em;
              text-transform:uppercase; color:var(--ink-3); }}
.systile-v {{ font-family:var(--mono); font-size:22px; font-weight:600;
              font-variant-numeric:tabular-nums; color:var(--sc); line-height:1.25;
              word-break:break-word; }}
.systile-s {{ font-size:11.5px; color:var(--ink-2); }}

.panel {{ background:var(--panel); border:1px solid var(--rule); border-radius:4px;
          padding:18px 20px 20px; margin:26px 0 0; }}
.panel h2 {{ margin:0 0 14px; font-size:16px; font-weight:640; display:flex;
             align-items:center; gap:10px; }}
.pill {{ font-family:var(--mono); font-size:11px; font-weight:500; color:var(--ink-2);
         border:1px solid var(--rule); border-radius:99px; padding:2px 10px; }}
.none {{ margin:0; color:var(--ink-2); font-size:14px; }}

.attlist {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:1px; }}
.att {{ display:grid; grid-template-columns:104px minmax(0,20rem) minmax(0,1fr);
        gap:4px 14px; padding:11px 14px; background:var(--panel-2);
        border-left:3px solid var(--sc); align-items:baseline; }}
.att-chip {{ font-family:var(--mono); font-size:10.5px; font-weight:600; letter-spacing:.1em;
             color:var(--sc); }}
.att-name {{ font-weight:600; font-size:14px; }}
.att-detail {{ font-size:13px; color:var(--ink-2); }}
.att-proof {{ grid-column:2 / -1; font-family:var(--mono); font-size:10.5px; color:var(--ink-3); }}

.bar {{ display:flex; height:22px; border-radius:3px; overflow:hidden; gap:1px; background:var(--rule); }}
.seg {{ background:var(--sc); min-width:2px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:12px 0 0; }}
.leg {{ font-family:var(--mono); font-size:11px; color:var(--ink-2);
        display:inline-flex; align-items:center; gap:6px; }}
.leg i {{ width:10px; height:10px; border-radius:2px; background:var(--sc); }}
.leg b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
.foot {{ margin:12px 0 0; font-size:12px; color:var(--ink-3); max-width:70ch; }}

.block {{ margin:34px 0 0; }}
.block h3 {{ display:flex; align-items:baseline; gap:12px; margin:0 0 10px;
             font-size:15px; font-weight:640; padding-bottom:8px;
             border-bottom:1px solid var(--rule); }}
.bidx {{ font-family:var(--mono); font-size:11px; color:var(--ink-3); }}
.bcount {{ margin-left:auto; font-family:var(--mono); font-size:11px; color:var(--ink-3); }}

.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:8px; }}
.tile {{ background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--sc);
         border-radius:3px; padding:10px 12px; display:flex; flex-direction:column; gap:3px;
         min-height:74px; }}
.tile[data-state="not_built"] {{ background:transparent; border-color:var(--rule); opacity:.62; }}
.tile-top {{ display:flex; align-items:center; gap:8px; }}
.tile-phase {{ font-family:var(--mono); font-size:10px; color:var(--ink-3);
               border:1px solid var(--rule); border-radius:2px; padding:0 5px; }}
.tile-state {{ font-family:var(--mono); font-size:10px; font-weight:600; letter-spacing:.1em;
               color:var(--sc); margin-left:auto; }}
.tile-name {{ font-size:13px; font-weight:560; line-height:1.35; }}
.tile-detail {{ font-size:11.5px; color:var(--ink-2); line-height:1.4; }}
.tile-proof {{ font-family:var(--mono); font-size:10px; color:var(--ink-3);
               margin-top:auto; padding-top:4px; word-break:break-word; }}
.tile[data-state="not_built"] .tile-proof {{ display:none; }}

footer {{ margin:56px 0 0; padding-top:18px; border-top:1px solid var(--rule);
          font-family:var(--mono); font-size:11px; color:var(--ink-3);
          display:flex; flex-wrap:wrap; gap:6px 22px; }}
@media (max-width:720px) {{
  .att {{ grid-template-columns:1fr; }}
  .att-proof {{ grid-column:1; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
<div class="wrap">
  <header class="top">
    <div class="kicker">Autonomous crypto trading system · status wall</div>
    <h1>What is actually running, right now, on this machine.</h1>
    <p class="sub">Every tile below came from a probe that ran at the timestamp shown.
    Features with no probe behind them read NOT BUILT — the board never guesses, and never
    defaults to healthy. A mostly-dark wall is the correct picture of a system that is
    mostly still designed.</p>
    {_system_banner(data.facts)}
    {_system_tiles(data.facts)}
  </header>
  {_attention_panel(data)}
  {_coverage_bar(data)}
  <main>{_wall(data)}</main>
  <footer>
    <span>Generated by src/statuswall · never hand-edited</span>
    <span>Catalogue source: ~/research/FEATURES.md</span>
    <span>Measured {_esc(data.facts.measured_at)}</span>
  </footer>
</div>
"""
