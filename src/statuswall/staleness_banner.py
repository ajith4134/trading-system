"""A board that reports its own age, in the reader's browser.

The failure this exists for, in full: on 2026-08-10 12:53 the boards generator
stopped writing and the box went off. The server kept serving, the tunnel kept
resolving, and every page kept rendering a complete, plausible, entirely green
board — five days out of date. The generation timestamp was printed at the top
the whole time. Nobody noticed, because a timestamp is a thing you have to read
and then do arithmetic on, and nobody does arithmetic on a page that looks fine.

Rule 8 says a display shows measured state. The measurement here is the age of
the page itself, and the only clock that can make it is the reader's — a
server-rendered "generated 5 days ago" is impossible, because the server that
would render it is the one that stopped.

So: the generation instant is embedded as an epoch, and a few lines of script
compute the age against `Date.now()` on every view. Past the threshold the banner
turns red and says so in words. It degrades honestly in both directions:

  * script blocked -> the element keeps its server-rendered text, which names the
    generation time and says the age is unknown. Not green, not silent.
  * clock skew on the reader -> ages can only be wrong by the skew, and a machine
    minutes out shows minutes, not days.

No external requests, no libraries. A staleness check that needed a CDN would go
blank in exactly the network conditions that produce a stale board.
"""
from __future__ import annotations

import html

# Fifteen minutes. The generator rebuilds every 300s, so this is three missed
# rebuilds: past it, something is wrong rather than slow.
STALE_AFTER_SECONDS = 15 * 60


def render_staleness_banner(generated_at_epoch_s: int, label: str,
                            stale_after_seconds: int = STALE_AFTER_SECONDS
                            ) -> str:
    """The banner element and the script that ages it.

    `generated_at_epoch_s` is when the page was built. `label` names the page, so
    a reader with two boards open knows which one went quiet.
    """
    safe_label = html.escape(label)
    return f"""
<div id="staleness" class="staleness staleness-unknown"
     data-generated="{int(generated_at_epoch_s)}"
     data-stale-after="{int(stale_after_seconds)}">
  <!-- Server-rendered fallback. If the script does not run, this is what shows:
       it names the generation instant and admits the age is unknown, which is
       the honest state for a page that cannot measure itself. It is deliberately
       NOT the healthy styling. -->
  AGE UNKNOWN — this page was generated at
  <span class="staleness-when">{safe_label}</span> and could not measure how long
  ago that was
</div>
<style>
  .staleness {{ display:block; margin:0 0 18px; padding:10px 14px;
    border-radius:8px; font-size:12px; letter-spacing:.04em;
    border:1px solid var(--rule); background:var(--panel); color:var(--ink-2); }}
  .staleness-unknown {{ border-color:var(--warn); color:var(--warn); }}
  .staleness-fresh {{ border-color:var(--rule); color:var(--ink-3); }}
  .staleness-stale {{ border-color:var(--fail); color:var(--fail);
    font-weight:600; }}
</style>
<script>
(function () {{
  var node = document.getElementById('staleness');
  if (!node) return;
  var generated = parseInt(node.getAttribute('data-generated'), 10);
  var staleAfter = parseInt(node.getAttribute('data-stale-after'), 10);
  if (!generated) return;

  function describe(seconds) {{
    if (seconds < 90) return Math.max(0, Math.round(seconds)) + 's';
    if (seconds < 5400) return Math.round(seconds / 60) + ' min';
    if (seconds < 172800) return (seconds / 3600).toFixed(1) + 'h';
    return (seconds / 86400).toFixed(1) + ' days';
  }}

  function paint() {{
    var age = (Date.now() / 1000) - generated;
    node.className = 'staleness ' + (age > staleAfter ? 'staleness-stale'
                                                      : 'staleness-fresh');
    if (age > staleAfter) {{
      // Named as a broken generator rather than as an old page, because that is
      // the thing to go and fix, and it is what nobody worked out for five days.
      node.textContent = 'STALE — this board is ' + describe(age) + ' old. '
        + 'The generator has stopped writing it; what you are reading is not '
        + 'the current state of the system.';
    }} else {{
      node.textContent = 'live — regenerated ' + describe(age) + ' ago';
    }}
  }}

  paint();
  // Repainted while the tab sits open, so a board left on a second monitor goes
  // red by itself instead of staying green until someone reloads it.
  setInterval(paint, 15000);
}})();
</script>
"""
