#!/usr/bin/env bash
# Serve the capture console at http://127.0.0.1:$CONSOLE_PORT/
#
# Two loops, one page. A renderer rewrites index.html on a timer and a stock
# http.server serves the directory; the page carries its own meta-refresh, so
# nothing here needs JavaScript, a framework, or a build step.
#
# Bound to 127.0.0.1 deliberately and not configurably. This box has a public
# IP, and the console exposes which instruments are being captured and how much
# - reach it with `ssh -L 8787:localhost:8787 <host>` rather than by opening a
# firewall port.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
CONSOLE_DIR="$CAPTURE_ROOT/console"
CONSOLE_PORT=${CONSOLE_PORT:-8787}
RENDER_EVERY=${RENDER_EVERY:-20}

mkdir -p "$CONSOLE_DIR"

renderer_pid=""
server_pid=""

stop_both() {
    [ -n "$renderer_pid" ] && kill "$renderer_pid" 2>/dev/null
    [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null
    exit 0
}
trap stop_both INT TERM

PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.capture_console \
    --root "$CAPTURE_ROOT" --html "$CONSOLE_DIR/index.html" \
    --watch "$RENDER_EVERY" >>"$CONSOLE_DIR/renderer.log" 2>&1 &
renderer_pid=$!

"$REPO/.venv/bin/python" -m http.server "$CONSOLE_PORT" \
    --bind 127.0.0.1 --directory "$CONSOLE_DIR" >>"$CONSOLE_DIR/server.log" 2>&1 &
server_pid=$!

echo "console  http://127.0.0.1:$CONSOLE_PORT/  (renderer $renderer_pid, server $server_pid)"
wait
