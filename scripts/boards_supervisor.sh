#!/usr/bin/env bash
# Keep the boards reachable: an authenticated local server, and a tunnel to it.
#
# Two children, supervised independently, because they fail independently: the
# server dies on a code error, the tunnel dies when Cloudflare drops the edge
# connection. Restarting both when only one died would change the public URL for
# no reason.
#
# THE URL IS NOT STABLE, and no amount of supervision makes it so. A
# `trycloudflare.com` quick tunnel is issued a random hostname per tunnel
# process; a named tunnel with a fixed hostname requires a Cloudflare account and
# a domain. So the current URL is written to $URL_FILE on every tunnel start,
# which is the only place that is guaranteed to be right.
#
# Backoff is capped exponential with jitter, matching capture_supervisor.sh. The
# jitter matters less here than there - there is only one of each child - but a
# tight restart loop against Cloudflare's edge earns a rate limit.
#
# Usage: boards_supervisor.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BOARDS_DIR=${BOARDS_DIR:-$HOME/research/dashboard}
STATE_DIR=${BOARDS_STATE_DIR:-$HOME/capture/boards}
CREDENTIALS=${BOARDS_CREDENTIALS_FILE:-$HOME/.config/boards/credentials}
PORT=${BOARDS_PORT:-8787}
PYTHON=${BOARDS_PYTHON:-$REPO/.venv/bin/python}
CLOUDFLARED=${CLOUDFLARED:-$HOME/.local/bin/cloudflared}

SERVER_LOG="$STATE_DIR/server.log"
TUNNEL_LOG="$STATE_DIR/tunnel.log"
URL_FILE="$STATE_DIR/current-url.txt"
RESTART_LOG="$STATE_DIR/restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120

mkdir -p "$STATE_DIR"

server_pid=""
tunnel_pid=""

record_restart() {
  printf '{"ts":"%s","child":"%s","exit_code":%s,"ran_seconds":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$RESTART_LOG"
}

# Forward the stop signal rather than dying and orphaning the children: an
# orphaned cloudflared keeps a tunnel alive that nothing is supervising, and the
# next start would publish a second URL to the same boards.
stop_children() {
  for pid in "$server_pid" "$tunnel_pid"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  exit 0
}
trap stop_children TERM INT

ensure_credentials() {
  if [ -s "$CREDENTIALS" ]; then
    return
  fi
  echo "generating credentials at $CREDENTIALS"
  "$PYTHON" - "$CREDENTIALS" <<'PY'
import sys
from pathlib import Path
from statuswall.board_server import generate_credentials
user, password = generate_credentials(Path(sys.argv[1]))
print(f"boards credentials created — user: {user}  password: {password}")
PY
}

next_delay() {
  local delay=$1
  local doubled=$(( delay * 2 ))
  [ "$doubled" -gt "$MAX_DELAY" ] && doubled=$MAX_DELAY
  # Jitter in [0, delay) so repeated failures do not retry on a fixed cadence.
  echo $(( doubled + (RANDOM % (delay > 0 ? delay : 1)) ))
}

publish_tunnel_url() {
  # cloudflared prints the assigned hostname once, at startup, inside a banner.
  # Poll the log rather than parse stdout live: the banner can take several
  # seconds and arrives after the process is already healthy.
  local waited=0
  while [ "$waited" -lt 40 ]; do
    local url
    url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -1)
    if [ -n "$url" ]; then
      printf '%s\n' "$url" > "$URL_FILE"
      echo "boards reachable at $url"
      return 0
    fi
    sleep 1
    waited=$(( waited + 1 ))
  done
  echo "tunnel started but printed no URL within 40s; see $TUNNEL_LOG" >&2
  return 1
}

supervise() {
  local name=$1 delay=$MIN_DELAY
  while true; do
    local started ended ran code
    started=$(date +%s)

    if [ "$name" = "server" ]; then
      : > "$SERVER_LOG"
      BOARDS_CREDENTIALS_FILE="$CREDENTIALS" "$PYTHON" -m statuswall.board_server \
        --directory "$BOARDS_DIR" --port "$PORT" >> "$SERVER_LOG" 2>&1 &
      server_pid=$!
      wait "$server_pid"
      code=$?
      server_pid=""
    else
      # Truncated each start so publish_tunnel_url cannot read the previous
      # tunnel's hostname and report a URL that no longer resolves.
      : > "$TUNNEL_LOG"
      "$CLOUDFLARED" tunnel --url "http://127.0.0.1:$PORT" >> "$TUNNEL_LOG" 2>&1 &
      tunnel_pid=$!
      publish_tunnel_url &
      wait "$tunnel_pid"
      code=$?
      tunnel_pid=""
    fi

    ended=$(date +%s)
    ran=$(( ended - started ))
    record_restart "$name" "$code" "$ran"
    echo "$name exited ($code) after ${ran}s; restarting in ${delay}s" >&2

    # A child that ran healthily and then dropped once should not inherit the
    # accumulated delay of every restart before it.
    if [ "$ran" -ge "$HEALTHY_RUN_SECONDS" ]; then
      delay=$MIN_DELAY
    else
      delay=$(next_delay "$delay")
    fi
    sleep "$delay"
  done
}

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ensure_credentials

supervise server &
supervise tunnel &
wait
