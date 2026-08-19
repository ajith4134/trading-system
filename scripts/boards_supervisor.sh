#!/usr/bin/env bash
# Keep the boards reachable AND current: a generator, an authenticated local
# server, and a tunnel to it.
#
# Four children, supervised independently, because they fail independently: the
# generator dies on a probe error, the blotter on a torn journal read, the server
# on a code error, the tunnel when Cloudflare drops the edge connection.
# Restarting all four when only one died would change the public URL for no reason.
#
# The blotter is separate from the generator rather than part of its pass, and the
# reason is measured: the wall's pass takes ten to nineteen minutes because every
# probe reads the store, while the blotter reads one directory of NDJSON in about
# 0.2 seconds. Riding the pass meant the blotter could never be fresher than the
# slowest probe on the board.
#
# The generator exists because this script used to serve boards and never rebuild
# them. On 2026-08-08 the wall being served was five days old - written 2026-08-03
# 17:46 - while this supervisor had been up the whole time reporting nothing wrong,
# because serving a file and refreshing it are different jobs and only one of them
# was anyone's. A board that is stale is worse than no board: it reassures exactly
# when attention was required. See Rule 8.
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

GENERATOR_LOG="$STATE_DIR/generator.log"
SERVER_LOG="$STATE_DIR/server.log"
TUNNEL_LOG="$STATE_DIR/tunnel.log"
URL_FILE="$STATE_DIR/current-url.txt"
RESTART_LOG="$STATE_DIR/restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120
# How often the wall is rebuilt from live measurement. Every probe reads the
# archive, the store and the repo, so this is not free - but it is seconds against
# an interval of minutes, and the alternative is a board whose age is unbounded.
REGENERATE_INTERVAL=${BOARDS_REGENERATE_INTERVAL:-300}
# The blotter has its own, much shorter interval and its own child. It reads one
# directory of NDJSON in about 0.2s, where the wall's measurement pass takes ten
# to nineteen minutes because every probe reads the store. Riding that pass meant
# the blotter could never be fresher than the slowest probe on the board - the
# user opened it 2026-08-16 and was served a page 3.5 hours old showing zero
# closed trades when four had closed.
BLOTTER_INTERVAL=${BOARDS_BLOTTER_INTERVAL:-60}
BLOTTER_LOG="$STATE_DIR/blotter.log"
WALL_OUT="$BOARDS_DIR/status-wall.html"

mkdir -p "$STATE_DIR"

generator_pid=""
wall_pid=""
server_pid=""
tunnel_pid=""
blotter_pid=""

record_restart() {
  printf '{"ts":"%s","child":"%s","exit_code":%s,"ran_seconds":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$RESTART_LOG"
}

# Forward the stop signal rather than dying and orphaning the children: an
# orphaned cloudflared keeps a tunnel alive that nothing is supervising, and the
# next start would publish a second URL to the same boards.
stop_children() {
  for pid in "$generator_pid" "$server_pid" "$tunnel_pid" "$blotter_pid" "$wall_pid"; do
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

    if [ "$name" = "generator" ]; then
      # Rebuilt in place on an interval. The wall is written whole by
      # `statuswall.cli`, so a reader either gets the previous file or the new one.
      # Failures are recorded and retried rather than fatal: a probe that raises
      # must not leave the board frozen with nothing saying so.
      wall_pid=""
      while true; do
        # **CHEAP BOARDS FIRST, AND NOTHING HERE BREAKS THE LOOP.** Measured
        # 2026-08-19 after a reboot: `statuswall.cli` was OOM-killed three times
        # (11.7 GB resident on a 30 GB box), and because the failure broke this
        # loop the segment and plan boards had not regenerated for 15 and 43
        # hours - while the code above them said in words that a failure is
        # "recorded and retried rather than fatal". A board that stops
        # regenerating is the Rule 8 failure this whole supervisor exists to
        # prevent, and it was caused by the ordering, not by the probes.
        #
        # BF-08: the four segment bots' wall. Cheap - four heartbeats and the
        # day's fill journals.
        if ! "$PYTHON" -m statuswall.segment_tiles --out-dir "$BOARDS_DIR" \
                >> "$GENERATOR_LOG" 2>&1; then
            printf '%s segment_tiles failed\n' "$(date -u +%FT%TZ)" >> "$GENERATOR_LOG"
        fi

        # The plan and ruling-conformance boards, in their OWN short-lived
        # process rather than inside `statuswall.cli`. Measured 2026-08-17:
        # statuswall.cli holds 4.8 GB while capture holds ~9 GB on a 30 GB box
        # with no swap, and the kernel OOM-killed the forward paper engine at
        # 09:49:54Z (exit 137). Folding a 2,738-member corpus sweep into that
        # same process would buy a tidier pass at the cost of the thing the
        # system exists to run.
        if ! "$PYTHON" -m plan.cli --out-dir "$BOARDS_DIR" >> "$GENERATOR_LOG" 2>&1; then
          echo "plan board regeneration failed; see $GENERATOR_LOG" >&2
        fi

        # **The feature wall runs BESIDE this loop, not inside it.** Measured
        # 2026-08-19: one wall pass ran 25 minutes and had not finished, so the
        # cheap boards above - which take 30 seconds between them - inherited its
        # cadence and were half an hour stale by the time it returned. Ordering
        # alone was not enough; the wall has to stop being the clock.
        #
        # At most ONE wall at a time. A second started while the first is still
        # walking would double a footprint already measured at 11.7 GB on a 30 GB
        # box, and OOM is how this board stopped updating in the first place.
        if [ -n "$wall_pid" ] && kill -0 "$wall_pid" 2>/dev/null; then
          printf '%s wall still running from an earlier pass (pid %s); not starting a second\n' \
              "$(date -u +%FT%TZ)" "$wall_pid" >> "$GENERATOR_LOG"
        else
          "$PYTHON" -m statuswall.cli --out "$WALL_OUT" >> "$GENERATOR_LOG" 2>&1 &
          wall_pid=$!
        fi
        sleep "$REGENERATE_INTERVAL"
      done &
      generator_pid=$!
      wait "$generator_pid"
      code=$?
      generator_pid=""
    elif [ "$name" = "blotter" ]; then
      "$PYTHON" -m statuswall.blotter_cli \
        --capture-root "$HOME/capture" \
        --out "$BOARDS_DIR/blotter.html" \
        --interval-seconds "$BLOTTER_INTERVAL" >> "$BLOTTER_LOG" 2>&1 &
      blotter_pid=$!
      wait "$blotter_pid"
      code=$?
      blotter_pid=""
    elif [ "$name" = "server" ]; then
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

supervise generator &
supervise blotter &
supervise server &
supervise tunnel &
wait
