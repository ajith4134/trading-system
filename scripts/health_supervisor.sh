#!/usr/bin/env bash
# Keep the venue halt registry fed.
#
# `VenueHaltRegistry` was complete and uncalled from the day it was written -
# `observe()` and `assess_venue()` had zero callers anywhere in src/ - so no
# degradation could halt a venue, while the status wall rendered "auto-halt
# armed" over the top of it. This is the process that makes the claim true.
#
# Separate from boards_supervisor.sh on purpose. The wall READS the registry and
# must never write to it: a display that decides whether a venue may be traded is
# a display whose refresh cadence and error handling become trading decisions.
# Keeping them apart also keeps their failures apart - this loop dying stops
# halting, and the wall reports that as a stale observation stamp rather than
# hiding it behind its own liveness.
#
# One pass per invocation, loop here, matching every other supervisor in this
# directory: a crash costs one interval, and the restart lands in the log an
# operator already reads.
#
# Usage: health_supervisor.sh [interval-seconds]
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STATE_DIR="$CAPTURE_ROOT/health"
PYTHON=${HEALTH_PYTHON:-$REPO/.venv/bin/python}

# Sixty seconds. The registry's own recovery dwell is thirty minutes, so the
# cadence that matters is "fast enough that the stamp never looks stale to the
# wall", and the wall grades an observation older than an hour as unarmed.
INTERVAL=${1:-60}

LOG="$STATE_DIR/health.log"
RESTART_LOG="$STATE_DIR/restarts.ndjson"

mkdir -p "$STATE_DIR"

child_pid=""

forward_stop() {
    # TERM, not INT: a non-interactive shell sets SIGINT to SIG_IGN for the
    # children it starts in the background, so `kill -INT` here is a no-op and
    # the wait below never returns. Measured on capture_supervisor.sh.
    if [ -n "$child_pid" ]; then
        kill -TERM "$child_pid" 2>/dev/null
        wait "$child_pid" 2>/dev/null
    fi
    printf '{"ts":"%s","event":"supervisor_stopped"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RESTART_LOG"
    exit 0
}
trap forward_stop INT TERM

printf '{"ts":"%s","event":"supervisor_started","interval_seconds":%s,"pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$INTERVAL" "$$" >>"$RESTART_LOG"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

while true; do
    # Backgrounded and waited on so the TERM trap can reach the child: with the
    # pass in the foreground, a stop signal would sit unhandled until it
    # finished, and the same shape left capture_supervisor.sh alive through a
    # kill on 2026-08-08.
    "$PYTHON" -m ops.venue_health_watch --capture-root "$CAPTURE_ROOT" >>"$LOG" 2>&1 &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    child_pid=""

    # Recorded and retried, never fatal. A venue whose report cannot be built
    # this minute is next minute's problem; exiting over it would stop halting
    # for every other venue too. Only failures are logged - a line per healthy
    # pass every 60s would bury the one that mattered.
    if [ "$exit_code" -ne 0 ]; then
        printf '{"ts":"%s","event":"pass_failed","exit_code":%d}\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" >>"$RESTART_LOG"
    fi

    sleep "$INTERVAL"
done
