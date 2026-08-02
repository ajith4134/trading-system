#!/usr/bin/env bash
# Keep one venue's recorder running.
#
# `cli._stream_frames` opens a single websocket and the frame iterator ends when
# that connection does - a disconnect stops the run and nothing restarts it. This
# is the smallest thing that makes unattended capture possible, and it is
# deliberately a shell loop rather than a supervision framework: the recorder
# already owns its own crash safety, so all that is missing is "start it again".
#
# Backoff is capped exponential with jitter. The jitter matters because both
# venues are supervised by identical loops and a venue-side outage stops them at
# the same instant - without it they reconnect in lockstep forever.
#
# A run that survived longer than HEALTHY_RUN_SECONDS resets the backoff. Without
# that, a process that runs happily for six hours and then drops once would come
# back with the full accumulated delay of every restart before it.
#
# Usage: capture_supervisor.sh <venue> <symbols>
#   e.g. capture_supervisor.sh binance BTCUSDT,ETHUSDT,SOLUSDT
set -uo pipefail

VENUE=${1:?venue required}
SYMBOLS=${2:?comma-separated symbols required}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STATE_DIR="$CAPTURE_ROOT/supervisor"
LOG="$STATE_DIR/$VENUE.log"
RESTART_LOG="$STATE_DIR/$VENUE.restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120

mkdir -p "$STATE_DIR"

child_pid=""

# Forward the stop signal rather than dying and orphaning the recorder: its
# close() writes the zstd footers and removes the .writing marker, and an hour
# left with a stale marker blocks the repair that is the way out of it.
forward_stop() {
    # TERM, not INT. A non-interactive shell sets SIGINT to SIG_IGN for the
    # children it starts in the background and the disposition survives exec, so
    # `kill -INT` here is a no-op and the wait below never returns. Measured: the
    # supervisor hung and the recorder kept running. The recorder installs a
    # SIGTERM handler that routes into its normal shutdown, so this closes the
    # zstd footers and removes the .writing markers.
    if [ -n "$child_pid" ]; then
        kill -TERM "$child_pid" 2>/dev/null
        wait "$child_pid" 2>/dev/null
    fi
    printf '{"ts":"%s","venue":"%s","event":"supervisor_stopped"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENUE" >>"$RESTART_LOG"
    exit 0
}
trap forward_stop INT TERM

printf '{"ts":"%s","venue":"%s","event":"supervisor_started","symbols":"%s","pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENUE" "$SYMBOLS" "$$" >>"$RESTART_LOG"

delay=$MIN_DELAY
while true; do
    started_at=$(date +%s)

    # An ungraceful stop tears the zstd frame of whichever streams were mid-write,
    # and RawWriter then refuses that hour for the rest of the hour. Repair is the
    # documented exit from that refusal and a restart is when it can safely run -
    # no writer holds the files. Measured cost of skipping it: seventeen minutes
    # of the two busiest depth streams, per ungraceful stop.
    PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.repair_archive \
        --root "$CAPTURE_ROOT" >>"$RESTART_LOG" 2>>"$LOG"

    PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.cli \
        --venue "$VENUE" --symbols "$SYMBOLS" \
        --root "$CAPTURE_ROOT" --seconds 0 >>"$LOG" 2>&1 &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    child_pid=""

    ran_for=$(( $(date +%s) - started_at ))
    if [ "$ran_for" -ge "$HEALTHY_RUN_SECONDS" ]; then
        delay=$MIN_DELAY
    fi

    # The dark window is the supervisor's own observation, recorded where a human
    # looks during an incident. The recorder cannot log its own downtime.
    printf '{"ts":"%s","venue":"%s","event":"recorder_exited","exit_code":%d,"ran_for_seconds":%d,"restart_in_seconds":%d}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENUE" "$exit_code" "$ran_for" "$delay" >>"$RESTART_LOG"

    sleep $(( delay + (RANDOM % (delay + 1)) ))

    delay=$(( delay * 2 ))
    [ "$delay" -gt "$MAX_DELAY" ] && delay=$MAX_DELAY
done
