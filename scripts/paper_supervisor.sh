#!/usr/bin/env bash
# The forward paper engine, supervised alongside capture, store and bars.
#
# This script is also what makes the paper subsystem REACHABLE. Until it existed,
# position_book, paper_broker, market_replay and forward_engine imported each
# other with nothing running any of them - a dead subsystem vouching for itself,
# which is precisely the shape integrity.unsupported_claims exists to catch, and
# which every one of their axis verdicts recorded as an honest DEPTH failure.
# `invoked_modules` reads roots out of this directory, so the `-m` below is not
# incidental: it is the thing that turns four modules from dead code into a path.
#
# THE STRATEGY IS NAMED ON THE COMMAND LINE AND HAS NO DEFAULT. The only signal
# that exists today is a plumbing signal that makes no edge claim, and an engine
# running it implicitly would accumulate a P&L series that reads, six weeks later,
# as a result. Every fill row carries makes_edge_claim=false for exactly this
# reason. When Phase C produces a model, it is named here and this comment is what
# should be re-read before it is.
#
# PARTICIPATION HAS NO DEFAULT EITHER. It is open question 1 of the paper
# execution design and is not settled; an invented rate is the single cheapest way
# to manufacture edge, so it must be typed by a human who knows they typed it.
# Until paper.participation_calibration has a receipt on disk, every fill carries
# uncalibrated=true and no promotion may read it.
#
# One long-running child, restarted on exit, matching the other supervisors here:
# the engine writes a heartbeat on EVERY poll, so a crash shows up on the wall as
# a stale heartbeat within one interval rather than as an absence that looks like
# a quiet market.
#
# Usage: paper_supervisor.sh [strategy] [participation] [interval-seconds]
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STORE_ROOT=${STORE_ROOT:-$CAPTURE_ROOT/store}
STATE_DIR="$CAPTURE_ROOT/paper/forward"
PYTHON=${PAPER_PYTHON:-$REPO/.venv/bin/python}

STRATEGY=${1:-plumbing-momentum}
PARTICIPATION=${2:-0.1}
INTERVAL=${3:-60}

LOG="$STATE_DIR/engine.log"
RESTART_LOG="$STATE_DIR/restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120

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

next_delay() {
    local delay=$1
    local doubled=$(( delay * 2 ))
    [ "$doubled" -gt "$MAX_DELAY" ] && doubled=$MAX_DELAY
    echo $(( doubled + (RANDOM % (delay > 0 ? delay : 1)) ))
}

printf '{"ts":"%s","event":"supervisor_started","strategy":"%s","participation":"%s","interval_seconds":%s,"pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STRATEGY" "$PARTICIPATION" "$INTERVAL" "$$" \
    >>"$RESTART_LOG"

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

delay=$MIN_DELAY
while true; do
    started=$(date +%s)

    # Backgrounded and waited on so the TERM trap can reach the child: with the
    # engine in the foreground a stop signal would sit unhandled until the next
    # poll returned, and the same shape left capture_supervisor.sh alive through
    # a kill on 2026-08-08.
    "$PYTHON" -m paper.forward_engine \
        --strategy "$STRATEGY" \
        --participation "$PARTICIPATION" \
        --store-root "$STORE_ROOT" \
        --capture-root "$CAPTURE_ROOT" \
        --interval-seconds "$INTERVAL" >>"$LOG" 2>&1 &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    child_pid=""

    ended=$(date +%s)
    ran=$(( ended - started ))
    printf '{"ts":"%s","event":"engine_exited","exit_code":%d,"ran_seconds":%d}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exit_code" "$ran" >>"$RESTART_LOG"

    # A kill switch is not a crash to retry into. ops.watchdog's kill file is the
    # one control that must not be defeated by a restart loop, so a refusal to
    # start under a kill backs off to the long delay rather than hammering.
    if [ "$ran" -ge "$HEALTHY_RUN_SECONDS" ]; then
        delay=$MIN_DELAY
    else
        delay=$(next_delay "$delay")
    fi
    sleep "$delay"
done
