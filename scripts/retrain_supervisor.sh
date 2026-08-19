#!/usr/bin/env bash
# LB-08: refit every segment's brains on a cadence, with nobody invoking anything.
#
# §1a L2 draws the line this script sits on: "if the only path that changes
# parameters is a script a human invokes, that is scheduled retraining - a
# legitimate but DIFFERENT claim, and it must be labelled as such". This is that
# path, made autonomous so it is at least not waiting on a person; the board
# labels what it produced "scheduled retrain" and labels the calibration the live
# loop fits "live loop", because they are different claims.
#
# NICED, and deliberately. Training reads the store, and the store read is what
# drove memory to 25 GB of 29 on 2026-08-18 and threatened the running bots - the
# same OOM that killed the paper engine on 2026-08-09. The bots trading on live
# prices matter more than a refit finishing quickly.
#
# A refit that finds no edge is a NORMAL outcome: `train_direction_model` refuses
# to register a model that only reproduced the majority class, the champion alias
# keeps pointing at whatever last passed, and the loop sleeps and tries again.
#
# It also RUNS the §1a axis probes after each fit and writes them beside the
# report, so the verdicts on the wall come from probes that ran rather than from a
# shell session somebody remembers having.
#
# Usage: retrain_supervisor.sh [interval-seconds] [hours-of-history]
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
PYTHON=${RETRAIN_PYTHON:-$REPO/.venv/bin/python}

INTERVAL=${1:-3600}
HOURS=${2:-14}
STATE_DIR="$CAPTURE_ROOT/learn"
LOG="$STATE_DIR/retrain.log"
OFF_SWITCH="$STATE_DIR/OFF"

mkdir -p "$STATE_DIR"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# **A fit that outlives the supervisor that started it is the bug this trap
# answers, measured 2026-08-19.** The trainer ran in the FOREGROUND here, so a
# TERM to this script could not be handled until the fit returned - up to forty
# minutes later - and killing the supervisor left the fit running as a child of
# init. The result was two `train_segment_model --segment perp` processes writing
# the SAME `perp-retrain.json`, holding a gigabyte each on a box that OOMs.
#
# TERM rather than INT for the reason capture_supervisor.sh measured on
# 2026-08-08: a non-interactive shell sets SIGINT to SIG_IGN for the children it
# backgrounds, so `kill -INT` here is a no-op and the wait below never returns.
child_pid=""

retrain_stop() {
    if [ -n "$child_pid" ]; then
        echo "$(stamp) stopping, terminating fit $child_pid" >> "$LOG"
        kill -TERM "$child_pid" 2>/dev/null
        wait "$child_pid" 2>/dev/null
    fi
    echo "$(stamp) retrain supervisor stopped" >> "$LOG"
    exit 0
}
trap retrain_stop TERM INT

# **A sleep in the foreground swallows the signal for its whole duration.** Bash
# runs a trap only when the current foreground command returns, and this loop
# sleeps FOUR HOURS between passes - so a plain `sleep "$INTERVAL"` means a stop
# request is honoured up to four hours later, which reads as a supervisor that
# ignored it. Backgrounded and waited on, the wait returns the moment the signal
# arrives.
interruptible_sleep() {
    sleep "$1" &
    child_pid=$!
    wait "$child_pid" 2>/dev/null
    child_pid=""
}

echo "$(stamp) retrain supervisor started interval=${INTERVAL}s hours=${HOURS} pid=$$" >> "$LOG"

while true; do
    if [ -e "$OFF_SWITCH" ]; then
        interruptible_sleep 60
        continue
    fi
    for SEGMENT in perp spot dated options; do
        echo "$(stamp) retraining $SEGMENT" >> "$LOG"
        # Backgrounded and waited on so the TERM trap can reach the fit. With it
        # in the foreground the trap sits unhandled until the fit returns, which
        # is how an orphan survived its supervisor on 2026-08-19.
        PYTHONPATH="$REPO/src" nice -n 15 "$PYTHON" -m learn.train_segment_model \
            --segment "$SEGMENT" --hours "$HOURS" \
            --report "$STATE_DIR/$SEGMENT-retrain.json" >> "$LOG" 2>&1 &
        child_pid=$!
        wait "$child_pid"
        fit_exit=$?
        child_pid=""
        [ "$fit_exit" -eq 0 ] \
            || echo "$(stamp) $SEGMENT retrain exited non-zero ($fit_exit)" >> "$LOG"
    done
    interruptible_sleep "$INTERVAL"
done
