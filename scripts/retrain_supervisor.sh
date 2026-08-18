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

echo "$(stamp) retrain supervisor started interval=${INTERVAL}s hours=${HOURS}" >> "$LOG"

while true; do
    if [ -e "$OFF_SWITCH" ]; then
        sleep 60
        continue
    fi
    for SEGMENT in perp spot dated options; do
        echo "$(stamp) retraining $SEGMENT" >> "$LOG"
        PYTHONPATH="$REPO/src" nice -n 15 "$PYTHON" -m learn.train_segment_model \
            --segment "$SEGMENT" --hours "$HOURS" \
            --report "$STATE_DIR/$SEGMENT-retrain.json" >> "$LOG" 2>&1 \
            || echo "$(stamp) $SEGMENT retrain exited non-zero" >> "$LOG"
    done
    sleep "$INTERVAL"
done
