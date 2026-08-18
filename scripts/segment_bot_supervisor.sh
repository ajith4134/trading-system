#!/usr/bin/env bash
# BF-06 / PB-08: one segment bot, supervised, 24/7 (RL-020), with an on/off switch.
#
# One child per segment, restarted on exit, matching the other supervisors here.
# The engine writes a heartbeat on EVERY poll, so a crash shows on the wall as a
# STALE heartbeat within one interval rather than as an absence that reads like a
# quiet market.
#
# THE SEGMENT IS NAMED ON THE COMMAND LINE AND HAS NO DEFAULT. Four bots exist and
# they are not interchangeable - RL-019 makes each its own architecture with its own
# features - so a supervisor that guessed would be starting a bot nobody chose.
#
# THE OFF SWITCH IS A FILE, and it is per segment: touch
# ~/capture/segment/<segment>/OFF and the loop stops that bot at its next check
# WITHOUT stopping capture, which is PB-08's acceptance in those words. Removing the
# file starts it again. A file rather than a signal because it survives a reboot and
# can be read by the board, so a bot that is off renders as deliberately off rather
# than as crashed.
#
# Usage: segment_bot_supervisor.sh <perp|spot|dated|options> [interval-seconds]
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
PYTHON=${SEGMENT_PYTHON:-$REPO/.venv/bin/python}

SEGMENT=${1:?segment required: perp, spot, dated or options}
INTERVAL=${2:-6}

STATE_DIR="$CAPTURE_ROOT/segment/$SEGMENT"
LOG="$STATE_DIR/engine.log"
RESTARTS="$STATE_DIR/restarts.ndjson"
OFF_SWITCH="$STATE_DIR/OFF"

mkdir -p "$STATE_DIR"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

record() {
    printf '{"ts":"%s","event":"%s","segment":"%s","detail":"%s"}\n' \
        "$(stamp)" "$1" "$SEGMENT" "${2:-}" >> "$RESTARTS"
}

record supervisor_started "interval=${INTERVAL}s pid=$$"

# Stop the child on TERM rather than leaving it orphaned holding a websocket.
CHILD=""
terminate() {
    record supervisor_stopped ""
    [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
    exit 0
}
trap terminate TERM INT

while true; do
    if [ -e "$OFF_SWITCH" ]; then
        # Deliberately off. Checked often enough to be responsive, and recorded once
        # per transition rather than every loop so the ledger stays readable.
        if [ "${WAS_OFF:-0}" != "1" ]; then
            record switched_off "$OFF_SWITCH exists"
            WAS_OFF=1
        fi
        sleep 10
        continue
    fi
    if [ "${WAS_OFF:-0}" = "1" ]; then
        record switched_on "$OFF_SWITCH removed"
        WAS_OFF=0
    fi

    STARTED=$(date +%s)
    PYTHONPATH="$REPO/src" "$PYTHON" -m segment.live_engine \
        --segment "$SEGMENT" \
        --root "$CAPTURE_ROOT/segment" \
        --interval-seconds "$INTERVAL" >> "$LOG" 2>&1 &
    CHILD=$!
    wait "$CHILD"
    CODE=$?
    CHILD=""
    RAN=$(( $(date +%s) - STARTED ))
    record engine_exited "exit_code=$CODE ran_seconds=$RAN"

    # A child that dies immediately and repeatedly would spin the box; back off to
    # one attempt every 15 seconds in that case. A long-running child that exits is
    # restarted at once, because that is a bot that should be trading.
    if [ "$RAN" -lt 15 ]; then
        sleep 15
    else
        sleep 2
    fi
done
