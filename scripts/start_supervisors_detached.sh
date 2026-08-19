#!/usr/bin/env bash
# Start every supervisor inside its own DETACHED screen session.
#
# **Why this exists, measured three times on 2026-08-19.** The segment bots, the
# retrainer and the boards generator were started as background jobs of an
# interactive session, and every time that session ended they died with it - the
# bots stopped trading and nothing said so until a probe was asked. A process
# started from a session is a child of that session; a detached `screen` is a
# child of init and outlives it.
#
# This is the WITHOUT-ROOT path. The durable path is the GCE metadata startup
# script, which runs as root at boot and is what brings everything back after a
# reboot - this script does not replace it and does not survive one.
#
#   scripts/start_supervisors_detached.sh          # start whatever is missing
#   screen -ls                                     # what is running
#   screen -r segment-perp                         # attach to one
#
# Idempotent: a supervisor already running is left alone rather than doubled,
# because two supervisors on one segment would both restart the same engine and
# fight over its journal.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_ROOT=${CAPTURE_ROOT:-$HOME/capture}

start_detached() {
    local name=$1 command=$2 pattern=$3
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "already running: $name"
        return 0
    fi
    screen -dmS "$name" bash -lc "cd $REPO && $command"
    sleep 1
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "started: $name"
    else
        echo "FAILED to start: $name" >&2
        return 1
    fi
}

failed=0
for segment in perp spot dated options; do
    start_detached "segment-$segment" \
        "scripts/segment_bot_supervisor.sh $segment 6 >> $LOG_ROOT/segment-$segment-supervisor.log 2>&1" \
        "segment_bot_supervisor.sh $segment" || failed=1
done

start_detached "retrain" \
    "scripts/retrain_supervisor.sh 14400 30 >> $LOG_ROOT/retrain-supervisor.log 2>&1" \
    "retrain_supervisor.sh" || failed=1

start_detached "boards" \
    "scripts/boards_supervisor.sh >> $LOG_ROOT/boards-supervisor.log 2>&1" \
    "boards_supervisor.sh" || failed=1

exit "$failed"
