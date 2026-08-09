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
# Usage: capture_supervisor.sh <venue> <core-symbols> [tail-symbols]
#   e.g. capture_supervisor.sh binance BTCUSDT,ETHUSDT,SOLUSDT ALL
#
# The third argument is the broad tail - the cheap channels across the wide
# universe, with no depth. "ALL" means "whatever the venue lists right now",
# resolved at startup and recorded as point-in-time membership before a frame
# is captured. Omitting it keeps the core-only behaviour this script had before
# the tail existed. Widening the tail is cheap today and impossible to backfill,
# so the default here is the one decision worth revisiting.
set -uo pipefail

VENUE=${1:?venue required}
SYMBOLS=${2:?comma-separated symbols required}
TAIL_SYMBOLS=${3:-}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STATE_DIR="$CAPTURE_ROOT/supervisor"
LOG="$STATE_DIR/$VENUE.log"
RESTART_LOG="$STATE_DIR/$VENUE.restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120

mkdir -p "$STATE_DIR"

# The directory this venue writes into, which is not always the name it is
# invoked by: `binance-funding` is a separate process polling the same venue, so
# its files land in raw/binance/ beside the trades.
#
# Resolved from the venue registry rather than mapped here, because a second copy
# of that mapping is a second thing to get wrong - and it WAS wrong: this script
# passed $VENUE straight to repair_archive, which scopes on the archive
# directory, so the funding process scoped its repair to a directory that does
# not exist and never repaired the hours it writes.
ARCHIVE_VENUE=$(PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -c \
    "from capture.cli import archive_name_for; print(archive_name_for('$VENUE'))" \
    2>/dev/null) || ARCHIVE_VENUE="$VENUE"
[ -n "$ARCHIVE_VENUE" ] || ARCHIVE_VENUE="$VENUE"

# One raw and one index file stay open per (stream, symbol) for the whole hour,
# so the broad tail needs file descriptors in proportion to the universe: 2,115
# symbols is over 4,000 before websockets and the ops files. The startup script
# runs this under `sudo -H bash -lc`, which hands the recorder PAM's default soft
# limit of 1024 - and the hard limit here is 524288, so the ceiling was never the
# system's, only the one inherited.
#
# Measured 2026-08-09: the binance recorder died on `OSError: [Errno 24] Too many
# open files` with exactly 1024 descriptors open, and the supervisor restarted it
# into the same wall. Median run length that day was 39 seconds against 880 the
# day before - each cycle losing the frames in flight and tearing whichever hour
# was mid-write.
#
# Raised, not removed: a leak should still fail rather than exhaust the box, and
# a failure to raise must not stop capture starting at the old limit.
NOFILE_TARGET=65536
if ! ulimit -n "$NOFILE_TARGET" 2>/dev/null; then
    printf '{"ts":"%s","venue":"%s","event":"nofile_raise_refused","target":%d,"soft":"%s","hard":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENUE" "$NOFILE_TARGET" \
        "$(ulimit -Sn)" "$(ulimit -Hn)" >>"$RESTART_LOG"
fi

child_pid=""
archive_repair_pid=""

# Forward the stop signal rather than dying and orphaning the recorder: its
# close() writes the zstd footers and removes the .writing marker, and an hour
# left with a stale marker blocks the repair that is the way out of it.
forward_stop() {
    # The background archive pass goes first. It only touches rotated hours, so
    # nothing is mid-capture in it, but leaving it running after the supervisor
    # exits means a reboot kills it mid-reconcile instead of letting it finish
    # the pair it is on.
    if [ -n "$archive_repair_pid" ]; then
        kill -TERM "$archive_repair_pid" 2>/dev/null
    fi
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
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENUE" "$SYMBOLS${TAIL_SYMBOLS:++tail:$TAIL_SYMBOLS}" "$$" >>"$RESTART_LOG"

# Rotated hours can be torn too - a stop tears whatever was open, and the hour
# it was open in has usually rotated by the time anyone looks. Repairing them
# still matters for the reader and the offload, but nothing is waiting on it, so
# it must not sit in front of capture. Backgrounded, once per supervisor start
# rather than once per restart, and disjoint from the blocking pass by scope:
# `archive` cannot reach the hour a recorder is about to open, which is what
# makes it safe to run beside a live writer at all.
PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.repair_archive \
    --root "$CAPTURE_ROOT" --venue "$ARCHIVE_VENUE" --scope archive \
    >>"$RESTART_LOG" 2>>"$LOG" &
archive_repair_pid=$!

delay=$MIN_DELAY
while true; do
    started_at=$(date +%s)

    # An ungraceful stop tears the zstd frame of whichever streams were mid-write,
    # and RawWriter then refuses that hour for the rest of the hour. Repair is the
    # documented exit from that refusal and a restart is when it can safely run -
    # no writer holds the files. Measured cost of skipping it: seventeen minutes
    # of the two busiest depth streams, per ungraceful stop.
    #
    # This pass is the current hour of this venue only, because that is the only
    # hour that can refuse the recorder about to start. Unscoped, each of the
    # three supervisors read every pair in the whole archive end to end first:
    # after the 2026-08-09 reboot that was 22,668 pairs and 1.9 GB, three times
    # over, and eighteen minutes later no venue had captured a frame. Everything
    # already rotated is repaired by the archive pass above, off this path.
    PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.repair_archive \
        --root "$CAPTURE_ROOT" --venue "$ARCHIVE_VENUE" --scope resumable \
        >>"$RESTART_LOG" 2>>"$LOG"

    # The tail argument is passed only when set, so a core-only invocation
    # builds exactly the command line it did before this existed.
    if [ -n "$TAIL_SYMBOLS" ]; then
        PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.cli \
            --venue "$VENUE" --symbols "$SYMBOLS" \
            --tail-symbols "$TAIL_SYMBOLS" \
            --root "$CAPTURE_ROOT" --seconds 0 >>"$LOG" 2>&1 &
    else
        PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m capture.cli \
            --venue "$VENUE" --symbols "$SYMBOLS" \
            --root "$CAPTURE_ROOT" --seconds 0 >>"$LOG" 2>&1 &
    fi
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
