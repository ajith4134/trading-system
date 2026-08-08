#!/usr/bin/env bash
# Runs the GCS offload on a loop, and records every run whether it worked or not.
#
# On a timer rather than after each hour rotates, because the archive has ~2,100
# streams whose hours do not rotate together, and because a run that finds
# nothing new costs 30 seconds. Missing an hour costs data that cannot be
# recovered.
#
# The failure this guards against is silence: an offload that stops working
# leaves the archive un-backed-up while everything else looks healthy. Every
# run appends a line, so "when did it last succeed" is answerable from disk.
#
# Usage: scripts/offload_supervisor.sh gs://bucket-name [interval-seconds]
set -uo pipefail

BUCKET="${1:?usage: offload_supervisor.sh gs://bucket-name [interval-seconds]}"
INTERVAL="${2:-1800}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STATE_DIR="$CAPTURE_ROOT/offload"
LOG="$STATE_DIR/offload.log"
RUNS="$STATE_DIR/runs.ndjson"

mkdir -p "$STATE_DIR"
export PATH="/snap/bin:$PATH"

trap 'exit 0' TERM INT

while true; do
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  started_epoch=$(date +%s)

  result=$("$REPO/scripts/offload_to_gcs.sh" "$BUCKET" "$CAPTURE_ROOT" 2>>"$LOG")
  status=$?

  # The offload script's own JSON is kept verbatim rather than re-summarised, so
  # the counts in the run log are the ones it actually measured.
  printf '{"ts":"%s","ran_seconds":%s,"exit_code":%s,"result":%s}\n' \
    "$started" "$(( $(date +%s) - started_epoch ))" "$status" \
    "${result:-null}" >> "$RUNS"

  sleep "$INTERVAL"
done
