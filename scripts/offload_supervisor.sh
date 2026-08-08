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

  # Eviction runs only behind a clean offload, and only ever removes a local file
  # whose object it has just seen listed in the bucket. Ordering it here rather
  # than on its own timer is the point: the inventory it checks against is the
  # one this pass just finished writing.
  #
  # KEEP_DAYS=0 disables it entirely, which is the default. Deleting the local
  # copy of the archive is not something a supervisor should start doing because
  # a script was updated - it is switched on deliberately, by setting the value.
  if [ "$status" -eq 0 ] && [ "${KEEP_DAYS:-0}" -gt 0 ]; then
    evicted=$(PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m ops.raw_eviction \
      --bucket "$BUCKET" --capture-root "$CAPTURE_ROOT" \
      --keep-days "$KEEP_DAYS" --apply 2>>"$LOG")
    printf '{"ts":"%s","eviction":%s}\n' "$started" "${evicted:-null}" >> "$RUNS"
  fi

  sleep "$INTERVAL"
done
