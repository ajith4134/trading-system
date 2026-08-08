#!/usr/bin/env bash
# Builds the Layer 1 datasets from the raw archive, on a loop.
#
# The datasets are what the cost engine reads. Without this running they exist
# as tested code and nothing else, which is the shape this project keeps paying
# for: correct logic that nothing calls.
#
# Builds today and yesterday on every pass. Today because hours are still
# closing, yesterday because the last hours of a day close after midnight and a
# single-day build would miss them permanently. Re-running a day already built
# is refused by the partition writer as a collision, so repetition is free.
#
# Usage: scripts/store_supervisor.sh [interval-seconds]
set -uo pipefail

INTERVAL="${1:-3600}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STORE_ROOT="$CAPTURE_ROOT/store"
STATE_DIR="$CAPTURE_ROOT/store-builds"
LOG="$STATE_DIR/build.log"
RUNS="$STATE_DIR/runs.ndjson"

# The symbols carrying depth and funding. Only the core has either: the broad
# tail subscribes trades alone, so there is nothing polled to build from it.
CORE="BTCUSDT,ETHUSDT,SOLUSDT"

mkdir -p "$STATE_DIR"

trap 'exit 0' TERM INT

build() {   # dataset venue date symbols
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m store.build_polled \
    --dataset "$1" --venue "$2" --date "$3" --symbols "$4" \
    --capture-root "$CAPTURE_ROOT" --store-root "$STORE_ROOT" 2>>"$LOG"
}

while true; do
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  today=$(date -u +%F)
  yesterday=$(date -u -d 'yesterday' +%F)

  for day in "$yesterday" "$today"; do
    for spec in "funding binance" "book binance" "book binance-spot"; do
      set -- $spec
      result=$(build "$1" "$2" "$day" "$CORE")
      printf '{"ts":"%s","day":"%s","result":%s}\n' \
        "$started" "$day" "${result:-null}" >> "$RUNS"
    done
  done

  sleep "$INTERVAL"
done
