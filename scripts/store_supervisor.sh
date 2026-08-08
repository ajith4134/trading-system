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
# The same three instruments as hyperliquid names them.
CORE_HYPERLIQUID="BTC,ETH,SOL"

mkdir -p "$STATE_DIR"

trap 'exit 0' TERM INT

build() {   # dataset venue date symbols
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m store.build_polled \
    --dataset "$1" --venue "$2" --date "$3" --symbols "$4" \
    --capture-root "$CAPTURE_ROOT" --store-root "$STORE_ROOT" 2>>"$LOG"
}

# Bars come from the trade tape, not the pollers, so they have their own entry
# point - and their own day. store.cli reads whole closed hours; the hour a
# capture writer still holds open is a partial zstd frame it refuses outright
# (TruncatedFrameFile), which is correct for a CLOSED hour and fatal for today.
# So bars build yesterday only. Today's bars appear tomorrow, and the lag is
# real: replay and calibration always trail the tape by up to a day.
#
# Symbols differ by venue because the venues name the same instrument
# differently - BTCUSDT on binance, BTC on hyperliquid.
# --batch-size bounds peak memory to the batch rather than the request: the
# builder accumulates every trade of every requested symbol before building, so
# an unbatched broad-universe day does not fit in this box's RAM. Harmless at
# three symbols, and correct if the list ever grows.
build_bars() {   # venue date symbols
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m store.cli \
    --venue "$1" --date "$2" --symbols "$3" --batch-size 5 \
    --capture-root "$CAPTURE_ROOT" --store-root "$STORE_ROOT" 2>&1
}
EXIT_ALREADY_BUILT=4

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

  # Recorded whether or not it succeeded. Bars went five days stale while the
  # polled datasets kept writing a line every pass, and nothing in the run log
  # said bars had been attempted at all - because they never had been. Every
  # pass now leaves a line naming the venue-day and whether it built.
  for spec in "binance $CORE" "binance-spot $CORE" "hyperliquid $CORE_HYPERLIQUID"; do
    set -- $spec
    # Three states, not two. The partition writer refuses a rewrite, so every
    # pass after the first exits non-zero on a day it already built - and
    # recording that as "failed" would bury a genuine failure in an hourly
    # stream of expected ones.
    # Keyed on the exit code, not on matching PartitionExistsError in the output.
    # That string came from an uncaught traceback, and it stopped appearing the
    # moment the builder started catching the collision per batch - a detector
    # tied to a message nobody meant as an interface. Every already-built day
    # would then have been recorded as a failure.
    output=$(build_bars "$1" "$yesterday" "$2")
    case $? in
      0) status=built ;;
      "$EXIT_ALREADY_BUILT") status=already-built ;;
      *) status=failed ;;
    esac
    # A day the capture never covered also exits 0, with nothing built. Recording
    # the status alone would read as success; the count is what distinguishes a
    # built day from an empty one.
    bars=$(printf '%s' "$output" | grep -oE '> [0-9]+ bars' | grep -oE '[0-9]+' | head -1)
    printf '%s\n' "$output" >> "$LOG"
    printf '{"ts":"%s","day":"%s","result":{"dataset":"bars","venue":"%s","date":"%s","status":"%s","bars":%s}}\n' \
      "$started" "$yesterday" "$1" "$yesterday" "$status" "${bars:-null}" >> "$RUNS"
  done

  sleep "$INTERVAL"
done
