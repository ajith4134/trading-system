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

# The symbols carrying depth. Only the core has it: the broad tail subscribes
# trades alone, and there is no all-market form of a depth snapshot.
CORE="BTCUSDT,ETHUSDT,SOLUSDT"
# The polled datasets are no longer core-only and their symbols are no longer
# named here. Since 2026-08-09 the venues poll for the whole market - 863
# instruments on binance, 232 on hyperliquid, 805 on bybit - and
# `build_polled --symbols ALL` reads the list off the archive.
#
# A hand-kept list is exactly what cost the trade tape 99.6% of itself: capture
# subscribed 2,098 symbols while this supervisor asked for 9, and raw is evicted
# after seven days so those days could not be recovered by fixing the list
# later. No polled dataset is going to repeat it.
#
# Named for what it means rather than for the one dataset that first needed it:
# `dated_futures` reads the same universe and is not funding.
UNIVERSE_FROM_ARCHIVE="ALL"
# No hyperliquid core list any more: it existed only for the bars loop, which now
# reads its symbols off the archive. An unused constant naming three symbols is
# how the next reader concludes the build is still core-only.

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
# --symbols ALL, not a list. The venues name the same instrument differently -
# BTCUSDT on binance, BTC on hyperliquid - so any hand-written list is three
# lists, and all three were the core: capture subscribed 2,098 symbols on
# 2026-08-08 and this loop asked for 9, so 99.6% of the tape was archived and
# never became a bar. A list also cannot be kept correct. A symbol listing
# mid-day is captured within the minute and would wait for a human, and the raw
# it was captured from is evicted after seven days - so a day built from a stale
# list is a day that cannot be built again. store.cli reads the symbols off the
# archive instead, which is the only thing that knows what was captured.
#
# --batch-size bounds peak memory to the batch rather than the request: the
# builder accumulates every trade of every requested symbol before building, so
# an unbatched broad-universe day does not fit in this box's RAM. Measured
# 2026-08-08: 2.7 GB peak in batches of five over 569 symbols, against ~25 GB
# extrapolated unbatched.
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
    for spec in "funding binance" "funding hyperliquid" "funding bybit" \
                "dated_futures bybit" "book binance" "book binance-spot"; do
      set -- $spec
      # Funding and the dated contracts read their universe off the archive -
      # they come out of the same bybit poll and neither list is ours to keep.
      # Depth is core-only: only the core subscribes it.
      case "$1" in
        funding|dated_futures) symbols="$UNIVERSE_FROM_ARCHIVE" ;;
        *) symbols="$CORE" ;;
      esac
      result=$(build "$1" "$2" "$day" "$symbols")
      printf '{"ts":"%s","day":"%s","result":%s}\n' \
        "$started" "$day" "${result:-null}" >> "$RUNS"
    done
  done

  # Recorded whether or not it succeeded. Bars went five days stale while the
  # polled datasets kept writing a line every pass, and nothing in the run log
  # said bars had been attempted at all - because they never had been. Every
  # pass now leaves a line naming the venue-day and whether it built.
  for venue in binance binance-spot hyperliquid; do
    # Three states, not two. The partition writer refuses a rewrite, so every
    # pass after the first exits non-zero on a day it already built - and
    # recording that as "failed" would bury a genuine failure in an hourly
    # stream of expected ones.
    # Keyed on the exit code, not on matching PartitionExistsError in the output.
    # That string came from an uncaught traceback, and it stopped appearing the
    # moment the builder started catching the collision per batch - a detector
    # tied to a message nobody meant as an interface. Every already-built day
    # would then have been recorded as a failure.
    output=$(build_bars "$venue" "$yesterday" ALL)
    code=$?
    # Four states. A day no capture covered exits 0 having built nothing, and
    # recording that as "built" is a green tile for work that did not happen -
    # which is what this loop already logged for 2026-08-07, three venues at once,
    # with a null bar count beside it. It gets its own name.
    case $code in
      0) status=built
         printf '%s' "$output" | grep -q 'nothing to build' && status=no-capture ;;
      "$EXIT_ALREADY_BUILT") status=already-built ;;
      *) status=failed ;;
    esac
    # And the counts regardless, because the status is a claim and these are the
    # evidence for it: "built" beside a null symbol count is not a build.
    bars=$(printf '%s' "$output" | grep -oE '> [0-9]+ bars' | grep -oE '[0-9]+' | head -1)
    # And the symbol count, because with ALL the breadth is no longer a constant
    # this script sets - it is whatever the archive held, and it is the number
    # that was wrong for five days while every other field read healthy. A run
    # log that reports bars without breadth cannot tell 9 symbols from 2,098.
    symbols=$(printf '%s' "$output" | grep -oE '^[0-9]+ symbol\(s\) captured' \
      | grep -oE '^[0-9]+' | head -1)
    printf '%s\n' "$output" >> "$LOG"
    printf '{"ts":"%s","day":"%s","result":{"dataset":"bars","venue":"%s","date":"%s","status":"%s","symbols":%s,"bars":%s}}\n' \
      "$started" "$yesterday" "$venue" "$yesterday" "$status" "${symbols:-null}" "${bars:-null}" >> "$RUNS"
  done

  sleep "$INTERVAL"
done
