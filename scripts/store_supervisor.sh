#!/usr/bin/env bash
# Builds the POLLED Layer 1 datasets from the raw archive, on a loop.
#
# Bars are not here. They are `bars_supervisor.sh`, split out 2026-08-10 - see
# the note where they used to run.
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

# No depth symbol list either, as of 2026-08-10. It was BTCUSDT,ETHUSDT,SOLUSDT
# - correct for the two binance venues and wrong for coinbase, which names the
# same markets BTC-USD, ETH-USD and SOL-USD. One list cannot spell three venues,
# and a book build asking binance names of coinbase would have found nothing and
# said "0 rows" like a quiet day.
#
# `--symbols ALL` reads the polled symbols off the archive, which resolves to
# exactly the symbols that were subscribed for depth. Same answer, no list.
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
# No per-venue symbol list at all any more. The last one existed for the bars
# loop, which lives in bars_supervisor.sh now and reads its own symbols off the
# archive. An unused constant naming three symbols is how the next reader
# concludes a build is still core-only.

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
    for spec in "funding binance" "funding hyperliquid" "funding bybit" \
                "dated_futures bybit" "book binance" "book binance-spot" \
                "book coinbase"; do
      set -- $spec
      # Every polled dataset reads its universe off the archive now, depth
      # included. Depth is still core-only in FACT - only the core subscribes a
      # snapshot - but that is a property of what was captured rather than a
      # list this script has to keep spelled correctly for three venues.
      result=$(build "$1" "$2" "$day" "$UNIVERSE_FROM_ARCHIVE")
      printf '{"ts":"%s","day":"%s","result":%s}\n' \
        "$started" "$day" "${result:-null}" >> "$RUNS"
    done
  done

  # Exchange reserves and netflow. Not a day-keyed build: it is a live poll of a
  # third-party aggregate with no history endpoint, so a reading not taken now
  # cannot be taken later at all - the same argument the raw archive rests on.
  # Once per pass, whatever the day loop did.
  reserves=$(PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" \
    -m store.exchange_reserves --store-root "$STORE_ROOT" 2>>"$LOG")
  printf '{"ts":"%s","result":%s}\n' "$started" "${reserves:-null}" >> "$RUNS"

  # Bars are NOT built here any more. They moved to bars_supervisor.sh on
  # 2026-08-10 because they were starving this loop: a universe-wide bars build
  # takes hours, it ran inside every pass, and the polled datasets above
  # therefore refreshed once per full pass rather than once an hour. Measured
  # that morning - the 05:36 pass was still in the bars loop at 07:45, and every
  # basis value in the system was being computed from funding polls three hours
  # old while reading as a current number.
  #
  # One slow job in a loop starves everything else in it. Same reason
  # binance-funding left the recorder.

  sleep "$INTERVAL"
done
