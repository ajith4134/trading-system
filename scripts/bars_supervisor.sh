#!/usr/bin/env bash
# Builds OHLCV bars from the trade tape, on a loop of its own.
#
# Split out of store_supervisor.sh on 2026-08-10, and the reason is measured
# rather than tidiness. Bars and polled datasets were one loop, so a pass looked
# like this:
#
#   funding + book + dated_futures + reserves   (seconds)
#   bars, four venues, --symbols ALL, one day   (HOURS)
#   sleep 3600
#
# The polled builds therefore refreshed once per FULL pass, not once an hour.
# Measured that morning: the 05:36 pass was still inside the bars loop at 07:45,
# and `features.staleness` - built the same day - reported 1,894 of 1,894 basis
# values STALE, computed from funding polls 3.03 hours old against a 61-second
# feed. Nothing had said so before, because a stale number looks exactly like a
# fresh one.
#
# This is the same split, for the same reason, as binance-funding leaving the
# recorder: one slow job inside a loop starves everything else in it.
#
# The bars build is idempotent and refuses a rebuild by snapshot id BEFORE
# parsing anything, so running it more often than a day changes costs little.
#
# TWO PASSES SINCE 2026-08-15, and the second one is why paper trading can tick
# at all. This comment used to say "bars build YESTERDAY only... today's bars
# appear tomorrow, and that lag is real", and treated that as a property of the
# store. It is not: it was a property of THIS FILE. store.cli already skips an
# hour a writer still holds, in the day's own folder as well as the lookahead,
# and names every skipped hour in the snapshot id - so a later pass, with more
# hours closed, is a different snapshot that appends the newly readable bars
# rather than a refused duplicate. Verified by running it: building 2026-08-15
# at 19:02 produced 88 BTCUSDT bars and skipped the live hour 19 by name, and
# the forward paper engine fed those 88 bars on its next poll and produced 43
# fills. Before that pass existed the engine had run 63 polls and traded nothing.
#
#   DAILY    yesterday, every venue, --symbols ALL. The complete build.
#   INTRADAY today, every venue, CORE SYMBOLS ONLY. What forward trading eats.
#
# The intraday pass is bounded to the core symbols on purpose. It rebuilds the
# whole of today from hour 0 each time - the day partition is written whole - so
# its cost grows through the day, and a universe-wide version of it would be
# quadratic in a way that eats the hour it runs in. The paper execution design
# already tiers exactly this way: broad discovery on the daily build, finalists
# on BTC/ETH/SOL. The core lists are per venue because the venues name the same
# instrument differently, which is the same reason --symbols ALL exists below.
#
# Usage: bars_supervisor.sh [interval-seconds]
set -uo pipefail

INTERVAL="${1:-3600}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STORE_ROOT="$CAPTURE_ROOT/store"
STATE_DIR="$CAPTURE_ROOT/store-builds"
LOG="$STATE_DIR/bars.log"
# The same run log the polled builds write to. One file, because "what did the
# store do last night" is one question, and two files is how half an answer
# gets read as the whole one.
RUNS="$STATE_DIR/runs.ndjson"

# Which venues have a trade tape. Not every captured venue does: bybit is polled
# for funding and bybit-liq records liquidations, so asking either for bars
# would log "no-capture" forever about a venue that was never going to have any.
VENUES="binance binance-spot hyperliquid coinbase"

# Core symbols per venue, for the intraday pass. Read off the archive on
# 2026-08-15 rather than assumed: binance perp and spot use BTCUSDT, hyperliquid
# uses bare BTC, coinbase uses BTC-USD. A single list would silently build three
# venues and miss the other one, which is exactly how 99.6% of the tape went
# unbuilt in August.
core_symbols_for() {
  case "$1" in
    binance|binance-spot) echo "BTCUSDT,ETHUSDT,SOLUSDT" ;;
    hyperliquid)          echo "BTC,ETH,SOL" ;;
    coinbase)             echo "BTC-USD,ETH-USD,SOL-USD" ;;
    *)                    echo "" ;;
  esac
}

mkdir -p "$STATE_DIR"

trap 'exit 0' TERM INT

# --symbols ALL, not a list. The venues name the same instrument differently -
# BTCUSDT on binance, BTC on hyperliquid, BTC-USD on coinbase - so any
# hand-written list is four lists, and all four were the core: capture
# subscribed 2,098 symbols on 2026-08-08 and the supervisor asked for 9, so
# 99.6% of the tape was archived and never became a bar. store.cli reads the
# symbols off the archive instead, which is the only thing that knows what was
# captured.
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
  yesterday=$(date -u -d 'yesterday' +%F)

  for venue in $VENUES; do
    # Four states, not two. The partition writer refuses a rewrite, so every
    # pass after the first exits non-zero on a day it already built - and
    # recording that as "failed" would bury a genuine failure in a stream of
    # expected ones.
    #
    # Keyed on the exit code, not on matching PartitionExistsError in the
    # output. That string came from an uncaught traceback and stopped appearing
    # the moment the builder caught the collision per batch - a detector tied to
    # a message nobody meant as an interface.
    output=$(build_bars "$venue" "$yesterday" ALL)
    code=$?
    # A day no capture covered exits 0 having built nothing, and recording that
    # as "built" is a green tile for work that did not happen - which is what
    # this loop logged for 2026-08-07, three venues at once, with a null bar
    # count beside it. It gets its own name.
    case $code in
      0) status=built
         printf '%s' "$output" | grep -q 'nothing to build' && status=no-capture ;;
      "$EXIT_ALREADY_BUILT") status=already-built ;;
      *) status=failed ;;
    esac
    # The counts regardless, because the status is a claim and these are the
    # evidence for it: "built" beside a null symbol count is not a build, and
    # breadth is the field that read healthy for five days while being wrong.
    bars=$(printf '%s' "$output" | grep -oE '> [0-9]+ bars' | grep -oE '[0-9]+' | head -1)
    symbols=$(printf '%s' "$output" | grep -oE '^[0-9]+ symbol\(s\) captured' \
      | grep -oE '^[0-9]+' | head -1)
    printf '%s\n' "$output" >> "$LOG"
    printf '{"ts":"%s","day":"%s","result":{"dataset":"bars","venue":"%s","date":"%s","status":"%s","pass":"daily","symbols":%s,"bars":%s}}\n' \
      "$started" "$yesterday" "$venue" "$yesterday" "$status" "${symbols:-null}" "${bars:-null}" >> "$RUNS"
  done

  # The intraday pass. Today's CLOSED hours, core symbols only. Each run is a
  # new snapshot because the set of still-live hours it skipped is part of the
  # id, so this appends the hour that just closed instead of being refused as a
  # duplicate - and the newly built bars carry a later availability time, which
  # is what the clock-gated reader's correction resolution keys on.
  #
  # already-built is the expected result when no hour has closed since the last
  # pass, and it is recorded under its own name rather than as a failure, for
  # the same reason the daily pass does it: a genuine failure must not arrive in
  # a stream of expected ones.
  today=$(date -u +%F)
  for venue in $VENUES; do
    core=$(core_symbols_for "$venue")
    [ -z "$core" ] && continue
    output=$(build_bars "$venue" "$today" "$core")
    code=$?
    case $code in
      0) status=built
         printf '%s' "$output" | grep -q 'nothing to build' && status=no-capture ;;
      "$EXIT_ALREADY_BUILT") status=already-built ;;
      *) status=failed ;;
    esac
    bars=$(printf '%s' "$output" | grep -oE '> [0-9]+ bars' | grep -oE '[0-9]+' | head -1)
    symbols=$(printf '%s' "$output" | grep -oE '^[0-9]+ symbol\(s\) captured' \
      | grep -oE '^[0-9]+' | head -1)
    printf '%s\n' "$output" >> "$LOG"
    printf '{"ts":"%s","day":"%s","result":{"dataset":"bars","venue":"%s","date":"%s","status":"%s","pass":"intraday","symbols":%s,"bars":%s}}\n' \
      "$started" "$today" "$venue" "$today" "$status" "${symbols:-null}" "${bars:-null}" >> "$RUNS"
  done

  sleep "$INTERVAL"
done
