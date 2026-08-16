#!/usr/bin/env bash
# Bars for the hour STILL BEING WRITTEN, so an intraday decision has intraday data.
#
# Added 2026-08-16 after the user ruled: "our entire tradin crypto bot is intraday
# on all sements spot,futures,options". Measured that day, before this existed:
# the newest bar the clock-gated reader would serve was **69 minutes old**. A
# strategy holding for minutes cannot be built on that, and nothing in the system
# said so - a stale bar reads exactly like a fresh one, which is the same failure
# `bars_supervisor.sh` was split for.
#
# ## Why this is not another pass inside bars_supervisor.sh
#
# For the reason that file's own header gives about the loop it was split out of:
# one slow job inside a loop starves everything else in it. Its intraday pass sits
# behind a daily `--symbols ALL` build across four venues, so it refreshes once per
# FULL pass, not once an hour. Coupling a 60-second job to that is how the blotter
# ended up 3.5 hours old and how the polled builds ended up 3 hours stale.
#
# ## Two tiers, and the numbers behind them
#
# `store.live_bars` re-reads the hour's capture files each pass, so a pass costs
# more the later in the hour it runs. Measured 2026-08-16 at minute 55, the worst
# point:
#
#   CORE  12 symbols across 4 venues     ~5s   -> 60s interval
#   ALL   2,303 symbols across 4 venues  95s   -> 300s interval
#
# 95 seconds does not fit in a 60-second loop, so a single universe-wide tier at
# 60s would fall behind and silently stop being intraday - the exact failure this
# exists to fix. The split is the same one the paper execution design already
# makes: broad discovery across the universe, finalists on BTC/ETH/SOL. The core
# tier is what forward trading eats; the universe tier is the watch list the user
# asked for - "keep an eye for all te univer symboles in all tree sements".
#
# ## What this does NOT fix
#
# OPTIONS. The ruling names three segments and this covers two: the archive holds
# no options venue at all, so there is nothing here to build fresh. See §3a of
# `docs/superpowers/specs/2026-08-08-final-project-goal-design.md`. Making options
# bars 60 seconds fresh is not a scheduling problem, it is a capture problem, and
# a supervisor that appeared to cover three segments while covering two would be
# worse than one that says this.
#
# Hyperliquid also captures only 3 symbols against binance's 570 and coinbase's
# 434. That is a capture-side gap this loop cannot close and must not disguise:
# `ALL` reads the archive, so the tier reports 3 rather than pretending to a
# universe that was never subscribed.
#
# Usage: live_bars_supervisor.sh [core-interval-seconds] [universe-interval-seconds]
set -uo pipefail

CORE_INTERVAL="${1:-60}"
UNIVERSE_INTERVAL="${2:-300}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE_ROOT=${CAPTURE_ROOT:-$HOME/capture}
STORE_ROOT="$CAPTURE_ROOT/store"
STATE_DIR="$CAPTURE_ROOT/store-builds"
CORE_LOG="$STATE_DIR/live-bars-core.log"
UNIVERSE_LOG="$STATE_DIR/live-bars-universe.log"
RESTART_LOG="$STATE_DIR/live-bars-restarts.ndjson"

MIN_DELAY=1
MAX_DELAY=60
HEALTHY_RUN_SECONDS=120

mkdir -p "$STATE_DIR"

core_pid=""
universe_pid=""

# Forward the stop signal rather than dying and orphaning the children. An
# orphaned tier keeps writing to the store with nothing supervising it, and the
# next start would run a second copy of the same loop against the same hour.
stop_children() {
  for pid in "$core_pid" "$universe_pid"; do
    [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  exit 0
}
trap stop_children TERM INT

record_restart() {
  printf '{"ts":"%s","tier":"%s","exit_code":%s,"ran_seconds":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$RESTART_LOG"
}

next_delay() {
  local delay=$1
  local doubled=$(( delay * 2 ))
  [ "$doubled" -gt "$MAX_DELAY" ] && doubled=$MAX_DELAY
  echo $(( doubled + (RANDOM % (delay > 0 ? delay : 1)) ))
}

# The venues that have a trade tape, and their core symbols. Per venue because
# the venues name the same instrument differently - BTCUSDT on binance, BTC on
# hyperliquid, BTC-USD on coinbase. A single list would build three venues and
# miss the fourth, which is how 99.6% of the tape went unbuilt in August.
CORE_VENUES=(
  --venue binance=BTCUSDT,ETHUSDT,SOLUSDT
  --venue binance-spot=BTCUSDT,ETHUSDT,SOLUSDT
  --venue hyperliquid=BTC,ETH,SOL
  --venue coinbase=BTC-USD,ETH-USD,SOL-USD
)
# ALL, not a list, and read off the archive rather than written here for the same
# reason `store.cli --symbols ALL` exists: only the archive knows what capture
# actually subscribed.
UNIVERSE_VENUES=(
  --venue binance=ALL
  --venue binance-spot=ALL
  --venue hyperliquid=ALL
  --venue coinbase=ALL
)

# The loop lives in Python (`--interval-seconds`), so the high-water mark of
# minutes already written survives across passes. Restarting the process is
# therefore not free - it re-reads the hour and re-offers rows the store already
# holds, which `store.live_bars` counts as already-written rather than raising -
# so this supervisor restarts a tier only when it actually dies.
supervise() {
  local tier=$1 delay=$MIN_DELAY
  while true; do
    local started ended ran code
    started=$(date +%s)

    if [ "$tier" = "core" ]; then
      PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m store.live_bars \
        --capture-root "$CAPTURE_ROOT" --store-root "$STORE_ROOT" \
        "${CORE_VENUES[@]}" --interval-seconds "$CORE_INTERVAL" \
        >> "$CORE_LOG" 2>&1 &
      core_pid=$!
      wait "$core_pid"
      code=$?
      core_pid=""
    else
      PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -m store.live_bars \
        --capture-root "$CAPTURE_ROOT" --store-root "$STORE_ROOT" \
        "${UNIVERSE_VENUES[@]}" --interval-seconds "$UNIVERSE_INTERVAL" \
        >> "$UNIVERSE_LOG" 2>&1 &
      universe_pid=$!
      wait "$universe_pid"
      code=$?
      universe_pid=""
    fi

    ended=$(date +%s)
    ran=$(( ended - started ))
    record_restart "$tier" "$code" "$ran"
    echo "live-bars $tier exited ($code) after ${ran}s; restarting in ${delay}s" >&2

    # A tier that ran healthily for hours and then dropped once should not
    # inherit the accumulated delay of every restart before it.
    if [ "$ran" -ge "$HEALTHY_RUN_SECONDS" ]; then
      delay=$MIN_DELAY
    else
      delay=$(next_delay "$delay")
    fi
    sleep "$delay"
  done
}

supervise core &
supervise universe &
wait
