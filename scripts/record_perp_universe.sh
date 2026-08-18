#!/usr/bin/env bash
# PB-01: publish the perp bot's tradable universe, so a probe has something
# measured to read instead of running a multi-minute store scan inside a board
# pass that already takes ten to nineteen minutes.
#
# Writes to $HOME/capture/perp/universe.json - outside the repo, because it is
# generated output that reports state and Rule 9 keeps that out of git.
#
# Safe to run at any time and safe to run twice: it reads the store through the
# clock gate and replaces one file atomically. It places no orders and writes
# nothing into the store.
#
# Usage: record_perp_universe.sh [as_of_ns]
set -euo pipefail

REPO="${REPO:-$HOME/trading-system}"
STORE="${STORE_ROOT:-$HOME/capture/store}"
OUT="${PERP_UNIVERSE_OUT:-$HOME/capture/perp/universe.json}"

cd "$REPO"
if [ "$#" -ge 1 ]; then
    PYTHONPATH="$REPO/src" .venv/bin/python -m perp.tradable_universe \
        --store-root "$STORE" --out "$OUT" --as-of-ns "$1"
else
    PYTHONPATH="$REPO/src" .venv/bin/python -m perp.tradable_universe \
        --store-root "$STORE" --out "$OUT"
fi
