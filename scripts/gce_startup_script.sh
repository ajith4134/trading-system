#!/bin/bash
# GCE startup-script — brings the long-running processes back after a reboot.
#
# INSTALL IT, do not just edit it. This file is a copy; the version that runs is
# the one in the instance's `startup-script` metadata:
#
#   gcloud compute instances add-metadata instance-20260801-081737 \
#     --zone=asia-south1-c --metadata-from-file startup-script=scripts/gce_startup_script.sh
#
# That works from the VM as of 2026-08-08 — the note here previously said it did
# not, citing 403s on `storage.buckets.create` and `services.list` from
# 2026-08-03. Those are different permissions; `compute.instances.setMetadata` is
# granted. The note was believed rather than retested, and in the meantime the
# metadata drifted three supervisors behind this file: on 2026-08-08 the live
# copy started boards, binance core capture and hyperliquid only — no store
# builds, no offload, no spot. A reboot would have come up with the archive
# silently un-backed-up.
#
# So: after editing, install it, then read it back from the metadata server and
# diff. The success message is not the verification.
#
#   curl -s -H "Metadata-Flavor: Google" \
#     http://metadata.google.internal/computeMetadata/v1/instance/attributes/startup-script
#
# WHY THIS AND NOT CRON OR LINGER, both of which were tried and failed here:
#   - `loginctl enable-linger`  -> "Access denied", and there is no sudo on this box
#   - user `@reboot` crontab    -> cron is NOT INSTALLED (no crontab binary, no
#                                  daemon) and installing it needs apt-get, which
#                                  needs sudo
#   - GCE startup-script        -> runs AS ROOT on every boot via google-guest-agent,
#                                  which is already enabled. No install, no linger.
#
# WHAT IT DOES NOT FIX: the boards' public hostname. A `trycloudflare.com` quick
# tunnel draws a new random hostname every time the tunnel process starts, so a
# reboot changes the URL. The current one is always in
# $HOME/capture/boards/current-url.txt. A fixed hostname needs a Cloudflare
# account and a domain, or a different host entirely.
set -uo pipefail

RUN_AS="anushadudekula71"
HOME_DIR="/home/${RUN_AS}"
REPO="${HOME_DIR}/trading-system"
BOOT_LOG="${HOME_DIR}/capture/boot.log"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) startup-script: $*" >> "$BOOT_LOG"; }

install -d -o "$RUN_AS" -g "$RUN_AS" "${HOME_DIR}/capture"
touch "$BOOT_LOG" && chown "${RUN_AS}:${RUN_AS}" "$BOOT_LOG"
log "boot"

# Drop to the owning user before touching that user's files. Running the
# supervisors as root leaves root-owned logs and state that the user's own
# processes then cannot write, which fails later and looks unrelated.
start_as_user() {
  local what=$1 cmd=$2
  if pgrep -u "$RUN_AS" -f "$what" > /dev/null 2>&1; then
    log "$what already running"
    return
  fi
  log "starting $what"
  setsid sudo -u "$RUN_AS" -H bash -lc "$cmd" >> "$BOOT_LOG" 2>&1 &
}

# The boards: an authenticated local server plus the tunnel in front of it.
start_as_user "boards_supervisor.sh" \
  "cd ${REPO} && nohup scripts/boards_supervisor.sh"

# Market data capture. Enabled 2026-08-03: the instance rebooted with nothing
# configured to restart it and the archive lost roughly seventeen hours, which is
# a gap no later analysis can fill — exchanges do not sell back the tape you
# failed to record. Disk runway measured 909 days at the observed write rate, so
# the cost of leaving it running is bounded and the cost of not is not.
start_as_user "capture_supervisor.sh binance" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh binance BTCUSDT,ETHUSDT,SOLUSDT ALL"
start_as_user "store_supervisor.sh" \
  "cd ${REPO} && nohup scripts/store_supervisor.sh"
# Retention for the local raw cache lives in a file, not here, so that changing
# it does not require re-pasting this script into instance metadata. This only
# creates the file if it is absent - an operator who has since changed the value,
# or switched eviction off with a 0, keeps their setting across reboots.
[ -e "${HOME_DIR}/capture/eviction-keep-days" ] || {
  echo 7 > "${HOME_DIR}/capture/eviction-keep-days"
  chown "${RUN_AS}:${RUN_AS}" "${HOME_DIR}/capture/eviction-keep-days"
  log "seeded eviction-keep-days=7"
}
start_as_user "offload_supervisor.sh" \
  "cd ${REPO} && nohup scripts/offload_supervisor.sh gs://capture-raw-data4134"
start_as_user "capture_supervisor.sh binance-spot" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh binance-spot BTCUSDT,ETHUSDT,SOLUSDT ALL"
start_as_user "capture_supervisor.sh hyperliquid" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh hyperliquid BTC,ETH,SOL"
# Binance funding, in its own process. Split from the recorder 2026-08-09: 857
# funding writers rotating inside one synchronous tick killed its websocket at
# every hour boundary, while binance-spot with a comparable trade load and no
# fan-out crossed four boundaries alive. Same venue name, so files land in
# raw/binance/ and the per-IP rate budget stays one bucket.
start_as_user "capture_supervisor.sh binance-funding" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh binance-funding BTCUSDT ALL"
# Funding only, and captured rather than traded - no key exists for this venue.
# Poll-only, so no symbols are named: the response is the whole linear market.
start_as_user "capture_supervisor.sh bybit" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh bybit BTCUSDT"
# Liquidations, market-wide - the feed Binance withholds from this host
# (DM-020). Its own process so a websocket cut cannot stall the funding poll.
start_as_user "capture_supervisor.sh bybit-liq" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh bybit-liq BTCUSDT ALL"
# Coinbase spot, keyless. The build plan carried this venue as blocked on a
# missing API key; probed 2026-08-10, only the private API needs one and the
# market data never did. Core three by product id (BTC-USD, not BTCUSDT - the
# venue's own name for the market), broad tail on trades alone.
start_as_user "capture_supervisor.sh coinbase" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh coinbase BTC-USD,ETH-USD,SOL-USD ALL"
# OPTIONS - the third segment, captured from 2026-08-16. Until this line existed
# the segment had ZERO data while the user's ruling names it as one of the three
# the bot trades. Polled, keyless, its own process: one response fans out into
# 1,502 files and there is no websocket here for a write storm to take down.
#
# With the capture supervisors rather than the builders, and that ordering is the
# whole argument for doing it now: a Deribit option chain cannot be backfilled at
# all - the endpoint ignores a `timestamp` parameter and returns the current
# chain - so an hour not captured is an hour that can never be reconstructed.
#
# The symbol argument is ignored by this venue, since one request returns the
# whole BTC or ETH chain; BTC is passed only because the supervisor requires one.
start_as_user "capture_supervisor.sh deribit" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh deribit BTC"
# Bars, in their own process since 2026-08-10. They were inside store_supervisor
# and a universe-wide build takes hours, so the polled datasets beside them
# refreshed once per full pass instead of once an hour - and every feature in
# the system was quietly reading three-hour-old funding as a current number.
start_as_user "bars_supervisor.sh" \
  "cd ${REPO} && nohup scripts/bars_supervisor.sh"
# Bars for the hour still being written, since 2026-08-16. Separate from
# bars_supervisor.sh for the same reason bars are separate from store: its
# intraday pass sits behind a daily universe-wide build, so it refreshes once per
# full pass. Measured before this existed: the newest bar the clock-gated reader
# would serve was 69 minutes old, against a bot the user has ruled is intraday on
# every segment. After: 1.8 minutes. Two tiers inside it - core at 60s, the whole
# captured universe at 300s - because a universe pass costs 95s at the end of an
# hour and would not fit in a 60s loop.
start_as_user "live_bars_supervisor.sh" \
  "cd ${REPO} && nohup scripts/live_bars_supervisor.sh"
# Last, and after the capture supervisors on purpose: it reports on what they
# produce, and starting it first would have it observe an empty archive and
# record "nothing captured" as this boot's first verdict.
start_as_user "health_supervisor.sh" \
  "cd ${REPO} && nohup scripts/health_supervisor.sh"
# The forward paper engine. Last, and after bars on purpose: it reads bars
# through the clock-gated reader and starting it first would prime an empty
# archive. The strategy and the participation rate are BOTH named here and both
# have no default in the engine - the only signal that exists makes no edge
# claim, and an invented participation rate is the cheapest way to manufacture
# edge. When Phase C produces a model, this line is where it is named.
# RETIRED 2026-08-18 under RL-025. `plumbing-momentum` made no edge claim and existed
# to prove the paper path was reachable at all. The four segment bots below now do
# that on live prices, so keeping it would put a second, no-edge-claim P&L series on
# the same board as the real bots - and six weeks from now that series reads as a
# result. Its journal stays on disk at ~/capture/paper/forward, marked as plumbing.
#
# start_as_user "paper_supervisor.sh" \
#   "cd ${REPO} && nohup scripts/paper_supervisor.sh plumbing-momentum 0.1 60"

# BF-06 / PB-08 / RL-020: the four segment bots, 24/7, one supervisor each. They take
# their prices from the venue feeds (RL-024) rather than the store, so they do NOT
# have to start after the store builders and are deliberately independent of them.
for SEGMENT in perp spot dated options; do
  start_as_user "segment_bot_supervisor.sh ${SEGMENT}" \
    "cd ${REPO} && nohup scripts/segment_bot_supervisor.sh ${SEGMENT} 6"
done

# LB-08 / RL-026: refit the brains on a cadence with nobody invoking anything, and
# run the §1a axis probes against whatever it registered. Four hours rather than
# one: a refit reads 30 hours of the store, and the store read is what drove memory
# to 25 GB of 29 on 2026-08-18. The bots trading on live prices come first.
start_as_user "retrain_supervisor.sh" \
  "cd ${REPO} && nohup scripts/retrain_supervisor.sh 14400 30"

log "startup-script finished"
