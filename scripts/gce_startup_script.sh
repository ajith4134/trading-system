#!/bin/bash
# GCE startup-script — brings the long-running processes back after a reboot.
#
# PASTE THE CONTENTS OF THIS FILE into the instance's `startup-script` metadata
# value. It cannot be installed from the VM: the attached service account has no
# project IAM (verified 2026-08-03 — `storage.buckets.create` and
# `services.list` both return 403), so setting instance metadata needs either a
# console edit or `gcloud auth login` as a human.
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

# Market data capture. UNCOMMENT TO ENABLE — deliberately off by default because
# starting it writes to disk and opens venue connections continuously, which is a
# decision to take on purpose rather than inherit from a reboot.
#
# start_as_user "capture_supervisor.sh binance" \
#   "cd ${REPO} && nohup scripts/capture_supervisor.sh binance BTCUSDT,ETHUSDT,SOLUSDT"
# start_as_user "capture_supervisor.sh hyperliquid" \
#   "cd ${REPO} && nohup scripts/capture_supervisor.sh hyperliquid BTC,ETH,SOL"

log "startup-script finished"
