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
  "cd ${REPO} && nohup scripts/capture_supervisor.sh binance-funding BTCUSDT"
# Funding only, and captured rather than traded - no key exists for this venue.
# Poll-only, so no symbols are named: the response is the whole linear market.
start_as_user "capture_supervisor.sh bybit" \
  "cd ${REPO} && nohup scripts/capture_supervisor.sh bybit BTCUSDT"
# Last, and after the capture supervisors on purpose: it reports on what they
# produce, and starting it first would have it observe an empty archive and
# record "nothing captured" as this boot's first verdict.
start_as_user "health_supervisor.sh" \
  "cd ${REPO} && nohup scripts/health_supervisor.sh"

log "startup-script finished"
