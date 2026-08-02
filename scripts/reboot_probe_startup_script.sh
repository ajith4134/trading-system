#!/bin/bash
# GCE startup-script — blocker B2 probe.
#
# PASTE THE CONTENTS OF THIS FILE into the instance's `startup-script` metadata
# value. Do not run it by hand; the point is to prove it runs unattended at boot.
#
# WHY THIS AND NOT CRON OR LINGER, both of which were tried and failed:
#   - `loginctl enable-linger`  -> "Access denied", and there is no sudo on this box
#   - user `@reboot` crontab    -> cron is NOT INSTALLED here (no crontab binary,
#                                  no daemon) and installing it needs apt-get, which
#                                  needs sudo
#   - GCE startup-script        -> runs AS ROOT on every boot via google-guest-agent,
#                                  which is already enabled. No install, no linger.
#
# WHAT IT PROVES: if the marker file below appears with a fresh timestamp after a
# reboot, then this host CAN start the recorder unattended, and B2 is solved by
# replacing the marker line with the recorder launch (see the commented block).
#
# The script runs as root, so it drops to the capture user before touching that
# user's files — otherwise the marker ends up root-owned and the recorder cannot
# write beside it later.

set -euo pipefail

CAPTURE_USER="anushadudekula71"
MARKER_DIR="/home/${CAPTURE_USER}/capture/health"
MARKER="${MARKER_DIR}/reboot_probe.log"

runuser -u "${CAPTURE_USER}" -- mkdir -p "${MARKER_DIR}"
runuser -u "${CAPTURE_USER}" -- \
  bash -c "printf '%s booted uid=%s\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"\$(id -u)\" >> '${MARKER}'"

# ---------------------------------------------------------------------------
# ONCE THE PROBE IS CONFIRMED, this is where the recorder starts. Left commented
# so the probe stays a probe — proving the boot hook fires is a separate question
# from whether the recorder survives, and conflating them makes a failure
# ambiguous.
#
# runuser -u "${CAPTURE_USER}" -- bash -lc '
#   export PATH="$HOME/.local/bin:$PATH"
#   cd "$HOME/trading-system/.claude/worktrees/layer0-raw-capture"
#   setsid nohup uv run --python 3.12 python -m capture.cli \
#       --venue binance --symbols BTCUSDT,ETHUSDT,SOLUSDT \
#       >> "$HOME/capture/health/recorder.out" 2>&1 &
# '
# ---------------------------------------------------------------------------
