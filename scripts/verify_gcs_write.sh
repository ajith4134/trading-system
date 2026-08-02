#!/usr/bin/env bash
# Proves the capture service can write, read back and delete an object in a
# GCS bucket, using whatever credential is active on this host.
#
# This is blocker B1's gate (spec section 10). The offloader must not be built
# against an unproven path, and the archive must not depend on one.
#
# Byte-exactness is checked, not assumed: the object is read back and compared
# to the source. A successful upload that silently corrupts is the failure this
# whole project is built to avoid.
#
# Usage: scripts/verify_gcs_write.sh gs://your-bucket-name
set -euo pipefail

BUCKET="${1:?usage: verify_gcs_write.sh gs://bucket-name}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL="$(mktemp)"
BACK="$(mktemp)"
REMOTE="${BUCKET}/_capture_write_probe_${STAMP}.txt"

cleanup() { rm -f "$LOCAL" "$BACK"; }
trap cleanup EXIT

echo "capture-write-probe ${STAMP}" > "$LOCAL"

echo "--- identity"
gcloud auth list --filter=status:ACTIVE --format='value(account)'

echo "--- 1/5 upload"
gcloud storage cp "$LOCAL" "$REMOTE"

echo "--- 2/5 read back"
gcloud storage cat "$REMOTE" > "$BACK"
cat "$BACK"

echo "--- 3/5 list"
gcloud storage ls "$BUCKET" | tail -3

echo "--- 4/5 byte-exact round trip"
if ! cmp -s "$LOCAL" "$BACK"; then
  echo "FAIL: object read back does not match what was written" >&2
  exit 1
fi
echo "identical"

echo "--- 5/5 delete"
gcloud storage rm "$REMOTE"

echo
echo "RESULT: GCS write access CONFIRMED for ${BUCKET}"
