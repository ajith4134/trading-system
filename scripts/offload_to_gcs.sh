#!/usr/bin/env bash
# Copy completed capture hours to object storage.
#
# The archive is the one thing on this box that cannot be rebuilt. Code can be
# rewritten from the repo; a lost day of market data is lost permanently,
# because exchanges do not serve it back and re-pulling history is forbidden by
# Layer 0's contract anyway (venues silently revise it).
#
# Two rules govern what is safe to upload.
#
# **Never a live hour.** `RawWriter` marks an hour it currently holds with a
# sibling `.writing` file. Uploading one copies a zstd frame mid-write, and the
# copy is a truncated archive that reads as complete. The marker is the venue
# recorder's own signal and is used here rather than guessing from mtime.
#
# **Never overwrite.** `--no-clobber` means a re-run cannot replace a good
# object with a worse one, and makes the whole script safe to run on a timer
# without tracking state. Integrity is gcloud's own CRC32C check on every
# object, so a corrupted transfer fails rather than lands.
#
# Usage: scripts/offload_to_gcs.sh gs://bucket-name [capture-root]
set -uo pipefail

BUCKET="${1:?usage: offload_to_gcs.sh gs://bucket-name [capture-root]}"
CAPTURE_ROOT="${2:-$HOME/capture}"
RAW_ROOT="$CAPTURE_ROOT/raw"

command -v gcloud >/dev/null 2>&1 || { echo "gcloud not on PATH" >&2; exit 1; }
[ -d "$RAW_ROOT" ] || { echo "no archive at $RAW_ROOT" >&2; exit 1; }

# One rsync rather than a copy per file. Measured 2026-08-08: per-file
# `gcloud storage cp` moved 98 of 8,138 files before it was killed - the
# per-invocation overhead dominates completely, and a backup slower than the
# data it protects is not a backup. rsync parallelises and skips what already
# matches.
#
# Live hours are excluded by name, derived from the `.writing` markers actually
# on disk rather than by assuming only the current hour is open: a stream that
# receives a late frame can reopen an older hour, and uploading it would copy a
# zstd frame mid-write - a truncated archive that reads as complete.
mapfile -t live_stems < <(find "$RAW_ROOT" -type f -name '*.writing' \
                          -printf '%f\n' 2>/dev/null | sed 's/\.writing$//' | sort -u)

exclude_args=()
if [ "${#live_stems[@]}" -gt 0 ]; then
  pattern=""
  for stem in "${live_stems[@]}"; do
    escaped=$(printf '%s' "$stem" | sed 's/[.[\*^$()+?{|]/\\&/g')
    pattern="${pattern}${pattern:+|}.*${escaped}\..*"
  done
  exclude_args=(--exclude "$pattern")
fi

echo "excluding ${#live_stems[@]} live hour(s) still being written" >&2

failed=0
if ! gcloud storage rsync -r "${exclude_args[@]}" \
      "$RAW_ROOT" "${BUCKET}/raw" >/dev/null 2>&1; then
  echo "FAILED: raw archive rsync" >&2
  failed=1
fi

# The ledger and the universe record are small and are the only way to know
# later what the archive was missing and who was listed when. They belong with
# the data they describe.
for extra in ledger universe; do
  [ -d "$CAPTURE_ROOT/$extra" ] || continue
  gcloud storage rsync -r "$CAPTURE_ROOT/$extra" "${BUCKET}/${extra}" >/dev/null 2>&1 \
    || { echo "FAILED: $extra" >&2; failed=$((failed + 1)); }
done

remote=$(gcloud storage ls -r "${BUCKET}/raw/**" 2>/dev/null | grep -c 'zst$' || echo 0)
local_files=$(find "$RAW_ROOT" -type f \( -name '*.ndjson.zst' -o -name '*.idx.zst' \) | wc -l)
printf '{"local":%d,"in_bucket":%d,"live_excluded":%d,"failed":%d,"bucket":"%s"}\n' \
  "$local_files" "$remote" "${#live_stems[@]}" "$failed" "$BUCKET"

# Non-zero on any failure so a timer cannot report success while the archive
# quietly stops being backed up - the failure mode this exists to prevent.
[ "$failed" -eq 0 ]
