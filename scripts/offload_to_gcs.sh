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

# Batched copies of exactly the completed files - not a regex exclude, and not
# a copy per file. Both of those were tried on 2026-08-08 and both were wrong:
#
#   Per-file `gcloud storage cp` moved 98 of 8,138 files before being killed.
#   The per-invocation overhead dominates completely.
#
#   `rsync --exclude` with one alternation per live stem builds a ~70KB regex
#   from ~1,750 stems, and gcloud silently over-matched it: files from a
#   *finished* hour were skipped because a *different* hour was open. The gap
#   was 38 files that direct copy uploaded without complaint.
#
# So: list the files that are genuinely complete, and hand them to cp in
# batches. `-n` means a re-run cannot replace a good object, which makes this
# safe to put on a timer without tracking state. Integrity is gcloud's own
# CRC32C check per object, so a corrupted transfer fails rather than lands.
#
# A file is complete when `RawWriter` is not holding it - marked by a sibling
# `.writing`. Deciding per file rather than per hour matters: markers were
# found from three different hours at once, including a stale one from a
# session that crashed six days earlier.
BATCH=400
failed=0
completed_list=$(mktemp)
trap 'rm -f "$completed_list"' EXIT

find "$RAW_ROOT" -type f \( -name '*.ndjson.zst' -o -name '*.idx.zst' \) |
  while IFS= read -r path; do
    stem="${path%.ndjson.zst}"; stem="${stem%.idx.zst}"
    [ -e "${stem}.writing" ] || printf '%s\n' "$path"
  done | sort > "$completed_list"

total=$(wc -l < "$completed_list")
live=$(find "$RAW_ROOT" -name '*.writing' | wc -l)
echo "offloading $total completed files, $live still being written" >&2

# `cp` takes many sources and one destination, and preserves nothing about the
# local tree - so each batch is copied per source directory to keep the layout.
while IFS= read -r dir; do
  find "$RAW_ROOT/$dir" -maxdepth 1 -type f \( -name '*.ndjson.zst' -o -name '*.idx.zst' \) \
    | while IFS= read -r f; do
        stem="${f%.ndjson.zst}"; stem="${stem%.idx.zst}"
        [ -e "${stem}.writing" ] || printf '%s\0' "$f"
      done \
    | xargs -0 -r -n "$BATCH" -- sh -c \
        'gcloud storage cp -n "$@" "'"${BUCKET}"'/raw/'"$dir"'/" >/dev/null 2>&1 || exit 1' _ \
    || { echo "FAILED: $dir" >&2; failed=$((failed + 1)); }
done < <(cd "$RAW_ROOT" && find . -mindepth 2 -maxdepth 2 -type d | sed 's|^\./||' | sort)

# The ledger and the universe record are small and are the only way to know
# later what the archive was missing and who was listed when. They belong with
# the data they describe.
#
# `store` is here and not only `raw` because funding has NO RAW COUNTERPART:
# it is polled directly into store/funding, so `raw/binance-funding` does not
# exist and a raw-only backup omits it entirely. That dataset's start date is
# the clock every promotion waits on, and `funding_reconstructed` cannot stand
# in - its availability times are the fetch, which is deliberately what makes
# it safe for research and useless for a backtest. Measured 2026-08-17: the
# bucket held raw/, ledger/ and universe/ and nothing else, so the observed
# record existed on exactly one disk.
#
# The exclude is the store's version of the `.writing` rule at the top of this
# file. Datasets are appended live and a part being written is named
# `.writing-part-<id>.parquet`; copying one lands a truncated parquet that
# reads as a complete dataset. Unlike the raw path's rejected `--exclude`
# experiment, this is one fixed pattern rather than ~1,750 alternations, so it
# cannot over-match a different partition.
# `.hourly-building` is excluded for the same reason and one level up: SL-15's
# migration builds a second copy of a dataset beside the live one, and a
# half-built copy in the bucket is a dataset that looks whole and is missing
# whatever had not been written when the pass ran. It is uploaded once it has
# been verified and renamed into place, which is the only state worth keeping.
for extra in ledger universe store; do
  [ -d "$CAPTURE_ROOT/$extra" ] || continue
  gcloud storage rsync -r -x '.*\.writing-part-.*|.*\.hourly-building/.*' \
      "$CAPTURE_ROOT/$extra" "${BUCKET}/${extra}" >/dev/null 2>&1 \
    || { echo "FAILED: $extra" >&2; failed=$((failed + 1)); }
done

remote=$(gcloud storage ls -r "${BUCKET}/raw/**" 2>/dev/null | grep -c 'zst$' || echo 0)
printf '{"completed_local":%d,"in_bucket":%d,"still_writing":%d,"failed":%d,"bucket":"%s"}\n' \
  "$total" "$remote" "$live" "$failed" "$BUCKET"

# Non-zero on any failure so a timer cannot report success while the archive
# quietly stops being backed up - the failure mode this exists to prevent.
[ "$failed" -eq 0 ]
