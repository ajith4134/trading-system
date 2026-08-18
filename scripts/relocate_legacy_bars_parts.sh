#!/usr/bin/env bash
# SL-15 straggler repair, step 1 of 2: move the legacy symbol-partitioned parts
# OUT of the live dataset and into the retired copy, so that step 2 can fold
# them back in as hour-partitioned rows.
#
# WHY MOVE RATHER THAN FOLD IN PLACE. A fold writes hour-partitioned copies and
# never deletes the originals - correctly, because deleting the only copy of a
# row on the strength of a migration that just ran is how archives are lost. But
# while both exist in the SAME dataset, every one of those rows is read TWICE:
# once as a legacy part with a NULL hour, once as an hour part. Doubled volume
# reads as real volume, and nothing in the store would flag it.
#
# So each part is moved, never copied and never deleted. A part exists in
# exactly one dataset at every instant, and the retired copy is the one place
# the fold reads from.
#
# NOTHING IS DELETED. The only removals are `rmdir` of directories that are
# already empty, which fails rather than forces if a part is still there.
#
# A target that already exists is SKIPPED, not overwritten: part names are
# derived from the content that produced them, so a name collision means the
# same rows, and overwriting would be a write with no possible benefit.
#
# Refuses to run while a writer is still landing parts in the old layout -
# moving a file being written is how a truncated part enters an archive.
#
# Usage:
#   relocate_legacy_bars_parts.sh              # dry run, prints what it would do
#   relocate_legacy_bars_parts.sh --apply      # actually move
set -euo pipefail

REPO="${REPO:-$HOME/trading-system}"
STORE="${STORE_ROOT:-$HOME/capture/store}"
# The dataset is the first NON-flag argument. Taking it as "$1" outright meant
# `--apply` became the dataset name, and the run refused with "no live dataset
# at .../--apply" - the right failure, but only because the refusal exists.
LIVE="bars_60000000000ns"
APPLY="no"
for argument in "$@"; do
    case "$argument" in
        --apply) APPLY="yes" ;;
        --*)     echo "unknown flag: $argument" >&2; exit 2 ;;
        *)       LIVE="$argument" ;;
    esac
done

cd "$REPO"
PYTHONPATH="$REPO/src" APPLY="$APPLY" .venv/bin/python - "$STORE" "$LIVE" <<'PY'
import os
import sys
import time
from pathlib import Path

store, live_name = Path(sys.argv[1]), sys.argv[2]
apply_changes = os.environ.get("APPLY") == "yes"
live = store / live_name
retired = store / f"{live_name}.legacy-symbol-layout"

if not live.is_dir():
    raise SystemExit(f"refused: no live dataset at {live}")
if not retired.is_dir():
    raise SystemExit(f"refused: no retired copy at {retired}. The fold reads "
                     f"from it, so relocating into nothing would strand the rows")

folders = sorted(p for p in live.iterdir()
                 if p.is_dir() and p.name.startswith("symbol="))
parts = [part for folder in folders for part in sorted(folder.glob("*.parquet"))]
if not parts:
    print(f"nothing to relocate: {live} holds no legacy symbol partition")
    raise SystemExit(0)

# A writer still landing parts here means the restart onto the hour layout has
# not happened, and moving a file mid-write is how a truncated part enters an
# archive. Sixty seconds is the bars cadence; anything newer is a live writer.
newest = max(part.stat().st_mtime for part in parts)
age_s = time.time() - newest
if age_s < 120:
    raise SystemExit(
        f"refused: the newest legacy part is {age_s:.0f}s old, so something is "
        f"still writing the old layout. Restart the writers onto the hour "
        f"layout first - until they are, this repair only opens a new window")

moved = skipped = 0
for part in parts:
    target = retired / part.parent.name / part.name
    if target.exists():
        skipped += 1
        continue
    if apply_changes:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same filesystem, so this is a rename: atomic, and it cannot leave the
        # part half-present in both places the way a copy-then-delete can.
        part.replace(target)
    moved += 1

emptied = 0
if apply_changes:
    for folder in folders:
        try:
            folder.rmdir()          # refuses if anything is still inside
            emptied += 1
        except OSError:
            pass

verb = "moved" if apply_changes else "would move"
print(f"{len(folders)} legacy symbol folder(s) holding {len(parts)} part(s)")
print(f"{verb} {moved}; skipped {skipped} already present in the retired copy")
if apply_changes:
    print(f"removed {emptied} emptied folder(s); "
          f"{len(folders) - emptied} still hold something and were left alone")
    remaining = sum(1 for p in live.iterdir()
                    if p.is_dir() and p.name.startswith("symbol="))
    print(f"top-level symbol= directories left in {live_name}: {remaining}")
    print("hour pruning turns itself back on when that reaches 0"
          if remaining else "hour pruning is now eligible to turn itself on")
else:
    print("dry run - nothing was moved. Re-run with --apply")
PY
