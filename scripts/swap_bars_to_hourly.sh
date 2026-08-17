#!/usr/bin/env bash
# SL-15: rename the verified hour-partitioned copy of the bars dataset into
# place, and the symbol-partitioned one aside.
#
# Two renames. Nothing is rewritten and NOTHING IS DELETED - the old layout
# stays at <dataset>.legacy-symbol-layout until a human retires it, which the
# plan row makes a decision rather than a cleanup step.
#
# Refuses on an unverified report, because the report is the only evidence that
# the copy holds what the original holds. Verified twice on 2026-08-17:
# 93,586 legacy parts, 3,568,232 rows in and 3,568,232 out, no column lost, no
# mismatched (hour, symbol) group.
#
# Reversal, if the new layout ever misbehaves:
#   mv bars_60000000000ns bars_60000000000ns.hourly-building
#   mv bars_60000000000ns.legacy-symbol-layout bars_60000000000ns
set -euo pipefail

REPO="${REPO:-$HOME/trading-system}"
DATASET="${1:-bars_60000000000ns}"
STORE="${STORE_ROOT:-$HOME/capture/store}"

cd "$REPO"
PYTHONPATH="$REPO/src" .venv/bin/python - "$DATASET" "$STORE" <<'PY'
import json
import sys
from pathlib import Path

from store.hourly_migration import MigrationReport, swap_in_migrated_dataset

dataset, store = sys.argv[1], Path(sys.argv[2])
report = MigrationReport(
    **json.loads((store / f"migration-report-{dataset}.json").read_text()))
print(f"report: {report.legacy_rows} legacy row(s), {report.migrated_rows} migrated, "
      f"columns lost {report.columns_lost or '()'}, "
      f"{len(report.mismatched_groups)} mismatched group(s), verified={report.verified}")
retired, live = swap_in_migrated_dataset(store, dataset, report)
print(f"retired: {retired}")
print(f"live:    {live}")
PY

echo
echo "Stragglers - parts a writer landed in the old directory during the swap -"
echo "are folded in by re-running the migration against the retired copy, or by"
echo "the straggler check in the session that ran this."
