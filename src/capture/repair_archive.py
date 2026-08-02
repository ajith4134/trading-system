"""Repair every damaged hour in the archive before a recorder starts.

`RawWriter` refuses to append to an hour whose raw/index pair is damaged, and
`reconcile_pair` is the documented way out - but nothing invokes it, so the
refusal has no exit and the affected stream stays dark until the hour rotates.
Measured on this box: one ungraceful stop tore the zstd frames of the two
busiest depth streams and cost them the remaining seventeen minutes of the hour,
while every other stream kept recording.

That is the right failure shape and the wrong recovery time. A restart is
exactly when repair should run: no writer holds the files, and the process about
to start is the one that needs them appendable.

Hours still held by a live writer are skipped, not waited for. `reconcile_pair`
swaps the index inode, and doing that under a running writer loses every index
entry it writes afterwards - so a busy hour is left alone and picked up on a
later pass.
"""
from __future__ import annotations

from pathlib import Path

from capture.raw_writer import (
    IDX_SUFFIX, RAW_SUFFIX, RawCaptureError, read_pair, reconcile_pair,
    is_hour_being_written,
)


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Every (raw, index) pair in the archive, oldest hour first.

    Quarantine copies are excluded: their suffix is appended to the full name
    precisely so they never match this glob.
    """
    raw_root = Path(root) / "raw"
    if not raw_root.is_dir():
        return []
    pairs = []
    for raw_path in sorted(raw_root.rglob(f"*{RAW_SUFFIX}")):
        idx_path = raw_path.with_name(raw_path.name[: -len(RAW_SUFFIX)] + IDX_SUFFIX)
        pairs.append((raw_path, idx_path))
    return pairs


def is_pair_readable(raw_path: Path, idx_path: Path) -> bool:
    """Whether the pair can be read end to end without error.

    `read_pair` is the reader every consumer uses, so its verdict is the one
    that matters. A missing index counts as damaged - `reconcile_pair` rebuilds
    it from the raw lines.
    """
    if not idx_path.exists():
        return False
    try:
        read_pair(raw_path, idx_path)
    except RawCaptureError:
        return False
    return True


def repair_archive(root: Path) -> dict:
    """Reconcile every damaged, unheld pair. Returns what was done.

    Errors are collected rather than raised: one unrepairable hour must not stop
    the pass, because the hours after it are the ones the recorder is about to
    need.
    """
    repaired, skipped_live, failed = [], [], []
    healthy = 0

    for raw_path, idx_path in find_pairs(Path(root)):
        if is_pair_readable(raw_path, idx_path):
            healthy += 1
            continue
        is_live, _, pid = is_hour_being_written(raw_path)
        if is_live:
            skipped_live.append((str(raw_path), pid))
            continue
        try:
            outcome = reconcile_pair(raw_path, idx_path)
        except Exception as exc:
            failed.append((str(raw_path), f"{type(exc).__name__}: {exc}"))
            continue
        repaired.append({
            "path": str(raw_path),
            "entries_rebuilt": outcome.entries_rebuilt,
            "entries_discarded": outcome.entries_discarded,
            "raw_frames_kept": outcome.raw_frames_kept,
            "raw_was_salvaged": outcome.raw_was_salvaged,
        })

    return {
        "healthy": healthy,
        "repaired": repaired,
        "skipped_live": skipped_live,
        "failed": failed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Repair damaged raw/index pairs so a recorder can resume into them.")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    args = parser.parse_args(argv)

    result = repair_archive(Path(args.root))
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    # A pair that could not be repaired is a real problem and the exit code says
    # so, but the recorder should still start - partial capture beats none.
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
