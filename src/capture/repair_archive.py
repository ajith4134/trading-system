"""Repair damaged hours so a recorder can resume into them.

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

## Why the pass is scoped

`is_pair_readable` decompresses the pair end to end. Run over the whole archive
that is not a check, it is a full re-read of every byte ever captured, and it
sits in front of the recorder.

Measured 2026-08-09, and the reason this scoping exists: after a reboot, three
venue supervisors each launched their own full-root pass over 22,668 pairs and
1.9 GB. Eighteen minutes later all three were still running at 44% CPU and **not
one frame of market data had been captured** since boot. The archive grows
280 MB/day, so the window in front of capture grows with it - and this ran on
every restart, of which 98 were recorded, not only at boot.

Recency alone does not fix it. Raw eviction keeps seven days, so a 48-hour
window is most of the archive: measured, 6,399 binance pairs and 364 seconds.
What actually bounds the blocking work is not age but **resumability**.

## The two scopes, and why they are disjoint

Only the **current hour** can be resumed. A rotated hour is never appended to
again, so a tear in it cannot refuse a recorder - it is an integrity problem,
which is worth repairing but is not worth blocking capture on.

- `Scope.RESUMABLE` - the current hour, this venue. What runs before the
  recorder starts. Measured at a busy moment: 15s for binance, and it grows with
  the symbol count rather than with the archive.
- `Scope.ARCHIVE` - everything strictly older than the current hour, this venue.
  Safe to run **alongside** a live recorder, which is the point: it is the same
  repair work, moved off the path that costs dark time.
- `Scope.ALL` - both, for a deliberate offline pass.

The disjointness is what makes `ARCHIVE` safe to background. `reconcile_pair`
swaps the index inode, so running it on an hour a writer is about to open would
silently drop every index entry written afterwards. `ARCHIVE` cannot reach the
current hour, so the recorder and the background pass never contend for one -
the `skipped_live` check is a second line of defence, not the only one.

**Markers are exempt from every bound and always swept.** A `.writing` marker
left by a dead process blocks the GCS offload from ever copying that hour, and
one survived six days undetected. Age is precisely what makes that case
dangerous, so it cannot be what excludes it from the sweep. The sweep walks
names and opens nothing, so it stays cheap over the whole archive.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from capture.raw_writer import (
    IDX_SUFFIX, RAW_SUFFIX, WRITING_MARKER_SUFFIX, RawCaptureError, read_pair,
    reconcile_pair, is_hour_being_written,
)


class Scope(str, Enum):
    """Which hours a pass reads end to end. Named for what they mean, not for
    how far back they reach - the whole point is that age is the wrong axis."""

    RESUMABLE = "resumable"   # the current hour: the only one a recorder can be refused by
    ARCHIVE = "archive"       # everything already rotated: repairable, but not blocking
    ALL = "all"               # both, as a deliberate offline pass


# `paths_for` builds the stem as f"{stream}_{symbol}_{hour}", so the hour is the
# trailing token. Anchored to the end rather than split on "_", because a stream
# or symbol containing an underscore would otherwise move which token is read.
_HOUR_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2})$")


def hour_of(raw_path: Path) -> datetime | None:
    """The UTC hour this pair records, read off its filename.

    None when the name does not carry one. A writer always builds its name
    through `paths_for`, which always stamps an hour, so a name without one is
    not a name any recorder will open - it belongs to `ARCHIVE`, where a slow
    surprise costs no dark time.
    """
    name = Path(raw_path).name
    stem = name[: -len(RAW_SUFFIX)] if name.endswith(RAW_SUFFIX) else name
    match = _HOUR_IN_NAME.search(stem)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def current_hour(now: datetime | None = None) -> datetime:
    """The hour a recorder starting now would open."""
    return (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)


def find_pairs(root: Path, venue: str | None = None) -> list[tuple[Path, Path]]:
    """Every (raw, index) pair in the archive, oldest hour first.

    Scoped to one venue's subtree when `venue` is given. Quarantine copies are
    excluded: their suffix is appended to the full name precisely so they never
    match this glob.
    """
    raw_root = Path(root) / "raw"
    base = raw_root if venue is None else raw_root / venue
    if not base.is_dir():
        return []
    pairs = []
    for raw_path in sorted(base.rglob(f"*{RAW_SUFFIX}")):
        idx_path = raw_path.with_name(raw_path.name[: -len(RAW_SUFFIX)] + IDX_SUFFIX)
        pairs.append((raw_path, idx_path))
    return pairs


def find_marked_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Every pair carrying a `.writing` marker, at any age and in any venue.

    Deliberately unscoped in both dimensions. A stale marker's whole danger is
    that it is old and silent - it keeps the GCS offload from ever copying that
    hour - so neither the venue bound nor the scope may hide it.
    """
    raw_root = Path(root) / "raw"
    if not raw_root.is_dir():
        return []
    pairs = []
    for marker in sorted(raw_root.rglob(f"*{WRITING_MARKER_SUFFIX}")):
        stem = marker.name[: -len(WRITING_MARKER_SUFFIX)]
        raw_path = marker.with_name(stem + RAW_SUFFIX)
        if raw_path.exists():
            pairs.append((raw_path, marker.with_name(stem + IDX_SUFFIX)))
    return pairs


def pairs_to_scan(root: Path, venue: str | None = None,
                  scope: Scope = Scope.RESUMABLE,
                  now: datetime | None = None) -> list[tuple[Path, Path]]:
    """The pairs a pass will read end to end, plus every marked one."""
    root = Path(root)
    in_scope = find_pairs(root, venue)
    if scope is not Scope.ALL:
        boundary = current_hour(now)
        wanted = (lambda h: h is not None and h >= boundary) if scope is Scope.RESUMABLE \
            else (lambda h: h is None or h < boundary)
        in_scope = [p for p in in_scope if wanted(hour_of(p[0]))]
    # Dict, not set: insertion order is the sorted order find_pairs produced, so
    # the oldest hour still runs first and the pass stays reproducible.
    return list(dict.fromkeys(in_scope + find_marked_pairs(root)))


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


def repair_archive(root: Path, venue: str | None = None,
                   scope: Scope = Scope.RESUMABLE,
                   now: datetime | None = None) -> dict:
    """Reconcile every damaged, unheld pair in scope. Returns what was done.

    Errors are collected rather than raised: one unrepairable hour must not stop
    the pass, because the hours after it are the ones the recorder is about to
    need.

    `scanned` is reported alongside the outcomes so the caller can see what the
    pass actually looked at. A scope bug reads as a quiet, healthy, empty result
    otherwise - which is the failure mode this whole change is about.
    """
    repaired, skipped_live, failed, cleared_markers = [], [], [], []
    healthy = 0
    in_scope = pairs_to_scan(root, venue, scope, now)

    for raw_path, idx_path in in_scope:
        # Asked for every pair, not only damaged ones. A healthy hour used to
        # `continue` before its marker was ever looked at, which is how a marker
        # from a process that died on 2026-08-02 was still on disk six days
        # later - and how that hour stayed out of every backup, silently,
        # because the offload correctly refuses to copy an hour a writer holds.
        is_live, marker, pid = is_hour_being_written(raw_path)
        if not is_live and marker.exists():
            # `is_hour_being_written` has already established the process is
            # gone. Removing the marker is what makes that judgement stick;
            # leaving it means the same dead pid blocks the hour forever.
            try:
                marker.unlink()
                cleared_markers.append(str(marker))
            except OSError as exc:
                failed.append((str(marker), f"{type(exc).__name__}: {exc}"))

        if is_pair_readable(raw_path, idx_path):
            healthy += 1
            continue
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
        "scanned": len(in_scope),
        "scope": {"venue": venue or "all", "hours": scope.value},
        "healthy": healthy,
        "repaired": repaired,
        "skipped_live": skipped_live,
        "cleared_markers": cleared_markers,
        "failed": failed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Repair damaged raw/index pairs so a recorder can resume into them.")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    parser.add_argument("--venue", default=None,
                        help="repair only this venue's subtree; omit for every venue")
    parser.add_argument("--scope", type=Scope, choices=list(Scope), default=Scope.RESUMABLE,
                        help="'resumable' (default) is the current hour - the only one that "
                             "can refuse a recorder, so the only one worth blocking on. "
                             "'archive' is everything already rotated and is safe to run "
                             "beside a live recorder. 'all' is a deliberate offline pass")
    args = parser.parse_args(argv)

    result = repair_archive(Path(args.root), venue=args.venue, scope=args.scope)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    # A pair that could not be repaired is a real problem and the exit code says
    # so, but the recorder should still start - partial capture beats none.
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
