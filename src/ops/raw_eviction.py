"""Drop the local copy of raw hours already proven to be in the bucket.

Nothing here deletes data. `offload_to_gcs.sh` mirrors every closed hour to object
storage, so the local tree is a **cache** of an archive that lives elsewhere, and
this evicts from that cache. The distinction is the whole safety argument: a bug
here costs a re-download, not a day of market data that no exchange will serve
back.

The design rule that follows from it: **every question this module cannot answer
resolves to keep.** An unparseable date, a missing bucket entry, a file whose name
does not look like an archive file - all kept. There is no path where uncertainty
deletes.
"""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

# The two suffixes `RawWriter` produces and `offload_to_gcs.sh` uploads. Anything
# else under raw/ is left alone: a file this module does not recognise is a file
# it has no business removing.
ARCHIVE_SUFFIXES = (".ndjson.zst", ".idx.zst")


@dataclass(frozen=True)
class EvictionPlan:
    """What would be removed, what would not, and why not.

    `kept` carries the reason per file rather than a count. An eviction that
    reports only totals cannot be audited, and this is the one operation on this
    box whose mistakes are not reversible.
    """

    evictable: tuple[Path, ...]
    kept: tuple[tuple[Path, str], ...]
    bytes_reclaimable: int


def plan_raw_eviction(raw_root: Path, *, today: dt.date, keep_days: int,
                      bucket_objects: set[str]) -> EvictionPlan:
    """Decide, without touching anything.

    `bucket_objects` is passed in rather than fetched so the decision is a pure
    function of (tree, clock, inventory) - which is what makes it testable at all,
    and what keeps a network failure from turning into a deletion.
    """
    raw_root = Path(raw_root)
    cutoff = today - dt.timedelta(days=keep_days)

    evictable: list[Path] = []
    kept: list[tuple[Path, str]] = []
    reclaimable = 0

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        if not path.name.endswith(ARCHIVE_SUFFIXES):
            continue

        day = _day_of(path, raw_root)
        if day is None:
            kept.append((path, "unrecognised path"))
            continue
        if day > cutoff:
            kept.append((path, "within retention"))
            continue
        if _writing_marker_for(path).exists():
            # Old does not imply closed. The bucket's copy of a half-written hour
            # is a truncated file that reads as complete, so it is not a survivor
            # this may rely on.
            kept.append((path, "still being written"))
            continue
        if f"raw/{path.relative_to(raw_root).as_posix()}" not in bucket_objects:
            # Age is not evidence. A file that never uploaded is the only copy
            # there is, and no re-download undoes deleting it.
            kept.append((path, "not in bucket"))
            continue

        evictable.append(path)
        reclaimable += path.stat().st_size

    return EvictionPlan(tuple(evictable), tuple(kept), reclaimable)


def _day_of(path: Path, raw_root: Path) -> dt.date | None:
    """The UTC day from the `<venue>/<date>/` folder, or None if it is not one."""
    try:
        relative = path.relative_to(raw_root)
    except ValueError:
        return None
    if len(relative.parts) != 3:
        return None
    try:
        return dt.date.fromisoformat(relative.parts[1])
    except ValueError:
        return None


def _writing_marker_for(path: Path) -> Path:
    """The sibling marker `RawWriter` holds while an hour is open.

    Derived the same way `offload_to_gcs.sh` derives it - strip the archive
    suffix, add `.writing` - because a marker this module looked for under a
    different name is a guard that silently never fires.
    """
    stem = str(path)
    for suffix in ARCHIVE_SUFFIXES:
        stem = stem.removesuffix(suffix)
    return Path(stem + ".writing")


def evict_planned_files(plan: EvictionPlan) -> tuple[int, int]:
    """Remove the planned files. Returns (count removed, bytes freed).

    The marker is re-checked per file rather than trusted from the plan. Planning
    is a snapshot and the capture writers never stop: a restart between the two
    can reopen an hour the plan still names, and one stat per file removes that
    race entirely.

    A file that has already gone is not an error - the offload and the eviction
    both run on timers, and a missing file means the work is done.
    """
    removed = 0
    freed = 0
    for path in plan.evictable:
        if _writing_marker_for(path).exists():
            continue
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
        freed += size
    return removed, freed


def object_names_from_listing(lines, *, bucket: str) -> set[str]:
    """Bucket keys from `gcloud storage ls -r`, matched exactly or not at all.

    A key from another bucket is dropped rather than stripped. It cannot arrive
    through the normal call path, which is precisely why it would go unnoticed if
    it ever did - and the mistake it would cause is calling a file uploaded when
    it is not.

    Directory placeholders (trailing `/`) and blanks are discarded: they name no
    file and could never match one.
    """
    prefix = bucket.rstrip("/") + "/"
    keys = set()
    for line in lines:
        entry = line.strip()
        if not entry or entry.endswith("/") or not entry.startswith(prefix):
            continue
        keys.add(entry[len(prefix):])
    return keys


def _gcloud_listing(bucket: str) -> list[str]:
    """Every object under the bucket's raw/ prefix, one per line.

    One listing rather than a stat per file: the archive is tens of thousands of
    objects, and `offload_to_gcs.sh` already recorded what per-file gcloud calls
    cost - 98 files moved before it was killed.
    """
    import subprocess
    result = subprocess.run(
        ["gcloud", "storage", "ls", "-r", f"{bucket.rstrip('/')}/raw/**"],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"gcloud listing failed: {result.stderr.strip()[:300]}")
    return result.stdout.splitlines()


def main(argv: list[str] | None = None, *, list_objects=_gcloud_listing) -> int:
    """Report what is evictable; remove it only under `--apply`.

    Dry run is the default because the mistake is one-way: an operator running
    this to see what it would do must not learn the answer by having it done.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Drop local raw files already mirrored to object storage.")
    parser.add_argument("--bucket", required=True, help="gs://bucket-name")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--keep-days", type=int, default=7,
                        help="distinct UTC days kept locally, including today")
    parser.add_argument("--today", default=None,
                        help="UTC date to treat as today (YYYY-MM-DD); testing only")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it nothing is removed")
    args = parser.parse_args(argv)

    raw_root = Path(args.capture_root) / "raw"
    today = (dt.date.fromisoformat(args.today) if args.today
             else dt.datetime.now(dt.UTC).date())

    try:
        listing = list_objects(args.bucket)
    except Exception as exc:                                   # noqa: BLE001
        print(json.dumps({"applied": False, "refused": f"listing failed: {exc}"}))
        return 2

    objects = object_names_from_listing(listing, bucket=args.bucket)
    plan = plan_raw_eviction(raw_root, today=today, keep_days=args.keep_days,
                             bucket_objects=objects)

    reasons: dict[str, int] = {}
    for _, reason in plan.kept:
        reasons[reason] = reasons.get(reason, 0) + 1
    report = {"bucket": args.bucket, "keep_days": args.keep_days,
              "in_bucket": len(objects), "evictable": len(plan.evictable),
              "bytes_reclaimable": plan.bytes_reclaimable, "kept": reasons,
              "applied": False, "removed": 0, "freed_bytes": 0}

    if not objects:
        # Keeping everything is already the safe outcome; refusing loudly is what
        # stops a broken credential from reading as a healthy eviction run.
        report["refused"] = "bucket listing was empty - refusing to treat that as proof"
        print(json.dumps(report))
        return 3

    if args.apply:
        removed, freed = evict_planned_files(plan)
        report.update({"applied": True, "removed": removed, "freed_bytes": freed})

    print(json.dumps(report))
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
