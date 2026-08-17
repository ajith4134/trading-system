"""Rewrite a symbol-partitioned dataset into an hour-partitioned one, and prove it.

SL-15. The store was partitioned by `symbol` alone, so availability was nowhere
in the path and a poll asking "what arrived since my watermark" had to open every
file to find out. Measured 2026-08-17 on the live bars dataset: 52,487 fragments,
538 MB, and a scan filtered to match NO rows still cost 185.6s cold.

Putting the availability hour in the path lets pyarrow skip directories without
opening them. This module moves what is already on disk into that layout.

**The rewrite is the risk, not the layout.** A migration that drops a row, a
column or a symbol produces a store that reads fine and is wrong, and this store
has lost a column silently once already - `funding_interval_hours`, 2026-08-09,
which turns a 4-hourly funding rate into an 8-hourly one. So:

  * the legacy directory is never written to and never deleted, only renamed
    aside after the new one has been verified;
  * verification is on content - rows per (hour, symbol), total rows, and the
    column set - not on "the command exited 0";
  * the swap refuses on an unverified report rather than trusting its caller;
  * an interrupted run resumes, because a run over 52,487 files will be
    interrupted.

Compaction comes free with the rewrite: every legacy part for one (hour, symbol)
becomes one part, which is where the ~20x cut in file count comes from.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from store.parquet_partition import (
    HOUR_KEY, PartitionExistsError, append_partition, compute_snapshot_id,
    floor_to_hour, read_dataset,
)
from store.temporal_schema import AVAILABILITY_TIME, SYMBOL

BUILDING_SUFFIX = ".hourly-building"
LEGACY_SUFFIX = ".legacy-symbol-layout"
REPORT_PREFIX = "migration-report-"
# Leading dot on purpose: pyarrow's dataset discovery skips names beginning with
# "." or "_", so the manifest can live inside the dataset it describes without
# being discovered as a malformed parquet part and failing every read.
MANIFEST_FILE = ".migrated-parts.json"

# What a report is entitled to say. A migration builds a copy nothing else
# writes to, so it can claim equality. A fold's target is the LIVE dataset,
# which capture keeps appending to while the fold runs, so rows the source
# never held are expected rather than a fault - and the claim has to be weaker
# by exactly that much, in writing, or a fold report reads as a migration one.
COPY_CLAIM = "every legacy row is present in the new layout, and no other row is"
FOLD_CLAIM = ("every group of the retired layout is present in the live one with "
              "at least its row count; the live layout may hold newer rows besides")


class MigrationRefused(RuntimeError):
    """The migration will not run, or its result will not be swapped in."""


@dataclass(frozen=True)
class MigrationReport:
    """What the run did, in the terms the acceptance is written in.

    A dataclass rather than a log line because the row says "every row of the old
    layout is present in the new one before the old is retired", and that is a
    claim someone has to be able to re-read tomorrow.
    """

    dataset: str
    building_dataset: str
    legacy_parts: int
    migrated_parts: int
    legacy_rows: int
    migrated_rows: int
    symbols: int
    hours: int
    columns_lost: tuple[str, ...]
    mismatched_groups: tuple[str, ...]
    verified: bool
    # Defaulted so a report written before folds existed still loads. The
    # default is the stronger claim, which is the one those reports made.
    claim: str = COPY_CLAIM


def _legacy_symbol_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.startswith("symbol="))


def _hour_partitioned_parts(root: Path) -> list[Path]:
    """Parts already in the new layout, which a live dataset acquires on its own.

    The store supervisors reload their child on restart, so `append_partition`
    starts writing hour-partitioned parts as soon as the code lands - measured
    2026-08-17, the live bars dataset held 4,595 of them beside 2,231 legacy
    symbol directories within minutes, before any migration had been run. A
    migration that refused a dataset in that state would refuse every dataset
    that matters, and one that ignored those parts would leave their rows out of
    the store it swapped in.
    """
    return sorted(root.glob(f"{HOUR_KEY}=*/symbol=*/*.parquet"))


def _link_into_building(store_root: Path, dataset: str, building: str,
                        part: Path) -> None:
    """Put an already-hour-partitioned part into the building copy, unchanged.

    Hard link rather than copy: the parts are immutable by the store's central
    promise, the link is instant and costs no space, and a link cannot produce a
    truncated file the way an interrupted copy can. Falls back to a copy across
    filesystems, which is the only case where linking cannot work.
    """
    target = store_root / building / part.relative_to(store_root / dataset)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(part, target)
    except OSError:
        shutil.copy2(part, target)


def _read_legacy_parts(folder: Path, parts: list[Path]) -> pd.DataFrame:
    """Those parts' rows, with SYMBOL put back from the directory name.

    Read one symbol at a time rather than the whole dataset: 538 MB fits in
    memory today and the next dataset may not, and a migration that only works
    while the store is small is a migration that fails when it is needed.
    """
    if not parts:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    frame[SYMBOL] = folder.name.removeprefix("symbol=")
    return frame


def _dataset_relative(entry: str) -> str:
    """An entry keyed on the store root, re-keyed on the dataset root.

    The swap renames the directory the manifest names. Keyed on the store root
    every entry would then miss, and the next pass would re-migrate all 93,588
    parts - writing every row a second time under a fresh content-derived
    snapshot id, a duplication that reads as real volume rather than as an
    error. Keyed on the dataset, a rename costs nothing.

    A first component with no `=` is a dataset directory: hive partition
    directories always carry one, and no dataset in this store has one in its
    name. So the rule is decidable from the entry alone, which is what lets an
    old manifest and a new one live in the same file.
    """
    head, _, rest = entry.partition("/")
    return rest if rest and "=" not in head else entry


def _read_consumed(store_root: Path, building: str) -> set[str]:
    """Legacy parts an earlier pass already migrated, by path relative to the dataset.

    A manifest rather than "has a part for this symbol been written": capture
    keeps appending to the legacy layout while the migration runs, so a second
    pass has to migrate the parts that arrived since and ONLY those. Without
    this, the second pass re-reads every part for the symbol, writes them all
    under a new content-derived snapshot id, and the store quietly holds every
    one of those rows twice - a duplication that reads as real volume.
    """
    manifest = Path(store_root) / building / MANIFEST_FILE
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:
        return set()
    # A JSON array is the format the first version wrote, and one such manifest
    # exists from the live bars migration that was already running when this
    # changed. Reading it is three lines; orphaning it would mean re-migrating
    # 62,000 parts that are already on disk and correct.
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            return {_dataset_relative(entry) for entry in json.loads(text)}
        except ValueError:
            return set()
    # One path per line. A truncated final line is a crash mid-append and is
    # dropped rather than trusted: re-migrating a part is wasteful, whereas
    # trusting a half-written name could skip a part nothing migrated.
    consumed = set()
    for line in text.splitlines():
        entry = line.strip()
        if entry:
            consumed.add(_dataset_relative(entry))
    if text and not text.endswith("\n"):
        consumed.discard(_dataset_relative(text.splitlines()[-1].strip()))
    return consumed


def _record_consumed(store_root: Path, building: str, fresh: list[str]) -> None:
    """APPEND the paths just migrated. Written after the parts, never before.

    A manifest ahead of the write it claims would make the next pass skip parts
    nothing migrated, and those rows would be missing from the new layout rather
    than merely migrated twice.

    Append rather than rewrite, and this is not micro-optimisation. The first
    version wrote the whole set after every symbol, so by symbol 2,000 it
    rewrote a 60,000-entry JSON document each time. Measured on the live bars
    migration: **7.8 GB written for a 538 MB dataset**, and a tail that crawled
    because each remaining symbol paid for every symbol before it. Quadratic IO
    in a routine whose whole purpose is to stop paying per-file costs.

    `fsync` on each append: the manifest's only job is to be true after a crash,
    and an entry sitting in the page cache when the box dies claims work that
    the next pass will then skip.
    """
    folder = Path(store_root) / building
    folder.mkdir(parents=True, exist_ok=True)
    if not fresh:
        return
    with (folder / MANIFEST_FILE).open("a", encoding="utf-8") as handle:
        handle.write("".join(f"{entry}\n" for entry in sorted(fresh)))
        handle.flush()
        os.fsync(handle.fileno())


def migrate_dataset_to_hourly(store_root: Path, dataset: str,
                              into: str | None = None) -> MigrationReport:
    """Build the hour-partitioned copy beside the legacy one. Swaps nothing.

    `into` names an existing dataset to fold the rows into instead of building
    a fresh copy, and it exists because the swap is two renames while capture
    never stops. Parts written to the old directory between the last pass and
    the rename are stranded there - measured on the live bars store 2026-08-17,
    16,517 parts, about 56 minutes of bars across 2,235 symbols. Nothing is
    lost, but they sit outside what readers now open, and only a fold puts them
    where the readers are.

    A fold is a genuinely weaker operation and the report says so: see
    FOLD_CLAIM. `swap_in_migrated_dataset` refuses a fold report outright,
    because a fold's "building" dataset is the live one and renaming that aside
    would move the store out from under every reader.
    """
    store_root = Path(store_root)
    root = store_root / dataset
    if not root.is_dir():
        raise MigrationRefused(f"no dataset at {root}")
    folders = _legacy_symbol_folders(root)
    already_hourly = _hour_partitioned_parts(root)
    if not folders and already_hourly:
        raise MigrationRefused(
            f"{root} is already partitioned by {HOUR_KEY} and holds no legacy "
            f"symbol directory; migrating it again would rewrite an "
            f"hour-partitioned store into itself")
    if not folders:
        raise MigrationRefused(f"no dataset partitions under {root}")

    building = into or f"{dataset}{BUILDING_SUFFIX}"
    consumed = _read_consumed(store_root, building)
    legacy_parts = migrated_parts = legacy_rows = 0
    hours: set[str] = set()
    # Per (hour, symbol) row counts, so verification compares the thing that
    # matters rather than a total that can net two errors against each other.
    # Built from EVERY legacy part, including ones an earlier pass consumed:
    # verification asks "does the new layout hold what the old one holds", which
    # is a question about the store, not about this run.
    expected: dict[tuple[str, str], int] = {}
    legacy_columns: set[str] = set()

    for folder in folders:
        parts = sorted(folder.glob("*.parquet"))
        if not parts:
            continue
        symbol = folder.name.removeprefix("symbol=")
        legacy_parts += len(parts)
        by_part = {part: pd.read_parquet(part) for part in parts}
        for part, rows in by_part.items():
            legacy_rows += len(rows)
            legacy_columns |= set(rows.columns) | {SYMBOL}
            for hour, group in rows.groupby(rows[AVAILABILITY_TIME].map(floor_to_hour)):
                hours.add(hour)
                expected[(hour, symbol)] = expected.get((hour, symbol), 0) + len(group)

        fresh = [part for part in parts
                 if str(part.relative_to(root)) not in consumed]
        if not fresh:
            continue
        frame = _read_legacy_parts(folder, fresh)
        # The snapshot id is derived from the parts that fed it, so the name
        # still answers "which data produced this" after compaction merged them.
        snapshot = compute_snapshot_id(fresh)
        try:
            migrated_parts += len(append_partition(store_root, building, frame, snapshot))
        except PartitionExistsError:
            # An earlier pass wrote these exact parts and died before recording
            # them. The snapshot id is content-derived, so the file already on
            # disk holds exactly these rows; recording it now is the repair.
            pass
        newly = [str(part.relative_to(root)) for part in fresh]
        consumed |= set(newly)
        _record_consumed(store_root, building, newly)

    # Parts the live writers already put in the new layout. They are carried
    # across unchanged rather than rewritten: their path already says which hour
    # they belong to, and rewriting them would change a snapshot id that answers
    # "which data produced this".
    linked: list[str] = []
    for part in already_hourly:
        hour = part.parent.parent.name.removeprefix(f"{HOUR_KEY}=")
        symbol = part.parent.name.removeprefix("symbol=")
        rows = pd.read_parquet(part)
        legacy_parts += 1
        legacy_rows += len(rows)
        legacy_columns |= set(rows.columns) | {SYMBOL}
        hours.add(hour)
        expected[(hour, symbol)] = expected.get((hour, symbol), 0) + len(rows)
        key = str(part.relative_to(root))
        if key not in consumed:
            _link_into_building(store_root, dataset, building, part)
            migrated_parts += 1
            consumed.add(key)
            linked.append(key)
    _record_consumed(store_root, building, linked)

    report = _verify(store_root, dataset, building, legacy_parts, migrated_parts,
                     legacy_rows, len(folders), len(hours), expected, legacy_columns,
                     exact=into is None)
    _record_report(store_root, report)
    return report


def _verify(store_root: Path, dataset: str, building: str, legacy_parts: int,
            migrated_parts: int, legacy_rows: int, symbols: int, hours: int,
            expected: dict[tuple[str, str], int],
            legacy_columns: set[str], exact: bool = True) -> MigrationReport:
    """Compare the built dataset against the legacy one on content.

    Group counts rather than a single total: two errors of opposite sign net to
    zero in a total, and "the numbers matched" would then be the last thing said
    before the store was swapped.

    `exact` is the difference between a migration and a fold. A migration owns
    its target, so a group the target holds and the source does not is a fault.
    A fold's target is live and capture is still appending to it, so the same
    observation is expected - and under the strict rule no fold could ever
    verify, which would make the check something to be worked around instead of
    obeyed. What a fold still refuses is the failure it exists to catch: a group
    of the source missing, or short, in the target.
    """
    migrated = read_dataset(store_root, building)
    mismatched: list[str] = []
    if migrated.empty:
        mismatched.append("the migrated dataset read back empty")
        observed_columns: set[str] = set()
    else:
        observed_columns = set(migrated.columns)
        counted = (migrated
                   .assign(**{HOUR_KEY: migrated[AVAILABILITY_TIME].map(floor_to_hour)})
                   .groupby([HOUR_KEY, SYMBOL]).size().to_dict())
        for key, rows in sorted(expected.items()):
            short = counted.get(key, 0) != rows if exact else counted.get(key, 0) < rows
            if short:
                mismatched.append(f"{key[0]}/{key[1]}: {rows} legacy row(s), "
                                  f"{counted.get(key, 0)} migrated")
        if exact:
            for key in sorted(set(counted) - set(expected)):
                mismatched.append(f"{key[0]}/{key[1]}: {counted[key]} row(s) the legacy "
                                  f"layout does not have")

    lost = tuple(sorted(legacy_columns - observed_columns))
    counts_agree = len(migrated) == legacy_rows if exact else len(migrated) >= legacy_rows
    return MigrationReport(
        dataset=dataset, building_dataset=building, legacy_parts=legacy_parts,
        migrated_parts=migrated_parts, legacy_rows=legacy_rows,
        migrated_rows=int(len(migrated)), symbols=symbols, hours=hours,
        columns_lost=lost, mismatched_groups=tuple(mismatched),
        verified=(not lost and not mismatched and counts_agree),
        claim=COPY_CLAIM if exact else FOLD_CLAIM)


def _record_report(store_root: Path, report: MigrationReport) -> None:
    """Write the claim to a file, because a claim in a transcript is not evidence."""
    target = Path(store_root) / f"{REPORT_PREFIX}{report.dataset}.json"
    target.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def swap_in_migrated_dataset(store_root: Path, dataset: str,
                             report: MigrationReport) -> tuple[Path, Path]:
    """Rename the legacy layout aside and the new one into place. Deletes nothing.

    Two renames rather than a copy: a rename is atomic and reversible, and the
    reverse is what a person will want at 3am if the new layout misbehaves. The
    legacy directory stays until a human retires it - the plan row says "before
    the old is retired", and retiring is not this function's decision.
    """
    if report.claim == FOLD_CLAIM:
        raise MigrationRefused(
            f"refusing to swap a fold report: its building dataset is "
            f"{report.building_dataset}, which is live. Renaming a live "
            f"dataset aside would move the store out from under every reader "
            f"and leave nothing in its place. A fold has already put its rows "
            f"where the readers are - there is nothing left to swap")
    if not report.verified:
        raise MigrationRefused(
            f"refusing to swap an unverified migration of {dataset}: "
            f"columns lost {report.columns_lost or '()'}, "
            f"{len(report.mismatched_groups)} mismatched group(s), "
            f"{report.legacy_rows} legacy row(s) against {report.migrated_rows}")

    store_root = Path(store_root)
    live = store_root / dataset
    built = store_root / report.building_dataset
    retired = store_root / f"{dataset}{LEGACY_SUFFIX}"
    if not built.is_dir():
        raise MigrationRefused(f"nothing built at {built}")
    if retired.exists():
        raise MigrationRefused(
            f"{retired} already exists - an earlier migration has not been "
            f"retired, and overwriting it would destroy the only copy of the "
            f"layout this one replaced")

    live.rename(retired)
    built.rename(live)
    return retired, live


def main(argv: list[str] | None = None) -> int:
    """Build, report, and swap only when asked and only when verified.

    Two steps rather than one flag-free command: the build is safe and can be
    run while capture continues, the swap changes which directory readers open.
    Making the second explicit means nobody performs it by running the first one
    again.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="store.hourly_migration",
        description="Rewrite a symbol-partitioned dataset as an hour-partitioned one.")
    parser.add_argument("--dataset", required=True,
                        help="dataset directory name, e.g. bars_60000000000ns")
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--into",
                        help="fold the rows into this EXISTING dataset instead of "
                             "building a fresh copy. For stragglers written to the "
                             "old directory between the last pass and the swap. The "
                             "report then makes the weaker fold claim, and --swap "
                             "refuses it")
    parser.add_argument("--swap", action="store_true",
                        help="after a VERIFIED build, rename the legacy layout "
                             "aside and the new one into place. Nothing is deleted")
    args = parser.parse_args(argv)

    try:
        report = migrate_dataset_to_hourly(Path(args.store_root), args.dataset,
                                           into=args.into)
    except MigrationRefused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2

    print(f"{report.dataset}: {report.legacy_parts} legacy part(s), "
          f"{report.legacy_rows} row(s), {report.symbols} symbol(s) -> "
          f"{report.migrated_parts} part(s) across {report.hours} hour(s), "
          f"{report.migrated_rows} row(s)", file=sys.stderr)
    if not report.verified:
        print(f"NOT VERIFIED - columns lost {report.columns_lost or '()'}, "
              f"{len(report.mismatched_groups)} mismatched group(s)", file=sys.stderr)
        for line in report.mismatched_groups[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"verified: {report.claim}", file=sys.stderr)

    if args.swap:
        retired, live = swap_in_migrated_dataset(Path(args.store_root), args.dataset,
                                                 report)
        print(f"swapped: {live} is now hour-partitioned; the old layout is kept at "
              f"{retired} and is not deleted by this command", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
