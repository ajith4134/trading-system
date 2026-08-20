"""Append-only Parquet parts, partitioned by symbol, identified by content.

Corrections are new parts. Nothing here opens an existing file for writing, and
`append_partition` refuses a snapshot id that already exists rather than
overwriting it - a rebuild that silently replaced a part would erase exactly the
history the store promises to keep, and the result would look like a clean run.

ZSTD because the archive it derives from is already ZSTD and the ratio on
columnar float data is worth more than the CPU at this volume.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# The archive writer already solved "a crash must not leave a half-written file
# under a name a reader trusts", down to the order of the two fsyncs. Its helper
# is imported rather than reimplemented: `capture.raw_writer._utc_moment` records
# what happened the last time this project kept three copies of one conversion -
# each copy had the bug, independently.
from capture.raw_writer import _fsync_directory
from store.temporal_schema import AVAILABILITY_TIME, SYMBOL, validate_temporal_frame

_CHUNK = 1 << 20

# The hive key that carries availability into the path. Reading it back is
# pyarrow's job; no caller ever sees it - `read_dataset` drops it, because the
# partition key is an index and the frame's shape is a contract.
HOUR_KEY = "availability_hour"

# pyarrow's dataset discovery skips names beginning with "." or "_", so an
# in-progress part is invisible to `read_dataset` even while it is being written,
# and stays invisible if a crash strands it. Without the dot the stranded file is
# discovered as a Parquet part and fails the whole dataset - the very failure the
# rename is here to prevent.
_PARTIAL_PREFIX = ".writing-part-"

# **Every dataset partitions by availability hour ALONE, with SYMBOL in the file
# body** (SL-17, RL-032; extended to all datasets by RL-034).
#
# Measured 2026-08-19: one sealed `funding` hour held 1,892 fragments for 26,015
# rows and 16.2 MiB - 13.8 rows and 8.8 KiB per file - at 29 ms each to open, so
# the whole 850 MB dataset cost 30.6 minutes to read. That is what froze the
# status wall for two days.
#
# **The list was originally a split by access pattern, and that split turned out
# to rest on a premise the code does not support.** `bars` and `book` were kept
# per-symbol on the grounds that they are read one symbol at a time - but
# `ClockGatedReader` applies its `symbols` argument to the ALREADY-MATERIALISED
# frame, and the only pushdown filters this store builds are on availability time
# and the hour (see `_availability_bound`). No path anywhere prunes by the
# `symbol=` partition, so that level cost `bars_60000000000ns` 208,880 fragments
# and pruned for nobody.
#
# A symbol level earns its keep again the day a reader pushes a symbol filter
# INTO the scan. Until such a reader exists, this frozenset is every dataset, and
# adding one to it is how that decision gets reversed.
WHOLE_UNIVERSE_DATASETS = frozenset({
    "funding", "funding_reconstructed", "option_chain",
    "bars_60000000000ns", "bars_reconstructed_60000000000ns",
    "book", "dated_futures", "exchange_reserves",
})

# `hourly_migration` builds its converted copy under this suffix and swaps it in.
# Defined here rather than there because the LAYOUT decision has to see through
# the suffix: a building copy of `funding` is still funding, and if it were
# treated as an unknown dataset the migration would rebuild the old layout and
# the conversion would be a no-op that looked like a success.
BUILDING_SUFFIX = ".hourly-building"


def partitions_by_hour_alone(dataset: str, root: Path | None = None) -> bool:
    """Whether this dataset's path carries the hour and nothing else.

    **The DATA decides, not the constant.** A dataset holding one `symbol=` part
    beside one hour-only part is UNREADABLE, not merely slow: the path-derived
    column infers as `string` and the body one as `large_string`, and pyarrow
    refuses to merge them - `ArrowTypeError`, measured 2026-08-19. A constant
    alone would produce exactly that state, because it becomes true the moment
    the code ships and the existing parts are converted later.

    So a dataset that already holds a `symbol=` directory keeps being written the
    way it already is, whatever the constant says. The conversion flips the
    layout once, and the writer follows it.
    """
    if dataset.endswith(BUILDING_SUFFIX):
        dataset = dataset[:-len(BUILDING_SUFFIX)]
    if dataset not in WHOLE_UNIVERSE_DATASETS:
        return False
    root = Path(root) if root is not None else None
    if root is None or not root.is_dir():
        return True              # nothing on disk to mix with, so start converted
    # **Cheap on purpose.** This runs on every append, and an `rglob` over the
    # 63,063 fragments funding held on 2026-08-19 would make the check its own
    # hot spot - the exact mistake SL-14 recorded. Two listings answer it: a
    # legacy `symbol=` at the top, or a `symbol=` inside the newest hour, which
    # is the one a converted dataset would have written into.
    if any(root.glob("symbol=*")):
        return False
    hours = sorted(entry.name for entry in os.scandir(root)
                   if entry.is_dir() and entry.name.startswith(f"{HOUR_KEY}="))
    if not hours:
        return True
    return not any(entry.is_dir() and entry.name.startswith("symbol=")
                   for entry in os.scandir(root / hours[-1]))

# The unified schema of each dataset, and the exact set of fragment paths it was
# built from. Keyed by dataset root, because two datasets share nothing.
#
# Reading a fragment's `physical_schema` opens its file to read the footer, and
# `read_dataset` did that for every fragment on every call. Measured 2026-08-17
# on the live bars dataset: 49,100 fragments over 2,229 symbol partitions, a
# read filtered to return ZERO rows still unfinished after 10 minutes, and the
# boards generator stalled for 1h50m holding 5.3 GB. The walk is unavoidable -
# see the comment in `read_dataset` for the column it exists to save - but doing
# it again for a fragment already read is not.
#
# The path set is the generation marker, and it is deliberately not an mtime: a
# restore from GCS rewrites every mtime while the data is unchanged, so an
# mtime-keyed cache would rebuild pointlessly, and an mtime-keyed *skip* would
# omit real rows. A path either was folded in or was not, and that survives a
# restore, a reboot and a copy.
_UNIFIED_SCHEMAS: dict[Path, tuple[frozenset[str], pa.Schema]] = {}
_SCHEMA_CACHE_LOCK = threading.Lock()
_FRAGMENT_SCHEMA_READS = 0


def count_fragment_schema_reads() -> int:
    """How many fragment footers this process has opened for their schema.

    The number SL-14 is accepted on. It has to grow with what arrived since the
    last read rather than with the size of the archive, and a test that asserts
    that is the only thing standing between here and the 49,100-file walk
    reappearing the next time someone edits the read path.
    """
    return _FRAGMENT_SCHEMA_READS


def clear_schema_cache() -> None:
    """Forget every cached schema. For tests, and for a caller that must not
    inherit a generation observed before some external change to the store."""
    global _FRAGMENT_SCHEMA_READS
    with _SCHEMA_CACHE_LOCK:
        _UNIFIED_SCHEMAS.clear()
        _FRAGMENT_SCHEMA_READS = 0


def unify_dataset_schema(root: Path, dataset_handle: ds.Dataset) -> pa.Schema | None:
    """The union of every fragment's schema, reading each fragment once.

    Returns None for a dataset with no fragments, which is "nothing captured
    yet" rather than an error.

    Incremental because union is associative: a schema already unified stays
    valid when new fragments are unified onto it, so an append costs only the
    appended files. It is also MONOTONIC, which is the dangerous half - union
    can only add columns, so a cache that only ever grew would keep reporting a
    column whose only source file had been removed by a compaction or a partial
    restore, and a long-lived process would disagree with a fresh one about what
    the dataset holds. A path that has disappeared therefore forces a full
    rebuild rather than an incremental step.
    """
    global _FRAGMENT_SCHEMA_READS
    fragments = {fragment.path: fragment for fragment in dataset_handle.get_fragments()}
    if not fragments:
        with _SCHEMA_CACHE_LOCK:
            _UNIFIED_SCHEMAS.pop(root, None)
        return None
    present = frozenset(fragments)

    with _SCHEMA_CACHE_LOCK:
        cached = _UNIFIED_SCHEMAS.get(root)
    if cached is not None and cached[0] <= present:
        known, unified = cached
        unread = [fragments[path] for path in present - known]
    else:
        unified, unread = None, list(fragments.values())

    schemas = []
    for fragment in unread:
        schemas.append(fragment.physical_schema)
        _FRAGMENT_SCHEMA_READS += 1
    if schemas or unified is None:
        # `dataset_handle.schema` is unioned back in because `physical_schema`
        # is what is IN each file, which excludes the hive partition column:
        # `symbol` lives in the directory name, not in the parquet, and it is a
        # required column here.
        unified = pa.unify_schemas(
            [*([unified] if unified is not None else []), *schemas,
             dataset_handle.schema])

    with _SCHEMA_CACHE_LOCK:
        _UNIFIED_SCHEMAS[root] = (present, unified)
    return unified


class PartitionExistsError(FileExistsError):
    """A part with this snapshot id is already on disk."""


def floor_to_hour(availability_ns: int) -> str:
    """The partition value for an availability time: `2026-08-17T11`, UTC.

    ONE key rather than a date key and an hour key: one added column, one
    directory level, and a fixed-width string whose lexicographic order IS its
    chronological order - so `>=` on the path segment is a correct time bound
    with no parsing on the read side.

    FLOORED, never rounded. Flooring includes the partly-consumed hour a
    watermark sits in, and the exact row filter then removes the rows already
    seen. Rounding up would skip the rest of that hour permanently: a later poll
    carries a later watermark, so nothing ever reaches back for them.

    UTC because the store is UTC everywhere else, and a local-time path would
    silently reorder itself across a daylight-saving boundary.
    """
    # Integer division, never `ns / 1e9`. A float carries 53 bits of mantissa and
    # an epoch nanosecond needs 61, so the division rounds - measured here, an
    # availability time one nanosecond before 06:00 floored to 06 instead of 05,
    # which would file a row under an hour it did not happen in and let a poll
    # bounded at 06:00 skip it. The failure is silent and the row never returns.
    moment = dt.datetime.fromtimestamp(int(availability_ns) // 1_000_000_000, dt.UTC)
    return moment.strftime("%Y-%m-%dT%H")


def _availability_bound(not_before_ns: int | None, *, partitioned_by_hour: bool):
    """The read filter for a lower bound on availability, or None for no bound.

    Two conditions doing different jobs. The hour condition is answered from the
    PATH, so pyarrow skips whole directories without opening a file - that is
    the entire point of the layout, and the reason a poll's cost stopped growing
    with the archive. The row condition is the exact bound, applied inside the
    few files that survive.

    `partitioned_by_hour` is asked of the dataset rather than assumed, and that
    is not defensive tidiness: naming a field a dataset does not have makes
    pyarrow raise `ArrowInvalid: No match for FieldRef.Name(availability_hour)`
    and take the whole read down. Every dataset is in the old layout until its
    migration has run, the paper engine polls with a bound every 60 seconds, and
    a store that only reads correctly after a migration nobody has run yet is a
    store that stops the moment this lands.
    """
    if not_before_ns is None:
        return None
    row_bound = ds.field(AVAILABILITY_TIME) >= int(not_before_ns)
    if not partitioned_by_hour:
        return row_bound
    return (ds.field(HOUR_KEY) >= floor_to_hour(not_before_ns)) & row_bound


def _is_partitioned_by_hour(root: Path, dataset_handle: ds.Dataset) -> bool:
    """True only when EVERY fragment carries an hour in its path.

    The schema alone is not enough, and getting this wrong loses rows silently.
    A dataset part-way through its migration holds both layouts: legacy parts at
    `symbol=<S>/`, new ones at `availability_hour=<H>/symbol=<S>/`. Hive
    partitioning gives the legacy parts a NULL hour, `NULL >= '2026-08-17T12'`
    is null rather than true, and every legacy row is therefore dropped from
    every bounded read - no error, no warning, just a smaller answer.

    Measured 2026-08-17: the live bars dataset entered exactly that state within
    minutes of this code landing, because the store supervisors reload their
    child on restart and a builder wrote 4,595 hour-partitioned parts beside
    2,231 legacy symbol directories.

    So the presence of a top-level `symbol=` directory disables hour pruning for
    the whole dataset. Reads stay correct and merely lose the speed-up until the
    migration finishes, at which point no such directory remains and pruning
    turns itself back on.
    """
    if HOUR_KEY not in dataset_handle.schema.names:
        return False
    return not any(root.glob("symbol=*"))


def select_fragments(store_root: Path, dataset: str,
                     not_before_ns: int | None = None) -> list[str]:
    """The fragment paths a read with this bound would open. Opens none of them.

    Exists to be measured. "The poll got faster" is an assertion; "the poll
    selected 2 of 52,487 fragments" is a measurement, and SL-15's acceptance is
    written in those terms.
    """
    root = Path(store_root) / dataset
    if not root.is_dir():
        return []
    handle = ds.dataset(root, format="parquet", partitioning="hive")
    bound = _availability_bound(
        not_before_ns, partitioned_by_hour=_is_partitioned_by_hour(root, handle))
    return [fragment.path for fragment in handle.get_fragments(filter=bound)]


def compute_snapshot_id(paths: Sequence[Path],
                        unread_input_names: Sequence[str] = ()) -> str:
    """A short digest over the *contents* of the inputs, order-independent.

    Content rather than filenames: the id answers "which data produced this
    result", and a rename must not read as different data. Sorted per-file
    digests rather than a running hash so the caller's iteration order cannot
    change the answer.

    `unread_input_names` names inputs the caller COULD not read - a file still
    being appended to - and folds them in by name, deliberately not by content.
    Their bytes are changing while this runs, so a digest of them would answer a
    question about an instant that has already passed; the name is the stable
    part and it is the part that matters, because it says which input this result
    is missing. A run that skipped something produced different data from one that
    read everything, and an id that cannot tell those apart lets `append_partition`
    refuse the complete rebuild as a duplicate of the incomplete one - in a store
    with no delete path, that makes the incomplete result permanent.

    Names, not paths: the same inputs read from a copied archive are the same
    inputs, exactly as the content digest already treats them.
    """
    digests = sorted(_digest_file(Path(path)) for path in paths)
    unread = sorted(f"unread:{name}" for name in unread_input_names)
    combined = hashlib.sha256("".join(digests + unread).encode("utf-8")).hexdigest()
    return combined[:16]


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def append_partition(store_root: Path, dataset: str, frame: pd.DataFrame,
                     snapshot_id: str) -> list[Path]:
    """Write one part per availability hour per symbol. Never touches an existing file.

    Grouped by (hour, symbol) rather than by symbol alone because a frame is not
    one hour: a builder handing over three hours of bars for one symbol has to
    land in three directories, or the hour in the path would be a lie and the
    pruning built on it would skip real rows.
    """
    # Validated before any directory is created so a refused write leaves no trace
    # and cannot be mistaken for a partial success.
    validate_temporal_frame(frame)
    if frame.empty:
        return []

    hours = frame[AVAILABILITY_TIME].map(floor_to_hour)
    # A whole-universe dataset groups by the hour alone, so one poll of 1,287
    # symbols is one part rather than 1,287 (SL-17, RL-032).
    by_hour_alone = partitions_by_hour_alone(dataset,
                                             Path(store_root) / dataset)
    groups = (list(frame.groupby(hours.rename(HOUR_KEY), sort=True))
              if by_hour_alone
              else list(frame.groupby([hours.rename(HOUR_KEY), SYMBOL], sort=True)))

    # Every target is checked before any part is written. Checking and writing
    # group-by-group in one pass would let a frame with N groups write the first
    # N-1 parts and only then discover the Nth collides, leaving those N-1 behind as
    # a half-written snapshot - exactly the partial state this store promises never
    # to hold, and the promise Task 4's reader is built on.
    plan: list[tuple[str, pd.DataFrame, Path, Path]] = []
    for key, group in groups:
        hour, symbol = (key, None) if by_hour_alone else key
        folder = Path(store_root) / dataset / f"{HOUR_KEY}={hour}"
        if symbol is not None:
            folder = folder / f"symbol={symbol}"
        target = folder / f"part-{snapshot_id}.parquet"
        if target.exists():
            raise PartitionExistsError(
                f"{target} already exists; snapshot '{snapshot_id}' has been written for "
                f"{symbol or hour}. Corrections are new snapshots, never rewrites")
        plan.append((symbol, group, folder, target))

    written: list[Path] = []
    for symbol, group, folder, target in plan:
        folder.mkdir(parents=True, exist_ok=True)
        # SYMBOL is dropped from the file body because it is already encoded in the
        # "symbol=..." directory name. Writing it into the file too gives pyarrow two
        # sources of truth for the same field - the file column infers as large_string,
        # the hive path segment infers as string - and ds.dataset() refuses to merge
        # them (ArrowTypeError) the moment more than one part exists for a symbol.
        # Dropping the duplicate is the standard hive-partitioning convention, not a
        # loss: read_dataset reconstructs the column from the path on every read.
        # SYMBOL is dropped only when the PATH carries it. A whole-universe
        # dataset has no `symbol=` segment to reconstruct it from, so the column
        # travels in the body - and it is the column this store has already lost
        # once, silently: `funding_interval_hours`, 2026-08-09, which turned a
        # 4-hourly funding rate into an 8-hourly one.
        body = (group.reset_index(drop=True) if symbol is None
                else group.drop(columns=[SYMBOL]).reset_index(drop=True))
        table = pa.Table.from_pandas(body, preserve_index=False)
        _write_part_atomically(table, folder, target)
        written.append(target)
    return written


def _write_part_atomically(table: pa.Table, folder: Path, target: Path) -> None:
    """Write a part to a temporary name and rename it into place once complete.

    `pq.write_table` straight to the target writes in place, so a crash or a
    power loss mid-write leaves a truncated Parquet file under the name readers
    trust. `read_dataset` then raises ArrowInvalid for the ENTIRE dataset - every
    symbol becomes unreadable, not only the one being written - and the rebuild
    that would repair it collides with the truncated part, because the snapshot
    id is content-derived and therefore unchanged. With no delete path by design,
    recovery means manual filesystem surgery.

    Renaming makes a part either absent or complete, which leaves the rebuild as
    the whole recovery path. This does not weaken append-only: the rename target
    is still refused by the pre-flight collision check in `append_partition`
    before any bytes are written, so a rename can only ever create a name that
    did not exist.

    The temporary file is fsynced BEFORE the rename and the directory after it,
    the discipline `capture.raw_writer._write_lines` documents: without the
    first, a power loss can leave the new name pointing at an empty inode -
    atomic in the rename sense and still data loss; without the second, the
    rename itself may not survive.
    """
    handle, temporary_name = tempfile.mkstemp(dir=folder, prefix=_PARTIAL_PREFIX,
                                              suffix=".parquet")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as sink:
            pq.write_table(table, sink, compression="zstd")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, target)
        _fsync_directory(folder)
    except BaseException:
        # An abandoned temporary is invisible to dataset discovery (see
        # _PARTIAL_PREFIX), but leaving it would still accumulate dead bytes on
        # every failed write.
        temporary.unlink(missing_ok=True)
        raise


def read_dataset(store_root: Path, dataset: str,
                 not_before_ns: int | None = None,
                 columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Every part, concatenated. Absent datasets read empty, not as an error.

    An empty store is the correct state before the first build, and raising there
    would make "nothing captured yet" indistinguishable from a bug.

    `not_before_ns` pushes a lower bound on AVAILABILITY time into the parquet
    scan, so a caller that already holds everything older does not pay to
    materialise it again. Default None reads exactly what this function read
    before it existed, and every existing caller gets that.

    The bound is on availability rather than event time, and that is the whole
    reason it is safe: a correction to an old bar carries a LATER availability
    time by definition, so it still arrives through the bound, while rows already
    consumed cannot come back. Bounding on event time instead would hide
    corrections, which is the one thing this store exists to preserve.

    Added 2026-08-17 after the forward paper engine was OOM-killed (exit 137,
    fourth restart that day): it re-read the whole bars dataset every 60-second
    poll, holding 5.5 GB, on a 30 GB box with no swap.

    `columns` narrows the read to the columns named, and it does two things at
    once. Measured 2026-08-19: the bars dataset holds **168,639 parquet
    fragments** and funding holds 58,841. A caller that wants a row count or a
    freshness stamp was materialising every column of every one of them, and
    before that it was opening every fragment's FOOTER to unify the schema -
    which is why a status board took 25 minutes and 11.7 GB to answer "how many
    rows are in the store".

    **Schema unification is skipped when every requested column is already in the
    dataset's own schema, and that is safe for exactly that case.** Unification
    exists because pyarrow infers a dataset's schema from the first fragment, so
    a column added by a LATER partition would be silently dropped. A column that
    is already in the inferred schema cannot be the column that goes missing. Ask
    for a column that is not there and the full unified path runs, so a caller
    can never lose data by naming a column.
    """
    root = Path(store_root) / dataset
    if not root.is_dir():
        return pd.DataFrame()
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive")

    if columns is not None:
        wanted = list(dict.fromkeys(columns))
        if wanted and set(wanted) <= set(dataset_handle.schema.names):
            table = dataset_handle.to_table(
                columns=wanted,
                filter=_availability_bound(
                    not_before_ns,
                    partitioned_by_hour=_is_partitioned_by_hour(root,
                                                                dataset_handle)))
            if HOUR_KEY in table.column_names:
                table = table.drop_columns([HOUR_KEY])
            return table.to_pandas()

    # Unify the schemas across every fragment before reading, and this is not
    # tidiness - without it the reader SILENTLY DROPS COLUMNS.
    #
    # pyarrow infers a dataset's schema from the first fragment it discovers, so
    # any column added by a later partition is absent from that inferred schema
    # and never appears in the result. The data is on disk; the read simply does
    # not return it, and nothing says so.
    #
    # Measured 2026-08-09: `funding_interval_hours` was written into bybit's
    # partition - it is the only venue that publishes a per-symbol funding
    # interval, and annualising a 4-hourly rate as 8-hourly is wrong by a factor
    # of two - and `read_dataset` returned a frame without the column at all.
    # An append-only store whose reader quietly forgets a field is worse than one
    # that refuses: the refusal is visible.
    #
    # A column absent from an older partition reads as null there, which is the
    # honest answer - it was not recorded then.
    # Each fragment's footer is opened once per generation, not once per read -
    # see `unify_dataset_schema`, and `count_fragment_schema_reads` for the
    # measurement that keeps it that way.
    unified = unify_dataset_schema(root, dataset_handle)
    if unified is None:
        return pd.DataFrame()
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive",
                                schema=unified)
    # Inclusive at the boundary, matching `read_as_of`'s upper bound. Off by one
    # in this comparison silently drops a bar on every poll of a caller that
    # advances its watermark to the newest availability time it has seen.
    table = dataset_handle.to_table(
        columns=list(dict.fromkeys(columns)) if columns else None,
        filter=_availability_bound(
            not_before_ns,
            partitioned_by_hour=_is_partitioned_by_hour(root, dataset_handle)))
    # The hour key is an index, not data. Dropping it here keeps the frame's
    # shape a contract: no caller learns that the store gained a partition key,
    # and a reader written before this change sees exactly what it saw before.
    if HOUR_KEY in table.column_names:
        table = table.drop_columns([HOUR_KEY])
    return table.to_pandas()
