"""Append-only Parquet parts, partitioned by symbol, identified by content.

Corrections are new parts. Nothing here opens an existing file for writing, and
`append_partition` refuses a snapshot id that already exists rather than
overwriting it - a rebuild that silently replaced a part would erase exactly the
history the store promises to keep, and the result would look like a clean run.

ZSTD because the archive it derives from is already ZSTD and the ratio on
columnar float data is worth more than the CPU at this volume.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
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
from store.temporal_schema import SYMBOL, validate_temporal_frame

_CHUNK = 1 << 20

# pyarrow's dataset discovery skips names beginning with "." or "_", so an
# in-progress part is invisible to `read_dataset` even while it is being written,
# and stays invisible if a crash strands it. Without the dot the stranded file is
# discovered as a Parquet part and fails the whole dataset - the very failure the
# rename is here to prevent.
_PARTIAL_PREFIX = ".writing-part-"


class PartitionExistsError(FileExistsError):
    """A part with this snapshot id is already on disk."""


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
    """Write one part per symbol. Never touches an existing file."""
    # Validated before any directory is created so a refused write leaves no trace
    # and cannot be mistaken for a partial success.
    validate_temporal_frame(frame)
    if frame.empty:
        return []

    groups = list(frame.groupby(SYMBOL, sort=True))

    # Every target is checked before any part is written. Checking and writing
    # symbol-by-symbol in one pass would let a frame with N symbols write the first
    # N-1 parts and only then discover the Nth collides, leaving those N-1 behind as
    # a half-written snapshot - exactly the partial state this store promises never
    # to hold, and the promise Task 4's reader is built on.
    plan: list[tuple[str, pd.DataFrame, Path, Path]] = []
    for symbol, group in groups:
        folder = Path(store_root) / dataset / f"symbol={symbol}"
        target = folder / f"part-{snapshot_id}.parquet"
        if target.exists():
            raise PartitionExistsError(
                f"{target} already exists; snapshot '{snapshot_id}' has been written for "
                f"{symbol}. Corrections are new snapshots, never rewrites")
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
        body = group.drop(columns=[SYMBOL]).reset_index(drop=True)
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


def read_dataset(store_root: Path, dataset: str) -> pd.DataFrame:
    """Every part, concatenated. Absent datasets read empty, not as an error.

    An empty store is the correct state before the first build, and raising there
    would make "nothing captured yet" indistinguishable from a bug.
    """
    root = Path(store_root) / dataset
    if not root.is_dir():
        return pd.DataFrame()
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive")

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
    # `physical_schema` is what is IN each file, which excludes the hive
    # partition column - `symbol` lives in the directory name, not the parquet.
    # Unifying those alone and handing the result back as the dataset schema
    # therefore loses the partition field, which is a required column here.
    # The inferred schema is unioned back in to keep it.
    fragment_schemas = [f.physical_schema for f in dataset_handle.get_fragments()]
    if not fragment_schemas:
        return pd.DataFrame()
    unified = pa.unify_schemas([*fragment_schemas, dataset_handle.schema])
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive",
                                schema=unified)
    return dataset_handle.to_table().to_pandas()
