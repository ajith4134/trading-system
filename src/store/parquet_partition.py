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
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from store.temporal_schema import SYMBOL, validate_temporal_frame

_CHUNK = 1 << 20


class PartitionExistsError(FileExistsError):
    """A part with this snapshot id is already on disk."""


def compute_snapshot_id(paths: Sequence[Path]) -> str:
    """A short digest over the *contents* of the inputs, order-independent.

    Content rather than filenames: the id answers "which data produced this
    result", and a rename must not read as different data. Sorted per-file
    digests rather than a running hash so the caller's iteration order cannot
    change the answer.
    """
    digests = sorted(_digest_file(Path(path)) for path in paths)
    combined = hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()
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

    written: list[Path] = []
    for symbol, group in frame.groupby(SYMBOL, sort=True):
        folder = Path(store_root) / dataset / f"symbol={symbol}"
        target = folder / f"part-{snapshot_id}.parquet"
        if target.exists():
            raise PartitionExistsError(
                f"{target} already exists; snapshot '{snapshot_id}' has been written for "
                f"{symbol}. Corrections are new snapshots, never rewrites")
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
        pq.write_table(table, target, compression="zstd")
        written.append(target)
    return written


def read_dataset(store_root: Path, dataset: str) -> pd.DataFrame:
    """Every part, concatenated. Absent datasets read empty, not as an error.

    An empty store is the correct state before the first build, and raising there
    would make "nothing captured yet" indistinguishable from a bug.
    """
    root = Path(store_root) / dataset
    if not root.is_dir():
        return pd.DataFrame()
    dataset_handle = ds.dataset(root, format="parquet", partitioning="hive")
    return dataset_handle.to_table().to_pandas()
