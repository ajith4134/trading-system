"""Build the polled Layer 1 datasets - funding and book - from the raw archive.

Separate from `store.cli`, which builds trade bars, because the two problems are
not the same shape. A bar has to close: it spans an interval, its availability
depends on the last trade that composed it, and a trade landing in the wrong
day's folder still belongs to the day its event time names. None of that applies
here. A poll is a point-in-time reading that is knowable the moment it lands, so
the day-boundary, lookahead and stranded-trade machinery would be ceremony
around a much simpler fact.

What *is* shared, and is not negotiable: a live hour is never read. `RawWriter`
marks an hour it holds with a sibling `.writing`, and reading one gets a zstd
frame mid-write - `read_pair` refuses it, correctly, but skipping it up front
means a build is not counted as failed for meeting a file that is simply still
open.

Idempotent by snapshot id: rebuilding the same day is refused by
`append_partition` as a collision rather than silently doubling every row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from capture.raw_writer import read_pair
from store.book_snapshots import build_book_frame, extract_book_snapshot
from store.funding_rates import build_funding_frame, extract_funding
from store.parquet_partition import append_partition

# Which archived stream feeds which dataset, and how to read it. Keyed by
# dataset name so the CLI, the builder and the reader all say the same word -
# `cost.funding_carry` and `cost.spread_and_depth` name the same two.
DATASETS: dict[str, tuple[str, Callable, Callable]] = {
    "funding": ("premiumIndex", extract_funding, build_funding_frame),
    "book": ("depthSnapshot", extract_book_snapshot, build_book_frame),
}


def _hour_files(capture_root: Path, venue: str, date: str, stream: str,
                symbol: str) -> list[Path]:
    """Completed hour files for one (stream, symbol) on one day.

    A live hour is skipped rather than read. It will be picked up by the next
    run once the writer closes it, and the alternative - a refused build every
    time the timer fires during an open hour - would mean the dataset only ever
    built by accident.
    """
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    files = []
    for raw in sorted(folder.glob(f"{stream}_{symbol}_*.ndjson.zst")):
        stem = str(raw)[: -len(".ndjson.zst")]
        if Path(stem + ".writing").exists():
            continue
        files.append(raw)
    return files


def build_for_day(capture_root: Path, store_root: Path, venue: str, date: str,
                  symbols: Sequence[str], dataset: str) -> dict:
    """Read one venue-day of a polled stream and append it to its dataset."""
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    stream, extract, build = DATASETS[dataset]

    observations = []
    frames = 0
    files_read = 0
    for symbol in symbols:
        for raw_path in _hour_files(capture_root, venue, date, stream, symbol):
            idx_path = Path(str(raw_path)[: -len(".ndjson.zst")] + ".idx.zst")
            if not idx_path.exists():
                continue
            files_read += 1
            for payload, entry in read_pair(raw_path, idx_path):
                frames += 1
                observations.extend(extract(payload, entry, venue, symbol))

    frame = build(observations)
    if frame.empty:
        # No rows is not an empty dataset written over the old one. Nothing is
        # appended, so a day with no polls leaves whatever was already there.
        return {"dataset": dataset, "venue": venue, "date": date,
                "files_read": files_read, "frames": frames, "rows": 0,
                "appended": False}

    snapshot_id = f"{dataset}-{venue}-{date}"
    appended = True
    try:
        append_partition(store_root, dataset, frame, snapshot_id=snapshot_id)
    except FileExistsError:
        # Already built. A collision is the partition writer refusing to double
        # every row, which is the behaviour a timer needs rather than an error.
        appended = False

    return {"dataset": dataset, "venue": venue, "date": date,
            "files_read": files_read, "frames": frames,
            "rows": int(len(frame)), "appended": appended}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-polled",
        description="Build the funding or book dataset from the raw archive.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--venue", required=True)
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root",
                        default=str(Path.home() / "capture" / "store"))
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    import json
    print(json.dumps(build_for_day(
        Path(args.capture_root), Path(args.store_root),
        args.venue, args.date, symbols, args.dataset)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
