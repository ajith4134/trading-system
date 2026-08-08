"""Republishes a venue-day's bars as a bitemporal correction, never as a rewrite.

Written to remediate a real defect. Until 2026-08-08 `_extract_binance` turned
Binance's zero-price placeholder frames into trades at price 0.0, and
`low=("price", "min")` needs one of those to ruin a bar. 746 of the store's 1,671
bars carried a non-positive price: binance BTCUSDT, ETHUSDT and SOLUSDT across
2026-08-02 and 2026-08-03. `store.trade_bars.is_tradeable` stops it happening again
and does nothing about the bars already written.

Those bars cannot be deleted and must not be overwritten - "append-only, corrections
are new rows" is the store's first sentence, and it is what lets a backtest ask what
was believed at a past moment. So the fix is a second row per affected bar, carrying
the corrected OHLCV and a LATER availability time.

Later is not cosmetic. `ClockGatedReader.read_as_of` resolves duplicates with
`sort_values(availability_time, kind="mergesort")` then
`drop_duplicates(keep="last")`, so two rows tying on availability time are separated
only by parquet discovery order - which depends on the filesystem and on whichever
snapshot ids a build happened to produce. A correction stamped with the same
availability time as the row it corrects would win on one machine and lose on
another. `build_bars_for_day` derives availability time from the data (the later of
the bar's close and its last ingestion), so a plain rebuild recomputes the SAME
value and is exactly that non-correction.

The correction is therefore stamped at the moment the correction became knowable,
which is also the bitemporally honest answer: nobody could have acted on the
corrected number before the defect was found. A `read_as_of` before that instant
still returns the corrupt bar, on purpose - that is what the store was believed to
hold at the time, and hiding it would make the archive lie about its own past.

The rebuild itself is delegated to `build_bars_for_day` against a scratch store
rather than reimplemented here. A second copy of the extraction, event-time
filtering, lookahead and quarantine logic would be a second thing to keep correct,
and the whole point is that the corrected rows come from the same code path every
other bar came from.

    python -m store.correct_bars --venue binance --date 2026-08-03 \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT --reason "zero-price frames" --apply
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from store.cli import DEFAULT_INTERVAL_NS, build_bars_for_day
from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from store.temporal_schema import AVAILABILITY_TIME, EVENT_TIME, SYMBOL, VENUE


def correction_snapshot_id(venue: str, date: str, reason: str, at_ns: int) -> str:
    """A distinct, reproducible id for one correction.

    Distinct because `append_partition` refuses to touch an existing part, and a
    correction rebuilt from the same raw files digests to the same snapshot id as the
    build it is correcting - so it would be refused as an accidental re-run. Derived
    rather than random so the same correction, replayed, names the same part.
    """
    digest = hashlib.sha256(
        f"correction:{venue}:{date}:{reason}:{at_ns}".encode("utf-8")).hexdigest()
    return f"fix{digest[:13]}"


def nonpositive_bars(store_root: Path, dataset: str) -> pd.DataFrame:
    """Bars a reader would serve today whose prices are not prices.

    Read through `ClockGatedReader` rather than off the parquet files, because what
    matters is what a consumer gets after correction resolution - a corrupt row that
    has already been superseded is not a problem to fix twice.
    """
    frame = ClockGatedReader(store_root, dataset).read_as_of(2**62)
    if frame.empty:
        return frame
    bad = (frame["open"] <= 0) | (frame["high"] <= 0) | (frame["low"] <= 0) | (frame["close"] <= 0)
    return frame[bad]


def republish_day(capture_root: Path, store_root: Path, venue: str, date: str,
                  symbols: Sequence[str], reason: str, *,
                  interval_ns: int = DEFAULT_INTERVAL_NS,
                  correction_time_ns: int | None = None,
                  apply: bool = True) -> dict:
    """Rebuild one venue-day and append it as a correction. Returns what it did.

    `apply=False` rebuilds and reports without writing, so the count of rows a
    correction would change can be seen before anything is committed to an
    append-only store.
    """
    correction_time_ns = time.time_ns() if correction_time_ns is None else correction_time_ns
    dataset = f"bars_{interval_ns}ns"

    with tempfile.TemporaryDirectory(prefix="bar-correction-") as scratch:
        summary = build_bars_for_day(
            capture_root=Path(capture_root), store_root=Path(scratch), venue=venue,
            date=date, symbols=list(symbols), interval_ns=interval_ns)
        rebuilt = ClockGatedReader(Path(scratch), dataset).read_as_of(2**62)

    if rebuilt.empty:
        return {"venue": venue, "date": date, "rebuilt_rows": 0, "written": 0,
                "reason": "the rebuild produced no bars; nothing to correct"}

    rebuilt = rebuilt[rebuilt[VENUE] == venue].copy()
    # The one field that makes this a correction rather than a duplicate.
    rebuilt[AVAILABILITY_TIME] = correction_time_ns

    snapshot_id = correction_snapshot_id(venue, date, reason, correction_time_ns)
    written: list[Path] = []
    if apply:
        written = append_partition(Path(store_root), dataset, rebuilt, snapshot_id)

    return {
        "venue": venue, "date": date,
        "rebuilt_rows": len(rebuilt),
        "written": len(written),
        "snapshot_id": snapshot_id,
        "correction_time_ns": correction_time_ns,
        "stranded": summary.get("trades_stranded", 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="correct_bars",
        description="Republish a venue-day's bars as a bitemporal correction.")
    parser.add_argument("--venue", required=True)
    parser.add_argument("--date", required=True, action="append", dest="dates",
                        help="UTC date, repeatable")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--reason", required=True,
                        help="why this day is being corrected; part of the snapshot id")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    # Off by default. Appending to an append-only store cannot be undone, so the
    # dry run is what a reader gets unless they say otherwise.
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without it, rebuild and report only")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    store_root, dataset = Path(args.store_root), f"bars_{args.interval_ns}ns"
    before = len(nonpositive_bars(store_root, dataset))
    print(f"bars a reader would serve with a non-positive price, before: {before}",
          file=sys.stderr)

    for date in args.dates:
        result = republish_day(
            Path(args.capture_root), store_root, args.venue, date, symbols,
            args.reason, interval_ns=args.interval_ns, apply=args.apply)
        print(f"  {args.venue} {date}: rebuilt {result['rebuilt_rows']} row(s), "
              f"wrote {result['written']} part(s), snapshot {result.get('snapshot_id')}",
              file=sys.stderr)

    after = len(nonpositive_bars(store_root, dataset))
    print(f"bars a reader would serve with a non-positive price, after:  {after}",
          file=sys.stderr)
    if not args.apply:
        print("dry run: nothing was written. Re-run with --apply", file=sys.stderr)
    return 0


if __name__ == "__main__":       # pragma: no cover - entry point
    raise SystemExit(main())
