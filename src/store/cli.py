# src/store/cli.py
"""Builds the bitemporal store from the raw archive.

Reads through `capture.raw_writer.read_pair`, which refuses a torn file rather
than returning its readable prefix. That refusal is load-bearing here: silently
building from a truncated hour produces a store that is quietly missing trades,
and every statistic computed from it is wrong in a way nothing reports.

    python -m store.cli --venue binance --date 2026-08-02 --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from capture.raw_writer import RAW_SUFFIX, read_pair
from store.parquet_partition import append_partition, compute_snapshot_id
from store.trade_bars import build_bars, extract_trades

_TRADE_STREAMS = {"binance": "trade", "hyperliquid": "trades"}
DEFAULT_INTERVAL_NS = 60_000_000_000


def _hour_files(capture_root: Path, venue: str, date: str,
                stream: str, symbol: str) -> list[Path]:
    folder = Path(capture_root) / "raw" / venue / date
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{stream}_{symbol}_*{RAW_SUFFIX}"))


def build_bars_for_day(capture_root: Path, store_root: Path, venue: str, date: str,
                       symbols: Sequence[str], interval_ns: int) -> dict:
    """Read one venue-day of trades and append the resulting bars."""
    stream = _TRADE_STREAMS.get(venue)
    if stream is None:
        raise SystemExit(f"no trade stream known for venue '{venue}'")

    trades = []
    sources: list[Path] = []
    frames = 0
    for symbol in symbols:
        for raw_path in _hour_files(capture_root, venue, date, stream, symbol):
            idx_path = raw_path.with_name(raw_path.name.replace(".ndjson.zst", ".idx.zst"))
            for payload, entry in read_pair(raw_path, idx_path):
                frames += 1
                trades.extend(extract_trades(payload, entry, venue, symbol))
            sources.extend([raw_path, idx_path])

    if not trades:
        return {"frames": frames, "trades": 0, "bars": 0, "snapshot_id": None}

    bars = build_bars(trades, interval_ns)
    snapshot_id = compute_snapshot_id(sources)
    append_partition(store_root, f"bars_{interval_ns}ns", bars, snapshot_id)
    return {"frames": frames, "trades": len(trades), "bars": len(bars),
            "snapshot_id": snapshot_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="store", description="Build the bitemporal store from captured frames.")
    parser.add_argument("--venue", required=True, choices=sorted(_TRADE_STREAMS))
    parser.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")

    summary = build_bars_for_day(
        Path(args.capture_root), Path(args.store_root),
        args.venue, args.date, symbols, args.interval_ns)
    print(f"{summary['frames']} frames -> {summary['trades']} trades -> "
          f"{summary['bars']} bars (snapshot {summary['snapshot_id']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
