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


class NoHourFilesForSymbol(FileNotFoundError):
    """A requested symbol matched no captured hour files for this venue and day.

    Raised only when the day itself was captured (the venue/date folder exists).
    An entirely absent day is a legitimate zero - no capture has run yet, and
    `build_bars_for_day` reports that quietly by design. But inside a real
    capture, a symbol matching nothing is almost always a typo or the wrong
    stream name, and folding it into a normal-looking aggregate is exactly the
    silent loss this module refuses everywhere else: the caller asked for that
    symbol and must find out before the store quietly excludes it, not from a
    smaller total nobody thought to check.
    """


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

    day_folder = Path(capture_root) / "raw" / venue / date
    symbol_files = {
        symbol: _hour_files(capture_root, venue, date, stream, symbol) for symbol in symbols
    }

    # Every requested symbol is checked before any frame is read, mirroring the
    # pre-check-then-write shape `append_partition` already uses: the failure
    # must arrive before work is done, not after some symbols were already read.
    # Gated on the day folder existing - see NoHourFilesForSymbol's docstring for
    # why an entirely absent day is not the same failure as a symbol missing from
    # a day that WAS captured.
    if day_folder.is_dir():
        missing = [symbol for symbol, files in symbol_files.items() if not files]
        if missing:
            raise NoHourFilesForSymbol(
                f"no {stream!r} hour files under {day_folder} for symbol(s) {missing} "
                f"(venue={venue!r}, date={date!r}); check for a typo or the wrong stream name")

    trades = []
    sources: list[Path] = []
    frames = 0
    # Per symbol, not just a running total: a symbol that contributes zero frames
    # or zero trades must stay visible in the result rather than disappear into
    # an aggregate that looks identical to one where every symbol pulled its
    # weight.
    by_symbol: dict[str, dict[str, int]] = {}
    for symbol in symbols:
        symbol_frames = 0
        symbol_trades = 0
        for raw_path in symbol_files[symbol]:
            idx_path = raw_path.with_name(raw_path.name.replace(".ndjson.zst", ".idx.zst"))
            for payload, entry in read_pair(raw_path, idx_path):
                frames += 1
                symbol_frames += 1
                extracted = extract_trades(payload, entry, venue, symbol)
                trades.extend(extracted)
                symbol_trades += len(extracted)
            sources.extend([raw_path, idx_path])
        by_symbol[symbol] = {"frames": symbol_frames, "trades": symbol_trades}

    if not trades:
        return {"frames": frames, "trades": 0, "bars": 0, "snapshot_id": None,
                "by_symbol": by_symbol}

    bars = build_bars(trades, interval_ns)
    snapshot_id = compute_snapshot_id(sources)
    append_partition(store_root, f"bars_{interval_ns}ns", bars, snapshot_id)
    return {"frames": frames, "trades": len(trades), "bars": len(bars),
            "snapshot_id": snapshot_id, "by_symbol": by_symbol}


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
    for symbol, counts in summary["by_symbol"].items():
        print(f"  {symbol}: {counts['frames']} frames -> {counts['trades']} trades",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
