"""Funding history fetched from REST, into a dataset that cannot be backtested.

Carry is made of funding and the archive holds days of it. The breadth gate of
the goal spec §5a.4 - *do triggers cluster and do the return streams correlate*
- needs months, and it gates all of Phase 4:

    If triggers cluster and returns correlate, the architecture is one macro bet
    wearing 1,290 hats and must be redesigned.

Binance publishes 166 days of settled funding over REST, so the *numbers* are
recoverable. What is not recoverable is that we observed them. They arrive with
no capture provenance, no availability time we witnessed, and no raw frame
behind them - in a store whose entire claim is that every row records when we
could have known it.

So they do not go in `funding`. They go here, in a dataset named for what it is.

## The property that makes this safe, and it is mechanical rather than a rule

**`availability_time` is when we FETCHED the row, not when the venue settled
it.** Everything follows from that one line:

- `read_as_of(t)` for any `t` before the fetch returns **nothing**. A backtest
  cannot consume this dataset by accident, because at every simulated instant in
  the past it is empty. The clock gate refuses it for free.
- `read_as_of(now)` returns all of it, which is exactly what a research question
  about historical clustering needs.

That is the whole design. A comment saying "do not backtest on this" would be a
rule someone can forget; an availability time that is honest is a rule the
reader enforces without knowing it exists.

The alternative - stamping availability at settlement time, as though we had
been watching - is the one thing this must never do. It would make the rows
indistinguishable from captured ones to every consumer, and Layer 1 exists to
make that class of leak structurally impossible rather than a matter of care.

`is_reconstructed` is on every row as well, so a frame that gets copied,
filtered or joined somewhere else still says what it is.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import pandas as pd

from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

# Named for what it is, and deliberately not a variant spelling of `funding`. A
# reader that has to notice a suffix to know what it is holding will one day not
# notice it.
DATASET = "funding_reconstructed"

_BINANCE_FUNDING_HISTORY = "https://fapi.binance.com/fapi/v1/fundingRate"
# The venue caps the page at 500 regardless of what is asked for - measured
# 2026-08-09, `limit=1000` returned 500 rows spanning 166 days. Asking for the
# documented maximum and receiving half of it silently is exactly the kind of
# thing that turns into "we have a year of history" in someone's head.
_PAGE_LIMIT = 500
_MS_TO_NS = 1_000_000
_REQUEST_WEIGHT = 1


@dataclass(frozen=True)
class ReconstructedFunding:
    """One settled funding rate, fetched long after it settled."""

    symbol: str
    venue: str
    funding_rate: Decimal
    mark_price: Decimal | None
    settled_at_ns: int          # the venue's settlement time - a real event time
    fetched_at_ns: int          # when we asked, and the only time we witnessed


def parse_binance_history(payload: str, venue: str,
                          fetched_at_ns: int) -> list[ReconstructedFunding]:
    """Read one `fundingRate` page. A row without a usable rate is skipped.

    Rates are parsed from the venue's decimal string rather than through float,
    the same reason `fee_schedule` and `funding_rates` do: a representation
    error here prices every carry trade that reads it.
    """
    try:
        rows = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(rows, list):
        return []

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rate, settled = row.get("fundingRate"), row.get("fundingTime")
        symbol = row.get("symbol")
        if not (isinstance(rate, str) and rate and isinstance(settled, int)
                and isinstance(symbol, str) and symbol):
            continue
        mark = row.get("markPrice")
        parsed.append(ReconstructedFunding(
            symbol=symbol,
            venue=venue,
            funding_rate=Decimal(rate),
            mark_price=Decimal(mark) if isinstance(mark, str) and mark else None,
            settled_at_ns=settled * _MS_TO_NS,
            fetched_at_ns=fetched_at_ns,
        ))
    return parsed


def build_reconstructed_frame(rows: Sequence[ReconstructedFunding]) -> pd.DataFrame:
    """Bitemporal rows whose availability time is the fetch, not the settlement.

    See the module docstring: this single choice is what stops the dataset being
    backtestable. Every simulated clock in the past sees an empty frame.
    """
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: r.symbol,
        VENUE: r.venue,
        "funding_rate": str(r.funding_rate),
        "mark_price": None if r.mark_price is None else str(r.mark_price),
        # The real settlement instant. Honest as an event time - it is when the
        # rate applied - and useless for leaking, because availability governs
        # what a reader may see.
        EVENT_TIME: r.settled_at_ns,
        INGESTION_TIME: r.fetched_at_ns,
        AVAILABILITY_TIME: r.fetched_at_ns,
        # Carried per row so a frame that is filtered, joined or copied
        # elsewhere still says what it is, long after it left this dataset.
        "is_reconstructed": True,
    } for r in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME):
        frame[column] = frame[column].astype("int64")
    for column in ("funding_rate", "mark_price"):
        frame[column] = frame[column].astype("string")
    frame["is_reconstructed"] = frame["is_reconstructed"].astype("bool")
    return frame


def _fetch(url: str, timeout: float = 20.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_binance_history(symbol: str, venue: str = "binance", budget=None,
                          fetch=_fetch, now_ns=time.time_ns) -> list[ReconstructedFunding]:
    """One symbol's settled funding, as far back as the venue will serve it.

    Returns [] rather than raising on a failed request. One symbol's history is
    not worth abandoning the other 856 for, and a symbol that returns nothing is
    visible as a gap in the built dataset.
    """
    if budget is not None and not budget.try_spend(_REQUEST_WEIGHT):
        return []
    try:
        payload = fetch(f"{_BINANCE_FUNDING_HISTORY}?symbol={symbol}&limit={_PAGE_LIMIT}")
    except (urllib.error.URLError, OSError, TimeoutError):
        return []
    return parse_binance_history(payload, venue, now_ns())


def backfill(store_root: Path, symbols: Sequence[str], venue: str = "binance",
             budget=None, fetch=_fetch, now_ns=time.time_ns,
             on_progress=None) -> dict:
    """Fetch every symbol's history and append it as one partition.

    One partition per run rather than per symbol: this is a single act of
    reconstruction at a single instant, and splitting it would suggest the parts
    were observed at different times, which is the impression the whole module
    exists to avoid.
    """
    from store.parquet_partition import append_partition

    started_ns = now_ns()
    rows: list[ReconstructedFunding] = []
    empty: list[str] = []
    for index, symbol in enumerate(symbols):
        fetched = fetch_binance_history(symbol, venue, budget=budget,
                                        fetch=fetch, now_ns=lambda: started_ns)
        if fetched:
            rows.extend(fetched)
        else:
            empty.append(symbol)
        if on_progress is not None:
            on_progress(index + 1, len(symbols), len(rows))

    frame = build_reconstructed_frame(rows)
    if frame.empty:
        return {"dataset": DATASET, "venue": venue, "symbols": len(symbols),
                "rows": 0, "empty_symbols": empty, "appended": False}

    # The snapshot id names the instant, so a second run is a second
    # reconstruction rather than a collision. Two reconstructions of the same
    # history are two observations of our own fetching, and both are true.
    snapshot_id = f"{DATASET}-{venue}-{started_ns}"
    appended = True
    try:
        append_partition(store_root, DATASET, frame, snapshot_id=snapshot_id)
    except FileExistsError:
        appended = False

    return {"dataset": DATASET, "venue": venue, "symbols": len(symbols),
            "rows": int(len(frame)), "empty_symbols": empty,
            "fetched_at_ns": started_ns, "appended": appended}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from ops.rate_budget import RateBudget
    from store.build_polled import polled_symbols

    parser = argparse.ArgumentParser(
        prog="funding-backfill",
        description="Fetch settled funding history into a dataset that cannot be "
                    "backtested - see the module docstring on why availability "
                    "time is the fetch, not the settlement.")
    parser.add_argument("--venue", default="binance", choices=["binance"],
                        help="only binance so far; it serves 166 days at weight 1")
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, or ALL for every symbol whose "
                             "funding this archive is already capturing")
    parser.add_argument("--date", default=None,
                        help="the captured day to read the ALL symbol list from; "
                             "defaults to today")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    args = parser.parse_args(argv)

    if args.symbols.strip().upper() == "ALL":
        date = args.date or time.strftime("%Y-%m-%d", time.gmtime())
        # The symbols we are already capturing live, so the reconstructed history
        # lines up with the observed record rather than covering a different set.
        symbols = polled_symbols(Path(args.capture_root), args.venue, date,
                                 "premiumIndex")
        if not symbols:
            print(f"no premiumIndex polls on disk for {args.venue} {date}; "
                  f"nothing to backfill against", file=sys.stderr)
            return 1
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    def progress(done, total, rows):
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total} symbols, {rows} rows", file=sys.stderr)

    print(f"fetching {len(symbols)} symbol(s) of settled funding from {args.venue}",
          file=sys.stderr)
    result = backfill(Path(args.store_root), symbols, args.venue,
                      budget=RateBudget(Path(args.capture_root) / "ops", args.venue),
                      on_progress=progress)
    print(json.dumps({k: v if k != "empty_symbols" else len(v)
                      for k, v in result.items()}))
    if result["empty_symbols"]:
        print(f"{len(result['empty_symbols'])} symbol(s) returned no history: "
              f"{sorted(result['empty_symbols'])[:10]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
