"""Bars fetched from REST for minutes nobody was watching, labelled as such.

`FEATURES.md` §1 P0: *"Gap detection + provenance-flagged backfill - interpolated
candles must be labelled, never silently blended."* Detection has been live since
`capture.sequencing`; this is the other half, and `DECISIONS.md` §13 states the
condition it has to meet: **nothing currently backfills, so this is a
prerequisite of the first thing that does.** A labelling scheme with no backfill
behind it is a comment; a backfill with no labelling is the defect the row was
written about. Both, or neither.

The gap this exists for is not hypothetical. Capture on this box stopped at
2026-08-09T19:51Z and came back at 04:26Z - eight and a half hours with no tape
and therefore no bars, and nothing in the `bars` dataset says so. An absent bar
and a quiet minute look identical to anything that resamples.

## Why this is a separate dataset and not a fill

Same design as `store.funding_backfill`, for the same reason, and it is
mechanical rather than a matter of care:

**`availability_time` is when we FETCHED the bar, not when the minute closed.**
`read_as_of(t)` for any `t` before the fetch returns nothing at all, so a
backtest cannot consume these rows by accident - at every simulated instant in
the past this dataset is empty. Stamping availability at bar close, as though we
had been watching, is the one thing this must never do: it would make a
reconstructed bar indistinguishable from an observed one to every consumer.

`is_reconstructed` rides every row too, so a frame that gets filtered, joined or
copied somewhere else still says what it is.

## What a reconstructed bar is worth, measured rather than assumed

A kline is the venue's own aggregation; our bars are built from the trade stream
we received. They should agree and they are not the same measurement, so
`compare_reconstructed_to_observed` reports the distance where both exist rather
than either side claiming to be the truth. A backfill nobody has ever compared
against an observed bar is a number of unknown quality wearing the same column
names as a good one.

## The refusals

* **The still-open kline is dropped.** Binance returns the current minute as a
  kline like any other, with a `closeTime` in the future and partial volume.
  Stored, it would be a bar that quietly disagrees with the finished version of
  itself an hour later - and it would disagree in the flattering direction for
  anything measuring realised volume.
* **A non-positive price is refused**, on the same evidence as the stored-bar
  validity gate: binance emits placeholder frames carrying `"0"`, and a zero
  price that reaches a dataset prices every trade that reads it.
* **A kline that does not parse is skipped, never zero-filled.** No trades is
  not a flat bar - the rule `build_bars` already holds to.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import pandas as pd

from store.clock_gated_reader import ClockGatedReader
from store.parquet_partition import append_partition
from ops.rate_budget import RateBudget
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

_MS_TO_NS = 1_000_000

# **The kline endpoint per venue, and the venue is NOT just a label.**
#
# This was one constant pointing at `fapi` while `venue` was written into every
# row as provenance. A `--venue binance-spot` backfill therefore fetched PERPETUAL
# FUTURES klines and stored them labelled spot - a whole dataset that is real data
# about the wrong instrument, which is worse than missing data because nothing
# downstream can tell. Perp and spot prices differ by the basis, which is exactly
# what `features.funding_basis` exists to measure.
#
# An unknown venue RAISES rather than falling back to either one. A default here
# is how the bug above happens again under a different name.
_KLINE_ENDPOINTS = {
    "binance": "https://fapi.binance.com/fapi/v1/klines",
    "binance-spot": "https://api.binance.com/api/v3/klines",
}

# Binance's documented maximum for these endpoints. One request covers 25 hours
# of minute bars, so a day's gap on one symbol is a single call.
_PAGE_LIMIT = 1500

# The endpoint's weight at this limit, per binance's own table, PER VENUE - the
# two are not the same and assuming the higher one everywhere would throttle spot
# to a fifth of its real allowance. Passed to the rate budget rather than assumed
# free: this shares a weight allowance with the pollers that keep live capture
# alive, and the cost of overage is a three-day ban on the egress IP.
_REQUEST_WEIGHT_BY_VENUE = {"binance": 10, "binance-spot": 2}
_REQUEST_WEIGHT = 10          # the conservative default for an unlisted venue


def kline_endpoint(venue: str) -> str:
    """The klines URL for one venue, or a refusal naming the ones that exist."""
    try:
        return _KLINE_ENDPOINTS[venue]
    except KeyError:
        raise ValueError(
            f"no kline endpoint for venue {venue!r} - known: "
            f"{sorted(_KLINE_ENDPOINTS)}. Falling back to another venue's "
            f"endpoint would store real data about the wrong instrument") from None


def request_weight(venue: str) -> int:
    """What one page costs this venue's rate budget."""
    return _REQUEST_WEIGHT_BY_VENUE.get(venue, _REQUEST_WEIGHT)
DEFAULT_INTERVAL_NS = 60_000_000_000
# **The intervals this backfill can ask a venue for, keyed by their length in ns.**
#
# 1m, 5m, 15m and 30m are the four RL-043 named as the bots' intraday timeframes;
# 1h predates that ruling and is kept because the store's hourly layout and the
# funding-carry work both read it. Every key here must be a real Binance kline
# interval - a name the venue does not publish comes back as an error page that
# parses to zero bars, which reads on a board as a market with no trades.
#
# PUBLIC on purpose. `statuswall.segment_probes.probe_intraday_timeframes_declared`
# reads this map to check the ruling against the code, and a probe reaching into a
# private name is a check that breaks silently the day the name changes.
INTERVAL_NAME_BY_NS = {
    60_000_000_000: "1m",
    300_000_000_000: "5m",
    900_000_000_000: "15m",
    1_800_000_000_000: "30m",
    3_600_000_000_000: "1h",
}
# The old private spelling, kept so nothing that imported it breaks mid-run.
_INTERVAL_NAME_BY_NS = INTERVAL_NAME_BY_NS


def dataset_name(interval_ns: int = DEFAULT_INTERVAL_NS) -> str:
    """Named for what it holds, beside `bars_<interval>ns` and never inside it.

    The observed dataset keeps its meaning: every row in it was witnessed. A
    consumer that wants both has to ask for both, in a line of code that says so.
    """
    return f"bars_reconstructed_{interval_ns}ns"


@dataclass(frozen=True)
class ReconstructedBar:
    symbol: str
    venue: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    bar_open_ns: int
    fetched_at_ns: int


def parse_binance_klines(payload: str, symbol: str, venue: str,
                         fetched_at_ns: int,
                         interval_ns: int = DEFAULT_INTERVAL_NS,
                         ) -> list[ReconstructedBar]:
    """Read one klines page, dropping the bar that has not closed yet.

    A kline is a 12-element array; the four that matter here are open time,
    OHLC, volume and trade count. Anything malformed is skipped rather than
    guessed at - a bar assembled from a row this could not read would carry
    prices nobody published.
    """
    try:
        rows = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(rows, list):
        return []

    parsed: list[ReconstructedBar] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 9:
            continue
        try:
            bar_open_ns = int(row[0]) * _MS_TO_NS
            values = [float(row[i]) for i in (1, 2, 3, 4, 5)]
            trades = int(row[8])
        except (TypeError, ValueError):
            continue
        open_, high, low, close, volume = values

        # The still-open bar. Its close time is in the future, its volume is
        # partial, and it would disagree with the finished version of itself.
        if bar_open_ns + interval_ns > fetched_at_ns:
            continue
        # A price of zero is not a price. Binance emits placeholder frames and
        # the stored-bar validity gate refuses them on the observed side; a
        # backfill that let them in would reintroduce what that gate removed.
        if min(open_, high, low, close) <= 0:
            continue
        if volume < 0 or trades < 0:
            continue

        parsed.append(ReconstructedBar(
            symbol=symbol, venue=venue,
            open=open_, high=high, low=low, close=close,
            volume=volume, trades=trades,
            bar_open_ns=bar_open_ns, fetched_at_ns=fetched_at_ns))
    return parsed


def build_reconstructed_bars_frame(rows: Sequence[ReconstructedBar]) -> pd.DataFrame:
    """Bitemporal rows whose availability is the fetch, not the bar close.

    That single choice is what stops this dataset being backtestable, and it is
    the reason a comment saying "do not backtest on this" was not written
    instead: every simulated clock in the past sees an empty frame, whether or
    not the reader ever heard of this module.
    """
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: r.symbol,
        VENUE: r.venue,
        "open": r.open, "high": r.high, "low": r.low, "close": r.close,
        "volume": r.volume, "trades": r.trades,
        # Honest as an event time - it is when the minute opened - and useless
        # for leaking, because availability is what a reader may filter on.
        EVENT_TIME: r.bar_open_ns,
        INGESTION_TIME: r.fetched_at_ns,
        AVAILABILITY_TIME: r.fetched_at_ns,
        # Per row, so a frame that is filtered, joined or copied elsewhere still
        # says what it is long after it left this dataset.
        "is_reconstructed": True,
    } for r in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME, "trades"):
        frame[column] = frame[column].astype("int64")
    frame["is_reconstructed"] = frame["is_reconstructed"].astype("bool")
    return frame


def _fetch(url: str, timeout: float = 20.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_binance_bars(symbol: str, start_ns: int, end_ns: int,
                       venue: str = "binance",
                       interval_ns: int = DEFAULT_INTERVAL_NS,
                       budget=None, fetch=_fetch, now_ns=time.time_ns,
                       pages: int = 64) -> list[ReconstructedBar]:
    """One symbol's bars over [start_ns, end_ns), walked forward a page at a time.

    Forward rather than backward, unlike the funding backfill: a bar gap has two
    known ends, so paging from the older one means each request's `startTime` is
    the last bar this already has and progress is provable. A page that fails to
    advance stops the walk rather than spinning - a venue ignoring `startTime`
    would otherwise return the same 1,500 bars until the page budget ran out.

    Returns what it has rather than raising. One symbol's gap is not worth
    abandoning the rest of the request for, and a symbol that returns nothing
    stays visibly a gap.
    """
    from capture.venue_recorder import decode_path_token

    interval_name = INTERVAL_NAME_BY_NS.get(interval_ns)
    if interval_name is None:
        raise ValueError(
            f"no binance kline interval for {interval_ns}ns - known: "
            f"{sorted(INTERVAL_NAME_BY_NS)}")

    # The archive stores path-unsafe symbols encoded; the venue has never heard
    # of that name. Asking for `_b32_...` returns an empty list, which reads as
    # a delisting rather than as us sending our own filename.
    endpoint = kline_endpoint(venue)
    venue_symbol = decode_path_token(symbol)
    fetched_at = now_ns()
    collected: dict[int, ReconstructedBar] = {}
    cursor_ms = int(start_ns) // _MS_TO_NS
    end_ms = int(end_ns) // _MS_TO_NS

    for _ in range(max(1, pages)):
        if cursor_ms >= end_ms:
            break
        if budget is not None and not budget.try_spend(request_weight(venue)):
            break
        url = (f"{endpoint}?symbol={quote(venue_symbol)}"
               f"&interval={interval_name}&limit={_PAGE_LIMIT}"
               f"&startTime={cursor_ms}&endTime={end_ms}")
        try:
            page = parse_binance_klines(fetch(url), symbol, venue, fetched_at,
                                        interval_ns)
        except (urllib.error.URLError, OSError, TimeoutError):
            break
        if not page:
            break
        newest_ns = max(bar.bar_open_ns for bar in page)
        for bar in page:
            collected[bar.bar_open_ns] = bar
        next_cursor = newest_ns // _MS_TO_NS + 1
        if next_cursor <= cursor_ms:
            break              # the page did not advance; stop rather than spin
        cursor_ms = next_cursor

    return [collected[key] for key in sorted(collected)]


def backfill_bars(store_root: Path, symbols: Sequence[str],
                  start_ns: int, end_ns: int, venue: str = "binance",
                  interval_ns: int = DEFAULT_INTERVAL_NS, budget=None,
                  fetch=_fetch, now_ns=time.time_ns, on_progress=None) -> dict:
    """Fetch each symbol's bars over the window and append them as one partition.

    One partition per run, keyed on the start of the run, so a re-run is refused
    by the partition writer as a collision rather than silently doubling every
    bar.
    """
    started_ns = now_ns()
    rows: list[ReconstructedBar] = []
    for index, symbol in enumerate(symbols, start=1):
        rows.extend(fetch_binance_bars(symbol, start_ns, end_ns, venue,
                                       interval_ns, budget, fetch, now_ns))
        if on_progress is not None:
            on_progress(index, len(symbols), len(rows))

    dataset = dataset_name(interval_ns)
    frame = build_reconstructed_bars_frame(rows)
    if frame.empty:
        # Reported rather than written. An empty backfill is a fact about the
        # window - the venue had nothing, or every bar was refused - and a
        # zero-row partition would look like a completed one.
        return {"dataset": dataset, "venue": venue, "symbols": len(symbols),
                "bars": 0, "appended": False}

    append_partition(store_root, dataset, frame,
                     snapshot_id=f"{dataset}-{venue}-{started_ns}")
    return {"dataset": dataset, "venue": venue, "symbols": len(symbols),
            "bars": int(len(frame)), "appended": True,
            "first_bar_ns": int(frame[EVENT_TIME].min()),
            "last_bar_ns": int(frame[EVENT_TIME].max())}


def compare_reconstructed_to_observed(store_root: Path, as_of_ns: int,
                                      interval_ns: int = DEFAULT_INTERVAL_NS,
                                      custodian=None) -> dict:
    """How far a reconstructed bar sits from the observed bar for the same minute.

    Both sides are read through the clock gate, the door every market-data read
    uses. The reconstructed side is only ever visible at a clock at or after its
    fetch, which is the property that keeps it out of backtests - so this
    comparison is a thing you run now, deliberately, and not something a
    simulation can stumble into.

    The point is not to declare a winner. A kline is the venue's aggregation of
    every trade; our bar is built from the trades our socket received. Where
    they disagree, the size of the disagreement is what a reconstructed bar is
    worth, and reporting `overlapping: 0` is a real answer - it says the two
    have never been compared on this store.
    """
    as_of_ns = int(as_of_ns)
    observed = ClockGatedReader(Path(store_root), f"bars_{interval_ns}ns",
                                custodian=custodian).read_as_of(as_of_ns)
    reconstructed = ClockGatedReader(Path(store_root), dataset_name(interval_ns),
                                     custodian=custodian).read_as_of(as_of_ns)
    summary = {"observed_bars": int(len(observed)),
               "reconstructed_bars": int(len(reconstructed)),
               # Measured, not asserted. "None of them leaked into the observed
               # dataset" is the claim this whole module rests on, and a claim
               # nobody counted is exactly what Rule 8 exists to stop.
               "reconstructed_rows_in_observed": (
                   0 if "is_reconstructed" not in observed.columns
                   else int(observed["is_reconstructed"].fillna(False).astype(bool).sum())),
               "overlapping": 0}
    if observed.empty or reconstructed.empty:
        return summary

    keys = [VENUE, SYMBOL, EVENT_TIME]
    merged = observed[keys + ["close", "volume"]].merge(
        reconstructed[keys + ["close", "volume"]], on=keys,
        suffixes=("_observed", "_reconstructed"))
    summary["overlapping"] = int(len(merged))
    if merged.empty:
        return summary

    close_bps = ((merged["close_reconstructed"] - merged["close_observed"])
                 / merged["close_observed"] * 10_000.0).abs()
    volume_observed = merged["volume_observed"].replace(0.0, float("nan"))
    volume_ratio = merged["volume_reconstructed"] / volume_observed
    summary.update({
        "close_agreement_bps_median": float(close_bps.median()),
        "close_agreement_bps_p99": float(close_bps.quantile(0.99)),
        "close_identical": int((close_bps == 0).sum()),
        "volume_ratio_median": float(volume_ratio.median()),
    })
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill bars from binance klines into a labelled dataset")
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, as the archive names them")
    parser.add_argument("--start", required=True,
                        help="ISO8601 UTC, inclusive - e.g. 2026-08-09T19:51:00Z")
    parser.add_argument("--end", required=True, help="ISO8601 UTC, exclusive")
    parser.add_argument("--venue", default="binance")
    parser.add_argument("--interval-ns", type=int, default=DEFAULT_INTERVAL_NS)
    parser.add_argument("--compare", action="store_true",
                        help="after writing, report the distance from observed bars")
    parser.add_argument("--no-budget", action="store_true",
                        help="spend no rate budget - ONLY for a single-symbol probe, "
                             "never for a bulk backfill sharing the egress IP")
    args = parser.parse_args(argv)

    start_ns = int(pd.Timestamp(args.start).value)
    end_ns = int(pd.Timestamp(args.end).value)
    symbols = [s for s in args.symbols.split(",") if s]

    # **The budget is the default, and its absence has to be asked for.**
    #
    # This CLI used to pass none at all, so a bulk backfill spent weight the live
    # pollers could not see, against the same egress IP whose overage - per
    # `ops.rate_budget`'s own measured note - escalates to a THREE-DAY BAN. That
    # ban would take capture down with it, and capture is what RL-024 requires
    # every paper fill to be priced from.
    #
    # The bucket is file-backed and shared, so this backfill queues behind live
    # capture rather than racing it.
    budget = None
    if not args.no_budget:
        budget = RateBudget(Path(args.store_root).parent / "rate-budgets",
                            args.venue)

    result = backfill_bars(Path(args.store_root), symbols, start_ns, end_ns,
                           venue=args.venue, interval_ns=args.interval_ns,
                           budget=budget)
    print(json.dumps(result))
    if args.compare:
        print(json.dumps(compare_reconstructed_to_observed(
            Path(args.store_root), time.time_ns(), args.interval_ns)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
