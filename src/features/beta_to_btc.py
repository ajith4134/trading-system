"""Beta and correlation to BTC, on returns that actually share a clock.

`FEATURES.md` §2 (P2): *"Correlation / beta to BTC"*. Ledger FE-018.

Crypto is a one-factor market most of the time. A symbol's move is mostly BTC's
move times a number, and the number is what this reports: `beta` for how much,
`correlation` for how reliably. Together they separate the two things a
position-sizing or hedging consumer needs to keep apart — an altcoin with beta
1.8 and correlation 0.9 is a leveraged BTC position, while beta 1.8 with
correlation 0.2 is an instrument that happens to have moved a lot, and sizing
them the same way is how a book ends up accidentally concentrated.

## Returns are joined on the clock, never zipped by position

The defect this module is built around: two symbols' bar series are different
lengths, because they list at different times, halt at different times, and one
of them is thin. Zipping them by position pairs BTC's 09:04 return with an
altcoin's 11:17 return and computes a correlation between two unrelated moments.
It produces a number in [-1, 1], it looks entirely normal, and it is noise.

So the returns are keyed by the **timestamp of the bar that closed them**, and
only timestamps present in both series contribute. `overlapping_returns` counts
what survived, and it rides every row, because a beta from 40 shared minutes and
a beta from 4,000 are different claims that print the same way.

## Only adjacent bars make a return, on both sides

A "return" spanning a five-day hole is a price ratio, not a return. It would
enter the covariance as one enormous observation and dominate every statistic
computed from it — the same rule `features.calendar_effects` applies for the same
reason, and it is applied here to the benchmark as well as to the symbol. A gap
in BTC's own tape must not be paired with a clean altcoin return.

## The benchmark is BTC on the SAME venue

`BENCHMARK_BY_VENUE` names it per venue, because the ticker differs — `BTCUSDT`
on Binance, `BTC` on Hyperliquid, `BTC-USD` on Coinbase — and a venue not on the
list is REFUSED rather than guessed at. Using another venue's BTC would import
that venue's basis, its outages and its own microstructure into every beta
measured here, and the contamination is largest exactly when venues diverge,
which is when a beta matters most.

## BTC's beta to itself is excluded, not reported as 1.0

It is 1.0 by construction, with correlation 1.0, and it is arithmetic performed
on a tautology rather than a measurement — the same refusal
`features.calendar_effects` makes for a venue that settles every hour. Reported,
it would sit in the table looking like the most reliably measured row there and
would be the first thing a consumer scanning for "beta near 1" found.

## A flat benchmark refuses; it does not divide

`beta = cov(symbol, benchmark) / var(benchmark)`. An hour in which BTC did not
move has zero variance in the denominator, and the answer is not a large beta but
no beta at all. Counted as `benchmark_did_not_move` so a quiet window is
distinguishable from a thin one.

## Correlation and beta are both reported, because neither implies the other

Beta without correlation cannot say whether the relationship is real, and
correlation without beta cannot size anything. A module reporting one is asking
its consumer to assume the other.

## Staleness (FE-001)

Stamped per (venue, symbol) against that key's whole visible bar history. A beta
is a statement about a relationship that was true over a window, and the window
having ended an hour ago is the thing the stamp exists to say.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from features.realized_volatility import BAR_INTERVAL_NS, decimal_close
from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"
_NS_PER_MINUTE = 60_000_000_000

# BTC's ticker on each venue. A venue absent from this map is refused, never
# guessed: picking the wrong symbol produces a beta to something that is not the
# market factor, and nothing downstream would notice.
BENCHMARK_BY_VENUE: dict[str, str] = {
    "binance": "BTCUSDT",
    "binance-spot": "BTCUSDT",
    "bybit": "BTCUSDT",
    "hyperliquid": "BTC",
    "coinbase": "BTC-USD",
}

# The windows this measures over. Two, declared, never searched - the same
# posture `features.cross_sectional` takes for the same reason.
HORIZONS_NS: dict[str, int] = {
    "1h": 60 * _NS_PER_MINUTE,
    "24h": 24 * 60 * _NS_PER_MINUTE,
}

# A beta from a handful of shared minutes is noise with two decimal places. 30
# is the smallest sample in which the covariance is a sum rather than a few
# terms; the count rides every row so a caller can hold out for more.
MIN_OVERLAPPING_RETURNS = 30

_COLUMNS = ("venue", "symbol", "benchmark", "horizon", "horizon_ns",
            "beta", "correlation", "overlapping_returns",
            "window_start_ns", "window_end_ns")

_REFUSAL_REASONS = ("unknown_benchmark_symbol", "benchmark_absent",
                    "benchmark_did_not_move", "symbol_did_not_move",
                    "too_few_overlapping_returns", "non_positive_close",
                    "unparseable_close")


@dataclass(frozen=True)
class BetaTable:
    """Beta and correlation to BTC per (venue, symbol, horizon).

    `refused` counts what did not produce a row, by reason. `benchmark_absent`
    and `too_few_overlapping_returns` are separate because they call for
    different work: one is a venue whose BTC feed is down, the other is a thin
    instrument.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def adjacent_log_returns(times: list[int], closes: list[Decimal],
                         ) -> dict[int, float] | str:
    """Log returns keyed by the timestamp of the bar that CLOSED them.

    Returns a refusal reason instead when a close cannot be trusted. Only
    adjacent bars produce a return - a ratio across a gap would enter every
    covariance as one enormous observation and dominate it.
    """
    returns: dict[int, float] = {}
    for i in range(1, len(closes)):
        if times[i] - times[i - 1] != BAR_INTERVAL_NS:
            continue
        returns[times[i]] = float((closes[i] / closes[i - 1]).ln())
    return returns


def _parse_closes(frame: pd.DataFrame) -> tuple[list[int], list[Decimal]] | str:
    closes: list[Decimal] = []
    for raw_close in frame["close"]:
        close = decimal_close(raw_close)
        if close is None:
            return "unparseable_close"
        if close <= 0:
            return "non_positive_close"
        closes.append(close)
    return [int(t) for t in frame["event_time_ns"]], closes


def beta_and_correlation(symbol_returns: list[float],
                         benchmark_returns: list[float],
                         ) -> tuple[float, float] | str:
    """Beta and Pearson correlation of one return series against the benchmark.

    Both series are already aligned by the caller - this function never sees
    timestamps, which is deliberate: an alignment bug should be impossible to
    introduce here, because there is nothing here to align.

    A zero-variance series on either side is a refusal, not a number. On the
    benchmark it is a division by zero; on the symbol it is a correlation that
    is undefined rather than zero, and returning 0.0 would report "moves
    independently of BTC" for an instrument that did not move at all.
    """
    n = len(symbol_returns)
    mean_symbol = sum(symbol_returns) / n
    mean_benchmark = sum(benchmark_returns) / n
    covariance = sum((s - mean_symbol) * (b - mean_benchmark)
                     for s, b in zip(symbol_returns, benchmark_returns)) / n
    variance_benchmark = sum((b - mean_benchmark) ** 2
                             for b in benchmark_returns) / n
    variance_symbol = sum((s - mean_symbol) ** 2 for s in symbol_returns) / n
    if variance_benchmark <= 0.0:
        return "benchmark_did_not_move"
    if variance_symbol <= 0.0:
        return "symbol_did_not_move"
    beta = covariance / variance_benchmark
    correlation = covariance / (variance_benchmark * variance_symbol) ** 0.5
    return beta, correlation


def compute_beta_to_btc(store_root: Path, as_of_ns: int,
                        custodian=None) -> BetaTable:
    """Beta and correlation to same-venue BTC per (venue, symbol, horizon).

    One clock-gated read; both horizons are views onto the same visible history.
    The benchmark's returns are extracted once per venue and looked up by
    timestamp, so a symbol contributes only the minutes BTC also printed.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return BetaTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for venue, venue_frame in frame.groupby("venue", sort=True):
        benchmark_symbol = BENCHMARK_BY_VENUE.get(venue)
        if benchmark_symbol is None:
            refused["unknown_benchmark_symbol"] += 1
            continue

        ordered = venue_frame.sort_values("event_time_ns")
        benchmark_frame = ordered[ordered["symbol"] == benchmark_symbol]
        if benchmark_frame.empty:
            refused["benchmark_absent"] += 1
            continue
        parsed = _parse_closes(benchmark_frame)
        if isinstance(parsed, str):
            refused[parsed] += 1
            continue
        benchmark_returns = adjacent_log_returns(*parsed)

        for horizon_name, horizon_ns in HORIZONS_NS.items():
            window_start = as_of_ns - horizon_ns
            window = ordered[ordered["event_time_ns"] > window_start]

            for symbol, symbol_frame in window.groupby("symbol", sort=True):
                if symbol == benchmark_symbol:
                    # Beta to itself is 1.0 by construction. Reported, it would
                    # look like the best-measured row in the table.
                    continue
                parsed_symbol = _parse_closes(symbol_frame)
                if isinstance(parsed_symbol, str):
                    refused[parsed_symbol] += 1
                    continue
                symbol_returns = adjacent_log_returns(*parsed_symbol)

                shared = [t for t in symbol_returns
                          if t in benchmark_returns and t > window_start]
                if len(shared) < MIN_OVERLAPPING_RETURNS:
                    refused["too_few_overlapping_returns"] += 1
                    continue

                measured = beta_and_correlation(
                    [symbol_returns[t] for t in shared],
                    [benchmark_returns[t] for t in shared])
                if isinstance(measured, str):
                    refused[measured] += 1
                    continue
                beta, correlation = measured

                out["venue"].append(venue)
                out["symbol"].append(symbol)
                out["benchmark"].append(benchmark_symbol)
                out["horizon"].append(horizon_name)
                out["horizon_ns"].append(horizon_ns)
                out["beta"].append(beta)
                out["correlation"].append(correlation)
                out["overlapping_returns"].append(len(shared))
                out["window_start_ns"].append(window_start)
                out["window_end_ns"].append(as_of_ns)

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return BetaTable(rows=stamp(rows, ages, ["venue", "symbol"]), refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
