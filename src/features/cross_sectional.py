"""Rank across the universe at one instant — the only feature here that is relative.

`FEATURES.md` §2 (P2): *"Cross-sectional ranking across pairs"*. Ledger FE-017.

Every other feature in this package answers a question about **one instrument
over time**: is this funding rate high for this symbol, is this hour volatile for
this symbol. This one answers the orthogonal question — *of everything trading
right now, where does this one sit* — and the two are not substitutes. A symbol
in the 90th percentile of its own volatility history during a market-wide
sell-off is unremarkable; the same reading while the rest of the universe is calm
is the entire signal. Only a cross-section can tell those apart.

## The universe is a fact about the answer, so it is carried on every row

A rank among four symbols and a rank among four hundred are different claims, and
they print identically as `0.75`. `universe_size` rides every row for the reason
`features.funding_basis` carries `observations`: a normalised number without its
sample is a number nobody can weight.

`MIN_UNIVERSE_SIZE` refuses the venue outright below its floor. A "cross-section"
over six symbols is a small sample with a percentile's name on it, and the
temptation it creates — reading the top-ranked of six as though it were the top
percentile of six hundred — is the mistake the rank was supposed to prevent.

## Ranked within a venue, never across venues

`BTCUSDT` on Binance and `BTC` on Hyperliquid are the same asset. Pooling them
would make the cross-section partly a measurement of the venues' own differences
— basis, fee structure, who is trading there — while presenting itself as a
measurement of the assets. Each venue is ranked as its own universe, and a
consumer comparing a symbol's rank across two venues is then comparing two
answers rather than reading one contaminated number.

## Survivorship is handled by requiring both ends of the return, not by a filter

The classic cross-sectional bias is ranking history against the universe that
*still exists today*. The clock-gated reader closes half of it — a past clock
sees only rows available by then — but not the other half: a delisted symbol's
old rows stay visible forever, so nothing stops it appearing in **today's**
cross-section on three-day-old prices.

What closes it is the window requirement. A symbol enters the cross-section only
if it has a usable close **at both ends** of the horizon window, so an instrument
that stopped printing drops out of every window that ends after it died, and
stays in every window it actually traded through. That is the correct behaviour
in both directions, and it is a consequence of how the return is defined rather
than a survivorship filter bolted on — a filter would be one more thing to keep
in agreement with the reader.

`incomplete_window` counts what dropped out, so a universe that quietly halved is
visible rather than being read as a market where nothing moved.

## Two ranked quantities, and volume is the one with the caveat

`return_rank` ranks trailing log return over the horizon — cross-sectional
momentum, the family with the longest-standing literature outside this corpus.

`volume_rank` ranks traded volume over the same window, and it is reported with a
named limitation rather than as a clean measurement: reported volume on crypto
venues is inflated by wash trading, which `features.volume_quality` exists to
discount and this module does not apply. Ranking raw volume is defensible where
ranking raw volume *levels* would not be — wash trading inflates a venue's
symbols unevenly, so the rank is contaminated too, just less than the level is.
It is carried because volume rank is a standard cross-sectional conditioner and
suppressing it would be worse than shipping it labelled.

## Midranks, computed once for the whole vector

Ties take the average of the positions they span, so twenty symbols with
identical volume all report the middle of that block rather than an ordering
invented by whatever the sort was stable on. Computed for the whole universe in
one pass rather than by asking each symbol where it sits, which would be
quadratic in a universe of several hundred and would compute the same sort
several hundred times.

## Staleness (FE-001)

Stamped per (venue, symbol) against that key's whole visible bar history. The
window requirement already guarantees a recent print, so this stamp is rarely the
thing that fires — but a key can clear the window and still be the stalest thing
in the universe, and a rank against a universe of fresher instruments is a
comparison nobody should make silently.
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

# The horizons this ranks over. Two, not five: a cross-section costs a full
# universe sort per horizon, and the two here are the ones a consumer of a
# relative signal actually separates - "moving now" against "moved today".
# Declared, never searched, for the reason `har_rv` declares its cascade.
HORIZONS_NS: dict[str, int] = {
    "1h": 60 * _NS_PER_MINUTE,
    "24h": 24 * 60 * _NS_PER_MINUTE,
}

# Below this, a percentile is a small sample wearing a percentile's name, and it
# invites reading the top of six as though it were the top percentile of six
# hundred.
MIN_UNIVERSE_SIZE = 20

_COLUMNS = ("venue", "symbol", "horizon", "horizon_ns", "universe_size",
            "trailing_log_return", "return_rank", "window_volume",
            "volume_rank", "window_start_ns", "window_end_ns")

_REFUSAL_REASONS = ("universe_too_small", "incomplete_window",
                    "non_positive_close", "unparseable_close")


@dataclass(frozen=True)
class CrossSectionalTable:
    """Where each symbol sits in its venue's universe, per horizon.

    `refused` counts symbols dropped from a window and venues refused whole. The
    distinction matters and both are here: a universe that quietly halved
    produces the same-looking table as a market where nothing moved.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def midranks(values: list[float]) -> list[float]:
    """Rank each value in [0, 1], averaging the positions that ties span.

    Returned in the input's order. The midrank is what makes a block of
    identical values report one shared position rather than an ordering invented
    by whatever the sort happened to be stable on - and identical values are
    common here, since a quiet symbol's volume and return are frequently exactly
    the same as its neighbours'.

    A single-element universe would divide by zero; the caller's
    `MIN_UNIVERSE_SIZE` floor makes that unreachable, and this asserts nothing
    about it rather than returning a number for a rank that has no meaning.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (end + 1 < len(order)
               and values[order[end + 1]] == values[order[position]]):
            end += 1
        # Positions `position..end` inclusive share a value: they share a rank.
        shared = (position + end) / 2
        for index in range(position, end + 1):
            ranks[order[index]] = shared / (len(values) - 1)
        position = end + 1
    return ranks


@dataclass(frozen=True)
class _WindowObservation:
    """One symbol's usable measurements over one horizon window."""
    symbol: str
    log_return: float
    volume: float


def _window_observation(symbol: str, window: pd.DataFrame,
                        ) -> _WindowObservation | str:
    """A symbol's return and volume over the window, or a refusal reason.

    Both ends are required. A return computed from one close is not a return,
    and a symbol contributing only its newest print would enter the cross-section
    with a zero return - ranking it in the middle of the universe rather than
    leaving it out, which is the flattering direction for anything selecting on
    extremes.
    """
    if len(window) < 2:
        return "incomplete_window"
    first = decimal_close(window["close"].iloc[0])
    last = decimal_close(window["close"].iloc[-1])
    if first is None or last is None:
        return "unparseable_close"
    if first <= 0 or last <= 0:
        return "non_positive_close"
    volume = float(pd.to_numeric(window["volume"], errors="coerce").fillna(0).sum())
    return _WindowObservation(symbol=symbol,
                              log_return=float((last / first).ln()),
                              volume=volume)


def compute_cross_sectional(store_root: Path, as_of_ns: int,
                            custodian=None) -> CrossSectionalTable:
    """Cross-sectional ranks per (venue, symbol, horizon) at this clock.

    One clock-gated read of the bar dataset; both horizons are views onto the
    same visible history, and the gate is the expensive part. The universe is
    whatever cleared the window at this clock - never a symbol list held
    somewhere, which would be a second thing to keep in agreement with the store
    and would go stale in the direction that keeps dead instruments in.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return CrossSectionalTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for venue, venue_frame in frame.groupby("venue", sort=True):
        ordered = venue_frame.sort_values("event_time_ns")
        for horizon_name, horizon_ns in HORIZONS_NS.items():
            window_start = as_of_ns - horizon_ns
            window = ordered[ordered["event_time_ns"] > window_start]

            observations: list[_WindowObservation] = []
            for symbol, symbol_window in window.groupby("symbol", sort=True):
                result = _window_observation(symbol, symbol_window)
                if isinstance(result, str):
                    refused[result] += 1
                    continue
                observations.append(result)

            if len(observations) < MIN_UNIVERSE_SIZE:
                # Refused whole. Every symbol that DID clear the window is
                # counted, so the tally reflects what was lost rather than
                # reporting one refusal for a venue of four hundred.
                refused["universe_too_small"] += len(observations)
                continue

            return_ranks = midranks([o.log_return for o in observations])
            volume_ranks = midranks([o.volume for o in observations])
            for observation, return_rank, volume_rank in zip(
                    observations, return_ranks, volume_ranks):
                out["venue"].append(venue)
                out["symbol"].append(observation.symbol)
                out["horizon"].append(horizon_name)
                out["horizon_ns"].append(horizon_ns)
                out["universe_size"].append(len(observations))
                out["trailing_log_return"].append(observation.log_return)
                out["return_rank"].append(return_rank)
                out["window_volume"].append(observation.volume)
                out["volume_rank"].append(volume_rank)
                out["window_start_ns"].append(window_start)
                out["window_end_ns"].append(as_of_ns)

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return CrossSectionalTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                               refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
