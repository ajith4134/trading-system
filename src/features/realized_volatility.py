"""Realized volatility, five horizons deep, from the 1-minute bar tape.

`FEATURES.md` §2 (P0). Every downstream consumer that sizes a position, sets
a stop, or prices an option needs a volatility number, and every one of them
needs a DIFFERENT one: a market maker quoting the next five minutes cares
about the last five minutes, not yesterday. A single window is a choice
disguised as a measurement - it silently answers "volatile over what
horizon?" with whatever the author happened to pick, and every consumer
inherits that pick whether it fits them or not. This module computes the
same quantity - realized volatility from log returns of bar closes, per
(venue, symbol) - over several windows at once (`HORIZONS_NS`) and reports
each independently, so the choice of horizon moves to the caller, where it
belongs.

## What is measured, precisely

Over a window, the log returns of consecutive available closes are
`r_i = ln(close_i / close_{i-1})`. Two figures are reported per horizon:

* `realized_vol` - `sqrt(sum(r_i ** 2))`, the quadratic-variation estimator
  of the volatility actually realized over that exact window. It is a
  horizon-native number: a 5-minute `realized_vol` and a 24-hour one are not
  comparable to each other, because the second is a sum over roughly 288
  times as many terms and only grows with more data.
* `annualised_vol` - `sqrt(mean(r_i ** 2) * periods_per_year)`, the same
  quadratic variation expressed as a PER-BAR rate and then scaled to a
  year. Unlike `realized_vol` this figure IS comparable across horizons,
  because averaging removes the dependence on how many returns went in -
  which is why the test that checks a short horizon reacts to a recent
  shock while a long one does not compares this figure, not the raw one.

## Annualisation is derived from the bar interval, never sqrt(252)

`_DATASET = "bars_60000000000ns"` names its own contract: bars are
`BAR_INTERVAL_NS` nanoseconds wide, and the number appears in the dataset
name so a mismatch between the two would be a visible constant, not a typo
waiting to be found. `PERIODS_PER_YEAR` is `365 days-in-ns // BAR_INTERVAL_NS`
- an exact integer division, 525,600 one-minute bars in a year, checked as
exact in the tests rather than assumed. 365 days, not 252 trading days:
crypto has no exchange holiday and no weekend close, so a trading-day
convention borrowed from equities would understate every annualised figure
here by roughly 30% for no reason grounded in this market - the same
reasoning `features.term_structure` already applies to a dated contract's
calendar tenor.

## The minimum observation count, and why it is not one number

A realized-vol estimate from a handful of returns is not an estimate, it is
noise wearing a measurement's clothes: with one return, `realized_vol` is
just `|r|`, a single draw passed through unchanged, and it would move
however violently that one bar happened to move regardless of what the
instrument actually does. `MIN_REQUIRED_RETURNS` requires at least HALF of
what an unbroken tape would offer for that horizon (`expected_returns // 2`),
floored at 2 - the smallest sample in which a sum of squares is actually a
sum rather than one term. The half-of-expected rule scales the floor with
the horizon itself: `peg_monitor` and `volume_quality` each pick one fixed
minimum for their one fixed window, and holding a 5-minute horizon to a
24-hour horizon's bar count here would mean the short horizons never fire at
all. A window that cannot meet its own floor is REFUSED for that
(venue, symbol, horizon) alone - the other horizons for the same key are
judged on their own data and are not held back by one horizon's gap.

## Zero and negative closes are refused, never divided by

A documented past defect in this store: placeholder frames carry price
`"0"` (`store.trade_bars.is_tradeable` catches this at bar-build time for
NEW bars, but the store also holds bars built before that guard existed).
`ln` of a non-positive close is undefined or a negative infinity wearing a
number's shape, and either one poisons every return computed across it. On
finding one, the WHOLE window is refused (`non_positive_close`) rather than
silently recomputed with the bad bar dropped - a horizon whose bar count
quietly shrank around a defect would still report a number, and that number
would carry no sign of what it was missing. The same treatment applies to a
close that cannot be parsed at all (`unparseable_close`): a NaN closing a
bar is data corruption dressed as an ordinary value, not a smaller sample.

## Prices are Decimal, and here specifically that matters

House convention: prices and bps are `Decimal`, never `float`. Two sibling
modules over this same bars dataset (`peg_monitor`, `volume_quality`) treat
`close` as float because they only ever compare or average it. This module
takes a natural log of a ratio of two closes and squares the result before
summing hundreds of terms - the kind of computation where float's binary
representation of a decimal price compounds fastest. `Decimal(str(close))`
before any arithmetic, `Decimal.ln()` and `Decimal.sqrt()` throughout, so
the numbers this module reports are as precise as the store's own decimal
prices, not as precise as IEEE-754 happens to allow.

## Staleness (FE-001)

Every row is stamped via `features.staleness`, keyed on (venue, symbol) and
measured against the WHOLE visible bar history for that key - not just the
bars inside one horizon's window - because staleness answers "how old is the
newest thing this venue has told us", and that question does not change
per horizon. A venue whose feed died an hour ago can still produce a valid
24-hour realized-vol row (the window reaches back far enough to find enough
returns) while its 5-minute row correctly refuses for lack of data - and the
staleness stamp on that surviving 24-hour row is what tells a caller the
newest input behind it is an hour stale, which the refusal accounting alone
would not show.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# One minute, in nanoseconds - the number the dataset name above already
# encodes. Repeated here as an int rather than parsed from the string so a
# typo in the dataset name shows up as two mismatched constants instead of a
# silent width change nothing checks.
BAR_INTERVAL_NS = 60_000_000_000

_NS_PER_MINUTE = 60_000_000_000

# The five lookback windows this module reports, nanoseconds out from the
# as-of clock. Ordered short to long because every table this module
# produces is read that way - a caller scanning for "did anything change
# recently" reads top to bottom.
HORIZONS_NS: dict[str, int] = {
    "5m": 5 * _NS_PER_MINUTE,
    "15m": 15 * _NS_PER_MINUTE,
    "1h": 60 * _NS_PER_MINUTE,
    "4h": 4 * 60 * _NS_PER_MINUTE,
    "24h": 24 * 60 * _NS_PER_MINUTE,
}

# 365 days, not 252 trading days - see the module docstring. The division is
# exact (checked in tests): 365 * 24 * 60 one-minute bars in a year.
_YEAR_NS = 365 * 24 * 3600 * 1_000_000_000
PERIODS_PER_YEAR = Decimal(_YEAR_NS // BAR_INTERVAL_NS)


def _minimum_required_returns(horizon_ns: int) -> int:
    """At least half of an unbroken tape's returns for this horizon, floored
    at 2. See the module docstring for why neither half alone is the rule."""
    expected_bars = horizon_ns // BAR_INTERVAL_NS
    expected_returns = max(expected_bars - 1, 1)
    return max(2, expected_returns // 2)


MIN_REQUIRED_RETURNS: dict[str, int] = {
    name: _minimum_required_returns(horizon_ns)
    for name, horizon_ns in HORIZONS_NS.items()
}

_COLUMNS = ("venue", "symbol", "horizon", "horizon_ns", "observations",
            "realized_vol", "annualised_vol", "window_start_ns", "window_end_ns")


@dataclass(frozen=True)
class RealizedVolatilityTable:
    """Realized volatility per (venue, symbol, horizon), and what was refused.

    `rows` carries up to `len(HORIZONS_NS)` rows per (venue, symbol) - one per
    horizon that had enough clean data, never fewer than a horizon's own
    floor and never a value interpolated for one that did not. `refused`
    counts every (venue, symbol, horizon) that could not be priced, by
    reason - visible, because a table with three of five horizons silently
    missing reads identically to a symbol that is simply quiet.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def decimal_close(value) -> Decimal | None:
    """The row's close as a Decimal, or None if it cannot be trusted as one.

    `str(value)` before `Decimal(...)`, matching `spot_perp_basis` and
    `term_structure`: constructing straight from a float would import that
    float's own binary rounding as if it were a digit of the stored price.

    Public because `features.har_rv` reads the same bars with the same defect
    history and must parse them the same way. A private copy there would be a
    second place for the placeholder-price rule to live, and the copy that drifts
    is always the one nobody re-derived.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def compute_realized_volatility(store_root: Path, as_of_ns: int,
                                custodian=None) -> RealizedVolatilityTable:
    """Realized volatility per (venue, symbol), at every horizon it can support.

    Reads the 1-minute bar dataset once through `ClockGatedReader`, so a
    backtest asking for this as of a past clock sees exactly the bars that
    had closed and arrived by then - and computes all five horizons from
    that one read, because the clock gate is the expensive part and the
    horizons are all views onto the same visible history.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {"too_few_observations": 0, "non_positive_close": 0,
               "unparseable_close": 0}
    if frame.empty:
        return RealizedVolatilityTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        ordered = group.sort_values("event_time_ns")
        for horizon_name, horizon_ns in HORIZONS_NS.items():
            window_start = as_of_ns - horizon_ns
            # Strictly after the window start: a lookback window is
            # half-open on its old end, so the boundary bar of a shorter
            # horizon is never double-counted against the longer one that
            # starts at the very same instant.
            window = ordered[ordered["event_time_ns"] > window_start]

            closes: list[Decimal] = []
            refusal_reason = None
            for raw_close in window["close"]:
                close = decimal_close(raw_close)
                if close is None:
                    refusal_reason = "unparseable_close"
                    break
                if close <= 0:
                    # A zero or negative close is the placeholder-price
                    # defect wearing a bar's clothes. ln() of it is either
                    # undefined or -inf, and dropping just this bar would
                    # silently shrink the window around the defect instead
                    # of refusing to have measured it at all.
                    refusal_reason = "non_positive_close"
                    break
                closes.append(close)

            if refusal_reason is not None:
                refused[refusal_reason] += 1
                continue

            returns = [(closes[i] / closes[i - 1]).ln()
                      for i in range(1, len(closes))]
            if len(returns) < MIN_REQUIRED_RETURNS[horizon_name]:
                refused["too_few_observations"] += 1
                continue

            sum_sq = sum((r * r for r in returns), start=Decimal(0))
            realized_vol = sum_sq.sqrt()
            mean_sq = sum_sq / Decimal(len(returns))
            annualised_vol = (mean_sq * PERIODS_PER_YEAR).sqrt()

            out["venue"].append(venue)
            out["symbol"].append(symbol)
            out["horizon"].append(horizon_name)
            out["horizon_ns"].append(horizon_ns)
            out["observations"].append(len(returns))
            out["realized_vol"].append(realized_vol)
            out["annualised_vol"].append(annualised_vol)
            out["window_start_ns"].append(window_start)
            out["window_end_ns"].append(as_of_ns)

    rows = pd.DataFrame(out)
    # FE-001: staleness is measured against the WHOLE visible history per
    # key, not per horizon - see the module docstring for why a dying feed's
    # 24h row can still be valid data with a stale stamp on it.
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return RealizedVolatilityTable(rows=stamp(rows, ages, ["venue", "symbol"]),
                                   refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
