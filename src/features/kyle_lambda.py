"""Kyle's lambda: the price-impact coefficient, and the honest limits of
measuring it here.

`FEATURES.md` §2 (P2) is one line: **"Liquidity-regime descriptor for sizing,
not a standalone signal."** Kyle (1985) models a market maker who cannot see
whether incoming flow is informed and prices to survive adverse selection
anyway: price moves `lambda` per unit of net signed order flow, and `lambda`
is large exactly where the book cannot absorb size without moving the quote -
a thin, illiquid instrument. That makes it an input to how big a position can
be, never a reason to take one: a high lambda says "this book moves under
size", not "buy" or "sell". This module cannot stop a caller from misreading
it as a trigger; the catalogue note and this docstring are the only guard, and
the axis verdict records that limitation rather than papering over it.

## What is fit, per (venue, symbol) window

    close_t - close_{t-1} = lambda * signed_flow_t + intercept + residual

by ordinary least squares over the newest `WINDOW_BARS` one-minute bars
knowable at the as-of clock, read through `ClockGatedReader` so a backtest
asking for the fit as of a past clock sees only bars that had closed and
arrived by then. `lambda_price_impact` is the slope.

## The regressor is a proxy, and it is named as one

Kyle's lambda is defined against *signed* order flow: net buyer-minus-seller
volume. `store.trade_bars` builds this store's bars from `open, high, low,
close, volume, trades` alone - no venue adapter here persists a trade's side
(`capture.venues.binance`'s aggTrade carries a maker flag `m` and is not kept;
`capture.venues.coinbase`'s `match` carries `side` and is not kept either), so
no signed-flow feed exists anywhere in this store today. Handing `volume`
alone to this regression would be exactly the unlabelled-proxy failure this
repo keeps finding - a person's guess wearing the name of a measurement - so
this module does not do that.

Instead it builds `signed_flow_source = "tick_rule_signed_volume_proxy"`:
`sign(close_t - close_{t-1}) * volume_t`, the standard substitute when
trade-side data is unavailable (the tick rule, Lee & Ready 1991). The name
rides on every row as a column, not only in this docstring, because a caller
reading the table without the source open must still see that this is a
proxy.

**Stated plainly because it would otherwise be found the hard way**: the tick
rule signs each bar's flow FROM the same price change the regression explains,
so `sign(signed_flow_t) == sign(delta_price_t)` by construction whenever the
price moved. The fit is therefore not an independent test of whether flow
predicts price direction - direction already agrees by definition. What it
CAN measure is whether the MAGNITUDE of volume predicts the MAGNITUDE of the
move, given that the sign is not in question. `r_squared` here will run higher
than an estimate from genuine trade-side data would, for a reason that has
nothing to do with liquidity, and a caller must not read it as evidence of
causal price impact beyond that. The feed that would remove this ceiling is
per-trade buy/sell classification captured at ingestion; until
`store.trade_bars` carries it, `lambda_price_impact` is a same-window
self-consistency measure of illiquidity, not a causal one.

## Refusals - no value from a default

A regression needs points to fit, variance in what explains, and variance in
what is explained:

* fewer than `MIN_OBSERVATIONS` price-change/flow pairs in the window -
  `too_few_observations`. A slope through a handful of points is a line
  through noise wearing a coefficient's clothes.
* zero variance in the flow proxy over the window - `zero_variance_regressor`.
  Every bar traded the same size in the same direction (or the window is one
  long silence); the OLS slope is 0/0, and returning zero would read as "this
  instrument has no price impact" rather than "this window could not measure
  it".
* zero variance in the price change itself - `zero_price_variance`. The close
  moved by the identical amount every bar; there is nothing left for R² to
  explain (also 0/0), and a slope with no fit quality beside it is the
  unjudgeable number the catalogue note warns a sizing caller against.
* any non-positive or unparseable close in the window - `non_positive_price`
  / `unparseable_bar`. `store.trade_bars.is_tradeable` already refuses
  zero-price trades before a bar is built, so these should not fire in
  practice; they are kept as a second gate rather than trusted to the first,
  because a defect reaching this regression corrupts a price CHANGE at both
  ends it touches, not just one bar.

## FE-001

Every row is stamped against the newest bar in its OWN window via
`measure_staleness`, keyed on (venue, symbol) - a lambda fit from a book that
stopped updating an hour ago is a confident number about a market that has
moved on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# Two hours of one-minute bars. Long enough that a handful of thin minutes
# cannot dominate the fit, short enough that the "regime" in "liquidity-regime
# descriptor" is this session rather than last week's. Disclosed here rather
# than tuned - one of the two arbitrary numbers this module carries.
WINDOW_BARS = 120

# Below this many price-change/flow pairs, the fit is not returned. The same
# floor `features.price_divergence` and `features.consolidated_price` use for
# trusting a scale built from a symbol's own history - fewer points than that
# and a fitted slope is a coincidence wearing a coefficient's clothes.
MIN_OBSERVATIONS = 20

# What the regressor actually is. Carried as a column value, not only stated
# in the docstring, so a consumer reading the table without the source open
# still sees this is a proxy for signed order flow and not the real thing.
_SIGNED_FLOW_SOURCE = "tick_rule_signed_volume_proxy"

_COLUMNS = ("venue", "symbol", "event_time_ns", "lambda_price_impact",
            "signed_flow_source", "r_squared", "residual_std", "observations",
            "window_bars", "window_start_ns", "window_end_ns")


@dataclass(frozen=True)
class KyleLambdaTable:
    """Per (venue, symbol) price-impact fit, and what could not be measured.

    `rows` carries the newest fit each window supports - one per (venue,
    symbol), not a history; a caller wanting the history calls this at the
    clocks it cares about, each read clock-gated. `refused` counts what the
    bars offered that could not be fit, by reason, so a consumer cannot
    mistake a quiet instrument for one with no price impact.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def compute_kyle_lambda(store_root: Path, as_of_ns: int,
                        custodian=None) -> KyleLambdaTable:
    """Fit lambda per (venue, symbol) over the newest bars knowable at `as_of_ns`.

    Liquidity-regime descriptor for sizing (`FEATURES.md` §2) - not a
    standalone signal, and not one this function can enforce; see the module
    docstring for why the fit is a proxy and what it can and cannot show.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {"too_few_observations": 0, "zero_variance_regressor": 0,
               "zero_price_variance": 0, "non_positive_price": 0,
               "unparseable_bar": 0}
    if frame.empty:
        return KyleLambdaTable(rows=_empty_rows(), refused=refused)

    out = {column: [] for column in _COLUMNS}
    ages: dict[tuple[str, str], object] = {}

    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        recent = group.sort_values("event_time_ns", kind="mergesort").tail(WINDOW_BARS)
        closes = pd.to_numeric(recent["close"], errors="coerce").to_numpy(dtype="float64")
        volumes = pd.to_numeric(recent["volume"], errors="coerce").to_numpy(dtype="float64")

        if np.isnan(closes).any() or np.isnan(volumes).any():
            # An unparseable close or volume is a different defect from a
            # non-positive one, and it is counted separately so the reason
            # says which repair is needed.
            refused["unparseable_bar"] += 1
            continue
        if (closes <= 0).any():
            refused["non_positive_price"] += 1
            continue

        delta_price = np.diff(closes)
        n = len(delta_price)
        if n < MIN_OBSERVATIONS:
            refused["too_few_observations"] += 1
            continue

        # sign() is 0 for an unchanged close: a bar the tick rule cannot
        # classify contributes no flow rather than a guessed direction.
        signed_flow = np.sign(delta_price) * volumes[1:]

        flow_mean = float(np.mean(signed_flow))
        flow_var = float(np.mean((signed_flow - flow_mean) ** 2))
        if flow_var == 0.0:
            refused["zero_variance_regressor"] += 1
            continue

        price_mean = float(np.mean(delta_price))
        ss_tot = float(np.sum((delta_price - price_mean) ** 2))
        if ss_tot == 0.0:
            refused["zero_price_variance"] += 1
            continue

        covariance = float(np.mean((signed_flow - flow_mean) * (delta_price - price_mean)))
        slope = covariance / flow_var
        intercept = price_mean - slope * flow_mean
        residuals = delta_price - (intercept + slope * signed_flow)
        ss_res = float(np.sum(residuals ** 2))
        r_squared = 1.0 - ss_res / ss_tot
        # Residual standard error: two parameters fit (slope, intercept), so
        # n - 2 degrees of freedom. MIN_OBSERVATIONS >= 20 keeps this away
        # from the n <= 2 singularity.
        residual_std = float(np.sqrt(ss_res / (n - 2)))

        newest_event_ns = int(recent["event_time_ns"].iloc[-1])
        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["event_time_ns"].append(newest_event_ns)
        out["lambda_price_impact"].append(slope)
        out["signed_flow_source"].append(_SIGNED_FLOW_SOURCE)
        out["r_squared"].append(r_squared)
        out["residual_std"].append(residual_std)
        out["observations"].append(n)
        out["window_bars"].append(len(recent))
        out["window_start_ns"].append(int(recent["event_time_ns"].iloc[0]))
        out["window_end_ns"].append(newest_event_ns)

        ages[(venue, symbol)] = measure_staleness(
            group["event_time_ns"].astype("int64"), as_of_ns)

    rows = pd.DataFrame(out)
    if not rows.empty:
        rows = rows.sort_values(["venue", "symbol"], ignore_index=True)
    return KyleLambdaTable(rows=stamp(rows, ages, ["venue", "symbol"]), refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
