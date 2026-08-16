"""HAR-RV: the volatility cascade, fitted walk-forward and judged out of sample.

`FEATURES.md` §2 (P1) — *"HAR-RV — beats GARCH-family for short-horizon
crypto"*. Ledger FE-004. The claim traces to Corsi, F. (2009), *"A Simple
Approximate Long-Memory Model of Realized Volatility"*, Journal of Financial
Econometrics 7(2), 174-196, and the crypto half to the two studies
`~/research/finml-feature-engineering.md` §3 names — both of which found HAR on
realized variance from high-frequency data generally beating GARCH-family
out-of-sample for short-horizon crypto, *"no single model dominating
universally"*. That last clause is why this module ships with a verdict
attached rather than with a citation attached.

## What HAR actually says

Volatility is persistent, and the persistence has structure: traders operating
on different horizons each leave a footprint, and a short-horizon trader watches
the last hour, a medium one the last shift, a long one the last day. Corsi's
model is that cascade written as one linear regression — next period's realized
variance on three lagged averages of it, short, medium and long. It is not a
long-memory model; it is three short-memory terms whose sum imitates one, which
is the whole reason it is cheap enough to refit continuously.

    RV_next = b0 + b_short * RV_short + b_medium * RV_medium + b_long * RV_long

## The cascade is in bars, and the ratios are Corsi's, not fitted

Corsi's cascade is 1 day / 5 days / 22 days because that is the equity market's
week and month. Crypto has neither — no weekend close, no exchange holiday, and
`features.realized_volatility` already refuses the 252-day convention for the
same reason. What carries over is the *ratio* structure, roughly 1 : 5 : 22, and
the requirement that the base period be short enough to matter to a decision.

So the base period here is one hour (`SHORT_BARS`, 60 one-minute bars), and the
cascade is 1 : 6 : 24 — hour, shift, day. Every one of those three numbers is a
declared constant with the ratio written next to it, and **none of them is
fitted**. A cascade whose horizons were selected by search would be three more
hyperparameters entering through the back door, and the Trial Registry would
never see them: the model would be picked from a family and then reported as
though one thing had been tried. `FEATURES.md` §2 marks the same trap on
triple-barrier width, which does go through the registry precisely because it is
a searched number.

## Variance, not volatility, and all four quantities in one unit

The regressand and all three regressors are realized **variance per base
period** — sum of squared log returns over an hour. The medium and long terms
are *averages of hourly variances*, not sums over six and twenty-four hours, so
a coefficient is a weight on a comparable quantity rather than on a number that
is six times larger for arithmetic reasons. Regressing a sum on sums makes
`b_medium` absorb the period-length ratio and the cascade stops being readable.

Volatility (the square root) is what a position sizer wants and is one call
away; the regression lives in variance because that is the quantity that is
additive over time, and additivity is the only reason averaging the lags is a
meaningful operation at all.

## Overlapping regressors are the model; overlapping targets are a defect

HAR's three regressors overlap by construction — the day average contains the
shift average contains the hour. That is the model and it is fine.

The *targets* are a different matter. Consecutive base periods here are strictly
non-overlapping: period `i` covers bars `[i*60, (i+1)*60)` and shares no bar with
period `i+1`. Sliding the target one bar at a time would multiply the observation
count sixtyfold while adding almost no information, and every significance test
downstream would read that inflated count as evidence. `features.sample_
uniqueness` exists because overlapping labels violate IID; this module simply
never creates the overlap.

## Fitted walk-forward, and no in-sample number is reported at all

Every reported observation is out of sample. For evaluation point `i`, the
coefficients are fitted on observations `0..i-1` only and then used once, on `i`.
The fit expands as the record grows, which is what a live refit does.

**No in-sample R², t-statistic or p-value appears in the output**, and that is a
deliberate omission rather than a gap. HAR's regressors are overlapping and
strongly autocorrelated; in-sample OLS inference on them is inflated in the
flattering direction, and an inflated t-statistic on a volatility cascade is one
of the easiest impressive numbers in this literature to produce. `numpy.linalg.
lstsq` is used for the solve rather than `statsmodels.OLS` for exactly this
reason — not because statsmodels is unavailable (it is a declared dependency,
`features.fractional_differentiation` uses it) but because importing a machinery
whose headline output is that inference invites reporting it.

## It has to beat the random walk, and it says so when it does not

`models.naive_baseline.judge_against_naive_baseline` — MD-001/VX-121, mandatory
for every model — is called on the out-of-sample forecasts, against the naive
benchmark *"next period's variance is this period's variance"*. The benchmark is
passed explicitly rather than left to the default: the default derives it from
the previous element of the actual series, which is only the same thing when the
evaluation points are contiguous, and a gap in the tape makes them not.

The verdict is a field of the output, both ways. A HAR fit that did not beat the
random walk is reported as one — `beats_naive=False`, with the p-value and both
mean losses beside it — because the finding *"the cascade bought nothing on this
symbol"* is the honest result on a market where the cited literature itself
reports no universal winner.

## Negative forecasts are clamped and counted, never just clamped

OLS is unconstrained and can predict a negative variance, which is not a small
number but an impossible one. Each is clamped to zero for scoring and counted in
`negative_forecasts`. Clamping silently would hide a fit that is extrapolating
outside the region its data supports; the count is what makes that visible, and
a row with many of them should not be believed however good its p-value looks.

## Refusals, and why a refused period poisons its whole window

A base period needs at least half of an unbroken hour's returns to be a variance
rather than a rumour — the same half-of-expected floor `features.realized_
volatility` applies, for the same reason. A period that misses the floor is not
computed, and **every cascade window containing it is skipped as well**: an
average over a 24-hour window with six missing hours is not a 24-hour average,
and computing it anyway would let the tape's holes into the regressors without
leaving a mark. `insufficient_period_coverage` counts the windows lost that way.

## Prices are Decimal; the regression is float, and the boundary is deliberate

Closes are parsed and differenced in `Decimal` exactly as `features.realized_
volatility` does — same defect history (placeholder frames carrying price `"0"`),
same refusal to divide by them. The conversion to float happens once the
quantities have stopped being prices: a realized variance is a dimensionless sum
of squared log ratios, and a least-squares solve over a design matrix has no
Decimal implementation worth writing. The boundary is named here so it is a
decision rather than a place where the convention quietly lapsed.

## Staleness (FE-001)

Stamped per (venue, symbol) against the whole visible bar history for that key,
matching `features.realized_volatility`: a fit can be perfectly valid while the
newest bar behind it is an hour old, and the stamp is the only thing that says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from features.realized_volatility import BAR_INTERVAL_NS, decimal_close
from features.staleness import measure_staleness, stamp
from models.naive_baseline import (
    DegenerateBenchmark, judge_against_naive_baseline,
)
from store.clock_gated_reader import ClockGatedReader
from validation.superior_predictive_ability import NotEnoughPaths

_DATASET = "bars_60000000000ns"

# The cascade, in base periods. Corsi's 1 : 5 : 22 is a trading week and month;
# this is 1 : 6 : 24 - an hour, a shift, a day - because crypto has neither of
# the former. Declared, never searched: see the module docstring.
SHORT_BARS = 60                     # one hour of one-minute bars
MEDIUM_PERIODS = 6                  # six hours
LONG_PERIODS = 24                   # one day

# At least half of an unbroken hour's returns before a base period is a variance
# at all. Same floor and same reasoning as `realized_volatility`, whose
# `_minimum_required_returns` applies the identical half-of-expected rule.
MIN_RETURNS_PER_PERIOD = max(2, (SHORT_BARS - 1) // 2)

# Four coefficients, so a fit on a handful of points is an interpolation rather
# than an estimate. Thirty is not a magic number - it is the smallest sample at
# which the four-parameter fit has an order of magnitude more observations than
# parameters, which is the weakest defensible statement about it.
MIN_FIT_OBSERVATIONS = 30

# Mirrors `naive_baseline.MIN_OBSERVATIONS`, which mirrors the SPA machinery's
# own floor. Checked here so "too little history to test" is diagnosed as that,
# ahead of the exception it would otherwise surface as. Pinned equal in tests.
MIN_EVALUATION_OBSERVATIONS = 20

_COLUMNS = ("venue", "symbol", "n_periods", "n_fit", "n_evaluated",
            "beta_intercept", "beta_short", "beta_medium", "beta_long",
            "mean_loss_har", "mean_loss_naive", "loss_reduction_pct",
            "p_value", "beats_naive", "reproduced_lag_one_trap",
            "negative_forecasts", "window_start_ns", "window_end_ns")

_REFUSAL_REASONS = (
    "too_few_bars",
    "insufficient_period_coverage",
    "too_few_periods_for_cascade",
    "too_few_observations_to_fit",
    "too_few_observations_to_test",
    "singular_design",
    "degenerate_benchmark",
    "non_positive_close",
    "unparseable_close",
)


@dataclass(frozen=True)
class HarRvTable:
    """One out-of-sample HAR-RV verdict per (venue, symbol), and what was refused.

    `rows` carries at most one row per key, and only for keys that produced a
    walk-forward record long enough to test. `refused` counts every key or window
    that did not, by reason - visible for the reason every sibling feature makes
    it visible: a table with two rows where six symbols were expected reads
    identically to a market where four symbols were quiet.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


@dataclass(frozen=True)
class BasePeriod:
    """Realized variance over one non-overlapping base period.

    `start_ns` is carried because the periods are indexed by position in a list
    and the list can have holes; a window is only a 24-hour window if its periods
    are actually consecutive, and the timestamp is what proves it.
    """
    start_ns: int
    end_ns: int
    variance: float
    n_returns: int


def base_period_variances(event_time_ns, closes: list[Decimal],
                          ) -> tuple[list[BasePeriod | None], dict[str, int]]:
    """Split a bar series into consecutive non-overlapping base periods.

    Periods are cut on the clock, not on bar count: period `k` covers
    `[origin + k*SHORT_BARS*BAR_INTERVAL_NS, origin + (k+1)*...)`. Cutting on
    bar count instead would make a period silently longer whenever the tape had
    a hole, and the two would be indistinguishable in the output - which is the
    same class of defect as an average over a window with missing hours.

    Returns one entry per period slot, `None` where the slot did not clear
    `MIN_RETURNS_PER_PERIOD`, so a caller can tell a missing hour from a quiet
    one. The refusal counter is returned alongside rather than raised: one bad
    hour is not a reason to refuse a symbol's whole history.
    """
    refused = {"insufficient_period_coverage": 0}
    if len(closes) < 2:
        return [], refused

    times = [int(t) for t in event_time_ns]
    period_ns = SHORT_BARS * BAR_INTERVAL_NS
    origin = times[0] - (times[0] % period_ns)
    n_slots = (times[-1] - origin) // period_ns + 1

    # Bucket returns by the period their CLOSING bar falls in. A return spans two
    # bars; attributing it to the later one is what makes the periods disjoint,
    # since every bar closes exactly one return.
    buckets: dict[int, list[float]] = {}
    for i in range(1, len(closes)):
        previous, current = closes[i - 1], closes[i]
        log_return = float((current / previous).ln())
        slot = (times[i] - origin) // period_ns
        buckets.setdefault(slot, []).append(log_return)

    periods: list[BasePeriod | None] = []
    for slot in range(n_slots):
        returns = buckets.get(slot, [])
        if len(returns) < MIN_RETURNS_PER_PERIOD:
            refused["insufficient_period_coverage"] += 1
            periods.append(None)
            continue
        periods.append(BasePeriod(
            start_ns=origin + slot * period_ns,
            end_ns=origin + (slot + 1) * period_ns,
            variance=sum(r * r for r in returns),
            n_returns=len(returns)))
    return periods, refused


def cascade_observations(periods: list[BasePeriod | None],
                         ) -> tuple[list[tuple[float, float, float]], list[float],
                                    list[float], int]:
    """Design rows, targets, and the naive benchmark, from a period series.

    One observation per period slot `i` that has `LONG_PERIODS` unbroken periods
    behind it and a period at `i + 1` to predict. A single `None` anywhere in
    that span drops the observation - see the docstring's note on why a window
    with holes is not a shorter window.

    The naive benchmark is returned as its own list rather than derived later
    from the targets: consecutive TARGETS are only consecutive in time when no
    observation was dropped between them, and the random walk's claim is about
    time, not about list position.
    """
    design: list[tuple[float, float, float]] = []
    targets: list[float] = []
    naive: list[float] = []
    skipped = 0

    for i in range(LONG_PERIODS - 1, len(periods) - 1):
        window = periods[i - LONG_PERIODS + 1:i + 1]
        target_period = periods[i + 1]
        if target_period is None or any(p is None for p in window):
            skipped += 1
            continue
        variances = [p.variance for p in window]          # type: ignore[union-attr]
        short = variances[-1]
        medium = sum(variances[-MEDIUM_PERIODS:]) / MEDIUM_PERIODS
        long = sum(variances) / LONG_PERIODS
        design.append((short, medium, long))
        targets.append(target_period.variance)
        # "Next period looks like this period" - the random walk on variance.
        naive.append(short)
    return design, targets, naive, skipped


def _fit_coefficients(design: np.ndarray, targets: np.ndarray) -> np.ndarray | None:
    """OLS coefficients with an intercept, or None if the design is degenerate.

    `lstsq` returns a least-norm answer for a rank-deficient design rather than
    raising, which would turn "these three regressors are the same number" into a
    confident-looking fit. The rank is checked explicitly and the fit refused.
    """
    with_intercept = np.column_stack([np.ones(len(design)), design])
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        with_intercept, targets, rcond=None)
    if rank < with_intercept.shape[1]:
        return None
    return coefficients


def walk_forward_forecasts(design: list[tuple[float, float, float]],
                           targets: list[float],
                           ) -> tuple[list[float], list[int], int, np.ndarray | None]:
    """Forecast each point from a fit that never saw it.

    Returns the forecasts, the indices they correspond to, how many were negative
    before clamping, and the LAST fitted coefficient vector - which is the one a
    live caller would hold, and the only one worth reporting out of the several
    hundred this function fits.

    The fit expands rather than rolls. An expanding window is what a system that
    keeps its whole record does; a rolling one is a window length, which is
    another unsearched hyperparameter, and this module already declares as many
    of those as it can defend.
    """
    forecasts: list[float] = []
    indices: list[int] = []
    negative = 0
    coefficients = None
    design_array = np.asarray(design, dtype=float)
    target_array = np.asarray(targets, dtype=float)

    for i in range(MIN_FIT_OBSERVATIONS, len(design)):
        fitted = _fit_coefficients(design_array[:i], target_array[:i])
        if fitted is None:
            continue
        coefficients = fitted
        raw = float(fitted[0] + fitted[1:] @ design_array[i])
        if raw < 0.0:
            negative += 1
            raw = 0.0
        forecasts.append(raw)
        indices.append(i)
    return forecasts, indices, negative, coefficients


def compute_har_rv(store_root: Path, as_of_ns: int, custodian=None) -> HarRvTable:
    """Fit and judge a HAR-RV cascade per (venue, symbol) at this clock.

    Reads the one-minute bar dataset once through `ClockGatedReader`, so a
    backtest asking as of a past clock sees exactly the bars that had closed and
    arrived by then. Every forecast inside is produced by coefficients fitted
    only on observations strictly before it, so the walk-forward record is
    honest at both gates - the store's and the fit's.
    """
    as_of_ns = int(as_of_ns)
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)

    refused = {reason: 0 for reason in _REFUSAL_REASONS}
    if frame.empty:
        return HarRvTable(rows=_empty_rows(), refused=refused)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        ordered = group.sort_values("event_time_ns")

        closes: list[Decimal] = []
        refusal_reason = None
        for raw_close in ordered["close"]:
            close = decimal_close(raw_close)
            if close is None:
                refusal_reason = "unparseable_close"
                break
            if close <= 0:
                # The placeholder-price defect. Refuse the key rather than drop
                # the bar: a period whose bar count quietly shrank around a bad
                # price still reports a variance, and it carries no sign of it.
                refusal_reason = "non_positive_close"
                break
            closes.append(close)
        if refusal_reason is not None:
            refused[refusal_reason] += 1
            continue

        if len(closes) < SHORT_BARS * (LONG_PERIODS + 1):
            refused["too_few_bars"] += 1
            continue

        periods, period_refusals = base_period_variances(
            ordered["event_time_ns"].tolist(), closes)
        refused["insufficient_period_coverage"] += period_refusals[
            "insufficient_period_coverage"]

        design, targets, naive, skipped = cascade_observations(periods)
        refused["too_few_periods_for_cascade"] += skipped
        if len(design) <= MIN_FIT_OBSERVATIONS:
            refused["too_few_observations_to_fit"] += 1
            continue

        forecasts, indices, negative, coefficients = walk_forward_forecasts(
            design, targets)
        if coefficients is None:
            refused["singular_design"] += 1
            continue
        if len(forecasts) < MIN_EVALUATION_OBSERVATIONS:
            refused["too_few_observations_to_test"] += 1
            continue

        evaluated = [targets[i] for i in indices]
        benchmark = [naive[i] for i in indices]
        try:
            verdict = judge_against_naive_baseline(
                evaluated, forecasts, benchmark)
        except DegenerateBenchmark:
            # A tape that did not move. Not a HAR failure and not a HAR success.
            refused["degenerate_benchmark"] += 1
            continue
        except NotEnoughPaths:
            refused["too_few_observations_to_test"] += 1
            continue

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["n_periods"].append(sum(1 for p in periods if p is not None))
        out["n_fit"].append(MIN_FIT_OBSERVATIONS)
        out["n_evaluated"].append(verdict.n_observations)
        out["beta_intercept"].append(float(coefficients[0]))
        out["beta_short"].append(float(coefficients[1]))
        out["beta_medium"].append(float(coefficients[2]))
        out["beta_long"].append(float(coefficients[3]))
        out["mean_loss_har"].append(verdict.mean_loss_model)
        out["mean_loss_naive"].append(verdict.mean_loss_naive)
        out["loss_reduction_pct"].append(verdict.loss_reduction_pct)
        out["p_value"].append(verdict.p_value)
        out["beats_naive"].append(verdict.beats_naive)
        out["reproduced_lag_one_trap"].append(verdict.reproduced_lag_one_trap)
        out["negative_forecasts"].append(negative)
        out["window_start_ns"].append(int(ordered["event_time_ns"].iloc[0]))
        out["window_end_ns"].append(as_of_ns)

    rows = pd.DataFrame(out)
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return HarRvTable(rows=stamp(rows, ages, ["venue", "symbol"]), refused=refused)


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])
