"""Pegged assets, found by behaviour and judged against their own history.

`FEATURES.md` §1 marks this `[MISSED]` with the case that earns it: **USDe hit
$0.65 only on Binance during Oct 2025.** Per venue, not per asset - a peg that
holds on one venue and breaks on another is exactly the shape a cross-venue
average hides, so nothing here averages across venues.

## Three decisions, and why none of them is a typed number

**Which assets are pegged is measured, never spelled.** The obvious
implementation - a list of stablecoin tickers, or matching a `USD*` prefix -
fails on this box's own data in both directions: `TUSDT` is the *T* token at
$0.0037 and would be read as TrueUSD by any prefix rule, while `EURIUSDT` is a
genuine peg the rule would keep but mis-price, because it is pegged to the euro
near $1.16 rather than to a dollar. So an asset is treated as pegged when its
own price series *behaves* like one: tight relative dispersion around its own
central level, over a window long enough to have seen movement. The T token's
volatility disqualifies it on the numbers; EURI qualifies at its own level.

**Where the peg sits is fitted, never assumed to be 1.0.** The level is the
median of the asset's own recent closes on that venue. A hardcoded 1.0 is wrong
for every non-dollar peg and, worse, silently right-looking for dollar ones.

**How far is too far is the worst this asset wobbled while it was holding**,
not a bps number someone chose. USDC's ordinary wobble and USDe's are different
sizes, and a single shared threshold is either deaf to one or noisy on the
other. What IS fixed is the window length, the two dispersion bands separating
peg-like behaviour from token-like, and the minimum history - `_HISTORY_BARS`,
`_PEGGED_MAX_DISPERSION`, `_UNDECIDED_MIN_DISPERSION`, `_MIN_REFERENCE_BARS`
below. Per §2.1 of the intelligence record, the question is never whether fixed
structure exists - it must - but whether the *fitted* part moves with the data
and the fixed part is declared. Those four are the declared part.

## What it changes when it is wrong

Two facts, reported separately because both can be true at once: `verdict` is
`BREACHED` when the latest bar is further from the reference level than
anything that level ever saw, and `level_moved` is True when the peg has
settled somewhere new. A spike that returns and a permanent 3% move are
identical under either one alone.

The intended consumer is position sizing on anything quoted in that asset.
**No sizer exists yet, so nothing acts on a breach** - the axis verdict for
this module fails DEPTH for exactly that reason rather than claiming a
consumer it does not have.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# --- the declared fixed structure, all three of it ---------------------------
# How many of an asset's own most recent bars decide its level and its ordinary
# deviation. One trading day at this bar width. Shorter and a slow depeg becomes
# the new normal within hours; longer and a genuinely new regime takes a day to
# register.
_HISTORY_BARS = 1440
# An asset qualifies as pegged when its relative dispersion over that window is
# tighter than this. Chosen an order of magnitude above the tightest real peg's
# wobble and an order below an ordinary token's, which the live data separates
# cleanly: USDC/FDUSD/USDE sit near 1e-4 and the T token near 1e-1. Anything
# landing between them is not classified as either - see `undecided`.
_PEGGED_MAX_DISPERSION = 0.005
_UNDECIDED_MIN_DISPERSION = 0.001
# How many bars the reference half must hold before it can classify anything.
# The reference is half the window, so this is also the practical floor on how
# much history an asset needs before it is judged at all.
_MIN_REFERENCE_BARS = 150


@dataclass(frozen=True)
class PegTable:
    """Per (venue, asset) peg state, and an account of what was not judged.

    `rows` carries one verdict per venue-asset pair that could be judged;
    `skipped` counts the pairs that could not be, by reason. A monitor that
    quietly judged fewer assets each day would look identical to a calm market,
    which is the failure this counter exists to make impossible.
    """
    rows: pd.DataFrame
    skipped: dict[str, int]


def monitor_pegs(store_root: Path, as_of_ns: int, custodian=None) -> PegTable:
    """Judge every asset that behaves like a peg, per venue, at this clock."""
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    skipped = {"too_few_bars": 0, "not_pegged": 0, "undecided": 0,
               "no_positive_level": 0}
    if frame.empty:
        return PegTable(rows=_empty_rows(), skipped=skipped)

    out = {"venue": [], "symbol": [], "peg_level": [], "close": [],
           "deviation": [], "breach_threshold": [], "verdict": [],
           "level_moved": [], "drift": [], "threshold_floored": [],
           "bars_judged": []}

    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        recent = group.sort_values("event_time_ns").tail(_HISTORY_BARS)
        closes = pd.to_numeric(recent["close"], errors="coerce").dropna()
        if len(closes) < 2 * _MIN_REFERENCE_BARS + 1:
            skipped["too_few_bars"] += 1
            continue

        # Nothing being judged enters the fit that judges it. The window splits
        # in three: an older REFERENCE half answers "was this behaving as a
        # peg, and at what level"; the newer half answers "is it still there";
        # the latest bar answers "is it there right now".
        #
        # Both halves of that split are load-bearing, and each was found by a
        # failing test rather than reasoned out in advance:
        #
        # - Judging the latest bar against statistics it contributed to made a
        #   depeg invisible. A break to $0.65 inflates the dispersion it is
        #   measured against, so the asset stops qualifying as pegged and is
        #   silently skipped as just another volatile token - precisely when
        #   the monitor is needed.
        # - Classifying on the WHOLE window made a peg that walked to a new
        #   level disappear the same way: 1.00 for half the window and 0.97
        #   for the other half is 3% dispersion, which reads as "never was a
        #   peg". An asset that was pegged and left is the event, not a
        #   reason to stop watching it.
        #
        # Same family of rule as the purge in the validation stack: fit on
        # what came before, judge what came after.
        fit, latest_close = closes.iloc[:-1], float(closes.iloc[-1])
        split = len(fit) // 2
        reference, since = fit.iloc[:split], fit.iloc[split:]
        if len(reference) < _MIN_REFERENCE_BARS:
            skipped["too_few_bars"] += 1
            continue

        level = float(reference.median())
        if level <= 0:
            skipped["no_positive_level"] += 1
            continue

        deviations = (reference - level).abs() / level
        dispersion = float(deviations.max())
        if dispersion >= _PEGGED_MAX_DISPERSION:
            skipped["not_pegged"] += 1
            continue
        if dispersion >= _UNDECIDED_MIN_DISPERSION:
            # Between the two bands. Not called either way, because an asset
            # that is drifting toward or away from a peg is precisely the case
            # where a forced binary answer would be wrong and confident.
            skipped["undecided"] += 1
            continue

        # The worst this asset wobbled while it was behaving, floored at what
        # it costs to act on the move. Not a quantile of the wobble: at these
        # window lengths a 99.9th percentile IS the maximum, so quoting a
        # quantile would have been a parameter that changed nothing except the
        # number of knobs.
        #
        # The floor is what the first live run demanded. Against the real
        # store, six assets read BREACHED on deviations of two to four basis
        # points, one of them against a threshold of 1e-6 - a single price
        # tick. A peg quiet enough to have almost no observed wobble gets a
        # threshold near zero, and then every tick is an alarm. The floor is
        # the round-trip cost from the cost engine: a deviation smaller than
        # the cost of trading it cannot be acted on, so calling it a breach is
        # a claim nobody could use. That is a measured number from verified
        # fee schedules, not a chosen one - and when the quote refuses, the
        # unfloored threshold stands rather than a default appearing.
        floor = _cost_floor(venue, symbol, as_of_ns)
        threshold = max(dispersion, floor)
        latest_deviation = abs(latest_close - level) / level

        # Has the peg itself moved? A depeg that persists becomes the new
        # normal and every individual bar then looks calm at the broken level
        # - the confident-staleness failure, arriving on a delay measured in
        # hours. Measured against the reference level, which is why the
        # reference half must not include the drift.
        drift = abs(float(since.median()) - level) / level

        # Two separate facts, reported separately. "It is off its peg right
        # now" and "it has settled somewhere new" are both true of an asset
        # that walked down and stayed, and collapsing them into one verdict
        # meant picking a precedence - which silently discarded whichever
        # fact lost. A spike back to the peg and a permanent move look
        # identical under `verdict` alone; `level_moved` is what separates
        # them, and a consumer that only reads the verdict is no worse off
        # than before.
        verdict = "BREACHED" if latest_deviation > threshold else "HELD"

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["peg_level"].append(level)
        out["close"].append(latest_close)
        out["deviation"].append(latest_deviation)
        out["breach_threshold"].append(threshold)
        out["verdict"].append(verdict)
        out["level_moved"].append(bool(drift > threshold))
        out["drift"].append(drift)
        # Whether the cost engine could price this venue-symbol at all. An
        # unfloored threshold is a weaker claim than a floored one - it can
        # sit below the cost of acting - and the two must not read alike.
        # Measured 2026-08-09: binance perps and hyperliquid floor; every
        # binance-spot pair does not, because no verified spot fee schedule
        # exists yet.
        out["threshold_floored"].append(floor > 0.0)
        out["bars_judged"].append(len(fit))

    return PegTable(rows=pd.DataFrame(out), skipped=skipped)


def _cost_floor(venue: str, symbol: str, at_ns: int) -> float:
    """Round-trip cost as a fraction, or 0.0 when it cannot be priced.

    Zero on refusal rather than a stand-in cost: an unpriceable symbol falls
    back to its own measured wobble, which is a real threshold. Substituting a
    plausible cost would be the module inventing the one input it exists to
    take from measurement.

    Taker on both legs - the pessimistic side. A peg breach is not a resting
    order's kind of event.
    """
    from cost.round_trip_cost import CostRefused, quote_round_trip_cost

    try:
        quote = quote_round_trip_cost(
            venue, symbol, Decimal("1000"), order_type="taker", at_ns=int(at_ns),
            instrument_kind="spot" if venue.endswith("-spot") else "perp")
    except Exception:
        return 0.0
    if isinstance(quote, CostRefused):
        return 0.0
    return float(quote.breakeven_bps) / 10_000.0


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({"venue": [], "symbol": [], "peg_level": [], "close": [],
                         "deviation": [], "breach_threshold": [], "verdict": [],
                         "level_moved": [], "drift": [], "threshold_floored": [],
                         "bars_judged": []})
