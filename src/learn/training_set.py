"""LB-01: the pooled cross-sectional training set — features at a bar, outcome after it.

## Why pooled, and not one model per symbol

Measured 2026-08-18: the store holds **67 hour-partitions across 8 non-contiguous days**
— 2026-08-02, 03, 08, 09, 15, 16, 17 and 18 — roughly 860,000 bar rows over 1,698
symbols. That is about **500 bars per symbol**.

Five hundred bars cannot fit a model. Pooling every symbol into one cross-sectional
dataset turns 1,698 thin series into one usable one, and it is also the honest shape for
what the brains actually do: they rank the whole universe at each moment (RL-009,
RL-014), so a model that learns *what a promising symbol looks like* is closer to the
job than 1,698 models each learning one symbol's habits.

The cost is real and is not hidden: a pooled model cannot learn that BTC behaves unlike
a microcap. Cross-sectional normalisation is what stands in for that — every feature is
expressed in units comparable across symbols (returns, ratios, ranks) and never in
price, so a $64,000 instrument and a $0.30 one produce comparable rows.

## Labels: the barrier that resolved first, not the return at a horizon

`strategy.deterministic_exit.run_exit_policy` already implements the policy — triple
barrier with an ATR-scaled stop and a ratchet — and it is reused rather than
reimplemented, for a reason beyond tidiness. `FEATURES.md` line 99: that deterministic
policy is *the baseline PROFIT-TAIL must beat and the thing that generates its training
data*. Labelling with a different rule would train the model against outcomes the live
system will never produce.

A fixed-horizon return would be the easier label and the wrong one. It answers "where
was price in 15 minutes", when the live bot exits at whichever rail it touches first —
so a trade that went to target in two minutes and one that bled for fifteen would carry
the same label, and the model would learn to predict a number nothing in the system acts
on.

## The leakage rule, and where it is enforced

**Every feature on a row is computable from bars up to and including that row's own bar.
Every label comes from bars strictly after it.** `build_rows` never lets the two touch:
features come from a trailing window ending at `i`, the outcome from the path starting
at `i + 1`.

This is the failure that makes a model look brilliant and lose money, and it is
invisible in the metrics — a leaked feature scores well precisely because it contains
the answer. The purged cross-validation in `models.gradient_boosted_trees` guards the
*fold* boundary; nothing but this function guards the *row* boundary.

## Label spans ride with every row

`LabelSpan(event_index, touched_at_index)` records the bars a label's information window
occupied, so `uniqueness_weights` can discount overlapping labels. Without it, a cluster
of twenty overlapping labels counts as twenty independent observations when it is closer
to one, and every significance test downstream is computed against an inflated N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from features.sample_uniqueness import LabelSpan
from strategy.deterministic_exit import (
    Bar, ExitPolicy, average_true_range, run_exit_policy,
)


@dataclass(frozen=True)
class MarketBar:
    """A full OHLCV bar with its moment.

    `strategy.deterministic_exit.Bar` is deliberately only (high, low, close) - it
    is the shape a policy walks, and it validates itself. Features need open,
    volume, trade count and the timestamp, so those live here and the policy's Bar
    is constructed from this when a path is walked. Two types rather than widening
    a validated one that other code depends on.
    """

    event_time_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: float

    def as_path_bar(self) -> Bar:
        return Bar(high=Decimal(str(self.high)), low=Decimal(str(self.low)),
                   close=Decimal(str(self.close)))

# The trailing window each feature row is computed from, in bars. One hour of
# one-minute bars: long enough for a volatility estimate to mean something, short
# enough that a symbol with a few hundred bars still yields many rows.
FEATURE_WINDOW_BARS = 60
# How far ahead a label may look for its barrier. Beyond this the trade is
# unresolved and the row is dropped rather than labelled by the horizon - RL-018
# makes every segment intraday, and a label that needed four hours to resolve is
# not a label for an intraday bot.
MAX_LABEL_BARS = 30
# The minimum bars a symbol must have before it contributes at all.
MIN_BARS_PER_SYMBOL = FEATURE_WINDOW_BARS + MAX_LABEL_BARS + 5

# The feature names, in the order the model receives them. ORDER IS PART OF THE
# CONTRACT: the registered model is a matrix of splits over column indices, so a
# reordering here silently feeds every value to the wrong split. `FEATURE_NAMES`
# is stored in the model's metrics at registration and checked at load.
FEATURE_NAMES = (
    "return_1",
    "return_5",
    "return_15",
    "return_60",
    "volatility_15",
    "volatility_60",
    "volatility_ratio",
    "range_position",
    "volume_ratio",
    "trade_intensity",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_to_vwap",
    "momentum_consistency",
)


@dataclass
class TrainingRows:
    """The dataset, with the coverage it was built from stated alongside it."""

    features: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    spans: list = field(default_factory=list)
    # Parallel to the rows: which symbol and which bar each came from. Kept so a
    # model's behaviour can be traced back to the data that produced it, and so a
    # holdout can be split by symbol rather than only by time.
    symbols: list = field(default_factory=list)
    event_times_ns: list = field(default_factory=list)
    # The forward P&L per unit for each row, signed. PROFIT-TAIL (LB-06) fits its
    # quantiles on this; the directional brains use only the binary label.
    outcomes: list = field(default_factory=list)
    holding_bars: list = field(default_factory=list)
    max_favourable: list = field(default_factory=list)
    max_adverse: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def describe(self) -> dict:
        positive = sum(1 for label in self.labels if label == 1)
        days = sorted({_day_of(ns) for ns in self.event_times_ns})
        return {
            "rows": len(self.labels),
            "symbols": len(set(self.symbols)),
            "positive_rate": (positive / len(self.labels)) if self.labels else None,
            "days_covered": days,
            "n_days": len(days),
            "features": list(FEATURE_NAMES),
            "dropped_by_reason": dict(self.dropped),
            # Stated because §1a L6 keys on it: eight days of one regime is not a
            # regime change, and a model fitted here has not been stressed
            # out-of-regime whatever its accuracy says.
            "regime_note": ("all rows come from one regime; §1a L6 (out-of-regime "
                            "stress) cannot be satisfied by this dataset"),
        }

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _day_of(ns: int) -> str:
    import time
    return time.strftime("%Y-%m-%d", time.gmtime(ns / 1e9))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """A ratio, or 0.0 when the denominator is not usable.

    Zero is the right answer here and not a silent default: these are all ratios
    of a quantity to a recent average of itself, so "no recent activity to compare
    against" genuinely is no signal, and the alternative is dropping most rows of
    every thin symbol - which would quietly rebuild the narrow universe RL-014
    exists to widen.
    """
    if denominator is None or denominator == 0 or not math.isfinite(denominator):
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def compute_features(window) -> list | None:
    """One feature row from a trailing window of bars, ending at the decision bar.

    `window` is oldest-first and its LAST element is the bar being decided on. No
    element after it is visible here, which is the row-level leakage guard.
    """
    if len(window) < FEATURE_WINDOW_BARS:
        return None
    closes = [b.close for b in window]
    highs = [b.high for b in window]
    lows = [b.low for b in window]
    volumes = [b.volume for b in window]
    trades = [b.trades for b in window]

    last = closes[-1]
    if last <= 0:
        return None

    def horizon_return(n: int) -> float:
        if len(closes) <= n or closes[-1 - n] <= 0:
            return 0.0
        return math.log(last / closes[-1 - n])

    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i] > 0 and closes[i - 1] > 0
    ]

    def stdev(values) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    volatility_15 = stdev(log_returns[-15:])
    volatility_60 = stdev(log_returns)
    window_high, window_low = max(highs), min(lows)
    span = window_high - window_low

    recent_volume = sum(volumes[-15:]) / 15 if len(volumes) >= 15 else 0.0
    baseline_volume = sum(volumes) / len(volumes)
    recent_trades = sum(trades[-15:]) / 15 if len(trades) >= 15 else 0.0
    baseline_trades = sum(trades) / len(trades)

    bar = window[-1]
    high, low, close, open_ = bar.high, bar.low, bar.close, bar.open
    bar_span = high - low
    body_top, body_bottom = max(open_, close), min(open_, close)

    # A crude VWAP over the window: sum(close * volume) / sum(volume). Crude is
    # correct here - the store keeps OHLCV bars, not prints, so a true VWAP is not
    # recoverable and pretending otherwise would put a fabricated number in the
    # feature vector.
    total_volume = sum(volumes)
    vwap = (sum(c * v for c, v in zip(closes, volumes)) / total_volume
            if total_volume > 0 else last)

    # How one-directional the recent path was: the fraction of the last 15 bars
    # whose return shared the sign of the 15-bar return. Separates a clean trend
    # from a round trip that happened to end higher.
    recent = log_returns[-15:]
    direction = 1.0 if horizon_return(15) >= 0 else -1.0
    consistency = (sum(1 for r in recent if (r >= 0) == (direction >= 0)) / len(recent)
                   if recent else 0.0)

    return [
        horizon_return(1),
        horizon_return(5),
        horizon_return(15),
        horizon_return(60),
        volatility_15,
        volatility_60,
        _safe_ratio(volatility_15, volatility_60),
        _safe_ratio(close - window_low, span),
        _safe_ratio(recent_volume, baseline_volume),
        _safe_ratio(recent_trades, baseline_trades),
        _safe_ratio(high - body_top, bar_span),
        _safe_ratio(body_bottom - low, bar_span),
        _safe_ratio(close - vwap, vwap),
        consistency,
    ]


def default_policy(max_label_bars: int = MAX_LABEL_BARS) -> ExitPolicy:
    """The labelling policy, with its horizon bound to the label horizon.

    `ExitPolicy.max_holding_bars` defaults to 480 - the carry family's horizon, set
    when this policy served a different consumer. Left at 480 while labels look
    ahead 30 bars, the time rail could never fire and every label would come from a
    price rail, which is a different policy from the one the live bot runs.
    """
    return ExitPolicy(max_holding_bars=max_label_bars)


def build_rows(bars_by_symbol: dict, *, policy: ExitPolicy,
               side: str = "LONG",
               feature_window: int = FEATURE_WINDOW_BARS,
               max_label_bars: int = MAX_LABEL_BARS,
               stride: int = 5) -> TrainingRows:
    """Features at each decision bar, labelled by what the exit policy did next.

    `stride` samples every Nth bar rather than every bar. Adjacent rows over a
    60-bar window are almost the same row, and sampling them all inflates N without
    adding information — which is the same error `uniqueness_weights` corrects for
    downstream, applied here where it is cheaper.

    A row is labelled 1 when the policy's realised P&L was positive. That is the
    question the BULL brain actually faces: *if this trade were taken now under the
    policy that will manage it, would it end in profit* — not "will price be higher
    later", which nothing in the system acts on.
    """
    rows = TrainingRows()
    for symbol, bars in bars_by_symbol.items():
        if len(bars) < MIN_BARS_PER_SYMBOL:
            rows._drop("TOO_FEW_BARS")
            continue
        last_start = len(bars) - max_label_bars - 1
        for i in range(feature_window - 1, last_start, stride):
            window = bars[i - feature_window + 1:i + 1]
            features = compute_features(window)
            if features is None:
                rows._drop("FEATURES_NOT_COMPUTABLE")
                continue
            # The path starts at the NEXT bar. An entry filled on the decision bar
            # would be an entry at a price the decision could not have known.
            path_bars = bars[i + 1:i + 1 + max_label_bars]
            if len(path_bars) < 2:
                rows._drop("PATH_TOO_SHORT")
                continue
            entry = path_bars[0].open
            if entry <= 0:
                rows._drop("NON_POSITIVE_ENTRY")
                continue
            try:
                # ATR measured on the trailing window - information available at
                # the decision bar, never from the path it is about to walk.
                atr = average_true_range([b.as_path_bar() for b in window])
                if atr <= 0:
                    rows._drop("NO_TRUE_RANGE")
                    continue
                record = run_exit_policy(
                    side, Decimal(str(entry)),
                    [b.as_path_bar() for b in path_bars],
                    atr_at_entry=atr, policy=policy)
            except Exception:                                  # noqa: BLE001
                # A corrupt bar or a degenerate path. Counted, never defaulted.
                rows._drop("EXIT_POLICY_REFUSED")
                continue
            if record is None:
                rows._drop("UNRESOLVED_WITHIN_HORIZON")
                continue

            pnl = float(record.pnl_per_unit)
            rows.features.append(features)
            rows.labels.append(1 if pnl > 0 else 0)
            rows.spans.append(LabelSpan(event_index=i,
                                        touched_at_index=i + record.holding_bars))
            rows.symbols.append(symbol)
            rows.event_times_ns.append(bars[i].event_time_ns)
            rows.outcomes.append(pnl / entry if entry else 0.0)
            rows.holding_bars.append(record.holding_bars)
            rows.max_favourable.append(float(record.max_favourable_excursion))
            rows.max_adverse.append(float(record.max_adverse_excursion))
    return rows


def bars_from_frame(frame) -> dict:
    """Group a store frame into per-symbol, time-ordered `Bar` lists.

    Rows whose OHLC cannot form a valid bar are dropped and counted by the caller
    rather than repaired: `deterministic_exit.Bar` refuses a corrupt bar on
    purpose, and silently mending one here would hand the model a price the market
    never printed.
    """
    ordered = frame.sort_values(["symbol", "event_time_ns"])
    grouped: dict = {}
    for symbol, block in ordered.groupby("symbol", sort=False):
        bars = []
        for row in block.itertuples(index=False):
            try:
                high, low = float(row.high), float(row.low)
                close, open_ = float(row.close), float(row.open)
            except (TypeError, ValueError):
                continue
            # The same refusals `deterministic_exit.Bar` makes, applied here so a
            # corrupt bar never reaches a feature row either. Repairing one would
            # hand the model a price the market never printed.
            if min(high, low, close, open_) <= 0 or low > high:
                continue
            bars.append(MarketBar(
                event_time_ns=int(row.event_time_ns), open=open_, high=high,
                low=low, close=close, volume=float(row.volume or 0),
                trades=float(getattr(row, "trades", 0) or 0)))
        if bars:
            grouped[symbol] = bars
    return grouped


# ---------------------------------------------------------------------------
# Reading the store for training. RESEARCH path, not a trading path.
#
# CLAUDE.md keeps the two apart on purpose: research and training reads go
# through Layer 1, and trading prices come from `live.live_feed` (RL-024). This
# is the research side, and it is the only place in the learned-brain stack that
# touches the store at all.
# ---------------------------------------------------------------------------

STORE_BARS = Path.home() / "capture" / "store" / "bars_60000000000ns"
_STORE_COLUMNS = ("symbol", "venue", "event_time_ns", "open", "high", "low",
                  "close", "volume", "trades")

# Which venues each segment's model is fitted on. **RL-019: each segment is its
# own bot with its own data**, and a shared dataset makes that false however
# separate the code looks.
#
# Measured 2026-08-18: perp and spot were fitted on the same pooled bars and
# produced byte-identical accuracy (0.5329 against a 0.5126 base rate) - two
# models that were one model wearing two aliases. The venue column is what
# separates them.
#
# `dated` and `options` are deliberately absent. Their brains reason about basis,
# time to expiry and implied volatility, none of which is in a bar; a bar-fitted
# direction model would be answering a question they do not ask. They keep their
# rule brains until they have their own datasets, and the board says so.
SEGMENT_VENUES = {
    "perp": ("binance", "hyperliquid"),
    "spot": ("binance-spot", "coinbase"),
}


def _symbol_from_path(path: str) -> str | None:
    """The symbol out of a hive path segment `symbol=BTCUSDT`."""
    for part in path.split("/"):
        if part.startswith("symbol="):
            value = part.split("=", 1)[1]
            return value or None
    return None


def read_recent_bars(n_hours: int = 14, root: Path = STORE_BARS,
                     venues=None) -> tuple[dict, dict]:
    """Per-symbol bar lists from the most recent `n_hours` hour partitions.

    **One hour DIRECTORY at a time, never `ds.dataset(root)`.** Opening the whole
    dataset discovers every fragment under all 67 partitions before a filter can
    prune anything - measured 2026-08-18, that read had not returned after ten
    minutes and drove memory to 25 GB of 29, which is the OOM that killed the paper
    engine on 2026-08-09. Pointing the reader at one hour opens only that hour.

    `partitioning="hive"` is required: the layout is
    `availability_hour=X/symbol=Y/part.parquet`, so `symbol` is a partition key and
    not a column in the files.

    Returns `(bars_by_symbol, report)`. The report names every hour that was
    skipped and why - a training set quietly built from half the hours it was asked
    for is a training set whose coverage nobody knows.
    """
    import pyarrow.dataset as ds

    root = Path(root)
    hours = sorted(h.name for h in root.iterdir()
                   if h.name.startswith("availability_hour="))[-n_hours:]
    bars: dict = {}
    wanted_venues = set(venues) if venues else None
    report = {"hours_requested": len(hours), "hours_read": 0, "skipped": {},
              "rows": 0, "venues": sorted(wanted_venues) if wanted_venues else "all"}
    for hour in hours:
        # **Whole hour first, fragments only as a fallback.**
        #
        # A whole-hour read is one scan and is fast. Per-fragment reading is
        # resilient but measured ~40x slower - 17,610 fragments in three hours,
        # each with its own open and close - which makes an hourly retrain loop
        # impossible. So: try the cheap path, and drop to the expensive one only
        # for the hours that actually need it, which are the newest ones where
        # capture has a file mid-write.
        #
        # `symbol` is a PARTITION KEY and is not in a fragment's physical schema,
        # so the fallback path reads it from the hive path instead of asking the
        # file for it.
        file_columns = [c for c in _STORE_COLUMNS if c != "symbol"]
        columns = {c: [] for c in _STORE_COLUMNS}
        try:
            dataset = ds.dataset(root / hour, format="parquet", partitioning="hive")
        except Exception as exc:                              # noqa: BLE001
            reason = f"hour:{type(exc).__name__}"
            report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
            continue

        try:
            table = dataset.to_table(columns=list(_STORE_COLUMNS))
            for c in _STORE_COLUMNS:
                columns[c] = table.column(c).to_pylist()
            del table
            report.setdefault("hours_whole", 0)
            report["hours_whole"] += 1
        except Exception:                                     # noqa: BLE001
            report.setdefault("hours_by_fragment", 0)
            report["hours_by_fragment"] += 1
            for fragment in dataset.get_fragments():
                try:
                    piece = fragment.to_table(columns=file_columns)
                except Exception as exc:                      # noqa: BLE001
                    reason = f"fragment:{type(exc).__name__}"
                    report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
                    continue
                symbol = _symbol_from_path(str(fragment.path))
                if symbol is None:
                    report["skipped"]["no_symbol_in_path"] = (
                        report["skipped"].get("no_symbol_in_path", 0) + 1)
                    continue
                for c in file_columns:
                    columns[c].extend(piece.column(c).to_pylist())
                columns["symbol"].extend([symbol] * piece.num_rows)
                del piece
        if not columns["symbol"]:
            continue
        for i in range(len(columns["symbol"])):
            try:
                high = float(columns["high"][i]); low = float(columns["low"][i])
                close = float(columns["close"][i]); open_ = float(columns["open"][i])
            except (TypeError, ValueError):
                continue
            if min(high, low, close, open_) <= 0 or low > high:
                continue
            if wanted_venues is not None and columns["venue"][i] not in wanted_venues:
                report["skipped"]["other_venue"] = (
                    report["skipped"].get("other_venue", 0) + 1)
                continue
            # Keyed by (venue, symbol): BTCUSDT on binance futures and BTCUSDT on
            # binance spot are different instruments, and merging their bars would
            # interleave two price series into one nonsensical path.
            bars.setdefault(f"{columns['venue'][i]}:{columns['symbol'][i]}", []).append(MarketBar(
                event_time_ns=int(columns["event_time_ns"][i]), open=open_,
                high=high, low=low, close=close,
                volume=float(columns["volume"][i] or 0),
                trades=float(columns["trades"][i] or 0)))
            report["rows"] += 1
        report["hours_read"] += 1
    for symbol in bars:
        bars[symbol].sort(key=lambda b: b.event_time_ns)
    report["symbols"] = len(bars)
    return bars, report
