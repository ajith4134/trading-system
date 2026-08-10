"""Delta vs price-hold: whether directional pressure actually moved price, or
was absorbed.

`FEATURES.md` §2 marks this `[MISSED]` with the case that names the whole
failure: *"Positive delta at a high is meaningless if price cannot hold - that
is absorption, not strength, and is bearish. Raw signed delta is as naive as
level-1 imbalance."* A bar can carry heavy, one-sided volume and still tell a
strategy nothing about direction, because the two facts that matter - how hard
the market pushed, and whether the push stuck - are not the same fact. Scoring
either one alone is the naivety the catalogue names; this module keeps them as
two separate fields and reports their conjunction (`absorption`), rather than
collapsing them into one score a reader would have to un-collapse to trust. A
row can say "strong push, price held" (`proxy_flow_strong=True,
price_held=True`, not absorption) as distinctly as it can say "strong push,
price gave it back" (absorption) - see `tests/test_absorption.py` for both,
plus the case that actually exercises the conjunction: a bar that failed to
hold with NO real push behind it, which must not read as absorption either.

## There is no signed delta in this store, and this module does not pretend there is

`bars_60000000000ns` carries `open, high, low, close, volume, trades` - no
taker-side or maker/taker-flagged volume, so no column here says which side of
a trade was the aggressor. Computing "delta" from close-versus-open and
calling it delta would be exactly the fabrication this repo keeps finding and
keeps being burned by: a number shaped like order flow that is actually a
candle description.

So every value this module produces is named a proxy - in the column
(`proxy_delta`), in the verdict field (`proxy_flow_strong`), and here: the
proxy is volume multiplied by the bar's own intrabar excursion away from open,
toward whichever of high/low it travelled further (`high - open` against
`open - low`, signed positive for the up side). That measures how hard price
was pushed away from where it started, weighted by how much traded while it
got there. It is NOT a classification of any individual trade's side, and it
cannot become one without trade-level data carrying an aggressor flag - e.g.
binance's `m` (buyer-is-maker) field on the raw trade stream, or a
Lee-Ready-classified tape - neither of which Layer 1 stores today.
`features.kyle_lambda` hit the identical wall and documented it precisely:
`capture.venues.binance` does not persist aggTrade's maker flag and
`capture.venues.coinbase` does not persist `match.side`, so no signed-flow
column exists anywhere downstream of capture. That module's proxy is not this
one's - it signs bar-to-bar volume by the tick rule
(`sign(close_t - close_{t-1})`) because a regression needs one flow number per
bar. Absorption is asking a different question, about the SHAPE of a single
bar rather than its relationship to the bar before it, so the proxy here is
built from that bar's own open/high/low/close instead. A future module reading
a real aggressor-flagged feed would compute real delta, and this module's
proxy would be the thing to delete, not the thing to extend.

## "Held" is measured against the bar's own range, never an absolute number

A basis-point threshold would be deaf to a $0.001 token and hysterical about
BTC. `retention` is the fraction of the bar's own dominant push that survived
into the close: 1.0 means the close sits at the extreme the push reached, 0.0
means it gave the whole move back to the open, negative means it reversed past
the open. Failing to hold is `retention < 0.5` - more of the push surrendered
by the close than kept. The boundary sits at half by disclosed choice, not
fit: it is the point past which "gave some back" becomes "gave most of it
back", and a bar retaining exactly half is treated as held rather than forced
into a tie it did not clearly lose.

## "Strong" is measured against the symbol's own recent bars, never a shared constant

What counts as forceful for a thinly-traded pair is routine for BTC. The
threshold is twice the *median* of `abs(proxy_delta)` over the trailing
window, excluding the bar being judged - nothing being judged enters the fit
that judges it, the same discipline `peg_monitor` uses for its dispersion
bands. The baseline floor, `_MIN_BASELINE_BARS = 20`, is lifted from
`staleness.CADENCE_SAMPLE` on purpose: it is the same question - how many
recent observations before a baseline means anything - asked of a different
series, and a second number here would drift from that one without either
being wrong.

## Refusal, not a default

A bar whose own range is zero (`high == low`) cannot say which way price was
pushed or whether it held - `retention` is undefined there, not zero. Scoring
it "no absorption" would be the confident-staleness failure in a new shape: a
value that could not be judged, reading identically to one that was judged and
came back calm. It is refused and counted instead, and so is a window with no
volume, a non-positive price, too little baseline history to trust a median,
or a bar whose own OHLC values are internally inconsistent (e.g. low above
high) - a corrupt row must never silently drive a verdict.

## Staleness (FE-001)

Every row carries `measure_staleness` against that (venue, symbol)'s own
event-time history: a feature value with no age attached is a value nobody can
tell has stopped being true.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATASET = "bars_60000000000ns"

# One hour of one-minute bars: long enough to see the symbol's routine push
# size across more than a few minutes of noise, short enough that yesterday's
# regime does not dominate today's baseline. Shorter than `peg_monitor`'s day-
# long window on purpose - flow strength moves faster than a peg's level does.
_LOOKBACK_BARS = 60
# The floor on baseline size before a median means anything. Lifted from
# `staleness.CADENCE_SAMPLE` - see the module docstring.
_MIN_BASELINE_BARS = 20
# Twice the symbol's own routine push counts as strong. See module docstring.
_STRONG_FLOW_MULTIPLE = Decimal("2")
# More of the push surrendered by the close than kept counts as not held.
_RETENTION_HOLD_THRESHOLD = Decimal("0.5")


@dataclass(frozen=True)
class AbsorptionTable:
    """The absorption verdict per (venue, symbol), and what could not be judged.

    `rows` carries the newest judged bar's verdict per key: `proxy_delta`
    (the volume-weighted push, signed, named `proxy` because it is not real
    signed order flow - see the module docstring), `retention` (how much of
    that push survived to the close, relative to the bar's own range), and
    `absorption` - the conjunction of `proxy_flow_strong` and NOT
    `price_held`, reported alongside both inputs rather than folded into one
    score. `refused` counts the keys that could not be judged, by reason.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def compute_absorption(store_root: Path, as_of_ns: int,
                       custodian=None) -> AbsorptionTable:
    """The latest judged bar's absorption verdict per (venue, symbol), at `as_of_ns`."""
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    refused = {"too_few_bars": 0, "non_positive_price": 0,
               "unparseable_price": 0, "invalid_bar_shape": 0,
               "zero_range_bar": 0, "no_volume": 0}
    if frame.empty:
        return AbsorptionTable(rows=_empty_rows(), refused=refused)

    history = frame
    out = {"venue": [], "symbol": [], "event_time_ns": [], "direction": [],
           "proxy_delta": [], "proxy_delta_threshold": [],
           "proxy_flow_strong": [], "retention": [], "price_held": [],
           "absorption": [], "absorption_signal": [], "bars_in_baseline": []}

    for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=True):
        recent = group.sort_values("event_time_ns").tail(_LOOKBACK_BARS)
        if len(recent) < _MIN_BASELINE_BARS + 1:
            refused["too_few_bars"] += 1
            continue

        parsed, reason = _parse_bars(recent)
        if reason is not None:
            refused[reason] += 1
            continue

        baseline, judged = parsed[:-1], parsed[-1]
        o, h, l, c, v, event_ns = judged

        if h == l:
            # No range to have pushed within - direction and retention are
            # both undefined, not zero. See module docstring.
            refused["zero_range_bar"] += 1
            continue

        total_volume = sum(bar[4] for bar in parsed)
        if total_volume <= 0:
            refused["no_volume"] += 1
            continue

        direction, push = _dominant_push(o, h, l)
        proxy_delta = v * push if direction == "up" else -(v * push)

        baseline_abs_deltas = []
        for (bo, bh, bl, bc, bv, _) in baseline:
            if bh == bl:
                baseline_abs_deltas.append(Decimal(0))
                continue
            _, b_push = _dominant_push(bo, bh, bl)
            baseline_abs_deltas.append(bv * b_push)
        threshold = _STRONG_FLOW_MULTIPLE * statistics.median(baseline_abs_deltas)
        flow_strong = abs(proxy_delta) > threshold

        retention = (c - o) / push if direction == "up" else (o - c) / push
        price_held = retention >= _RETENTION_HOLD_THRESHOLD

        absorption = bool(flow_strong and not price_held)
        signal = None
        if absorption:
            signal = "bearish" if direction == "up" else "bullish"

        out["venue"].append(venue)
        out["symbol"].append(symbol)
        out["event_time_ns"].append(event_ns)
        out["direction"].append(direction)
        out["proxy_delta"].append(proxy_delta)
        out["proxy_delta_threshold"].append(threshold)
        out["proxy_flow_strong"].append(bool(flow_strong))
        out["retention"].append(retention)
        out["price_held"].append(bool(price_held))
        out["absorption"].append(absorption)
        out["absorption_signal"].append(signal)
        out["bars_in_baseline"].append(len(baseline))

    rows = pd.DataFrame(out)
    # FE-001: age every row against that (venue, symbol)'s own full visible
    # history, not just the window judged - the cadence a key keeps is a
    # property of the whole series, and trimming it to the window would make
    # the routine gap depend on how far back this module happened to look.
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           int(as_of_ns))
        for (venue, symbol), group in history.groupby(["venue", "symbol"], sort=False)
    }
    return AbsorptionTable(rows=stamp(rows, ages, ["venue", "symbol"]), refused=refused)


def _dominant_push(o: Decimal, h: Decimal, l: Decimal) -> tuple[str, Decimal]:
    """Which side of open the bar travelled further toward, and by how much.

    Ties go to `up` - an arbitrary but disclosed choice; a bar with an
    identical up-push and down-push is symmetric and calling it either way is
    equally defensible, so the tie-break is named here rather than left to
    fall out of an unexplained `>=`.
    """
    push_up, push_down = h - o, o - l
    if push_up >= push_down:
        return "up", push_up
    return "down", push_down


def _parse_bars(recent: pd.DataFrame) -> tuple[list[tuple], str | None]:
    """OHLCV as Decimal for the whole window, or the one reason the key is refused.

    All-or-nothing per key: a median fitted on a baseline that silently
    dropped its bad rows would be a baseline nobody can trust, and a mix of
    real and refused bars reads exactly like a calm one.
    """
    parsed = []
    for row in recent.itertuples(index=False):
        values = (row.open, row.high, row.low, row.close, row.volume)
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return [], "unparseable_price"
        try:
            o, h, l, c, v = (Decimal(str(value)) for value in values)
        except InvalidOperation:
            return [], "unparseable_price"
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return [], "non_positive_price"
        if v < 0:
            return [], "unparseable_price"
        if not (l <= h and l <= o <= h and l <= c <= h):
            return [], "invalid_bar_shape"
        parsed.append((o, h, l, c, v, int(row.event_time_ns)))
    return parsed, None


def _empty_rows() -> pd.DataFrame:
    empty = pd.DataFrame({
        "venue": [], "symbol": [], "event_time_ns": [], "direction": [],
        "proxy_delta": [], "proxy_delta_threshold": [], "proxy_flow_strong": [],
        "retention": [], "price_held": [], "absorption": [],
        "absorption_signal": [], "bars_in_baseline": [],
    })
    return stamp(empty, {}, ["venue", "symbol"])
