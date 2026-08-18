"""BF-02: the live feature frame — what the brains read, computed from the ticks a poll delivered.

## Rolling state, and why it lives here rather than in the feed

A single poll cannot produce a return or a realized volatility; both need a history.
The feed is deliberately stateless past its buffer — it hands over ticks and forgets
them — so the memory lives here, one small ring per symbol.

That placement matters for a reason beyond tidiness: a bot restarting must not
inherit a stale view of the market. This state is in-process and dies with the
engine, so a restarted bot has NO features until it has watched live ticks for its
warm-up window, and it refuses to trade during it. The alternative — priming from the
store — is the exact thing RL-024 forbids, and it would let a bot's first trades after
every restart be taken on hours-old prices.

## A refusal names what was missing; nothing is defaulted

Repo rule, and `perp.tradable_universe` already works this way: a missing or stale
input produces a refusal naming what was missing rather than a default. A zero
spread, an assumed mid, a volatility of 0.0 standing in for "not enough samples" —
each is a number that reads as information and is not. `FrameRefused` carries the list.

The specific trap: a feature frame that quietly returns zeros makes every brain
decline for reasons that look like market conditions, and the bot goes quiet while
every log line reads normal.

## Order flow imbalance needs the taker side, and the sign is easy to get backwards

Binance sends `m` = is-buyer-maker. `m` true means the resting order was the buyer,
so the AGGRESSOR was a seller. `live.live_feed` resolves that once, into
`extra["taker_side"]`, and nothing downstream re-derives it. Inverting this sign
produces a feature that is exactly as strong as the correct one and points the wrong
way, which backtests beautifully and loses money live.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal

# How many mid observations a symbol needs before a return or a volatility is
# reported. Below this the frame refuses rather than reporting a number computed from
# two points, which is a number with no error bar wearing the same units as one that has.
MIN_SAMPLES = 12

# The rolling window, IN TIME. Sixty seconds of live mid observations.
#
# **It was observation-counted first, and that was wrong in a way worth recording.**
# A 120-observation window sounds venue-neutral and is the opposite: measured
# 2026-08-18 on the live perp feed, `bookTicker` delivered ~2,750 updates in 30
# seconds for ONE symbol, so 120 observations spanned about a tenth of a second.
# Momentum over 0.1s is 0.0 to the precision of the quote, and the first live run of
# the perp bot abstained on 81 of 81 decisions with `momentum: 0.0` and a realized
# volatility of 1.5e-05. The same 120 observations on the 5-second Deribit poll would
# have spanned ten minutes. One constant, two segments, two windows three orders of
# magnitude apart - and neither is the window anybody intended.
#
# Time is what the thresholds in every brain are stated against, so time is what the
# window is measured in.
WINDOW_NS = 60_000_000_000
# Ring capacity. Bounded by memory rather than by meaning - the window above decides
# what is USED; this only decides how much can be held before the oldest is dropped.
MAX_OBSERVATIONS = 20_000


@dataclass(frozen=True)
class FrameRefused:
    """The frame could not be computed, and this names what was missing."""

    venue: str
    symbol: str
    missing: tuple[str, ...]
    at_ns: int

    def get(self, key, default=None):
        """Frames are read with `.get`; a refusal answers every key with the default.

        Deliberately NOT raising. A brain that asks a refused frame for a feature
        gets None and declines naming it, which is the behaviour wanted; raising
        would make one bad symbol take down a poll covering hundreds.
        """
        return default

    @property
    def is_refusal(self) -> bool:
        return True


class _SymbolState:
    """The rolling memory for one instrument."""

    __slots__ = ("mids", "last_trade", "buy_volume", "sell_volume", "trade_count",
                 "last_quote_ns", "last_trade_ns", "bid", "ask", "bid_size", "ask_size",
                 "extra")

    def __init__(self) -> None:
        # (received_ns, mid). Timestamped so the window is a duration, not a count.
        self.mids: deque = deque(maxlen=MAX_OBSERVATIONS)
        self.last_trade: Decimal | None = None
        self.buy_volume = Decimal(0)
        self.sell_volume = Decimal(0)
        self.trade_count = 0
        self.last_quote_ns: int | None = None
        self.last_trade_ns: int | None = None
        self.bid: Decimal | None = None
        self.ask: Decimal | None = None
        self.bid_size: Decimal | None = None
        self.ask_size: Decimal | None = None
        self.extra: dict = {}


class LiveFeatureFrames:
    """Consumes each poll's ticks and answers with one frame per symbol.

    `update()` then `frames()`, every poll. The two are separate because a poll may
    deliver ticks for symbols the bot is not admitted to trade, and admission is
    SB-01/DB-01/OB-01/PB-01's decision rather than this module's.
    """

    def __init__(self, *, segment: str, window_ns: int = WINDOW_NS,
                 min_samples: int = MIN_SAMPLES) -> None:
        self.segment = segment
        self._window_ns = window_ns
        self._min_samples = min_samples
        self._state: dict[tuple[str, str], _SymbolState] = defaultdict(_SymbolState)

    def update(self, ticks) -> int:
        """Fold a poll's ticks into the rolling state. Returns how many were used."""
        used = 0
        for tick in ticks:
            state = self._state[(tick.venue, tick.symbol)]
            if tick.bid is not None or tick.ask is not None:
                if tick.bid is not None:
                    state.bid = tick.bid
                    state.bid_size = tick.bid_size
                if tick.ask is not None:
                    state.ask = tick.ask
                    state.ask_size = tick.ask_size
                state.last_quote_ns = tick.received_ns
                mid = tick.mid
                if mid is None and state.bid is not None and state.ask is not None:
                    # The tick carried one side; the other is the last one seen. This
                    # is a composed quote, not an invented one - both halves were
                    # printed by the venue.
                    if state.ask >= state.bid:
                        mid = (state.bid + state.ask) / 2
                if mid is not None:
                    state.mids.append((tick.received_ns, float(mid)))
                used += 1
            if tick.price is not None:
                state.last_trade = tick.price
                state.last_trade_ns = tick.received_ns
                state.trade_count += 1
                quantity = tick.quantity or Decimal(0)
                if tick.extra.get("taker_side") == "SELL":
                    state.sell_volume += quantity
                else:
                    state.buy_volume += quantity
                used += 1
            if tick.extra:
                state.extra.update(
                    {k: v for k, v in tick.extra.items() if k != "taker_side"})
        return used

    def symbols(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._state.keys())

    def frame(self, venue: str, symbol: str, now_ns: int) -> dict | FrameRefused:
        """One symbol's features, or a refusal naming what was missing."""
        state = self._state.get((venue, symbol))
        if state is None:
            return FrameRefused(venue=venue, symbol=symbol,
                                missing=("no_ticks_seen",), at_ns=now_ns)

        # Only observations inside the window count. An instrument that stopped
        # quoting keeps its old points in the ring and must NOT be able to produce a
        # momentum from them - that would be a stale number wearing a live label.
        cutoff = now_ns - self._window_ns
        windowed = [mid for stamp, mid in state.mids if stamp >= cutoff]

        missing = []
        if state.bid is None or state.ask is None:
            missing.append("two_sided_quote")
        if len(windowed) < self._min_samples:
            missing.append(f"samples({len(windowed)}<{self._min_samples})")
        if missing:
            return FrameRefused(venue=venue, symbol=symbol,
                                missing=tuple(missing), at_ns=now_ns)

        bid, ask = state.bid, state.ask
        mid = (bid + ask) / 2
        relative_spread = (ask - bid) / mid if mid > 0 else None

        mids = windowed
        returns = [
            math.log(mids[i] / mids[i - 1])
            for i in range(1, len(mids))
            if mids[i] > 0 and mids[i - 1] > 0
        ]
        # Population standard deviation of log returns over the window. NOT
        # annualised: every consumer here is intraday (RL-018) and an annualised
        # number would have to be un-annualised by each of them, which is one
        # conversion per caller and one chance each to get it wrong.
        volatility = _stdev(returns)
        # **Two volatilities, because one of them is not comparable to anything a
        # brain reasons about.** `realized_volatility` above is the standard
        # deviation of ONE observation's log return - on a feed delivering thousands
        # of quote updates a minute that is ~1e-05, a number with no relationship to
        # a 25 basis point take-profit.
        #
        # Measured 2026-08-18: PROFIT-TAIL compared exactly those two and computed a
        # negative expectancy on every symbol of every poll, so the perp bot selected
        # 9 trades and took none of them. The units were the whole bug.
        #
        # `window_volatility` is the movement over the WINDOW - per-observation sigma
        # scaled by the root of the observation count. That is a fraction of price
        # over a stated duration, which is what a take-profit and a spread also are,
        # so the three can be compared without anybody converting anything.
        window_volatility = volatility * math.sqrt(len(returns)) if returns else 0.0
        # Return across the window: newest against the oldest still inside it.
        momentum = (mids[-1] - mids[0]) / mids[0] if mids[0] > 0 else 0.0
        window_span_ns = 0
        if len(state.mids) >= 2:
            inside = [stamp for stamp, _ in state.mids if stamp >= cutoff]
            if len(inside) >= 2:
                window_span_ns = inside[-1] - inside[0]

        flow_total = state.buy_volume + state.sell_volume
        order_flow_imbalance = (
            float((state.buy_volume - state.sell_volume) / flow_total)
            if flow_total > 0 else None)

        quote_age_ns = (now_ns - state.last_quote_ns
                        if state.last_quote_ns is not None else None)

        return {
            "segment": self.segment,
            "venue": venue,
            "symbol": symbol,
            "at_ns": now_ns,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last_trade": state.last_trade,
            "relative_spread": relative_spread,
            "realized_volatility": Decimal(str(volatility)),
            "window_volatility": Decimal(str(window_volatility)),
            "momentum": momentum,
            "order_flow_imbalance": order_flow_imbalance,
            "trade_count": state.trade_count,
            "buy_volume": state.buy_volume,
            "sell_volume": state.sell_volume,
            "samples": len(mids),
            # Published so a threshold can never be read against a window that was
            # not the one it was stated for.
            "window_span_ns": window_span_ns,
            "window_ns": self._window_ns,
            "quote_age_ns": quote_age_ns,
            "has_two_sided_quote": True,
            # Whatever the venue sent that has no shared shape: an option's mark IV
            # and underlying, a dated contract's delivery time and funding rate. The
            # segment brains that need these know their keys.
            **{f"venue_{k}": v for k, v in state.extra.items()},
        }

    def frames(self, now_ns: int, only=None) -> dict:
        """Every known symbol's frame, or just the admitted ones."""
        keys = self.symbols() if only is None else tuple(only)
        return {key: self.frame(key[0], key[1], now_ns) for key in keys}

    def reset_flow(self) -> None:
        """Clear the per-poll trade tallies.

        Called by the engine after a poll so order-flow imbalance measures the recent
        window rather than everything since the process started - a cumulative
        imbalance converges to a constant and stops being a signal, slowly enough
        that nothing looks broken.
        """
        for state in self._state.values():
            state.buy_volume = Decimal(0)
            state.sell_volume = Decimal(0)
            state.trade_count = 0


def _stdev(values) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
