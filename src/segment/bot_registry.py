"""BF-07: each segment bot declared in one place — its feed, its features, its three brains.

## Why a declaration and not four hard-wired engines

**RL-019, in the user's words:** *"each segment are like there own bots with its own
architecture, data features etc like this so make sure you do not assume and build
different from what..."*.

A declaration makes each segment's architecture READABLE — one row, four bots, no
segment inheriting another's brains by default, which is BF-07's acceptance in those
words. The alternative, four engines that each grew their own defaults, is how the
2026-08-02 examination-hall ruling got assigned once and read as resolved four times.

**What is shared is the harness, never the decisions.** The engine loop, the risk
gate, the fill model, the journal and the broker are one implementation used four
times — they are plumbing and four copies would be four places for the same bug. The
feed, the features, the three brains and every threshold are per segment and are
declared below.

## The four, and the honest state of each

| segment | venue | source | what its brains read | live today |
|---|---|---|---|---|
| perp | binance futures | websocket | order flow, momentum, funding | yes |
| spot | binance spot | websocket | momentum, aggressor flow (no divergence yet) | yes |
| dated | bybit linear | REST poll | annualised basis, time to expiry | yes |
| options | deribit | REST poll | mark IV against the chain median | yes, thin |

**The symbol lists are a starting universe, not the final one.** RL-009 and RL-014
want every instrument considered rather than a chosen few, and that is the direction
this grows. It starts bounded because a websocket URL has a byte budget
(`capture.venues.shard_by_url_budget` exists for exactly this) and because a first
live run should be verifiable by reading it. The count is published on every board
tile, so a narrow universe reads as narrow rather than as complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from dated import segment_brains as dated_brains
from dated import tradable_universe as dated_universe
from live import live_feed
from options import segment_brains as options_brains
from options import tradable_universe as options_universe
from perp import segment_brains as perp_brains
from perp import tradable_universe as perp_universe
from spot import segment_brains as spot_brains
from spot import tradable_universe as spot_universe

PERP = "perp"
SPOT = "spot"
DATED = "dated"
OPTIONS = "options"
SEGMENTS = (PERP, SPOT, DATED, OPTIONS)

# The perp and spot starting universes. The three majors carry the deepest book and
# the densest tape, which `perp.tradable_universe` measured as the only symbols with
# `book` coverage at all.
_MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# Beyond the majors: liquid enough to quote continuously, varied enough that the
# brains are not tested on three correlated instruments.
_BROADER = ["BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
            "LINKUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT"]


@dataclass(frozen=True)
class SegmentBot:
    """One segment's complete declaration. Nothing about it is inferred elsewhere."""

    segment: str
    venue: str
    build_feed: object
    bull: object
    bear: object
    profit_tail: object
    admit: object
    describe_universe: object
    # Order size in the instrument's own units. Small and fixed: sizing off an
    # uncalibrated confidence is the error `dual-agent-spec.md` names, so until the
    # brains are calibrated every order is the same size and the P&L reflects the
    # signal rather than a sizing rule nobody validated.
    quantity: Decimal
    # Passed to `segment.arbiter.select`. Per segment because the segments' evidence
    # is not equally strong: a dated basis is an observable convergence force, an
    # options IV edge is relative value against a thin chain.
    min_confidence: Decimal
    min_margin: Decimal
    max_loss_tail: Decimal | None
    band: str
    # Extra keyword the brains take, if any. The options brains need the chain median,
    # which is a property of the whole poll rather than of one instrument.
    # The feature window, per segment, because the feeds do not deliver at the same
    # rate. **Measured 2026-08-18:** a shared 60-second window with a 12-observation
    # floor refused every frame on both REST-polled segments - 829 of 829 on dated,
    # 786 of 786 on options - because a 10-second poll can only put 6 observations
    # inside 60 seconds. The floor is what makes a window meaningful and the cadence
    # is what makes it reachable, so the two are declared together, per segment.
    feature_window_ns: int = 60_000_000_000
    min_samples: int = 12
    # The hard stop the RISK GATE sets at fill, as a fraction of entry price.
    #
    # **Per segment because "price" means different things.** On a perp, spot pair or
    # dated future the price is the instrument's value and 1% is a real move. On an
    # option the price is the PREMIUM: a 1% move in the underlying can be a 10% move
    # in the premium, and the quoted spread alone can be 5%.
    #
    # Measured 2026-08-18 with a shared 1% stop: the options bot opened 156 positions
    # and closed 155 of them as STOP_BREACHED with ZERO wins. Entry crosses the spread,
    # the position is marked at the other side, and the round trip was already past a
    # 1% stop before the market moved at all. Every trade stopped out on the spread.
    hard_stop_fraction: Decimal = Decimal("0.010")
    brains_need_chain_median: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _perp_feed():
    return live_feed.websocket_feed(
        venue="binance-futures",
        url=live_feed.binance_futures_url(_MAJORS + _BROADER),
        detail=f"trade+bookTicker, {len(_MAJORS + _BROADER)} perpetuals")


def _spot_feed():
    return live_feed.websocket_feed(
        venue="binance-spot",
        url=live_feed.binance_spot_url(_MAJORS + _BROADER),
        detail=f"trade+bookTicker, {len(_MAJORS + _BROADER)} spot pairs")


def _dated_feed():
    return live_feed.rest_poll_feed(
        venue="bybit",
        url="https://api.bybit.com/v5/market/tickers?category=linear",
        fan_out=live_feed.fan_out_bybit_linear,
        interval_seconds=5.0,
        detail="v5 linear tickers, whole board, dated split out by deliveryTime")


def _deribit_feed():
    return live_feed.rest_poll_feed(
        venue="deribit",
        url=("https://www.deribit.com/api/v2/public/"
             "get_book_summary_by_currency?currency=BTC&kind=option"),
        fan_out=live_feed.fan_out_deribit_chain,
        interval_seconds=10.0,
        detail="get_book_summary_by_currency BTC options, whole chain per request")


def _registry() -> dict[str, SegmentBot]:
    return {
        PERP: SegmentBot(
            segment=PERP, venue="binance-futures", build_feed=_perp_feed,
            bull=perp_brains.PerpBullBrain(), bear=perp_brains.PerpBearBrain(),
            profit_tail=perp_brains.perp_profit_tail("fast"),
            admit=perp_universe.admit_live,
            describe_universe=perp_universe.describe_live,
            quantity=Decimal("0.002"),
            min_confidence=Decimal("0.55"), min_margin=Decimal("0.05"),
            max_loss_tail=Decimal("0.05"), band="fast",
            notes="RL-022: directional scalping, fast band. PB-06's slow band runs "
                  "as a second supervised process on the same declaration."),
        SPOT: SegmentBot(
            segment=SPOT, venue="binance-spot", build_feed=_spot_feed,
            bull=spot_brains.SpotBullBrain(), bear=spot_brains.SpotBearBrain(),
            profit_tail=spot_brains.spot_profit_tail("fast"),
            admit=spot_universe.admit,
            describe_universe=spot_universe.describe,
            quantity=Decimal("0.002"),
            min_confidence=Decimal("0.58"), min_margin=Decimal("0.06"),
            max_loss_tail=Decimal("0.05"), band="fast",
            notes="No funding and no leverage; cross-venue divergence is not yet "
                  "available and every decision says so."),
        DATED: SegmentBot(
            segment=DATED, venue="bybit", build_feed=_dated_feed,
            bull=dated_brains.DatedBullBrain(), bear=dated_brains.DatedBearBrain(),
            profit_tail=dated_brains.dated_profit_tail("slow"),
            admit=dated_universe.admit,
            describe_universe=dated_universe.describe,
            quantity=Decimal("0.010"),
            min_confidence=Decimal("0.55"), min_margin=Decimal("0.05"),
            max_loss_tail=Decimal("0.08"), band="slow",
            feature_window_ns=300_000_000_000,   # 5s poll x 60 observations
            min_samples=12,
            notes="Basis convergence, not momentum. Perpetuals in the same response "
                  "are rejected as NOT_DATED and the count is published."),
        OPTIONS: SegmentBot(
            segment=OPTIONS, venue="deribit", build_feed=_deribit_feed,
            bull=options_brains.OptionsBullBrain(),
            bear=options_brains.OptionsBearBrain(),
            profit_tail=options_brains.options_profit_tail("slow"),
            admit=options_universe.admit,
            describe_universe=options_universe.describe,
            quantity=Decimal("0.100"),
            min_confidence=Decimal("0.55"), min_margin=Decimal("0.05"),
            max_loss_tail=Decimal("0.30"), band="slow",
            hard_stop_fraction=Decimal("0.35"),
            feature_window_ns=600_000_000_000,   # 10s poll x 60 observations
            min_samples=12,
            brains_need_chain_median=True,
            notes="Relative value on implied volatility. An option is never taken as "
                  "a plain directional bet; the history cannot be backfilled."),
    }


def segment_bot(segment: str) -> SegmentBot:
    registry = _registry()
    if segment not in registry:
        raise ValueError(
            f"unknown segment {segment!r}; the four are {', '.join(SEGMENTS)}")
    return registry[segment]


def all_segments() -> tuple[str, ...]:
    return SEGMENTS
