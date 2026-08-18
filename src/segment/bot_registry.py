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
from live import live_feed, universe_discovery
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

# **No typed symbol lists.** Every universe comes from `live.universe_discovery`,
# which asks the venue what it lists (BF-09, RL-009, RL-014). Thirteen symbols were
# typed here until 2026-08-18; the venues list 570 perpetuals, 1,361 spot pairs, 48
# dated crypto contracts and 1,442 options.
#
# Discovery raising is deliberate and is NOT caught here. A bot started against an
# empty universe admits nothing, journals nothing and reads on the board exactly like
# a bot in a market with no opportunities - so a failed listing must stop the start,
# and the supervisor's restart loop is what retries it.


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
    # **A wide universe needs a longer window, and this is why.** Measured 2026-08-18
    # on the full board: `!bookTicker` delivered ~4 quote updates per symbol per 30
    # seconds averaged across 740 perpetuals - the majors get hundreds and the long
    # tail gets a handful. A 60-second window with a 12-sample floor therefore admits
    # the majors and refuses most of the tail, which would quietly hand back the
    # breadth RL-009 and RL-014 asked for while the code looked unchanged.
    #
    # 180 seconds at a 10-sample floor keeps the tail eligible. It is still intraday
    # (RL-018) and still far inside the fast band's five-minute hold, and the window
    # is published on every frame so no threshold is read against the wrong one.
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
    # How many positions this bot may hold at once.
    #
    # **It was 8, hardcoded in the engine, and 8 is a limit for a 13-symbol universe.**
    # Against the full board it binds immediately: measured 2026-08-18, the perp bot
    # selected 104 trades in its first four polls and could act on eight of them, so
    # the breadth RL-009 and RL-014 asked for was scanned and then discarded at the
    # gate. Scanning everything and being able to act on almost none of it is the same
    # outcome as never scanning it, arrived at more expensively.
    #
    # Still a real limit, not a removal - it bounds how much of one bot's book can be
    # wrong at once, and every order still passes the spread and price checks.
    max_open_positions: int = 40
    brains_need_chain_median: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _perp_feed():
    """Every listed perpetual: two all-market streams plus sharded trade streams."""
    universe = universe_discovery.discover("perp")
    symbols = universe.symbols
    urls = live_feed.binance_futures_all_market_urls()
    urls += live_feed.binance_trade_shard_urls(
        "wss://fstream.binance.com/stream?streams=", symbols)
    return live_feed.composite_feed(
        venue="binance-futures", websocket_urls=urls,
        rest_endpoints=[(live_feed.BINANCE_PREMIUM_INDEX,
                         live_feed.binance_funding_fan_out(symbols))],
        rest_interval_seconds=10.0,
        detail=(f"!bookTicker whole board + @trade for {len(symbols)} perpetuals "
                f"across {len(urls)} sockets, funding via premiumIndex poll"))


def _spot_feed():
    """Every listed spot pair: one all-market quote stream plus sharded trades."""
    universe = universe_discovery.discover("spot")
    symbols = universe.symbols
    # No all-market quote stream exists on spot any more - see
    # `binance_spot_all_market_url`. Quotes and trades are both per symbol here.
    urls = live_feed.binance_quote_and_trade_shard_urls(
        "wss://stream.binance.com:9443/stream?streams=", symbols)
    return live_feed.websocket_feed(
        venue="binance-spot", urls=urls,
        detail=(f"@bookTicker + @trade for {len(symbols)} spot pairs across "
                f"{len(urls)} connections; the venue serves no all-market quote stream"))


def _dated_feed():
    """Every listed dated crypto contract, across binance and bybit.

    Three endpoints because no one of them carries what the brains need: binance
    `premiumIndex` for mark and index, binance `ticker/bookTicker` for the two-sided
    quote, and bybit's own board. Until 2026-08-18 this segment read bybit alone.
    """
    universe = universe_discovery.discover("dated")
    delivery = {i.symbol: i.detail.get("delivery_ns")
                for i in universe.instruments if i.venue == "binance-futures"}
    endpoints = [
        (live_feed.BINANCE_PREMIUM_INDEX, live_feed.binance_dated_fan_out(delivery)),
        (live_feed.BINANCE_BOOK_TICKER, live_feed.binance_dated_quote_fan_out(delivery)),
        ("https://api.bybit.com/v5/market/tickers?category=linear",
         live_feed.fan_out_bybit_linear),
        ("https://api.bybit.com/v5/market/tickers?category=inverse",
         live_feed.fan_out_bybit_linear),
    ]
    return live_feed.multi_rest_poll_feed(
        venue="dated-multi", endpoints=endpoints, interval_seconds=5.0,
        detail=(f"{len(universe.instruments)} dated contracts: "
                f"{len(delivery)} binance, {len(universe.instruments) - len(delivery)} "
                f"bybit; premiumIndex + bookTicker + both bybit boards"))


def _deribit_feed():
    """The whole option chain, every currency Deribit lists it for."""
    endpoints = [
        (live_feed.DERIBIT_CHAIN_URL.format(currency=currency),
         live_feed.fan_out_deribit_chain)
        for currency in universe_discovery.DERIBIT_CURRENCIES
    ]
    return live_feed.multi_rest_poll_feed(
        venue="deribit", endpoints=endpoints, interval_seconds=10.0,
        detail=("get_book_summary_by_currency for "
                f"{', '.join(universe_discovery.DERIBIT_CURRENCIES)}; "
                "whole chain per request per currency"))


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
            feature_window_ns=180_000_000_000,   # see the wide-universe note below
            min_samples=10,
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
            feature_window_ns=180_000_000_000,   # see the wide-universe note below
            min_samples=10,
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
            max_open_positions=60,
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
