"""BF-09: every instrument a venue lists, so a segment's universe is the venue's and not a typed list.

## The ruling, and the measurement that showed it was not being met

**RL-014, in the user's words:** *"insted of picking and selecting few and trying to get
a profit trade from them my idea of considering all trades that exist in all segments"*.
**RL-009** says the same about not skipping symbols with profit potential.

The four bots started on thirteen hand-typed symbols. Measured 2026-08-18, the venues
list:

| segment | typed | listed |
|---|---|---|
| perp | 13 | **570** binance perpetuals |
| spot | 13 | **1,361** binance spot pairs |
| dated | 40 | **48** — 4 binance quarterlies, 40 bybit linear, 4 bybit inverse |
| options | 786 | **1,442** — Deribit BTC 786 and ETH 656 |

A universe that is a typed list is a universe that is whatever somebody remembered on
the day.

**The dated row was 217 in the first draft of this module, and the correction is the
more useful fact.** That count came from treating any contract with a delivery date as
dated, which admitted 169 `TRADIFI_PERPETUAL` contracts — Tesla, Intel, gold, silver,
Korean and Hong Kong equities — each carrying a delivery date set to the year 2100.
They are perpetuals, so a basis-convergence brain would have found no convergence in
them, and they are not crypto, which the spec puts out of scope in one line. The
universe count went UP, which is what breadth is supposed to look like. See
`DATED_CONTRACT_TYPES`.

## Discovery failing is not an empty market

Every function here raises `DiscoveryFailed` rather than returning an empty list. The
two are indistinguishable downstream and mean opposite things: an empty universe from
a failed HTTP call produces a bot that admits nothing, journals nothing, and reads on
the board exactly like a bot in a market with no opportunities.

## What is dropped is named, and counted

`Universe.dropped` carries one entry per instrument excluded at DISCOVERY, with the
reason — not trading, no contract type, a quote asset the segment cannot settle. This
is deliberately separate from the per-segment `tradable_universe` admission decision,
which excludes on LIVE evidence (stale quote, no two-sided market, thin tape).

Two different questions, asked at different times, and merging them would lose the
distinction between *the venue does not list this* and *the venue lists it and it is
not quoting right now*.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

_TIMEOUT = 30
_HEADERS = {"User-Agent": "ajit-universe-discovery/1.0"}

BINANCE_FUTURES_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_SPOT_INFO = "https://api.binance.com/api/v3/exchangeInfo"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers?category={category}"
DERIBIT_CHAIN = ("https://www.deribit.com/api/v2/public/"
                 "get_book_summary_by_currency?currency={currency}&kind=option")

PERPETUAL = "PERPETUAL"
NOT_TRADING = "NOT_TRADING"
NOT_PERPETUAL = "NOT_PERPETUAL"
NOT_DATED = "NOT_DATED"
NOT_CRYPTO = "NOT_CRYPTO"

# **The contract types that are genuinely dated crypto futures, and nothing else.**
#
# Measured 2026-08-18, binance futures `contractType`: PERPETUAL 570,
# TRADIFI_PERPETUAL 169, CURRENT_QUARTER 2, NEXT_QUARTER 2.
#
# `TRADIFI_PERPETUAL` is the trap. Those 169 are Tesla, Intel, gold, silver, platinum,
# palladium and Korean, Hong Kong and Chinese equities - and every one of them carries
# `deliveryDate: 4133404800000`, the year 2100. A "dated" test of `deliveryTime > 0`
# therefore admits all 169, and the dated bot's universe reads 173 instead of 4.
#
# It fails twice over. They are PERPETUALS, so a basis-convergence brain would be
# trading an instrument with no convergence to trade. And they are not crypto, which
# `~/research/bull-bear-profit-agents-spec.md` §1 puts outside scope in one line.
#
# The failure direction is the dangerous one: the universe count goes UP, which looks
# like the breadth RL-014 asked for.
DATED_CONTRACT_TYPES = ("CURRENT_QUARTER", "NEXT_QUARTER",
                        "CURRENT_QUARTER_DELIVERING", "NEXT_QUARTER_DELIVERING")
# Crypto only. COIN is a coin-margined or coin-underlying contract; INDEX is a crypto
# index. EQUITY, COMMODITY, KR_EQUITY, HK_EQUITY, CN_EQUITY and PREMARKET are not.
CRYPTO_UNDERLYING = ("COIN", "INDEX")


class DiscoveryFailed(RuntimeError):
    """A venue listing could not be read.

    Raised rather than returning nothing, because a bot handed an empty universe
    trades nothing and looks identical to a bot in a quiet market.
    """


@dataclass(frozen=True)
class Instrument:
    """One listed instrument, with what the venue said about it."""

    venue: str
    symbol: str
    kind: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Universe:
    """What a venue lists for one segment, and what discovery dropped."""

    segment: str
    instruments: tuple
    dropped: tuple

    @property
    def symbols(self) -> list:
        return [i.symbol for i in self.instruments]

    def describe(self) -> dict:
        reasons: dict = {}
        for _, reason in self.dropped:
            reasons[reason] = reasons.get(reason, 0) + 1
        venues: dict = {}
        for instrument in self.instruments:
            venues[instrument.venue] = venues.get(instrument.venue, 0) + 1
        return {
            "segment": self.segment,
            "listed": len(self.instruments) + len(self.dropped),
            "admitted_at_discovery": len(self.instruments),
            "dropped_at_discovery": len(self.dropped),
            "dropped_by_reason": reasons,
            "by_venue": venues,
        }


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise DiscoveryFailed(f"{url}: {type(exc).__name__}: {exc}") from exc


def _binance_symbols(payload: dict) -> list:
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise DiscoveryFailed("binance exchangeInfo carried no symbols")
    return symbols


def discover_binance_perpetuals() -> Universe:
    """Every binance USD-M perpetual that is currently trading."""
    rows = _binance_symbols(_get(BINANCE_FUTURES_INFO))
    kept, dropped = [], []
    for row in rows:
        symbol = row.get("symbol")
        if row.get("status") != "TRADING":
            dropped.append((symbol, NOT_TRADING))
            continue
        if row.get("contractType") != PERPETUAL:
            # TRADIFI_PERPETUAL lands here too, and correctly: it IS a perpetual but
            # it is a perpetual on Tesla or gold, and this is a crypto bot.
            dropped.append((symbol, NOT_PERPETUAL))
            continue
        if row.get("underlyingType") not in CRYPTO_UNDERLYING:
            dropped.append((symbol, NOT_CRYPTO))
            continue
        kept.append(Instrument(
            venue="binance-futures", symbol=symbol, kind=PERPETUAL,
            detail={"quote_asset": row.get("quoteAsset"),
                    "underlying_type": row.get("underlyingType"),
                    "price_precision": row.get("pricePrecision")}))
    return Universe("perp", tuple(kept), tuple(dropped))


def discover_binance_spot() -> Universe:
    """Every binance spot pair that is currently trading."""
    rows = _binance_symbols(_get(BINANCE_SPOT_INFO))
    kept, dropped = [], []
    for row in rows:
        symbol = row.get("symbol")
        if row.get("status") != "TRADING":
            dropped.append((symbol, NOT_TRADING))
            continue
        kept.append(Instrument(
            venue="binance-spot", symbol=symbol, kind="SPOT",
            detail={"quote_asset": row.get("quoteAsset")}))
    return Universe("spot", tuple(kept), tuple(dropped))


def discover_dated_futures() -> Universe:
    """Every dated contract across binance and both bybit categories.

    **Three sources, because no one of them is the market.** Binance lists 4 genuinely
    dated crypto contracts, bybit's linear board 40 and its inverse board 4.

    **A contract is dated only if the venue names a dated contract TYPE.** Not if it
    has a delivery date - see `DATED_CONTRACT_TYPES` for the 169 TradFi perpetuals
    that carry one set to the year 2100. Not if its name looks dated either;
    `capture.venues.bybit` established that a name looking dated to a human is not
    evidence.

    Bybit is the exception that proves the rule, and is handled differently on
    purpose: its tickers response has no contract-type field at all, so
    `deliveryTime` is the only signal it offers. That is acceptable there because
    bybit's linear and inverse boards are crypto throughout - it lists no equity or
    commodity contracts to be confused with.
    """
    kept, dropped = [], []

    for row in _binance_symbols(_get(BINANCE_FUTURES_INFO)):
        symbol = row.get("symbol")
        if row.get("status") != "TRADING":
            dropped.append((symbol, NOT_TRADING))
            continue
        contract_type = row.get("contractType")
        if contract_type not in DATED_CONTRACT_TYPES:
            # Allow-list, never a "not perpetual" test. See DATED_CONTRACT_TYPES:
            # TRADIFI_PERPETUAL carries a year-2100 delivery date and would pass any
            # test phrased as an exclusion.
            dropped.append((symbol, NOT_DATED))
            continue
        if row.get("underlyingType") not in CRYPTO_UNDERLYING:
            dropped.append((symbol, NOT_CRYPTO))
            continue
        kept.append(Instrument(
            venue="binance-futures", symbol=symbol, kind=contract_type,
            detail={"delivery_ns": int(row.get("deliveryDate") or 0) * 1_000_000,
                    "underlying_type": row.get("underlyingType"),
                    "quote_asset": row.get("quoteAsset")}))

    for category in ("linear", "inverse"):
        payload = _get(BYBIT_TICKERS.format(category=category))
        rows = (payload.get("result") or {}).get("list")
        if not isinstance(rows, list) or not rows:
            raise DiscoveryFailed(f"bybit {category} tickers carried no list")
        for row in rows:
            symbol = row.get("symbol")
            try:
                delivery = int(row.get("deliveryTime") or 0)
            except (TypeError, ValueError):
                delivery = 0
            if delivery <= 0:
                dropped.append((symbol, NOT_DATED))
                continue
            kept.append(Instrument(
                venue="bybit", symbol=symbol, kind=f"BYBIT_{category.upper()}",
                detail={"delivery_ns": delivery * 1_000_000, "category": category}))

    return Universe("dated", tuple(kept), tuple(dropped))


# Deribit lists options on a handful of currencies and zero on the rest. Measured
# 2026-08-18: BTC 786 instruments, ETH 656, SOL 0, XRP 0. The zeros are asked for
# anyway - a currency that starts listing should appear without anybody editing this.
DERIBIT_CURRENCIES = ("BTC", "ETH", "SOL", "XRP")


def discover_deribit_options(currencies=DERIBIT_CURRENCIES) -> Universe:
    """Every quoted option instrument across Deribit's listed currencies.

    A currency listing zero instruments is NOT a discovery failure - Deribit answers
    correctly with an empty chain for SOL and XRP - but a request that fails is, and
    the two are separated. `chains` in the detail records what each currency
    contributed, so a currency silently dropping to zero is visible rather than
    absorbed into a total.
    """
    kept, dropped, chains = [], [], {}
    for currency in currencies:
        rows = _get(DERIBIT_CHAIN.format(currency=currency)).get("result")
        if rows is None:
            raise DiscoveryFailed(f"deribit {currency}: no result field")
        chains[currency] = len(rows)
        for row in rows:
            instrument = row.get("instrument_name")
            if not isinstance(instrument, str) or not instrument:
                dropped.append((instrument, "NO_INSTRUMENT_NAME"))
                continue
            kept.append(Instrument(
                venue="deribit", symbol=instrument, kind="OPTION",
                detail={"currency": currency,
                        "underlying_price": row.get("underlying_price")}))
    universe = Universe("options", tuple(kept), tuple(dropped))
    object.__setattr__(universe, "chains", chains)
    return universe


DISCOVERERS = {
    "perp": discover_binance_perpetuals,
    "spot": discover_binance_spot,
    "dated": discover_dated_futures,
    "options": discover_deribit_options,
}


def discover(segment: str) -> Universe:
    if segment not in DISCOVERERS:
        raise ValueError(f"unknown segment {segment!r}")
    return DISCOVERERS[segment]()
