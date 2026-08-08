"""Which symbols are priced in dollars, decided from the venue's own field.

Settled 2026-08-08 as open question 4 of the paper-execution design: of the 1,377
tradeable Binance spot pairs the capture subscribes, only 838 are quoted in
dollars. The rest are quoted in TRY (312 pairs), USDC's non-dollar cousins, EUR,
JPY, IDR, BRL, MXN, ARS, ZAR, UAH, COP, KZT, AED, or in BTC, ETH, BNB, XRP and
SOL. A TRY pair's returns carry Turkish lira moves and a BTC-quoted pair's carry
bitcoin's; feeding either into a dollar-denominated P&L measures something nobody
asked about. The ruling was to filter to dollar quotes rather than convert -
conversion needs an FX rate the archive does not capture, and a wrong rate
corrupts a P&L silently, where an excluded pair is merely absent.

**Never parse the quote out of the symbol string.** That is not a style
preference; measured against the live endpoints on 2026-08-08 it is simply wrong:

  * `BTCU` is BTC quoted in `U`, and 45 other spot pairs share that quote. Two
    perpetuals are quoted in `U` as well.
  * `XRPRLUSD` is XRP quoted in `RLUSD`. A longest-suffix rule holding `USD`
    reads it as `XRPRL` quoted in dollars - a pair that does not exist, admitted
    to a dollar universe.
  * `EUREURI` is EUR quoted in `EURI`, so the base and the quote are both
    suffixes of each other's name.

Add `U` to a suffix table to fix the first and every symbol ending in U becomes
ambiguous. The venue publishes `quoteAsset` per symbol; `capture.venues` records
it point-in-time into the universe snapshot, and this module reads that.

The filter is applied where symbols are SELECTED, never where bars are BUILT. The
bars build takes every captured symbol: bars are cheap - measured 2026-08-08, the
whole 1,363-symbol spot day costs 100s - and the raw they come from is evicted
after seven days, so a symbol filtered out at build time is a symbol that can
never be built. A pair excluded here can be admitted tomorrow by changing one
frozenset; a day not built is gone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from capture.universe_tracker import UniverseTracker

# Quote currencies that ARE the dollar for P&L purposes: the dollar itself and
# the stablecoins that track it. Membership is a judgement about denomination,
# not about credit risk - USDT depegging is a risk this system carries, and it is
# priced by the strategy layer, not hidden by excluding the pair from the
# universe.
#
# Every entry was observed as a live `quoteAsset` on Binance spot or futures on
# 2026-08-08, except USD, which is hyperliquid's uniform perp denomination.
DOLLAR_QUOTE_ASSETS = frozenset({
    "USD", "USDT", "USDC", "FDUSD", "USD1", "USDS", "RLUSD", "TUSD", "USDP",
    "PYUSD", "BUSD", "DAI",
})

# Quote currencies known NOT to be dollars, listed rather than inferred so that a
# quote asset nobody has classified stays visible as `unknown` instead of being
# absorbed into this set by default. Observed live 2026-08-08.
#
# `U` is here on purpose and is the reason this module refuses string parsing:
# it is a real Binance quote asset, 46 spot pairs and 2 perpetuals.
NON_DOLLAR_QUOTE_ASSETS = frozenset({
    "TRY", "EUR", "EURI", "JPY", "BRL", "IDR", "MXN", "ARS", "ZAR", "UAH",
    "COP", "KZT", "AED", "PLN", "RON", "CZK", "GBP", "AUD", "NGN", "RUB",
    "BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "U",
})


class QuoteAssetsNotRecorded(LookupError):
    """No point-in-time quote map exists for this venue at this time.

    Refused rather than defaulted, in either direction, because both defaults are
    wrong and neither is visible. Treating an empty map as "nothing is
    dollar-quoted" empties the universe and looks like a market with no
    candidates; treating it as "everything is" admits 312 lira-quoted pairs into
    a dollar P&L. Both read as a working system.

    Reachable for any snapshot recorded before 2026-08-08, when quote assets were
    not captured at all. The fix is to record a snapshot, not to guess.
    """


@dataclass(frozen=True)
class QuotePartition:
    """Symbols split three ways by what they are priced in.

    Three groups, not two, because "not known to be a dollar" and "known not to
    be a dollar" are different facts and only one of them is a finding. A new
    Binance stablecoin lands in `unknown`, where a caller reports it and someone
    classifies it; folded into `non_dollar` it would silently shrink the tradeable
    universe on the day it listed, and the only symptom would be a number nobody
    was watching.
    """
    dollar: tuple[str, ...]
    non_dollar: Mapping[str, str]
    unknown: Mapping[str, str]

    def describe(self) -> str:
        """One line naming all three counts, for a run log or a board tile.

        The unknown count is printed even when it is zero. A reader must not have
        to infer it from silence - see Rule 8: absence of evidence renders as its
        own state, never as health.
        """
        return (f"{len(self.dollar)} dollar-quoted, {len(self.non_dollar)} "
                f"non-dollar, {len(self.unknown)} unclassified quote asset(s)")


def partition_by_quote(quote_assets: Mapping[str, str]) -> QuotePartition:
    """Split a symbol-to-quote map into dollar, non-dollar and unclassified.

    Takes the map rather than reading it, so the classification is testable
    without an archive and usable against a map from any source.
    """
    if not quote_assets:
        raise QuoteAssetsNotRecorded(
            "an empty quote map cannot be filtered: read as 'no dollar pairs' it "
            "empties the universe, read as 'all dollar pairs' it admits every "
            "lira- and bitcoin-quoted pair into a dollar P&L, and both look like "
            "a working system. Record a universe snapshot carrying quote assets.")

    dollar, non_dollar, unknown = [], {}, {}
    for symbol, quote in sorted(quote_assets.items()):
        if quote in DOLLAR_QUOTE_ASSETS:
            dollar.append(symbol)
        elif quote in NON_DOLLAR_QUOTE_ASSETS:
            non_dollar[symbol] = quote
        else:
            unknown[symbol] = quote
    return QuotePartition(tuple(dollar), non_dollar, unknown)


def dollar_quoted_symbols(capture_root: Path, venue: str, as_of_ns: int) -> QuotePartition:
    """Partition the venue's universe as it was known at `as_of_ns`.

    Point-in-time through `UniverseTracker.load_last_quote_assets`, so a snapshot
    recorded after `as_of_ns` is not used: knowing which pairs existed later is
    the same lookahead the universe record exists to prevent, and here it would
    also mean classifying a symbol by a listing that had not happened.
    """
    quote_assets = UniverseTracker(Path(capture_root), venue).load_last_quote_assets(as_of_ns)
    if not quote_assets:
        raise QuoteAssetsNotRecorded(
            f"no universe snapshot with quote assets for {venue} at or before "
            f"{as_of_ns}. Every snapshot recorded before 2026-08-08 predates quote "
            f"capture; record one with `python -m capture.record_universe_snapshot "
            f"--venue {venue}` rather than assuming a denomination.")
    return partition_by_quote(quote_assets)
