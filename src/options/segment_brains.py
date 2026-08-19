"""OB-02: the options segment's BULL, BEAR and PROFIT-TAIL — and the rule that an option is not a punt.

## The constraint that shapes everything here

`~/research/dual-agent-spec.md`, carried forward into
`bull-bear-profit-agents-spec.md` §1: **options are not a directional bet.** OB-02's
acceptance requires a test that trips on violating it.

The violation being guarded against is specific and is the default way retail trades
options: form a view on the underlying, buy a call, call it a long. That position is
short volatility premium, short theta and long a strike-and-expiry choice nobody
reasoned about, and it loses money slowly while looking like a directional view that
was simply wrong.

So a proposal from these brains is **never justified by underlying momentum alone**.
Every one carries the instrument's expiry, its moneyness, and the volatility reading
that made it attractive — `named_expiry_and_moneyness` is asserted on every decision,
and `OptionsTakenAsDirectionalBet` is raised if a caller tries to build a proposal
without them.

## What these brains actually read

The quoted chain: `bid_price`, `ask_price`, `mark_iv`, `underlying_price`. Deribit's
`get_book_summary_by_currency` carries all of them for the whole chain in one request.

The signal is **relative implied volatility**, not direction:

* **BULL** proposes buying an option when its mark IV sits low against the chain's own
  median for that expiry — cheap volatility, defined risk, premium paid.
* **BEAR** proposes selling premium when mark IV sits high against that median.

The direction of the underlying enters as a filter on WHICH instrument, never as the
reason for the trade.

## Two measured facts this segment has to live with

**Most fields are null.** `capture.venues.deribit` measured `high`, `low` and
`price_change` null on 588 of 818 instruments. A null bid on an option means nobody is
bidding, and `live.live_feed._decimal_or_none` refuses to turn that into a zero — a
zero bid on a call is a free option and a fill model would happily oblige.

**The history is two days old and cannot be backfilled.** The chain endpoint ignores
its `timestamp` parameter, so every hour of capture from 2026-08-16 is the only
history this segment will ever have. Nothing here may be trained on depth that does
not exist, and the board tile publishes the thinness rather than hiding it.
"""
from __future__ import annotations

import statistics
from decimal import Decimal

from segment.brain import BEAR, BULL, decline, propose
from segment.profit_tail import ProfitTail

# An option whose quoted spread is wider than this cannot be scalped: the round trip
# costs twice the spread and option spreads are quoted in premium, where 5% of a
# small premium is a large fraction of any move.
MAX_RELATIVE_SPREAD = Decimal("0.05")
# How far from the money an instrument may be. Deep wings have almost no delta, quote
# terribly, and their premium is dominated by the tail nobody here is modelling.
MAX_MONEYNESS_DISTANCE = Decimal("0.15")
# How far IV must sit from the chain's median before it is worth acting on.
MIN_IV_EDGE = Decimal("0.08")
MIN_CHAIN_SAMPLE = 8


class OptionsTakenAsDirectionalBet(RuntimeError):
    """A proposal was built without the expiry, moneyness and volatility reading.

    The guard exists because the failure it catches is invisible: a call bought on a
    bullish view IS a valid-looking long position, and only the missing reasoning
    distinguishes it from a trade this segment is allowed to make.
    """


def _parse_instrument(symbol: str) -> dict:
    """Deribit names carry the contract: `BTC-26SEP25-70000-C`."""
    parts = (symbol or "").split("-")
    if len(parts) != 4:
        return {}
    underlying, expiry, strike, kind = parts
    try:
        strike_value = Decimal(strike)
    except (TypeError, ArithmeticError):
        return {}
    return {"underlying": underlying, "expiry_label": expiry,
            "strike": strike_value, "option_kind": kind.upper()}


def _moneyness(frame, contract: dict) -> Decimal | None:
    underlying = frame.get("venue_underlying_price")
    strike = contract.get("strike")
    if underlying is None or strike is None:
        return None
    try:
        underlying_d = Decimal(str(underlying))
    except (TypeError, ArithmeticError):
        return None
    if underlying_d <= 0:
        return None
    return (strike - underlying_d) / underlying_d


def _evidence(frame, contract: dict, moneyness, chain_median) -> dict:
    return {
        "expiry_label": contract.get("expiry_label"),
        "strike": None if contract.get("strike") is None else str(contract["strike"]),
        "option_kind": contract.get("option_kind"),
        "moneyness": None if moneyness is None else str(moneyness),
        "mark_iv": frame.get("venue_mark_iv"),
        "chain_median_iv": chain_median,
        "underlying_price": frame.get("venue_underlying_price"),
        "relative_spread": None if frame.get("relative_spread") is None else float(frame["relative_spread"]),
        "named_expiry_and_moneyness": True,
        "directional_bet": False,
        # **The window this frame was computed from (BF-02).** Its acceptance is
        # that every feature row NAMES the window behind it, and this brain was
        # the only evidence on the board that did not - so a reader could not
        # tell a decision made on two observations from one made on two hundred.
        "samples": frame.get("samples"),
        "window_ns": frame.get("window_ns"),
        "rule_brain": True,
    }


def chain_median_iv(frames) -> float | None:
    """The chain's own median mark IV. The reference every proposal is relative to.

    Computed across the instruments seen this poll rather than from a constant: a
    fixed IV threshold would mean the bot bought every option in a calm regime and
    sold every option in a volatile one, which is a bet on the level of volatility
    dressed up as a bet on its relative value.
    """
    values = []
    for frame in frames:
        if hasattr(frame, "is_refusal"):
            continue
        iv = frame.get("venue_mark_iv")
        if iv is None:
            continue
        try:
            values.append(float(iv))
        except (TypeError, ValueError):
            continue
    if len(values) < MIN_CHAIN_SAMPLE:
        return None
    return statistics.median(values)


def _blocked_reason(frame, contract, moneyness, chain_median) -> str | None:
    if not contract:
        return "INSTRUMENT_NAME_NOT_PARSED"
    if chain_median is None:
        return "CHAIN_TOO_THIN_FOR_A_REFERENCE"
    if frame.get("venue_mark_iv") is None:
        return "NO_MARK_IV"
    if moneyness is None:
        return "NO_UNDERLYING_PRICE"
    if abs(moneyness) > MAX_MONEYNESS_DISTANCE:
        return "TOO_FAR_FROM_THE_MONEY"
    spread = frame.get("relative_spread")
    if spread is None:
        return "NO_TWO_SIDED_QUOTE"
    if Decimal(str(spread)) > MAX_RELATIVE_SPREAD:
        return "SPREAD_WIDER_THAN_EDGE"
    return None


def _iv_edge(frame, chain_median) -> Decimal | None:
    iv = frame.get("venue_mark_iv")
    if iv is None or not chain_median:
        return None
    try:
        return (Decimal(str(iv)) - Decimal(str(chain_median))) / Decimal(str(chain_median))
    except (TypeError, ArithmeticError):
        return None


def _confidence(edge: Decimal) -> Decimal:
    excess = (abs(edge) - MIN_IV_EDGE) / MIN_IV_EDGE
    scaled = min(Decimal("1"), max(Decimal("0"), excess))
    return Decimal(str(round(float(Decimal("0.55") + Decimal("0.4") * scaled), 4)))


class OptionsBullBrain:
    """Buys volatility when the chain prices this instrument cheap against itself.

    LONG here means long the option — long premium, long vega, defined risk. It is
    not a bullish view on the underlying, and the evidence says so on every row.
    """

    name = "options-bull-rule"
    stance = BULL
    makes_edge_claim = False

    def __call__(self, frame, chain_median=None):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        contract = _parse_instrument(symbol)
        moneyness = _moneyness(frame, contract) if contract else None
        evidence = _evidence(frame, contract, moneyness, chain_median)

        blocked = _blocked_reason(frame, contract, moneyness, chain_median)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)

        edge = _iv_edge(frame, chain_median)
        if edge is None:
            return decline(self, venue=venue, symbol=symbol, reason="NO_IV_EDGE",
                           evidence=evidence, at_ns=at_ns)
        if edge > -MIN_IV_EDGE:
            return decline(self, venue=venue, symbol=symbol,
                           reason="VOLATILITY_NOT_CHEAP_AGAINST_CHAIN",
                           evidence={**evidence, "iv_edge": str(edge)}, at_ns=at_ns)

        return propose(self, venue=venue, symbol=symbol, confidence=_confidence(edge),
                       evidence={**evidence, "iv_edge": str(edge),
                                 "rule": "mark IV cheap against the chain median for a "
                                         "near-the-money instrument; long premium, "
                                         "not a directional view on the underlying"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


class OptionsBearBrain:
    """Sells volatility when the chain prices this instrument rich against itself.

    SHORT here means short the option — short premium, short vega, and an undefined
    loss tail that the risk gate, never this brain, is responsible for bounding.
    """

    name = "options-bear-rule"
    stance = BEAR
    makes_edge_claim = False

    def __call__(self, frame, chain_median=None):
        venue, symbol = frame.get("venue"), frame.get("symbol")
        at_ns = frame.get("at_ns") or 0
        contract = _parse_instrument(symbol)
        moneyness = _moneyness(frame, contract) if contract else None
        evidence = {**_evidence(frame, contract, moneyness, chain_median),
                    "undefined_loss_tail": True}

        blocked = _blocked_reason(frame, contract, moneyness, chain_median)
        if blocked:
            return decline(self, venue=venue, symbol=symbol, reason=blocked,
                           evidence=evidence, at_ns=at_ns)

        edge = _iv_edge(frame, chain_median)
        if edge is None:
            return decline(self, venue=venue, symbol=symbol, reason="NO_IV_EDGE",
                           evidence=evidence, at_ns=at_ns)
        if edge < MIN_IV_EDGE:
            return decline(self, venue=venue, symbol=symbol,
                           reason="VOLATILITY_NOT_RICH_AGAINST_CHAIN",
                           evidence={**evidence, "iv_edge": str(edge)}, at_ns=at_ns)

        return propose(self, venue=venue, symbol=symbol, confidence=_confidence(edge),
                       evidence={**evidence, "iv_edge": str(edge),
                                 "rule": "mark IV rich against the chain median for a "
                                         "near-the-money instrument; short premium, "
                                         "not a directional view on the underlying"},
                       at_ns=at_ns, calibrated=False, makes_edge_claim=False)


def options_profit_tail(band: str = "slow") -> ProfitTail:
    """Options move in premium terms, so the take-profit is a premium fraction.

    Wider than the linear segments because an option's premium is a small number and
    a 40 bp move in the underlying can be a 5% move in the premium. The numbers below
    are fractions OF THE PREMIUM, not of the underlying, and confusing the two is the
    fastest way to build an exit rule that never fires.
    """
    return ProfitTail(
        segment="options", name=f"options-profit-tail-{band}",
        take_profit=Decimal("0.1200"), ratchet_trigger=Decimal("0.0700"),
        ratchet_give_back=Decimal("0.0300"),
        signal_expiry_ns=180_000_000_000,
        max_hold_ns=7_200_000_000_000)            # 2 hours
