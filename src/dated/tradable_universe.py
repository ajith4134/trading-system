"""DB-01: which dated futures contracts the bot may trade, and the reason for every one it may not.

## Admission is decided from the LIVE feed, not from the store

`perp.tradable_universe` decides admission from captured history — it was written
before RL-024 and answers a different question: which symbols have enough archive to
model. This module answers *which symbols are tradable right now*, and the only
evidence that can answer it is what the venue is quoting this second.

Both are legitimate and they are not merged. A symbol with deep history that stopped
quoting an hour ago passes the first and must fail this one.

## Five refusals — the four the linear segments share, plus expiry

A dated contract has a death date, so admission carries one refusal no other segment
has: `INSIDE_FINAL_SETTLEMENT_WINDOW`. Near delivery the basis has already converged,
liquidity is at its worst and settlement mechanics dominate price. DB-02's brains
repeat the check, deliberately - a brain that depends on its caller having filtered
correctly has an unstated precondition.

`NOT_DATED` is the other addition. Bybit's `v5/market/tickers?category=linear` returns
perpetuals and dated futures in one response - 765 and 40 when `capture.venues.bybit`
measured it - and `deliveryTime` is the only thing separating them. A perpetual
admitted here would be the perp bot's instrument traded by the dated bot's brains,
which is the RL-019 failure in its purest form.

## Four refusals shared with the other segments

* `NEVER_QUOTED` — the feed has carried no tick for this symbol at all.
* `NO_TWO_SIDED_QUOTE` — prints, but not both sides. A one-sided market cannot be
  entered and exited, and a fill model given one side will invent the other.
* `QUOTE_STALE` — both sides seen, but not recently. The venue may have halted the
  symbol while the connection stayed healthy, which looks identical to a quiet market
  from anywhere except the timestamp.
* `TAPE_TOO_THIN` — quoting, but nothing trades. A quote nobody hits is not a market;
  modelling fills against it is the flattering direction, and therefore the one that
  goes unnoticed.

## Counts, never bare percentages

`describe()` reports the denominator with every number, following
`perp.tradable_universe`: "92% tradable" reads identically over 12 symbols and over
1,800 and is a different statement about each.
"""
from __future__ import annotations

from dataclasses import dataclass

NOT_DATED = "NOT_DATED"
INSIDE_FINAL_SETTLEMENT_WINDOW = "INSIDE_FINAL_SETTLEMENT_WINDOW"
NEVER_QUOTED = "NEVER_QUOTED"
NO_TWO_SIDED_QUOTE = "NO_TWO_SIDED_QUOTE"
QUOTE_STALE = "QUOTE_STALE"
TAPE_TOO_THIN = "TAPE_TOO_THIN"

# The dated feed is a REST poll, so its freshness bound is its interval and not a
# websocket's. Set well above the poll interval so a single slow response does not
# empty the universe.
MAX_QUOTE_AGE_NS = 120_000_000_000
# A dated contract can go minutes without a print and still be perfectly tradable;
# the tape-thinness test the linear segments apply would exclude most of the board.
MIN_TRADES = 0
FINAL_SETTLEMENT_WINDOW_NS = 2 * 24 * 3_600_000_000_000


@dataclass(frozen=True)
class Admission:
    """One instrument's decision, with what produced it."""

    venue: str
    symbol: str
    admitted: bool
    reason: str
    evidence: dict


def admit(frames: dict, *, max_quote_age_ns: int = MAX_QUOTE_AGE_NS,
          min_trades: int = MIN_TRADES,
          final_window_ns: int = FINAL_SETTLEMENT_WINDOW_NS) -> tuple[Admission, ...]:
    """Decide every symbol the feed has seen. Excluded rows are RETAINED.

    An excluded symbol that vanished from the output would make the excluded count
    unverifiable, and the count is the acceptance.
    """
    decisions = []
    for (venue, symbol), frame in frames.items():
        if hasattr(frame, "is_refusal"):
            missing = getattr(frame, "missing", ())
            reason = (NO_TWO_SIDED_QUOTE if "two_sided_quote" in missing
                      else NEVER_QUOTED if "no_ticks_seen" in missing
                      else TAPE_TOO_THIN)
            decisions.append(Admission(venue, symbol, False, reason,
                                       {"missing": list(missing)}))
            continue
        quote_age = frame.get("quote_age_ns")
        trades = frame.get("trade_count") or 0
        delivery_ns = frame.get("venue_delivery_ns") or 0
        remaining = int(delivery_ns) - int(frame.get("at_ns") or 0) if delivery_ns else 0
        evidence = {"quote_age_ns": quote_age, "trade_count": trades,
                    "samples": frame.get("samples"),
                    "delivery_ns": delivery_ns or None,
                    "time_to_expiry_ns": remaining or None}
        if not delivery_ns:
            decisions.append(Admission(venue, symbol, False, NOT_DATED, evidence))
            continue
        if remaining <= final_window_ns:
            decisions.append(Admission(venue, symbol, False,
                                       INSIDE_FINAL_SETTLEMENT_WINDOW, evidence))
            continue
        if quote_age is None or quote_age > max_quote_age_ns:
            decisions.append(Admission(venue, symbol, False, QUOTE_STALE, evidence))
            continue
        if trades < min_trades:
            decisions.append(Admission(venue, symbol, False, TAPE_TOO_THIN, evidence))
            continue
        decisions.append(Admission(venue, symbol, True, "ADMITTED", evidence))
    return tuple(decisions)


def describe(decisions) -> dict:
    """Counts with their denominator, and every exclusion reason named."""
    considered = len(decisions)
    admitted = [d for d in decisions if d.admitted]
    reasons: dict[str, int] = {}
    for decision in decisions:
        if not decision.admitted:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    return {
        "segment": "dated",
        "considered": considered,
        "admitted": len(admitted),
        "excluded": considered - len(admitted),
        "excluded_by_reason": reasons,
        "admitted_symbols": [d.symbol for d in admitted],
        # Published because the perp/dated split is the thing most likely to go
        # wrong silently, and a count of what was rejected as a perpetual is the
        # cheapest way to see it has not.
        "rejected_as_perpetual": reasons.get(NOT_DATED, 0),
    }
