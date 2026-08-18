"""OB-01: which option instruments the bot may trade, and the reason for every one it may not.

## Admission is decided from the LIVE feed, not from the store

`perp.tradable_universe` decides admission from captured history — it was written
before RL-024 and answers a different question: which symbols have enough archive to
model. This module answers *which symbols are tradable right now*, and the only
evidence that can answer it is what the venue is quoting this second.

Both are legitimate and they are not merged. A symbol with deep history that stopped
quoting an hour ago passes the first and must fail this one.

## The thinness of this segment is published, never hidden

The Deribit chain endpoint ignores its `timestamp` parameter, so it cannot be
backfilled: every hour of capture from 2026-08-16 is the only history this segment
will ever have. `describe()` carries `history_note` so any board rendering this
universe states the constraint rather than showing a healthy-looking instrument count
that implies depth behind it.

`capture.venues.deribit` measured 818 BTC instruments with `high`, `low` and
`price_change` null on 588 of them. A mostly-null chain is the base case here, and an
admission list that treated a null as a zero would admit instruments that are not
quoted at all.

## Refusals, in a fixed precedence

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

NEVER_QUOTED = "NEVER_QUOTED"
NO_TWO_SIDED_QUOTE = "NO_TWO_SIDED_QUOTE"
QUOTE_STALE = "QUOTE_STALE"
TAPE_TOO_THIN = "TAPE_TOO_THIN"

# One REST poll covers the whole chain, so every instrument's quote is as fresh as
# the last poll. The bound is set to several poll intervals: a single slow response
# must not empty the universe.
MAX_QUOTE_AGE_NS = 180_000_000_000
# Options do not print continuously. Most instruments on the chain trade rarely and
# are still perfectly quotable, so a trade-count floor would exclude almost the whole
# board for a reason that is not about tradability.
MIN_TRADES = 0


@dataclass(frozen=True)
class Admission:
    """One instrument's decision, with what produced it."""

    venue: str
    symbol: str
    admitted: bool
    reason: str
    evidence: dict


def admit(frames: dict, *, max_quote_age_ns: int = MAX_QUOTE_AGE_NS,
          min_trades: int = MIN_TRADES) -> tuple[Admission, ...]:
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
        evidence = {"quote_age_ns": quote_age, "trade_count": trades,
                    "samples": frame.get("samples")}
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
        "segment": "options",
        "considered": considered,
        "admitted": len(admitted),
        "excluded": considered - len(admitted),
        "excluded_by_reason": reasons,
        "admitted_symbols": [d.symbol for d in admitted],
        "history_note": ("the chain endpoint ignores its timestamp parameter, so this "
                         "segment cannot be backfilled; captured history begins "
                         "2026-08-16 and is the only history it will ever have"),
    }
