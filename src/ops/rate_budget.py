"""One rate budget per venue, shared by every caller in every process.

`ARCHITECTURE.md` Layer 3: *"Rate limits are an architectural constraint, not a
config value: per-IP, exchange-wide. Binance `418` bans scale **2 minutes → 3
days** for repeat offenders. One strategy's burst bans all of them."*

Three design consequences, each load-bearing.

**Weight, not request count.** Measured 2026-08-08 from the `x-mbx-used-weight`
header: a `limit=1000` spot depth snapshot costs **50**; `premiumIndex` costs
**1**. A budgeter counting requests would let fifty snapshots through as cheaply
as fifty funding polls and earn exactly the ban it exists to prevent.

**File-backed, not in-memory.** Capture runs one process per venue and the limit
is per-IP across all of them, so a bucket living in one process's memory protects
nothing. State is a file, read and written under an exclusive lock, so a spend by
one process is visible to the next.

**The venue's word beats ours.** A `Retry-After` header, or a 418, is the venue
telling us our model of its limit is wrong. A full local bucket during a ban is
not permission to continue - it is the model being wrong - so a recorded ban
blocks every spend regardless of what the bucket says.

Refusal, never a hidden sleep. `try_spend` returns False and the caller decides
whether to wait; a budgeter that quietly blocks is how a capture loop stalls with
nothing reporting why.
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Callable

# Measured against the live endpoints on 2026-08-08 rather than recalled, and
# deliberately conservative: these are the venue's published per-minute ceilings
# with headroom, because the cost of being under is a slower poll and the cost of
# being over escalates to a three-day ban.
#
# The reported figures are 2400/min on fapi and 6000/min on spot. Two thirds of
# each leaves room for the request weight this process cannot see - anything
# else sharing the egress IP.
KNOWN_CAPACITY = {
    "binance": 1600,            # fapi, per minute
    "binance-spot": 4000,       # api/v3, per minute
    "hyperliquid": 1000,        # consensus-bound and shaped differently; a guess
}


class RateBudget:
    """A weight-based token bucket for one venue, shared through a state file."""

    def __init__(self, root: Path, venue: str, capacity: int | None = None,
                 refill_per_second: float | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._root = Path(root)
        self._venue = venue
        self._clock = clock
        self._capacity = float(
            capacity if capacity is not None else KNOWN_CAPACITY.get(venue, 600))
        # Default refill spreads the per-minute ceiling evenly. A bucket that
        # refills in one lump at the top of each minute permits a burst the
        # venue's own sliding window would still count as an overage.
        self._refill = float(
            refill_per_second if refill_per_second is not None
            else self._capacity / 60.0)
        self._path = self._root / f"rate_budget_{venue}.json"

    # --- state, always under a lock ------------------------------------------

    def _with_state(self, mutate):
        """Read, mutate and write the state file while holding an exclusive lock.

        The lock is the whole reason this is not a plain read-modify-write: two
        capture processes polling at the same instant would each read the same
        balance, both decide they could afford it, and together spend twice the
        budget.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        with self._path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read()
                try:
                    state = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    state = {}
                state.setdefault("tokens", self._capacity)
                state.setdefault("updated_at", self._clock())
                state.setdefault("blocked_until", 0.0)

                result = mutate(state)

                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(state))
                handle.flush()
                return result
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _refill_into(self, state: dict) -> None:
        now = self._clock()
        elapsed = max(0.0, now - float(state["updated_at"]))
        # Capped at capacity: a bucket that overfills while idle hands out a
        # burst on the next tick, which is the shape that earns a ban.
        state["tokens"] = min(self._capacity,
                              float(state["tokens"]) + elapsed * self._refill)
        state["updated_at"] = now

    # --- the two calls that matter -------------------------------------------

    def try_spend(self, weight: int) -> bool:
        """Take `weight` from the budget, or return False having taken nothing."""
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}; a free "
                             f"request is a bug in the caller, not a discount")

        def mutate(state):
            # The ban is checked before the bucket, deliberately. During a
            # Retry-After window our balance is irrelevant - the venue has
            # already answered.
            if self._clock() < float(state["blocked_until"]):
                return False
            self._refill_into(state)
            if float(state["tokens"]) < weight:
                return False
            state["tokens"] = float(state["tokens"]) - weight
            return True

        return self._with_state(mutate)

    def note_rate_limited(self, retry_after_seconds: float) -> None:
        """Record that the venue pushed back, and for how long.

        Honoured over our own schedule, per `ARCHITECTURE.md`. Also empties the
        bucket: if the venue is rate-limiting us, our model of its limit was
        wrong, and resuming at a full balance the moment the window closes would
        repeat the mistake straight into the next ban tier.
        """
        def mutate(state):
            state["blocked_until"] = self._clock() + max(0.0, float(retry_after_seconds))
            state["tokens"] = 0.0
            state["updated_at"] = self._clock()
            return None

        self._with_state(mutate)

    def remaining(self) -> float:
        """Weight available right now, after refill. Zero during a ban."""
        def mutate(state):
            if self._clock() < float(state["blocked_until"]):
                return 0.0
            self._refill_into(state)
            return float(state["tokens"])

        return self._with_state(mutate)

    def blocked_for(self) -> float:
        """Seconds until a ban lifts, or 0.0 when not banned."""
        def mutate(state):
            return max(0.0, float(state["blocked_until"]) - self._clock())

        return self._with_state(mutate)
