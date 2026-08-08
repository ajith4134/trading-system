"""Write the intent, then send. Never the other way round, never blind-retry.

`DECISIONS.md` §6 / `FEATURES.md` §9: *"Write-ahead order-intent log with startup
reconciliation."* `FEATURES.md` §5, EX-001: *"Client order ID derived
deterministically; query by original ID on timeout, **never blind-retry**."*

The ledger's citation for the cost of getting this wrong: **Everbright Securities
lost roughly $3.8 billion** doing the opposite. Blind-retrying an order whose
response was lost is not error recovery; it is a second order.

Two mechanisms, load-bearing only in combination.

**Write before send, fsynced.** If the process dies between the write and the send,
recovery finds an intent of unknown fate and can ask the venue about it. If the log
were written *after* sending - the natural way to write it - a death in between
leaves an order live at the venue that nothing local knows exists. The position is
then invisible to every risk check that matters.

**Deterministic client order id.** The same intent hashes to the same id, always.
That id is what makes "query by the original id" possible; a random or wall-clock id
turns every uncertain send into a guaranteed duplicate on retry. Creation time is
part of the hash, so two genuinely separate decisions to buy the same size stay
distinct rather than the second being swallowed as a duplicate of the first.

An intent that has been submitted once cannot be submitted again. The way out of an
unknown outcome is `resolve()` after asking the venue - not another send.

`EX-003` lives here too: an intent past its validity window is refused, because *"a
signal computed 5 minutes ago must not fire now."* Refused *before* the log write, so
no record implies it might have gone out - but the expiry itself is recorded, since
a system that is consistently too slow to act on its own signals should be visible
rather than quiet.

**No venue contact anywhere in this module.** `submit` takes a transport callable.
There are no credentials, no network, no order placement here - this is durability
and recovery. Wiring a real transport is a separate decision, and under `CLAUDE.md`
Rule 0 an explicitly human one.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable

WAL_FILE = "order_intents.ndjson"

# Binance caps newClientOrderId at 36 characters. An id the venue truncates is an id
# that cannot be queried by, which removes the only alternative to resending.
_MAX_ID_LENGTH = 36
_ID_PREFIX = "ts"

OUTCOME_UNKNOWN = "unknown"


class WalCorrupt(RuntimeError):
    """The write-ahead log cannot be read, so outstanding orders are unknown.

    Fatal rather than empty. An empty history reads as "nothing outstanding", which
    is the most dangerous available misreading of an unreadable WAL.
    """


class IntentExpired(RuntimeError):
    """The intent's validity window closed before it was submitted."""


@dataclass(frozen=True)
class OrderIntent:
    """A decision to trade, before anything has been sent.

    `valid_for_ns` is mandatory and positive: an intent with no expiry is one that
    can fire arbitrarily late, which is EX-003's failure exactly.
    """

    strategy: str
    symbol: str
    venue: str
    side: str
    quantity: Decimal
    created_at_ns: int
    valid_for_ns: int
    price: Decimal | None = None
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if self.valid_for_ns <= 0:
            raise ValueError(
                f"valid_for_ns must be > 0, got {self.valid_for_ns}; an intent with "
                f"no expiry can fire arbitrarily late")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")

    def is_expired_at(self, now_ns: int) -> bool:
        return now_ns > self.created_at_ns + self.valid_for_ns

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "symbol": self.symbol, "venue": self.venue,
            "side": self.side, "quantity": str(self.quantity),
            "created_at_ns": self.created_at_ns, "valid_for_ns": self.valid_for_ns,
            "price": str(self.price) if self.price is not None else None,
            "reduce_only": self.reduce_only,
        }


def derive_client_order_id(intent: OrderIntent) -> str:
    """A deterministic, venue-safe id for this exact intent.

    Hashed over every field that distinguishes one decision from another, including
    `created_at_ns`. Two separate decisions to buy the same size at different moments
    must not collide, or the second is silently discarded as a duplicate.
    """
    material = "|".join((
        intent.strategy, intent.symbol, intent.venue, intent.side.upper(),
        str(intent.quantity), str(intent.created_at_ns),
        str(intent.valid_for_ns), str(intent.price), str(intent.reduce_only),
    ))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{_ID_PREFIX}{digest}"[:_MAX_ID_LENGTH]


class OrderIntentWal:
    """Append-only intent log. Durable before the send, queryable after a crash."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._path = self._root / WAL_FILE

    # --- reading --------------------------------------------------------------

    @property
    def has_torn_tail(self) -> bool:
        """Whether the last WAL line is a partial write from a killed process.

        A property that re-reads rather than a flag set as a side effect of some
        earlier call: an attribute whose truth depends on whether something else has
        run yet is a flag that will be read before it is correct, and here that reads
        as "no torn tail" - the flattering answer.
        """
        return self._read()[1]

    def _read(self) -> tuple[list[dict], bool]:
        """(events, a_torn_final_line_was_seen)."""
        if not self._path.exists():
            return [], False
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise WalCorrupt(f"cannot read {self._path}: {exc}") from exc

        lines = [ln for ln in text.split("\n") if ln.strip()]
        events: list[dict] = []
        torn = False
        for index, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                if index == len(lines) - 1 and not text.endswith("\n"):
                    # Killed mid-append. The intent it was recording may already
                    # have reached the venue, so it must surface as unresolved.
                    torn = True
                    continue
                raise WalCorrupt(
                    f"{self._path} line {index + 1} is not valid JSON ({exc}); "
                    f"outstanding orders cannot be determined, and reporting an "
                    f"empty history would read as 'nothing outstanding'") from exc
        return events, torn

    def _folded(self) -> dict[str, dict]:
        """Intents keyed by client order id, with later events applied over earlier."""
        folded: dict[str, dict] = {}
        for event in self._read()[0]:
            if event.get("event") == "expired":
                continue
            identifier = event.get("client_order_id")
            if identifier is None:
                continue
            if identifier in folded:
                folded[identifier].update(
                    {k: v for k, v in event.items() if k != "client_order_id"})
            else:
                folded[identifier] = dict(event)
        return folded

    def records(self) -> list[dict]:
        """Every submitted intent, folded to its latest known state."""
        return list(self._folded().values())

    def expiries(self) -> list[dict]:
        return [e for e in self._read()[0] if e.get("event") == "expired"]

    def unresolved(self) -> list[dict]:
        """Intents whose fate at the venue is unknown.

        These are the startup work list: for each, query the venue by
        `client_order_id` and call `resolve`. Never resend.
        """
        events, torn = self._read()
        out = [r for r in self._folded().values()
               if r.get("outcome") == OUTCOME_UNKNOWN]
        if torn:
            # A torn tail is an intent whose record never completed. Its id is not
            # recoverable from the partial line, so it is surfaced as a synthetic
            # unresolved entry rather than being dropped.
            out.append({"client_order_id": None, "outcome": OUTCOME_UNKNOWN,
                        "error": "torn final WAL line - the intent it recorded may "
                                 "have reached the venue; reconcile against venue "
                                 "open orders before trading",
                        "event": "torn"})
        return out

    # --- writing --------------------------------------------------------------

    def _append(self, event: dict) -> None:
        """Append one event and fsync it.

        The fsync is the whole point of a write-ahead log. Without it the record
        lives in the page cache and a power loss takes it, which puts us back to
        having sent an order nothing knows about.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def submit(self, intent: OrderIntent,
               transport: Callable[[OrderIntent, str], dict],
               now_ns: int | None = None) -> dict:
        """Record the intent durably, then hand it to `transport`.

        Raises `IntentExpired` if the validity window has closed, `RuntimeError` if
        this exact intent was already submitted, and re-raises whatever the transport
        raised - having first recorded the outcome as unknown, which is the state
        that prompts a query rather than a resend.
        """
        now_ns = time.time_ns() if now_ns is None else now_ns
        identifier = derive_client_order_id(intent)

        if intent.is_expired_at(now_ns):
            # Recorded, but as an expiry rather than a submission: nothing was sent,
            # and no record should imply it might have been.
            self._append({"event": "expired", "client_order_id": identifier,
                          "at_ns": now_ns, **intent.to_dict()})
            raise IntentExpired(
                f"intent {identifier} was created at {intent.created_at_ns} with a "
                f"{intent.valid_for_ns}ns window and it is now {now_ns} - a signal "
                f"computed that long ago must not fire")

        if identifier in self._folded():
            raise RuntimeError(
                f"intent {identifier} was already submitted. Blind-retrying an order "
                f"whose response was lost is a second order, not a recovery - query "
                f"the venue by this id and call resolve() instead")

        self._append({"event": "submitted", "client_order_id": identifier,
                      "at_ns": now_ns, "outcome": OUTCOME_UNKNOWN,
                      "error": None, **intent.to_dict()})

        try:
            response = transport(intent, identifier)
        except BaseException as exc:
            # A timeout is not a rejection. The venue may well have accepted it, so
            # the outcome stays unknown - the only state that triggers a query.
            self._append({"client_order_id": identifier,
                          "event": "transport_failed", "at_ns": now_ns,
                          "outcome": OUTCOME_UNKNOWN,
                          "error": f"{type(exc).__name__}: {exc}"})
            raise

        self._append({"client_order_id": identifier, "event": "acknowledged",
                      "at_ns": now_ns, "outcome": "acknowledged",
                      "response": response, "error": None})
        return response

    def resolve(self, client_order_id: str, outcome: str, detail: str,
                now_ns: int | None = None) -> None:
        """Close out an unknown intent with what the venue actually said."""
        if client_order_id not in self._folded():
            raise KeyError(
                f"no intent {client_order_id!r} in the WAL; resolving an id that was "
                f"never submitted would invent a history")
        if outcome == OUTCOME_UNKNOWN:
            raise ValueError(
                "resolving to 'unknown' is not a resolution; query the venue until "
                "the fate is known, or halt")
        self._append({"client_order_id": client_order_id, "event": "resolved",
                      "at_ns": time.time_ns() if now_ns is None else now_ns,
                      "outcome": outcome, "detail": detail})
