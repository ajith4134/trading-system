"""Rebuild on startup, reconcile against venue truth, refuse to start on mismatch.

`FEATURES.md` §6 / `DECISIONS.md` §6, RX-005: *"Rebuild local state on every startup;
**refuse to start if reconciliation fails**"*, with NautilusTrader's invariant
tolerances - quantity to instrument precision, average price within 0.01%.

**Why refuse rather than adopt.** The prior art (`live_trader.py`, ledger RX-005)
reconciles positions at boot and then *adopts or closes* whatever mismatched. That is
defensible right up to the point where the mismatch is a symptom rather than an
artefact - a fill handler dropping updates, a WAL never replayed, a second process
trading the same account. Adopting makes the symptom disappear while the cause keeps
running. Refusing to start costs a human a few minutes; trading on a position model
already known to be wrong costs whatever the position does next.

The same ledger row records that the donor's *periodic* reconcile loop **only re-syncs
capital, not positions, despite its name** - flagged in its own source as "narrower
than its name implies". A reconciliation that does not compare positions is the exact
shape of a control that reads as present and is not.

Three sources must agree:

  1. the **WAL** - what we intended, including intents whose fate is unknown
  2. the **local position model** - what we believe we hold
  3. **venue truth** - what the exchange reports

Any disagreement outside tolerance blocks the start. Unresolved intents block it
regardless of whether the positions happen to match right now, because an order that
may be live is a position that may be about to exist.

There is deliberately **no `adopt_mismatches` parameter.** Its absence is the
requirement, not an omission.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from execution.order_intent_wal import OrderIntentWal

# NautilusTrader's average-price invariant: 0.01%, i.e. one basis point. A wrong
# entry price is a wrong unrealised P&L, and unrealised P&L is what the drawdown
# circuit breakers read.
_PRICE_TOLERANCE = Decimal("0.0001")


class ReconciliationFailed(RuntimeError):
    """Local state and venue truth disagree, so trading must not resume."""


@dataclass(frozen=True)
class VenuePosition:
    """A position as one source reports it. Used for both local and venue sides."""

    symbol: str
    venue: str
    quantity: Decimal
    average_price: Decimal

    @property
    def key(self) -> tuple[str, str]:
        return (self.venue, self.symbol)


@dataclass(frozen=True)
class ReconciliationReport:
    """Whether to start, and every reason not to.

    `mismatches` holds all of them rather than the first: a gate that stops at the
    first problem turns one restart into five, and the operator needs the whole list
    to fix it once.
    """

    may_start: bool
    mismatches: tuple[str, ...]
    n_positions_checked: int
    n_unresolved_intents: int
    detail: str


def recover_and_reconcile(
    wal: OrderIntentWal,
    local_positions: list[VenuePosition],
    venue_positions: list[VenuePosition],
    quantity_steps: dict[str, Decimal],
    raise_on_failure: bool = False,
) -> ReconciliationReport:
    """Compare the three sources and decide whether trading may resume.

    `quantity_steps` maps symbol to the venue's quantity step. A symbol with no
    declared step is itself a mismatch: without a step there is no defined tolerance,
    and choosing one here would silently decide how wrong the position model is
    allowed to be.

    Raises `WalCorrupt` (not a mismatch) when the WAL cannot be read - the set of
    orders that might be live is then unknown, and there is nothing to reconcile
    against.
    """
    mismatches: list[str] = []

    # Reading the WAL first, and letting WalCorrupt propagate. A corrupt WAL is not
    # one finding among several; it means the question cannot be asked.
    unresolved = wal.unresolved()
    for entry in unresolved:
        mismatches.append(
            f"unresolved order intent {entry.get('client_order_id')}: "
            f"{entry.get('error') or 'fate at the venue unknown'} - an order that "
            f"may be live is a position that may be about to exist, so matching "
            f"positions now proves nothing")

    local_by_key = {p.key: p for p in local_positions}
    venue_by_key = {p.key: p for p in venue_positions}

    for key in sorted(set(local_by_key) | set(venue_by_key)):
        venue_name, symbol = key
        ours = local_by_key.get(key)
        theirs = venue_by_key.get(key)

        if ours is None:
            mismatches.append(
                f"{symbol} on {venue_name}: the venue reports {theirs.quantity} and "
                f"we hold no position - an unmanaged live position that nothing will "
                f"size, stop or close, because nothing knows it exists")
            continue
        if theirs is None:
            mismatches.append(
                f"{symbol} on {venue_name}: we believe we hold {ours.quantity} and "
                f"the venue reports nothing - every exit order against it would be "
                f"rejected, and risk counts capital that is not at risk")
            continue

        step = quantity_steps.get(symbol)
        if step is None:
            mismatches.append(
                f"{symbol} on {venue_name}: no instrument quantity step declared, so "
                f"there is no defined precision to reconcile to. Refused rather than "
                f"guessed - a guessed step decides how wrong the position model may "
                f"be")
            continue

        if abs(ours.quantity - theirs.quantity) >= step:
            mismatches.append(
                f"{symbol} on {venue_name}: quantity mismatch - we hold "
                f"{ours.quantity}, the venue reports {theirs.quantity}, step {step}")

        if theirs.average_price != 0:
            relative = (abs(ours.average_price - theirs.average_price)
                        / abs(theirs.average_price))
            if relative > _PRICE_TOLERANCE:
                mismatches.append(
                    f"{symbol} on {venue_name}: average price mismatch - ours "
                    f"{ours.average_price}, venue {theirs.average_price}, "
                    f"{relative * 100:.4f}% apart against a "
                    f"{_PRICE_TOLERANCE * 100}% tolerance")

    n_checked = len(set(local_by_key) | set(venue_by_key))
    report = ReconciliationReport(
        may_start=not mismatches,
        mismatches=tuple(mismatches),
        n_positions_checked=n_checked,
        n_unresolved_intents=len(unresolved),
        detail=(f"reconciled {n_checked} position(s) across local state and venue "
                f"truth with {len(unresolved)} unresolved intent(s); "
                f"{len(mismatches)} mismatch(es)"),
    )

    if raise_on_failure and not report.may_start:
        raise ReconciliationFailed(
            "refusing to start - local state and venue truth disagree:\n  "
            + "\n  ".join(report.mismatches))
    return report
