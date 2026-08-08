"""Drives one crude signal through every layer, to prove the wiring carries a fill.

**This is not a strategy and its result is not evidence of edge.** The signal here
was chosen because it is trivial to state and produces trades on the tape that
exists - not because anything suggests it works. No purge, no embargo, no holdout,
no Deflated Sharpe, no promotion gate. A P&L number printed by this module means
"the pipes connect", and reading it as anything else is the exact mistake
`~/research/ledger/merged/risk-execution-validation.md` records three prior systems
making.

What it actually proves, end to end and on real captured data:

  ClockGatedReader   - every bar is read at a simulated clock that advances one bar
                       at a time, so the loop can never see a bar that had not become
                       available yet. The gate is the reason this is not a backtest
                       that peeks.
  quote_round_trip_cost - both legs priced before a trade is accepted, and a
                       `CostRefused` stops the trade instead of defaulting a number.
  breakeven gate     - a signal whose target does not clear `breakeven_bps` is
                       refused and counted, which is most of them.
  OrderIntentWal     - every decision is journalled durably BEFORE it is acted on,
                       with a derived client order id, so a crash leaves a record
                       that prompts a query rather than a resend.
  Order              - real lifecycle transitions; a fill on a terminal order raises
                       rather than being tolerated.
  simulate_fills     - both accountings, optimistic and pessimistic, never blended.
  signed_cash        - fees charged at the rate for the liquidity each fill took,
                       from the venue's own schedule.

Three honest limits of this run, none of them hidden:

  * **The touch is derived from the bar, not read from a book.** The store's book
    dataset covers 2026-08-08 10:07-10:59 and its bars cover 2026-08-02 15:38 to
    2026-08-03 18:12 - they do not overlap, so there is no real bid/ask at the
    timestamps being traded. `best_bid`/`best_ask` are therefore the bar's low and
    high, which makes the pessimistic crossing price the bar's extreme: worse than
    reality, which is the safe direction, and stated rather than assumed away.
  * **Participation is uncalibrated and must be declared.** It is open question 1 of
    the paper-execution design and is not settled, so `--participation` has no
    default. Every result carries `uncalibrated`.
  * **Spread and impact are zero in the cost quote**, as `quote_round_trip_cost`
    already reports through its own unverified inputs. That understates cost.

    python -m paper.prove_plumbing --participation 0.1 --symbol BTCUSDT --venue binance
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from cost.round_trip_cost import CostRefused, quote_round_trip_cost
from execution.order_intent_wal import OrderIntent, OrderIntentWal
from execution.order_lifecycle import Order
from paper.fill_model import (
    MarketEvent, Participation, RestingOrder, signed_cash, simulate_fills,
)
from store.clock_gated_reader import ClockGatedReader

BARS_DATASET = "bars_60000000000ns"
STRATEGY = "prove-plumbing-crude-momentum"
# One bar. An intent that outlives the bar it was decided on would fire against a
# market it never saw, which is the failure `valid_for_ns` exists to prevent.
INTENT_VALID_NS = 60_000_000_000


@dataclass
class Tally:
    """What happened, counted by outcome rather than summarised into one number."""

    bars_read: int = 0
    signals: int = 0
    refused_non_positive_price: int = 0
    refused_by_cost_engine: int = 0
    refused_below_breakeven: int = 0
    orders_journalled: int = 0
    filled: int = 0
    unfilled: int = 0
    optimistic_cash: Decimal = Decimal("0")
    pessimistic_cash: Decimal = Decimal("0")
    uncalibrated_fills: int = 0


def _paper_transport(intent: OrderIntent, client_order_id: str) -> dict:
    """Stands in for a venue. Accepts, and touches no network.

    Named a transport because that is the seam the real one plugs into: the WAL
    calls it after the intent is durable, so the ordering that matters - record,
    then act - is exercised here exactly as it would be live.
    """
    return {"status": "accepted", "client_order_id": client_order_id,
            "venue": intent.venue, "paper": True}


def run(store_root: Path, wal_root: Path, symbol: str, venue: str,
        participation: Decimal, target_bps: Decimal,
        notional: Decimal, out=sys.stdout) -> Tally:
    """Walk the tape one bar at a time and report every outcome."""
    reader = ClockGatedReader(store_root, BARS_DATASET)
    # Read once at the end of time only to learn WHICH bars exist; every decision
    # below re-reads at that bar's own availability time.
    catalogue = reader.read_as_of(2**62, symbols=[symbol])
    catalogue = catalogue[catalogue["venue"] == venue].sort_values("event_time_ns")
    if catalogue.empty:
        print(f"no bars for {symbol} on {venue}; nothing to prove", file=out)
        return Tally()

    wal = OrderIntentWal(wal_root)
    tally = Tally()
    schedule_bps_maker = Decimal("0")
    previous_close: Decimal | None = None

    for row in catalogue.itertuples(index=False):
        available_at = int(row.availability_time_ns)
        # THE point of the gate: ask for everything knowable at this bar's
        # availability time. A loop that read the whole frame once and iterated it
        # would be indistinguishable in output and would have seen the future.
        visible = reader.read_as_of(available_at, symbols=[symbol])
        visible = visible[visible["venue"] == venue]
        if visible.empty:
            continue
        tally.bars_read += 1

        close = Decimal(str(row.close))
        low, high = Decimal(str(row.low)), Decimal(str(row.high))
        volume = Decimal(str(row.volume))

        # A zero price is not a price, and this is not hypothetical: the real store
        # holds a BTCUSDT bar at event_time 1785694440000000000 with open 63292.5,
        # high 63294.8, 377 trades - and low and close both 0.0. Refused and counted
        # by name rather than skipped, because a bar like that is a defect upstream
        # and a loop that quietly steps over it is how it stays unfixed.
        if close <= 0 or low <= 0 or high <= 0:
            tally.refused_non_positive_price += 1
            print(f"  REFUSED (price) {symbol} bar at {int(row.event_time_ns)}: "
                  f"open={row.open} high={high} low={low} close={close} "
                  f"trades={getattr(row, 'trades', '?')} - a zero price is not a price",
                  file=out)
            continue

        if previous_close is None or volume <= 0:
            previous_close = close
            continue

        # The crude signal, stated plainly so nobody mistakes it for research: rest a
        # BUY one tick below the previous close and hope the bar trades through it.
        # It is a coin flip with extra steps.
        limit = previous_close
        previous_close = close
        tally.signals += 1

        quote = quote_round_trip_cost(
            venue, symbol, notional, order_type="maker",
            at_ns=int(row.event_time_ns), store_root=store_root)
        if isinstance(quote, CostRefused):
            tally.refused_by_cost_engine += 1
            print(f"  REFUSED (cost) {symbol} @ {limit}: {quote.reason}", file=out)
            continue
        schedule_bps_maker = quote.fee_bps

        if target_bps <= quote.breakeven_bps:
            tally.refused_below_breakeven += 1
            continue

        intent = OrderIntent(
            strategy=STRATEGY, symbol=symbol, venue=venue, side="BUY",
            quantity=(notional / limit), created_at_ns=int(row.event_time_ns),
            valid_for_ns=INTENT_VALID_NS, price=limit)
        # Journalled BEFORE the fill is simulated, in that order, because that is
        # the ordering the live path depends on and the one worth proving.
        receipt = wal.submit(intent, _paper_transport, now_ns=int(row.event_time_ns))
        tally.orders_journalled += 1
        order = Order(client_order_id=receipt["client_order_id"], symbol=symbol,
                      venue=venue, side="BUY", quantity=intent.quantity)

        # One event per bar: the print is the close, and the touch is the bar's own
        # range. See the module docstring - there is no book at these timestamps.
        event = MarketEvent(trade_price=close, trade_quantity=volume,
                            best_bid=low, best_ask=high)
        outcome = simulate_fills(
            RestingOrder(side="BUY", limit_price=limit, remaining=intent.quantity),
            (event,),
            participation=Participation(fraction=participation, calibrated=False))

        if not outcome.optimistic:
            tally.unfilled += 1
            wal.resolve(receipt["client_order_id"], "cancelled",
                        "bar closed without printing through the resting price",
                        now_ns=int(row.event_time_ns) + INTENT_VALID_NS)
            continue

        tally.filled += 1
        if outcome.uncalibrated:
            tally.uncalibrated_fills += 1
        for fill in outcome.optimistic:
            order = order.fill(fill.quantity, fill.price)
        wal.resolve(receipt["client_order_id"], "filled",
                    f"paper fill {order.filled_quantity} @ {order.average_fill_price}",
                    now_ns=int(row.event_time_ns) + 1)

        taker_bps = quote.fee_bps
        tally.optimistic_cash += signed_cash(
            outcome.optimistic, side="BUY",
            maker_bps=schedule_bps_maker, taker_bps=taker_bps)
        tally.pessimistic_cash += signed_cash(
            outcome.pessimistic, side="BUY",
            maker_bps=schedule_bps_maker, taker_bps=taker_bps)

    return tally


def _report(tally: Tally, symbol: str, venue: str, out=sys.stdout) -> None:
    print(f"\n{symbol} on {venue} — plumbing walkthrough", file=out)
    print(f"  bars read through the clock gate : {tally.bars_read}", file=out)
    print(f"  signals raised                   : {tally.signals}", file=out)
    print(f"  refused, non-positive price      : {tally.refused_non_positive_price}", file=out)
    print(f"  refused by the cost engine       : {tally.refused_by_cost_engine}", file=out)
    print(f"  refused below breakeven          : {tally.refused_below_breakeven}", file=out)
    print(f"  intents journalled to the WAL    : {tally.orders_journalled}", file=out)
    print(f"  filled / unfilled                : {tally.filled} / {tally.unfilled}", file=out)
    print(f"  fills carrying `uncalibrated`     : {tally.uncalibrated_fills}", file=out)
    # Labelled as spend, not as P&L, because this demo never closes a position:
    # every order is a BUY and no exit leg is simulated. A figure called "P&L" here
    # would be a loss that is really just the cost of buying, which is precisely the
    # kind of number that gets quoted later without its caveat.
    print(f"  entry spend, optimistic fills    : {tally.optimistic_cash:.2f} "
          f"(NOT P&L - no exit leg)", file=out)
    print(f"  entry spend, pessimistic fills   : {tally.pessimistic_cash:.2f} "
          f"(NOT P&L - no exit leg)", file=out)
    # Never one blended number, and the invariant is asserted rather than trusted:
    # pessimistic must never look better than optimistic, or the accounting is
    # producing a free lunch the strategy did not earn.
    if tally.filled and tally.pessimistic_cash > tally.optimistic_cash:
        print("  INVARIANT VIOLATED: pessimistic beat optimistic", file=out)
    print("\n  This is a wiring proof. No purge, no embargo, no holdout, no "
          "Deflated Sharpe,\n  and no exit leg - positions are opened and never "
          "closed. It says the pipes\n  connect. It says nothing about edge.",
          file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prove_plumbing",
        description="Drive one crude signal through the whole paper path. Not a strategy.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--venue", default="binance")
    # No default, deliberately. The participation fraction to use when no depth was
    # measured is open question 1 of the paper-execution design and is not settled;
    # defaulting it here would answer it by accident, in a demo.
    parser.add_argument("--participation", required=True, type=Decimal,
                        help="share of printed volume the queue would have taken "
                             "(REQUIRED: the uncalibrated default is not decided)")
    parser.add_argument("--target-bps", default=Decimal("50"), type=Decimal,
                        help="the move the signal claims, in bps. Must clear breakeven")
    parser.add_argument("--notional", default=Decimal("1000"), type=Decimal)
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--wal-root", required=True,
                        help="where the order intent WAL is written. Required so a "
                             "demo run cannot land in a real trading journal")
    args = parser.parse_args(argv)

    if args.participation <= 0 or args.participation > 1:
        parser.error("--participation must be in (0, 1]")

    tally = run(Path(args.store_root), Path(args.wal_root), args.symbol, args.venue,
                args.participation, args.target_bps, args.notional)
    _report(tally, args.symbol, args.venue)
    return 0


if __name__ == "__main__":       # pragma: no cover - entry point
    raise SystemExit(main())
