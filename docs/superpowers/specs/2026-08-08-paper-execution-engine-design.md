# Paper Execution Engine — design

Date: 2026-08-08
Status: **design, pending review.** No code written.
Preceded by: `2026-08-08-final-project-goal-design.md`, the execution-layer floor (`832adef`).

## Why this exists

`ARCHITECTURE.md:92` states that *"paper mode manufactures edge"* — that is the failure this
design is written against, not a risk it mentions in passing. Every decision below is chosen
to make an optimistic assumption impossible to hold silently.

The engine is also the missing consumer for two things already built and unwired:

- the **Holdout Custodian** (`src/validation/holdout_custodian.py`) has had no reader
  attached to it. `ClockGatedReader.__init__` takes `custodian` as an *optional* argument,
  and its own docstring warns that a reader built without one is unguarded. The paper engine
  is the first component that must always pass one.
- the **execution floor** — `OrderIntentWal`, `Order`, `recover_and_reconcile` — has had no
  transport. This design supplies one that is purely local.

Ledger rows this touches: EX-004 (order types, partial), EX-007 (maker-vs-taker per order),
EX-013 (strategy-facing cost gate), EX-017 (queueing / maker fill probability, partial),
VX-012 (shadow alignment), VX-014 (backtest-vs-live divergence), VX-125 (shadow deployment),
SP-054 (paper-wallet fill fidelity — design-only in every prior repo, never built).

## Decisions taken (interview, 2026-08-08)

| Question | Decision |
|---|---|
| What does it run on first? | **Tiered.** Discovery across all 2,123 symbols on top-of-book + the cost engine; finalists re-run on real 20-level depth (BTC/ETH/SOL). |
| Replay or forward? | **Both.** On-demand replay over the bitemporal archive, plus a long-running forward supervisor alongside capture/store/offload. |
| Maker fills, with no queue data? | **Score every strategy under both accountings.** Gate promotion on the pessimistic one; report the gap as a first-class output. |

The third decision is a refinement of "maker only on trade-through", after the user asked for
all options to be thought through. Three flaws in the plain version, and their fixes:

1. **Trade-through is volume-blind.** One lot printing through a 100-lot resting order is not
   a fill. Fill is capped at `min(remaining, volume_that_printed_through)`.
2. **It still assumes you get all of that volume.** Others are in the queue. That needs a
   participation rate, and an invented rate is exactly what manufactures edge — so it is
   **calibrated from the depth archive** (resting size at the touch on BTC/ETH/SOL) and
   carried as a receipted, declared number. One day of depth is thin; a measured starting
   point with recorded provenance still beats a round number.
3. **The maker-or-taker choice is avoidable.** Score both. The gap between them *is* VX-012's
   "realized-vs-assumed fill gap" and VX-014's divergence signal. A strategy that survives
   only under optimistic fills is the single most important fact about it, and this makes it
   visible rather than buried inside one assumption.

```
                        optimistic            pessimistic
resting BUY @ 100.00    maker, 2.0 bps        taker, 5.0 bps
  trade prints 99.98    fill min(remaining,   cross the spread
  size 40               40 x participation)   at the touch
  no print through      unfilled              unfilled

promotion gate  -> requires PASS on pessimistic
execution risk  -> pessimistic minus optimistic, reported per strategy
participation   -> calibrated from depth archive, receipted, default conservative
```

Fee rates above are the **verified** ones, not a fee page: `~/capture/fee-verification/latest.json`
records binance perp maker 2.0 / taker 5.0 bps and hyperliquid 1.5 / 4.5 bps, fetched from
signed endpoints at the account's own tier. Maker-both-legs is a 4 bps round-trip hurdle
against taker's 10 — a bar 2.5x lower. That single assumption is the largest lever in the
engine, which is why it is never taken unilaterally.

Cost of scoring both: one extra cost accounting per fill (cheap), and a calibration job that
must refuse when depth is absent rather than default silently.

## Architecture — `src/paper/`

The load-bearing idea: **the paper broker is a transport for the WAL already built.**

`OrderIntentWal.submit(intent, transport, now_ns=None)` takes
`transport: Callable[[OrderIntent, str], dict]` — exactly the callable a paper broker can be.
So paper inherits, for free and without a parallel implementation:

- deterministic idempotency (`derive_client_order_id`, sha256 over every distinguishing field)
- write-before-send durability (the `submitted` record is appended and fsynced first)
- signal expiry (`IntentExpired` raised pre-write when `valid_for_ns` has closed)
- the unknown-outcome path (a transport that raises leaves `outcome=unknown`, the only state
  that prompts a query rather than a resend)

This is `ARCHITECTURE.md`'s "same code path across backtest/paper/live" made literal: paper
exercises the code a live transport would, rather than a simulator beside it.

| Component | Responsibility | Depends on |
|---|---|---|
| `fill_model.py` | Pure. (resting order, market events, participation) → fills under **both** accountings | nothing |
| `participation_calibration.py` | Measures capture fraction from the depth archive; **refuses** when depth absent; writes a receipt | `store`, `cost` |
| `paper_broker.py` | The WAL transport. Registers resting orders, applies fills via `Order.fill()` | `order_lifecycle`, `fill_model` |
| `market_replay.py` | Feeds events from `ClockGatedReader` — simulated clock for replay, wall clock forward. One door | `store` |
| `position_book.py` | The local position model; supplies `local_positions` to `state_recovery` | `order_lifecycle` |
| `scripts/paper_supervisor.sh` | The forward process, alongside capture/store/offload | all |

`fill_model.py` is pure on purpose: it is the one component whose invariant must be provable
by property test, and a pure function of (order, events, participation) is testable without a
store, a clock, or a filesystem.

## Data flow

```
strategy -> OrderIntent -> WAL.submit(intent, paper_broker)
                              |  (durable + fsynced before the broker sees it)
                              v
                        paper_broker  <- market_replay <- ClockGatedReader(custodian)
                              |                              (holdout sealed)
                              v
                        fill_model -> optimistic fills | pessimistic fills
                              |
                              v
                     Order.fill() -> position_book -> cost engine (both accountings)
                              |
                              v
                  TrialRegistry.evaluate -> promotion_gate (gated on pessimistic)
                                            + fill-gap metric (VX-012/VX-014)
```

Note the reader is constructed **with** a custodian, always. `ClockGatedReader.read_as_of`
calls `custodian.assert_readable(sim_clock_ns)` before touching any data, so a refused query
never loads the rows it was refused.

## The two tiers, and what tier 1 honestly cannot do

**Tier 1 — 2,123 symbols, since Aug 3.** Trades and book ticker, no depth. `impact_bps` in
`src/cost/spread_and_depth.py` **refuses** rather than extrapolating, and
`quote_round_trip_cost` returns `CostRefused` rather than a number. That behaviour is
inherited, not re-implemented.

Consequence, stated plainly: **tier 1 cannot price impact.** So it refuses order sizes above a
declared fraction of observed trade volume instead of pretending to fill them. Discovery at
this tier answers "is there a signal at all", never "what would it have earned at size".

**Tier 2 — BTC/ETH/SOL, depth archive.** Real 20-level depth, so impact is priced and the
shadow stage's execution-quality metrics become measurable. Finalists only.

## Error handling — all refusals, no defaults

| Condition | Behaviour |
|---|---|
| No depth, size too large | `CostRefused` propagates; the signal is recorded as refused, not silently skipped |
| Participation uncalibrated | conservative default **plus** an `uncalibrated` flag on the result; tier-2 promotion requires calibration |
| Venue halted (`VenueHaltRegistry`) | no fills, positions frozen |
| Kill file present (`watchdog.is_killed`) | engine refuses to start |
| Replay touches the sealed holdout | `HoldoutSealed` from the reader |
| Fill arrives on a terminal order | `InvalidTransition` from `Order.fill()` — raised, never tolerated |

`CostRefused.__float__` raises by design; nothing in this engine may reach for a number on a
refusal and find one.

## Testing

**The load-bearing invariant, as a property test: pessimistic P&L ≤ optimistic P&L, always,**
for every strategy and every event sequence. A violation means the two accountings have
diverged such that the "optimistic" path is conservative somewhere — and the entire promotion
argument, which rests on gating the pessimistic number, collapses.

Alongside it:

- never fill more than the volume that printed through
- never fill without a print
- hand-computed expected fills over synthetic event sequences (the arithmetic checked by hand,
  not by the code under test)
- a replay whose clock enters the sealed holdout raises `HoldoutSealed` and reads nothing
- an intent submitted twice raises rather than producing a second order
- calibration with no depth present **refuses** and writes no receipt

TDD per the standing rule: test first, watched failing, then the minimal code.

## Status wall (Rule 8)

Two tiles, both measured, neither asserted:

- **paper engine** — reads the supervisor's own heartbeat and last-processed event time.
  `NOT BUILT` until the process has run; never green from the presence of the code.
- **participation calibration** — reads the receipt. `NOT MEASURED` when absent, with the
  depth-coverage window and measurement timestamp shown when present.

## Scope boundary

No live transport, no credentials, no network. `paper_broker` is a pure local object. The
existence of verified fee data does not make this engine able to trade, and nothing here is a
step toward placing an order — that remains an explicit, separately-authorised decision.

## Open, and deliberately not decided here

1. **Participation default when uncalibrated** — a number is needed; the honest one is
   conservative, but "conservative" needs a value. Proposed at build time from the calibration
   run's own lower bound, not chosen now.
2. **Declared volume fraction for tier-1 size refusal** — same shape; wants the trade-volume
   distribution measured before a fraction is picked.
3. **Forward-supervisor cadence** — event-driven off the capture feed vs a fixed tick. Affects
   ops cost more than correctness.

## Review checkpoint

This document is the deliverable of the brainstorming phase. **Nothing is implemented.**
Next step is the user's review of this design; code begins only after it.
