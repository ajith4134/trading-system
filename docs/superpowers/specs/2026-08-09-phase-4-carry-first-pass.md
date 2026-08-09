# Phase 4 — one family end to end: funding carry

**First pass, 2026-08-09. Written to be argued with, not executed.** Section 11 is the part
that needs a decision before any of the rest is worth building.

`ARCHITECTURE.md` §3: *"Phase 4 — One family, end to end. Carry (funding/basis) — latency-immune,
viable at this scale. Take it through the full pipeline to reduced-size live. Prove the machine on
the family most likely to work."*

The phrase that governs this spec is **prove the machine**. The deliverable is a pipeline that can
refuse a strategy, demonstrated on a family chosen because it is the one most likely to survive.
A promoted strategy is a possible *outcome*; it is not the definition of done.

---

## 1. The measured starting position

Everything here was read off the live store on 2026-08-09, not recalled.

| Dataset | Rows | Symbols | Venues | Span |
|---|---|---|---|---|
| `bars_60000000000ns` | 715,339 | 1,718 | binance, binance-spot, hyperliquid | 2026-08-02 → 2026-08-08 |
| `funding` | 67,763 | **3** | **binance only** | **2026-08-08 04:06 → 2026-08-09 05:59** |
| `book` | 540 | 3 | binance, binance-spot | 2026-08-08 → 2026-08-09 |

**Carry is made of funding, and we have 26 hours of it on 3 symbols.** At 8-hourly settlement that
is roughly three funding prints per symbol. Six days of bars is thin; three funding events is not
thin, it is nothing.

What *is* ready: the clock-gated reader, the cost engine (`quote_round_trip_cost`, live fee
verification), the paper fill model, the order-intent WAL with a kill-switch gate, and the whole of
`validation/` — Trial Registry, Holdout Custodian, CPCV, deflated Sharpe, PBO, SPA, promotion gate.
That library has been complete and **callerless since it was written**. Phase 4 is its first caller,
and wiring it clears nine of the eleven baselined unsupported claims.

---

## 2. The one thing that must happen today, whatever else is decided

**Widen funding capture to the whole perp universe, now.**

`binance.poll_specs` fetches `premiumIndex` **per symbol**, and `capture.cli` passes it the three
core symbols. The adapter's own comment explains why:

> omitting `symbol` returns every perpetual on the venue at request weight 10, and writing several
> hundred instruments to disk in order to read three of them is not a raw archive of what was asked
> for.

That reasoning was right when the universe was three symbols and is now the binding constraint on
the entire phase. One all-market request at weight 10 returns every perp's funding, against a
2,400/minute budget.

This is urgent for one reason and it is not effort: **funding history cannot be backfilled into the
archive after the fact.** Binance publishes historical funding over REST, so the *numbers* are
recoverable — but they arrive with no capture provenance, no availability time we observed, and no
place in a bitemporal store whose entire claim is that every row records when we could have known
it. Every hour this waits is an hour of carry history that can only ever be reconstructed, never
captured. The same argument the capture supervisor already makes about the broad tail:

> Widening the tail is cheap today and impossible to backfill, so the default here is the one
> decision worth revisiting.

Do this before the spec is settled, because it is right under every branch of §11.

---

## 3. Which carry, and why not the others

Three candidates sit in the ledger. They differ in what they need.

| Variant | Ledger | Needs | Available? |
|---|---|---|---|
| **Funding carry, perp vs spot hedge** | SP-056 | perp funding + spot price for the same asset | binance perps + binance-spot — **yes, once §2 lands** |
| Spot-perp basis | SP-058 | same two legs, traded on the price gap rather than the funding print | same |
| Cross-venue funding differential | SP-059 | funding on two venues | **no** — hyperliquid funding is not captured at all |

**Recommendation: SP-056, funding carry, delta-hedged against spot.** It is the row
`ARCHITECTURE.md` marks *"start here"*, both legs are on venues already captured, and it is the
variant whose edge is a *scheduled, non-discretionary flow* — which SP-100 in the ledger calls the
most durable edge class there is.

SP-058 is close enough that it should be treated as the same build with a different trigger, not a
second project. SP-059 is out until hyperliquid funding is captured, which §2 should cover for both
venues while it is in there.

---

## 4. Universe selection — a hard requirement, not a detail

The store contains **89,097 bars across 536 symbols priced in TRY, EUR, JPY, IDR, BRL, BTC and
ETH** — 29.4% of binance-spot's rows. The builder stopped adding more on 2026-08-09; the store is
append-only, so those rows are there forever.

**Universe selection must call `dollar_quoted_symbols`.** Carry ranks cross-sectionally, and a lira
price against a USDT price is not a comparison. This is written here because it is exactly the kind
of requirement that gets rediscovered as a bug.

The tradeable set is then narrowed further, in this order:

1. dollar-quoted (above)
2. perp exists **and** a spot leg exists for the same asset — carry without a hedge is a
   directional bet wearing a carry name
3. clears the minimum viable size of §5a.5: expected carry per settlement must exceed
   `quote_round_trip_cost` for the round trip, with funding charged at actual settlement times

Step 3 is a **cost-engine call, not an assumption**, and it will eliminate most of the universe.
That is the correct outcome and the number is worth reporting: how many of ~566 perps carry enough
to pay for their own hedge is itself a finding.

---

## 5. The signal, and the one rule that must not bend

§5a.5 of the goal spec:

> **Setup definitions are universal and parameter-free across symbols.** Per-symbol variation comes
> only from *normalisation* — z-scores or percentiles against that symbol's own history — never
> from fitted per-symbol parameters. One setup on 1,290 symbols is **one hypothesis with 1,290
> samples**. It becomes 1,290 hypotheses the moment per-symbol tuning is allowed. This is the
> single most dangerous thing that could be implemented here.

So the signal is one expression, evaluated identically on every symbol:

- **carry** = funding rate at the next settlement, annualised, net of the round-trip cost of holding
  the hedged pair through it
- **normalised** as a percentile against that symbol's own funding history, never against the
  cross-section
- **entered** when net carry clears the cost gate *and* the acceptance threshold, which per §5a.5
  **rises with opportunity flow** — the more symbols qualifying, the pickier each must be
- **default action is stand aside**

Free parameters exist — lookback for the percentile, the threshold curve, hold horizon. Each one is
a Trial Registry entry, and the registry counts cumulative N across *all* searches including
abandoned ones. That is what makes the deflated Sharpe honest downstream.

---

## 6. The breadth gate comes before the strategy, not after

§5a.4 is explicit, and it gates this whole phase:

> Define candidate setups, fire them historically across the universe, and measure (1) clustering
> of trigger times, (2) correlation of the resulting return streams. **If triggers cluster and
> returns correlate, the architecture is one macro bet wearing 1,290 hats and must be redesigned.**

Funding is *unusually* well placed to pass this — settlement is scheduled per venue, and
symbol-specific funding is named in §5a.4 as one of the idiosyncratic setups that stay independent.
But "well placed to pass" is a hypothesis, and this gate exists because that hypothesis is the
expensive one to get wrong.

**It cannot run on 3 symbols and 26 hours.** It runs when §2 has accumulated enough funding history,
and it runs *before* the strategy is built rather than as a post-hoc check on results.

---

## 7. Validation — the part that is the actual deliverable

Every module in `validation/` gets its first caller here, in this order:

1. **Trial Registry** — structurally impossible to evaluate a configuration without incrementing it
2. **Holdout Custodian** — passed to `ClockGatedReader` so the holdout range physically refuses
   reads. The reader already accepts one and no caller has ever supplied it
3. **Purged CPCV** — purge and embargo configured for *this family's* label horizon, not shared
4. **Deflated Sharpe** — as the fitness function inside the search loop, not a report on the winner,
   fed the registry's cumulative N and not the CPCV path count
5. **PBO / SPA** — on the finalist set
6. **Promotion gate** — the single call that says yes or no

The prior-art warning in `CLAUDE.md` applies directly and was earned on exactly these modules: a
donor CPCV purge was one-sided and a donor deflated-Sharpe gate was fed the path count as its trial
count. **All three prior-art defects found that day failed in the flattering direction.** Read the
code, not the ledger's note about it.

---

## 8. Definition of done

Phase 4 is done when **all** of these are true:

1. The breadth gate has been run on real captured funding and its result is written down —
   including if the answer is "redesign"
2. A carry configuration has been through the full pipeline and the promotion gate has returned a
   verdict. **A refusal is a pass for this phase.** The machine working is the deliverable
3. `validation/` is no longer reachable-by-nothing — the ratchet baseline drops VX-001 … VX-011
4. If and only if the gate promotes: reduced-size live, under the §6 human gate, with
   `state_recovery` wired because a live position now exists to reconcile
5. Each new module carries its learning / reasoning / depth verdict per §1a.6, and the status wall
   renders them — `NOT MEASURED` where untested, never green

---

## 9. What this phase does NOT do

- No second family. §5 of the goal spec sequences that to Phase 6
- No portfolio allocation — one strategy has nothing to allocate between
- No options. Closed until P3, and short options are excluded from the default action space
- No per-symbol parameter fitting, under any framing (§5)

---

## 10. Intelligence axes, declared up front

Per §1a.6, no module ships without a verdict. Declared here so they are designed for rather than
retrofitted:

| Axis | How this phase must satisfy it |
|---|---|
| **Learning** | Every number traces to captured data or to a registered trial. The threshold curve is fitted from opportunity flow, not chosen. A hardcoded funding threshold would be "a person's guess wearing a variable name" |
| **Reasoning** | The mechanism declaration is mandatory: *who pays this carry, and why do they keep paying?* Plus a measurable proxy for that mechanism's health, so decay is detectable before P&L can show it |
| **Depth** | The master test — **does it change what the system does when it is wrong?** Concretely: the strategy must be able to stand aside, and the mechanism proxy must be able to retire it while its P&L still looks fine |

---

## 11. Open questions — the reason this is a first pass

These are decisions, not gaps. Each changes what gets built.

1. **Do we wait for real funding history, or backfill from REST?** Waiting means the breadth gate
   is weeks away and Phase 4 stalls. Backfilling gets an answer in days, from data with no capture
   provenance in a store whose whole claim is provenance. My lean: **capture broad now (§2), and
   backfill into a separate clearly-labelled dataset for the breadth gate only — never into
   `funding`, and never as an input to a promotable backtest.**

2. **Is the hedge leg spot, or is it nothing?** Delta-hedged carry is the textbook trade and doubles
   the cost. Unhedged funding capture is cheaper and is a directional bet wearing a carry name.
   There is a middle — hedge only above a size threshold — which is a parameter and therefore a
   trial.

3. **What is the tail cap for this family, in numbers?** §0's objective is *subject to* a hard cap,
   and §6 says the ceiling is yours alone to set. I cannot pick it. Nothing sizes until it exists.

4. **What counts as enough history before the promotion gate may return yes?** MinBTL will have an
   opinion given the trial count, but the honest answer may be "more calendar time than we have",
   and it is better to agree that now than to discover it at the gate.

5. **Scope of §2 — binance perps only, or hyperliquid too?** Hyperliquid funding unlocks SP-059
   cross-venue carry later, costs little now, and is equally unbackfillable.
