# THE GOAL — what this system is, when it is finished

**Status:** settled with the user in interview, 2026-08-08.
**Authority:** this document defines the *goal*. `ARCHITECTURE.md` defines the *structure*,
`FEATURES.md` the *capability list*, `DECISIONS.md` the *record of why*. Where any of them
conflicts with this file on a question of purpose, this file wins. Where this file is silent
on structure, they win.

**Read this first.** Every prior session re-derived the goal from scattered documents and got a
slightly different answer each time. That is what this file exists to stop.

---

## 0. Prime directive

> **Earn money constantly, at the smoothest curve honestly achievable, without ever selling
> insurance to get there.**

Formally, the objective function is:

**Maximise the fraction of green days, subject to a hard tail-loss cap the system cannot raise.**

Not a target return. Not a Sharpe number. A *shape*: up most days, and never a single day that
undoes a year.

> **The tail cap in this objective is the immutable ceiling of §6, and nothing else.** The system
> may move its own operating limits freely *inside* that ceiling — that is the autonomy decision in
> §6 — but the ceiling is the cap this objective is subject to, and no component can edit it. Read
> §0 and §6 together: **operating limits are the system's to move; the ceiling is the user's
> alone.**

### Why the cap is part of the objective, not a constraint bolted on afterwards

"Maximise green days" on its own is the objective that selects for **selling insurance** — short
volatility, short options, premium harvest with a hidden left tail. Those strategies are green
almost every day. LTCM was. The 2018 short-VIX funds were. They return the entire multi-year gain
in a single session.

The tail cap is the term that makes the objective safe to optimise against. It is not a safety
tax; without it the objective is actively dangerous. `ARCHITECTURE.md` already excludes short
options from the default action space, and a daily-green objective without the cap would fight
that rule.

### What was asked for, and the arithmetic that shaped it

The user's words were **"green every day."** That is recorded here as the *aim*, together with why
it is not the *target*:

- At annual Sharpe 1.5 — the "genuinely good" band in `DECISIONS.md` §11 — daily Sharpe is
  1.5 ÷ √252 = **0.094**, which puts **~46% of individual days negative**. Roughly 116 red days a
  year.
- To make even **95% of days green** requires an annual Sharpe near **26**. `DECISIONS.md` §11 says
  above 3, suspect a bug. Medallion is estimated near 7.
- The gap is arithmetic. No amount of engineering closes it.

So the aim is expressed as a maximisation, not a promise. The system reports its actual green-day
rate on the wall and never asserts a rate it has not measured (Rule 8).

**Reachable, in order of smoothness:**

| Profile | Green days | Return | Note |
|---|---|---|---|
| Market-neutral carry (funding + basis, delta-hedged) | ~70–85% | ~2.8% annualised venue-average | The smoothest curve available. Compressed from 10–30% — `DECISIONS.md` §11 |
| Carry + low-frequency directional | ~55–60% | Sharpe 1.4–1.7 | Losing months ~1 in 3 |
| Full menu | ~52–55% | Highest expected | Multi-month red stretches |

---

## 1. The learning curve is a tracked metric, not a claim

**As cumulative trades grow, win rate and the count of profitable trades must both rise.**

This is the operational meaning of "self-improving." It is measured, not asserted:

- Win rate is computed **out-of-sample only** — on paper-forward, shadow and live fills, never
  in-sample backtest fills.
- It is plotted against **cumulative trade count**, not calendar time. Observation frequency buys
  statistical power; elapsed days do not (`DECISIONS.md` §11).
- A rising win rate is only counted as improvement when **total P&L and the green-day rate rise
  with it**. A system can trivially raise win rate by taking only tiny certain trades, or by
  cutting winners early and holding losers. Requiring all three to move together removes both
  cheats.
- The curve appears on the status wall with its own tile, and a flat or falling curve over a
  meaningful trade count is a **finding the system must report**, not a state it may hide.

---

## 2. What the finished system does on a normal day

Nobody touches it.

It wakes into a capture stream it never stopped writing. It reconciles local state against the
exchange before trading a single unit, and **refuses to start if reconciliation fails**.

Nine bots think. Three arbiters select. PROFIT-TAIL times entries and works the open positions.
The risk gate refuses anything that breaks a limit. The allocator moves capital toward what is
currently paying and starves what is not.

In the background a research loop that never stops: it reads its own ledger, forms a hypothesis,
writes it as executable code, runs it against sealed data through a deterministic evaluator it
cannot edit, and either promotes it to paper or records the failure. Most fail. **The failures are
the asset** — they are what makes the Deflated Sharpe honest.

A digest arrives at day's end. The wall is there if wanted. Nothing asks for anything unless it
wants more capital or wants to go live.

---

## 3. The earning core, and how the tree earns its place

Two of the user's answers pull against each other, and this section is the ruling that reconciles
them.

- The objective demands **delta-neutral carry** — market-neutral, smooth, ~70–85% green days.
- The definition of done demands **nine directional bots** — a directional book runs ~52–55% green
  days and *lowers* the portfolio rate when added naively.

**Ruling: the objective becomes the allocator's fitness function.**

**Carry is the earnings core.** Funding-rate carry and spot-perp basis are what "constantly" rests
on. They are latency-immune and viable at this capital — the only two families `FEATURES.md` §4
marks "start here."

**Directional bots do not receive capital for being clever, or for a good standalone Sharpe.** They
receive capital only when adding them **measurably raises the portfolio's green-day rate at the
same tail cap**. That happens when they diversify the carry book. It does not happen when they
merely add return with correlated risk.

This needs no new machinery — the "starving" allocator already in `DECISIONS.md` becomes the
enforcer of the prime directive. The full tree still gets built and measured. It competes *for*
smoothness rather than against it.

---

## 4. Capital is a dial

The user sets it. Anywhere from roughly **$2k to $100k+**. It may be changed at any time and the
system must keep functioning across the whole range.

**Every size in the system is a fraction of NAV, never an absolute.**

The system can be capital-*agnostic* in sizing. It cannot be capital-*indifferent* in its strategy
menu — exchange minimum notionals (~$5–10) and options contract granularity are physics, not
design choices.

**Therefore: a capital feasibility gate.** Each strategy declares its **minimum viable capital**.
The gate enables and disables families automatically as the dial moves. The wall shows which
families are off and the exact reason. Same machine, smaller menu, and it says so rather than
silently underperforming.

**The system may request more capital. It may never grant itself any.**

---

## 5. Scope

**Crypto only.** Confirmed 2026-08-08, consistent with `bull-bear-profit-agents-spec.md` §1.

**Zero NSE.** No Indian equities, no F&O, no currency, no commodity segment. No forex, no
equities of any market. The NSE material in the corpus and in `ajith4134/nse-botonly` is a
**feature donor** — a source of capabilities to adapt for crypto — never a target market.

### Instrument segments

| Segment | Phase | Venue |
|---|---|---|
| Spot | P1 | Binance, Hyperliquid |
| Perpetual futures | P1 | Binance USDⓈ-M, Hyperliquid |
| Dated futures (calendar) | P2 | Binance |
| Options | P3 / Phase 6 | Deribit — behind an IV surface and a second, Greeks-based risk gate |

### Closed segments — refused with reasons, not overlooked

Market making (no reachable rebate tier, 10–20× queue disadvantage) · latency arbitrage (5–10 μs
races against an ~8 ms cloud floor) · triangular arbitrage (found never profitable after fees,
2024 Binance study).

### Venues

**Execution:** Binance + Hyperliquid — deliberately unalike, so they fail differently.
**Reference price only:** Kraken, OKX, Coinbase — feeding the liquidity-weighted consolidated
price, carrying no key or execution risk. **Deferred:** Bybit. **Options:** Deribit, Phase 6 only.

---

## 5a. Universe-wide scanning — a property of the system, not a strategy

**Full analysis: `~/research/DESIGN-NOTE-universe-wide-scanning.md`, rated 8/10 conditional on the
gate in §5a.4.**

> **The system watches every tradeable symbol in every segment, and acts on any one only when that
> symbol's own condition fires.**

**Selectivity moves from space to time.** Not "pick K symbols from N and trade those, skipping the
rest even though the skipped ones had profit in them." Instead: watch all N, wait for each symbol's
own moment.

**The unit of edge is `(symbol, condition, moment)`** — not `(symbol)`, not `(strategy)`. Most
systematic design says "I have a momentum model, I apply it to a universe." This says each symbol
has moments when it becomes tradeable and the job is to be present for them. Event-driven-desk
thinking, not factor-quant thinking.

This is not a strategy family. It is how the whole system relates to the market, and it sits
underneath every family in `FEATURES.md` §4.

### 5a.1 The universe, measured 2026-08-08

| Segment | Venue | Listed | Actively TRADING |
|---|---|---|---|
| Spot | Binance | 3,680 | **1,377** (489 USDT-quoted) |
| Perpetual futures | Binance USDⓈ-M | 854 | **569** crypto + 153 TradFi |
| Dated futures | Binance | — | 4 (current + next quarter) |
| Perpetual futures | Hyperliquid | — | **232** |
| Options | Deribit | — | **not measured** — Phase 6 |

**~1,290 reachable symbols excluding options. Capture currently runs 6.** That is 0.5% of the
universe the directive depends on.

### 5a.2 The mechanism — attention scarcity, and nothing weaker

"Every symbol has profit potential" fails the *who loses when I win?* test. The declarable
mechanism is:

> Mid- and small-cap symbols get **episodic** coverage. Most of the time nobody competent is
> watching them properly. When something happens — unlock, listing, funding dislocation,
> liquidation cascade, venue-specific break — there is a window before adequate participants
> arrive. **A machine watching 1,290 symbols is structurally present for windows humans and small
> desks physically cannot cover.** Large funds ignore the region on capacity grounds; retail
> watches only what is trending. There is a genuine coverage gap in the middle tail.

**This wording goes in the Mechanism Declaration. "Potential" never does.**

### 5a.3 Why it serves the prime directive rather than competing with it

Carry (§3) is the base income. Universe-wide scanning is the **breadth engine**.

Grinold: **IR ≈ IC × √breadth**. At ~2 idiosyncratic setups per symbol per year, 1,290 symbols is
roughly 3 opportunities a day. Breadth converts a weak, rare edge into steady flow — and a steady
flow of *independent* setups is exactly what raises the portfolio green-day rate, which is exactly
the test §3 makes every strategy pass to earn allocation.

§3 rewards independence. This note says idiosyncratic setups are the independent ones. They are the
same argument.

### 5a.4 THE GATE — the breadth test, before anything is built

**Breadth means independent bets, not symbols.** Crypto cross-sectional correlation is severe;
1,290 symbols may be one BTC factor plus noise, collapsing effective breadth to ~5–15 and gutting
the proposition entirely.

- **Idiosyncratic** setups — unlock schedules, listing events, single-venue dislocations,
  symbol-specific funding — are far more independent than price returns.
- **Systematic** setups — volatility or momentum shaped — fire together in regimes, and effective
  breadth collapses toward 1.

> **The test:** define candidate setups, fire them historically across the universe, and measure
> (1) clustering of trigger times, (2) correlation of the resulting return streams.
>
> **If triggers cluster and returns correlate, the architecture is one macro bet wearing 1,290 hats
> and must be redesigned.** Cheap, answerable in days, and it gates everything downstream.

It cannot be run on 6 symbols. It is blocked on §10.3.

### 5a.5 The rules this property imposes

| Rule | Why |
|---|---|
| **Setup definitions are universal and parameter-free across symbols.** Per-symbol variation comes only from *normalisation* — z-scores or percentiles against that symbol's own history — never from fitted per-symbol parameters | One setup on 1,290 symbols is **one hypothesis with 1,290 samples**, which is statistically favourable. It becomes 1,290 hypotheses the moment per-symbol tuning is allowed. This is the single most dangerous thing that could be implemented here |
| **The acceptance threshold RISES with opportunity flow** | Triggers cluster, so the binding constraint is having capital free when the good ones fire — not finding setups. The rule is *"is this setup better than the option value of waiting for a better one?"* Dry powder is a held option with computable value, not idle cash. **The more symbols watched, the pickier each trade must be** |
| **Default action is stand aside** | The examiner posture: willing to examine every symbol, and fails almost all of them, almost always. Not a teacher maximising pass rate |
| **Exposure limits bind at factor level, not per symbol** | 40 simultaneous triggers can be one bet |
| **Minimum viable setup size, per symbol** | The tail has the least competition *and* the least capacity. Expected profit must exceed all-in cost — fees, slippage, operational, inference |
| **Two-stage funnel** | Cheap coarse screen across the whole universe; expensive evaluation only on candidates. Watching 1,290 symbols must not cost like evaluating 1,290 symbols |
| **Every scan counts in the Trial Registry** | Corrected on both axes — trial count *and* effective sample size |
| **Edge lifecycle is per symbol** | A symbol's setup can die while others live. Hazard models apply per symbol, not per strategy |

### 5a.6 The new risk this creates — manufactured setups

**A universe-wide scanner with deterministic triggers is a target.** In a thin symbol, moving the
book enough to fabricate the entry condition is cheap — far cheaper than the position that can then
be unloaded into you. The broader and thinner the tail, the cheaper it is to bait.

Required defences: **corroboration across independent data types** (trade flow *and* funding *and*
open interest — faking three at once costs much more) · **persistence requirements** (conditions
hold for a duration, never fire instantaneously) · an explicit **"why is this liquidity available
to me?"** pre-trade check · **randomised trigger latency** (bounded jitter, breaks exact timing
attacks) · a **per-symbol tradability gate evaluated before the setup gate**.

### 5a.7 The two Layer 0 consequences — one met, one not

| | Requirement | State 2026-08-08 |
|---|---|---|
| **R1** | **The broad tail must be genuinely broad.** Widening is cheap today and impossible to backfill. Do not settle for a token 20 symbols | **UNMET.** `tail_specs()` exists on both venues with passing tests, but `capture/cli.py:169` calls only `core_specs()` and there is no `--tail-symbols` argument. Running 3 symbols per venue. See §10.3 |
| **R2** | **Point-in-time universe membership** — listings, delistings, renames, contract migrations, status changes, as timestamped first-class events. Backtesting "watch every symbol" against *today's* list silently conditions on survival, and no purge/embargo scheme catches it | **MET.** `src/capture/universe_tracker.py`. Append-only, and refuses rather than writing a plausible guess |

---

## 6. Autonomy, and the two human gates

### Gates that stay human

1. **The capital dial.** Only the user sets how much money the system runs.
2. **Paper → real money.** No strategy touches capital without an explicit human promotion. This is
   the manual button of `DECISIONS.md` §4 stage 5.

Everything else is autonomous.

### Self-modification of safety controls — decided 2026-08-08

**The system may raise and lower its own risk limits, and clear its own halts, inside a ceiling the
user sets once.** The ceiling is the tail-loss cap of §0 — the same object, named twice.

This was chosen after the counter-argument was put and reaffirmed. The counter-argument is recorded
here so it is a decision and not an oversight:

> `DECISIONS.md` §7 records that Sakana's AI Scientist edited its own code to extend a timeout
> rather than fix the speed problem, and concludes: *"the evaluator must be outside the system's
> write access… our system will otherwise widen a stop, extend a lookback, or relax a threshold —
> cheapest path to a better metric."* An optimiser that can raise its own cap will eventually
> discover that raising the cap scores better than earning under it.

**Consequence, accepted:** the optimiser will drift toward the outer bound. **The ceiling the user
sets is the real risk level, not the intended one.** Set it as though the system will live there,
because it will.

**The one mitigation, which does not violate the decision:** the ceiling itself is **outside the
system's write access** — a checksummed configuration the risk gate enforces, which no component,
LLM or otherwise, can edit. The status wall shows current limits against the ceiling, so limit
creep is visible rather than silent.

### Rules that hold regardless of autonomy level

- No strategy — however trusted, however well it has performed, however highly the meta-model rates
  it — bypasses the risk gate. Both FTX/Alameda and bZx died from exactly such an exemption.
- No path from web content to capital that skips the promotion gate.
- The research loop's evaluator is deterministic, non-LLM, and outside the system's write access.

---

## 7. How the system reaches the user

All four channels, as requested:

| Channel | Content |
|---|---|
| **Live status wall** | Always current. Honest about measured vs unknown — absence of a probe renders as `NOT MEASURED`, never as green (Rule 8) |
| **Daily digest** | P&L, green-day rate, learning curve, what the research loop tried, what was promoted or retired, what is degrading |
| **Weekly review** | Deeper — allocation shifts, decay analysis, mechanism health, ledger growth |
| **Push** | The two gates, and any halt the system cannot resolve. Fallback push if a gate sits unanswered |

---

## 8. Definition of done

The project is finished when **all** of the following hold:

1. **All nine bots live** — three brains (hours→minutes, minutes→seconds, sub-second) × BULL /
   BEAR / PROFIT-TAIL, each an independent bot with its own features, architecture, training
   pipeline and validation history in the ledger.
2. **Portfolio netting layer** operating across brains.
3. **Research loop closing unattended** — hypothesis → executable code → sealed evaluation →
   promote or retire, with no human in the cycle.
4. **Live capital under the gate, earning**, at a green-day rate the wall reports honestly.
5. **The learning curve is rising** — win rate and profitable-trade count increasing with cumulative
   out-of-sample trades, alongside total P&L and green-day rate. "Rising" means the most recent
   quartile of cumulative out-of-sample trades beats the first quartile on all three measures, with
   the comparison stated on the wall rather than asserted.
6. **Every `FEATURES.md` Phase 0 minimum item satisfied.**
7. **Every Requirements Ledger row resolved** — CLAIMED by a module, or DECLINED in writing with a
   reason. No row may be left unaddressed.
8. **Universe-wide scanning live** (§5a) — the full tradeable universe watched across every segment,
   not a selected subset; the breadth test of §5a.4 passed with idiosyncratic setups shown to be
   genuinely independent; point-in-time universe membership recorded throughout.

Brain 3 (sub-second) is built even though `DECISIONS.md` §3 expects the allocator to starve it. It
is built so the answer is **measured rather than asserted**. If it cannot clear costs, that is a
finding, and it belongs on the wall.

---

## 9. Explicitly not

| Not doing | Why |
|---|---|
| NSE, equities, forex, commodities | Crypto only. Settled 2026-08-08 |
| Market making, latency arbitrage, HFT | Structurally closed at cloud latency and this capital |
| Triangular arbitrage | Never profitable after fees in a 2024 Binance study |
| Technical-indicator zoo | 7,846 rules on 100 years of Dow data — best failed out-of-sample once corrected for search size |
| Deep learning by default | LightGBM is the default. Every neural component must beat LightGBM **and** a linear baseline on our own data before it ships |
| Targeting lines of code | Institutional systems are large because of what they must handle. Size follows requirements — `DECISIONS.md` §0 |
| Multi-agent committee for research | Evidence is negative when budget-matched. Single strong agent holding full context — `DECISIONS.md` §7 |
| Chasing a Sharpe above 3 | `DECISIONS.md` §11 — suspect a bug |

---

## 10. Named work this document creates

These are consequences of the goal that do not yet exist anywhere in the plan.

### 10.1 The Requirements Ledger — **prerequisite for §8.7**

The feature set is scattered across five uninventoried sources and **nothing reconciles them**:

| Source | Size | Inventoried? |
|---|---|---|
| Research corpus (`FEATURES.md` + 6 × `IDEAS-*.md` + others) | ~950 table rows | No |
| `nse-crypto-bot-final` | 3,009 files / 864 Python / 404 markdown | Postmortem only, 6 items carried forward |
| `ai-crypto-trading-bot` | 512 files / 316 Python | Postmortem only |
| `ai-advanced-crypto-bot-final`, `ajith-ai-crypto-trading-bot`, `crypto-linix-server-bot` | ~370 files each | Postmortem only |
| `pattern-brain` | 162 files / 119 Python / 48 tests | Postmortem only |
| **`crypto-bot`** (104 MB) | unknown | **No — appears in no document** |
| **`nse-botonly`** (16 MB) | unknown | **No — appears in no document** |
| `~/video-notes/` | 7 breakdowns | No |

`prior-attempts-postmortem.md` §6 states plainly: *"The bulk of the 864 Python files is unread."*

**Deliverable:** one numbered, deduplicated list. Each row carries source, one-line description,
category, and a status of **CLAIMED** (a named module satisfies it), **PLANNED** (assigned to a
phase), or **DECLINED** (with the reason, written down).

**The property that makes it work: declining is allowed, forgetting is not.** The build reconciles
against the ledger twice — once when each module ships, once in a final sweep.

### 10.2 Spot capture does not exist — **blocks the earning core**

Measured 2026-08-08. Binance capture points at `wss://fstream.binance.com` — **USDⓈ-M futures
only**. Channels: `depth@100ms`, `trade`, `forceOrder` (withheld by the venue), plus polled
`premiumIndex`. Hyperliquid subscribes perps.

**No spot data is being captured on either venue.** But **spot-perp basis is a P1 "start here"
family**, and the prime directive rests on carry. Half of carry's data does not exist.

Universe is also three symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`.

This is a blocker for the earning core, not a nice-to-have, and it likely precedes parts of Phase 1.

### 10.3 The broad tail is built but not wired — **blocks the breadth gate**

Measured 2026-08-08. `tail_specs()` exists on both `BinanceVenue` and `HyperliquidVenue`, returning
the cheap channel set (`trade`, `forceOrder`) intended for wide coverage. It has roughly ten passing
tests. **Nothing calls it.** `src/capture/cli.py:169` calls `core_specs()` unconditionally, and the
CLI exposes no `--tail-symbols` argument.

Running now: `binance BTCUSDT,ETHUSDT,SOLUSDT` and `hyperliquid BTC,ETH,SOL`. **Six symbols out of
~1,290 reachable.**

Two consequences, both severe:

1. **The breadth test of §5a.4 cannot run.** It gates the entire universe-wide architecture and
   needs the universe.
2. **Tail history cannot be backfilled.** Every day the tail stays unwired is a day of coverage
   permanently lost — `DESIGN-NOTE-universe-wide-scanning.md` R1 is explicit that this is
   unrecoverable.

Work: a `--tail-symbols` path through the CLI and supervisor, a rate-limit and connection budget
that admits several hundred symbols across the venue's stream limits, and a storage estimate before
turning it on.

### 10.4 Capital feasibility gate

Per-strategy minimum viable capital, checked against current NAV, families auto-enabled and
disabled as the dial moves, with the reason surfaced. Does not exist in any current plan.

### 10.5 Green-day objective wired into the allocator

The allocator's fitness function is the prime directive (§3). Currently specified as discounted
Thompson sampling with no stated objective. Needs the green-day-rate-at-fixed-tail-cap criterion.

### 10.6 Learning-curve instrumentation

Win rate and profitable-trade count against cumulative out-of-sample trades, gated on total P&L and
green-day rate moving together (§1). A wall tile and a digest line. Does not exist.

### 10.7 The immutable ceiling

A checksummed risk-limit ceiling outside all component write access, enforced by the risk gate,
with current-limits-vs-ceiling displayed (§6).

---

## 11. Open, and deliberately not decided here

Carried from `ARCHITECTURE.md` §4 and `DECISIONS.md` §12, unchanged by this document:

1. **Paper-mode fill fidelity** — determines whether shadow is a separate stage or already built.
2. **Historical L2 data budget** — 1–5 TB/year per symbol-venue pair; Kaiko ≈ $28.5k/yr
   (unverified). Gates how much microstructure work is feasible.
3. **Binance matching-engine region** — sources conflict; measure with an RTT probe before
   committing infrastructure.
4. **Liquidation feed has no source** — Binance withholds `forceOrder` from this host and
   `allForceOrders` was withdrawn from the public REST API. Needs a second venue or a paid feed.
5. **Fable and data retention** — 30-day retention mandated, unavailable under ZDR. Routing
   proprietary strategy code through it is a policy decision.
6. **The numeric value of the risk ceiling** (§6) — the user has not yet set it, and per §6 it is
   the single most consequential number in the system.

---

## Provenance

Settled in interview with the user, 2026-08-08, eight questions. Grounded on a full read of
`ARCHITECTURE.md`, `DECISIONS.md`, `FEATURES.md`, `bull-bear-profit-agents-spec.md`,
`prior-attempts-postmortem.md`, the three implementation plans, the built source tree, and the live
capture archive. Test suite at time of writing: **414 passed, 1 skipped**.
