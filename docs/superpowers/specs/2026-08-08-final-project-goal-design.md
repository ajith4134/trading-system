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

## 1a. The intelligence standard — what "real intelligence" means here, and how it is checked

**Requirement, in the user's words:** the bots must have *real intelligence, real AI in the code —
not hardcoded values or instructions*. And, clarified separately: **not only learning, but real
intelligence.** Those are two different things and this section keeps them apart.

**Sources.** `~/research/IDEAS-INTELLIGENCE.md` (the corpus already answers much of this),
plus four reports commissioned 2026-08-08: `learned-vs-hardcoded-boundary.md`,
`reasoning-vs-learning.md`, `what-makes-code-deep.md`, `strategies-as-evolved-programs.md`.
Each carries its own `UNVERIFIED` appendix. Re-read those before overriding anything below.

### 1a.0 The honest boundary — stated first, so the standard does not become the overclaim it exists to prevent

`IDEAS-INTELLIGENCE.md` §0 opens with it:

> No architecture available in 2026 produces understanding. LLMs do not know things; they produce
> text conditioned on text. Anything promising "real intelligence" from a model swap is selling
> something.

The empirical answer to that hope is already in the corpus: **Alpha Arena, six frontier models given
$10k each on Hyperliquid perps. In 17 days four of six lost 30–63%.** Causes were mundane —
over-trading into fees, rigid directional bias, no stop discipline.

**But the distance between a script and an adapting, self-aware system is real, and it is
crossable.** What crosses it is not a smarter model. It is beliefs that carry provenance and expire;
calibrated knowledge of its own competence; learning from its own history as data; self-directed
acquisition of what it lacks; and modelling itself as a participant rather than a ghost. All five are
buildable with unglamorous engineering, and are almost universally skipped for exactly that reason.

**The failure mode this standard is designed against is not stupidity. It is confident staleness** —
a system that learned something true, never noticed it stopped being true, and keeps betting on it.

### 1a.1 Three axes, judged separately

A module can pass one and fail the others. All three are review gates, not aspirations.

| Axis | The question | Failure looks like |
|---|---|---|
| **Learning** | Where did this number come from? | A person's guess wearing a variable name |
| **Reasoning** | Does it know *why*, and does it know when it doesn't? | Fluent causal-sounding language over pattern completion |
| **Depth** | Is the code itself deep or shallow? | Elaborate interfaces hiding little |

### 1a.2 The learning axis — tests

From `learned-vs-hardcoded-boundary.md` §5. Apply to any module named, documented or described as
learning, adaptive, AI, or intelligent.

| # | Test | Disqualifier |
|---|---|---|
| L1 | **Provenance-of-constants audit** | Every numeric parameter in the decision path traces to a `fit()`/`train()` call with a logged loss and dataset reference, or to an explicitly disclosed design choice. "Someone typed it" is hardcoded, whatever the variable is named |
| L2 | **Update-in-the-runtime-loop** | Does the live decision loop itself update parameters from post-deployment data? If the only path that changes parameters is a script a human invokes, that is scheduled retraining — a legitimate but *different* claim, and it must be labelled as such |
| L3 | **Parameter-randomisation** | Swap trained values for random ones and rerun on held-out data. Materially unchanged output means the parameters are decorative |
| L4 | **Ablation** | Freeze or remove the component, replace with its long-run average, rerun. Statistically indistinguishable means it contributes nothing measurable |
| L5 | **Freezing / inner-loop** | For anything claiming fast adaptation: freeze the adaptable part at deployment. If performance barely moves, that is feature reuse, not rapid learning |
| L6 | **Out-of-regime stress** | Edge must survive a *different* regime, not a held-out slice of the same one. Nov 20 2024 is a known boundary in crypto structure |
| L7 | **Trial-count accounting** | Every search, retune and backtest counted, with DSR or PBO applied. Uncounted retuning makes the reported edge unverified by definition. **The count keys on candidates *evaluated*, never on candidates *retained*** — see the archive clause below |
| L8 | **Objective-vs-intent audit** | Stress the reward against degenerate inputs and check whether the learner finds the exploit. ~60 catalogued real cases of specification gaming exist |
| L9 | **Catastrophic-forgetting check** | Interleave a regression suite of past scenarios into every online update; alert when old competence degrades as new data arrives |
| L10 | **Documentation reconciliation** | Every place the system is called AI, adaptive or self-improving must point at an artifact satisfying L1–L9 |

> **If only two can be run: L1 + L4.** Show where the values came from, and show that removing the
> component changes measured behaviour. Fail either and it is a parameterised script wearing the
> costume of learning.

This is not an abstract standard. **The SEC sanctioned Delphia and Global Predictions in March 2024
for precisely this shape** — AI/ML claims with no underlying training artifact to point at.

#### The archive clause — a diversity archive buys no discount on multiple testing

From `strategies-as-evolved-programs.md`, and it is the single most important constraint that report
imposes.

Quality-diversity search — maintaining an archive of strategies that are good **and behaviourally
different**, MAP-Elites style — is the right shape for this system. `IDEAS-INTELLIGENCE.md` §9 already
rates it ★ HIGH, because correlation is what kills portfolios and a stable of decorrelated mediocre
strategies beats one optimised strategy that dies in one regime.

**But a diverse archive is a *larger* search, so the correction gets harsher, not gentler.**

> **The multiple-testing correction keys on the total number of candidates evaluated to build the
> archive — never on the number retained in it.**

The report's finding on why this matters: **five of the six finance-specific evolved-alpha papers
surveyed omit exactly that number.** It is the one figure needed before any of their out-of-sample
claims can be trusted, and it is the one figure missing.

The Trial Registry must therefore be **structurally impossible to bypass** during archive
construction, not merely conventionally used — which is already its contract in `ARCHITECTURE.md`
Layer 2.

### 1a.3 The reasoning axis — tests

From `reasoning-vs-learning.md` §6. **None of these requires trusting the component's self-report.**
That is the point — every one is checkable from outside, on data you control.

| # | Test | What it catches |
|---|---|---|
| R1 | **Falsifiable side-prediction** | Ask for an auxiliary consequence of the explanation that has not been checked, then check it. No checkable side-prediction, or one that merely restates the claim, means pattern-matching in causal language |
| R2 | **DoWhy refutation battery** | Placebo-treatment, random-common-cause, data-subset refuters. Passing is necessary, not sufficient — **failing any one is a hard mechanical disqualifier** |
| R3 | **Evidence-order permutation** | Same facts, different order. A component reasoning over structure is invariant. Measured LLM-as-judge position bias: Claude-v1 flipped 76% of the time, GPT-4 30% — on a task with no principled order dependence |
| R4 | **Abstention with teeth** | "I don't know" must be backed by a coverage guarantee (conformal `q̂`, or an SGR bound), not a verbalised hedge. Track realised accuracy on non-abstained predictions against the promised bound |
| R5 | **Counterfactual on held-out structure** | Synthetic environment with a known confounder; train on observational data only; ask for an intervention never shown. The only test here that separates Pearl's rung 1 from rung 2 |
| R6 | **Self-correction without ground truth** | Let it self-correct with no external feedback and measure before/after. Real systems *degrade* — one documented case 75.8% → 38.1%. If accuracy does not improve, the correction step is theatre |
| R7 | **Multi-voice vs single-voice + resampling** | Benchmark any debate or committee design against one agent with self-consistency at matched compute. An ICML 2024 replication found debate does not reliably beat it, and an assigned adversarial persona made it worse |
| R8 | **Model-trust boundary** | Does planning depth *shrink as the world model's measured error grows*, or is lookahead a fixed hyperparameter? MBPO's own bound implies planning zero steps into an uncalibrated model; a documented 45% reward collapse (176→98) came from a model whose training loss never moved |
| R9 | **Mechanism-swap / label invariance** | Rename `funding_rate` to `widget_7`, preserving its statistical role. A component reasoning over structure answers identically; one leaning on lexical association does not |

### 1a.4 The depth axis — tests

From `what-makes-code-deep.md`. Depth is a defined concept, not a compliment. Ousterhout:
*"It is more important for a module to have a simple interface than a simple implementation."*

| # | Test | Threshold |
|---|---|---|
| D1 | **Interface-to-implementation ratio** | Public symbols exposed, against lines of implementation behind them. The defining criterion, and the only one applicable from a module's own source without tracing call sites |
| D2 | **Concepts a caller must hold** | A caller needing to understand the internals to use it correctly means the abstraction leaks |
| D3 | **Error paths per happy path** | Institutional code is mostly non-happy-path. A module with one error branch per twenty happy-path lines has not been written for production |
| D4 | **No shallow-module proliferation** | Many small classes each exposing nearly as much interface as implementation is the documented failure pattern, and the one generated code tends to produce |

**Honest gap, recorded rather than papered over:** the depth report found **no published study
measuring module depth in LLM-generated code directly.** Adjacent evidence exists on duplication and
defect density; the specific claim does not. D1–D4 are therefore applied as a review discipline
grounded in design literature, not as an empirically validated predictor.

### 1a.5 The master test

From `IDEAS-INTELLIGENCE.md`, and it governs everything above:

> ### Does it change what the system does when it is wrong?

Features that only improve behaviour when the system is already right are decoration. Every
capability admitted under this standard must change behaviour *under error* — by refusing to act, by
expiring a belief, by catching a contradiction, by rolling back, or by noticing the world moved.

### 1a.6 How this binds

- **No module ships without an axis verdict.** Learning / reasoning / depth, each pass or fail with
  the evidence named. A module may legitimately be *not applicable* on an axis — a Parquet writer
  does not reason — but that must be stated, not left blank.
- **The status wall carries the verdicts** (Rule 8). A module claimed intelligent with no passing
  test renders as `NOT MEASURED`, never as green.
- **Ledger rows inherit it.** Any Requirements Ledger row describing an intelligent capability
  carries its axis tests; without them the row cannot move to CLAIMED.
- **The ten to build first** are already ordered in `IDEAS-INTELLIGENCE.md` by leverage per unit of
  effort: belief records with provenance and half-life · read/verified/observed epistemic classes
  where **only observed may size a position** · verification-before-ingestion · abstention as a real
  action with its P&L measured · calibration scoring · "who loses when I win?" as a required
  declaration · meta-analysis over the Trial Registry · own-footprint attribution · cost of operation
  inside the objective · property-based invariants across backtest, shadow and live.

That list is deliberately boring, and `IDEAS-INTELLIGENCE.md` says why: **boring is what compounds.**

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

## 3a. INTRADAY, on all three segments — user ruling, 2026-08-16

> **"our entire trading crypto bot is intraday on all segments spot, futures, options."**

Recorded verbatim because it changes §3's ruling, and §3 was itself settled with the
user. Where the two conflict, this section is the later word.

### What it changes

**Carry is no longer the earnings core by right.** §3 makes funding carry and
spot-perp basis the core *because they are latency-immune* — that is the stated
reason, in those words. An intraday mandate removes the property the ruling rests
on. Carry does not stop being a family; it stops being the thing "constantly"
rests on, and it now has to win capital from the allocator on the same terms as
everything else.

**The three segments are named and unequal today:**

| Segment | Captured | State |
|---|---|---|
| **Spot** | `binance-spot`, `coinbase` | live, byte-exact |
| **Futures — perpetual** | `binance`, `bybit`, `hyperliquid` | live, plus funding |
| **Futures — dated** | `bybit` | live, 48 contracts, `features.term_structure` reads them |
| **Options** | **nothing** | **no venue is captured at all** |

Options is not a phase behind — it is a segment with **zero data**. `FEATURES.md`
§5b is eleven unbuilt rows at P3, and its own header says the layer is *"required
before any options position"*. Deribit carries most crypto options volume, and
§5b already records that trading it means accepting single-venue concentration
against the per-venue exposure cap. Nothing about that is decided by this ruling;
what the ruling does is move it from "later" to "named, missing, and blocking."

**The data path becomes the binding constraint, and today it fails the mandate.**
Measured 2026-08-16: raw capture is live to the second on every venue, and the
newest bar the clock-gated reader serves was **69 minutes old**. Bars are built
only for hours that have closed, in passes roughly every 35 minutes. For carry —
8-hourly settlement — that lag is immaterial, which is why it was never a
problem. **For an intraday system it is disqualifying.** No intraday strategy can
be honestly judged on a feed that stale, and none should be.

**Costs bind harder than anything else.** A round trip is ~24 bps on this venue
set. Carry earns across an 8-hour settlement; an intraday trade must clear the
same 24 bps in minutes to hours. `cost.round_trip_cost.is_signal_viable` was
already the gate every signal passes — under an intraday mandate it becomes the
constraint that decides whether the mandate is achievable at all, and the honest
answer for most instruments is expected to be no.

### What it does NOT change

**The brain structure already spans intraday.** §8's definition of done is three
brains — *hours→minutes, minutes→seconds, sub-second* — so an intraday mandate is
consistent with the architecture rather than a replacement for it. What changes is
which brain carries the weight, not that a brain exists for it.

**The prime directive is untouched.** §0 stands: maximise the fraction of green
days subject to a tail cap the system cannot raise. Intraday is a statement about
holding period, not about objective, and the tail cap of §6 remains the user's
alone.

**`DECISIONS.md` §3's caveat still stands and is now more load-bearing, not
less.** Sub-second was kept as a branch the system *measures and prunes* rather
than one either party asserts about, on the record that HFT from a cloud VM is
not realistically winnable — market making ~75% captured by HFT firms, triangular
arbitrage on Binance found never profitable after fees. An intraday mandate does
not repeal that evidence. It raises the stakes on measuring it honestly.

### What this ruling requires, in order

1. ~~**Fix the data lag**~~ — **DONE 2026-08-16**, `store.live_bars` +
   `scripts/live_bars_supervisor.sh`, commit `3569c80`. Bars are now built for the
   hour still being written, from the readable prefix that `capture.raw_writer`
   guarantees by closing a zstd frame and fsyncing every 30 seconds.

   Measured through the clock-gated reader, before and after: **69 minutes → 1.8
   minutes** on all four venues, and the forward paper engine's newest event went
   to **103 seconds** old. Two tiers, sized by measurement at the worst point of an
   hour: core 12 symbols ~5s at a 60s interval, the whole captured universe of
   2,303 symbols 95s at 300s.

   **The provisional bars are not an approximation.** Only minutes that closed
   before that symbol's newest readable trade are published, and stamps are left
   data-derived so the complete build supersedes correctly. Of **63,147** hour-15
   bars published provisionally, **zero** differed from the complete build on
   close, high, low, open or volume.

   Two gaps this did NOT close, and neither is a scheduling problem:
   **options** still has no venue to build from (item 2), and **hyperliquid**
   subscribes `trades_*` for only BTC/ETH/SOL, so 174 of its 177 symbols in the
   store are a week old — a capture-breadth gap that belongs with item 2.
2. **Capture an options venue** — the segment has no data, and no options feature
   can be built or probed against nothing.
3. **Re-run the cost gate as a feasibility question**, per segment and per
   holding period, and write down what survives. If most of the universe cannot
   clear 24 bps intraday, that is the finding and it belongs on the board.
4. **Re-rank `FEATURES.md` §4.** "Start here" currently marks the two
   latency-immune families, and that marking was reasoned from the property this
   ruling removes.

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

### 5a.6b The second new attack surface — the price feed is attacker-authored text

Found by the Requirements Ledger, 2026-08-08, as row **OGR-057** — sourced to `IDEAS-STRATEGIC.md` §9
and **named in none of the six core design documents.**

The plan quarantines *web* content: `FEATURES.md` §11 and `DECISIONS.md` §7 specify the Dual-LLM
pattern so the privileged model never sees raw untrusted pages. **Market data was never classified as
an untrusted-content channel.** It is one:

- **Token names and symbols** are chosen by whoever mints the token.
- **On-chain memos and NFT metadata** are free-form attacker-controlled strings.
- **Listing announcements** are prose, ingested as prose.

§5a multiplies this by ~1,290, and concentrates it precisely in the thin tail where minting a token
costs almost nothing — the same region §5a.6 already identifies as cheap to bait.

Four distinct threats, none currently addressed:

| Threat | Shape |
|---|---|
| **Direct injection via the feed** | Attacker-authored strings reaching any reasoning context as text |
| **Slow context poisoning** | Adversarial content accumulating in long-lived memory, where a single day's ingest looks harmless |
| **Semantic denial-of-service** | Content crafted to consume the autonomous researcher's curiosity budget (§5a.5) on nothing |
| **Content targeting models, not people** | Text written to move a model's output rather than to persuade a human reader |

**The rule this imposes: structured extraction only, never free text from a market feed into a
reasoning context.** A symbol is an identifier to match against a universe record, never a string to
interpret. Anything narrative — announcements, memos, metadata — goes through the same quarantine as
a web page, and enters as a hypothesis, never a decision.

Two of the mitigations already exist elsewhere in the plan and should be pointed at this threat
explicitly: **belief provenance with retraction propagation** (`IDEAS-INTELLIGENCE.md` §1, ledger
OGR-065) so a falsified source re-scores everything derived from it, and the **capped curiosity
budget** (§5a.5) so semantic DoS has a ceiling. **The threat-model architecture itself is unbuilt and
unplanned.**

### 5a.7 The two Layer 0 consequences — one met, one not

| | Requirement | State 2026-08-08 |
|---|---|---|
| **R1** | **The broad tail must be genuinely broad.** Widening is cheap today and impossible to backfill. Do not settle for a token 20 symbols | **MET 2026-08-08.** `--tail-symbols ALL` resolves the universe from the venue at startup and shards it by measured URL bytes. Running in production: **569 Binance perpetuals, 177 Hyperliquid**, against 3 per venue before |
| **R2** | **Point-in-time universe membership** — listings, delistings, renames, contract migrations, status changes, as timestamped first-class events. Backtesting "watch every symbol" against *today's* list silently conditions on survival, and no purge/embargo scheme catches it | **MET 2026-08-08 — and it was not before.** `src/capture/universe_tracker.py` existed from Layer 0 with passing tests and **no caller outside them**, so nothing was ever recorded. `--tail-symbols ALL` now writes a snapshot before the first frame of every run. This row previously read MET on the strength of the module existing, which is the same reads-as-built error the ledger exists to catch |

---

## 6. Autonomy, and the two human gates

### Gates that stay human

1. **The capital dial.** Only the user sets how much money the system runs.
2. **Paper → real money.** No strategy touches capital without an explicit human promotion. This is
   the manual button of `DECISIONS.md` §4 stage 5.

Everything else is autonomous.

### Self-modification of safety controls — decided 2026-08-08, amended the same day

**The system may raise and lower its own risk limits, and clear its own halts, inside a ceiling the
user sets once.** The ceiling is the tail-loss cap of §0 — the same object, named twice.

**Amendment: every self-modification passes through a bounded canary before it takes effect.** The
system still changes itself without asking. It simply cannot change itself faster than it can
measure whether the change helped.

#### Why the amendment exists — first-party evidence, not an argument from analogy

The original ruling was made against the Sakana precedent in `DECISIONS.md` §7, and reaffirmed.
Then the repo mining found the same experiment already run **on this project's own money**.

`ai-crypto-trading-bot` contains two internal audits — `PROFESSOR_AUDIT.md` and
`BLUEPRINT_COMPLIANCE_AUDIT.md` — recording, with code citations:

| Finding | Detail |
|---|---|
| **Net −$837 over 10,240 live trades** | Empirically measured. Not a backtest |
| **Fake attribution** | The feature-governance controller credited the same bot-wide win/loss to *every* active feature. It **deactivated all 36 features simultaneously, twice, in production** |
| **~775 lines of dead code** | In the exit-management monolith |
| **Three concurrent self-modification loops** | Daily code-rewriting, prompt evolution, and a genetic algorithm — all running against the system being debugged |

**The market did not beat it. The self-modification loops corrupted the measurement they fed on.**
With attribution wrong and three rewrite loops competing, the system could not tell what was
working, and responded by switching everything off. Over 10,240 trades it did not learn its way to
profit.

That failure mode is fatal to this design specifically, because §0 rests on *"the failures are the
asset."* If attribution is wrong, the ledger is wrong, the Deflated Sharpe is wrong, and the
learning curve of §1 measures noise.

#### The canary contract

Adapted from `signals/scibrain/` in `nse-crypto-bot-final`, which already implements this shape —
`changespec.py`, `evaluator.py`, `rigor.py`, `validator.py`, `change_report.py`, `canary.py`.

Every self-initiated change to a risk limit, threshold, or strategy parameter must:

1. Be expressed as a **typed `ChangeSpec`** — bounded, validated, no free-form code path.
2. Be evaluated against a **matched cohort**, not against its own post-hoc performance.
3. Pass a **statistical rigor gate** before promotion, with the trial counted in the Trial Registry.
4. Run as a **canary** on a bounded slice before applying broadly.
5. **Auto-roll-back** on failure, with a change report written to the ledger either way.

**Only one self-modification loop may be active at any moment.** Three concurrent loops was the
documented root cause, and this rule is not negotiable by the system.

**Per-feature attribution must be correct before any loop runs at all.** A loop fed by bot-wide
P&L credited to every feature is not learning; it is amplifying noise with authority. See §10.8.

#### What still holds from the original ruling

**Consequence, accepted:** the optimiser will drift toward the outer bound. **The ceiling the user
sets is the real risk level, not the intended one.** Set it as though the system will live there,
because it will.

**The ceiling is outside the system's write access** — a checksummed configuration the risk gate
enforces, which no component, LLM or otherwise, can edit. The status wall shows current limits
against the ceiling, so limit creep is visible rather than silent.

### The kill switch is weaker than the design calls for — measured 2026-08-08

`ARCHITECTURE.md` Layer 3 specifies the kill switch as a separate OS process that kills the bot
**and drops outbound network at the firewall**, and is explicit about why the firewall carries the
weight: *"No major exchange appears to expose programmatic self-revocation of your own key. Do not
design a kill switch assuming it — the firewall is the reliable mechanism."*

**On this host, the reliable mechanism does not exist.** Measured, not assumed:

| Probe | Result |
|---|---|
| `sudo -n true` | denied outright — not password-prompted, refused |
| `iptables` on PATH | absent |
| `nft` on PATH | absent |

`src/ops/watchdog.py` therefore ships three mechanisms and **records which ones fired**, rather than
reporting a kill it did not perform:

1. **A persistent kill file** (`KILLED.json`) — every trading path reads it and refuses. Written
   first, before anything that can fail, and it survives a restart on purpose: a kill the supervisor
   undoes on its next loop looks like it fired and did not. The first breach's reason is never
   overwritten by a later one, because whatever fires last is usually the symptom.
2. **Credential denial** — the age identity is moved aside, so `secret_store` cannot decrypt and no
   signed request can be constructed. It is *set aside, not destroyed*: a kill switch that loses the
   only copy of a key turns a drawdown into a permanent outage.
3. **SIGKILL** of the watched PIDs — not SIGTERM, because a wedged process is the case the watchdog
   exists for and may never service a handler.

**Credential denial is genuinely weaker than a packet filter, and the gap is recorded rather than
glossed:** an already-open socket survives it, and unauthenticated endpoints stay reachable. Every
trip writes `network_dropped: false` with the measured reason, so nobody later reads a halt and
assumes egress was cut. The capability is re-probed on each trip rather than held as a constant —
the answer changes the moment a packet filter and sudo exist on the host.

**Consequence for §6's promotion gate:** the firewall drop is an open item (§11), and until it is
closed the paper → real-money crossing carries a kill switch that can stop new orders but cannot
guarantee stopping an in-flight one. This is a fact about the host, not a design choice, and it
belongs in the decision the user makes at that gate.

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
| **`crypto-bot`** (104 MB) | **not a prior bot** — a publish mirror of the *current* work | Mined; see correction below |
| **`nse-botonly`** (16 MB) | 587 Python files, near-zero stubs | **No — appeared in no document** |
| `~/video-notes/` | 7 breakdowns | No |

`prior-attempts-postmortem.md` §6 states plainly: *"The bulk of the 864 Python files is unread."*

**Correction, 2026-08-08.** `ajith4134/crypto-bot` is not a seventh trading bot. Its root tree is
`README.md · claude-config · research · trading-system · video-notes · push-to-github.sh`, matching
`~/crypto-bot-publish` exactly — it is the staging mirror behind Rule 9's push workflow, carrying
the *current* project plus a `layer0-raw-capture` branch. Roughly 554 of the 723 rows mined from it
are therefore duplicates of `research-corpus.md` and of the built Layer 0, and collapse at merge.

**`nse-botonly` is the real find.** 587 Python files, near-zero stubs, a substantially complete and
tested system: brokers, execution, risk, backtesting and validation, plus an LLM, governance and
self-healing stack. Its one genuine gap, flagged in its own `BACKLOG B42`, is a missing top-level
strategy router — the "main AI brain." It is the most complete system in the account, it is NSE, and
per §5 it is a **feature donor**: the execution, risk and validation machinery ports; the market does
not.

**Deliverable:** one numbered, deduplicated list. Each row carries source, one-line description,
category, and a status of **CLAIMED** (a named module satisfies it), **PLANNED** (assigned to a
phase), or **DECLINED** (with the reason, written down).

**The property that makes it work: declining is allowed, forgetting is not.** The build reconciles
against the ledger twice — once when each module ships, once in a final sweep.

### 10.2 ~~Spot capture does not exist~~ — **DONE 2026-08-08**

Measured 2026-08-08. Binance capture points at `wss://fstream.binance.com` — **USDⓈ-M futures
only**. Channels: `depth@100ms`, `trade`, `forceOrder` (withheld by the venue), plus polled
`premiumIndex`. Hyperliquid subscribes perps.

**Resolved.** `binance-spot` is a separate venue — sharing the futures venue's name would have filed
spot BTCUSDT and perpetual BTCUSDT as one instrument, and the basis is precisely the difference
between them. Live in production across all 1,377 spot symbols, at 26 MB/hour.

Building it exposed a coupling worth recording: the recorder chose its depth gap tracker with
`venue.name == "binance"`, correct only while exactly one Binance venue existed. Spot would have
fallen through to plain staleness tracking, and every dropped depth update would have gone
unreported. `BinanceDepthTracker` already handled both chains — futures on `pu == prev.u`, spot on
`U == prev.u + 1` — and nothing could reach its spot branch.

**Capture now stands at 2,123 symbols across three venues**, against six this morning: 569 Binance
perpetuals, 1,377 Binance spot, 177 Hyperliquid. Combined ~3.4 GB/day against 89 GB free.

### 10.3 ~~The broad tail is built but not wired~~ — **DONE 2026-08-08**

Measured 2026-08-08. `tail_specs()` exists on both `BinanceVenue` and `HyperliquidVenue`, returning
the cheap channel set (`trade`, `forceOrder`) intended for wide coverage. It has roughly ten passing
tests. **Nothing calls it.** `src/capture/cli.py:169` calls `core_specs()` unconditionally, and the
CLI exposes no `--tail-symbols` argument.

**Resolved.** `--tail-symbols ALL`, sharded by measured URL bytes (fstream refuses past ~16.3KB with
HTTP 414, and rejects SUBSCRIBE over the socket, so the URL is the only path and it must shard). The
core keeps its own connection so a tail disconnect cannot take depth with it. Live in production on
both venues. Measured cost: **111 MB/hour, 2.7 GB/day.**

Two defects surfaced by the first live tail run, both silent and both fixed: Binance lists
perpetuals with CJK symbols and every one was being filed as `trade_unknown` — one shared bucket,
0 dropped, 0 malformed; and a symbol of exactly `..` matched the "safe" path-token regex because `.`
is in its character class. **The breadth gate of §5a.4 is now unblocked.**

Two consequences, both severe:

1. **The breadth test of §5a.4 cannot run.** It gates the entire universe-wide architecture and
   needs the universe.
2. **Tail history cannot be backfilled.** Every day the tail stays unwired is a day of coverage
   permanently lost — `DESIGN-NOTE-universe-wide-scanning.md` R1 is explicit that this is
   unrecoverable.

Work: a `--tail-symbols` path through the CLI and supervisor, a rate-limit and connection budget
that admits several hundred symbols across the venue's stream limits, and a storage estimate before
turning it on.

### 10.3b Capture continuity — **blocks the attention-scarcity mechanism itself**

Measured 2026-08-08. **Capture has covered 20.0% of wall-clock time since it began.**

| | |
|---|---|
| First capture | 2026-08-02 15:42 UTC |
| Clean stop | 2026-08-03 18:12:27 UTC — `supervisor_stopped` logged, graceful SIGTERM, no torn hour |
| Restart | 2026-08-08 04:05:54 UTC, 56 seconds after boot |
| Outage | **4 days 9 hours 53 minutes** |
| Days with any data | 3 — Aug 2, 3, 8 |

**Not a fault.** The instance is `preemptible: FALSE`, `automaticRestart: TRUE`. The VM was
deliberately stopped, and the user confirms this is **cost control, done on purpose**.

**The boot-recovery claim is verified.** `DECISIONS.md` §13.3 says capture restarts on boot; the
startup script fired correctly after a real four-day outage. That claim is now proven against a
genuine failure rather than a test.

**Why this is a goal problem and not an ops footnote.** §5a's declared mechanism is *attention
scarcity* — being structurally present for windows others cannot cover. **A machine that is off 80%
of the time has no such advantage; it has a worse version of the gap it claims to exploit.** And
none of it is recoverable: Layer 0's contract is never re-pull history.

**Measured sizing, 2026-08-08.** Capture uses **3.1% of one vCPU and 117 MB RSS** for six symbols,
on an `e2-custom-12-30720` — 12 vCPU, 30 GB. Data rate 31.2 MB/hour for six symbols, of which
`depth@100ms` is roughly 95%; the broad tail uses only `trade` and `forceOrder`.

**Then measured properly, on the full tail.** A 30-second run over all 569 Binance perpetuals plus a
three-symbol core: **111 MB/hour, 2.7 GB/day, roughly 1 TB/year.** Against 89 GB free that is about
33 days of runway before storage needs a decision.

*The earlier extrapolation in this section said ~15 GB/day and ~5.5 TB/year — wrong by 5×, from a
six-symbol sample dominated by depth. Left visible rather than deleted: it is a small worked example
of why this document prefers a measurement to an estimate, and the estimate was mine.*

**Consequence:** the thing that must never stop needs a fraction of one core, and it is being stopped
to avoid paying for twelve. Capture belongs on its own minimal always-on instance, with this machine
free to be stopped whenever it is not building. Dollar figures deliberately not quoted — the Cloud
Billing API rejects unauthenticated callers. **Price it before it sizes anything.**

*Correction, later the same day: an earlier version of this paragraph said gcloud was authenticated
to a different project than the instance. That was wrong — both are `project-e760fdd7-f8da-46a5-8c4`,
and the `gcloud compute instances describe` failure that suggested otherwise was the zone or the
instance name. Left visible rather than deleted, on the same principle as the storage estimate above.*

**The archive now has a backup**, which it did not this morning: `scripts/offload_supervisor.sh`
copies every completed hour to `gs://capture-raw-data4134` on a 30-minute loop, wired into the boot
script. Verified 2026-08-08: **0 completed files missing from the bucket.** Live hours are excluded
per file via `RawWriter`'s own `.writing` marker, because copying a zstd frame mid-write produces a
truncated archive that reads as complete.

### 10.3c Fees dominate by ~1,300×, not 5-10× — measured 2026-08-08

`ARCHITECTURE.md` Layer 1 argues that **fees dominate breakeven by roughly 5-10× over slippage**, and
the whole Cost Engine exists on the strength of that claim. It is now measured on this project's own
captured books, at 10:50 UTC on 2026-08-08:

| Venue | Touch | Half-spread |
|---|---|---|
| Binance perp BTCUSDT | 64980.60 / 64980.70 | **0.0077 bps** |
| Binance spot BTCUSDT | 65010.00 / 65010.01 | **0.0008 bps** |

Against a **10 bps** taker round-trip fee, that is fees dominating by roughly **1,300×** on the perp
and **12,000×** on spot — two to three orders of magnitude beyond the figure the argument was built
on.

**The direction of the claim is right and its magnitude was understated.** The practical consequence
sharpens rather than changes: for a strategy trading liquid BTC at this account's clip sizes, spread
is very nearly free and **the fee schedule is essentially the entire cost model.** Venue and fee-tier
selection outrank every execution refinement by a wider margin than the design document assumed, and
effort spent modelling slippage before fees are *verified* is effort spent on the smaller term by a
factor of a thousand.

Two caveats, both real. This is BTC, the most liquid instrument on either venue — the mid-tail
symbols the §5a breadth engine is built for will be far wider, and this measurement says nothing
about them. And impact behaves entirely differently at size: a $500k clip against 20 levels of
Binance **spot** was refused outright, because the visible book could only fill $196,888. Spread is
free; depth is not.

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

### 10.8 Correct per-feature attribution — **prerequisite for every learning loop**

The documented cause of the −$837 / 10,240-trade result was not a bad model. It was that the
feature-governance controller credited **the same bot-wide win/loss to every active feature**, so
every feature's score moved together and carried no information. Acting on it deactivated all 36
features at once, twice, in production.

Nothing in the current plan specifies how credit is assigned to a feature, a signal, or a strategy.
Until that is correct and tested, **no self-modification loop, no meta-model over the ledger, and no
learning curve (§1) means anything** — they would all be reading the same corrupted signal.

Required: attribution that can distinguish one feature's contribution from another's, validated
against a synthetic case where the true contributions are known before it is trusted on real fills.

### 10.9 The bounded canary

Implement the §6 canary contract: typed `ChangeSpec`, matched-cohort evaluator, statistical rigor
gate, canary trial on a bounded slice, auto-rollback, change report to the ledger, and the
one-loop-at-a-time constraint. `signals/scibrain/` in `nse-crypto-bot-final` is the reference
implementation to adapt, not to copy blind — it has never been validated against a gate.

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
7. **The firewall kill mechanism has no host** — added 2026-08-08 from measurement, not carried.
   `ARCHITECTURE.md` names dropping egress at the firewall as the *reliable* kill mechanism, and this
   host has neither sudo nor a packet filter (§6). Credential denial ships as the substitute and is
   weaker. Closing this needs a decision the user owns: grant passwordless sudo for a single
   pre-authorised `iptables` rule, run the watchdog on a second host that can cut the instance's
   egress at the VPC firewall, or accept the weaker mechanism and record that at the promotion gate.

---

## Provenance

Settled in interview with the user, 2026-08-08, eight questions. Grounded on a full read of
`ARCHITECTURE.md`, `DECISIONS.md`, `FEATURES.md`, `bull-bear-profit-agents-spec.md`,
`prior-attempts-postmortem.md`, the three implementation plans, the built source tree, and the live
capture archive. Test suite at time of writing: **414 passed, 1 skipped**.
