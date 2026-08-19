# RECONCILIATION — what was decided, what exists, what was skipped

**Measured 2026-08-19 18:0x UTC.** Every number here came from running something, not from
reading a plan. Where a figure could not be measured it says so.

Ordered by RL-045. All four segment bots were switched off for the duration so nothing moved
underneath the audit.

---

## 0. The one number

**90 of 1,491 requirements-ledger rows are satisfied by a named module. That is 6%.**

The ledger is the anti-forgetting mechanism: every feature, constraint and idea found across the
research corpus, eight prior repositories and seven video breakdowns, deduplicated into one list.

| Status | Count | Meaning |
|---|---:|---|
| **CLAIMED** | 90 | a named module in this project satisfies it |
| **PLANNED** | 497 | named in a design document, not built |
| **PRIOR-ART** | 599 | working code in a prior repository, in no current plan |
| **UNRESOLVED** | 108 | no home in any plan and no implementation — the gaps |
| **DECLINED** | 61 | refused, with the reason written down |
| *unclassified* | 136 | rows in tables carrying no status column |
| **TOTAL** | **1491** | across 8 slice files, merged from 3,214 raw rows |

On 2026-08-08 this read **29 CLAIMED**. So 61 rows have been built in eleven days — real
progress, and it changes the 6% to a trajectory rather than a verdict. But **599 rows are
working code in repositories no plan references.** For every capability this project satisfies,
roughly seven exist already, written, somewhere the plan never looks.

---

## 1. What the plan says, and what the probes measure

The AJIT MASTER PLAN holds **71 rows across 10 slices**. Every row names a probe; there are **55
distinct probes** and **no row without one**. That part of the governance works.

| slice | rows | note |
|---|---:|---|
| slice-0 (foundation, governance) | 18 | |
| bot-framework | 12 | |
| spot-bot | 2 | |
| perp-bot | 16 | 3 deliberately BLOCKED behind a performance verdict |
| capital | 7 | added today |
| champion-gate | 3 | added today |
| dated-bot | 2 | bot switched off (RL-039) |
| options-bot | 2 | bot switched off (RL-039) |
| learned-brains | 9 | |
| slice-5 (allocator, portfolio risk, brains 2-3) | **0** | **named, never written** |

### Probe results, run today

**30 ok · 8 partial · 8 degraded · 7 not measured · 2 failing**

Seven of the "not measured" are a direct consequence of switching the bots off for this audit —
they measure a live loop that is deliberately not running. That is the OFF switch working, not a
finding.

**The two failing probes:**

- `probe_only_perp_and_spot_running` — perp and spot are governed but not running. Correct: I
  switched them off. It will clear when they restart.
- `probe_margin_per_trade_within_band` — reports the **lifetime** journal, so it still carries
  3,340 pre-capital-slice opens with a 6.1e9 spread. **This is a real defect in my own work:**
  the capital design document says *"the journal must mark the cutover"* and I did not implement
  that, so the probe cannot tell the old regime from the new one. It is measuring a fixed
  problem and reporting it as broken.

### What DID verify today

- `probe_capital_declaration_is_enforced` → **ok**, enforced on perp
- `probe_excursions_journalled_on_close` → **ok**, 26 closed trades carry both excursions and
  hold the ordering invariant
- `probe_leverage_declared_and_bounded` → **ok**, 24 opens within ceiling with a reachable stop
- `probe_axis_verdicts_present` → **ok**, 159/159 modules carry an intelligence verdict, 27 of
  them an explicit FAIL

---

## 2. The intelligence standard — the largest gap, and the one that matters most

The goal document's §1a defines **23 mechanical tests**: L1–L10 (learning), R1–R9 (reasoning),
D1–D4 (depth). This is the standard the project owner set and sharpened twice — *"not only
learning but also real intelligence"*.

**Six of the twenty-three are mechanically run. Seventeen are not run at all.**

```
probe_axis_tests_run -> partial
perp, spot ran the axis probes:
  L1=PASS   L3=PASS   L4=PASS   L6=FAIL   R4=NOT MEASURED   R9=PASS
failing: perp L4, perp L6, spot L6
```

What that means test by test:

- **L1 provenance-of-constants — PASS.** No numeric parameter in the decision path is untraceable.
- **L3 parameter-randomisation — PASS.** Trained values are not decorative.
- **L4 ablation — FAILS on perp.** Freeze the component and the output is statistically
  indistinguishable. On perp, **the model is not measurably contributing.** This is the single
  most important red flag in this document.
- **L6 out-of-regime — FAILS on both.** Expected and disclosed: the record covers one regime and
  four days. It cannot pass until there is history spanning a structural boundary.
- **R4 abstention with teeth — NOT MEASURED.** The bots abstain constantly (223,987 abstentions
  on perp in 600 polls) and the coverage guarantee behind those abstentions is unverified.
- **R9 mechanism-swap — PASS.** The brains reason over structure, not feature names.

**Not run at all: L2, L5, L7, L8, L9, L10, R1, R2, R3, R5, R6, R7, R8, D1, D2, D3, D4.**

Among those, three are load-bearing and absent:

- **L7 trial-count accounting** with DSR/PBO applied. `validation/promotion_gate.py` implements
  it; nothing on the model path calls it. Uncounted retuning makes a reported edge *unverified by
  definition*, and the champion path retrains on a supervisor loop.
- **L9 catastrophic-forgetting check.** Live calibration updates on every close (LB-05) with no
  regression suite behind it.
- **R2 DoWhy refutation battery** — §1a calls failing any one of these *"a hard mechanical
  disqualifier"*, and it has never been run.

---

## 3. What was ruled and never built

Forty-five rulings are recorded. These are the ones whose subject does not yet exist in code.

| Ruling | What it asked for | Measured state |
|---|---|---|
| **RL-009 / RL-014** | Universe-wide opportunity scanning as a property of each bot — every symbol watched, the bot waits for each symbol's own setup, "like a teacher trying to pass every student" | **Partly built.** 570 perp and 1,351 spot symbols ARE watched live. But there is no *opportunity monitor*: the bot evaluates each symbol independently per poll and has no cross-sectional ranking, no notion of a symbol's setup maturing, no memory of near-misses. `src/strategy/` holds two modules and neither is one. |
| **RL-010 / RL-013** | Real intelligence — reading books/articles, research capability, not code that follows instructions | **Not built.** No research or ingestion capability of any kind exists. |
| **RL-011 / RL-023** | Two direction agents plus profit-tail, per segment | **Built for 2 of 4 segments.** perp and spot have learned bull/bear/profit-tail. dated and options run rule brains with no champion registered. |
| **RL-019** | Each segment is its own bot with its own architecture, data and features | **Built structurally, thin in practice.** Four bots exist with their own feeds and universes, but all four share one feature set (`live_features`), so "its own features" is not yet true. |
| **RL-022** | Market making, mean reversion, and funding/basis carry, after directional scalping is judged | **3 rows BLOCKED**, correctly, pending PB-10's verdict. Not forgotten. |
| **RL-027** | Future features and intelligence learning connecting to the brains | **Not built.** No extension point exists. |
| — | **slice-5: allocator, portfolio risk, brains 2–3** | **Zero rows written.** The slice is named in the plan and has no contents. Portfolio-level risk — correlation caps, portfolio heat, drawdown circuit breakers — does not exist anywhere in the system. |

---

## 4. How we got here — the process finding

Today's work, in order: capital declaration, shared pool, per-trade leverage, excursion columns,
15m/30m backfill, endpoint routing, rate-budget truncation, champion promotion gate, sample
uniqueness pooling.

**Nine pieces of work. Every one of them was a defect found while looking at something else.**
Not one came from the plan asking to be built. That is Rule 6 run backwards, and it is invisible
precisely because the work was genuinely useful — each fix was real, each was verified, several
were serious (the champion gate was replacing better models with worse ones; the sizing defect
meant the system could not tell whether it made money).

The cost is not wasted effort. The cost is that **slice-5 has zero rows, seventeen of
twenty-three intelligence tests have never run, and 599 ledger rows sit in repositories nothing
references** — while nine days of good engineering went into the things that happened to be
broken in front of us.

---

## 5. What the measured state says to build next

Ranked by what the evidence in this document supports, not by what is interesting.

1. **Find out whether there is any edge at all.** L4 ablation FAILS on perp: freezing the model
   leaves the output statistically unchanged. Before anything is added to a bot, establish
   whether its brain contributes. A 1.2 percentage-point out-of-fold edge on four days is the
   claim under test, and the ablation says it may be zero.
2. **Mark the cutover in the journal**, so `probe_margin_per_trade_within_band` can tell the old
   sizing regime from the new one. Small, and it unblocks the acceptance test for the whole
   capital slice.
3. **Wire L7 into the model path** — trial counting with DSR/PBO. The machinery exists and is
   called by nothing. Until it runs, every reported edge is unverified by §1a's own definition.
4. **Write slice-5's rows**: correlation caps, portfolio heat, drawdown circuit breaker. Forty
   concurrent crypto longs are one leveraged BTC bet, and nothing currently says otherwise.
5. **Build the opportunity monitor** RL-009 and RL-014 actually asked for — cross-sectional
   ranking and setup tracking, not per-symbol independent evaluation.
6. **Search the ledger before each of the above.** 599 PRIOR-ART rows. Four capabilities were
   designed from scratch on 2026-08-08 that the corpus already held.

---

## 6. What this document does not know

- **The real per-symbol label uniqueness.** The pooling defect is fixed, but the live measurement
  timed out. The number arrives on the next retrain.
- **Whether paper P&L predicts live P&L.** No queue position, no partial fills, no latency in the
  fill model. Optimistic and pessimistic prices bracket it, which is better than a single number,
  but the bracket has never been validated against a real fill.
- **Whether the intelligence standard is passable.** Seventeen of its tests have never been run,
  so their outcome is unknown rather than assumed.
