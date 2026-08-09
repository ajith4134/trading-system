# FULL BUILD — reordered recommendation (expert opinion, recorded)

Date: 2026-08-09
Status: **RECOMMENDATION, decision pending.** The active plan remains
`2026-08-09-full-build-master-plan.md` (strict order) until the user chooses.
If the user says "reorder", this file's order replaces §Phases there and this
line changes to ACTIVE. If the user says "keep strict", this file stays as the
record of the alternative that was put and declined.

## The question

The user asked for an expert opinion on the strict ordering: complete every
plan, idea, feature and goal first; start paper trading only after.

## Verdict

Strict order is the highest-risk ordering available, and the risk is documented
in the project's own history. One change fixes it: **move the paper engine
(Phase J) to immediately after Phase A.** Everything else — full completeness,
all 186 features, nine bots, ledger closure — stays identical.

## Evidence the verdict rests on (all measured earlier, none asserted fresh)

1. **The six prior repos.** `prior-attempts-postmortem.md`, read from GitHub via
   `gh` 2026-08-01: `nse-crypto-bot-final` at 864 Python files and 243 tests,
   complete-looking, validation last, abandoned. Six attempts in twelve weeks;
   three inside seven days. The postmortem's diagnosis was never engineering
   quality — it was building on untested assumptions until the pile could not be
   trusted, then restarting. Strict order is that shape.

2. **This repo's defect record.** Every defect it caught was caught at the
   moment of DRIVING, never at the moment of building:
   - promotion pipeline: lookahead found only when run — noise scored Sharpe
     3.90 past three of five gates (DECISIONS §13 step 6)
   - HoldoutCustodian: accepted argument no caller supplied; guarded nothing
   - auto-halt tile: read "armed" while `observe()` had no caller
   - `tail_specs`, `spread_and_depth`, `drawdown_distribution`,
     `participation_calibration`: built, tested, reached by nothing
   Built-undriven code rots silently. Strict order manufactures ~150 features of
   it simultaneously.

3. **Phases C, D, E are unverifiable without paper execution.** The consumer of
   a model is a strategy; the consumer of a strategy is execution. A BULL bot
   with no forward-paper run is "done" only as files-exist — Rule 0 has nothing
   to run and Rule 8's board would light on probes that never touched a market
   decision.

4. **Serial costs the live date twice.** Live promotion requires the
   forward-paper track record (pipeline stage 4) AND the observed-history clock
   (started 2026-08-08). Paper-last runs build-weeks then paper-weeks in
   series. Paper-early runs build, paper record and observed history
   concurrently. Same end state; live arrives weeks sooner.

## What the strict instinct gets right, and how it is kept

- **Completeness** is guaranteed by Phase I (ledger closure: 0 UNRESOLVED, every
  PLANNED row CLAIMED or DECLINED-with-reason) — by the ledger, not by the
  ordering. Nothing in the reorder skips anything.
- **One committed push, no restarts** — unchanged.

## The recommended order

> H-early → A → **J (paper engine live)** → B → C → D → E → F → G → H → I

| Step | Content | Change from strict |
|---|---|---|
| H-early | build-progress board on the wall | none — first in both |
| A | data & ingestion completion | none — second in both |
| **J** | **paper engine built; forward paper supervisor joins boot chain** | **moved from last to third** |
| B–I | features, models, bots, portfolio, execution, brains, dashboard, ledger closure | order unchanged; each phase now verified against live paper contact as it lands |

Paper trading here is not the reward at the end — it is the **test instrument**
every later phase runs against. Defects surface one at a time on contact,
instead of as an avalanche after the last feature.

## Definition of done

Unchanged from the master plan, all five clauses — plus one: from step J
onward, no strategy-touching module (phases B–E) is marked done without a
forward-paper or replay run naming it as consumer.
