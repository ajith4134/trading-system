# Autonomous Crypto Trading System

**This file is an index, not a rulebook.** The rules already exist, were researched, and were argued
through. Read the source rather than a paraphrase of it — a summary written from memory is how
things get quietly re-derived wrong.

## Read before doing anything here

| Document | What it settles |
|---|---|
| **`docs/AJIT-MASTER-PLAN.md`** | **The order.** What to build next, its slices, its gates, and every ruling. **Top of the authority chain** — where any document below disagrees with it about the order of work, it wins. Measured state: `~/research/dashboard/ajit-master-plan.html` |
| **`docs/rulings.json`** | **What the user settled**, verbatim and dated. 21 rulings, each with a scope and a probe. Recorded the moment a ruling is given, before other work continues |
| `docs/superpowers/specs/2026-08-08-final-project-goal-design.md` | **The goal.** Prime directive, the human-settled rulings including **§3a** (intraday on all segments) and **§3b** (each segment is its own bot), definition of done, and **§1a — the intelligence standard with its 23 mechanical tests** |
| `~/research/ADVANCED-INTELLIGENT-CODE.md` | **Why §1a says what it says.** The consolidated evidence on learning, reasoning, depth and evolved-program search, with every claim traced to a primary source |
| `~/research/IDEAS-INTELLIGENCE.md` | The corpus's own answer — 12 capability areas, the ten to build first, and the master test |
| `~/research/ledger/` | **1,485 rows.** Every feature, constraint and idea from the corpus, eight prior repos and seven video breakdowns. Rebuild the index: `python3 ~/research/scripts/build_ledger_index.py` |
| `~/research/ARCHITECTURE.md`, `DECISIONS.md`, `FEATURES.md` | Structure, the record of why, and the exhaustive capability list |

## The standing requirement — do not make the user restate it

They asked for **real learning, real reasoning and real depth**, then sharpened it: *"not only
learning but also real intelligence."* Those are separate axes and are judged separately. A model
retraining hourly learns; it does not reason.

**The standard is §1a. Do not re-derive it, do not summarise it from memory, and do not soften its
honest boundary into a sales pitch.**

## Before writing anything new

**Search the ledger first.** 599 of its rows are PRIOR-ART — working code in prior repos that no plan
references. On 2026-08-08, four capabilities were designed from scratch that the corpus already held.

**Declining a row is allowed. Forgetting one is not.**

**And read the donor code, not the ledger's note about it.** Also 2026-08-08: two rows flagged
"reusable as-is" were wrong once fetched raw. A CPCV purge was one-sided (leaked a label horizon of
training rows into every test block) and a Deflated-Sharpe gate was fed the CPCV *path* count as its
trial count, deflating a candidate picked from thousands as though 15 things were tried. Both read as
rigorous. **All three prior-art defects found that day failed in the flattering direction** — which is
why the systems that shipped them never noticed. See the addendum in
`~/research/ledger/merged/risk-execution-validation.md`.

## Working rules for this repo

- **Python 3.12** via `uv`. Never system Python (3.14.4 — too new for polars).
- Tests: `.venv/bin/python -m pytest -q`. Baseline **2,192 passed, 1 skipped** (2026-08-19); it takes ~10 minutes on this box while the bots are trading.
- **Every RESEARCH and TRAINING market-data read goes through `store.clock_gated_reader`.** No
  direct Parquet reads. That is the whole reason Layer 1 exists, and it is unchanged for anything
  that learns, backtests, validates or promotes.
- **Trading prices do NOT come from Layer 1 — RL-024, 2026-08-18.** *"sould te paper tradin sould be
  done on live data on live crypto prices not on old data"*. Every segment bot takes its prices from
  `live.live_feed`, which holds the venue websocket or REST poll directly. Measured 2026-08-18: a
  cold filtered store scan took 185.6s, an hour-pruned three-symbol read was killed at 300s having
  returned nothing, and the running engine sat 14 minutes without completing one poll — while the
  live feed delivered 501 trades and 3,406 two-sided quotes in 15 seconds at a newest-tick age of
  1 millisecond. The old rule forbade live REST in a pricing path; that prohibition was written
  before there was a trading loop, and a bot polling Layer 1 is backtesting on a delay while
  carrying the name paper trading. **The two paths are separate on purpose: the store is the
  corpus, the feed is the clock, and neither is allowed to do the other's job.**
- **No quote may be produced from a default.** A missing or stale input produces a refusal naming
  what was missing.
- Timestamps int64 nanoseconds UTC. Fees `Decimal` basis points, never float.
- Docstrings explain *why*. Tests are named for the behaviour they defend.
- Displays show measured state; absence renders as `NOT MEASURED`, never green (Rule 8).
- Never commit generated output that reports state — commit the generator (Rule 9).

## The trap this codebase keeps falling into

`tail_specs()` — built, tested, called by nothing. `ai-scientist/` and `fable5/` — session logs
dressed as autonomous machinery. Five circuit breakers and health checks living only in docstrings.
A Dreamer rollout rewarding itself with `np.random.normal`. A live system that lost **$837 over
10,240 trades** because it credited the same P&L to all 36 of its features.

**Run the check and paste the output before writing that something works.** If it was not run, say so.
