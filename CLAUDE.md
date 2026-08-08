# Autonomous Crypto Trading System — project rules

**Read `docs/superpowers/specs/2026-08-08-final-project-goal-design.md` before doing anything here.**
It is the goal. This file is the short version that must not be forgotten.

---

## The standing requirement — do not make the user restate this

Every session, without being asked: **the bots must contain real learning, real reasoning, and real
depth. Not hardcoded values with intelligent-sounding names.**

Learning and reasoning are **different things** and are judged separately. A model that retrains
hourly learns. It does not reason.

Full standard with all 23 mechanical tests: **goal doc §1a**. The short form:

| Axis | Question | Fails when |
|---|---|---|
| **Learning** | Where did this number come from? | Someone typed it |
| **Reasoning** | Does it know *why*, and know when it doesn't? | Fluent causal language over pattern completion |
| **Depth** | Deep module or shallow? | Elaborate interface hiding little |

**The three tests to run before claiming any of the three:**

1. **Provenance + ablation.** Show the fitting process — loss, dataset, re-runnable script — that
   produced the current values. Then freeze or remove the component and show measured behaviour
   changes. Fail either and it is a parameterised script wearing the costume of learning.
2. **Falsifiable side-prediction.** The claimed mechanism must imply some *other* checkable
   consequence, checked on data never used to fit it. A curve-fit has no side-predictions.
3. **Interface-to-implementation ratio.** Public symbols exposed, against lines of implementation
   behind them. Ousterhout: *"It is more important for a module to have a simple interface than a
   simple implementation."*

**Every module ships with an axis verdict** — learning / reasoning / depth, each pass, fail, or
explicitly *not applicable* with a reason. Blank is not allowed. A Parquet writer does not reason;
say so rather than leaving it empty.

**The master test, which governs all of the above:**

> ### Does it change what the system does when it is wrong?

Anything that only improves behaviour when the system is already right is decoration.

**And the honest boundary, which must never be softened into a sales pitch:** no architecture
available in 2026 produces understanding. Alpha Arena — six frontier models, $10k each, four down
30–63% in 17 days. What is reachable is the distance between a script and an adapting, self-aware
system, and that distance is crossable with unglamorous engineering.

---

## Prime directive

**Maximise the fraction of green days, subject to a hard tail-loss cap the system cannot raise.**

The cap is part of the objective, not a constraint added afterwards. Without it, "maximise green
days" selects for selling insurance — green almost every day until it returns the multi-year gain in
one session.

Carry is the earnings core. Directional strategies earn allocation **only by measurably raising the
portfolio's green-day rate at the same tail cap.**

The learning curve is a tracked metric: win rate and profitable-trade count must rise with cumulative
**out-of-sample** trades, and only counts as improvement when total P&L and green-day rate rise with
them.

---

## Non-negotiable

- **No strategy bypasses the risk gate.** FTX/Alameda and bZx both died from exactly that exemption.
- **No path from web content — or market-feed text — to capital that skips the promotion gate.**
  Token names, on-chain memos and listing announcements are attacker-authored strings. Structured
  extraction only, never free text into a reasoning context (§5a.6b).
- **The research loop's evaluator is deterministic, non-LLM, and outside the system's write access.**
- **Self-modification goes through the bounded canary** — typed `ChangeSpec`, matched-cohort
  evaluator, rigor gate, canary, auto-rollback. **One loop active at a time.** Reference
  implementation: `signals/scibrain/` in `nse-crypto-bot-final` (ledger row OGE-072).
- **Correct per-feature attribution before any learning loop runs** (§10.8). The prior system lost
  **$837 over 10,240 live trades** because it credited bot-wide P&L to all 36 features and
  deactivated all of them, twice, in production.
- **Two human gates only:** the capital dial, and paper → real money.

---

## Scope

**Crypto only. Zero NSE.** Spot, perps, dated futures; options at Phase 6. Execution on Binance and
Hyperliquid; Kraken/OKX/Coinbase reference-price only.

The NSE repos (`nse-botonly`, `nse-crypto-bot-final`) are **feature donors** — machinery ports, the
market does not.

**Universe-wide.** Watch every tradeable symbol in every segment; act on any one only when that
symbol's own condition fires. Selectivity in time, not space. The unit of edge is
`(symbol, condition, moment)`.

---

## Before adding anything, check the ledger

`~/research/ledger/` — 1,485 rows across 8 slices. Rebuild the index with:

```
python3 ~/research/scripts/build_ledger_index.py
```

**599 rows are PRIOR-ART: working code in prior repos that no plan references.** Four times on
2026-08-08 a capability was designed from scratch that the corpus already held. Search the ledger
before writing something new.

**Declining is allowed. Forgetting is not.** Every ledger row must end CLAIMED, PLANNED or DECLINED
with a written reason.

---

## Working rules for this repo

- **Python 3.12** via `uv`. Never system Python (3.14.4 — too new for polars).
- Run tests with `.venv/bin/python -m pytest -q`. Current baseline: **414 passed, 1 skipped.**
- **Every market-data read goes through `store.clock_gated_reader`.** No direct Parquet reads, no
  live REST in a pricing path. That is the whole reason Layer 1 exists.
- **No quote may be produced from a default.** A missing or stale input produces a refusal that names
  what was missing.
- Timestamps are int64 nanoseconds UTC everywhere. Fees are `Decimal` basis points, never float.
- Docstrings explain *why*. Tests are named for the behaviour they defend.
- **Displays show measured state.** Absence renders as `NOT MEASURED`, never green, never blank.
- **Never commit generated output that reports state.** Commit the generator.

## The trap this project keeps falling into

Things that read as built and are not. `tail_specs()` — built, tested, called by nothing.
`ai-scientist/` and `fable5/` — session logs dressed as autonomous machinery. Five circuit breakers
and health checks existing only in docstrings. A Dreamer rollout rewarding itself with
`np.random.normal`.

**Before writing that something works, run the check and paste the output.** If it was not run, say
so. That is Rule 0, and it is the rule this codebase most needs.
