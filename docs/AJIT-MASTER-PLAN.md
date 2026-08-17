# AJIT MASTER PLAN

**This file is the top of the authority chain for WHAT TO BUILD NEXT.** Where the goal document,
`ARCHITECTURE.md`, `DECISIONS.md`, `FEATURES.md` or any plan disagrees with this file about the
order of work, this file wins. They keep what they own: the goal document keeps the goal and the
§1a intelligence standard, `ARCHITECTURE.md` keeps structure, `DECISIONS.md` keeps the record of
why, `FEATURES.md` keeps the capability list, and `~/research/ledger/` keeps the research corpus.

`docs/superpowers/plans/2026-08-09-full-build-master-plan.md` is **superseded by this file** and
retained as the record of what was decided on 2026-08-09.

**Nothing in this file reports state.** Every row's `state:` field names a probe. The measured
values live at `~/research/dashboard/ajit-master-plan.html`, generated on every boards pass and
never committed (Rule 8, Rule 9). If you find a status value typed below, the parser is broken —
it is supposed to refuse.

Design: `docs/superpowers/specs/2026-08-17-ajit-master-plan-design.md`
Rulings: `docs/rulings.json` — 21 rulings, verbatim, dated, each with a scope
Implementation: `docs/superpowers/plans/2026-08-17-ajit-master-plan-slice-0-governance.md`

## How to read a row

Eight fields, none optional. Nothing is buildable until all eight are filled, so an unwritten
decision shows up as a blank field **before** any code is written rather than as a wrong module
afterwards.

`does` what it is responsible for · `satisfies` the ruling ids it serves · `sources` the corpus
members it discharges · `depends on` rows that must land first · `probe` the thing that measures it
· `accepts` the one sentence that decides whether it works · `state` always `measured by <probe>`,
or `BLOCKED`/`DECLINED` which are decisions rather than measurements.

## Gates

**BUILT** — every row in the slice lit, the bot running under its own supervisor, journalling fills,
`integrity.unsupported_claims` reporting zero for it, its board tile measured. *The next slice starts
here.*

**EARNING** — a strategy of that bot promoted through the full gate stack on observed history.
Tracked per bot. **Never blocks the build**, because that clock is calendar time on a record that
started 2026-08-08 and cannot be shortened by reconstruction.

## Scope, and why coverage is a fraction

Every ruling binds at a level, and the level is the denominator. `per-segment` needs a row in all
four segment slices; `per-brain` in all twelve brains; `shared` and `system` need one row anywhere.
Assigned once is not resolved four times — that is how a cross-cutting capability reads as done
while three of four bots never received it, and it is what happened to the examination-hall ruling
of 2026-08-02.

---

## SLICE slice-0 — FOUNDATION & GOVERNANCE

The plan, the machinery that measures it, the hooks that enforce it, and the two operational fixes
that cannot wait for a later slice. The shared bot framework — brain interface, journal format,
risk-gate interface, on/off switch — is deliberately NOT here: it is a separate subsystem and gets
its own plan when slice 1's row inventory is reviewed.

### SL-01
  slice:      slice-0
  does:       copy the store to GCS beside the raw tape on every offload pass
  satisfies:  RL-020
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: none
  probe:      probe_store_offloaded
  accepts:    gs://capture-raw-data4134/store/funding lists non-empty, and a part
              being written is never uploaded
  state:      measured by probe_store_offloaded

### SL-02
  slice:      slice-0
  does:       make every session reach the same project memory whatever its directory
  satisfies:  RL-016
  sources:    2026-08-17-ajit-master-plan-design.md#0. Why this exists — the measured failure it answers
  depends on: none
  probe:      probe_memory_reachable
  accepts:    a session started from / sees the same memory count as one started
              from the home directory
  state:      measured by probe_memory_reachable

### SL-03
  slice:      slice-0
  does:       load and validate the rulings register, refusing a typed status
  satisfies:  RL-021 RL-016
  sources:    2026-08-17-ajit-master-plan-design.md#6. Enforcement — four mechanisms plus the path fix
  depends on: none
  probe:      probe_rulings_register_loads
  accepts:    all 21 rulings parse, every scope is from the closed set, and a row
              carrying a status field is refused
  state:      measured by probe_rulings_register_loads

### SL-04
  slice:      slice-0
  does:       parse the spine into slices and eight-field rows, refusing a typed state
  satisfies:  RL-021 RL-007
  sources:    2026-08-17-ajit-master-plan-design.md#4. Row anatomy, states and gates
  depends on: SL-03
  probe:      probe_spine_parses
  accepts:    the real spine parses into six slices and every row cites a ruling
              that exists in the register
  state:      measured by probe_spine_parses

### SL-05
  slice:      slice-0
  does:       compute per-scope coverage so partial delivery reads as a fraction
  satisfies:  RL-021 RL-009 RL-019
  sources:    2026-08-17-ajit-master-plan-design.md#5. Reconciliation — how the file proves nothing was skipped
  depends on: SL-04
  probe:      probe_scope_arithmetic
  accepts:    a per-segment ruling with rows in three of four slices reports 3/4
              and is unresolved, naming the missing segment
  state:      measured by probe_scope_arithmetic

### SL-06
  slice:      slice-0
  does:       sweep the ledger, the catalogue and every design section, classifying each member
  satisfies:  RL-021 RL-007 RL-015
  sources:    2026-08-17-ajit-master-plan-design.md#5. Reconciliation — how the file proves nothing was skipped
  depends on: SL-04
  probe:      probe_reconciliation_sweeps
  accepts:    every member of all three populations resolves to exactly one of
              assigned, declined, blocked, prose or unassigned, and the unassigned
              count is published rather than hidden
  state:      measured by probe_reconciliation_sweeps

### SL-07
  slice:      slice-0
  does:       hold the written order and decisions as this document
  satisfies:  RL-007 RL-017 RL-021
  sources:    2026-08-17-ajit-master-plan-design.md#1. What the file is, and its authority
  depends on: SL-04
  probe:      probe_spine_present
  accepts:    the document parses, states its own authority, and names the plan
              it supersedes
  state:      measured by probe_spine_present

### SL-08
  slice:      slice-0
  does:       measure each ruling against running code and render the conformance board
  satisfies:  RL-021 RL-012
  sources:    2026-08-17-ajit-master-plan-design.md#6. Enforcement — four mechanisms plus the path fix
  depends on: SL-03 SL-05
  probe:      probe_ruling_conformance
  accepts:    every ruling renders a row, one naming no probe renders NOT MEASURED,
              and no row is ever green without a probe having run
  state:      measured by probe_ruling_conformance

### SL-09
  slice:      slice-0
  does:       render the plan board with per-slice counts, the active slice and the next row
  satisfies:  RL-012 RL-017 RL-021
  sources:    2026-08-17-ajit-master-plan-design.md#8. Progress reporting
  depends on: SL-06 SL-07
  probe:      probe_plan_board_rendered
  accepts:    an empty slice renders 0 / 0 rather than complete, and the page
              carries no date or effort estimate
  state:      measured by probe_plan_board_rendered

### SL-10
  slice:      slice-0
  does:       refuse a new module under src/ that no plan row names
  satisfies:  RL-021 RL-007
  sources:    2026-08-17-ajit-master-plan-design.md#6. Enforcement — four mechanisms plus the path fix
  depends on: SL-07
  probe:      probe_enforcement_live
  accepts:    a new src/ file with no row exits 2, one with a row exits 0, and an
              edit to an existing file is never blocked
  state:      measured by probe_enforcement_live

### SL-11
  slice:      slice-0
  does:       inject the ruling list and the active slice into every session at start
  satisfies:  RL-021 RL-016 RL-001
  sources:    2026-08-17-ajit-master-plan-design.md#6. Enforcement — four mechanisms plus the path fix
  depends on: SL-03 SL-07
  probe:      probe_enforcement_live
  accepts:    the hook prints one line per non-superseded ruling and exits 0 on
              every path, including when the register is absent
  state:      measured by probe_enforcement_live

### SL-12
  slice:      slice-0
  does:       resume the paper engine from a persisted watermark instead of re-priming the archive
  satisfies:  RL-020 RL-005 RL-018
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: none
  probe:      probe_paper_engine_running
  accepts:    a restart reaches its first poll without re-feeding the archive, and
              no archived event is ever traded on resume
  state:      measured by probe_paper_engine_running

### SL-13
  slice:      slice-0
  does:       make this plan the document every other document points at
  satisfies:  RL-007 RL-019 RL-018 RL-021
  sources:    2026-08-17-ajit-master-plan-design.md#1. What the file is, and its authority
  depends on: SL-07
  probe:      probe_authority_chain_consistent
  accepts:    exactly one document claims authority over the order of work, the
              superseded one says so in its own text, and goal §3b records RL-019
              in the user's own words
  state:      measured by probe_authority_chain_consistent

---

## SLICE spot-bot — SPOT BOT

*Row inventory pending the reconciliation sweep, reviewed with the user before this slice starts
(design §12). An empty slice reports 0/0, which is the honest board for unplanned work.*

Known shape from the design: segment data completion → spot features → BULL, BEAR and PROFIT-TAIL
brains → segment risk gate → engine, journal, supervisor and on/off switch → board tile → three
axis verdicts per brain → reconciliation sweep. The universe-wide opportunity monitor (RL-009,
RL-014) is a row here and in each of the three slices below; it is `per-segment` and reads **0/4**
until all four exist.

---

## SLICE perp-bot — PERP BOT

*Row inventory pending review.* The running `plumbing-momentum` engine retires into this slice: its
journal is archived and marked as plumbing that made no edge claim, so it can never later be read
as a result. Until then it keeps running, because something trading is what the 24/7 record needs.

---

## SLICE dated-bot — DATED FUTURES BOT

*Row inventory pending review.* `bybit` carries 48 dated contracts and `features.term_structure`
already reads them.

---

## SLICE options-bot — OPTIONS BOT

*Row inventory pending review.* Less blocked than the record says: `store/option_chain` exists with
305 parquet files across per-instrument partitions and `option_chain deribit` is in the store
supervisor's rotation, so goal §3a item 2's *"no builder reads it yet"* is out of date. The chain
cannot be backfilled — the endpoint ignores a `timestamp` parameter — so every hour of capture from
2026-08-16 onward is the only history this segment will ever have.

---

## SLICE slice-5 — ALLOCATOR, PORTFOLIO RISK, BRAINS 2-3

*Row inventory pending review.* Nothing here has anything to allocate between until at least two
bots reach BUILT.
