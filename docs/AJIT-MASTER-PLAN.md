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

### SL-14
  slice:      slice-0
  does:       walk each parquet fragment's schema once per store generation, not once per read
  satisfies:  RL-020 RL-018 RL-005
  sources:    ~/research/DECISIONS.md#15.3 The paper engine was killing itself
  depends on: SL-12
  probe:      probe_poll_scan_cost
  accepts:    a poll opens fragment footers proportional to what arrived since
              the last read, and the heartbeat carries what the poll cost
  state:      measured by probe_poll_scan_cost

### SL-15
  slice:      slice-0
  does:       partition the store by availability hour, and compact a sealed hour to one part per symbol
  satisfies:  RL-020 RL-018 RL-005
  sources:    ~/research/DECISIONS.md#15.5 The scan is the hot spot, not the walk
  depends on: SL-14
  probe:      probe_poll_scan_cost
  accepts:    a 60-second poll opens the fragments of at most two hour
              directories however long capture has been running, and every row
              of the old layout is present in the new one before the old is
              retired
  state:      measured by probe_poll_scan_cost

> **Hour, not date, and the file count is why.** Measured 2026-08-17: 538 MB across 52,487 files
> is an average of **10 KB per file**, roughly 6,000 new files a day over 2,230 symbol directories.
> A date-grained path leaves a mature day's directory holding every file written that day, so an
> evening poll re-walks ~6,000 fragments and that number grows with the universe. An hour-grained
> path bounds a 60-second poll at the hour directories it actually needs, and the bound does not
> move as the archive grows.
>
> **Measured on a real 60-symbol slice of the live bars dataset, 2026-08-17.** A one-hour bound
> selects **59 of 1,805 fragments** under the hour layout and **1,587 of 1,587** under the legacy
> one, returning byte-identical rows. That is the whole row in one number.
>
> **The file count goes UP, not down, and the earlier claim of a ~20x cut was wrong.** Compaction
> merges every legacy part for one (hour, symbol) into one — but these writers already snapshot
> about hourly, so there is little to merge, and a legacy part spanning more than one hour is SPLIT
> across hour directories instead. Measured: bars 1,587 parts → 1,805, `dated_futures` 680 → 1,472.
> The saving is pruning, never fewer files, and a plan that promised both would have been checked
> against only the half that was true.
>
> Decision taken by the user 2026-08-17: hour grain, migrate the existing archive, compact sealed
> hours. The old layout stays on disk and in GCS until the new one verifies row for row.
>
> **The writers deployed themselves, and that is now a fact this plan has to hold.** Minutes after
> the new `append_partition` was saved, the live bars dataset held 4,595 hour-partitioned parts
> beside 2,231 legacy `symbol=` directories — the supervisors restart their child on a loop and a
> restarted child imports whatever is on disk. Hive partitioning gives the legacy parts a NULL
> hour, so an hour-bounded read dropped every one of them: 1 of 6 rows in the reproduction. Hour
> pruning is therefore enabled only when NO top-level `symbol=` directory remains, which makes a
> half-migrated dataset correct-but-slow rather than fast-and-wrong, and turns pruning back on by
> itself when the migration finishes. See `~/research/DECISIONS.md` §15.6.
>
> Design: `docs/superpowers/specs/2026-08-17-hourly-store-partitioning-design.md`. The row builds
> one new module, `store/hourly_migration.py`, and changes the layout knowledge already held by
> `store/parquet_partition.py`, `store/cli.py` and `statuswall/evidence.py`.

> **The swap ran 2026-08-17 16:17, and the two things it left behind are part of this row.**
> `bars_60000000000ns` is now the hour layout; the symbol layout is retained at
> `bars_60000000000ns.legacy-symbol-layout` and is not deleted by this row.
>
> **1. Stragglers.** The swap is two renames and capture never stops, so parts written to the old
> directory between the last migration pass (15:21) and the rename (16:17) are stranded in the
> retired copy: **16,517 parts, about 56 minutes of bars across 2,235 symbols.** Nothing is lost -
> they are on disk and in GCS - but they sit outside what readers open, so this row is not done
> until they are folded in. `migrate_dataset_to_hourly(..., into=<live dataset>)` is that fold.
>
> **A fold makes a weaker claim than a migration, and the report says which one it made.** A
> migration owns its target and can claim equality: every legacy row present, and no other row. A
> fold's target is the LIVE dataset, which capture keeps appending to while the fold runs, so rows
> the source never held are expected rather than a fault - under the strict rule no fold could ever
> verify, and a check nothing can pass becomes a check that gets worked around. `FOLD_CLAIM` is
> therefore *every group of the retired layout is present in the live one with at least its row
> count*. It still refuses the failure it exists to catch: a source group missing or short in the
> target. `swap_in_migrated_dataset` refuses a fold report outright, because a fold's building
> dataset IS the live one and renaming that aside would move the store out from under every reader.
>
> **2. The writers still have to be restarted, and until they are, pruning cannot switch on.**
> OUTSTANDING as of 2026-08-17 16:20. `store.live_bars` pids 1489 and 1491 have been up since the
> 07:50 boot, holding the pre-SL-15 `append_partition` in memory, and they created nine fresh
> top-level `symbol=` directories in the live dataset at 16:20 - after the swap. Hour pruning is
> disabled while any top-level `symbol=` directory remains (§15.6, and deliberately so), so **the
> swap buys nothing until those two processes are replaced by ones running the current code**, and
> folding the stragglers before then only opens a fresh straggler window. A layout change is not
> delivered when the code lands; it is delivered when every long-lived writer has been restarted
> onto it, and this row is not measured until `probe_poll_scan_cost` says so.

> **The 2026-08-17 11:19 diagnosis in this row was wrong, and the correction matters more than the
> fix.** That note named the schema walk as the hot spot, from a read filtered to zero rows that
> had not finished in 10 minutes. Re-measured at 11:50 on the live bars dataset, now **52,487
> fragments across 2,230 symbol partitions**, each part timed separately:
>
> | part of a poll | cold | warm |
> |---|---|---|
> | dataset discovery | 0.6s | 0.6s |
> | schema walk, every fragment footer | 8.3s | — |
> | schema walk, cached (SL-14) | — | **8 footers, 1.7s** |
> | scan filtered to match NOTHING | **185.6s** | 8.8s |
>
> So the walk was ~8s of a ~195s read. **The scan is the hot spot**, and it is cold-cache IO: the
> same zero-row scan costs 185.6s cold and 8.8s warm, which is why it read as unbounded on a box at
> load 27 with 22 of 29 GB in use and the page cache being evicted under it. The earlier 10-minute
> observation was real; the attribution was not.
>
> **SL-14 is still worth having and is now built** — it removes a duplicate walk of the whole
> archive per read, and it makes the cost measurable per poll rather than inferable after an OOM.
> It is not sufficient, and this file should not have implied it would be.
>
> **SL-15 is the fix.** The partition key is `symbol` and availability is not in the path, so
> `not_before_ns` (SL-12) filters ROWS after the scan and can never prune FILES. Only a date in the
> path lets pyarrow skip directories before opening anything.
>
> **The property both rows must keep.** The schema walk cannot simply be deleted: without it
> pyarrow infers the dataset schema from the first fragment and **silently drops** columns added by
> later partitions. On 2026-08-09 `funding_interval_hours` vanished from every read that way, and
> annualising a 4-hourly rate as 8-hourly is wrong by a factor of two. SL-14 keeps it by caching
> rather than skipping, and `tests/test_schema_cache.py` asserts the column survives a warm cache
> and that a disappearing fragment forces a full rebuild.
>
> **Rejected, and why.** *Prune fragments by file mtime* — cheapest and the most dangerous: mtime
> is not a data property, and a restore from GCS resets it, so the store would silently skip real
> rows. *Have the engine record consumed snapshot ids* — fixes one reader; the boards generator,
> which stalled from 09:41 to 11:51 on 2026-08-17, is a second reader and would need its own.
>
> Decision taken by the user 2026-08-17: cache now, repartition next.

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
