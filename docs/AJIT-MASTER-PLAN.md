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
  satisfies:  RL-021 RL-009 RL-019 RL-031
  sources:    2026-08-17-ajit-master-plan-design.md#5. Reconciliation — how the file proves nothing was skipped
  depends on: SL-04
  probe:      probe_scope_arithmetic
  accepts:    a per-segment ruling with rows in three of four slices reports 3/4
              and is unresolved, naming the missing segment
  state:      measured by probe_scope_arithmetic

> **The per-brain numerator could not count, measured 2026-08-19.** The denominator
> was twelve brains and the numerator compared SLICE KEYS against brain names. A
> slice key is `spot-bot` or `learned-brains`, never `spot-bot/BULL`, so the
> comparison could never match: all nine per-brain rulings read 0/12 and always
> would have — RL-026 "make the brains real ai" and RL-030 among them, the latter
> reading 0/12 on the day its row was built and its probe measured 2/4 bots live.
> **A fraction that can only ever be zero is not a measurement**, and it was sitting
> inside the one row whose whole job is to stop "assigned" meaning "done".
>
> **RL-031 settles the attribution.** A row in a SEGMENT slice covers that segment's
> three brains together, because that is how the bots are actually built — a
> capability lands in a segment bot and BULL, BEAR and PROFIT-TAIL receive it at
> once. Rows keep their eight fields; naming a brain per row would have made a ninth
> mandatory one and sent every existing row back for revision.
>
> **A shared or cross-cutting slice attributes to no brain, and that is the doctrine
> rather than an omission.** A capability built once is not thereby delivered to four
> bots. Letting `bot-framework` or `learned-brains` count for all twelve would
> reproduce inside the plan the exact failure the scope arithmetic exists to catch —
> which is why LB-09, sitting in `learned-brains`, correctly contributes nothing to
> RL-030's brain coverage until the segment slices carry rows for it.
>
> The probe also **names the gap** now rather than only counting it. It reported
> `RL-006 2/4` without saying which two bots were short, and a gap nobody names is a
> gap nobody closes.

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

> **Repointed 2026-08-19, and the row is unchanged.** `probe_paper_engine_running` measured
> `capture/paper/forward` - the `plumbing-momentum` engine RETIRED under RL-025 - so it reported
> DEGRADED on a heartbeat 38 hours old and deliberately never coming back. The four segment bots
> are the paper engines now and they satisfy this acceptance more strictly than the engine that
> was retired: they never read the parquet archive at all (RL-024), so "no archived event is
> traded on resume" holds by construction, and each journals what it rebuilt from its own fills
> before it polls. **A tile permanently red about something switched off on purpose is one
> everybody learns to skip, which is how the next real failure gets missed.**

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

> **Repointed 2026-08-19 with SL-12, and what it measures now.** Nothing polls the store on a
> loop any more - RL-024 moved every trading price to a live feed - so the subject of this
> measurement changed while the property being defended did not. What still reads the store on a
> cadence is the RETRAINER, and what makes that read cheap or ruinous is this row's partitioning.
> The probe therefore measures the layout directly: how many fragments a reader bounded to the
> newest hours must open, against how many exist. Measured 2026-08-19: **168,639 fragments across
> 73 hour directories**, with the newest two holding 4,288 of them.
>
> Its first run reported a healthy store as DEGRADED at `4288/2000+` - an exact numerator against
> a budgeted lower-bound denominator. A lower bound is a measurement and a ratio taken against one
> is not, so the share is now computed only from a complete count and reads PARTIAL otherwise.

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

### SL-16
  slice:      slice-0
  does:       finish the hour migration for the datasets still holding a legacy symbol layout
  satisfies:  RL-020 RL-018 RL-012 RL-032
  sources:    ~/research/DECISIONS.md#15.5 The scan is the hot spot, not the walk
  depends on: SL-15
  probe:      probe_hour_pruning_enabled
  accepts:    no dataset under the store holds a top-level symbol= directory, so
              _is_partitioned_by_hour returns true for every one of them, and each
              migrated dataset has a verified report before its legacy copy is
              retired
  state:      measured by probe_hour_pruning_enabled

### SL-17
  slice:      slice-0
  does:       write and compact a whole-universe dataset one part per venue per hour
  satisfies:  RL-020 RL-018 RL-012 RL-032
  sources:    ~/research/DECISIONS.md#15.5 The scan is the hot spot, not the walk
  depends on: SL-16
  probe:      probe_sealed_hour_compacted
  accepts:    a sealed hour of funding holds one part per venue rather than one per
              symbol, the rows and columns of that hour are unchanged across the
              compaction, and the hour still being written is never compacted
  state:      measured by probe_sealed_hour_compacted

> **Compaction alone cannot hold, and that is why this row writes as well as compacts.**
> `append_partition` groups by `(hour, symbol)` and drops SYMBOL from the file body
> because the `symbol=` path segment carries it. Compact a sealed hour into parts that
> sit directly under `availability_hour=` with the symbol back in the body, and the very
> next poll writes per-symbol parts beside them — leaving one dataset where some
> fragments take `symbol` from the path and others from the body. pyarrow infers those
> as different types, `string` against `large_string`, and **refuses to merge them**:
> the exact ArrowTypeError `append_partition` already documents itself as avoiding.
>
> So the writer moves with the layout. For a whole-universe dataset the path is
> `availability_hour=<H>/` alone, the venue is already in the part name
> (`part-funding-binance-…`), and SYMBOL travels in the body where it is the thing being
> distinguished rather than the thing being partitioned on. Compaction then merges the
> handful of parts an hour accumulates — one per poll per venue — into one per venue,
> and never touches the hour still open.
>
> **Which datasets, and why not all of them.** `funding` and `option_chain` are read
> whole-universe-for-a-window by every consumer, so a `symbol=` level buys nothing and
> costs 1,021 files an hour. `bars` and `book` are read per symbol, where that level is
> what makes the read cheap. The split is by access pattern, not by size.
>
> **No compaction module, and the measurement is why.** The intent here was a pass that
> merges a sealed hour's parts into one per venue. Counting the DISTINCT part names in a
> sealed hour first showed there is nothing to merge: funding's hour
> `2026-08-17T17` holds 1,892 fragments under **3 distinct names** — binance, bybit,
> hyperliquid — and option_chain's holds 1,436 under **one**. The fan-out is entirely
> the `symbol=` level, so converting the layout takes that hour to 3 files and 1 file
> respectively, and RL-032's target is met by the conversion alone.
>
> A merge pass would therefore be machinery for a problem that does not exist, which is
> what the counterweight in CLAUDE.md warns against: every automation is operational debt,
> and the question of whether it is worth maintaining comes before whether it is possible.
> **What would make it needed** is a build cadence that writes several snapshots per hour
> per venue — then `probe_sealed_hour_compacted` reports parts exceeding venues and says
> so, which is the trigger rather than a guess.
>
> **The conversion is `store/hourly_migration.py`, unchanged in shape.** Its building copy
> starts empty, so `partitions_by_hour_alone` sees a dataset with nothing to mix with and
> writes the converted layout; its existing verify-then-swap carries the rest. It verifies
> on content — rows per (hour, symbol), total rows, the column set — never on "the command
> exited 0", because this store has lost a column silently once already:
> `funding_interval_hours`, 2026-08-09, which turned a 4-hourly funding rate into an
> 8-hourly one.

> **What made the wall stop, measured 2026-08-19.** `status-wall.html` last completed on
> **2026-08-17 13:38** and the probe cache directory had never been created — which is the
> proof, because `measured_periodically` writes on the first success and no expensive probe
> had ever returned one. The board that reports whether the system is healthy had been frozen
> for two days while reading as if it were current, which is the Rule 8 failure it exists to
> prevent, wearing the costume of a working board.
>
> Two causes, and the second is the one that hurts.
>
> **The migration never finished for the datasets that needed it most.** `funding` holds 16
> `availability_hour=` directories beside **1,271 legacy `symbol=` ones**, and `option_chain`
> 16 beside **1,520**. `_is_partitioned_by_hour` disables hour pruning for a whole dataset
> while any top-level `symbol=` remains — deliberately, because a part-way dataset would
> otherwise drop every legacy row from a bounded read with no error. So SL-15's speed-up is
> switched off for the two largest datasets, and only `bars_60000000000ns` was ever migrated.
>
> **Per-symbol parts are too small to be worth opening.** One sealed funding hour holds
> **1,892 fragments for 26,015 rows and 16.2 MiB** — 13.8 rows and 8.8 KiB per file — and
> costs **29 ms per fragment** to open. The whole funding dataset therefore takes **30.6
> minutes** to read, for 850 MB, and it grows by ~1,021 files every hour. Column pushdown
> filters rows; **it cannot prune files**, which is the same lesson SL-14 recorded.
>
> **RL-032 settles the granularity: one part per hour per VENUE, not per symbol.** That hour
> becomes 3 parts instead of 1,892 — binance, bybit, hyperliquid. Per-symbol partitioning
> earns its keep only where reads are per-symbol, and nothing reads funding that way: every
> consumer wants the whole universe for a window. Bars and book keep their symbol layout for
> the opposite reason.
>
> **The symbol moves into the file body**, because it is no longer in the path to reconstruct
> it from. `append_partition` drops it today precisely because the path carried it, so this is
> forced by the layout rather than chosen — and it is the column this store has already lost
> once, silently, on 2026-08-09.

---

## SLICE bot-framework — SHARED LIVE BOT FRAMEWORK

**Written 2026-08-18 under RL-024 and RL-023.** slice-0 deliberately left the shared bot framework
out, to be planned when a segment slice needed it. All four segment slices need it at once, and two
rulings given on 2026-08-18 fixed its shape before it was built.

**RL-024 moved the market data path.** Every row below reads a LIVE venue feed. The parquet store
stays the research and training corpus and is never the trading clock. Measured 2026-08-18: the
running engine had been up 14 minutes without completing one poll, in uninterruptible IO, on a store
whose cold filtered scan measured 185.6s the day before; the raw tape behind that store flushes a
zstd frame every 30s. Neither is a live price, and a bot polling either is backtesting on a delay
while carrying the name paper trading.

**RL-023 fixed the brain count at three.** BULL, BEAR and PROFIT-TAIL. The arbiter is the selection
step that consumes all three, not a third brain. PROFIT-TAIL's expectancy and tail estimates are
inputs the arbiter consumes and never a veto; it owns entry timing and the whole position after fill;
it can neither reject a selected trade nor refuse to close a loser; the hard stop overrides it
absolutely.

### BF-01
  slice:      bot-framework
  does:       hold one live websocket feed per venue and hand each bot the ticks that arrived
              since its last poll, with the age of the newest tick
  satisfies:  RL-024 RL-018 RL-020
  sources:    ~/research/bull-bear-profit-agents-spec.md#2. Authority map
  depends on: none
  probe:      probe_live_feed_fresh
  accepts:    a bot's poll returns ticks whose newest is seconds old rather than hours, a feed
              that has gone quiet is reported as quiet rather than as an empty market, and no
              bot ever reads a price from the parquet store
  state:      measured by probe_live_feed_fresh

### BF-02
  slice:      bot-framework
  does:       compute the live feature frame per symbol from the ticks a poll delivered
  satisfies:  RL-010 RL-013 RL-024
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01
  probe:      probe_segment_features_current
  accepts:    every feature row names the tick window it was computed from, and a missing or
              stale input produces a refusal naming what was missing rather than a default
  state:      measured by probe_segment_features_current

### BF-03
  slice:      bot-framework
  does:       state what a BULL or BEAR brain is, so a rule brain and a trained model are the
              same interface to everything downstream
  satisfies:  RL-023 RL-025 RL-013 RL-011
  sources:    ~/research/bull-bear-profit-agents-spec.md#1. The three bots
  depends on: BF-02
  probe:      probe_segment_brains_reason
  accepts:    a brain returns a proposal carrying the features that produced it or a first-class
              decline, a BULL can never propose a short, a BEAR can never propose a long, and
              swapping a rule brain for a model changes no caller
  state:      measured by probe_segment_brains_reason

### BF-04
  slice:      bot-framework
  does:       own entry timing and the whole position after fill, without the power to refuse a
              selected trade or to refuse to close a loser
  satisfies:  RL-023 RL-011 RL-022
  sources:    ~/research/bull-bear-profit-agents-spec.md#4. PROFIT-TAIL in detail
  depends on: BF-02
  probe:      probe_profit_tail_authority
  accepts:    a test proves it cannot reject a selected trade, a test proves it cannot hold a
              position past the hard stop, a trade abandoned on signal expiry is journalled as a
              missed entry attributed to it, and every order it raises re-enters the risk gate
  state:      measured by probe_profit_tail_authority

### BF-05
  slice:      bot-framework
  does:       select the trade from the BULL and BEAR proposals with PROFIT-TAIL's expectancy as
              an input, yielding one side or an abstention
  satisfies:  RL-023 RL-011 RL-006
  sources:    ~/research/bull-bear-profit-agents-spec.md#2. Authority map
  depends on: BF-03 BF-04
  probe:      probe_segment_arbiter_decides
  accepts:    two brains proposing opposite sides yields one side or an abstention and never
              both, PROFIT-TAIL's numbers change the selection but can never veto it, and the
              rejected case is journalled with its reason
  state:      measured by probe_segment_arbiter_decides

### BF-06
  slice:      bot-framework
  does:       run one segment bot end to end on the live feed - features, three brains,
              selection, timing, risk gate, broker, journal, heartbeat
  satisfies:  RL-024 RL-019 RL-005 RL-020
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-05
  probe:      probe_segment_engine_running
  accepts:    the bot opens and closes positions on live prices, every fill is journalled under
              its own segment, and its heartbeat carries the age of the newest tick it acted on
  state:      measured by probe_segment_engine_running

### BF-07
  slice:      bot-framework
  does:       declare each segment bot - its venue, its streams, its feature set and its three
              brains - so a segment's architecture is stated in one place
  satisfies:  RL-019 RL-006 RL-023
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-06
  probe:      probe_segment_engine_running
  accepts:    all four segments are declared, each names its own feed and its own features, and
              no segment inherits another's brains by default
  state:      measured by probe_segment_engine_running

### BF-08
  slice:      bot-framework
  does:       show every segment bot's measured state on the wall, absence rendering as its own
              state and the rule brains carrying their no-edge-claim label
  satisfies:  RL-012 RL-025 RL-008
  sources:    2026-08-17-ajit-master-plan-design.md#8. Progress reporting
  depends on: BF-06
  probe:      probe_segment_tiles_measured
  accepts:    no tile is green without a probe having run, a bot that has never traded renders
              NOT MEASURED rather than blank or green, and every tile running a rule brain says
              so on its face
  state:      measured by probe_segment_tiles_measured

### BF-09
  slice:      bot-framework
  does:       enumerate every instrument each venue lists, so a segment's universe is the
              venue's own and never a list somebody typed
  satisfies:  RL-009 RL-014 RL-019 RL-024
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01
  probe:      probe_universe_is_the_venues
  accepts:    each segment's instrument count comes from a venue listing call rather than a
              constant, the count and what was dropped are published with the reason, and a
              venue that lists nothing reads as a failed discovery rather than an empty market
  state:      measured by probe_universe_is_the_venues

> **Builds `live/universe_discovery.py`. Measured 2026-08-18, and the numbers are the row's
> justification.** The bots started on 13 hand-typed symbols. The venues list:
>
> | segment | typed | listed |
> |---|---|---|
> | perp | 13 | **570** binance perpetuals |
> | spot | 13 | **1,361** binance spot pairs |
> | dated | 40 | **48** — 4 binance quarterlies, 40 bybit linear, 4 bybit inverse |
> | options | 786 | **1,442** — Deribit BTC 786 and ETH 656; SOL and XRP list zero |
>
> A segment whose universe is a typed list is a segment whose universe is whatever somebody
> remembered.
>
> **The dated count read 217 until a delivery date was found to mean nothing.** 169 binance
> `TRADIFI_PERPETUAL` contracts — Tesla, Intel, gold, silver, Korean and Hong Kong equities —
> carry a delivery date of the year 2100 and pass any "is it dated" test phrased as an
> exclusion. They are perpetuals with no basis to converge, and they are not crypto. **The
> wrong count was the LARGER one**, which is why this row's acceptance requires that what was
> dropped is published with its reason rather than only what was kept.
>
> **Subscription shape, and why it is not one stream per symbol.** Quotes come from the
> all-market `!bookTicker` stream and funding from `!markPrice@arr@1s` — one stream each,
> covering every symbol the venue lists, so breadth costs no extra connections. Only the
> per-symbol `@trade` streams are sharded, because there is no all-market trade stream and
> the aggressor side cannot be recovered without one. `capture.venues.shard_by_url_budget`
> already exists for that split and is reused rather than re-derived.

### BF-10
  slice:      bot-framework
  does:       implement the probes the framework and learned-brain rows name, so their
              state is measured rather than permanently unmeasured
  satisfies:  RL-021 RL-012 RL-008 RL-004
  sources:    2026-08-17-ajit-master-plan-design.md#8. Progress reporting
  depends on: BF-06 BF-08
  probe:      probe_probes_implemented
  accepts:    every probe a row names either exists and measures the running system, or is
              reported as unimplemented rather than rendering as passed, and no probe here
              reports whether a test passed instead of what is true on this box now
  state:      measured by probe_probes_implemented

> **Builds `statuswall/segment_probes.py`, and the audit that produced this row is the
> point of it.** Measured 2026-08-18: the spine's rows named **43 distinct probes and 29
> of them had no implementation anywhere**. Everything the bot framework and the learned
> brains had built was running, journalling and trading — and not one of those rows could
> light up, because `ruling_conformance` renders NOT MEASURED for a probe that is named
> and absent.
>
> That is the plan working, not a reporting bug. A row whose probe does not exist has not
> been measured, and Rule 8 says an unmeasured thing renders unmeasured. What was missing
> was the measurement, so this row supplies it.
>
> **Kept out of `statuswall/ruling_conformance.py` deliberately.** That module is imported
> by the boards generator, which measured 4.8 GB resident on 2026-08-17 and 11.7 GB on
> 2026-08-18. Probes that open model registries and journals have no business inflating
> it; they live in their own module and register by name.
>
> **Every journal read here is a bounded tail seek.** The options bot wrote 4.1 GB of
> decisions on 2026-08-18 alone. A probe that read one end to end would cost more each day
> it ran and would eventually take the boards process down with it, which is the failure
> immediately below.
>
> **What `probe_board_generated` found on its first run, 2026-08-19.** One of five boards
> was under six hours old: the plan board was **43 hours** stale and the segment wall
> **15**, while the bots they describe were trading. Cause was in
> `scripts/boards_supervisor.sh`: the generator ran the expensive feature wall FIRST and
> `break`-ed its own loop when that wall failed, so an OOM-killed wall stopped the two
> cheap boards from ever regenerating - directly under a comment saying failures are
> "recorded and retried rather than fatal". The cheap boards now run first and nothing in
> that loop is fatal. **A board that silently stops updating is the Rule 8 failure the
> whole supervisor exists to prevent, and only a probe measuring the board's own age
> could see it.**

### BF-11
  slice:      bot-framework
  does:       account, in USDT, for the capital each bot used and the profit or loss it made
              on it - peak at risk, turnover, and equity against a declared bankroll
  satisfies:  RL-028 RL-029 RL-005 RL-012
  sources:    docs/rulings.json#RL-028
  depends on: BF-06
  probe:      probe_capital_and_pnl_reported
  accepts:    every bot reports all three denominators or none of them, a fill priced in a
              currency other than USDT is converted at a rate journalled with that fill or
              is reported as unconvertible rather than converted at an unrecorded rate, and
              no return is published without the capital base it was computed against
  state:      measured by probe_capital_and_pnl_reported

> **Why three numbers rather than one, in the user's own decision (RL-028).** "Return" has
> no meaning until the denominator is named, and the three available denominators differ by
> orders of magnitude on the same trades: **peak concurrent notional** answers what the bot
> needed, **cumulative turnover** answers what it traded through, and **equity against a
> declared bankroll** answers what an account holder would have seen. A tile publishing one
> of them alone is publishing whichever flatters. So the row's acceptance is all three or
> none.
>
> **The unit trap this row exists to avoid (RL-029).** Deribit options are quoted in BTC.
> Three of the four bots are already in USDT and summing the fourth into a total without
> conversion is a unit error that reads as a result - the same shape as the $837 loss in
> `DECISIONS.md`, where one number was credited to 36 features. The conversion rate is
> journalled ON THE FILL, so every converted figure can be audited back to the price it was
> converted at, and a fill that carries no rate is reported as unconvertible rather than
> converted at a rate nobody saw.
>
> **The bankroll is DECLARED, not inferred.** A bot's bankroll is part of its declaration in
> `segment/bot_registry.py`, like its venue and its brains, because inferring it from the
> largest position the bot happened to take would make the denominator move with the
> numerator - a return that rises when the bot gets luckier about sizing.

### BF-12
  slice:      bot-framework
  does:       show each bot's capital and P&L on the wall, in USDT, with the conversion rate
              and its age where one was used
  satisfies:  RL-028 RL-029 RL-012 RL-008
  sources:    docs/rulings.json#RL-028
  depends on: BF-11 BF-08
  probe:      probe_capital_and_pnl_reported
  accepts:    the tile carries peak at risk, turnover and equity against bankroll, a bot with
              no closed trade reads NOT MEASURED rather than 0.0%, and a converted figure
              names the rate it was converted at
  state:      measured by probe_capital_and_pnl_reported

> **The modules these rows build.** Named here so the row and the file cannot drift apart, and so
> `require-plan-row.sh` admits them:
>
> | row | module |
> |---|---|
> | BF-01 | `live/live_feed.py` |
> | BF-02 | `segment/live_features.py` |
> | BF-03 | `segment/brain.py` |
> | BF-04 | `segment/profit_tail.py` |
> | BF-05 | `segment/arbiter.py` |
> | BF-06 | `segment/live_engine.py` |
> | BF-07 | `segment/bot_registry.py` |
> | BF-08 | `statuswall/segment_tiles.py` |
> | BF-09 | `live/universe_discovery.py` |
> | BF-10 | `statuswall/segment_probes.py` |
> | BF-11 | `segment/capital_accounting.py` |
> | BF-12 | `statuswall/segment_tiles.py` (the capital block) |
> | SB-01 | `spot/tradable_universe.py` |
> | SB-02 | `spot/segment_brains.py` |
> | DB-01 | `dated/tradable_universe.py` |
> | DB-02 | `dated/segment_brains.py` |
> | OB-01 | `options/tradable_universe.py` |
> | OB-02 | `options/segment_brains.py` |
> | PB-15 | `perp/segment_brains.py` |
>
> The per-segment brain modules share a file NAME and nothing else. RL-019 makes each segment its
> own bot with its own features, and the four files differ in every input they read: the perp
> brains read funding and order flow, the spot brains cross-venue divergence, the dated brains term
> structure and time to expiry, the options brains the quoted chain. A shared name is a convention
> for finding them; it is not a shared implementation, and a segment inheriting another's brains by
> default is what BF-07's acceptance refuses.

---

## SLICE spot-bot — SPOT BOT

*Row inventory pending the reconciliation sweep, reviewed with the user before this slice starts
(design §12). An empty slice reports 0/0, which is the honest board for unplanned work.*

**Rows written 2026-08-18 under RL-024 and RL-023.** The bot trades a live venue feed; the
remaining rows of this slice are still pending review.

### SB-01
  slice:      spot-bot
  does:       name every spot symbol the bot may trade and every one it may not, with the reason
  satisfies:  RL-014 RL-009 RL-019 RL-018
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01
  probe:      probe_spot_universe_measured
  accepts:    every symbol the live spot feed carries resolves to tradable or excluded with its
              reason named, and the excluded count is published rather than hidden
  state:      measured by probe_spot_universe_measured

### SB-02
  slice:      spot-bot
  does:       carry the spot BULL, BEAR and PROFIT-TAIL brains on spot's own features -
              cross-venue divergence and consolidated price rather than funding
  satisfies:  RL-023 RL-019 RL-025 RL-011
  sources:    ~/research/bull-bear-profit-agents-spec.md#1. The three bots
  depends on: BF-03 BF-04 SB-01
  probe:      probe_segment_brains_reason
  accepts:    the spot brains name spot features, no brain reads a perpetual-only input, and
              each decision carries the features that produced it
  state:      measured by probe_segment_brains_reason


Known shape from the design: segment data completion → spot features → BULL, BEAR and PROFIT-TAIL
brains → segment risk gate → engine, journal, supervisor and on/off switch → board tile → three
axis verdicts per brain → reconciliation sweep. The universe-wide opportunity monitor (RL-009,
RL-014) is a row here and in each of the three slices below; it is `per-segment` and reads **0/4**
until all four exist.

---

## SLICE perp-bot — PERP BOT

**Inventory reviewed with the user 2026-08-17 (RL-022).** This is the first segment bot built end
to end. Its edge is **directional scalping**. Two exit horizons are built and run side by side on
the same entry signal — a fast band of seconds to minutes and a slow band of five to sixty minutes
— and measured winrate and net profit decide which survives. Market making, mean reversion and
funding/basis carry were all chosen as wanted and are carried below as BLOCKED rows so they are
revisited rather than forgotten.

The running `plumbing-momentum` engine retires into this slice: its journal is archived and marked
as plumbing that made no edge claim, so it can never later be read as a result. Until then it keeps
running, because something trading is what the 24/7 record needs.

**Why the perp segment first.** It has the deepest captured history across three venues, the best
liquidity, no borrow to arrange for the bear side, and funding as a second observable. It is also
where the paper harness already runs, so the engine, journal, fill model and equity curve are
exercised code rather than new code.

**What this slice does NOT assume.** No row below says which model or method an agent uses. That is
settled per row against §1a's intelligence standard when the row is specced, not here — a plan that
picked the method in advance would be the hard-coded instruction-following RL-013 refuses.

### PB-01
  slice:      perp-bot
  does:       name every perp symbol whose bars, book and trades are complete enough to trade on,
              and name every one that is not, with the reason
  satisfies:  RL-014 RL-009 RL-019 RL-018
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: none
  probe:      probe_perp_universe_measured
  accepts:    every captured perp symbol resolves to tradable or excluded with its reason
              named, and the excluded count is published rather than hidden
  state:      measured by probe_perp_universe_measured

> **Builds one module, `perp/tradable_universe.py`, on top of
> `features/universe_coverage.py` rather than beside it.** That module already establishes the
> segment from *which dataset carries the key* — funding makes a key a perpetual — and already
> measures staleness against each series' own cadence rather than a shared constant. It is
> deliberately a **watch list**: a symbol that went quiet is its most interesting row. This row is
> the opposite decision — **admission to trading** — and the two must not be merged, because a
> watch list that drops what it stopped seeing cannot answer "is that instrument gone, or did our
> feed stop".
>
> **The measured constraint this row exists to surface.** Bars cover 2,235 symbols; the `book`
> dataset covers **six** — BTCUSDT, ETHUSDT, SOLUSDT, BTC-USD, ETH-USD, SOL-USD — and only from
> 2026-08-17T09. The capture configuration is `binance BTCUSDT,ETHUSDT,SOLUSDT ALL`: three named
> symbols get depth, `ALL` gets bars. Order-flow imbalance, microprice and absorption all need the
> book, so **RL-009/RL-014 breadth and book-based scalping cannot both hold today**. Admission
> therefore carries a tier — DEEP where the book exists, BARS-ONLY otherwise — so PB-02 and PB-03
> are specced against what is actually there rather than discovering it halfway through.

### PB-02
  slice:      perp-bot
  does:       compute the scalping feature frame per tradable symbol per bar from the feature
              modules that already exist and today feed nothing
  satisfies:  RL-010 RL-013 RL-018 RL-019
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-01
  probe:      probe_perp_features_current
  accepts:    every tradable symbol carries a feature row no older than one bar, and a missing
              or stale input produces a refusal naming what was missing rather than a default
  state:      measured by probe_perp_features_current

### PB-03
  slice:      perp-bot
  does:       propose long entries across the whole tradable universe, each with a confidence and
              the evidence that produced it
  satisfies:  RL-011 RL-013 RL-010 RL-006 RL-009
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: PB-02
  probe:      probe_perp_bull_agent_reasons
  accepts:    every proposal names the features that produced it, the same features at a later
              date can produce a different decision, and declining a symbol is a first-class
              outcome rather than the absence of one
  state:      measured by probe_perp_bull_agent_reasons

### PB-04
  slice:      perp-bot
  does:       propose short entries across the whole tradable universe on the same terms
  satisfies:  RL-011 RL-013 RL-010 RL-006 RL-009
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: PB-02
  probe:      probe_perp_bear_agent_reasons
  accepts:    as PB-03, and the bear agent is a separate decision rather than the bull agent's
              output negated
  state:      measured by probe_perp_bear_agent_reasons

### PB-05
  slice:      perp-bot
  does:       select the trade, or no trade at all, from the bull and bear proposals for a symbol,
              with PROFIT-TAIL's expectancy and tail estimates as inputs rather than as a veto
  satisfies:  RL-011 RL-006 RL-019 RL-023
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: PB-03 PB-04
  probe:      probe_perp_arbiter_decides
  accepts:    two agents proposing opposite sides on one symbol yields one side or an
              abstention, never both, the rejected case is journalled with its reason, and
              PROFIT-TAIL's numbers can change the selection but never refuse it
  state:      measured by probe_perp_arbiter_decides

### PB-06
  slice:      perp-bot
  does:       run the fast and the slow exit band on the same entry signal, journalled apart
  satisfies:  RL-022 RL-018 RL-005
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-05
  probe:      probe_perp_bands_journalled
  accepts:    each band's trades are separable in the journal, one trade never appears in both,
              and each band's exit rule is stated as a rule rather than a tuned constant
  state:      measured by probe_perp_bands_journalled

### PB-07
  slice:      perp-bot
  does:       refuse an order that breaches perp exposure, liquidation distance or entry funding cost
  satisfies:  RL-006 RL-019 RL-004
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-05
  probe:      probe_perp_risk_gate_refuses
  accepts:    every limit has a test that trips it, a refusal names the limit and the measured
              value that breached it, and no order reaches the broker without passing
  state:      measured by probe_perp_risk_gate_refuses

### PB-08
  slice:      perp-bot
  does:       run the perp bot 24/7 under its own supervisor with an on/off switch, journalling
              every fill
  satisfies:  RL-020 RL-005 RL-019 RL-006
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-06 PB-07
  probe:      probe_perp_engine_running
  accepts:    the bot returns after a reboot with nothing started by hand, the switch stops it
              without stopping capture, and its heartbeat carries what the poll cost
  state:      measured by probe_perp_engine_running

### PB-09
  slice:      perp-bot
  does:       show the perp bot's measured state on the wall, absence rendering as its own state
  satisfies:  RL-012 RL-004
  sources:    2026-08-17-ajit-master-plan-design.md#8. Progress reporting
  depends on: PB-08
  probe:      probe_perp_tile_measured
  accepts:    no tile is green without a probe having run, and a bot that has never traded
              renders NOT MEASURED rather than blank or green
  state:      measured by probe_perp_tile_measured

### PB-10
  slice:      perp-bot
  does:       measure trades, winrate and net profit per exit band from the journal alone
  satisfies:  RL-022 RL-005 RL-012
  sources:    2026-08-17-ajit-master-plan-design.md#8. Progress reporting
  depends on: PB-08
  probe:      probe_perp_band_performance
  accepts:    each band reports its trade count, winrate and profit net of fees and modelled
              slippage, and a band with too few trades to support a rate says so instead of
              reporting one
  state:      measured by probe_perp_band_performance

### PB-11
  slice:      perp-bot
  does:       decide whether a perp strategy has earned real money, on observed history only
  satisfies:  RL-005 RL-004
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-10
  probe:      probe_perp_promotion_gate
  accepts:    a strategy passes only on out-of-sample observed history, deflated by the number
              of trials it was actually selected from, and the gate refuses outright on
              reconstructed data
  state:      measured by probe_perp_promotion_gate

### PB-15
  slice:      perp-bot
  does:       own perp entry timing and the whole position after fill, on the segment's own
              horizon bands
  satisfies:  RL-023 RL-022 RL-011 RL-019
  sources:    ~/research/bull-bear-profit-agents-spec.md#4. PROFIT-TAIL in detail
  depends on: PB-02
  probe:      probe_profit_tail_authority
  accepts:    it cannot reject a selected perp trade and cannot hold one past the hard stop, a
              trade abandoned on signal expiry is journalled as a missed entry attributed to it,
              and its fast and slow bands are the two PB-06 journals rather than a third
  state:      measured by probe_profit_tail_authority

> **This row is the RL-023 correction, and it was a real gap.** PB-03 named the bull, PB-04 the
> bear and PB-05 an arbiter over the two - a two-brain design with a chooser, which is exactly the
> shape `dual-agent-spec.md` was superseded for on 2026-08-03. The user caught it on 2026-08-18.
> Three brains: BULL, BEAR, PROFIT-TAIL, and the arbiter is the selection step, not a brain.

### PB-16
  slice:      perp-bot
  does:       move the perp bot's market data path from the parquet store to the live feed
  satisfies:  RL-024 RL-018 RL-020
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01 BF-06
  probe:      probe_live_feed_fresh
  accepts:    the perp bot acts on ticks seconds old rather than hours, and no price it trades on
              is ever read from the store
  state:      measured by probe_live_feed_fresh

> **The three rows below are chosen, not declined.** The user selected market making, mean
> reversion and funding/basis carry alongside directional scalping (RL-022), then chose to run
> scalping first and judge it on measured winrate and profit before adding the rest. They are
> written as rows so the choice is revisited rather than remembered. Each is BLOCKED on PB-10
> returning a verdict — which is a decision about sequence, not a claim that they are hard.

### PB-12
  slice:      perp-bot
  does:       rest quotes on both sides of the perp book and earn the spread, managing inventory
  satisfies:  RL-022 RL-006
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-10
  probe:      probe_perp_band_performance
  accepts:    not specced. Before this row is built, the fill model must be shown to model queue
              position, because a market maker judged on a fill model that assumes fills is a
              market maker judged on nothing
  state:      BLOCKED

### PB-13
  slice:      perp-bot
  does:       fade short-horizon overextension against a cross-venue reference price
  satisfies:  RL-022 RL-006
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-10
  probe:      probe_perp_band_performance
  accepts:    not specced. `features.price_divergence` and `features.consolidated_price` already
              exist and are the intended inputs
  state:      BLOCKED

### PB-14
  slice:      perp-bot
  does:       capture funding and spot-perp basis as a held position rather than a scalp
  satisfies:  RL-022 RL-006 RL-018
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: PB-10
  probe:      probe_perp_band_performance
  accepts:    not specced. `strategy/funding_carry.py` already exists at 482 lines and is the
              only non-plumbing strategy file in the repo. Its holding period is not intraday,
              so RL-018 has to be reconciled with it before this row is built
  state:      BLOCKED

---

## SLICE dated-bot — DATED FUTURES BOT

*Row inventory pending review.* `bybit` carries 48 dated contracts and `features.term_structure`
already reads them.

**Rows written 2026-08-18 under RL-024 and RL-023.** The bot trades a live venue feed; the
remaining rows of this slice are still pending review.

### DB-01
  slice:      dated-bot
  does:       name every dated futures contract the bot may trade and every one it may not,
              with the reason and the contract's expiry
  satisfies:  RL-014 RL-009 RL-019 RL-018
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01
  probe:      probe_dated_universe_measured
  accepts:    every dated contract the live feed carries resolves to tradable or excluded with
              its reason named, and a contract inside its final settlement window is excluded
              with that as the reason
  state:      measured by probe_dated_universe_measured

### DB-02
  slice:      dated-bot
  does:       carry the dated BULL, BEAR and PROFIT-TAIL brains on term structure and basis
              rather than on spot momentum
  satisfies:  RL-023 RL-019 RL-025 RL-011
  sources:    ~/research/bull-bear-profit-agents-spec.md#1. The three bots
  depends on: BF-03 BF-04 DB-01
  probe:      probe_segment_brains_reason
  accepts:    the dated brains name term-structure features, and time to expiry is an input to
              every decision rather than an afterthought
  state:      measured by probe_segment_brains_reason


---

## SLICE options-bot — OPTIONS BOT

*Row inventory pending review.* Less blocked than the record says: `store/option_chain` exists with
305 parquet files across per-instrument partitions and `option_chain deribit` is in the store
supervisor's rotation, so goal §3a item 2's *"no builder reads it yet"* is out of date. The chain
cannot be backfilled — the endpoint ignores a `timestamp` parameter — so every hour of capture from
2026-08-16 onward is the only history this segment will ever have.

**Rows written 2026-08-18 under RL-024 and RL-023.** The bot trades a live venue feed; the
remaining rows of this slice are still pending review.

### OB-01
  slice:      options-bot
  does:       name every option instrument the bot may trade and every one it may not, with the
              reason, its expiry and its moneyness
  satisfies:  RL-014 RL-009 RL-019 RL-006
  sources:    2026-08-17-ajit-master-plan-design.md#2. The slices — vertical, one bot at a time
  depends on: BF-01
  probe:      probe_options_universe_measured
  accepts:    every instrument the live chain carries resolves to tradable or excluded with its
              reason named, an instrument with no two-sided quote is excluded as such, and the
              thinness of the captured history is published rather than hidden
  state:      measured by probe_options_universe_measured

### OB-02
  slice:      options-bot
  does:       carry the options BULL, BEAR and PROFIT-TAIL brains on the chain's own inputs, an
              option never being taken as a plain directional bet
  satisfies:  RL-023 RL-019 RL-025 RL-011 RL-006
  sources:    ~/research/bull-bear-profit-agents-spec.md#1. The three bots
  depends on: BF-03 BF-04 OB-01
  probe:      probe_segment_brains_reason
  accepts:    the options brains read the quoted chain, a decision names the instrument's expiry
              and moneyness, and the spec's rule that options are not a directional bet has a
              test that trips on violating it
  state:      measured by probe_segment_brains_reason


---

## SLICE learned-brains — REAL LEARNED BRAINS, AND WHAT ATTACHES TO THEM

**Written 2026-08-18 under RL-026 and RL-027.** RL-025's rule brains were scaffolding with an
honest label and this slice replaces them. The standard every row here is measured against is
**§1a of the goal document** — three axes, judged separately, with the master test governing all
of them: *does it change what the system does when it is wrong?*

**The honest boundary, stated first so this slice does not become the overclaim §1a exists to
prevent.** §1a.0: no architecture available in 2026 produces understanding, and six frontier
models given $10k each on Hyperliquid perps lost 30–63% in 17 days. What crosses the distance
between a script and an adapting system is not a smarter model — it is beliefs that carry
provenance and expire, calibrated knowledge of its own competence, and learning from its own
history as data. That is what this slice builds.

**The data constraint, measured 2026-08-18 and binding on every row below.** The store holds
**67 hour-partitions across 8 non-contiguous days** (2026-08-02, 03, 08, 09, 15, 16, 17, 18) —
roughly 860,000 bar rows over 1,698 symbols, about 500 bars per symbol. Two consequences, neither
negotiable: the model is **pooled cross-sectional** rather than per-symbol, because 500 bars
cannot fit a symbol; and **§1a L6 (out-of-regime stress) cannot pass**, because eight days of one
regime is not a regime change. L6 is recorded as FAILING rather than skipped.

### LB-01
  slice:      learned-brains
  does:       build the pooled cross-sectional training set from the store - features as of a bar,
              triple-barrier outcomes after it, and the label span each one occupied
  satisfies:  RL-026 RL-013 RL-010
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: none
  probe:      probe_training_set_built
  accepts:    every row's features are computable from information available at its own bar and no
              later, each row carries the label span its outcome occupied so uniqueness weighting
              is possible, and the row count and day coverage are published rather than implied
  state:      measured by probe_training_set_built

### LB-02
  slice:      learned-brains
  does:       fit the segment model under purged cross-validation, count the trial, and register the
              artefact with the loss that produced it
  satisfies:  RL-026 RL-013 RL-004
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: LB-01
  probe:      probe_model_registered_with_loss
  accepts:    every registered model names the trial that counted it and the out-of-fold score it
              earned, a fit that only reproduced the majority class is reported as such rather than
              as an accuracy, and no model reaches a brain without passing through the registry
  state:      measured by probe_model_registered_with_loss

### LB-03
  slice:      learned-brains
  does:       carry a claim with its provenance, its epistemic class and its half-life, so future
              intelligence attaches here rather than to the engine
  satisfies:  RL-027 RL-013 RL-010
  sources:    2026-08-08-final-project-goal-design.md#1a.6
  depends on: none
  probe:      probe_beliefs_carry_provenance
  accepts:    every belief names what produced it and when it expires, an expired belief cannot
              size a position, only an OBSERVED belief may size one at all, and a belief with no
              provenance cannot be constructed
  state:      measured by probe_beliefs_carry_provenance

### LB-04
  slice:      learned-brains
  does:       decide BULL and BEAR from the registered model, emitting a calibrated belief or an
              abstention with a coverage guarantee behind it
  satisfies:  RL-026 RL-023 RL-013 RL-011
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: LB-02 LB-03
  probe:      probe_brains_are_learned
  accepts:    no threshold in the decision path is a typed constant, an abstention is backed by a
              conformal quantile rather than a verbalised hedge, and swapping the trained
              parameters for random ones measurably changes the decisions
  state:      measured by probe_brains_are_learned

### LB-05
  slice:      learned-brains
  does:       update calibration and the abstention quantile inside the live loop from realised
              outcomes
  satisfies:  RL-026 RL-013 RL-010
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: LB-04
  probe:      probe_calibration_updates_live
  accepts:    a parameter the live loop itself changed is distinguishable from one the retrainer
              set, realised coverage is measured against the promised bound rather than assumed,
              and the two are labelled apart on the board
  state:      measured by probe_calibration_updates_live

### LB-06
  slice:      learned-brains
  does:       forecast the distribution of forward P&L rather than a point estimate, so entry
              timing and position management are learned
  satisfies:  RL-026 RL-023 RL-022
  sources:    ~/research/bull-bear-profit-agents-spec.md#4. PROFIT-TAIL in detail
  depends on: LB-02 LB-03
  probe:      probe_profit_tail_is_learned
  accepts:    expectancy and loss tail come from fitted quantiles rather than from a volatility
              proxy, PROFIT-TAIL's authority limits are unchanged by becoming learned, and the
              deterministic policy remains the baseline it must beat
  state:      measured by probe_profit_tail_is_learned

### LB-07
  slice:      learned-brains
  does:       run the §1a axis probes against the live brains - randomisation, ablation, label
              invariance and realised coverage
  satisfies:  RL-026 RL-013 RL-012 RL-004
  sources:    2026-08-08-final-project-goal-design.md#1a.2
  depends on: LB-04 LB-05
  probe:      probe_axis_tests_run
  accepts:    L3 randomisation and L4 ablation both change measured behaviour or the brain is
              reported as decorative, R9 label invariance holds when a feature is renamed, and
              L6 out-of-regime is reported as FAILING rather than omitted while the record is
              eight days of one regime
  state:      measured by probe_axis_tests_run

### LB-08
  slice:      learned-brains
  does:       refit on a cadence without a human invoking it, and publish which model each bot is
              running
  satisfies:  RL-026 RL-020 RL-012
  sources:    2026-08-08-final-project-goal-design.md#1a
  depends on: LB-02 LB-04
  probe:      probe_retrainer_running
  accepts:    a refit happens with nothing started by hand, the board names the model version and
              the out-of-fold score each bot is deciding on, and a bot running no model reads
              NOT MEASURED rather than green
  state:      measured by probe_retrainer_running

> **What a learned brain costs at every restart, measured 2026-08-19.** The perp bot swapped to
> its trained champion and then declined **152,334 times across 319 polls without one proposal**.
> Nothing was wrong: `compute_features` is fitted on exactly 60 sealed one-minute bars, so a
> learned brain cannot produce a vector at all until it has watched 60 minutes of live tape, and
> `segment.live_features` holds that state IN PROCESS - a restart discards it (RL-024 forbids
> priming from the store, which is the alternative). **On the board this was `0 proposals`, which
> is indistinguishable from a model whose threshold is never crossed.** The heartbeat now carries
> the bar count against the window and the minutes remaining, the tile renders WARMING UP, and
> `probe_brains_are_learned` counts a warming bot apart from a deciding one. LB-09's reload is
> what would remove the cost rather than only reporting it.

### LB-09
  slice:      learned-brains
  does:       make a bot decide from the champion that is registered NOW, not the one that
              was registered when its process started
  satisfies:  RL-030 RL-026 RL-020 RL-012
  sources:    docs/rulings.json#RL-030
  depends on: LB-08
  probe:      probe_champion_reload_current
  accepts:    a champion registered while a bot is running becomes the model it decides from
              without anybody restarting anything, the swap is journalled with both version
              ids, a position already open is managed to its close by the brains that opened
              it, and a bot deciding from a superseded model reads as superseded rather
              than as learned
  state:      measured by probe_champion_reload_current

> **Found by measurement on 2026-08-19, which is the point of BF-10.** `bot_registry`
> resolves the champion once, inside `segment_bot()`, so the swap happens at process start
> and never again. The retrainer refits every four hours (LB-08) and the bots run 24/7
> (RL-020), so **every model fitted between two restarts is registered and never used** -
> and nothing said so, because the heartbeat published the version the bot loaded rather
> than the version that exists.
>
> The probe is implemented ahead of the reload deliberately: it compares each bot's running
> `model_version` against the registry's current alias, so the gap is on the board while the
> reload is still unbuilt. **A row nobody can see is how this one survived.**
>
> **The design question is settled: RL-030, 2026-08-19.** A reload applies to NEW ENTRIES
> ONLY - a position already open is managed to its close by the brains that opened it - and
> the bot checks the champion alias on every poll, loading a model only when the version id
> actually changed. The user chose both. **A trade opened by one model and closed by another
> is attributable to neither**, which is the same failure as the live system in
> `DECISIONS.md` that credited one P&L to all 36 of its features; every fill already carries
> its brains by name and this keeps that record true.
>
> A champion that fails to load, or that the provenance check refuses, leaves the bot trading
> on the model it already had and journals `CHAMPION_SWAP_REFUSED`. Trading on the previous
> model is correct there; stopping is not.

> **The modules these rows build.**
>
> | row | module |
> |---|---|
> | LB-01 | `learn/training_set.py` |
> | LB-02 | `learn/train_segment_model.py` |
> | LB-03 | `learn/belief.py` |
> | LB-04 | `learn/learned_brains.py` |
> | LB-05 | `learn/online_calibration.py` |
> | LB-06 | `learn/learned_brains.py` (the PROFIT-TAIL half) |
> | LB-07 | `learn/axis_probes.py` |
> | LB-08 | `scripts/retrain_supervisor.sh` |
> | LB-09 | `segment/bot_registry.py` (the reload), `statuswall/segment_probes.py` (the gap) |
>
> **What RL-027 makes this slice responsible for.** The user's words: *"lots of futer featues and
> intelliences learnin will be connectin to tis"*. So `learn/belief.py` is deliberately the widest
> point of the design and the narrowest interface: a capability that produces a claim attaches as a
> belief SOURCE, one that challenges a claim attaches as a belief CRITIC, and neither touches the
> engine, the arbiter or any segment bot. The ten capabilities §1a.6 orders first — provenance and
> half-life, the read/verified/observed classes, verification before ingestion, abstention with its
> P&L measured, calibration scoring, own-footprint attribution — are all belief-shaped, which is why
> that is the carrier rather than a model output.

---

## SLICE slice-5 — ALLOCATOR, PORTFOLIO RISK, BRAINS 2-3

*Row inventory pending review.* Nothing here has anything to allocate between until at least two
bots reach BUILT.
