# SL-15 — hourly store partitioning, and the migration that proves nothing was lost

**Row:** `docs/AJIT-MASTER-PLAN.md` SL-15 · **depends on** SL-14
**Measured cause:** `~/research/DECISIONS.md` §15.5
**Decision taken by the user 2026-08-17:** hour grain, migrate the existing archive, compact sealed
hours, and keep the old layout until the new one verifies row for row.

## 0. The measured problem, in one table

Live bars dataset, 2026-08-17 11:50, **52,487 fragments across 2,230 symbol partitions, 538 MB**:

| part of a poll | cold | warm |
|---|---|---|
| dataset discovery | 0.6s | 0.6s |
| schema walk, every fragment footer | 8.3s | — |
| schema walk, cached (SL-14) | — | 8 footers, 1.7s |
| scan filtered to match NOTHING | **185.6s** | 8.8s |

The scan is the cost, and `not_before_ns` cannot reduce it: the partition key is `symbol`, so
availability is not in the path and pyarrow must open every fragment to discover that none of its
rows qualify. **A filter that cannot be answered from the path is answered from the file.**

538 MB / 52,487 = **10 KB per file**, ~6,000 new files a day. So date grain is not enough: an
evening poll under a date-grained path re-walks a whole day of files. Hour grain bounds a poll at
the hour directories it actually needs, and the bound does not move as the archive grows —
measured on a 60-symbol slice of the live dataset, a one-hour bound selects **59 of 1,805**
fragments against **1,587 of 1,587** under the legacy layout.

## 1. The layout

```
<store>/<dataset>/availability_hour=YYYY-MM-DDTHH/symbol=<SYMBOL>/part-<snapshot>.parquet
```

**One partition key for time, not two.** `availability_hour=2026-08-17T11` rather than
`availability_date=…/availability_hour=11`: one added column instead of two, one directory level
instead of two, and a single fixed-width string whose lexicographic order **is** its chronological
order, so `>=` on the path segment is a correct time bound with no parsing.

**Hour before symbol.** The pruning that matters is temporal — every poll is "what arrived since my
watermark" — and the top level is what gets pruned without a listing. A symbol-filtered read still
prunes correctly one level down; it pays one directory listing per hour directory, which is
readdir, not a parquet open.

**UTC, always.** The store is UTC everywhere else and a local-time path would silently reorder
across a DST boundary.

## 2. What must not change

These are the properties the store already promises. Each is currently asserted by a test, and the
migration is not done until each still is.

1. **Append-only.** A snapshot id already written is refused, never overwritten. Corrections are new
   parts. `PartitionExistsError`.
2. **All-or-nothing per call.** Every target is collision-checked before any part is written, so a
   frame with N symbols cannot leave N-1 parts behind. With hour grain the plan is over
   (hour, symbol) pairs and the property is unchanged.
3. **A part is absent or complete.** Write to `.writing-part-*`, fsync, rename, fsync the directory.
4. **No silently dropped column.** The schema walk of SL-14 stays; a column added by a later
   partition still appears, and a fragment that disappears still forces a full rebuild.
5. **The returned frame's shape.** `read_dataset` reconstructs `symbol` from the path today.
   It must now also **drop `availability_hour`** before returning, so no downstream reader sees a
   new column. The partition key is an index, not data — `AVAILABILITY_TIME` remains the only
   availability field a caller reads.
6. **Corrections still arrive.** The bound is on availability, never event time: a correction to an
   old bar carries a LATER availability time, so it lands in a LATER hour directory and a poll sees
   it. This is the whole reason hour-of-availability is a safe pruning key and hour-of-event would
   not be.

## 3. The read path

```python
read_dataset(store_root, dataset, not_before_ns=None)
```

* `not_before_ns is None` — unchanged, reads everything.
* `not_before_ns` given, dataset already migrated — filter on **both**:
  * `availability_hour >= floor_to_hour(not_before_ns)` — prunes directories, opens nothing;
  * `AVAILABILITY_TIME >= not_before_ns` — the exact bound, inside the surviving files.
* `not_before_ns` given, dataset **not yet migrated** — the row bound alone.

That last case is not defensiveness. Every dataset is in the old layout until its own migration has
run, and naming a field the dataset does not have makes pyarrow raise
`ArrowInvalid: No match for FieldRef.Name(availability_hour)` and take the entire read down. The
paper engine polls with a bound every 60 seconds, so without the fallback this change stops the
engine the moment it lands — before any migration could fix it. Found by running the migration
against a copy of real data rather than a fixture.

The hour bound is floored, never rounded: flooring includes the partly-consumed hour the watermark
sits in, and the exact row filter removes the rows already seen. Rounding up would skip rows in
that hour, permanently — they would never appear in any later poll either, because a later
watermark is further ahead still.

## 4. The migration

`src/store/hourly_migration.py`, run by
`python -m store.hourly_migration --dataset <name> [--swap]`. Two steps rather than one flag-free
command: the build is safe while capture continues, the swap changes which directory readers open,
and making the second explicit means nobody performs it by re-running the first.

Writes to a sibling directory, never in place:

```
<dataset>                              the live legacy layout, untouched
<dataset>.hourly-building              the new one, while it is being built
<dataset>.legacy-symbol-layout         the old one after the swap, retired by hand
```

Steps, in order:

1. **Read the legacy dataset one symbol directory at a time**, never the whole dataset — 538 MB
   fits in memory, and the next dataset may not.
2. **Group each symbol's rows by availability hour** and write one compacted part per
   (hour, symbol). Snapshot id is `compute_snapshot_id` over the contributing legacy parts, so the
   name still answers "which data produced this".
3. **Verify before swapping**, and verify on content rather than on counts alone:
   * row count per (hour, symbol) equals the legacy row count for those rows;
   * the total row count matches;
   * the unified schema of the new dataset is a superset of the legacy one — no column lost, which
     is the failure mode this store has actually suffered;
   * a sampled digest of sorted rows matches for a sample of symbols including the largest.
4. **Swap by rename**, both directions recorded in the run's report. A rename is atomic and
   reversible; a copy-and-delete is neither.
5. **Never delete the legacy directory.** The row says "before the old is retired", and retiring it
   is a separate, human decision after the boards have been green on the new layout.

**Restartable, by a manifest of consumed legacy parts** — `.migrated-parts.json` inside the
building directory, dot-prefixed so pyarrow's discovery skips it. The manifest is written AFTER the
parts it names, never before: a manifest ahead of its write would make the next pass skip parts
nothing migrated, and those rows would be missing from the new layout rather than merely migrated
twice.

**Capture keeps running throughout, and the manifest is what makes that safe.** Without it a second
pass re-reads every part for a symbol, writes them all under a new content-derived snapshot id, and
the store quietly holds those rows TWICE — duplicated bars that read as real volume, with nothing
raising. `test_a_second_pass_migrates_only_what_arrived_since_the_first` fails without it.

## 5. Compaction

Compaction is what the migration does by construction: all legacy parts for one (hour, symbol)
become one part.

**It does not reduce the file count here, and that was measured rather than assumed.** These
writers already snapshot about hourly, so there is little to merge — and a legacy part spanning
more than one hour is SPLIT across hour directories instead. Measured 2026-08-17: a 60-symbol
slice of bars went 1,587 parts → 1,805, and `dated_futures` went 680 → 1,472. An earlier draft of
this spec claimed a ~20x cut; it was wrong, and the only reason it did not survive is that the
trial run was done on a copy of real data before the live store was touched.

The saving is pruning and nothing else. On that same slice a one-hour bound selects **59 of 1,805**
fragments under the hour layout against **1,587 of 1,587** under the legacy one, returning
byte-identical rows.

Ongoing writes therefore need no separate compactor.

## 6. Call sites that know the layout

Three, all found by grep and all changed with the layout:

| file | what it assumes |
|---|---|
| `src/store/parquet_partition.py` | writes `symbol=<S>/part-<id>.parquet` |
| `src/store/cli.py` | builds a part path to check for a snapshot |
| `src/statuswall/evidence.py` | counts `symbol=` directories to report partition counts |

## 7. Acceptance — the row's own words

> a 60-second poll opens the fragments of at most two hour directories however long capture has been
> running, and every row of the old layout is present in the new one before the old is retired

Measured by `probe_poll_scan_cost`, which reads `fragment_schema_reads` and `poll_seconds` from the
engine's heartbeat and grades against the fragment count on disk. The migration's verification
report is the second half, and it is written to `~/capture/store/migration-report-<dataset>.json`
so the claim is a file rather than a sentence in a transcript.

## 8. What this does not do

Not a general compaction policy, not a retention policy, not a change to the raw tape, and not a
change to what `read_as_of` means. The dataset gains a pruning key; nothing gains or loses a row.
