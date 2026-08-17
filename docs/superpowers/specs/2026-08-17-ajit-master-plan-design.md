# AJIT MASTER PLAN — design

Date: 2026-08-17
Status: **DESIGN, awaiting user review.** No implementation until the user approves this file.
Authority once built: **top of the chain for what to build next.** See §1.
Interviewed and settled with the user on 2026-08-17 across 18 questions. Every choice below is
the user's, not a default.

---

## 0. Why this exists — the measured failure it answers

The user asked, on 2026-08-17: *"why did we fail to completely implement or start papper trading
according to the plan we discussed each segment are like there own bots with its own architecture,
data features etc."*

Four things were measured that day rather than recalled.

**The plan that says what to build next is eight days stale and contradicts what is running.**
`2026-08-09-full-build-master-plan.md` is 118 lines of pointers whose phase J places paper trading
last; paper trading has been running since 2026-08-15. `DECISIONS.md` §13 records the identical
defect about itself on 2026-08-09 — *"the one section whose job is to say what to do next had been
wrong for six days."* The same failure, twice, in the same project.

**Ideas reach a document and stop there.** Tested by name against the 1,491-row ledger, which is
the thing the build reconciles against:

| Idea | In design documents | Rows in the ledger |
|---|---|---|
| Goodhart defence — the metric never optimised | 4 files | **0** |
| Attention scarcity — the mechanism behind §5a | 2 files | **0** |
| The examination-hall framing (the user's own, 2026-08-02) | 3 files | **0** |

The examination-hall ruling was argued through, written into the goal document as roughly 140 lines
of design in §5a, and never became one row of work.

**Recorded is not reached.** 497 ledger rows sit in PLANNED — named in a design document, not
built, with no slice, no owner, no probe and no position in any order.

**Rulings lived only in conversation.** 21 human rulings were recovered on 2026-08-17 by reading
every user message in the project's transcripts. RL-011 — three independent bots with their own
data, features and architecture — was given on 2026-08-03 and is 2/22 built. §3a of the goal
document is the counter-example that worked: a ruling written verbatim the moment it was given.

**A fifth cause is mechanical, not editorial.** Project memory is keyed by working directory. Twelve
memories exist under `~/.claude/projects/-home-anushadudekula71/memory/` — including one titled
*"Interview, don't assume"* and one pointing at the goal document and the ledger. A session started
from `/` loads `~/.claude/projects/-/memory/`, which is empty. Every such session begins with no
project memory at all.

---

## 1. What the file is, and its authority

**`docs/AJIT-MASTER-PLAN.md`.** Named by the user. The file every session opens first.

It owns the build order, the slices, the rulings and the reconciliation. Where any other document
disagrees with it about **what to build next**, this file wins, and its own header says so.

| Document | Keeps |
|---|---|
| `docs/superpowers/specs/2026-08-08-final-project-goal-design.md` | the goal, the three human-settled rulings, §1a and its 23 tests |
| `~/research/ARCHITECTURE.md` | structure, components, contracts |
| `~/research/DECISIONS.md` | the record of why |
| `~/research/FEATURES.md` | the capability list |
| `~/research/ledger/` | the research corpus |
| `docs/superpowers/plans/2026-08-09-full-build-master-plan.md` | **superseded by this file**, retained as the record of what was decided on 2026-08-09 — not deleted |

### Written spine, generated status, never mixed

The user's ruling: the ORDER and the DECISIONS are written by hand and change only when the user
rules; the STATUS of every row is measured by probes and never typed.

This forces two artefacts, because three standing rules collide otherwise — Rule 8 says a display
of state must be generated, Rule 9 says generated state must never be committed, and a plan is a
record of decisions that must be committed.

| Artefact | Written or generated | Committed | Holds |
|---|---|---|---|
| `docs/AJIT-MASTER-PLAN.md` | written | yes | order, slices, rows, rulings, scopes, declines. Every row's `state:` field reads `measured by <probe>` — never a value |
| `~/research/dashboard/ajit-master-plan.html` | generated | no | the measured state of every row, the reconciliation counts, the active slice, the next row |

A value typed into the first file is a defect the tests catch (§9).

---

## 2. The slices — vertical, one bot at a time

Ordering principle chosen by the user over layer-by-layer: **each slice ends with a bot that
trades.** The layer-by-layer order is what produced the six-day gap in which nothing ran, and the
user already reversed it on 2026-08-15.

```
SLICE 0  foundation & governance                        ← next
SLICE 1  SPOT BOT           → BUILT, trading
SLICE 2  PERP BOT           → BUILT, trading
SLICE 3  DATED FUTURES BOT  → BUILT, trading
SLICE 4  OPTIONS BOT        → BUILT, trading
SLICE 5  allocator over the four · portfolio risk · brains 2 and 3
```

**Slice 0 contents**, all five chosen by the user:

1. **Store backup to GCS.** Measured 2026-08-17: the bucket holds `raw/`, `ledger/` and `universe/`
   — not `store/`. `store/funding` is 446 MB, is the OBSERVED record whose start date is the clock
   every promotion waits on, and has no raw counterpart at all (`~/capture/raw/binance-funding` does
   not exist; funding is polled straight into the store). Losing this disk resets that clock to
   zero. `funding_reconstructed` cannot substitute — its availability times are the fetch, which is
   deliberately what makes it useless for a backtest.
2. **The enforcement layer** — this plan, the rulings register, the conformance probe and board,
   the no-row-no-module hook, the session-start injection. Built first so every later slice is
   governed by it rather than retrofitted to it.
3. **The memory path fix** — so a session started from `/` reaches the same memories as one started
   from `~`.
4. **The paper cold-start fix.** Measured 2026-08-17: after boot the engine re-primes ~1.98M
   archived events before its first poll — about 11 minutes with nothing trading. With four bots
   that is four cold starts. Fixed before there are four.
5. **The shared bot framework** — brain interface, journal format, risk-gate interface, on/off
   switch. Built once as the contract all four segment bots implement, so the spot bot is the first
   implementation rather than the thing the other three are retrofitted to.

**Every segment slice is the same vertical:**

```
segment data completion
  → segment features
    → BULL brain · BEAR brain · PROFIT-TAIL brain
      → segment risk gate
        → engine, journal, supervisor, on/off switch
          → board tile
            → three axis verdicts per brain
              → reconciliation sweep
```

**Shared beneath all four**, and deliberately not duplicated: the raw store, the clock-gated reader,
the cost engine, the promotion pipeline and validation. Chosen by the user over per-bot copies —
they are venue-neutral, and four copies would produce four subtly different truths about the same
tape. The single copy of this machinery has already caught three defects.

### The running engine

`plumbing-momentum` keeps running exactly as it is through slice 1, so the 24/7 record keeps
accumulating. At slice 2 it is retired into the real perp bot and its journal is archived, marked
as plumbing that made no edge claim so it can never later be read as a result. Its board tile says
PLUMBING, never a P&L.

---

## 3. Bot topology

Two rulings compose, on two different axes:

- **RL-011** (2026-08-03) splits by **direction**: BULL, BEAR, PROFIT-TAIL, each with its own data,
  features, architecture, training and validation history.
- **RL-019** (2026-08-17) splits by **segment**: spot, perpetual futures, dated futures, options.

Taken as a product that is twelve independent bots. The user chose the nested reading:

```
SPOT BOT            PERP BOT           DATED BOT          OPTIONS BOT
  data: spot          data: perp         data: dated        data: chain + greeks
  ├ BULL brain        ├ BULL brain       ├ BULL brain       ├ BULL brain
  ├ BEAR brain        ├ BEAR brain       ├ BEAR brain       ├ BEAR brain
  └ PROFIT-TAIL       └ PROFIT-TAIL      └ PROFIT-TAIL      └ PROFIT-TAIL
  own risk gate       own risk gate      own risk gate      own risk gate
  own journal         own journal        own journal        own journal
  own on/off switch   own on/off switch  own on/off switch  own on/off switch

shared: store · clock-gated reader · cost engine · promotion pipeline · validation
```

**The bot boundary is the segment.** Four processes, twelve brains, four P&L curves. The three
brains inside a bot are independent of each other in features and architecture, and share that
segment's data pipeline — because spot ticks and an options chain are genuinely different data, and
BULL-on-spot has nothing to learn from BEAR-on-options.

### The on/off switch

A file on disk holds one line per segment bot plus an ALL master. Only the user sets it. Each bot
reads it every poll and stops **opening** on off; existing positions are still managed to exit,
because abandoning open risk is not what off should mean. The board shows each bot's switch state
beside its heartbeat, so off never reads as dead.

This is a **third** control, distinct from the two that exist: the §6 kill switch means *something
is wrong, stop*, and the capital feasibility gate disables families by itself when the dial is too
small. Conflating them would make a deliberate pause look like a fault on the board.

### The intelligence standard binds per brain

Chosen by the user over segment-level or system-level judging. Each of the twelve brains carries its
own verdict on **all three axes** — learning, reasoning, depth — measured by §1a's mechanical tests.
A brain failing an axis says so on its tile rather than shipping quietly. **36 verdicts**, reading
0/36 today. This is what stops a hardcoded rule wearing the word "brain", which is the substitution
the user ruled against three times (2026-08-01, 2026-08-02, 2026-08-08).

---

## 4. Row anatomy, states and gates

### The row is a contract — eight fields, none optional

```
SP-04  slice: spot-bot
  does:       compute spot-only microstructure features per symbol
  satisfies:  RL-006 RL-009 RL-014 RL-019
  sources:    FEATURES §2.4 · ledger FE-014, FE-021 · finml-feature-engineering.md
  depends on: SP-02 (spot bars), shared clock-gated reader
  probe:      probe_spot_features
  accepts:    every value carries a staleness stamp and no value is readable
              before its availability time
  state:      measured by probe_spot_features
```

Nothing is buildable until all eight are filled. An unwritten decision therefore shows up as a blank
field **before** any code is written, rather than as a wrong module afterwards.

### Two kinds of state, kept apart because they come from different places

| Kind | Values | Written by |
|---|---|---|
| **Measured** | `NOT BUILT · PARTIAL · DEGRADED · BUILT · OK · NOT MEASURED` | probes only, never typed. Reuses the status wall's existing vocabulary |
| **Decided** | `BLOCKED` (naming what it waits on) · `DECLINED` (with the user's reason) | the user, never inferred |

A row with no probe renders **NOT MEASURED**, never green (Rule 8).

### Two gates per slice

**BUILT** — every row in the slice lit, the bot running under its own supervisor, journalling fills,
`integrity.unsupported_claims` reporting zero for it, and its board tile measured. **The next slice
starts here.**

**EARNING** — a strategy of that bot promoted through the full gate stack on observed history.
Tracked per bot, shown on the board, and **never blocks the build**.

The separation is required by arithmetic already measured: §13 records that promotion needs weeks of
OBSERVED history, on a clock that started 2026-08-08 and cannot be shortened by reconstruction. One
combined gate would either stall the build for weeks or let a half-built bot be called finished.

---

## 5. Reconciliation — how the file proves nothing was skipped

Three populations are swept, not one. The third is new and is the one that would have caught the
user's own example.

| Population | Size | Every member must resolve to |
|---|---|---|
| Ledger rows | 1,491 | a plan row · DECLINED with reason · BLOCKED naming its blocker |
| `FEATURES.md` rows | 167 | a plan row · DECLINED · BLOCKED |
| **Design sections** | **826** (655 across 63 research files, 171 across the specs and plans) | a plan row · DECLINED · BLOCKED · **"prose, no work implied"** |

Anything resolving to none of these renders **UNASSIGNED in red**, with the count on the board. That
count starts large. That is the point — it is the number that has been invisible.

`"prose, no work implied"` is a real and necessary answer: §0 Prime directive implies no module. It
is **written by the assistant and reviewable by the user**, never inferred silently, because
inferring it is how 140 lines of §5a became zero rows of work.

### The scope rule — why "assigned once" is not enough

A cross-cutting capability assigned to one slice would read as fully resolved while three of four
bots silently never received it. That is the failure this file exists to prevent, reproduced inside
the file. So every ruling and every idea carries a **scope**, and reconciliation is arithmetic per
scope:

| Scope | Requirement | Renders UNASSIGNED when |
|---|---|---|
| `per-segment` | a row in **every** segment slice | any one segment lacks it |
| `per-brain` | a row for **each of the 12 brains** | any brain lacks it |
| `shared` | one row, in slice 0 or a named slice | no row anywhere |
| `system` | one row, judged at system level | no row anywhere |

Worked example, the user's own: universe-wide opportunity monitoring is `per-segment`. Four rows
required, **0/4 today**, red on the board from the first regeneration.

### Default scope assignment

Applied to all 383 idea candidates and every ledger row, then overridden individually by the user:

- Decides about a **symbol or a market** — opportunity monitoring, features, signals, entries, exits,
  sizing → **per-segment**
- About a brain's own **learning, beliefs, calibration or competence** → **per-brain**
- About **capture, store, cost, validation, registry, the tail cap or governance** → **shared**

Every assignment is written into the file and any of them can be overridden.

### Scopes already ruled by the user, 2026-08-17

| Capability | Scope | The reason it was not the obvious answer |
|---|---|---|
| Universe-wide opportunity monitor | **per-segment** | §5a currently calls it *"a property of the system"*. Amended to a per-bot property; recorded as an amendment to RL-009 and RL-014 rather than a silent reinterpretation |
| Trial registry | **system, one registry** | 12 brains each counting their own trials makes every promotion ~12× easier than correct. The project has already found one defect of exactly this shape |
| Belief records with provenance and half-life | **shared layer, per-brain beliefs inside** | one retraction mechanism, tagged by which brain formed the belief; no brain reads another's as its own |
| Abstention with measured P&L | **every level, measured separately** | brain, bot and system may each abstain; waiting for a symbol's moment IS abstention until then |
| Ergodic / time-average objective | **system, binding on every bot** | there is one account and one path through time, so one growth rate to maximise |
| Evolved-program strategy search | **per brain, one shared engine** | one engine, twelve independent populations; a spot BULL program never enters the options BEAR population |
| Autonomous internet access | **one shared quarantined service** | the whole prompt-injection surface in one place; §5a.6b already names market data as attacker-authored text |
| Diversity archive | **per brain, counted in the shared registry** | §1a.2's archive clause: a diversity archive buys no discount on multiple testing |

---

## 6. Enforcement — four mechanisms plus the path fix

All four chosen by the user. Prose in `CLAUDE.md` already said "search the ledger first" and it
still failed, which is why three of these are mechanical.

| Mechanism | What it does | Failure mode it closes |
|---|---|---|
| `docs/rulings.json` | 21 rulings recovered from transcripts, verbatim and dated; appended the moment a ruling is given | RL-011 sat unrecorded as a ruling for two weeks |
| **PreToolUse hook** | a **new** file under `trading-system/src/` is refused unless a plan row names it. Edits are never blocked | building something no plan row asked for |
| **Conformance probe** | one board row per ruling; a ruling with no probe renders NOT MEASURED | a ruling honoured on paper and not in running code |
| **SessionStart hook** | injects the ruling list and the active slice into every session | the plan being findable but not present |
| **Memory path fix** | every session reaches the same memories regardless of starting directory | a session from `/` starting with zero project memory |

The hook's escape hatch is **adding the row**, which is the behaviour wanted rather than an obstacle
to route around. Per Rule 4, PreToolUse fails closed: a wrong block is merely annoying.

---

## 7. Amendments — how the file survives new rulings

A new ruling goes into `docs/rulings.json` **the moment the user gives it** — verbatim, dated,
before other work continues. Then it is **placed**, explicitly, into one of three states:

1. rows in a named slice, or
2. DECLINED with the user's reason, or
3. DEFERRED to a named later slice.

It never sits recorded-but-unplaced. That is the state RL-011 was in for two weeks. The slice in
flight is not interrupted unless the user says so.

---

## 8. Progress reporting

Every regeneration stamps: rows BUILT / total per slice · which slice is active · **which single row
is next** · what each blocked row waits on · the three reconciliation counts. All measured.

**No effort estimates and no dates.** Chosen by the user. An estimate on this project would be a
guess presented as a number, and a wrong one is what makes *"why is it taking so long"*
unanswerable rather than answered. The user has asked that question six times between 2026-08-08 and
2026-08-17; measured counts are what answer it without asking.

---

## 9. What lands on disk

**New:**

| Path | Kind |
|---|---|
| `docs/AJIT-MASTER-PLAN.md` | written, committed |
| `docs/rulings.json` | written, committed — already created 2026-08-17 |
| `src/plan/master_plan.py` | parses the spine, sweeps the three populations, applies scope arithmetic |
| `src/plan/reconcile_sources.py` | reads the ledger, FEATURES and the 826 design sections |
| `src/statuswall/ruling_conformance.py` | probes per ruling, renders the conformance board |
| `src/statuswall/master_plan_board.py` | renders `ajit-master-plan.html` |
| `tests/test_master_plan.py`, `tests/test_reconcile_sources.py`, `tests/test_ruling_conformance.py` | |
| `~/.claude/hooks/require-plan-row.sh` | PreToolUse, Write |
| `~/.claude/hooks/inject-plan-index.sh` | SessionStart |

**Changed:** `scripts/boards_supervisor.sh` (render the two new boards) · `scripts/offload_to_gcs.sh`
(cover `store/`) · `scripts/paper_supervisor.sh` (cold start) · `~/.claude/settings.json` (register
two hooks) · `~/.claude/CLAUDE.md` (index table points at this plan first) ·
`docs/superpowers/plans/2026-08-09-full-build-master-plan.md` (marked superseded) ·
goal doc §5a (amended to a per-bot property) and a new §3b (RL-019 verbatim).

---

## 10. Testing

Tests are named for the behaviour they defend, per the repo's standing rule.

1. **No typed state.** Parsing `AJIT-MASTER-PLAN.md` fails if any row's `state:` field holds a
   status value rather than a probe name. This is the guard on the written/generated split.
2. **Eight fields or not buildable.** A row missing any of the eight fields is rejected, and the
   error names the missing field.
3. **Scope arithmetic.** A `per-segment` item with rows in three of four slices reports 3/4 and
   renders UNASSIGNED, not resolved. The same for `per-brain` at 11/12.
4. **Three populations swept.** Every ledger row, every `FEATURES.md` row and every design section
   appears exactly once in the reconciliation, and the totals match the source counts.
5. **Unknown resolution rejected.** A member resolving to anything outside the allowed set fails.
6. **Probe absence renders NOT MEASURED.** A row with `probe: null` never renders a lit state.
7. **Hook blocks and permits.** A new `src/` file with no plan row is refused (exit 2); one with a
   row is permitted; an edit to an existing file is always permitted. Tested by piping payloads,
   per Rule 4 — exit 2 blocks, exit 1 does not.
8. **Rulings register parses and is complete.** Every ruling has an id, date, verbatim text and
   either a probe or an explicit null.

Baseline to preserve: **857 passed, 1 skipped** as of 2026-08-09, via `.venv/bin/python -m pytest -q`.

---

## 11. Definition of done for this piece of work

1. `docs/AJIT-MASTER-PLAN.md` exists, parses, and every row carries all eight fields.
2. The reconciliation board renders and its three counts match the source populations.
3. The UNASSIGNED count is measured and published — whatever it is. A large number is the correct
   first result and must not be hidden.
4. Both hooks are registered and verified firing, not merely registered.
5. The conformance board shows all 21 rulings, each measured or explicitly NOT MEASURED.
6. Test suite green, count recorded rather than asserted.
7. Committed and pushed (Rule 9).

---

## 12. Open, and deliberately not decided here

- **Which rows fill slices 1–5.** This spec settles the file's structure, its gates and its
  enforcement. The row inventory is produced by the reconciliation sweep and reviewed with the user
  before slice 1 begins — writing rows now would be the assumption this whole design exists to stop.
- **The UNASSIGNED count.** Unknown until the sweep runs. It will be large.
- **Hyperliquid capture breadth.** `trades_*` is subscribed for BTC, ETH and SOL only, so 174 of its
  177 stored symbols are a week stale. A real row; its slice is decided when the inventory is
  reviewed.
- **Key scoping and `state_recovery` against venue truth.** Both still travel with the first live
  strategy, unchanged by this design.

---

## Provenance

Interviewed with the user on 2026-08-17 across 18 questions, following Rule 6 and the
`superpowers:brainstorming` architectural path. Classified architectural because it restructures how
the project is planned and how every future session decides what to build.

Every fact in §0 was measured on 2026-08-17, not recalled: the stale plan by reading it, the three
zero-row ideas by grepping the ledger, the 497 PLANNED count from `ledger/INDEX.md`, the 826 design
sections by counting headings, the store backup gap by listing the GCS bucket, the cold start by
watching the engine prime, and the memory path by listing both memory directories.
