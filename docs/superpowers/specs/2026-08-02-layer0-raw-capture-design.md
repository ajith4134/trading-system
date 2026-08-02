# Layer 0 — Raw Market Data Capture

**Date:** 2026-08-02
**Status:** design approved, pending spec review
**Sub-project 1 of 7** in the Autonomous Crypto Trading System (`~/research/DECISIONS.md`)

---

## 1. Context

The system has a settled stack (`~/research/ARCHITECTURE.md` §3c: Python 3.12, NautilusTrader,
Parquet + DuckDB, LightGBM), six research files with 418 catalogued candidate features, and **zero
lines of code**. Prior attempt: `github.com/ajith4134/nse-crypto-bot-final`.

The corpus converged independently, from four different altitudes, on one conclusion:
**build the thing that tells you when you are wrong before the thing that tries to be right.**
Layer 0 is the root of that dependency graph.

**Why this sub-project is first:** it is the only genuinely irreversible item. Validation, features,
backtest and live all consume it, and **market data you did not capture cannot be re-collected at
any price.** Everything else can be retrofitted.

## 2. Scope

Build a **raw capture service**: subscribe to venue websockets, write what arrives verbatim, forever.

### Non-goals — each is a later sub-project

Conflating any of these is how Layer 0 becomes unfinishable.

- No parsing, normalising, or schema-fitting of message bodies
- No Parquet store, no DuckDB, no feature computation
- **No clock-gated access API** — sub-project 2, reads from this archive
- No trading, no signals, no NautilusTrader integration
- No backfill of history — live capture only (backfill is a separate, later decision)

### Governing invariant

> **A written frame is never modified. The bytes on disk are exactly the bytes the venue sent.**
> All derived data — timestamps, sequence numbers, gap records — lives in **sidecar** files.

This is what keeps the archive an option on questions not yet conceived, and what keeps the
exchange-time vs receipt-time distinction (`~/research/IDEAS-STRATEGIC.md` §10) recoverable rather
than baked in wrong.

## 3. Decisions, with rationale

| Decision | Choice | Rationale |
|---|---|---|
| Sequencing | **Raw recorder first**, store/API later | Days of capture beat weeks of architecture. Raw bytes are kept forever, so the store can be rebuilt any number of times |
| Capture scope | **Tiered**: deep core + broad tail | Resolves the tension between "L2 depth is P0" (`ARCHITECTURE.md`) and "the edge is in capacity-constrained corners" (`IDEAS-STRATEGIC.md` §2). Depth is what explodes storage; only a handful of symbols get it |
| Hosting | **This GCE box + GCS offload** | 96 GB local is a ceiling, not a home, for a forever-archive |
| Integrity | **Gap-aware single recorder** | A recorder that silently drops data is worse than none, because it will be trusted |
| Implementation | **Purpose-built async Python** | See §3.1 |

### 3.1 Why not ccxt.pro or NautilusTrader adapters

- **ccxt.pro** normalises messages into its own schema. That **destroys raw fidelity**, which is the
  entire reason for capturing first. You would permanently store ccxt's interpretation with no way
  to recover what the venue actually sent. Also licensed/paid.
- **NautilusTrader adapters** invert a settled decision. `ARCHITECTURE.md` §3c: *"Nautilus must not
  own Layer 0 — the truth layer is ours, and Nautilus consumes from it."* Using its adapters here
  couples the irreplaceable archive to a framework version. Nautilus remains the execution path; it
  does not own capture.

### 3.2 Capture tiers

| Tier | Symbols | Streams |
|---|---|---|
| **Core** | `BTC`, `ETH`, `SOL` — Binance USDⓈ-M perps + Hyperliquid perps | Full L2 depth diffs + trades + funding + open interest + liquidations |
| **Tail** | Every perp each venue lists, discovered dynamically | Trades + funding + open interest + liquidations. **No depth** |

Exact instrument selection and the reasoning behind three-not-five, and futures-not-spot, are in §11.

### 3.3 How each data type is actually obtained — **corrected 2026-08-02 by live measurement**

The original design assumed every data type arrived over websocket. **Task 11's live smoke test proved
otherwise**, and the correction was verified independently with no capture code involved. Measured on
this host, 12 s per single stream on `wss://fstream.binance.com/ws/`:

```
depth@100ms  117    bookTicker 266    trade 501     <- WORK
aggTrade 0   markPrice@1s 0   forceOrder 0   kline_1m 0   miniTicker 0   !forceOrder@arr 0
```

Every `markPrice` and `forceOrder` variant was tested (`@1s`, bare, `!…@arr`, `!…@arr@1s`, and via
`/stream?streams=`). All returned zero. This is not a naming error — those stream families are
unavailable from this host, while REST serves the same data normally.

**Had this shipped, the archive would have contained depth only** — no trades, no funding, no
liquidations — and none of it recoverable later. This is the single strongest justification for
sequencing a live smoke test inside the capture sub-project rather than after it.

| Data | Source | Notes |
|---|---|---|
| **L2 depth** | Binance ws `@depth@100ms` · Hyperliquid ws `l2Book` | Unchanged |
| **Trades** | Binance ws **`@trade`** · Hyperliquid ws `trades` | **`@aggTrade` yields zero here.** `@trade` is *individual* trades — strictly more raw, so this is an improvement on the original spec, not a workaround |
| **Funding + mark** | **REST poll** | Binance `/fapi/v1/premiumIndex` **with no symbol returns 854 symbols in one call**; Hyperliquid `metaAndAssetCtxs` returns **232 assets in one call**. Hyperliquid ws `activeAssetCtx` also works |
| **Open interest** | **REST poll** | Binance `/fapi/v1/openInterest` (per symbol); Hyperliquid included in the same one-call payload |
| **Liquidations** | **OKX ws `liquidation-orders` (`instType: SWAP`)** | Binance has **no public path**: `allForceOrders` → 404, `forceOrders` → 401 (auth, own orders only). OKX verified working, one subscription covers every swap |

**The REST path is better than the websocket design it replaces.** One poll covers 854 Binance
symbols and one covers 232 Hyperliquid assets; per-symbol websocket streams would have needed
hundreds of subscriptions to match that. This materially strengthens **R1 (genuinely broad tail)**
rather than merely routing around a blocked stream.

### 3.4 OKX — reference-only venue (added 2026-08-02)

OKX is captured **solely for market-wide liquidation data** and is **never an execution venue**. This
boundary is permanent and must be enforced in code and in review: no order path, no credentials, no
adapter surface beyond reading the public liquidation stream.

Rationale: liquidation cascades are cross-venue correlated, so OKX liquidations are a usable proxy for
market-wide forced flow — and forced, non-discretionary flow is the most durable edge class in
`~/research/IDEAS-INTELLIGENCE.md` §6. The data is unrecoverable if not captured as it happens.

**A silent stream must never look like a healthy one.** A subscribed stream that has received zero
frames after a startup grace period is recorded as a ledger event, so this failure mode cannot recur
undetected on any venue.

Tail breadth is a requirement, not a default — see §8 R1.

## 4. Architecture

Six units, one responsibility each.

| Unit | Responsibility |
|---|---|
| `venue_recorder` | One process per venue. Owns subscriptions, reconnect, backoff |
| `raw_writer` | Appends frames verbatim, zstd, hourly rotation, one file per (venue, stream) |
| `capture_ledger` | Records gaps, disconnects, reconnects, queue overflow as first-class events |
| `universe_tracker` | Polls instrument lists; emits listing / delisting / rename / status-change events |
| `archive_offloader` | Checksums completed files, uploads to GCS, prunes local after verified upload |
| `capture_health` | Heartbeat, disk headroom, per-stream staleness, alerting |

**Flow:** websocket frame → bounded in-memory queue → `raw_writer` appends → hourly rotation →
checksum → GCS upload → verify → local prune after retention.

The bounded queue matters: writing must never block socket reads, or backpressure becomes silent
frame loss. **Queue overflow is itself a ledger event**, never a silent drop.

## 5. On-disk format

### 5.1 Verified venue behaviour

Probed live on 2026-08-02 (`scratchpad/probe_venues.py`, 6 frames per stream):

```
binance-spot-depth      frames=6 text=6 binary=0 multiline=0 max_len=1089
binance-futures-depth   frames=6 text=6 binary=0 multiline=0 max_len=1127
hyperliquid l2Book      frames=6 text=6 binary=0 multiline=0 max_len=1610
```

All text, zero binary, **zero embedded newlines** — so NDJSON line alignment is safe.

### 5.2 Two-file layout

```
{stream}_{symbol}_{hourUTC}.ndjson.zst   one venue frame per line, byte-identical to what arrived
{stream}_{symbol}_{hourUTC}.idx.zst      one line per frame, parallel:
                                         {"n":0,"t_recv_ns":…,"t_exch_ms":…,"seq":…,"kind":"data"}
```

Two parallel files rather than an envelope wrapping the payload:

- **Byte-exact** — no base64 (≈33% inflation, worse compression), no JSON re-encoding, no key reordering
- **All derived metadata in the index**, so the raw file is never touched to add a field later
- **Greppable with standard tools**, which matters during an incident
- **Newline guard**: if a payload ever contains a literal newline, escape it and set a flag in the
  index rather than silently corrupting line alignment

### 5.3 Directory layout

```
~/capture/raw/{venue}/{date}/{stream}_{symbol}_{hourUTC}.ndjson.zst + .idx.zst
~/capture/ledger/{venue}/{date}/events.ndjson
~/capture/universe/{venue}/{date}/instruments.ndjson
```

### 5.4 Per-venue sequencing — not uniform

Discovered by probe; designing one mechanism for both venues would have been wrong in both directions.

| Stream | Sequencing | Gap meaning |
|---|---|---|
| Binance spot depth | `U` / `u` (first/final update id) | **Stateful diffs** — gap corrupts the book until REST resync |
| Binance futures depth | `U` / `u` / **`pu`** (previous final id) | Same, plus explicit chain validation |
| Hyperliquid `l2Book` | **no sequence number** | **Stateless snapshots** — gap loses an observation; nothing corrupts |

**Binance futures carries both `E` (event time) and `T` (transaction time)**; spot carries only `E`.
The index records *which* timestamp field it captured per stream rather than assuming a single
"exchange timestamp".

**Control frames** (e.g. Hyperliquid's `subscriptionResponse`) are part of the session record:
**captured**, but flagged `kind:"control"` in the index so downstream does not parse them as data.

### 5.5 Storage projection

Measured frame sizes ~300–1600 bytes at ~10/sec per depth stream ≈ 400 MB/day/stream uncompressed,
~45 MB compressed. Core tier + broad trades-only tail ≈ **1–2 GB/day compressed**. Against 91 GB
free that is 45–90 days of headroom, so **7-day local retention + GCS offload is comfortable.**

## 6. Failure handling

**Principle: never drop data silently, never modify data to "fix" it.** Every anomaly becomes a
ledger event; raw bytes stay untouched even when malformed.

| Failure | Response |
|---|---|
| Websocket disconnect | Exponential backoff. Ledger event with disconnect + reconnect times. Binance depth: mark book invalid, REST snapshot resync, record the resync |
| Sequence-chain break | Ledger event, **severity=corrupting**, resync. Downstream must be able to exclude the window |
| Hyperliquid staleness | Time-based only. Per-stream expected interval is measured over a rolling window; **staleness fires at 10× the rolling median inter-frame gap, floored at 5 s** → **severity=observation-loss**. Both values are tunable per stream, not hardcoded constants |
| Writer queue overflow | Ledger event, **loud alert**. The only path to true data loss |
| Disk pressure | Degradation ladder — see below |
| GCS upload failure | Retry with backoff. **Never prune on a timer — only after checksum-verified upload.** Alert on backlog growth |
| Process crash | Restart writes a `gap_on_restart` event covering the dark window |
| Clock jump / NTP skew | Detect monotonic-vs-wall divergence, flag affected index entries. Timestamps cannot be repaired later |
| Malformed frame | **Written verbatim**, flagged in index, never discarded |
| Symbol delisted mid-stream | `universe_tracker` emits event; recorder unsubscribes cleanly and records why |

**Disk degradation ladder** (`IDEAS-FRONTIER.md` §7): normal → **tail stops, core continues** →
**core depth only** → **halt with loud alert**. Automatic in both directions. Core dies last,
because it is least replaceable.

## 7. Verification

### 7.1 Tests before unattended operation

- **Round-trip byte-exactness** — write frames, read back, assert identical to what arrived
- **Raw/index alignment** — property test: index line *n* always describes raw line *n*, including
  escaped-newline frames
- **Synthetic gap injection** — replay a stream with known dropped sequences; assert the ledger
  records *exactly* those gaps, no more, no fewer
- **Crash safety** — `kill -9` mid-write; assert no torn line breaks alignment on restart
- **Prune safety** — assert prune cannot execute against a file whose upload checksum is unverified

### 7.2 Standing artifact — daily capture integrity report

Per venue/stream: frames captured, sequence continuity, gap count by severity, total dark time,
bytes local vs uploaded, checksum failures.

This makes "is capture healthy?" answerable rather than assumed. Per the guardrail lesson already
recorded in memory: **a green test suite proves logic, not that the thing is firing in production.**

### 7.3 Acceptance criteria

**7 consecutive days unattended**, with every gap in the ledger explained, zero unverified prunes,
and zero silent losses.

## 8. Requirements from the universe-wide scanning design note

Source: `~/research/DESIGN-NOTE-universe-wide-scanning.md`. Both are unrecoverable if missed.

### R1 — The broad tail must be genuinely broad

The scanning idea's value is a **breadth** play (IR ≈ IC × √breadth). Widening the tail is cheap
today and impossible to backfill. **Do not settle for a token 20 symbols.**

### R2 — Point-in-time universe membership must be recorded

Backtesting "watch every symbol" against *today's* symbol list silently conditions on survival —
**survivorship bias entering through the universe definition rather than through prices.** No
purge/embargo scheme catches it.

`universe_tracker` must capture as timestamped first-class events: **listings, delistings, symbol
renames, contract migrations, status changes** (halts, maintenance, settlement changes), plus a
daily full snapshot.

Trivial now. Impossible to reconstruct later — exchanges do not reliably publish historical universe
membership, and third-party reconstructions are exactly the class of unverifiable secondary source
this project has already been burned by once.

## 9. Verified environment facts

Established by direct check on 2026-08-02, not assumed.

| Fact | Status |
|---|---|
| Disk | 96 GB total, **91 GB free** |
| Compute | 12 cores, 29 GB RAM — not a constraint |
| Host | GCE instance `instance-20260801-081737`, project `project-e760fdd7-f8da-46a5-8c4` |
| `gcloud` / `gsutil` | Present (SDK 577.0.0), SA `1095194309870-compute@developer.gserviceaccount.com` active, `cloud-platform` scope |
| `loginctl enable-linger` | **DENIED** — "Access denied" |
| `sudo -n` | **DENIED** — no passwordless sudo |
| `KillUserProcesses` | **false** → detached user processes **survive logout** |
| systemd user unit dir | Writable |

## 10. Blockers and preconditions

### ~~B1 — GCS write access is unproven~~ → **RESOLVED 2026-08-02**

> **Bucket `gs://capture-raw-data4134` created and `roles/storage.objectAdmin` granted to
> `1095194309870-compute@developer.gserviceaccount.com`.** Verified end to end by
> `scripts/verify_gcs_write.sh`, which uploads, reads back, lists, **compares byte-for-byte**, and
> deletes:
>
> ```
> RESULT: GCS write access CONFIRMED for gs://capture-raw-data4134
> ```
>
> The round-trip comparison is deliberate — a successful upload that silently corrupts is exactly the
> failure this project exists to avoid, so "the command exited 0" is not accepted as proof.
>
> **Consequence: `archive_offloader` is unblocked and may now be built.** The local-only-never-prunes
> rule below applies only until it exists. The runway table stays relevant as the deadline for
> building it, not for obtaining access.
>
> Original blocker text follows for the record.

### B1 (original) — GCS write access is unproven (blocking for offload only)

```
ERROR: HTTPError 403: ...does not have storage.buckets.list access to the project
```

The SA has `cloud-platform` *scope* but lacks the IAM *role*. **Nuance:** `storage.buckets.list` is
a **project-level** permission; writing objects to one **specific named** bucket requires
`storage.objects.create` on that bucket, which may still be granted independently. Untested — no
bucket name available, and creating a bucket is an outward billable action not taken unprompted.

**Resolution — either:**
1. Grant the SA `roles/storage.objectAdmin` on a dedicated bucket, **or**
2. Provide a bucket name so a real write test can run

**Gating rule:** the recorder ships and runs **local-only from day one** — capture starts
immediately, nothing irreplaceable is lost — and offload is switched on the moment the write test
passes. No code depends on GCS until proven.

#### Local-only mode semantics — resolves an otherwise fatal contradiction

§7.1 forbids pruning any file whose upload is not checksum-verified. In local-only mode **nothing is
ever uploaded, therefore nothing may ever be pruned.** Without an explicit rule the disk simply
fills. The rule:

- **Local-only mode never prunes and never deletes.** Retention is not enforced; the archive grows
  monotonically at ~1–2 GB/day.
- **The disk degradation ladder (§6) is the only defence**, and it protects capture, not the disk.
- `capture_health` escalates on **free-space runway**, not percentage: warn at **30 days**
  remaining, alert daily at **14 days**, and at **7 days** raise a **hard decision point** —
  by then either offload works, or a human must consciously choose what to stop capturing.
  Runway is computed from measured 7-day average daily bytes, not a fixed threshold.
- **B1 carries a deadline, not just a blocker** — see the two-stage runway below.

#### Disk runway with the 400 GB ceiling

Operator constraint (stated 2026-08-02): **the VM disk can be grown, up to a hard ceiling of 400 GB.**
That extends the runway substantially but does not remove the need for offload — capture is
unbounded and 400 GB is not.

| Stage | Usable | Runway at 1–2 GB/day |
|---|---|---|
| Current disk | 91 GB free | **45–90 days** |
| After growth to ceiling | ~395 GB | **~200–400 days total** |

**Conclusion: GCS offload is not urgent for months, but it is eventually mandatory.** The 400 GB
ceiling is a delay, not an escape — at which point the only remaining levers are offload, or
deliberately capturing less.

#### Disk growth requires a reboot — which makes B2 a prerequisite, not a parallel concern

Verified 2026-08-02: root is **ext4 on `/dev/sda1`**; `/etc/cloud/cloud.cfg` enables the
**`growpart`** module; `growpart` and `resize2fs` are installed but require root
(`open: Permission denied while opening /dev/root`).

Therefore, without sudo the growth path is:

1. Resize the disk in the GCP console — **online, no reboot needed for this step**
2. **Reboot**, so cloud-init's `growpart` extends the partition and filesystem automatically

Step 2 is the problem. **A reboot silently stops capture while B2 is unresolved** (no linger, no
root systemd, so nothing restarts the recorder). The two blockers compound:

> **B2 must be resolved before the first disk resize**, or growing the disk costs an
> unknown-duration capture outage — and the outage is silent, which is worse than its length.

Planning consequence: solving reboot persistence is **not** an optional hardening task to schedule
later. It gates the first disk growth, which the runway table puts at 45–90 days.

### B2 — Reboot persistence unresolved (**upgraded to blocking**, gates first disk growth)

`KillUserProcesses=false` means the recorder survives **logout**. It does **not** survive a VM
**reboot**, because linger cannot be enabled and there is no root systemd access.

**Originally assessed as non-blocking. Upgraded after the disk-growth analysis above:** the only
sudo-free path to a larger disk requires a reboot, so B2 gates the first resize — which the runway
table places at **45–90 days**. It is a scheduled dependency, not a background concern.

Mitigations to evaluate during planning, cheapest first:

1. **User `@reboot` crontab** — untested on this box; verify whether cron runs user `@reboot` jobs
   without linger. If it works, this is the whole solution and costs nothing
2. **External watchdog** — an off-box check that alerts (or restarts via SSH) on capture silence
3. **Obtain linger or root systemd** from whoever administers the GCP project — cleanest, but
   depends on access this account does not have

**Until resolved, `capture_health` must alert on absence, not merely on error** — a stopped recorder
produces no errors at all, which is exactly why silent outages go unnoticed.

## 11. Resolved defaults

Decided 2026-08-02 with stated reasoning. All are reversible before implementation; none block planning.

### Q1 — Core symbol list and tail universe

**Core (deep, L2 depth):** `BTC`, `ETH`, `SOL` on both venues. Three, not five — depth is what
explodes storage, and the runway is the binding constraint until B1 resolves. Expanding the core
later costs only storage; it does not require rework.

**Tail (broad, no depth):** every perpetual instrument each venue lists, discovered dynamically by
`universe_tracker` rather than hardcoded. A static list would silently miss new listings — which
are among the highest-signal events the scanning design depends on (R1).

### Q2 — Local retention window

**Moot until offload works.** Local-only mode never prunes (§10 B1), so retention is not enforced
and the archive grows monotonically. The 7-day window activates only once GCS upload is verified.
Revisit with measured volume at that point, not before.

### Q3 — Alert channel for `capture_health`

**Default: structured JSON lines to `~/capture/health/alerts.ndjson`, plus non-zero exit on a
`capture_health --check` invocation** so any external scheduler or human can poll it.

Deliberately no email/Slack/webhook in this sub-project — that is a credential-bearing outward
integration, and it belongs with the operations sub-project rather than being smuggled into Layer 0.
**The consequence must be stated plainly: until an external channel exists, alerts are pull-only and
nobody is notified of a silent outage.** That is acceptable only because B2 is scheduled work.

### Q4 — Binance spot, futures, or both

**Futures (USDⓈ-M perpetuals) for the core tier.** Three reasons:

1. **It is the tradeable surface.** Hyperliquid is perps-only, so futures keeps the two venues
   comparable and matches what execution will actually touch
2. **Better gap detection** — futures depth carries `pu` (previous final update id), giving explicit
   sequence-chain validation that spot's `U`/`u` alone does not
3. **Richer timestamps** — futures carries both `E` (event) and `T` (transaction); spot has only `E`

Spot is not captured in this sub-project. Adding it later is additive and costs no rework.
