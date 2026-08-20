# COMPARISON — this system against the named public trading systems (RL-045, part two)

**Written 2026-08-20.** Part one is `RECONCILIATION-2026-08-19.md`. This part answers the second
half of RL-045: *"search how other famous crypto bots … compete to what we did or implemented"*,
on the three axes the ruling names.

**Sources, all read from source code, none from marketing:**
- Ours: `docs/SELF-FACTS-2026-08-20.md` — every row cites `path:line` in this repo.
- Theirs: 15 repositories cloned and read 2026-08-20, one note each under
  `~/ajit-segment-bots/docs/research/repo-harvest/`, every cited path checked against the clone
  (`verify_citations.py`, 0 missing). Commercial platforms: `~/research/commercial-crypto-bot-platforms.md`.

**Grading.** Per mechanism: **A** = working, tested, the thing a desk would copy; **B** = present and
sound, narrower; **C** = present but unwired, partial, or contradicted by its own labels;
**D** = absent. Grades are about what the code does, not what a README says. Where a public
system's mechanism does not apply to intraday crypto (equities-only, daily bars) it is not graded
up for existing.

---

## 0. The one-paragraph verdict

On **paper-trading mechanics** this system is a **C**: the pieces a real simulator needs — slippage,
funding, depth, latency, liquidation — are each written as a module and **none is wired into the
fill path**; the two paper engines do not share fill code although one's docstring says they do.
NautilusTrader, Hummingbot and Jesse are A/B here and show exactly what wiring looks like.
On **intelligence and learning** this system is a **B**, and ahead of every public system read:
none of the fifteen has conformal abstention, online calibration, a DSR/PBO promotion gate and
automatic ablation probes — but only 14 hand features reach the model, bull and bear share one
booster, the out-of-regime probe is hard-coded to fail, and nothing reads outside price data.
On **architecture** this system is a **B−**: process-per-segment, fsynced WAL with SHA-256 client
ids, flock-guarded capital pool and a probe-driven status wall are all real; the hash chain the
names imply does not exist, the kill switch cannot drop the network on this host, and the
architecture doc three modules cite is not on disk.

---

## 1. Axis 1 — paper-trading mechanics

| mechanism | ours | grade | best public reference | grade | gap |
|---|---|---|---|---|---|
| Fill price | dual optimistic/pessimistic, never blended (A); single taker touch (B) | B | Hummingbot walks live book depth per fill (`OrderBook.simulate_buy`); Nautilus 11 named fill models with depth numbers | A | ours has no depth at all — every fill is "the touch" |
| Slippage | `impact_bps` exists, imported by nothing; `round_trip_cost` hardcodes 0 | C | Lumibot SMART_LIMIT asymmetric per-side ladder; Lean volume-share model (refuses crypto, honestly) | B | unwired module is a D in effect |
| Fees | hardcoded 2/5 bps labelled "VERIFIED" in one file, "DECLARED/unverified" in another | C | Nautilus six fee models behind one trait; Hummingbot quantises from live `exchangeInfo` | A | label contradiction is a Rule 8 defect |
| Funding / carry | `funding_carry.py` correct schedule, wired into nothing | C | Nautilus `settle_funding_rate`: idempotent, keyed on boundary, survives replay | A | — |
| Partial fills | yes, min(remaining, available), PENDING→PARTIAL→FILLED | B | Hummingbot closed `OrderState` enum + immutable updates | A | close |
| Latency | none; look-ahead prevented by event ordering only | D | Nautilus base + per-op latency, min-heap of in-flight commands | A | paper fills here are instantaneous |
| Queue position | scalar participation fraction, limit orders only; market orders ignore it | C | Hummingbot book walk; Qlib `_clip_amount_by_volume` participation cap | B | — |
| Liquidation | distance computed at open, never triggers | C | Jesse manufactures a reduce-only fill at bankruptcy price each candle; Nautilus `process_liquidations` on equity ≤ maintenance×ratio | A | a leveraged paper book that cannot be liquidated overstates survival |
| Order state machine | 5 states, illegal transitions raise | B | Hummingbot / OctoBot one state class per phase | A | close |
| Idempotency / WAL | fsync-before-transport, SHA-256 client id, resubmit raises | **A** | ccxt `newClientOrderId`; Hummingbot lost-order debouncer | B | **ours is the better design here** |
| Sizing + leverage | vol-targeted leverage, fit-to-stop, capital pool reservation | B | Nautilus fixed-risk size net of two-sided commission; freqtrade unlimited-stake split; Lean fee-aware iterative | A | ours does not net fees from size |
| Stops / targets | 4 rails, ATR-based, ratchet profit lock, same-bar ambiguity resolved | B | Nautilus `trailing_stop_calculate` 3×3 pure math; freqtrade monotonic ratchet with short-side rounding | A | ours measures ATR once at entry |
| Peak excursion | tracked per bar, rides every ExitRecord | **A** | vectorbt drawdown records (equity-level) | B | ours is per-position, the right grain |
| Capital declaration | Decimal-only JSON, whole-or-nothing, mtime reload | **A** | freqtrade `available_capital` back-out | B | — |
| USDT P&L | per-fill usd_rate journalled, three never-blended figures, gross of fees | B | Jesse expectancy decomposition on the ledger | B | "gross of fees" means no net P&L yet |
| Paper ↔ live parity | **no live submission exists; two engines, no shared fill code; docstring claims otherwise** | D | Jesse one `Strategy`, mode-branching; Lean paper = backtest brokerage on live data; Lumibot one code path | A | the largest single gap on this axis |

**Axis grade: C.** The design documents describe an A; the wired code is a C. The difference is
entirely "module exists, nothing imports it" — `spread_and_depth`, `round_trip_cost`,
`funding_carry`, `fee_fetcher`, `capital_declaration` (from `forward_engine`). That is five
imports, not five modules.

---

## 2. Axis 2 — intelligence and learning

| mechanism | ours | grade | best public reference | grade | gap |
|---|---|---|---|---|---|
| Brains | BULL/BEAR/PROFIT-TAIL; one LightGBM shared, BEAR reads it reversed with 0.05 penalty; tail cannot veto | C | TradingAgents: separate bull/bear researcher agents that debate, then a trader | B | RL-011 asked for two bots with own architecture; one reversed model is not that |
| Model class | LightGBM only; ridge-logistic meta-learners exist, uncalled | B | FinRL / TensorTrade deep RL; Qlib model zoo | B | theirs are research toys on daily bars; ours at least runs live |
| Features | 14 hand features from 60 bars; 22 of 23 feature modules unwired | C | FreqAI pipeline: variance → scale → PCA → SVM outlier → DI → DBSCAN | A | ours has the richer feature library on disk and the poorer one in the model |
| Out-of-distribution | none | D | FreqAI dissimilarity index + `do_predict` mask | A | model opines on inputs it has never seen |
| Calibration | 10-bin online reliability map, 30/bin floor | **A** | none of the fifteen | D | **ours alone** |
| Abstention | conformal, α=0.35, coverage audited | **A** | none of the fifteen | D | **ours alone** |
| Promotion gate | DSR ≥ 0.95, PBO ≤ 0.50 (CSCV), SPA, purge retention; plus strict no-regression edge gate | **A** | freqtrade hyperopt (pure in-sample); Qlib workflow | C | **ours alone**, but L7 is not wired to the model path (reconciliation §2) |
| Ablation / randomisation | auto-run after every retrain; ablation **FAILS on perp** | B | none | D | the probe is an A; the result is the reddest flag in the project |
| Regime | vol-decile exists, not wired; out-of-regime probe hard-coded FAIL | C | FinRL turbulence index, same code in backtest and live, forces liquidation | B | — |
| Memory of trades | `Belief` wrapper with half-life; ledger meta-model uncalled | C | TradingAgents reflection memory (prose, daily) | C | both thin |
| Universe | 570 perp + 1,361 spot watched; pooled cross-sectional training; **per-symbol independent inference** | B | freqtrade pairlist filters (spread, volume, TTL) | B | RL-009/014 need cross-sectional ranking at inference; `cross_sectional.py` exists, unwired |
| Research / ingestion | none | D | TradingAgents news + fundamentals agents (yfinance, daily) | C | RL-010/013 unmet |
| LLM use | none | D | TradingAgents LangGraph desk; ground-truth snapshot to stop hallucinated numbers | B | the one LLM idea worth copying is the snapshot, already harvested as `ground-truth-snapshot-builder` |

**Axis grade: B.** The verification machinery (calibration, conformal abstention, DSR/PBO, ablation)
is genuinely ahead of every public codebase read. The learning itself is behind the design: one
model for two bots, 14 features, no OOD gate, no regime input, no cross-sectional inference, and
the only honest ablation says perp's model may contribute nothing.

---

## 3. Axis 3 — architecture

| mechanism | ours | grade | best public reference | grade | gap |
|---|---|---|---|---|---|
| Process model | one OS process per segment, bash supervisor with backoff, per-segment OFF file | B | Nautilus Rust core, actor model; Hummingbot single asyncio process | B | fine |
| Data store | hour-partitioned ZSTD parquet, append-only, snapshot-id dedup | **A** | Jesse SQLite/Postgres; freqtrade feather | B | — |
| Journal integrity | fsynced NDJSON, **no hash chain** despite `integrity/` naming | C | none of the fifteen chain either | C | the blueprint's `journal-integrity-checker` requires one |
| Probes / status | ~30 probes over running artifacts only, bounded tails, worst-first | **A** | Hummingbot status CLI; OctoBot web UI | B | Rule 8 embodied |
| Kill switch | venue halt sticky 30 min; process kill writes KILLED.json; **network drop non-functional on host** | C | Hummingbot `ActiveKillSwitch` 10s PnL loop; Nautilus throttler synthesises local denials | B | a kill switch whose strongest step is known not to work |
| Rate budget | weight-based token bucket, file-backed, flock across processes | B | Hummingbot per-endpoint-class named budgets; ccxt leaky bucket | A | ours is one budget per venue, not per endpoint class |
| Segment isolation | own process, venue, brains, feed, journal; capital pool per-bot cap | **A** | none isolate this way | — | — |
| Config | no central config; per-domain files, Decimal-only validation | B | freqtrade single JSON schema | B | — |
| Reconciliation | capital pool replayed from ledger every call | B | Nautilus 0.01% tolerance, dedup discipline stated in module doc | A | no venue-position reconciliation exists because no venue orders exist |
| Tests | 142 test files | B | Nautilus thousands; freqtrade ~2,500 | A | — |
| Design doc | `docs/ARCHITECTURE.md` cited by three modules, **not on disk** | D | Nautilus docs/ | A | — |

**Axis grade: B−.** Storage, probes and isolation are strong. Three claims in names or docstrings
are not true in code: the hash chain, the network kill, and the architecture document.

---

## 4. Where the public systems are simply ahead, ranked by what it would cost us not to copy

1. **Fill realism** — book walk (Hummingbot), latency (Nautilus), liquidation as a fill (Jesse).
   Without these, paper P&L is an upper bound, and RL-005's "save what gets profits, use on live"
   promotes strategies that live fills would erase. Already harvested into the blueprint as
   `book-walk-fill-pricer`, `order-latency-simulator`, `paper-liquidation-simulator`.
2. **One code path for paper and live** — Jesse, Lumibot, Lean. Ours has two engines and no live
   path. The blueprint's `order-destination-router` + `money-mode-reader` is this design.
3. **Out-of-distribution gate** — FreqAI DI. Harvested as `forecast-distribution-gate` and the
   per-bot `outlier-rejector`.
4. **Per-endpoint rate budgets** — Hummingbot. Folded into `venue-rate-budgeter`'s role.
5. **Funding settled as an event** — Nautilus. Harvested as `funding-settlement-recorder`.

## 5. Where this system is ahead, and should not be talked out of it

1. **Conformal abstention with audited coverage** — no public bot abstains on a statistical
   guarantee. Keep it; wire R4's coverage check so it is measured, not promised.
2. **DSR / PBO / SPA promotion** — the only system read that refuses an edge on trial-count
   grounds. Wire L7 to the model path so the classifier is held to the same bar as the carry strategy.
3. **Probe-only status wall** — nothing on it is asserted. The public UIs show config, not proof.
4. **WAL-before-transport with content-derived client ids** — better than ccxt's random
   `newClientOrderId`, which cannot deduplicate a crash-retry.
5. **Capital pool with per-bot caps replayed from the ledger** — no public system isolates
   capital per bot; they size against one wallet.

## 6. What this changes in the build order

The reconciliation's ranked list (§5 there) stands. This comparison adds one item at the top of
Axis 1 and confirms two:

- **Before any new feature: wire the five unwired modules into the fill path** and make the two
  paper engines one. Until then every paper number is optimistic by an unmeasured amount.
- **L4 ablation failing on perp** is still item 1 overall — a model that does not contribute
  cannot be compared to anything.
- **Two bots, two models** (RL-011, RL-048) — the shared-reversed booster is the design the
  rulings explicitly rejected; the `bull-bot` / `bear-bot` blocks in the blueprint are the
  replacement.

## 7. What this document does not know

- Whether any public system's paper P&L predicts its live P&L either — none publishes that
  measurement. The right comparison is live-vs-paper drift, which neither side has.
- How our conformal coverage holds under the regime break L6 says we have never seen.
- The commercial platforms (`~/research/commercial-crypto-bot-platforms.md`) were graded from
  documentation, not source; they are excluded from the tables above for that reason.
