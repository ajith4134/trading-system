# FULL BUILD — master plan

Date: 2026-08-09
Status: **ACTIVE. Decided by the user, over the phased-with-paper-first alternative.**
Authority: order of work only. The goal stays `2026-08-08-final-project-goal-design.md`;
structure stays `ARCHITECTURE.md`; capability list stays `FEATURES.md`.

## The decision, verbatim intent

The user wants the **entire project completed — every plan, idea, feature and goal — before
paper trading starts.** Strict order: everything, then paper. This supersedes §13 step 9's
"wait for observed history" as the *ordering* of work (the observed-history clock still runs
on the calendar and still gates real-money promotion; nothing here changes that statistics).

The recommended alternative (paper engine first, rest in parallel) was put to the user and
declined. Recorded per the counterweight rule: this is the user's ordering to choose.

Two prior mischaracterisations this plan must not repeat:
- **Building everything does not shorten the observed-history wait** — that clock is
  calendar time on the OBSERVED funding dataset (started 2026-08-08) and gates *live*
  promotion, not paper.
- **The board never claims a feature exists until a probe measures it** (Rule 8). The build
  is done when the tile lights, not when the file lands.

## Credentials on hand (verified 2026-08-09)

| Venue | State | Proof |
|---|---|---|
| binance | WORKING | signed `GET /fapi/v1/commissionRate` succeeded, tier `account` |
| bybit | PRESENT, unexercised | key in sops store; no signed fetcher built yet |
| coinbase | INCOMPLETE | entry lacks `api_key` or `api_secret` — blocked on user |
| hyperliquid | n/a | public endpoints |

Keys unblock what §13 called impossible: `execution.state_recovery` against venue truth,
key scoping, RX-005, and a bybit liquidation feed for the FAILING liquidation tile.

## Phases — strict order, paper last

Sequencing rule inside each phase: **wire the already-built-but-unreached modules first**
(`spread_and_depth`, `drawdown_distribution`, `participation_calibration`), then build new.
Every module ships with tests, an axis verdict (§1a.6), a catalogue entry, and a wall tile
whose probe actually runs. Every phase ends with a ledger reconciliation sweep and a push
to GitHub (Rule 9).

### A. Data & ingestion completion (FEATURES §1)
Open interest; mark-vs-index-vs-oracle per venue; spot-perp basis term structure;
cross-venue liquidity-weighted consolidated price; stablecoin peg monitor; per-feed
data-quality score; wash-trading discount; provenance-flagged backfill (Phase 0's last
item); **liquidation via bybit public stream** (kills the FAILING tile without a paid
feed); exchange reserve/netflow (P2); coinbase venue once its key is complete.

### B. Feature engineering (FEATURES §2)
Multi-horizon realized vol; HAR-RV; depth-weighted OFI (never level-1); absorption
detection; microprice; Kyle's lambda; fractional differentiation (`fracdiff`);
triple-barrier labelling; meta-labelling; sample uniqueness + sequential bootstrap;
funding/basis spread features; funding-hour effects; cross-sectional ranking;
vol-regime decile; staleness timestamp on every value. All through the clock-gated
reader; nothing reads the store directly.

### C. Models (FEATURES §3)
Mandatory linear/naive baseline; gradient-boosted trees; rolling walk-forward retrain;
model registry with aliases; stacked ensemble; champion/challenger scaffolding;
meta-model over the experiment ledger. Every model must beat the baseline through the
existing promotion pipeline — the gates are already live and refuse things.

### D. Bots — brain 1 (bull-bear-profit-agents-spec.md)
BULL, BEAR, PROFIT-TAIL as three independent bots: own features, own architecture, own
training pipeline, own validation history. Authority map per the spec. This is the
largest single phase.

### E. Portfolio & risk (FEATURES §5-6, ARCHITECTURE Phase 5)
Allocator; position sizer; correlation breaker; VX-011 drawdown ladder wired to gross
(its first consumer — currently derived, not acted on); regime conditioning; drawdown
distribution finally driven.

### F. Execution & live-prep (FEATURES §7)
`state_recovery` driven against binance venue truth (key now exists); key scoping;
RX-005 closed; bybit signed fetcher; order-type coverage per EX-004.

### G. Brains 2-3 + intelligence layer
Second and third brains per the nine-bot maturity map; Dual-LLM quarantine for
news/social (P3); macro context feeds; on-chain (only if a DEX venue is decided —
currently none is, so this sub-item lands as DECLINED-unless-decided in the ledger).

### H. Dashboard as full product
The wall already auto-regenerates and serves at the tunnel URL. This phase adds: a
build-progress board over this plan's phases (generated, Rule 8 — each phase row shows
tiles-lit / tiles-total from the catalogue, never a hand-typed percentage), and the
feature-catalogue page grouped by phase with per-feature probe status.

### I. Ledger closure
Final sweep of the 1,485-row ledger: every PLANNED row CLAIMED or DECLINED with a
reason; every UNRESOLVED row resolved or DECLINED. Declining is allowed, forgetting is
not. `all expected slices present` must still hold.

### J. THEN: paper engine + paper trading
Build `2026-08-08-paper-execution-engine-design.md` as specced (tiered symbols, replay +
forward, dual maker/taker accounting, promotion gated on pessimistic fills). Forward
supervisor joins the boot chain. Paper trading starts here — after everything above, per
the user's explicit ordering.

## Definition of done, whole plan

1. Feature catalogue shows **0 rows NOT BUILT** that are not explicitly DECLINED/BLOCKED,
   and every BLOCKED row names what it waits on (a key, a paid feed, a user decision).
2. Ledger: 0 UNRESOLVED.
3. Test suite green; count recorded, not asserted.
4. Every module has an axis verdict; `integrity.unsupported_claims` reports 0.
5. Paper engine running as a supervised process, journalling fills under both
   accountings, visible on the wall.

## Honest scale statement

167 catalogue features + 555 planned ledger rows across phases A-I is **multi-week work
even at maximum pace**, and paper trading sits at the end of it by the user's choice.
Progress is visible continuously on the wall — tiles flip as probes light, and the
build-progress board (phase H, pulled early: built in the first work block so the user
can watch from day one) shows phase-by-phase counts.
