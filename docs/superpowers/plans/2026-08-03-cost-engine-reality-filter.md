# Cost Engine — the Reality Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Phase 1 of the build order in `ARCHITECTURE.md` §3.** Phase 0 (bitemporal store, clock-gated access) is built and running.

**Goal:** No signal may be accepted without first asking what it costs to trade. The engine
returns the round-trip breakeven for a `(venue, symbol, notional, order type)` and refuses to
answer when the inputs it would need are missing or stale — because a cost engine that guesses
is worse than none, and manufactures exactly the edge it exists to disprove.

**Why this is the biggest single gap** (`ARCHITECTURE.md` §Layer 1): fees dominate breakeven by
roughly 5–10× over slippage and adverse selection at this size. Without this gate, paper mode
manufactures edge that evaporates live. **Venue and fee-tier selection outrank every execution
algorithm that could be written.**

**Architecture:** One query path, `quote_round_trip_cost(...)`, returning a `CostQuote` that
carries not just a number but the provenance and age of every input that produced it. Fees come
from a versioned fee table with a recorded source per venue; spread and depth come from the Layer 1
store **through the clock-gated reader only**, never from a live socket, so a backtest cannot
price a fill against a spread from its own future.

**Tech Stack:** Python 3.12, pandas/pyarrow (already present), `store.clock_gated_reader`,
`capture.rest_poller` for the venues that publish a schedule, pytest.

## Global Constraints

- **Python `>=3.12,<3.13`** — matches `pyproject.toml`; do not widen it.
- **New package `src/cost`** must be added to `[tool.hatch.build.targets.wheel] packages`.
- **Every market-data read goes through `store.clock_gated_reader`.** No exceptions, no direct
  Parquet reads, no live REST call in a pricing path. This is the whole reason Layer 1 exists.
- **A quote may never be produced from a default.** No `fee_bps = 10.0` fallback anywhere. A
  missing or stale input produces a refusal that names what was missing (see Rule 8).
- **Fees are basis points as `Decimal`, never float.** A rounding error in a breakeven that gates
  every strategy is not an acceptable class of bug.
- **Timestamps are int64 nanoseconds UTC** everywhere, matching Layer 0/1.
- Docstrings explain *why*, matching `src/capture` and `src/store`.
- Tests are named for the behaviour they defend.

## Measured Facts This Plan Depends On

Probed from this host on **2026-08-03**. Not assumed, and the first two rows change the design.

| Venue | Fee schedule reachable unauthenticated? | Evidence |
|---|---|---|
| **Hyperliquid** | **Yes, in full** | `POST /info {"type":"userFees","user":"0x000…000"}` returns `feeSchedule` with base rates **and every VIP tier**. The zero address is accepted. |
| **Binance USDⓈ-M** | **No** | `/fapi/v1/commissionRate` → **401** `{"code":-2014}`. `/sapi/v1/asset/tradeFee` → **400**. `/fapi/v1/exchangeInfo` carries `liquidationFee` only — no maker/taker anywhere. |

Live Hyperliquid rates, fetched 2026-08-03:

| | Perp maker (`add`) | Perp taker (`cross`) | Spot maker | Spot taker |
|---|---|---|---|---|
| Base tier | **1.5 bps** | **4.5 bps** | 4 bps | 7 bps |
| First VIP cutoff | $5M 30-day notional → 1.2 / 4.0 bps | | | |

**Consequence for the design:** "live-fetched fee schedules per venue" is achievable for
Hyperliquid **today** and is not achievable for Binance without an API key — and exchange
selection is still open (`DECISIONS.md` §12.1), so no key exists. The engine therefore cannot
depend on live fetch being available, and must not silently substitute a guess where it is not.
Hence the declared-table-plus-verifier design below, rather than a pure fetcher.

**Corollary already settled** (`ARCHITECTURE.md` §Layer 1): no rebate tier is reachable at this
size — VIP needs $5M–$250M+ 30-day volume — so the tier input is base tier until measured
otherwise, and the engine should say so rather than model an aspiration.

## File Structure

| File | Responsibility |
|---|---|
| `src/cost/fee_schedule.py` | The declared fee table, its provenance, and staleness rules. No I/O. |
| `src/cost/fee_fetcher.py` | Fetches a venue's published schedule where the venue publishes one; records what it got and when. |
| `src/cost/spread_and_depth.py` | Half-spread and depth-consuming impact for a notional, read through the clock gate. |
| `src/cost/funding_carry.py` | Funding paid/earned over a holding period, from the `premiumIndex` archive. |
| `src/cost/round_trip_cost.py` | Composes the above into a `CostQuote`, or a typed refusal. |
| `src/cost/cli.py` | `python -m cost.cli --venue … --symbol … --notional …` for operator use. |

---

## Task 1 — The fee table and its provenance

- [ ] `FeeRate` carries `maker_bps`, `taker_bps`, both `Decimal`.
- [ ] `FeeSchedule` carries `venue`, `instrument_kind` (perp/spot), `FeeRate`, `tier`,
      `source` (one of `venue_api`, `declared`), `fetched_at_ns`, and `source_detail` naming the
      endpoint or document it came from.
- [ ] A schedule whose `source` is `declared` is **valid but marked unverified** — it is a written
      claim, not a measurement, and every quote built on it must inherit that mark.
- [ ] `is_stale(now_ns, max_age_ns)` — a fetched schedule ages; a declared one is stale from birth
      for the purpose of any decision that moves capital.
- [ ] Seed the table with the Hyperliquid rates above (`source=venue_api`, with the real
      `fetched_at_ns`) and Binance as `declared`, citing the public fee page and the 401 that
      blocks fetching it.
- [ ] **Tests:** a declared schedule never reports itself verified; staleness is computed from
      `fetched_at_ns` and not from process start; `Decimal` in, `Decimal` out, no float anywhere.

## Task 2 — Fetching what the venue will publish

- [ ] `fetch_hyperliquid_schedule()` — `POST /info {"type":"userFees", "user": <address>}`, parsing
      `feeSchedule.add` / `.cross` and the `tiers.vip` ladder. Defaults to the zero address, which
      is measured to work, so a schedule can be read before an account exists.
- [ ] `fetch_binance_schedule()` — **exists and raises `FeeScheduleUnavailable`**, naming the 401
      and the fact that it needs a key. A missing function invites a future caller to assume the
      fetch was simply never wired; an explicit refusal records *why*.
- [ ] Persist each fetch into the raw archive under its own stream (`userFees`), reusing
      `capture.rest_poller` — a fee schedule is market data and a change to it is an event this
      system must be able to reconstruct after the fact. Cadence: hourly, not 1 Hz.
- [ ] **Tests:** the real captured `userFees` body (checked in verbatim as a fixture, like
      `REAL_BINANCE_PREMIUM_INDEX_BODY`) parses to 1.5/4.5 bps; a body missing `feeSchedule`
      raises rather than defaulting; `fetch_binance_schedule` raises the typed error.

## Task 3 — Spread and impact, through the clock gate

- [ ] `half_spread_bps(venue, symbol, at_ns)` — from the archived L2 book, read via
      `clock_gated_reader` at `at_ns`. Refuses if no book is available at or before that instant.
- [ ] `impact_bps(venue, symbol, notional, side, at_ns)` — walk the archived book levels and
      compute the volume-weighted price for that notional against the mid. Refuses if the book's
      visible depth cannot fill the notional, rather than extrapolating past the last level —
      "the book could not fill this" is the answer, and it is a real one.
- [ ] **Leakage test, the important one:** ask for a quote at `t` and assert the values are
      identical whether the store also holds data from `t+1h` or not. A cost engine that is
      cheaper in backtest than in live is the single failure this whole layer exists to prevent.
- [ ] **Tests:** a notional larger than the whole book refuses; a one-level book still prices a
      small clip; the reader is the only I/O path (assert no direct Parquet/network use).

## Task 4 — Funding, for anything held across a settlement

- [ ] `funding_cost_bps(venue, symbol, held_from_ns, held_to_ns, side)` — count the settlements
      crossed and apply the archived rate at each, from the `premiumIndex` stream now being
      captured (`lastFundingRate`, `nextFundingTime`).
- [ ] **Venue asymmetry is explicit** (`FEATURES.md` §1, marked `[MISSED]`): Hyperliquid funds on
      *oracle* price and settles hourly; Binance USDⓈ-M funds on *mark* and settles 8-hourly. One
      shared assumption is wrong at both ends. Encode the settlement schedule per venue.
- [ ] **Tests:** a position held across zero settlements pays zero; across three pays three
      archived rates, not three copies of the latest one; the sign flips with side.

## Task 5 — The round-trip quote

- [ ] `CostQuote` carries `breakeven_bps`, the itemised `fee_bps` / `spread_bps` / `impact_bps` /
      `funding_bps`, and an `inputs` list naming each source with its age and whether it was
      verified. **A quote is a measurement with provenance, not a number** (Rule 8).
- [ ] `quote_round_trip_cost(venue, symbol, notional, order_type, at_ns, holding_ns)` composes
      the parts; `order_type` selects maker vs taker on each leg.
- [ ] `CostRefused` is a typed outcome, not an exception to be swallowed, and names the missing
      input. Callers must handle it; there is no numeric fallback to accidentally use.
- [ ] **The gate:** `is_signal_viable(expected_edge_bps, quote)` — the one call a strategy makes.
      It returns false when the quote is a refusal, and false when edge does not clear breakeven.
      **A strategy cannot reach a fill path without passing through this.**
- [ ] **Tests:** a taker round trip on Binance base tier prices at ≈20 bps, matching the figure
      `ARCHITECTURE.md` states, so the engine and the design document cannot silently disagree;
      an unverified fee marks the whole quote unverified; a refusal never coerces to a number.

## Task 6 — Make it visible

- [ ] `probe_cost_engine` in `statuswall/evidence.py`, and map it to the `FEATURES.md` cost rows.
- [ ] The tile reports what it measured: which venues have a **verified** schedule, the age of the
      newest fetch, and whether any quote path is running on `declared` numbers. A cost engine
      quoting from unverified fees must **not** render green.
- [ ] Add the `[ ]` → catalogue rows to `FEATURES.md` if the wording does not already cover
      "Cost Engine" — `verify_probe_coverage` fails loudly on a probe with no feature, by design.
- [ ] **Test:** the tile reads unverified while Binance is `declared`, and only goes green for a
      venue whose schedule was actually fetched.

---

## Verification for the whole plan

Per Rule 0, stated before starting, and to be run and reported at the end:

1. `pytest -q` — full suite green, including the leakage test in Task 3.
2. `python -m cost.cli --venue hyperliquid --symbol BTC --notional 10000` prints a real quote
   whose fee component matches the live-fetched 1.5/4.5 bps.
3. The same command for `binance` prints a **refusal or an explicitly-unverified quote**, never a
   confident number.
4. The status wall regenerates and the cost tile reflects (2) and (3) — not asserted, measured.

## Out of scope, deliberately

- **TWAP/VWAP/Almgren-Chriss slicing.** `ARCHITECTURE.md`: clips here are 0.0005–0.05% of Binance
  BTC ADV (~$19B/day), thousands of times below where slicing helps.
- **Rebate-tier modelling.** No reachable tier at this size; base tier until measured otherwise.
- **The Capacity Model.** It belongs to this layer but needs live-vs-shadow divergence to measure,
  which needs a running strategy. It follows Phase 4, not this plan.
