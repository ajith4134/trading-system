# Declared capital, per-trade leverage, and trade excursions

**Rulings discharged:** RL-040 (declared capital is a budget, shared pool, margin basis),
RL-041 (per-trade leverage on both bots), RL-042 (peak profit and loss per trade).
**Method constraint:** RL-038 — this EDITS the running system. No parallel engine, no second
journal, no second board. Every step must leave one system that still passes its probes.
**Scope:** perp and spot only (RL-036, RL-039). Dated and options are switched off.

---

## Why this comes before the swing conversion

Measured 2026-08-19 from the live journals, using the system's own `usd_rate_of` rule:

| | perp | spot |
|---|---|---|
| OPENs | 3,345 | 2,444 |
| unconvertible, excluded from every capital figure | 24 (1%) | 473 (19%) |
| **median margin per trade** | **0.00014 USDT** | **0.00066 USDT** |
| p75 | 0.0006 | 0.11 |
| max | 129.53 | 130.19 |
| turnover | 2,342.86 | 3,817.17 |

Turnover reconciles to the segment wall, so the measurement is faithful.

Capital per trade spans **nine orders of magnitude** — the min-to-max spread measured by
`probe_margin_per_trade_within_band` is 6.1e9 on perp and 3.3e9 on spot — from one line: `bot_registry.py` declares
`quantity=Decimal("0.002")`, a fixed *base-asset* quantity applied to every symbol in a
universe-wide scan (RL-009, RL-014). 0.002 BTC is $130. 0.002 FLOKI is $0.00000004.

The consequence is that the system currently cannot tell you whether it makes money. The
published `+1.8378` realised P&L is the P&L of a handful of BTC trades; the other ~3,300 moved
nothing measurable. The published `48.8%` winrate weights a $130 trade and a $0.0000001 trade
equally. Converting to swing (RL-035) without fixing this would measure a swing edge with the
same broken ruler.

---

## 1. The declaration — `~/capture/segment/capital.json`

One file. Both bots read it. It is the only place capital is declared.

```json
{
  "declared_at": "2026-08-19T14:30:00Z",
  "portfolio_usdt": "2000",
  "per_bot_cap_fraction": "0.60",
  "min_margin_per_trade_usdt": "5",
  "max_margin_per_trade_usdt": "50",
  "leverage": {
    "rule": "volatility_targeted",
    "ceiling": { "perp": "20", "spot": "5" },
    "floor": "1",
    "target_annual_vol_pct": "40"
  },
  "spot_borrow_annual_pct": "8.0",
  "maintenance_margin_rate": "0.005"
}
```

**Owned by** a new `src/segment/capital_declaration.py`. It parses, validates and reloads —
nothing else in the tree may read the raw file.

**Reload semantics (RL-030 pattern).** `stat()` the file each poll; parse only when `st_mtime_ns`
changed. A change applies to *new entries only* — an open position keeps the margin and leverage
it was opened under, because retroactively re-margining a live position is a different trade
than the one the journal recorded.

**Validation is refusal, not repair.** Every field is a `Decimal` parsed from a string, and the
file is rejected whole if any of these fail:

- `portfolio_usdt > 0`
- `0 < per_bot_cap_fraction <= 1`
- `0 < min_margin_per_trade_usdt <= max_margin_per_trade_usdt <= per_bot_cap_fraction × portfolio_usdt`
- `1 <= floor <= ceiling[segment]` for each of perp and spot
- `leverage.rule` is a name the code implements

**Failure behaviour, and why it is this way.** A rejected or missing file means **the bots stop
opening new positions** and say so with reason `NO_CAPITAL_DECLARATION`. They do *not* fall back
to a default and they do *not* keep the last good file across a restart. Existing positions are
still managed and closed normally — refusing to manage an open position would be worse than any
config error. A default here would be capital chosen by nobody, which is precisely the failure
this file exists to remove.

Because the bots are running now, deployment order matters: **write the file first, then deploy
the code.** A `scripts/write_capital_declaration.sh` emits a starting file so there is never a
window where a live bot has code that demands a file that does not exist.

---

## 2. The shared pool, and the one hard problem in this design

`portfolio_usdt` is a **budget that is spent**, overturning `capital_accounting.py:66`
(*"a measurement base, not a limit"*). The gate now reads it.

Two rules:
- **Pool headroom** = `portfolio_usdt − Σ open margin across BOTH bots`
- **Per-bot cap** = `per_bot_cap_fraction × portfolio_usdt`, so neither bot can take it all

### Why a naive implementation is wrong

perp and spot are **two independent OS processes** polling every 6 seconds. If each computes
headroom from its own view of the world, both can read "800 USDT free", both open 50 USDT, and
the pool is over-committed. It is a classic lost update, and at 6-second polls across 800
symbols it will not be rare — it will be constant.

### The reservation ledger

`~/capture/segment/pool.ndjson`, guarded by `fcntl.flock` on the same file. Both processes are
on one box and one filesystem, so `flock` is a real mutex and costs microseconds.

Sequence for one intended open:

1. take the exclusive lock
2. replay the ledger to current open margin per bot
3. check `pool headroom`, `per-bot cap`, `min/max margin per trade`
4. append a `RESERVE` row, or refuse with a named reason
5. release the lock
6. *only then* journal the fill

A `RELEASE` row is appended on every close. A reservation whose bot dies is reclaimed by a
startup sweep that releases any reservation with no matching open position in that bot's
journal — otherwise a crash silently shrinks the pool forever.

**Refusal reasons, all named and counted on the tile**, because a refusal that reports as an
abstention is a bot that looks idle when it is actually broke:

`POOL_EXHAUSTED` · `BOT_CAP_REACHED` · `BELOW_MIN_MARGIN` · `VENUE_MIN_NOTIONAL` ·
`NO_USD_RATE` · `STOP_OUTSIDE_LIQUIDATION` · `NO_CAPITAL_DECLARATION`

---

## 3. Sizing — margin in, quantity out

Capital per trade means **margin posted** (RL-040). This replaces `SegmentBot.quantity`, which
is deleted rather than defaulted, so nothing can silently keep the old fixed-quantity behaviour.

```
leverage  = choose_leverage(...)                       # §4, clamped to [floor, ceiling]
margin    = max_margin_per_trade_usdt                  # see below
notional  = margin × leverage                          # USDT
quantity  = notional / (price × usd_rate)              # base asset
```

**Margin is the declared maximum, and the minimum is a refusal floor.** The user asked for a
minimum and a maximum, not a sizing curve, so no curve is invented: the band bounds the trade at
the top and stops the bot dribbling out unmeasurable positions at the bottom once the pool runs
low. A sizing rule — risk-parity off the stop distance, or confidence-scaled — is a separate
decision and gets its own row when it is taken.

**Lot rounding is NOT implemented, and the journal says so.** *Measured 2026-08-19: no venue lot
size, step size or minimum notional exists anywhere in this repository.* There is nothing to
round to. So the quantity is unrounded, every OPEN carries `lot_size: null` and
`lot_rounding: "not modelled"`, and a real venue could reject the size. That is a known gap
stated on every fill rather than a silent one — and it needs its own plan row before this system
could place a real order.

Three refusals fall out of this, and each is a trade the current system takes blindly:
- rounded quantity is `0` → `VENUE_MIN_NOTIONAL`
- re-derived margin `< min_margin_per_trade_usdt` → `BELOW_MIN_MARGIN`
- no journalled `usd_rate` → `NO_USD_RATE` (this is the 19% of spot opens above)

**Every one of these is journalled on the OPEN**: `margin_usdt`, `notional_usdt`, `leverage`,
`usd_rate`, `lot_size`, `pre_round_quantity`. An audit that cannot recompute the size from the
journal is not an audit.

---

## 4. Per-trade leverage (RL-041)

`rule: "volatility_targeted"` — the default, and named in the file so it can be swapped:

```
realised_vol = annualised realised volatility of that instrument   # src/features/realized_volatility.py
leverage     = clamp(target_annual_vol_pct / realised_vol, floor, ceiling[segment])
```

Quiet instrument → more leverage; violent instrument → less. Each position therefore carries
roughly the same *risk*, which is the point. **Not confidence-scaled**: both bots journal
`calibrated=False`, and scaling leverage by an uncalibrated confidence compounds the
miscalibration exactly where it costs most.

An instrument with too few samples for a volatility estimate gets `floor`, not the ceiling, and
the reason is journalled. Unknown risk is not an argument for more leverage.

### 4a. Liquidation distance — the check that makes leverage safe

At `L` leverage an isolated position liquidates on roughly a `1/L` adverse move. A hard stop
wider than that **can never fire** — the position is liquidated first, and the stop the risk gate
believes it has is fiction.

```
liquidation_distance = (1 / leverage) − maintenance_margin_rate
require: hard_stop_fraction < liquidation_distance × 0.8      # 20% headroom
```

If it fails, **reduce leverage until the stop fits**; if it still fails at `floor`, refuse with
`STOP_OUTSIDE_LIQUIDATION`. Both the reduction and the refusal are journalled with both numbers.

### 4b. Spot borrow cost — no free loans

Leveraged spot is borrowed money. Not charging for it makes the spot bot's returns a free loan
that inflates every figure it publishes, and it would inflate them *more* the more leverage it
took — a bug that rewards recklessness.

```
borrowed  = notional − margin
interest  = borrowed × spot_borrow_annual_pct × (held_ns / year_ns)
net_pnl   = gross_pnl − interest − fees
```

Charged at close, journalled as its own field. Under RL-035's swing horizons this is not a
rounding error: 5x spot held five days at 8% annual is ~44 bp of the margin.

Perp pays funding instead, which `src/cost/funding_carry.py` already models and which becomes
material at swing horizons for the same reason.

---

## 5. Peak profit and loss per trade (RL-042)

**Current state, measured:** `peak_favourable` is declared at `profit_tail.py:180` and copied at
`live_engine.py:575`, and **is never assigned anywhere in the codebase**. It has been `None` for
every position that has ever existed. There is no adverse counterpart. This field looks built
and is not, which is worse than absent.

Tracked on **every poll** for every open position, and carried through the frozen-dataclass
replacement `live_engine` already performs on ratchet:

| journalled on CLOSE | meaning |
|---|---|
| `peak_favourable_fraction` | best move in the position's favour, fraction of entry |
| `peak_adverse_fraction` | worst move against it, fraction of entry |
| `peak_profit_usdt` | `peak_favourable_fraction × notional_usdt` |
| `peak_loss_usdt` | `peak_adverse_fraction × notional_usdt` |
| `excursion_samples` | how many polls the position was observed at |

**These are SAMPLED, not true extremes, and must be labelled so on the board.** The engine polls
every 6 seconds and cannot see between polls, so each figure is a *lower bound* on the real
excursion. `excursion_samples` sits beside them so a two-sample trade is not read as a measured
one. Rule 8: the display says what it actually measured.

**What these two columns buy you.** A trade that reached +0.9% and closed at −0.2% is a profit
tail that let go, and today that failure is completely invisible — the journal records only the
−0.2%. Across the population, `peak_favourable` versus realised P&L is the direct measurement of
whether the take-profit is set too wide or the ratchet too loose, and `peak_adverse` versus the
hard stop measures how close the losers actually came to stopping out. Neither question is
answerable from anything the system records today.

---

## 6. What the board shows

Edited in place (RL-038). No new page.

- **Tile event table** (`segment_tiles.py:247`) gains `margin USDT`, `leverage`, `peak +`,
  `peak −`. Sourced from the journalled fields, never recomputed at render — a display that
  recomputes can disagree with the journal, and then neither is trustworthy.
- **Capital header** on each tile: portfolio total, this bot's cap, margin in use, headroom,
  and the count of each named refusal reason.
- **A declaration banner**: the `declared_at` of the loaded file and its mtime, so a stale or
  rejected declaration is loud rather than assumed. If the file is rejected the tile renders
  `NO CAPITAL DECLARATION` and is not green (Rule 8).

---

## 7. Build order

Each step lands with tests and leaves one running system (RL-038).

1. `capital_declaration.py` + `scripts/write_capital_declaration.sh` — parse, validate, reload.
   Nothing reads it yet. Zero behaviour change.
2. Excursion tracking + journalling (§5). Pure addition to the journal, no sizing change.
   **This is the only step safe to ship while the bots run as they are**, and it starts
   collecting the data that judges everything after it.
3. Board columns for §5. Read-only.
4. Pool ledger + `flock` reservation (§2), with the startup reclaim sweep. Still not enforced.
5. Sizing switch (§3) — delete `SegmentBot.quantity`, enforce the pool. **Behaviour change.
   The paper track record before and after are not comparable and the journal must mark the
   cutover.**
6. Leverage (§4), including the liquidation check and spot borrow cost.
7. Board capital header and declaration banner (§6).

## 8. How this is verified (Rule 0)

- Re-run `probe_margin_per_trade_within_band` after step 5: it must go from `degraded` to `ok`,
  the distribution collapsing from a 6.1e9 spread into the declared `[min, max]` band, with every excluded trade carrying a
  named refusal reason. **This is the acceptance test for the whole document.**
- A concurrency test that runs two reservation loops against one pool file and asserts total
  reserved never exceeds `portfolio_usdt` — the lost-update failure of §2, proven absent rather
  than assumed.
- A liquidation test asserting no journalled OPEN has `hard_stop_fraction` beyond its
  leverage's liquidation distance.
- Excursions: assert `peak_favourable ≥ realised return ≥ −peak_adverse` on every closed trade
  in the live journal. A violation means the sampler missed the close.
