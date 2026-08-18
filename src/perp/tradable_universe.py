"""PB-01: which perpetuals the bot may trade, and the named reason for every one it may not.

## Why this is not `features.universe_coverage`

That module is a **watch list** and says so: a symbol that went quiet is its most
interesting row, and it deliberately never drops what it stopped seeing. This
module makes the opposite decision - **admission to trading** - and the two must
stay apart. Merging them would give the watch list a reason to forget, and the
question it exists to answer is "is that instrument gone, or did our feed stop".

So everything about *what is out there* is read from `compute_universe_coverage`:
the segment of a key comes from **which dataset carries it** (funding makes a key
a perpetual), and staleness is measured against **each series' own cadence**
rather than a shared constant. Neither is re-derived here.

## What admission adds

Four refusals, in a fixed precedence, each with a name that survives into the
report:

* `NOT_PERPETUAL` - no funding observation carries this key. Nothing here parses
  a venue or symbol name to decide this; `binance-spot` looking like spot to a
  human is not evidence.
* `TOO_LITTLE_HISTORY` - fewer bars than `CADENCE_SAMPLE`, so the series has
  never shown a cadence. Nothing can be said about whether it is fresh, and a
  rolling feature has nothing to roll over.
* `WENT_QUIET` - stale beyond the build's own lag, by this series' routine gap.
  The distinction matters: on 2026-08-16 every symbol in the store read stale
  because the Layer-1 build was an hour behind, and the feeds were fine.
* `ILLIQUID` - bars printed, but no volume and no trades in them. A bar the venue
  emitted is not a market. Scalping it would model fills against a book nobody is
  hitting, and the fill model would happily oblige - which is the flattering
  direction, and therefore the one that goes unnoticed.

## The tier, and the constraint it makes visible

Admission is not one thing, because the data is not uniform. **Measured
2026-08-17: bars cover 2,235 symbols and the `book` dataset covers six** -
BTCUSDT, ETHUSDT, SOLUSDT, BTC-USD, ETH-USD, SOL-USD - and only from 09:00 that
day. The capture configuration is `binance BTCUSDT,ETHUSDT,SOLUSDT ALL`: three
named symbols get depth, `ALL` gets bars.

Order-flow imbalance, microprice and absorption all need the book. So RL-009 and
RL-014 breadth and book-based scalping cannot both be true today, and an
admission list that hid that would be read as "2,235 tradable" - true about bars,
false about the strategy, and the first thing anyone would build against.

`DEEP` means the book is there. `BARS_ONLY` means it is not. Both are admitted,
because refusing the wide universe would throw away what the user asked for, and
calling it DEEP would let a book-based feature be built against symbols that have
no book.

## Counts, never bare percentages

`describe()` reports the denominator with every number. "92% tradable" reads
identically over 12 symbols and over 1,800 and is a different statement about
each.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from features.staleness import CADENCE_SAMPLE, STALE_MULTIPLE
from features.universe_coverage import (
    BARS_DATASET, PERPETUAL, compute_universe_coverage,
)
from store.clock_gated_reader import ClockGatedReader

BOOK_DATASET = "book"
# How far back the bars read reaches. Measured 2026-08-17: an UNBOUNDED read of
# the live bars dataset had not returned after 26 minutes, and hour pruning
# cannot help it - a read with no lower bound needs every hour by definition, so
# the layout change SL-15 delivered buys such a read nothing.
#
# This is not a knob that can be turned to make the universe look larger. The
# bot is intraday on every segment (RL-018), so a perpetual that has not printed
# a bar in a day is not something it could trade today whatever this number
# said. What the window changes is which REASON is reported, never whether a
# symbol is admitted, and the window is written into the report so the
# denominator is never read without it.
DEFAULT_LOOKBACK_NS = 24 * 3_600_000_000_000
# Outside the repo on purpose: this is generated output that reports state, and
# Rule 9 keeps that out of git. Committed, a timestamped admission list would
# read as a fact about the project rather than about one moment.
REPORT_PATH = Path.home() / "capture" / "perp" / "universe.json"

# The four refusals. Strings rather than an enum because they are written into a
# report, a board tile and a journal, and a name that survives serialisation
# unchanged is one fewer thing to keep in step.
NOT_PERPETUAL = "NOT_PERPETUAL"
TOO_LITTLE_HISTORY = "TOO_LITTLE_HISTORY"
WENT_QUIET = "WENT_QUIET"
ILLIQUID = "ILLIQUID"

# The two admission tiers.
DEEP = "DEEP"
BARS_ONLY = "BARS_ONLY"

_COLUMNS = ("venue", "symbol", "observations", "freshness", "age_ns",
            "routine_gap_ns", "age_beyond_build_ns", "traded_volume",
            "trade_count", "admitted", "tier", "reason")


@dataclass(frozen=True)
class PerpUniverse:
    """The admission decision for every perpetual visible at one clock.

    `rows` holds one row per (venue, symbol) CONSIDERED - admitted and excluded
    alike. An excluded symbol that vanished from the output would make the
    excluded count unverifiable, and the count is the acceptance.
    """

    rows: pd.DataFrame
    as_of_ns: int
    excluded_by_reason: dict[str, int] = field(default_factory=dict)
    lookback_ns: int = 0

    @property
    def considered(self) -> int:
        return int(len(self.rows))

    @property
    def admitted_count(self) -> int:
        return int(self.rows["admitted"].sum()) if self.considered else 0

    @property
    def excluded_count(self) -> int:
        return self.considered - self.admitted_count

    @property
    def deep_count(self) -> int:
        if not self.considered:
            return 0
        return int(((self.rows["tier"] == DEEP) & self.rows["admitted"]).sum())

    @property
    def admitted_symbols(self) -> list[tuple[str, str]]:
        if not self.considered:
            return []
        admitted = self.rows[self.rows["admitted"]]
        return [(str(row.venue), str(row.symbol))
                for row in admitted.itertuples(index=False)]

    def _row_for(self, venue: str, symbol: str) -> pd.Series | None:
        if not self.considered:
            return None
        match = self.rows[(self.rows["venue"] == venue)
                          & (self.rows["symbol"] == symbol)]
        return None if match.empty else match.iloc[0]

    def reason_for(self, venue: str, symbol: str) -> str | None:
        """The named refusal, or None where the symbol was admitted or unseen.

        None for "never seen" and "" for "admitted" are different answers on
        purpose: one says nothing was measured, the other says something was and
        it passed.
        """
        row = self._row_for(venue, symbol)
        return None if row is None else str(row["reason"])

    def tier_of(self, venue: str, symbol: str) -> str | None:
        row = self._row_for(venue, symbol)
        return None if row is None else str(row["tier"])

    def describe(self) -> str:
        if not self.considered:
            return ("perp universe: 0 considered - no perpetual has been seen "
                    "through the clock gate at this time")
        breakdown = ", ".join(f"{count} {reason}" for reason, count
                              in sorted(self.excluded_by_reason.items()))
        # The window rides every summary. "1,900 admitted" over an hour and over
        # a week are different statements, and a reader who is not told which
        # one they have will assume whichever suits them.
        window = (f", bars read over the last {self.lookback_ns / 3_600_000_000_000:.0f}h"
                  if self.lookback_ns else "")
        return (f"perp universe: {self.considered} considered, "
                f"{self.admitted_count} admitted "
                f"({self.deep_count} {DEEP}, "
                f"{self.admitted_count - self.deep_count} {BARS_ONLY}), "
                f"{self.excluded_count} excluded"
                + (f" - {breakdown}" if breakdown else "")
                + window)


def _symbols_with_depth(store_root: Path, as_of_ns: int,
                        custodian) -> set[tuple[str, str]]:
    """(venue, symbol) pairs the book dataset carries at this clock.

    A missing book dataset is not an error - it is the state of every store built
    before depth capture landed - and it is the difference between "no symbol has
    depth" and "we did not look". Both produce an empty set here; the count in
    the report is what tells them apart, because DEEP being zero against a
    non-zero considered count is itself the finding.
    """
    try:
        book = ClockGatedReader(store_root, BOOK_DATASET,
                               custodian=custodian).read_as_of(as_of_ns)
    except FileNotFoundError:
        return set()
    if book.empty:
        return set()
    return {(str(venue), str(symbol))
            for venue, symbol in zip(book["venue"], book["symbol"])}


def _traded(bars: pd.DataFrame) -> dict[tuple[str, str], tuple[float, int]]:
    """Volume and trade count per (venue, symbol), for the liquidity refusal.

    Summed over everything the clock gate returned rather than sampled from the
    last bar: one quiet minute on a liquid perpetual is normal, and refusing on
    it would exclude the whole universe at 3am.
    """
    if bars.empty:
        return {}
    columns = set(bars.columns)
    if "volume" not in columns and "trades" not in columns:
        return {}
    grouped = bars.groupby(["venue", "symbol"], sort=False)
    volume = (grouped["volume"].sum() if "volume" in columns
              else pd.Series(dtype="float64"))
    trades = (grouped["trades"].sum() if "trades" in columns
              else pd.Series(dtype="int64"))
    keys = set(volume.index) | set(trades.index)
    return {(str(venue), str(symbol)):
            (float(volume.get((venue, symbol), 0.0)),
             int(trades.get((venue, symbol), 0)))
            for venue, symbol in keys}


def _refusal(row: pd.Series, volume: float, trade_count: int,
             has_bars_in_window: bool) -> str:
    """The first refusal that applies, in a precedence that reports the CAUSE.

    Order matters and is not arbitrary. A symbol with 4 bars is also, trivially,
    a symbol with no measurable cadence and no established liquidity - reporting
    it as ILLIQUID would be true and useless, because the thing to do about it is
    wait for history, not remove it from capture. So the earliest refusal in the
    causal chain wins.

    `has_bars_in_window` is checked before the history count for the same reason.
    A perpetual is established by its FUNDING prints, so one that has stopped
    printing bars entirely still arrives here carrying a handful of funding
    observations - and reporting that as TOO_LITTLE_HISTORY would send someone
    looking for a young listing when what actually happened is that the series
    stopped.
    """
    if row["segment"] != PERPETUAL:
        return NOT_PERPETUAL
    if not has_bars_in_window:
        return WENT_QUIET
    if int(row["observations"]) < CADENCE_SAMPLE:
        return TOO_LITTLE_HISTORY
    gap = row["routine_gap_ns"]
    if pd.notna(gap) and float(row["age_beyond_build_ns"]) > STALE_MULTIPLE * float(gap):
        return WENT_QUIET
    if volume <= 0.0 and trade_count <= 0:
        return ILLIQUID
    return ""


def select_tradable_perp_universe(store_root, as_of_ns: int,
                                  custodian=None,
                                  lookback_ns: int = DEFAULT_LOOKBACK_NS) -> PerpUniverse:
    """Decide, for every symbol visible at this clock, whether the perp bot may trade it.

    Every symbol the store has seen is considered - including spot ones, which
    are refused as `NOT_PERPETUAL` rather than filtered out before counting. The
    difference is the whole acceptance: a population that excludes what it
    rejected cannot show that nothing was skipped.
    """
    store_root = Path(store_root)
    as_of_ns = int(as_of_ns)

    # Read once, used twice. The bars read is the expensive one - minutes
    # against the live store - and the roll-call and the liquidity refusal both
    # need it. Reading it separately for each cost double for no answer that
    # differed, and worse, the two reads would be at slightly different clocks.
    lookback_ns = int(lookback_ns)
    try:
        bars = ClockGatedReader(store_root, BARS_DATASET,
                                custodian=custodian).read_as_of(
                                    as_of_ns, not_before_ns=as_of_ns - lookback_ns)
    except FileNotFoundError:
        bars = pd.DataFrame()

    # The roll-call still sees every perpetual: a key is perpetual because
    # FUNDING carries it, and that read is unbounded. Bounding the bars only
    # changes what is known about each key's recent behaviour, which is the only
    # thing admission turns on.
    coverage = compute_universe_coverage(store_root, as_of_ns,
                                         custodian=custodian, bars=bars)
    with_depth = _symbols_with_depth(store_root, as_of_ns, custodian)
    traded = _traded(bars)

    built: dict[str, list] = {column: [] for column in _COLUMNS}
    excluded_by_reason: dict[str, int] = {}

    for row in coverage.rows.to_dict("records") if not coverage.rows.empty else []:
        key = (str(row["venue"]), str(row["symbol"]))
        volume, trade_count = traded.get(key, (0.0, 0))
        reason = _refusal(pd.Series(row), volume, trade_count, key in traded)
        admitted = reason == ""
        if not admitted:
            excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1

        built["venue"].append(key[0])
        built["symbol"].append(key[1])
        built["observations"].append(int(row["observations"]))
        built["freshness"].append(str(row["freshness"]))
        built["age_ns"].append(row["age_ns"])
        built["routine_gap_ns"].append(row["routine_gap_ns"])
        built["age_beyond_build_ns"].append(row["age_beyond_build_ns"])
        built["traded_volume"].append(volume)
        built["trade_count"].append(trade_count)
        built["admitted"].append(admitted)
        # A refused symbol carries no tier. Giving it one would put a symbol we
        # will not trade into the same column a feature builder reads to decide
        # what it may compute.
        built["tier"].append("" if not admitted
                             else DEEP if key in with_depth else BARS_ONLY)
        built["reason"].append(reason)

    return PerpUniverse(rows=pd.DataFrame(built, columns=list(_COLUMNS)),
                        as_of_ns=as_of_ns,
                        excluded_by_reason=excluded_by_reason,
                        lookback_ns=lookback_ns)


def record_perp_universe(store_root=Path.home() / "capture" / "store",
                         out_path: Path = REPORT_PATH,
                         as_of_ns: int | None = None,
                         custodian=None) -> Path:
    """Run the admission and write the result where a probe can read it.

    `record_` and not `get_` because it writes, and Rule 7 says a name that hides
    a write is a bug waiting. The selection itself reads the whole store, which
    is minutes of IO - far too expensive to run inside a board pass that already
    takes ten to nineteen minutes. So the expensive thing runs on its own and
    publishes; the probe reads what it published, and says NOT MEASURED when
    there is nothing there rather than running it inline.

    Every excluded symbol is written out with its reason, not just the counts.
    A total with no breakdown is the number that gets quoted and never acted on,
    and the acceptance sentence says published rather than hidden.
    """
    as_of_ns = int(time.time_ns() if as_of_ns is None else as_of_ns)
    universe = select_tradable_perp_universe(store_root, as_of_ns,
                                             custodian=custodian)
    rows = universe.rows
    payload = {
        "as_of_ns": universe.as_of_ns,
        "written_at_ns": time.time_ns(),
        "considered": universe.considered,
        "admitted": universe.admitted_count,
        "deep": universe.deep_count,
        "bars_only": universe.admitted_count - universe.deep_count,
        "excluded": universe.excluded_count,
        "excluded_by_reason": universe.excluded_by_reason,
        "summary": universe.describe(),
        "admitted_symbols": [
            {"venue": venue, "symbol": symbol,
             "tier": universe.tier_of(venue, symbol)}
            for venue, symbol in universe.admitted_symbols],
        "excluded_symbols": [
            {"venue": str(row["venue"]), "symbol": str(row["symbol"]),
             "reason": str(row["reason"]),
             "observations": int(row["observations"])}
            for _, row in (rows[~rows["admitted"]].iterrows()
                           if universe.considered else [])],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written whole then moved into place: a probe reading a half-written report
    # would grade the bot on a truncated universe.
    temporary = out_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Record the perp bot's tradable universe. PB-01."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="perp.tradable_universe",
        description="Decide which perpetuals the perp bot may trade, and why "
                    "every other symbol it can see is refused.")
    parser.add_argument("--store-root",
                        default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--as-of-ns", type=int, default=None,
                        help="the clock to read as of; defaults to now")
    args = parser.parse_args(argv)

    written = record_perp_universe(Path(args.store_root), Path(args.out),
                                   args.as_of_ns)
    report = json.loads(written.read_text(encoding="utf-8"))
    print(report["summary"], file=sys.stderr)
    print(f"written to {written}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# PB-16 / RL-024: live admission, alongside the archive admission above.
#
# Everything above this line decides admission from CAPTURED HISTORY - which
# perpetuals have enough archive to model, measured against each series' own
# cadence. That question is unchanged and is what the research and training path
# needs.
#
# This decides a different one: **which perpetuals are tradable right now**, from
# the live feed and nothing else. A symbol with a deep archive that stopped quoting
# a minute ago passes the first test and must fail this one, and the reverse is
# true of a newly listed contract. Merging them would give one answer to two
# questions and quietly get one of them wrong.
#
# It is written here rather than borrowed from `spot.tradable_universe` because
# BF-07's acceptance refuses a segment inheriting another's decisions by default.
# The perp thresholds are its own: perps quote continuously and a perp that has
# gone 15 seconds without a two-sided quote is a problem, where the same gap on a
# dated contract is normal.
# ---------------------------------------------------------------------------

LIVE_NEVER_QUOTED = "NEVER_QUOTED"
LIVE_NO_TWO_SIDED_QUOTE = "NO_TWO_SIDED_QUOTE"
LIVE_QUOTE_STALE = "QUOTE_STALE"
LIVE_TAPE_TOO_THIN = "TAPE_TOO_THIN"

# Tighter than spot's 30s: a perpetual on a major venue quotes many times a second,
# and a 15-second gap means the stream for that symbol has stopped even though the
# connection is up.
LIVE_MAX_QUOTE_AGE_NS = 15_000_000_000
# Perps are the densest tape of the four segments, and the scalping edge RL-022
# named needs prints to read aggressor flow from at all.
LIVE_MIN_TRADES = 2


@dataclass(frozen=True)
class LiveAdmission:
    """One perpetual's live trading decision, with what produced it."""

    venue: str
    symbol: str
    admitted: bool
    reason: str
    evidence: dict


def admit_live(frames: dict, *, max_quote_age_ns: int = LIVE_MAX_QUOTE_AGE_NS,
               min_trades: int = LIVE_MIN_TRADES) -> tuple:
    """Decide every perpetual the live feed has seen. Excluded rows are retained."""
    decisions = []
    for (venue, symbol), frame in frames.items():
        if hasattr(frame, "is_refusal"):
            missing = getattr(frame, "missing", ())
            reason = (LIVE_NO_TWO_SIDED_QUOTE if "two_sided_quote" in missing
                      else LIVE_NEVER_QUOTED if "no_ticks_seen" in missing
                      else LIVE_TAPE_TOO_THIN)
            decisions.append(LiveAdmission(venue, symbol, False, reason,
                                           {"missing": list(missing)}))
            continue
        quote_age = frame.get("quote_age_ns")
        trades = frame.get("trade_count") or 0
        evidence = {"quote_age_ns": quote_age, "trade_count": trades,
                    "samples": frame.get("samples"),
                    "funding_rate": frame.get("venue_funding_rate")}
        if quote_age is None or quote_age > max_quote_age_ns:
            decisions.append(LiveAdmission(venue, symbol, False,
                                           LIVE_QUOTE_STALE, evidence))
            continue
        if trades < min_trades:
            decisions.append(LiveAdmission(venue, symbol, False,
                                           LIVE_TAPE_TOO_THIN, evidence))
            continue
        decisions.append(LiveAdmission(venue, symbol, True, "ADMITTED", evidence))
    return tuple(decisions)


def describe_live(decisions) -> dict:
    """Counts with their denominator, following `describe()` above."""
    considered = len(decisions)
    admitted = [d for d in decisions if d.admitted]
    reasons: dict = {}
    for decision in decisions:
        if not decision.admitted:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    return {"segment": "perp", "considered": considered, "admitted": len(admitted),
            "excluded": considered - len(admitted), "excluded_by_reason": reasons,
            "admitted_symbols": [d.symbol for d in admitted]}
