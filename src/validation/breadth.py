"""The breadth gate: are these 850 bets, or one bet wearing 850 hats?

Goal spec §5a.4, which gates the whole of Phase 4:

    Breadth means independent bets, not symbols. Crypto cross-sectional
    correlation is severe; 1,290 symbols may be one BTC factor plus noise,
    collapsing effective breadth to ~5-15 and gutting the proposition entirely.

    The test: define candidate setups, fire them historically across the
    universe, and measure (1) clustering of trigger times, (2) correlation of
    the resulting return streams.

    If triggers cluster and returns correlate, the architecture is one macro bet
    wearing 1,290 hats and must be redesigned.

This module is that test. It answers a research question and it is not a
backtest: it reads `funding_reconstructed`, whose availability time is the fetch
rather than the settlement, precisely so no simulated clock can consume it.

## What is measured, and why these two numbers

**Trigger clustering** is the index of dispersion of triggers per settlement.
Under independence the count at each settlement is Binomial(n, p) with variance
`n*p*(1-p)`; the statistic is observed variance over that. **1.0 is what
independence looks like.** Ten means the triggers arrive in bursts ten times
more uneven than chance, which is a regime firing, not 850 opinions.

**Effective breadth** is the participation ratio of the return correlation
matrix's eigenvalues, `(sum L)^2 / sum(L^2)`. It is the number of independent
bets the correlated streams are worth. For perfectly correlated streams it is 1;
for independent ones it is the symbol count. This is the number §5a.4 warns
collapses to 5-15, and it is the one that decides the phase.

Mean pairwise correlation is reported beside it because the two fail
differently: a handful of tightly coupled clusters and a uniform mild
correlation can produce the same effective breadth, and they call for different
redesigns.

## The rule this must not break

§5a.5: *setup definitions are universal and parameter-free across symbols.
Per-symbol variation comes only from normalisation - z-scores or percentiles
against that symbol's own history - never from fitted per-symbol parameters.*

The setup here is one expression evaluated identically everywhere: fire when a
symbol's funding is in the top percentile of ITS OWN trailing history. The
lookback and the percentile are global and shared by every symbol. That makes it
one hypothesis with 850 samples rather than 850 hypotheses, which is the whole
statistical argument for scanning wide.

Both parameters are registered in the Trial Registry before the scan runs -
§5a.5 again: *every scan counts*. A breadth measurement is a look at the data
and the count is what keeps a later deflated Sharpe honest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pathlib import Path

from store.temporal_schema import EVENT_TIME, SYMBOL

# Enough settlements behind a symbol for a percentile of its own history to mean
# anything. At 8-hourly settlement, 90 is thirty days.
MIN_HISTORY = 90
# A symbol must be present for this share of the common grid to be compared with
# the others. Correlation over a handful of overlapping points is noise with a
# number attached.
MIN_COVERAGE = 0.80
# How much recent history one measurement looks at. A STATED choice, not a fitted
# one: the grid is ragged - symbols list and delist - so coverage has to be
# judged against a declared window rather than against the union of everything,
# where almost no symbol would reach 80% and the filter would empty the universe.
# A year is long enough to hold several funding regimes and short enough that
# most of the current universe existed for it.
WINDOW_DAYS = 365


@dataclass(frozen=True)
class BreadthMeasurement:
    """What one setup produced across the universe. Every field is measured."""

    setup: str
    lookback: int
    percentile: float
    window_days: int
    symbols: int
    settlements: int
    triggers: int
    trigger_rate: float
    # (1) clustering of trigger times
    clustering_ratio: float
    busiest_5pct_share: float
    # (2) correlation of the resulting return streams
    mean_pairwise_correlation: float
    effective_breadth: float
    effective_breadth_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def funding_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """DAILY carry per symbol: UTC days as rows, symbols as columns.

    Days rather than settlements, and that is forced by the data rather than
    chosen for convenience. Measured on the reconstructed history 2026-08-09:
    this universe does not share one settlement schedule. The modal gap between
    a symbol's settlements is 4 hours for 255,513 of them, 8 for 115,108, 1 for
    6,018 and 2 for 1,017 - and the timestamps carry millisecond jitter
    (`...200007`), so even same-schedule symbols do not land on identical
    instants.

    Pivoting on the raw instant therefore produced a grid of 6,363 columns of
    time where every symbol occupied its own, and the coverage filter correctly
    threw away all 850 of them.

    Summing to the day fixes both. It removes the jitter, and it makes a
    4-hourly symbol comparable with an 8-hourly one: the day's total funding is
    the day's carry either way, which is the quantity a carry bet actually earns.
    """
    if frame.empty:
        return pd.DataFrame()
    rates = frame.assign(
        rate=pd.to_numeric(frame["funding_rate"], errors="coerce"),
        day=pd.to_datetime(frame[EVENT_TIME], unit="ns", utc=True).dt.floor("D"))
    return rates.groupby(["day", SYMBOL])["rate"].sum().unstack().sort_index()


def recent_window(matrix: pd.DataFrame, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """The last `window_days` rows of the grid.

    Taken before the coverage filter, because coverage only means anything
    against a declared period. Measured on the reconstructed history: the full
    grid spans 1,083 days but the median symbol covers 334 of them, so judging
    coverage against the union would drop every symbol and report an empty
    universe as a finding.
    """
    if matrix.empty:
        return matrix
    return matrix.tail(window_days)


def well_covered(matrix: pd.DataFrame, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Drop symbols too sparse to correlate against the rest.

    Dropped rather than filled. A forward-filled funding rate for a symbol that
    had not listed is a fabricated observation, and it would correlate perfectly
    with whatever was used to fill it.
    """
    if matrix.empty:
        return matrix
    coverage = matrix.notna().mean()
    return matrix.loc[:, coverage >= min_coverage]


def percentile_triggers(matrix: pd.DataFrame, lookback: int = MIN_HISTORY,
                        percentile: float = 0.90) -> pd.DataFrame:
    """Fire where a symbol's funding tops the given percentile of its own history.

    Trailing and strictly backward-looking: the window ends at the settlement
    BEFORE the one being judged, so a rate never helps decide about itself. That
    is not a backtest safeguard here - nothing trades on this - it is so the
    trigger times being measured are the ones a live rule would actually produce.
    """
    if matrix.empty:
        return matrix
    threshold = (matrix.shift(1)
                 .rolling(lookback, min_periods=lookback)
                 .quantile(percentile))
    # A day with no rate is NOT a trigger, said explicitly rather than left as
    # NA to propagate. A symbol that had not listed yet must count as standing
    # aside, not as an unknown that poisons the arithmetic downstream - and a
    # boolean matrix carrying NA is one that silently becomes an object array.
    fired = (matrix > threshold) & threshold.notna() & matrix.notna()
    return fired.fillna(False).astype(bool)


def carry_returns(matrix: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    """The return stream each symbol's setup produced, zero where it stood aside.

    A short perp against a spot hedge collects the funding it is charged, so the
    per-settlement return of the hedged trade is the rate itself. Costs are not
    subtracted: this measures whether the bets are INDEPENDENT, and a cost
    applied identically to every symbol moves the level of every stream without
    changing how they co-move.

    Standing aside is a zero rather than a gap, and it matters. If triggers
    cluster then the zeros align too, and the correlation this measures picks
    that up - which is exactly the effect §5a.4 is asking about.
    """
    return matrix.where(triggers, 0.0).fillna(0.0)


def clustering_ratio(triggers: pd.DataFrame) -> tuple[float, float]:
    """`(index of dispersion, share of triggers in the busiest 5% of settlements)`.

    Under independence the count per settlement is Binomial(n, p) and its
    variance is `n*p*(1-p)`, so the ratio of observed variance to that is 1.0
    when symbols fire independently and rises as they fire together.

    The second number is there because the first is a summary: a ratio of 3 could
    be a mild everywhere-effect or a few enormous days, and only one of those is
    a regime.
    """
    if triggers.empty or triggers.shape[1] == 0:
        return float("nan"), float("nan")
    per_settlement = triggers.sum(axis=1).astype(float)
    n = float(triggers.shape[1])
    p = float(triggers.to_numpy().mean())
    expected_variance = n * p * (1.0 - p)
    ratio = (float(per_settlement.var(ddof=0)) / expected_variance
             if expected_variance > 0 else float("nan"))

    total = float(per_settlement.sum())
    if total <= 0:
        return ratio, float("nan")
    busiest = max(1, int(round(0.05 * len(per_settlement))))
    share = float(per_settlement.nlargest(busiest).sum() / total)
    return ratio, share


def effective_breadth(returns: pd.DataFrame) -> tuple[float, float]:
    """`(effective independent bets, mean pairwise correlation)`.

    The participation ratio of the correlation matrix's eigenvalues,
    `(sum L)^2 / sum(L^2)`. One dominant eigenvalue - every stream loading on the
    same factor - drives it to 1 however many columns there are. Independent
    streams give back the column count.

    Columns with no variance are dropped first: a symbol that never triggered has
    a constant zero stream, its correlation with everything is undefined, and
    leaving it in would inflate the count with a bet nobody placed.
    """
    if returns.empty or returns.shape[1] < 2:
        return float("nan"), float("nan")
    active = returns.loc[:, returns.std(ddof=0) > 0]
    if active.shape[1] < 2:
        return float("nan"), float("nan")

    correlation = active.corr().to_numpy()
    correlation = np.nan_to_num(correlation, nan=0.0)
    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        return float("nan"), float("nan")
    breadth = float(total ** 2 / float((eigenvalues ** 2).sum()))

    upper = correlation[np.triu_indices_from(correlation, k=1)]
    return breadth, float(np.mean(upper)) if upper.size else float("nan")


def measure(frame: pd.DataFrame, lookback: int = MIN_HISTORY,
            percentile: float = 0.90, window_days: int = WINDOW_DAYS,
            setup: str = "funding above its own trailing percentile") -> BreadthMeasurement:
    """Fire one setup across the universe and measure what came out."""
    matrix = well_covered(recent_window(funding_matrix(frame), window_days))
    triggers = percentile_triggers(matrix, lookback, percentile)
    returns = carry_returns(matrix, triggers)

    ratio, busiest = clustering_ratio(triggers)
    breadth, mean_correlation = effective_breadth(returns)
    symbols = int(matrix.shape[1])
    fired = int(triggers.to_numpy().sum()) if not triggers.empty else 0
    cells = int(triggers.notna().to_numpy().sum()) if not triggers.empty else 0

    return BreadthMeasurement(
        setup=setup,
        lookback=lookback,
        percentile=percentile,
        window_days=window_days,
        symbols=symbols,
        settlements=int(matrix.shape[0]),
        triggers=fired,
        trigger_rate=(fired / cells) if cells else float("nan"),
        clustering_ratio=ratio,
        busiest_5pct_share=busiest,
        mean_pairwise_correlation=mean_correlation,
        effective_breadth=breadth,
        effective_breadth_share=(breadth / symbols) if symbols else float("nan"),
    )


# The band §5a.4 names as the failure it fears: "collapsing effective breadth to
# ~5-15 and gutting the proposition entirely". Taken from the corpus rather than
# chosen here, so the gate is not grading itself against a number invented to
# pass it.
COLLAPSE_BAND = (5.0, 15.0)
# Independence gives an index of dispersion of 1.0. Two is a judgement: it allows
# the mild co-movement any shared market has, and refuses a regime. Stated in the
# report so a reader can disagree with the number rather than with the verdict.
CLUSTERING_LIMIT = 2.0

PASS = "PASS"
PASS_BURSTY = "PASS - BURSTY FLOW"
REDESIGN = "REDESIGN"
INCONCLUSIVE = "INCONCLUSIVE"


def verdict(measurement: BreadthMeasurement) -> tuple[str, str]:
    """`(verdict, why)`. The gate's answer, with the criteria it applied.

    The condition for REDESIGN is collapsed BREADTH, and clustering alone is not
    it. §5a.4 says *"if triggers cluster **and** returns correlate"* - a
    conjunction, and the first draft of this function read it as a disjunction
    and would have failed the gate on clustering by itself.

    That distinction is the whole finding, because the two say different things:

    - **Correlated returns collapse breadth**, and §5a.4 says that "guts the
      proposition entirely". There is no version of the strategy that survives
      it, so it is a redesign.
    - **Clustered triggers are a capital-timing problem**, and §5a.5 already
      prescribes the answer rather than treating it as fatal: *"the acceptance
      threshold RISES with opportunity flow... the binding constraint is having
      capital free when the good ones fire"*, and *"dry powder is a held option
      with computable value, not idle cash"*.

    So bursty flow gets its own verdict instead of being folded into either. A
    PASS that quietly contained a 28x clustering ratio would be the flattering
    kind of green this project keeps finding.
    """
    breadth = measurement.effective_breadth
    ratio = measurement.clustering_ratio
    if not np.isfinite(breadth) or not np.isfinite(ratio):
        return INCONCLUSIVE, ("not enough data to measure: "
                              f"{measurement.symbols} symbols, "
                              f"{measurement.settlements} days, "
                              f"{measurement.triggers} triggers")

    breadth_note = (
        f"effective breadth {breadth:.1f} of {measurement.symbols} symbols "
        f"({measurement.effective_breadth_share:.0%}), mean pairwise correlation "
        f"{measurement.mean_pairwise_correlation:.3f}")
    cluster_note = (
        f"triggers arrive {ratio:.0f}x more unevenly than independent firing, "
        f"{measurement.busiest_5pct_share:.0%} of them in the busiest 5% of days")

    if breadth <= COLLAPSE_BAND[1]:
        return REDESIGN, (
            f"{breadth_note} - inside the {COLLAPSE_BAND[0]:.0f}-"
            f"{COLLAPSE_BAND[1]:.0f} band §5a.4 calls a collapse that guts the "
            f"proposition. {cluster_note}")

    if ratio > CLUSTERING_LIMIT:
        return PASS_BURSTY, (
            f"{breadth_note} - clear of the {COLLAPSE_BAND[1]:.0f} collapse "
            f"ceiling, so the bets are independent enough. But {cluster_note}, "
            f"which is the §5a.5 case: the threshold must rise with opportunity "
            f"flow and dry powder priced as a held option. Not a redesign, and "
            f"not a clean pass either")

    return PASS, (f"{breadth_note}; triggers disperse at {ratio:.1f}x against a "
                  f"{CLUSTERING_LIMIT:.1f}x limit")


def run_gate(frame: pd.DataFrame, registry, lookback: int = MIN_HISTORY,
             percentile: float = 0.90, window_days: int = WINDOW_DAYS) -> dict:
    """Measure one setup, counted in the Trial Registry before it is measured.

    §5a.5: *every scan counts in the Trial Registry*. A breadth measurement is a
    look at the data, and the count is what keeps a later deflated Sharpe honest
    about how many things were tried before the winner.
    """
    from validation.trial_registry import TrialSpec

    spec = TrialSpec(
        name=f"breadth: funding percentile {percentile:.2f} over {lookback}d, "
             f"{window_days}d window",
        family="carry",
        params={"gate": "breadth §5a.4", "lookback": lookback,
                "percentile": percentile, "window_days": window_days},
    )

    def evaluate(_spec):
        measurement = measure(frame, lookback, percentile, window_days)
        answer, why = verdict(measurement)
        return {**measurement.as_dict(), "verdict": answer, "why": why}

    return registry.evaluate(spec, evaluate)


# The configurations one run measures. Several rather than one, because a single
# (lookback, percentile, window) that happened to pass would be a result chosen
# rather than found - and each is registered as its own trial, so the count of
# what was tried is on the record for whatever reads it later.
SCANS = (
    {"window_days": 365, "lookback": 90, "percentile": 0.90},
    {"window_days": 365, "lookback": 90, "percentile": 0.95},
    {"window_days": 300, "lookback": 60, "percentile": 0.90},
    {"window_days": 300, "lookback": 60, "percentile": 0.95},
    {"window_days": 200, "lookback": 60, "percentile": 0.90},
    {"window_days": 200, "lookback": 60, "percentile": 0.95},
)


def main(argv: list[str] | None = None) -> int:
    """Run the gate and write its answer down.

    §5a.4 calls this cheap and says it gates everything downstream, so the
    result belongs on disk rather than in a terminal that gets closed. Written
    as generated JSON beside a printed report - the file is the record, the
    report is for reading.
    """
    import argparse
    import datetime as dt
    import json
    import sys

    from store.clock_gated_reader import ClockGatedReader
    from store.funding_backfill import DATASET
    from validation.trial_registry import TrialRegistry

    parser = argparse.ArgumentParser(
        prog="breadth-gate",
        description="§5a.4: measure whether a setup produces independent bets.")
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--trials-root", default=str(Path.home() / "capture" / "trials"))
    parser.add_argument("--out", default=None,
                        help="where to write the result; defaults to "
                             "<store-root>/../breadth-gate.json")
    args = parser.parse_args(argv)

    store_root = Path(args.store_root)
    # The reconstructed dataset, deliberately. Its availability time is the
    # fetch, so nothing here could be mistaken for a backtest even by accident.
    frame = ClockGatedReader(store_root, DATASET).read_as_of(2**62)
    if frame.empty:
        print(f"no {DATASET} rows in {store_root}; run store.funding_backfill first",
              file=sys.stderr)
        return 1

    registry = TrialRegistry(Path(args.trials_root))
    results = [run_gate(frame, registry, **scan) for scan in SCANS]

    out = Path(args.out) if args.out else store_root.parent / "breadth-gate.json"
    out.write_text(json.dumps({
        "measured_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": DATASET,
        "rows": int(len(frame)),
        "symbols_present": int(frame[SYMBOL].nunique()),
        "cumulative_trials": registry.cumulative_count(),
        "collapse_band": list(COLLAPSE_BAND),
        "clustering_limit": CLUSTERING_LIMIT,
        "scans": results,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(frame)} rows, {frame[SYMBOL].nunique()} symbols, "
          f"{registry.cumulative_count()} cumulative trials")
    print(f"{'window':>7} {'look':>5} {'pct':>5} {'syms':>5} {'cluster':>8} "
          f"{'corr':>8} {'breadth':>8}  verdict")
    for r in results:
        print(f"{r['window_days']:>7} {r['lookback']:>5} {r['percentile']:>5.2f} "
              f"{r['symbols']:>5} {r['clustering_ratio']:>8.1f} "
              f"{r['mean_pairwise_correlation']:>8.4f} {r['effective_breadth']:>8.1f}"
              f"  {r['verdict']}")
    verdicts = {r["verdict"] for r in results}
    print(f"\nwrote {out}")
    print(f"verdicts across {len(results)} scans: {', '.join(sorted(verdicts))}")
    # Nonzero only on a redesign - the one answer that stops Phase 4.
    return 2 if REDESIGN in verdicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
