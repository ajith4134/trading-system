"""One carry candidate, end to end, through every gate that can refuse it.

This is the first caller `validation/` has ever had. Trial Registry, Holdout
Custodian, purged CPCV, deflated Sharpe, MinBTL, PBO and SPA were all written
complete and reached by nothing - `promotion_gate` had zero callers, and every
other module was reached only through it.

**A refusal is this phase succeeding.** The spec's definition of done says the
gate must return a verdict, not that the verdict must be yes. The machine
working is the deliverable; a promoted strategy is a possible outcome.

## What it must read, and what it must not

It reads the **observed** `funding` dataset through the clock gate, with a
`HoldoutCustodian` attached. It must never read `funding_reconstructed`: that
dataset's availability time is the fetch, so a backtest sees nothing in it - by
construction, and the breadth gate is what it exists for.

That is also why this will refuse today. The observed record began on
2026-08-08; the machinery is being proven against roughly a day of history. The
gates that fire will be MinBTL and the deflated Sharpe, and they will be right.

## The candidate

A carry rule of exactly the shape §5a.5 permits: one expression evaluated
identically on every symbol, with per-symbol variation coming only from
normalisation against that symbol's own history. Short the perp when its daily
carry sits in the top percentile of its own trailing window; the return is the
funding collected.

No per-symbol parameters. §5a.5 calls fitting them *"the single most dangerous
thing that could be implemented here"*, because one setup on 850 symbols is one
hypothesis with 850 samples and becomes 850 hypotheses the moment a symbol gets
its own knob.

## The holdout is sealed against the reader, not against the author

`ClockGatedReader` has accepted a custodian since it was written and no caller
has ever passed one - so the guard existed and guarded nothing. It is passed
here. A read whose simulated clock lands inside the sealed range raises rather
than returning rows, which is the difference between a rule and a mechanism.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from store.temporal_schema import EVENT_TIME, SYMBOL
from validation.backtest_overfitting import probability_of_backtest_overfitting
from validation.promotion_gate import GateResult, evaluate_for_promotion
from validation.superior_predictive_ability import superior_predictive_ability
from validation.trial_registry import TrialRegistry, TrialSpec

# The observed dataset. Named as a constant so the one thing this must not read
# is impossible to reach by editing a string in a call site.
OBSERVED_FUNDING = "funding"

# Carry pays on a schedule rather than on a random walk, so the natural period is
# the day. Matches the breadth gate, which is deliberate: the same setup measured
# two ways should be measured on the same clock.
PERIODS_PER_YEAR = 365

# Above this share of paths beating the median, the selection procedure is
# picking overfit configurations more often than not. Bailey et al put the
# interesting region around 0.5; anything at or above it means the search is
# choosing worse-than-median performers out of sample.
MAX_PBO = 0.50

# Hansen's SPA, at the conventional level. The incumbent is standing aside, so
# the null is "no configuration beats doing nothing" - a carry rule has to clear
# that before it is worth comparing with anything cleverer.
SPA_ALPHA = 0.05

# Below this many days, an effective-breadth estimate is noise and the gate falls
# back to counting calendar days. A correlation matrix over a handful of rows
# reports whatever it likes, and this multiplier only ever makes promotion
# easier - so the short-window case fails closed to the conservative basis.
MIN_DAYS_FOR_BREADTH = 30


def tail_cap_gate(series: pd.Series, cap) -> dict:
    """Would this candidate have lived inside the user's ceiling?

    The promotion input promised when the ceiling was built. Paper is
    unconstrained by ruling - fake capital does not need protecting - but a
    candidate tuned with no ceiling was optimised against a different objective
    from the one it must satisfy with money behind it. Promoting it hands capital
    to a strategy never once tested against its own constraint.

    So the run is replayed against the ceiling as an equity curve: worst single
    day, and worst peak-to-trough. Nothing was stopped when it happened; this is
    the record of whether it would have been.

    A missing ceiling FAILS rather than being skipped. §6 reserves the number to
    the user and says nothing trades real money without one - a gate that
    quietly disappeared when unset would be the absence of evidence rendering as
    green, which is the failure Rule 8 exists for.
    """
    if cap is None:
        return {"name": "tail_cap", "passed": False, "measured": float("nan"),
                "threshold": float("nan"),
                "detail": "no tail-loss ceiling set; §6 reserves it to the user "
                          "and nothing trades real money without one"}

    equity = (1.0 + series).cumprod()
    worst_day = float((-series).max())
    worst_drawdown = float((1.0 - equity / equity.cummax()).max())
    daily_limit = float(cap.daily_loss_fraction)
    drawdown_limit = float(cap.drawdown_fraction)
    inside = worst_day < daily_limit and worst_drawdown < drawdown_limit

    return {"name": "tail_cap", "passed": inside,
            "measured": max(worst_day / daily_limit,
                            worst_drawdown / drawdown_limit),
            "threshold": 1.0,
            "detail": (f"worst day {worst_day * 100:.3f}% against a "
                       f"{daily_limit * 100:.2f}% ceiling, worst peak-to-trough "
                       f"{worst_drawdown * 100:.3f}% against "
                       f"{drawdown_limit * 100:.2f}%"
                       + ("" if inside else " - would have breached"))}


@dataclass(frozen=True)
class CarryCandidate:
    """One configuration of the carry rule. Every field is a global parameter."""

    lookback_days: int = 30
    percentile: float = 0.90

    @property
    def name(self) -> str:
        return f"carry: top {self.percentile:.0%} of own {self.lookback_days}d funding"

    def params(self) -> dict:
        return {"lookback_days": self.lookback_days, "percentile": self.percentile}


@dataclass
class PipelineResult:
    """What the pipeline decided, and everything it decided it from."""

    candidate: dict
    promoted: bool
    rows: int
    symbols: int
    days: int
    n_trials: int
    n_cpcv_paths: int
    observed_sharpe: float
    deflated_sharpe: float
    gates: list[dict] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def daily_carry(frame: pd.DataFrame) -> pd.DataFrame:
    """Days as rows, symbols as columns, that day's total funding as the value.

    Summed to the day for the reason the breadth gate found the hard way: this
    universe does not share a settlement schedule - 4-hourly, 8-hourly and hourly
    symbols side by side, with millisecond jitter on the timestamps - so a grid
    keyed on the raw instant gives every symbol a column of time to itself.
    """
    if frame.empty:
        return pd.DataFrame()
    rates = frame.assign(
        rate=pd.to_numeric(frame["funding_rate"], errors="coerce"),
        day=pd.to_datetime(frame[EVENT_TIME], unit="ns", utc=True).dt.floor("D"))
    return rates.groupby(["day", SYMBOL])["rate"].sum().unstack().sort_index()


def candidate_returns(matrix: pd.DataFrame, candidate: CarryCandidate) -> pd.Series:
    """The portfolio's daily return: the mean carry across whatever triggered.

    Equally weighted across the symbols firing that day, and zero on a day
    nothing fires. Standing aside is a real outcome with a real return, and
    dropping those days would score the strategy only on the days it chose to
    act - which is how a rule that trades twice a year reports a wonderful
    Sharpe.
    """
    if matrix.empty:
        return pd.Series(dtype=float)
    # The signal is YESTERDAY's carry against a threshold built from the days
    # before it. Both shifted, and the second shift is the one that matters.
    #
    # Without it this compared TODAY's funding to a past threshold and then
    # collected today's funding - deciding to have been positioned for a rate
    # that had not settled yet. Measured before the fix, on 400 days of PURE
    # NOISE across 30 symbols: observed Sharpe 3.90, and the deflated Sharpe,
    # MinBTL and purge-retention gates all PASSED it. Only PBO refused.
    #
    # The mechanism is worth naming because it looks like an edge: selecting the
    # top decile of a symmetric distribution and averaging it returns a positive
    # number every single day, with little variance across days. That is a
    # selection artefact of the return definition, not a strategy, and it failed
    # in the flattering direction exactly as this project's other three did.
    signal = matrix.shift(1)
    threshold = (matrix.shift(2)
                 .rolling(candidate.lookback_days, min_periods=candidate.lookback_days)
                 .quantile(candidate.percentile))
    fired = ((signal > threshold) & threshold.notna() & signal.notna()).fillna(False)
    # Fired on yesterday's information, paid at today's rate.
    collected = matrix.where(fired)
    return collected.mean(axis=1).fillna(0.0).astype(float)


def fold_returns(series: pd.Series, fold) -> list[float]:
    """The out-of-sample returns for one CPCV fold: its test blocks only.

    The fold's train blocks are already purged and embargoed on both sides by
    `combinatorial_purged_folds`. Nothing here touches them - a backtest that
    fitted on the train range would still be honest only if it never saw the test
    range, and this candidate fits nothing, so the test blocks are the whole
    answer.
    """
    values = series.to_numpy()
    out: list[float] = []
    for start, end in fold.test_blocks:
        out.extend(float(v) for v in values[start:end])
    return out


def effective_sample_multiplier(matrix: pd.DataFrame,
                                candidate: CarryCandidate) -> float:
    """How many independent bets a day of this setup is worth. 1.0 when unknown.

    §5a.5 corrects on two axes, and this is the second: *trial count AND
    effective sample size*. A day in which the setup fired across a wide
    cross-section carries more evidence than a single portfolio return implies -
    but not 850 times more, because crypto cross-sectional correlation is severe
    and correlated streams are not independent samples.

    So the multiplier is the participation ratio of the per-symbol return
    streams' correlation matrix - the same measure the §5a.4 breadth gate uses -
    computed on the data being scored rather than inherited from that gate's run
    over reconstructed history.

    **Fails closed.** Below `MIN_DAYS_FOR_BREADTH` the estimate is noise and this
    returns 1.0, the calendar-days basis. A correlation matrix over a handful of
    rows reports whatever it likes, and this number only ever makes promotion
    easier - so the uncertain case takes the conservative side.
    """
    from validation.breadth import carry_returns, effective_breadth, percentile_triggers

    if len(matrix) < MIN_DAYS_FOR_BREADTH:
        return 1.0
    triggers = percentile_triggers(matrix, candidate.lookback_days, candidate.percentile)
    breadth, _ = effective_breadth(carry_returns(matrix, triggers))
    if not np.isfinite(breadth) or breadth < 1.0:
        return 1.0
    # Never more independent bets than symbols, whatever the arithmetic says.
    return float(min(breadth, matrix.shape[1]))


def run(frame: pd.DataFrame, registry: TrialRegistry,
        candidates: list[CarryCandidate] | None = None,
        min_deflated_sharpe: float = 0.95, cap=None) -> PipelineResult:
    """Take the best candidate through every gate, and report what refused it.

    Candidates are scored first, then the best is put through the gate - and the
    N that deflates it is the registry's cumulative count, so every configuration
    looked at raises the bar for the one that wins. That is the whole point of
    the registry and it is why the search happens inside it rather than beside
    it.
    """
    candidates = candidates or [CarryCandidate()]
    matrix = daily_carry(frame)
    if matrix.empty:
        return PipelineResult(candidate={}, promoted=False, rows=0, symbols=0,
                              days=0, n_trials=registry.cumulative_count(),
                              n_cpcv_paths=0, observed_sharpe=float("nan"),
                              deflated_sharpe=float("nan"),
                              refusals=["no funding rows to score"])

    # Every configuration's full return series, kept for PBO: CSCV needs the
    # WHOLE search's results, not the finalists', or it measures the wrong
    # selection procedure.
    series_by_candidate = {c.name: candidate_returns(matrix, c) for c in candidates}
    scored = {name: s for name, s in series_by_candidate.items() if s.std() > 0}
    if not scored:
        return PipelineResult(
            candidate={}, promoted=False, rows=int(matrix.notna().sum().sum()),
            symbols=int(matrix.shape[1]), days=int(matrix.shape[0]),
            n_trials=registry.cumulative_count(), n_cpcv_paths=0,
            observed_sharpe=float("nan"), deflated_sharpe=float("nan"),
            refusals=["no candidate produced a return series with any variance; "
                      "there is not enough history for the trailing window to "
                      "have opened"])

    best_name = max(scored, key=lambda n: scored[n].mean() / (scored[n].std() or 1))
    best = next(c for c in candidates if c.name == best_name)
    series = scored[best_name]

    # §5a.5's second axis: effective sample size, measured on THIS data rather
    # than carried over from the breadth gate's run on reconstructed history.
    # One observation per day understates a setup firing across a cross-section;
    # 850 correlated symbols overstate it just as badly. Effective breadth is the
    # number between them, and it is measured here at gate time.
    multiplier = effective_sample_multiplier(matrix, best)

    spec = TrialSpec(name=best.name, family="carry", params=best.params())
    verdict = evaluate_for_promotion(
        registry, spec,
        backtest_fold=lambda fold: fold_returns(series, fold),
        n_rows=len(series),
        min_deflated_sharpe=min_deflated_sharpe,
        periods_per_year=PERIODS_PER_YEAR,
        effective_sample_multiplier=multiplier,
    )

    gates = [asdict(g) for g in verdict.gates]
    refusals = list(verdict.rejection_reasons())

    # PBO over the whole search, and SPA against standing aside. Both are added
    # here rather than inside `evaluate_for_promotion` because both need the
    # OTHER candidates, which that function deliberately never sees - it scores
    # one candidate and knows nothing about the search around it.
    aligned = [list(s.to_numpy()) for s in scored.values()]
    if len(aligned) >= 2:
        pbo = probability_of_backtest_overfitting(aligned)
        gates.append({"name": "pbo", "passed": pbo < MAX_PBO, "measured": pbo,
                      "threshold": MAX_PBO,
                      "detail": (f"CSCV probability {pbo:.3f} that this selection "
                                 f"picks a below-median performer out of sample, "
                                 f"across {len(aligned)} configurations")})
        if pbo >= MAX_PBO:
            refusals.append(f"pbo {pbo:.3f} >= {MAX_PBO}")

        # Standing aside is the incumbent. A carry rule has to beat doing
        # nothing before it is compared with anything cleverer.
        standing_aside = [0.0] * len(series)
        spa = superior_predictive_ability(
            standing_aside, {n: list(s.to_numpy()) for n, s in scored.items()},
            alpha=SPA_ALPHA)
        # Hansen's consistent p-value, not White's reality check, and not the
        # `superior` list on its own. `superior` is the per-name outcome; the
        # gate is whether the FAMILY beat the null at all, which is the p-value.
        beat = spa.p_value < SPA_ALPHA
        gates.append({"name": "spa", "passed": beat, "measured": spa.p_value,
                      "threshold": SPA_ALPHA,
                      "detail": (f"SPA p={spa.p_value:.4f} (reality-check p="
                                 f"{spa.p_value_reality_check:.4f}) against standing "
                                 f"aside, over {spa.n_challengers} configuration(s); "
                                 f"superior: {list(spa.superior) or 'none'}")})
        if not beat:
            refusals.append(
                f"spa p={spa.p_value:.4f} did not reject the null that nothing "
                f"beats standing aside")

    tail = tail_cap_gate(series, cap)
    gates.append(tail)
    if not tail["passed"]:
        refusals.append(f"tail_cap: {tail['detail']}")

    return PipelineResult(
        candidate=best.params() | {"name": best.name},
        promoted=verdict.promoted and not refusals,
        rows=int(matrix.notna().sum().sum()),
        symbols=int(matrix.shape[1]),
        days=int(matrix.shape[0]),
        n_trials=verdict.n_trials,
        n_cpcv_paths=verdict.n_cpcv_paths,
        observed_sharpe=verdict.observed_sharpe,
        deflated_sharpe=verdict.deflated_sharpe,
        gates=gates,
        refusals=refusals,
    )


def sealed_reader(store_root: Path, custodian_root: Path,
                  holdout_start_ns: int, holdout_end_ns: int,
                  dataset: str = OBSERVED_FUNDING):
    """A reader with the holdout actually attached.

    `ClockGatedReader` has taken a custodian since it was written and no caller
    has ever passed one, so the guard existed and guarded nothing. Passing it is
    the difference between a rule and a mechanism: a read whose clock lands
    inside the sealed range raises, rather than politely returning the rows the
    selection was supposed to never see.

    The reader is built here rather than by the caller so there is one place
    where the holdout is either attached or visibly absent.
    """
    from store.clock_gated_reader import ClockGatedReader
    from validation.holdout_custodian import HoldoutCustodian

    custodian = HoldoutCustodian(custodian_root, holdout_start_ns, holdout_end_ns)
    return ClockGatedReader(store_root, dataset, custodian=custodian), custodian


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and write the verdict down.

    Exit 0 whether or not the candidate is promoted: a refusal is this working.
    Nonzero is reserved for the pipeline itself failing to reach a verdict.
    """
    import argparse
    import datetime as dt
    import sys

    parser = argparse.ArgumentParser(
        prog="promotion-pipeline",
        description="Take one carry candidate through every gate that can refuse it.")
    parser.add_argument("--store-root", default=str(Path.home() / "capture" / "store"))
    parser.add_argument("--trials-root", default=str(Path.home() / "capture" / "trials"))
    parser.add_argument("--holdout-days", type=int, default=30,
                        help="seal the most recent N days from selection")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    store_root = Path(args.store_root)
    now_ns = int(dt.datetime.now(dt.timezone.utc).timestamp()) * 1_000_000_000
    holdout_start = now_ns - args.holdout_days * 86_400 * 1_000_000_000

    reader, custodian = sealed_reader(
        store_root, Path(args.trials_root), holdout_start, now_ns)

    # The ceiling is read, never chosen here. A missing one fails the gate rather
    # than being skipped - §6 reserves the number to the user.
    from risk.tail_cap import CeilingNotSet, read_ceiling
    try:
        cap = read_ceiling(store_root.parent / "risk")
    except CeilingNotSet as unset:
        print(f"note: {unset}", file=sys.stderr)
        cap = None

    # Read strictly BEFORE the holdout. Asking for anything later is what the
    # custodian is there to refuse, and it does.
    frame = reader.read_as_of(holdout_start - 1)
    registry = TrialRegistry(Path(args.trials_root))

    grid = [CarryCandidate(lookback_days=lb, percentile=p)
            for lb in (10, 20, 30) for p in (0.80, 0.90, 0.95)]
    result = run(frame, registry, grid, cap=cap)

    out = Path(args.out) if args.out else store_root.parent / "promotion.json"
    out.write_text(json.dumps({
        "measured_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": OBSERVED_FUNDING,
        "holdout": {"start_ns": holdout_start, "end_ns": now_ns,
                    "frozen": custodian.is_frozen(),
                    "consumed": custodian.is_consumed()},
        "cumulative_trials": registry.cumulative_count(),
        **result.as_dict(),
    }, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"selection window: {result.days} days x {result.symbols} symbols "
          f"({result.rows} observations), holdout of {args.holdout_days} days sealed")
    print(f"candidate: {result.candidate.get('name', '-')}")
    print(f"PROMOTED: {result.promoted}")
    for gate in result.gates:
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {gate['name']}: {gate['detail']}")
    for refusal in result.refusals:
        print(f"  refused: {refusal}")
    print(f"\ncumulative trials: {registry.cumulative_count()}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
