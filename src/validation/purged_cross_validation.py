"""Combinatorial Purged Cross-Validation, with a purge that reaches both ways.

CPCV (López de Prado, *Advances in Financial Machine Learning*, ch. 7 & 12)
replaces one walk-forward Sharpe with a distribution over C(n_groups, k_test)
recombinations. `deflated_sharpe` and `probability_of_backtest_overfitting` both
need that distribution; one number cannot separate selection bias from skill.

**Why this was not ported from the corpus.** Both donor implementations were read
raw, and the ledger's "reusable as-is" flag does not survive contact with either.

`nse-crypto-bot-final/trading/strategy/cpcv.py` builds the right combinations and
has real purge machinery, but its exclusion zone is `[test_start, test_end +
embargo)`. Its docstring states the correct contract - *"drop training rows whose
label window overlaps a test block"* - and the code implements the narrower "drop
training rows at or after the test start". The gap is exactly one label horizon on
the **left** edge: a training row at `test_start − 5` with a 10-bar label matures
inside the test block, so it was fit on an outcome that block is about to be
scored on. It leaks, and nothing reports it.

`nse-botonly`'s version purges nothing - defensible for a label-free realized
returns series - but retains `embargo_group_count: int = 1`, a config knob wired
to nothing.

**The consequence for design, not just correctness.** A correct left edge requires
a label horizon, so VX-004's per-family horizon configuration is not a refinement
of purging - it is what makes purging possible. A funding-carry label matures at
the next settlement; a trend label matures in weeks. Getting this from a
per-family declaration means a family nobody has declared has to be a refusal: a
defaulted horizon of zero silently disables the purge, and it would do so for the
family least likely to have been thought about.

Every fold also reports what purging cost it. A fold that quietly lost most of its
training data still returns a Sharpe, and that Sharpe means much less.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

# Label horizons in **rows**, per strategy family. `FEATURES.md` §8: these differ
# by orders of magnitude, which is why a single global embargo cannot be correct.
#
# Stated in rows rather than wall-clock because the caller knows its own bar size
# and the purge operates on indices. The values assume 1-minute bars, the capture
# archive's native resolution.
FAMILY_LABEL_HORIZONS = {
    "carry": 480,        # funding settles every 8h on binance perps -> 480 1m bars
    "mean-reversion": 60,
    "trend": 10_080,     # a week of 1m bars
    "breakout": 1_440,   # a day
    "microstructure": 5,
}


class UnknownStrategyFamily(KeyError):
    """No label horizon is declared for this family, so purging cannot be correct.

    Fail closed. The alternative - defaulting to zero - disables the purge
    entirely, and does it for whichever family was least considered.
    """


@dataclass(frozen=True)
class CpcvFold:
    """One out-of-sample path: its test blocks, and what survived purging.

    `rows_purged` and `train_fraction_retained` are part of the result rather than
    a log line: a fold that lost 80% of its training data to a long horizon is
    still a fold, and whoever reads its Sharpe needs to know that.
    """

    test_groups: tuple[int, ...]
    test_blocks: tuple[tuple[int, int], ...]
    train_blocks: tuple[tuple[int, int], ...]
    rows_purged: int
    train_fraction_retained: float


def n_cpcv_paths(n_groups: int, k_test: int) -> int:
    """C(n_groups, k_test) - the number of out-of-sample paths."""
    return comb(n_groups, k_test)


def _group_bounds(n_rows: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous near-equal half-open [start, end) ranges spanning [0, n_rows).

    The remainder is spread over the leading groups rather than dumped on the
    last one, so no single fold is evaluated on a materially longer block than
    its peers - which would bias the Sharpe distribution the gates deflate
    against.
    """
    base, remainder = divmod(n_rows, n_groups)
    bounds, cursor = [], 0
    for index in range(n_groups):
        size = base + (1 if index < remainder else 0)
        bounds.append((cursor, cursor + size))
        cursor += size
    return bounds


def purge_and_embargo(train: tuple[int, int],
                      test_blocks: list[tuple[int, int]],
                      label_horizon: int,
                      embargo: int) -> list[tuple[int, int]]:
    """The parts of `train` that survive purge and embargo around every test block.

    The exclusion zone for a test block `[ts, te)` is

        [ts − label_horizon,  te + embargo)

    The left edge is the correction to the donor implementation. A training row at
    `ts − 1` whose label matures `label_horizon` rows later is scored on an
    outcome inside the test block; keeping it leaks the test set into training.
    The right edge is the embargo - serial correlation runs forward too, so a row
    just after the test block carries information about it even with no label
    overlap.

    Applied around **every** test block. With `k_test > 1` a path has several
    disjoint test blocks, and a purge that handles only the first leaks around all
    the rest.
    """
    if label_horizon < 0 or embargo < 0:
        raise ValueError(
            f"label_horizon and embargo must be >= 0, got {label_horizon} and "
            f"{embargo}")

    segments = [train]
    for test_start, test_end in test_blocks:
        low = test_start - label_horizon
        high = test_end + embargo
        survivors: list[tuple[int, int]] = []
        for start, end in segments:
            if end <= low or start >= high:
                survivors.append((start, end))
                continue
            if start < low:
                survivors.append((start, low))
            if end > high:
                survivors.append((high, end))
        segments = survivors
    # A block of one row cannot support a return, let alone a fit. Dropped rather
    # than carried, but the caller sees the loss through rows_purged.
    return [(s, e) for s, e in segments if e - s >= 2]


def combinatorial_purged_folds(n_rows: int, family: str, *,
                               n_groups: int = 6, k_test: int = 2,
                               embargo_pct: float = 0.01,
                               label_horizon: int | None = None) -> list[CpcvFold]:
    """Build the CPCV folds for a strategy family.

    `label_horizon` defaults to the family's declared horizon and can be
    overridden by a strategy that knows its own label window - the override exists
    so a correct horizon is always expressible, rather than being approximated by
    whichever family looks closest.
    """
    if n_groups < 3:
        raise ValueError(
            f"n_groups must be >= 3 for a meaningful CPCV, got {n_groups}")
    if not 1 <= k_test < n_groups:
        raise ValueError(f"k_test must be in [1, {n_groups}), got {k_test}")
    if n_rows < n_groups * 2:
        raise ValueError(
            f"need >= {n_groups * 2} rows for {n_groups} groups, got {n_rows} - "
            f"refused rather than producing empty groups, which would yield a "
            f"Sharpe distribution computed over nothing")

    if label_horizon is None:
        if family not in FAMILY_LABEL_HORIZONS:
            raise UnknownStrategyFamily(
                f"no label horizon declared for family {family!r}. Declare it in "
                f"FAMILY_LABEL_HORIZONS or pass label_horizon explicitly - "
                f"defaulting to zero would silently disable the purge for the "
                f"family least likely to have been thought about. Known: "
                f"{sorted(FAMILY_LABEL_HORIZONS)}")
        label_horizon = FAMILY_LABEL_HORIZONS[family]

    bounds = _group_bounds(n_rows, n_groups)
    embargo = max(1, int(n_rows * embargo_pct))

    folds: list[CpcvFold] = []
    for combo in combinations(range(n_groups), k_test):
        test_blocks = [bounds[i] for i in combo]
        train_source = [bounds[j] for j in range(n_groups) if j not in combo]
        available = sum(e - s for s, e in train_source)

        train_blocks: list[tuple[int, int]] = []
        for block in train_source:
            train_blocks += purge_and_embargo(block, test_blocks,
                                              label_horizon, embargo)

        retained = sum(e - s for s, e in train_blocks)
        folds.append(CpcvFold(
            test_groups=combo,
            test_blocks=tuple(test_blocks),
            train_blocks=tuple(train_blocks),
            rows_purged=available - retained,
            train_fraction_retained=(retained / available) if available else 0.0,
        ))
    return folds
