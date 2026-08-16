"""The paper equity curve, and the adaptive cap measured against it.

`risk.paper_tail_cap` was built on the user's instruction that the paper ceiling
adapt — *"for paper trading it is for practising and experimenting so make these
dynamic or auto adjust"* — and until now nothing fed it. A cap that governs
nothing is a cap nobody has to obey, and this is what makes it operative.

## Two accountings, all the way through

Equity is tracked under both, never blended. `paper.paper_broker` keeps two
position books that fill at identical sizes and different prices, and a single
equity curve would have to pick one — which on a strategy whose participation is
uncalibrated is picking how good the answer is allowed to look.

The cap is derived from and assessed against the **pessimistic** curve. Deriving
it from the optimistic one would set a limit around a book that never existed,
and the whole purpose of a drawdown limit is to bound the bad case.

## What is a "return" here, and the honest limit of it

The series fed to the bootstrap is the **per-poll change in equity as a fraction
of the declared NAV**. Two things about that are worth stating rather than
leaving to be discovered:

* A poll is not a fixed interval. The engine polls on a timer, but a poll with no
  new bars produces no change, so the series is irregular in wall-clock terms and
  a drawdown measured over it is a drawdown over *activity*, not over time. On a
  feed that goes quiet for an hour, that understates how long a drawdown lasted
  and says nothing wrong about how deep it went.
* Unrealised P&L needs a mark, and an unmarked position contributes **nothing**
  rather than zero — the same distinction `paper.blotter` makes. So an equity
  curve built while positions are unmarked is a curve of realised P&L only, and
  `unmarked_positions` rides every sample so a reader knows which they are
  looking at.

## Paper breaches are recorded, never enforced

`risk.tail_cap.assess` in `PAPER` mode returns `would_have_breached` rather than
halting, by the design's own ruling, and that is kept: nothing here stops the
engine. A paper run that could not have lived inside its ceiling has not earned
capital, and the record of that is the deliverable — enforcing it would destroy
the evidence by preventing the breach it exists to observe.

The **live** ceiling is never touched. `risk.paper_tail_cap` cannot produce one
and this module cannot ask for one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from risk.paper_tail_cap import PaperCap, derive_paper_cap
from risk.tail_cap import PAPER, CapAssessment, TailCap, assess

# How many equity samples before the adaptive cap is even attempted. Below it
# `derive_paper_cap` passes the live ceiling through unchanged and says so; this
# constant exists so the caller does not run a bootstrap on four points to be
# told that.
MIN_SAMPLES_TO_DERIVE = 100


@dataclass(frozen=True)
class EquitySample:
    """One observation of the book, under both accountings.

    `unmarked_positions` rides every sample because an equity curve built while
    positions are unmarked is a curve of REALISED P&L only, and a reader who
    cannot tell the two apart will read one as the other.
    """
    at_ns: int
    optimistic: Decimal
    pessimistic: Decimal
    unmarked_positions: int


@dataclass
class EquityCurve:
    """The paper book's equity through time, and the cap measured against it.

    Holds no policy: the ceiling comes from `risk.tail_cap`, the adaptive paper
    cap from `risk.paper_tail_cap`, and the breach verdict from `assess`. This
    assembles them and records what they said.
    """
    nav: Decimal
    live_ceiling: TailCap
    samples: list[EquitySample] = field(default_factory=list)
    would_have_breached: int = 0
    breaches_by_reason: dict[str, int] = field(default_factory=dict)
    _peak: Decimal | None = None
    _day_open: Decimal | None = None

    def record(self, *, at_ns: int, realised_optimistic: Decimal,
               realised_pessimistic: Decimal,
               unmarked_positions: int = 0) -> EquitySample:
        """Add one observation. Equity is NAV plus realised P&L.

        Unrealised is deliberately absent rather than assumed zero: an unmarked
        position has an unknown contribution, and folding it in as zero would
        report a flat book when the truth is an unmeasured one.
        """
        sample = EquitySample(
            at_ns=int(at_ns),
            optimistic=self.nav + realised_optimistic,
            pessimistic=self.nav + realised_pessimistic,
            unmarked_positions=int(unmarked_positions))
        self.samples.append(sample)
        if self._day_open is None:
            self._day_open = sample.pessimistic
        self._peak = (sample.pessimistic if self._peak is None
                      else max(self._peak, sample.pessimistic))
        return sample

    # --- the cap ----------------------------------------------------------

    def returns(self) -> list[float]:
        """Per-poll equity change as a fraction of NAV, pessimistic.

        Pessimistic because the cap's purpose is to bound the bad case, and a
        limit derived from the optimistic curve would be set around a book that
        never existed.
        """
        if len(self.samples) < 2 or self.nav <= 0:
            return []
        return [float((later.pessimistic - earlier.pessimistic) / self.nav)
                for earlier, later in zip(self.samples, self.samples[1:])]

    def paper_cap(self, *, n_samples: int = 1000, block_size: int = 50,
                  seed: int = 0) -> PaperCap:
        """The adaptive cap derived from this curve, bounded both ways.

        Returns the live ceiling passed through - `is_derived` False - whenever
        there is too little history, rather than raising, because the early state
        of every paper run is "not enough yet" and a board should render that
        without a try block.
        """
        series = self.returns()
        block = min(block_size, max(2, len(series) - 1)) if series else block_size
        return derive_paper_cap(self.live_ceiling, series,
                                n_samples=n_samples, block_size=block,
                                seed=seed)

    def assess_latest(self, cap: TailCap | None = None) -> CapAssessment | None:
        """Measure the newest sample against the cap. PAPER mode, always.

        `None` before there is anything to measure. Paper mode means a breach is
        reported as `would_have_breached` and nothing is stopped - enforcing it
        would destroy the evidence by preventing the breach it exists to observe.
        """
        if not self.samples:
            return None
        latest = self.samples[-1]
        judged = assess(
            cap=cap or self.paper_cap().cap,
            nav=latest.pessimistic,
            day_open_nav=self._day_open or latest.pessimistic,
            peak_nav=self._peak or latest.pessimistic,
            mode=PAPER)
        if judged.would_have_breached:
            self.would_have_breached += 1
            reason = judged.reason()
            self.breaches_by_reason[reason] = (
                self.breaches_by_reason.get(reason, 0) + 1)
        return judged

    def describe(self) -> str:
        if not self.samples:
            return "no equity samples - the paper book has not been observed yet"
        latest = self.samples[-1]
        unmarked = (f", {latest.unmarked_positions} position(s) unmarked so this "
                    f"is realised P&L only" if latest.unmarked_positions else "")
        breaches = (f"; {self.would_have_breached} would-have-breach(es) "
                    f"recorded and none enforced"
                    if self.would_have_breached else "; no breach recorded")
        return (f"{len(self.samples)} sample(s); equity "
                f"{latest.pessimistic} pessimistic against "
                f"{latest.optimistic} optimistic, peak {self._peak}"
                f"{unmarked}{breaches}")
