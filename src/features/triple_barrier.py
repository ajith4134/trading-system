"""Labels for what a trade would actually have done, not for where price ended.

`FEATURES.md` §2, FE-008: *"Triple-barrier labelling — barrier width is a
hyperparameter, it goes through the Trial Registry."* The corpus states the flaw
it repairs in `finml-feature-engineering.md` §2: naive fixed-horizon labelling
*"ignores path — can label a trade 'profitable' even though it breached the
actual stop-loss mid-window"*.

That is the whole design brief. A label reading +1 for a trade that would have
been stopped out two bars earlier is not a noisy label; it is a label for a trade
that never happened, and it is wrong in the direction that makes a backtest look
good. Three barriers — profit-take, stop-loss, and a holding-period ceiling —
and whichever is touched **first** decides the label.

**Not a port.** Ledger VX-055 records López de Prado triple-barrier code in an
early repo, and `CLAUDE.md` requires reading a donor rather than its ledger note.
That donor is not on this machine — `crypto-bot-publish` mirrors the current
repos, not the prior ones — so this is written from the method as the corpus
states it, and claims no inheritance it cannot show.

Four decisions, each one a way the naive version flatters itself:

**Path, never endpoints.** Barriers are checked against every bar's high and low.
A bar that closes up having traded through the stop is a stop.

**Ambiguity resolves to the stop, and says so.** When a single bar's range spans
both barriers, OHLC cannot say which was touched first. Awarding the profit-take
is a free win drawn from missing information — and drawn precisely on the most
volatile bars in the sample, which is where a strategy's worst trades live. The
stop takes it and `is_ambiguous` is set, so a caller can measure how much of the
dataset rests on that rule instead of inheriting it silently.

**An unresolved event is not a zero.** An event whose vertical barrier runs past
the end of the data has not been resolved. Labelling it 0 fills the newest
stretch of every dataset — the part closest to live — with an outcome nobody
observed. `label` is `None` and `is_resolved` is False; training on it is then a
deliberate act rather than an accident.

**Volatility is supplied, not computed here.** The barrier width scales with a
volatility estimate, and one computed over the whole series has seen the future.
The caller passes one value per EVENT, trailing as of that event, and a
mismatched length refuses rather than being broadcast into the wrong alignment.

Pure: no store, no clock, no filesystem. The same reason `paper.fill_model` is —
a function of (bars, events, widths) is provable without standing up an archive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

_REASON_PROFIT = "profit_take"
_REASON_STOP = "stop_loss"
_REASON_VERTICAL = "vertical"
_REASON_UNRESOLVED = "unresolved"


class UnalignedVolatility(ValueError):
    """One volatility estimate per event was not supplied.

    Refused rather than broadcast. A caller passing one value per BAR has almost
    certainly aligned the series to bars rather than to events, and the barriers
    would then be scaled by the wrong number for every event but the first -
    without anything complaining.
    """


@dataclass(frozen=True)
class Bars:
    """The path, as OHLC. `open` is not taken because nothing here uses it, and
    a field nobody reads is a field that goes stale without being noticed."""

    high: Sequence[float]
    low: Sequence[float]
    close: Sequence[float]

    def __post_init__(self) -> None:
        if not (len(self.high) == len(self.low) == len(self.close)):
            raise ValueError(
                f"bar series must be the same length: {len(self.high)} high(s), "
                f"{len(self.low)} low(s), {len(self.close)} close(s)")
        for index, (high, low) in enumerate(zip(self.high, self.low)):
            if low > high:
                raise ValueError(
                    f"bar {index} has a low ({low}) above its high ({high}). Not a "
                    f"labelling question, but a bar like this silently decides "
                    f"barrier touches, and this store has served malformed bars "
                    f"before - 746 of them, with a placeholder frame folded into "
                    f"`low` by min()")

    def __len__(self) -> int:
        return len(self.close)


@dataclass(frozen=True)
class BarrierTouch:
    """How one event resolved, and how long its information window really was.

    `touched_at_index` and `holding_bars` exist for FE-009 (sample uniqueness)
    and for the purge horizon. A label without its touch time forces both to
    assume the maximum holding period, which over-purges every fast trade and
    quietly shrinks the training set that survives.
    """

    event_index: int
    label: int | None
    reason: str
    touched_at_index: int | None
    entry_price: float
    upper_barrier: float
    lower_barrier: float
    is_ambiguous: bool

    @property
    def is_resolved(self) -> bool:
        return self.label is not None

    @property
    def holding_bars(self) -> int | None:
        if self.touched_at_index is None:
            return None
        return self.touched_at_index - self.event_index


def label_triple_barrier(
    bars: Bars,
    event_indices: Sequence[int],
    volatility: Sequence[float],
    *,
    profit_take_multiple: float,
    stop_loss_multiple: float,
    max_holding_bars: int,
) -> list[BarrierTouch]:
    """Label each event by whichever barrier its path touches first.

    `volatility[i]` is the trailing estimate as of `event_indices[i]`, expressed
    as a fraction of price. The barriers sit at

        upper = entry * (1 + profit_take_multiple * vol)
        lower = entry * (1 - stop_loss_multiple  * vol)

    The two multiples are separate because they are separate hyperparameters -
    every real risk rule is asymmetric, and one shared width cannot express that.
    Both go through the Trial Registry, per FE-008: a barrier width tuned until
    the labels look good is a search, and an untracked search is what the
    registry's N exists to count.

    The search for a touch starts at the bar AFTER the event. An event is decided
    on that bar's information, and a position opened on it cannot be filled and
    stopped by the same bar it was decided from.
    """
    if max_holding_bars <= 0:
        raise ValueError(
            f"max_holding_bars must be > 0, got {max_holding_bars}; a vertical "
            f"barrier at the event itself resolves every label before the "
            f"position exists")
    if len(volatility) != len(event_indices):
        raise UnalignedVolatility(
            f"{len(event_indices)} event(s) against {len(volatility)} volatility "
            f"estimate(s). One per EVENT, trailing as of that event - a series "
            f"aligned to bars instead would scale every barrier but the first by "
            f"the wrong number, and nothing downstream would notice")

    n = len(bars)
    touches: list[BarrierTouch] = []

    for event_index, event_volatility in zip(event_indices, volatility):
        if not 0 <= event_index < n:
            raise IndexError(
                f"event index {event_index} is outside a series of {n} bar(s)")
        if event_volatility <= 0:
            raise ValueError(
                f"volatility must be > 0, got {event_volatility} at event "
                f"{event_index}. Zero puts both barriers on the entry price, "
                f"where the first bar touches both and every label becomes an "
                f"ambiguous stop")

        entry = float(bars.close[event_index])
        upper = entry * (1.0 + profit_take_multiple * event_volatility)
        lower = entry * (1.0 - stop_loss_multiple * event_volatility)

        vertical_index = event_index + max_holding_bars
        # The horizontal barriers are searched only as far as the vertical one,
        # and only as far as the data goes. Those are different limits and the
        # difference is the whole unresolved case below.
        last_searchable = min(vertical_index, n - 1)

        resolution: BarrierTouch | None = None
        for index in range(event_index + 1, last_searchable + 1):
            hit_upper = float(bars.high[index]) >= upper
            hit_lower = float(bars.low[index]) <= lower
            if not (hit_upper or hit_lower):
                continue
            # Both inside one bar: unknowable from OHLC, so the stop takes it.
            ambiguous = hit_upper and hit_lower
            label = -1 if hit_lower else 1
            reason = _REASON_STOP if hit_lower else _REASON_PROFIT
            resolution = BarrierTouch(
                event_index=event_index, label=label, reason=reason,
                touched_at_index=index, entry_price=entry,
                upper_barrier=upper, lower_barrier=lower,
                is_ambiguous=ambiguous)
            break

        if resolution is None:
            if vertical_index < n:
                # The holding period ran out with the price between the barriers.
                # That is a real, observed outcome and 0 is its label.
                resolution = BarrierTouch(
                    event_index=event_index, label=0, reason=_REASON_VERTICAL,
                    touched_at_index=vertical_index, entry_price=entry,
                    upper_barrier=upper, lower_barrier=lower, is_ambiguous=False)
            else:
                # The data ran out first. Nothing was observed, so nothing is
                # claimed - and 0 is emphatically not the answer, because 0 is a
                # measured outcome and this is the absence of one.
                resolution = BarrierTouch(
                    event_index=event_index, label=None,
                    reason=_REASON_UNRESOLVED, touched_at_index=None,
                    entry_price=entry, upper_barrier=upper, lower_barrier=lower,
                    is_ambiguous=False)
        touches.append(resolution)

    return touches
