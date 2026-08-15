"""FE-008 — labels that reflect what a trade would actually have done.

`FEATURES.md` §2: *"Triple-barrier labelling — barrier width is a hyperparameter,
it goes through the Trial Registry."* The corpus states the flaw it repairs in
`finml-feature-engineering.md` §2: naive fixed-horizon labelling *"ignores path —
can label a trade 'profitable' even though it breached the actual stop-loss
mid-window"*.

That sentence is the whole test suite. A label that says +1 for a trade which
would have been stopped out two bars earlier is not a slightly noisy label, it is
a label for a trade that never happened — and it is wrong in the direction that
makes a backtest look good.

Four things are defended here, each a way the naive version flatters itself:

**Path, not endpoints.** The barriers are checked against every bar's high and
low, never its close alone. A bar that closes up having traded through the stop
is a stop.

**Ambiguity is refused, not guessed.** When one bar's range spans both barriers,
OHLC cannot say which was touched first. Awarding the profit-take there is a free
win drawn from missing information. The stop takes it, and the label is flagged
ambiguous so the caller can count how much of the dataset rests on that rule.

**An unresolved event is not a zero.** An event whose vertical barrier extends
past the end of the data has not been resolved. Labelling it 0 is the quiet bug:
it fills the most recent stretch of every dataset — the part closest to live —
with a label meaning "nothing happened" that was never observed.

**Volatility must be trailing.** The barrier width is scaled by a volatility
estimate, and one computed over the whole series has seen the future. The
estimate is supplied per event by the caller, and a missing one refuses.
"""
import pytest

from features.triple_barrier import (
    BarrierTouch,
    Bars,
    UnalignedVolatility,
    label_triple_barrier,
)


def _bars(rows):
    """rows: (high, low, close). Open is not used and is not invented."""
    return Bars(high=[r[0] for r in rows],
                low=[r[1] for r in rows],
                close=[r[2] for r in rows])


def _flat(n, price=100.0):
    return _bars([(price, price, price)] * n)


# --- the vertical barrier ----------------------------------------------------

def test_an_event_that_touches_nothing_is_labelled_zero_at_the_vertical_barrier():
    bars = _flat(10)
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.10],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == 0
    assert touch.reason == "vertical"
    assert touch.touched_at_index == 5


def test_the_vertical_barrier_is_counted_in_bars_from_the_event():
    bars = _flat(20)
    (touch,) = label_triple_barrier(
        bars, event_indices=[3], volatility=[0.10],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=4)
    assert touch.touched_at_index == 7


# --- the profit and stop barriers -------------------------------------------

def test_a_move_through_the_upper_barrier_is_labelled_plus_one():
    # 1% vol, 2x multiple -> +2% barrier at 102. Bar 2 highs to 103.
    bars = _bars([(100, 100, 100), (100.5, 99.5, 100), (103, 100, 102.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == 1
    assert touch.reason == "profit_take"
    assert touch.touched_at_index == 2


def test_a_move_through_the_lower_barrier_is_labelled_minus_one():
    bars = _bars([(100, 100, 100), (100.5, 99.5, 100), (100, 97, 97.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == -1
    assert touch.reason == "stop_loss"


def test_the_first_barrier_touched_wins_not_the_larger_move():
    """A trade exits at the first barrier it touches. A later, bigger move in the
    other direction never happened to that position."""
    bars = _bars([(100, 100, 100), (100, 97, 97.5), (110, 100, 109)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == -1
    assert touch.touched_at_index == 1


# --- path, not endpoints. The defect the method exists to repair -------------

def test_a_bar_that_closes_up_having_traded_through_the_stop_is_a_stop():
    """The exact failure `finml-feature-engineering.md` names: naive labelling
    'can label a trade profitable even though it breached the actual stop-loss
    mid-window'. Close-only labelling calls this +1."""
    bars = _bars([(100, 100, 100), (103, 97, 102.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == -1, "close-only labelling would call this a win"


def test_a_bar_spanning_both_barriers_is_flagged_ambiguous():
    bars = _bars([(100, 100, 100), (103, 97, 100)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.is_ambiguous is True


def test_an_ambiguous_bar_resolves_to_the_stop_not_the_profit():
    """OHLC cannot say which barrier was touched first inside one bar. Awarding
    the profit-take is a free win drawn from missing information, and it is
    drawn on exactly the most volatile bars in the sample."""
    bars = _bars([(100, 100, 100), (103, 97, 100)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.label == -1


def test_an_unambiguous_touch_is_not_flagged():
    bars = _bars([(100, 100, 100), (103, 100, 102.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.is_ambiguous is False


# --- an unresolved event is not a zero ---------------------------------------

def test_an_event_whose_vertical_barrier_runs_past_the_data_is_unresolved():
    """The quiet bug: labelling it 0 fills the newest stretch of every dataset -
    the part closest to live - with an outcome nobody observed."""
    bars = _flat(4)
    (touch,) = label_triple_barrier(
        bars, event_indices=[2], volatility=[0.10],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=5)
    assert touch.reason == "unresolved"
    assert touch.label is None


def test_an_unresolved_event_that_touched_a_barrier_first_is_still_resolved():
    """Running out of data only matters if nothing happened before it did."""
    bars = _bars([(100, 100, 100), (100, 100, 100), (103, 100, 102.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=99)
    assert touch.label == 1


def test_resolved_labels_can_be_separated_from_unresolved_ones():
    bars = _bars([(100, 100, 100)] * 3 + [(103, 100, 102.5)] + [(100, 100, 100)] * 2)
    touches = label_triple_barrier(
        bars, event_indices=[0, 5], volatility=[0.01, 0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=4)
    resolved = [t for t in touches if t.is_resolved]
    assert len(resolved) == 1 and resolved[0].label == 1


# --- volatility is trailing, supplied, and mandatory -------------------------

def test_a_volatility_series_of_the_wrong_length_is_refused():
    """One volatility per event. A caller passing the whole series' worth has
    almost certainly aligned it to bars rather than to events, and the barriers
    would then be scaled by the wrong number without complaint."""
    with pytest.raises(UnalignedVolatility):
        label_triple_barrier(_flat(10), event_indices=[0, 1], volatility=[0.01],
                             profit_take_multiple=2.0, stop_loss_multiple=2.0,
                             max_holding_bars=3)


def test_a_non_positive_volatility_is_refused_rather_than_collapsing_the_barriers():
    """Zero volatility puts both barriers on the entry price, where the first
    bar touches both and every label becomes an ambiguous stop."""
    with pytest.raises(ValueError, match="volatility"):
        label_triple_barrier(_flat(10), event_indices=[0], volatility=[0.0],
                             profit_take_multiple=2.0, stop_loss_multiple=2.0,
                             max_holding_bars=3)


def test_the_barrier_widths_scale_with_the_supplied_volatility():
    """Same path, five times the volatility, and the wider barriers are not
    reached. Padded out to the vertical barrier deliberately: with too few bars
    the wide case comes back UNRESOLVED rather than 0, which is correct and
    would make this test pass for the wrong reason."""
    bars = _bars([(100, 100, 100), (103, 100, 102.5)] + [(102.5, 102.5, 102.5)] * 4)
    tight = label_triple_barrier(bars, event_indices=[0], volatility=[0.01],
                                 profit_take_multiple=2.0, stop_loss_multiple=2.0,
                                 max_holding_bars=5)[0]
    wide = label_triple_barrier(bars, event_indices=[0], volatility=[0.05],
                                profit_take_multiple=2.0, stop_loss_multiple=2.0,
                                max_holding_bars=5)[0]
    assert tight.label == 1
    assert wide.label == 0


def test_asymmetric_multiples_are_honoured_independently():
    """Profit-take and stop-loss widths are separate hyperparameters. A single
    shared width cannot express the asymmetry every real risk rule has."""
    bars = _bars([(100, 100, 100), (102, 98.5, 100)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=5.0, stop_loss_multiple=1.0, max_holding_bars=5)
    assert touch.label == -1 and touch.reason == "stop_loss"


# --- refusals ----------------------------------------------------------------

def test_an_event_index_outside_the_series_is_refused():
    with pytest.raises(IndexError):
        label_triple_barrier(_flat(5), event_indices=[9], volatility=[0.01],
                             profit_take_multiple=2.0, stop_loss_multiple=2.0,
                             max_holding_bars=2)


def test_bars_of_unequal_length_are_refused():
    with pytest.raises(ValueError):
        Bars(high=[1.0, 2.0], low=[1.0], close=[1.0, 2.0])


def test_a_low_above_its_own_high_is_refused_as_a_corrupt_bar():
    """It is not a labelling question, but a bar like this silently decides
    barrier touches, and the store has served malformed bars before."""
    with pytest.raises(ValueError, match="high"):
        Bars(high=[100.0], low=[101.0], close=[100.5])


def test_a_non_positive_holding_period_is_refused():
    with pytest.raises(ValueError, match="max_holding_bars"):
        label_triple_barrier(_flat(5), event_indices=[0], volatility=[0.01],
                             profit_take_multiple=2.0, stop_loss_multiple=2.0,
                             max_holding_bars=0)


# --- the touch index is what FE-009 and purged CV need -----------------------

def test_every_touch_carries_the_index_it_resolved_at():
    """Sample uniqueness (FE-009) and the purge horizon both key on how long a
    label's information window actually was. A label without its touch time
    forces both to assume the maximum, which over-purges every fast trade."""
    bars = _bars([(100, 100, 100), (100, 100, 100), (103, 100, 102.5)])
    (touch,) = label_triple_barrier(
        bars, event_indices=[0], volatility=[0.01],
        profit_take_multiple=2.0, stop_loss_multiple=2.0, max_holding_bars=9)
    assert isinstance(touch, BarrierTouch)
    assert touch.touched_at_index == 2
    assert touch.holding_bars == 2
