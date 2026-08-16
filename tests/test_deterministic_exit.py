"""A lock that can loosen is not a lock, and a bar is a summary, not a path.

Two properties carry this module. The ratchet must be monotone under every input,
because the spec says "it never widens, under any condition, for any model
output" - and the way that dies is a helpful branch someone adds later. And a bar
that touched both rails contains both outcomes in an unknown order, so the
adverse one has to win; assuming the favourable one is worth between a few basis
points and the whole trade, on exactly the bars that matter most.
"""
from decimal import Decimal

import pytest

from strategy.deterministic_exit import (
    ATR_PERIOD,
    HARD_STOP,
    LONG,
    PRECEDENCE,
    PROFIT_LOCK,
    PROFIT_TARGET,
    SHORT,
    STILL_OPEN,
    VERTICAL_BARRIER,
    Bar,
    ExitPolicy,
    ExitSummary,
    NoTrueRange,
    average_true_range,
    ratchet_profit_lock,
    run_exit_policy,
    true_range,
)

_ATR = Decimal("1")
_ENTRY = Decimal("100")


def _bar(high, low, close=None):
    high, low = Decimal(str(high)), Decimal(str(low))
    return Bar(high=high, low=low,
               close=Decimal(str(close)) if close is not None else (high + low) / 2)


def _flat(n, price="100"):
    return [_bar(price, price, price) for _ in range(n)]


# --- the ratchet ----------------------------------------------------------

def test_the_lock_never_moves_against_a_long():
    """Monotone by construction. The way this dies is a helpful branch someone
    adds later, so it is tested as a property rather than on one example."""
    lock = None
    for candidate in (Decimal(90), Decimal(95), Decimal(80), Decimal(97),
                      Decimal(10), Decimal(99)):
        new = ratchet_profit_lock(LONG, lock, candidate)
        if lock is not None:
            assert new >= lock, f"the lock moved down from {lock} to {new}"
        lock = new

    assert lock == Decimal(99), "it ends at the highest candidate ever seen"


def test_the_lock_never_moves_against_a_short():
    lock = None
    for candidate in (Decimal(110), Decimal(105), Decimal(120), Decimal(103)):
        new = ratchet_profit_lock(SHORT, lock, candidate)
        if lock is not None:
            assert new <= lock
        lock = new

    assert lock == Decimal(103)


def test_an_unknown_side_is_refused():
    with pytest.raises(ValueError):
        ratchet_profit_lock("FLAT", None, Decimal(1))


# --- the ATR --------------------------------------------------------------

def test_the_true_range_includes_the_gap_from_the_previous_close():
    """The whole reason it is not just high minus low: an instrument that gapped
    overnight moved, and a range that ignores the gap reports a quiet bar on the
    most violent night of the year."""
    gapped = _bar(110, 108)

    assert true_range(gapped, previous_close=Decimal(100)) == Decimal(10)
    assert gapped.high - gapped.low == Decimal(2)


def test_the_atr_is_the_mean_of_the_last_period_true_ranges():
    bars = [_bar(101, 99, 100) for _ in range(ATR_PERIOD + 1)]

    assert average_true_range(bars) == Decimal(2)


def test_too_few_bars_refuses_rather_than_defaulting_to_a_percentage():
    """A trailing distance that silently stops being volatility-scaled either
    strangles the tail or protects nothing, and neither shows in the output."""
    with pytest.raises(NoTrueRange, match="strangles the tail"):
        average_true_range([_bar(101, 99)] * 3)


# --- precedence -----------------------------------------------------------

def test_the_hard_stop_fires_before_anything_else():
    """§3b: the hard stop overrides absolutely. Its mandate to close in profit is
    an objective, not a veto."""
    # This bar touches the target at 104 AND the hard stop at 97.
    record = run_exit_policy(LONG, _ENTRY, [_bar(105, 96, 100)], _ATR)

    assert record.exit_reason == HARD_STOP
    assert record.exit_price == Decimal(97)


def test_a_bar_touching_both_rails_is_marked_ambiguous():
    """A one-minute bar is a summary, not a path. The order within it is unknown,
    and assuming the favourable one is worth the whole trade on exactly the bars
    that matter most."""
    record = run_exit_policy(LONG, _ENTRY, [_bar(105, 96, 100)], _ATR)

    assert record.is_ambiguous


def test_a_clean_target_hit_is_not_ambiguous():
    """A flag that fires on everything says nothing."""
    record = run_exit_policy(LONG, _ENTRY, [_bar(105, 99.5, 104)], _ATR)

    assert record.exit_reason == PROFIT_TARGET
    assert not record.is_ambiguous


def test_the_declared_precedence_is_the_evaluated_one():
    """The order changes the reported P&L of every trade, so it is declared
    rather than emergent."""
    assert PRECEDENCE == (HARD_STOP, PROFIT_LOCK, PROFIT_TARGET,
                          VERTICAL_BARRIER)


# --- the four exits, each reachable --------------------------------------

def test_a_run_to_target_exits_on_the_target():
    path = [_bar(101, 100, 100.5), _bar(103, 101, 102), _bar(105, 103, 104)]
    record = run_exit_policy(LONG, _ENTRY, path, _ATR)

    assert record.exit_reason == PROFIT_TARGET
    assert record.exit_price == Decimal(104)
    assert record.pnl_per_unit == Decimal(4)


def test_a_move_that_stalls_exits_on_the_lock():
    """The rail the policy exists for: a gain that arrived and then gave back."""
    path = [_bar(103, 100, 102.5),      # arms the ratchet at +3 ATR
            _bar(103, 101, 101.5),
            _bar(102, 100, 100.5)]      # retraces into the lock
    record = run_exit_policy(LONG, _ENTRY, path, _ATR)

    assert record.exit_reason == PROFIT_LOCK
    assert record.exit_price > _ENTRY, "the lock kept a gain, which is its job"
    assert record.final_lock is not None


def test_a_position_that_does_nothing_exits_on_time():
    policy = ExitPolicy(max_holding_bars=5)
    record = run_exit_policy(LONG, _ENTRY, _flat(10), _ATR, policy)

    assert record.exit_reason == VERTICAL_BARRIER
    assert record.holding_bars == 5


def test_a_path_that_runs_out_is_still_open_not_closed():
    """Reporting the last close as an exit would invent a trade that never
    closed - the same refusal `features.triple_barrier` makes for an unresolved
    label."""
    record = run_exit_policy(LONG, _ENTRY, _flat(3), _ATR)

    assert record.exit_reason == STILL_OPEN


def test_every_exit_reason_is_reachable():
    """A rail nothing can trigger is a rail that is not in the policy."""
    policy = ExitPolicy(max_holding_bars=4)
    reasons = {
        run_exit_policy(LONG, _ENTRY, [_bar(101, 96, 97)], _ATR).exit_reason,
        run_exit_policy(LONG, _ENTRY, [_bar(105, 100, 104)], _ATR).exit_reason,
        run_exit_policy(LONG, _ENTRY, [_bar(103, 100, 102.5), _bar(103, 101, 101.5),
                                       _bar(102, 100, 100.5)], _ATR).exit_reason,
        run_exit_policy(LONG, _ENTRY, _flat(8), _ATR, policy).exit_reason,
    }

    assert reasons == {HARD_STOP, PROFIT_TARGET, PROFIT_LOCK, VERTICAL_BARRIER}


# --- shorts are not longs with a sign -------------------------------------

def test_a_short_stops_out_upward():
    record = run_exit_policy(SHORT, _ENTRY, [_bar(104, 100, 103)], _ATR)

    assert record.exit_reason == HARD_STOP
    assert record.exit_price == Decimal(103)
    assert record.pnl_per_unit == Decimal(-3)


def test_a_short_takes_profit_downward():
    record = run_exit_policy(SHORT, _ENTRY, [_bar(100, 95, 96)], _ATR)

    assert record.exit_reason == PROFIT_TARGET
    assert record.exit_price == Decimal(96)
    assert record.pnl_per_unit == Decimal(4)


# --- the dataset PROFIT-TAIL trains on ------------------------------------

def test_the_record_carries_the_shape_of_the_move_not_just_its_end():
    """§3b lists "realised path data - the shape of the move after entry, not
    just its endpoint" as what PROFIT-TAIL needs and the directional bots do not.
    A record with only the exit price cannot tell a trade that went straight to
    target from one that nearly stopped out first."""
    path = [_bar(101, 97.5, 98),        # nearly stopped out
            _bar(105, 100, 104)]        # then ran to target
    record = run_exit_policy(LONG, _ENTRY, path, _ATR)

    assert record.exit_reason == PROFIT_TARGET
    assert record.max_favourable_excursion == Decimal(5)
    assert record.max_adverse_excursion == Decimal("-2.5")


def test_the_atr_is_measured_once_at_entry_and_does_not_follow():
    """A trailing distance that re-measured every bar would tighten into a
    volatility collapse and loosen into a spike - a second undeclared policy
    riding inside the first."""
    record = run_exit_policy(LONG, _ENTRY, [_bar(105, 100, 104)], _ATR)

    assert record.atr_at_entry == _ATR


def test_a_zero_atr_is_refused():
    """It puts every rail on the entry price, where the first bar touches all of
    them and the exit reason becomes whichever was checked first."""
    with pytest.raises(NoTrueRange, match="checked first"):
        run_exit_policy(LONG, _ENTRY, [_bar(101, 99)], Decimal(0))


def test_an_empty_path_is_refused():
    with pytest.raises(ValueError, match="nothing to exit through"):
        run_exit_policy(LONG, _ENTRY, [], _ATR)


# --- the summary ----------------------------------------------------------

def test_the_summary_reports_by_reason_rather_than_one_win_rate():
    """The reasons are the diagnosis: mostly vertical means it is not trading,
    mostly hard stop means the rails are in the wrong order, mostly lock means it
    is doing what it was built to do. One number hides all three."""
    records = [
        run_exit_policy(LONG, _ENTRY, [_bar(105, 100, 104)], _ATR),
        run_exit_policy(LONG, _ENTRY, [_bar(101, 96, 97)], _ATR),
    ]
    summary = ExitSummary(records)

    assert summary.by_reason[PROFIT_TARGET] == 1
    assert summary.by_reason[HARD_STOP] == 1
    assert summary.total_pnl_per_unit() == Decimal(1)      # +4 and -3
    assert "profit_target 1" in summary.describe()


def test_the_summary_counts_ambiguous_bars(tmp_path):
    summary = ExitSummary([
        run_exit_policy(LONG, _ENTRY, [_bar(105, 96, 100)], _ATR)])

    assert summary.ambiguous == 1
    assert "resolved against the position" in summary.describe()


def test_an_empty_summary_says_so():
    assert ExitSummary([]).describe() == "no exits recorded"


# --- what is deliberately absent ------------------------------------------

def test_nothing_here_places_an_order():
    """MD-018 requires the lock to rest at the exchange as a reduce-only stop,
    because a lock held only in memory protects nothing during a crash, deploy or
    partition - exactly the events it exists for. That is a separate row needing
    an order path this system does not have, and it is recorded as absent rather
    than implied."""
    import ast

    import strategy.deterministic_exit as module

    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not any("execution" in name or "paper" in name for name in imported)
