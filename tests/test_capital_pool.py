"""CL-04: the shared pool serialises two bots and never over-commits — RL-040.

The test that matters is `test_two_concurrent_reservers_never_overcommit`. The rest
exist so a failure there can be localised.
"""

import json
import multiprocessing
from decimal import Decimal

import pytest

from segment.capital_declaration import parse_declaration
from segment.capital_pool import (
    ALREADY_RESERVED,
    BELOW_MIN_MARGIN,
    BOT_CAP_REACHED,
    POOL_EXHAUSTED,
    CapitalPool,
    ledger_path,
)


def _declaration(portfolio="1000", cap="0.60", low="5", high="500"):
    return parse_declaration({
        "declared_at": "2026-08-19T00:00:00Z",
        "portfolio_usdt": portfolio,
        "per_bot_cap_fraction": cap,
        "min_margin_per_trade_usdt": low,
        "max_margin_per_trade_usdt": high,
        "leverage": {"rule": "fixed", "floor": "1",
                     "ceiling": {"perp": "10", "spot": "3"},
                     "target_annual_vol_pct": "40"},
        "spot_borrow_annual_pct": "8",
        "maintenance_margin_rate": "0.005",
    })


def _pool(tmp_path):
    return CapitalPool(tmp_path / "pool.ndjson")


def test_an_empty_pool_holds_nothing(tmp_path):
    assert _pool(tmp_path).held_by_segment() == {}


def test_a_reservation_is_granted_and_shows_up_as_held(tmp_path):
    pool = _pool(tmp_path)
    decision = pool.reserve(segment="perp", venue="binance-futures",
                            symbol="BTCUSDT", margin_usdt=Decimal("50"),
                            declaration=_declaration(), at_ns=1)

    assert decision.granted
    assert pool.held_by_segment() == {"perp": Decimal("50")}


def test_the_pool_refuses_once_the_portfolio_is_committed(tmp_path):
    # Cap at the whole portfolio so the PER-BOT cap cannot bite first: this test
    # is about the pool running out, and the two refusals have different fixes.
    declaration = _declaration(cap="1", high="1000")
    pool = _pool(tmp_path)
    for i in range(10):
        assert pool.reserve(segment="perp", venue="v", symbol=f"S{i}",
                            margin_usdt=Decimal("100"),
                            declaration=declaration, at_ns=i).granted

    refused = pool.reserve(segment="perp", venue="v", symbol="OVER",
                           margin_usdt=Decimal("100"),
                           declaration=declaration, at_ns=99)

    assert not refused.granted
    assert refused.reason == POOL_EXHAUSTED
    assert refused.headroom_usdt == Decimal("0")


def test_one_bot_cannot_take_more_than_its_cap(tmp_path):
    # 1000 x 0.60 = 600. The pool still has 400 free, so this refusal is the CAP
    # and must say so - the two are different problems with different fixes.
    declaration = _declaration()
    pool = _pool(tmp_path)
    for i in range(6):
        assert pool.reserve(segment="perp", venue="v", symbol=f"S{i}",
                            margin_usdt=Decimal("100"),
                            declaration=declaration, at_ns=i).granted

    refused = pool.reserve(segment="perp", venue="v", symbol="CAP",
                           margin_usdt=Decimal("100"),
                           declaration=declaration, at_ns=99)

    assert not refused.granted
    assert refused.reason == BOT_CAP_REACHED
    assert refused.headroom_usdt == Decimal("400")


def test_the_cap_on_one_bot_leaves_the_other_bot_its_share(tmp_path):
    declaration = _declaration()
    pool = _pool(tmp_path)
    for i in range(6):
        pool.reserve(segment="perp", venue="v", symbol=f"S{i}",
                     margin_usdt=Decimal("100"), declaration=declaration, at_ns=i)

    granted = pool.reserve(segment="spot", venue="v", symbol="BTCUSDT",
                           margin_usdt=Decimal("100"),
                           declaration=declaration, at_ns=99)

    assert granted.granted
    assert pool.held_by_segment() == {"perp": Decimal("600"), "spot": Decimal("100")}


def test_a_margin_below_the_declared_minimum_is_refused(tmp_path):
    refused = _pool(tmp_path).reserve(
        segment="perp", venue="v", symbol="S", margin_usdt=Decimal("1"),
        declaration=_declaration(), at_ns=1)

    assert refused.reason == BELOW_MIN_MARGIN


def test_reserving_the_same_instrument_twice_is_refused(tmp_path):
    # The second would overwrite the first and lose its margin from the total
    # while the position stayed open.
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="perp", venue="v", symbol="S", margin_usdt=Decimal("50"),
                 declaration=declaration, at_ns=1)

    again = pool.reserve(segment="perp", venue="v", symbol="S",
                         margin_usdt=Decimal("50"), declaration=declaration, at_ns=2)

    assert again.reason == ALREADY_RESERVED
    assert pool.held_by_segment() == {"perp": Decimal("50")}


def test_releasing_gives_the_margin_back(tmp_path):
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="perp", venue="v", symbol="S", margin_usdt=Decimal("50"),
                 declaration=declaration, at_ns=1)

    pool.release(segment="perp", venue="v", symbol="S", at_ns=2)

    assert pool.held_by_segment() == {}


def test_releasing_twice_is_safe(tmp_path):
    # A double close must not append a second RELEASE that a later replay could
    # pair with a genuine RESERVE.
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="perp", venue="v", symbol="S", margin_usdt=Decimal("50"),
                 declaration=declaration, at_ns=1)
    pool.release(segment="perp", venue="v", symbol="S", at_ns=2)
    pool.release(segment="perp", venue="v", symbol="S", at_ns=3)

    assert pool.held_by_segment() == {}
    rows = [json.loads(line) for line in
            (tmp_path / "pool.ndjson").read_text().splitlines() if line.strip()]
    assert sum(1 for r in rows if r["event"] == "RELEASE") == 1


def test_an_orphaned_reservation_is_reclaimed_at_startup(tmp_path):
    # Without this, every crash permanently shrinks the pool, and the shrinking
    # looks exactly like a bot that has grown cautious.
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="perp", venue="v", symbol="LIVE",
                 margin_usdt=Decimal("50"), declaration=declaration, at_ns=1)
    pool.reserve(segment="perp", venue="v", symbol="ORPHAN",
                 margin_usdt=Decimal("50"), declaration=declaration, at_ns=2)

    reclaimed = pool.reclaim_orphans(segment="perp", open_keys=[("v", "LIVE")],
                                     at_ns=3)

    assert reclaimed == [("v", "ORPHAN")]
    assert pool.held_by_segment() == {"perp": Decimal("50")}


def test_the_sweep_never_touches_the_other_bots_reservations(tmp_path):
    # One process deciding what a running process does not own is how a live
    # position loses its margin.
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="spot", venue="v", symbol="SPOTPOS",
                 margin_usdt=Decimal("50"), declaration=declaration, at_ns=1)

    pool.reclaim_orphans(segment="perp", open_keys=[], at_ns=2)

    assert pool.held_by_segment() == {"spot": Decimal("50")}


def test_a_torn_final_line_does_not_break_the_replay(tmp_path):
    # What a kill mid-write leaves behind. The reservation it would have described
    # was never completed, so no position exists against it.
    pool, declaration = _pool(tmp_path), _declaration()
    pool.reserve(segment="perp", venue="v", symbol="S", margin_usdt=Decimal("50"),
                 declaration=declaration, at_ns=1)
    with (tmp_path / "pool.ndjson").open("a") as handle:
        handle.write('{"at_ns": 2, "event": "RESE')

    assert pool.held_by_segment() == {"perp": Decimal("50")}


def _reserve_many(path, segment, count, portfolio):
    """One child process hammering the pool, for the concurrency test."""
    pool = CapitalPool(path)
    declaration = _declaration(portfolio=portfolio, cap="1", high=str(portfolio))
    granted = 0
    for i in range(count):
        if pool.reserve(segment=segment, venue="v", symbol=f"{segment}-{i}",
                        margin_usdt=Decimal("10"), declaration=declaration,
                        at_ns=i).granted:
            granted += 1
    return granted


def _child(path, segment, count, portfolio, queue):
    queue.put(_reserve_many(path, segment, count, portfolio))


def test_two_concurrent_reservers_never_overcommit(tmp_path):
    """**The reason this module exists.**

    Two processes, 60 attempts each at 10 USDT, against a 400 USDT pool. Without
    the lock both read the same headroom and both grant, and the total lands above
    400 - the lost update, which drifts in the direction that always looks fine.
    """
    path = tmp_path / "pool.ndjson"
    portfolio = "400"
    queue = multiprocessing.Queue()
    children = [
        multiprocessing.Process(target=_child,
                                args=(path, segment, 60, portfolio, queue))
        for segment in ("perp", "spot")
    ]
    for child in children:
        child.start()
    granted = sum(queue.get() for _ in children)
    for child in children:
        child.join(timeout=60)
        assert child.exitcode == 0

    held = CapitalPool(path).held_by_segment()
    total = sum(held.values(), Decimal(0))

    assert total <= Decimal(portfolio), f"pool over-committed to {total}"
    assert granted == 40, f"400 / 10 = 40 reservations should fit, got {granted}"
    assert total == Decimal("400")


def test_the_default_ledger_sits_beside_the_segment_state(tmp_path):
    assert ledger_path(tmp_path) == tmp_path / "pool.ndjson"


@pytest.mark.parametrize("segment", ["perp", "spot"])
def test_the_decision_reports_the_pool_state_it_results_in(tmp_path, segment):
    # A decision that does not carry the numbers behind it sends the reader to
    # guess, which is where the wrong dial gets turned. Resulting state, not
    # prior: a grant reports the pool as a reader would now see it.
    pool, declaration = _pool(tmp_path), _declaration()
    decision = pool.reserve(segment=segment, venue="v", symbol="S",
                            margin_usdt=Decimal("50"), declaration=declaration,
                            at_ns=1)

    assert decision.headroom_usdt == Decimal("950")
    assert decision.held_by_segment == {segment: Decimal("50")}
    assert decision.total_held_usdt == Decimal("50")
    assert decision.as_dict()["margin_usdt"] == "50"


@pytest.mark.parametrize("segment", ["perp", "spot"])
def test_a_refusal_reports_the_unchanged_pool(tmp_path, segment):
    pool, declaration = _pool(tmp_path), _declaration()
    refused = pool.reserve(segment=segment, venue="v", symbol="S",
                           margin_usdt=Decimal("1"), declaration=declaration,
                           at_ns=1)

    assert not refused.granted
    assert refused.headroom_usdt == Decimal("1000")
    assert refused.held_by_segment == {}
