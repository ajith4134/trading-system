"""The central rate budget — one bucket per venue, shared by every caller.

`ARCHITECTURE.md` Layer 3 is emphatic that this is architecture rather than
configuration: *"Rate limits are an architectural constraint, not a config value:
per-IP, exchange-wide. Binance `418` bans scale **2 minutes → 3 days** for repeat
offenders. One strategy's burst bans all of them."*

Two consequences the tests below pin down.

**Weight, not request count.** Measured 2026-08-08 off the `x-mbx-used-weight`
header: a `limit=1000` spot depth snapshot costs **50**, while `premiumIndex`
costs **1**. A budgeter counting requests would let fifty snapshots through as
cheaply as fifty funding polls and earn the ban it exists to prevent.

**Shared across processes.** Capture runs one process per venue and the limit is
per-IP, so a bucket living in one process's memory protects nothing. Two
budgeters pointed at the same state must see each other's spending.
"""
import pytest

from ops.rate_budget import RateBudget


def test_spending_inside_the_budget_is_allowed(tmp_path):
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=10,
                   clock=lambda: 0.0)
    assert b.try_spend(50)
    assert b.remaining() == 50


def test_spending_past_the_budget_is_refused_not_queued(tmp_path):
    """Refused rather than silently delayed: the caller decides whether to wait,
    and a hidden sleep inside a budgeter is how a capture loop stalls without
    anything reporting why."""
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=10,
                   clock=lambda: 0.0)
    assert b.try_spend(100)
    assert not b.try_spend(1)
    assert b.remaining() == 0


def test_weight_is_what_is_counted_not_requests(tmp_path):
    """A limit=1000 spot snapshot is 50 weight; premiumIndex is 1. Counting
    requests would let fifty snapshots through as cheaply as fifty polls."""
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=10,
                   clock=lambda: 0.0)
    for _ in range(50):
        assert b.try_spend(1), "fifty cheap polls should fit in 100 weight"
    assert b.try_spend(50)
    assert not b.try_spend(1)


def test_the_budget_refills_over_time(tmp_path):
    now = {"t": 0.0}
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=10,
                   clock=lambda: now["t"])
    assert b.try_spend(100)
    assert not b.try_spend(10)
    now["t"] = 2.0
    assert b.try_spend(20), "two seconds at 10/s should restore 20"


def test_refill_never_exceeds_capacity(tmp_path):
    """A bucket that overfills while idle hands out a burst the venue will ban."""
    now = {"t": 0.0}
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=10,
                   clock=lambda: now["t"])
    now["t"] = 10_000.0
    assert b.remaining() == 100


def test_two_budgeters_on_the_same_state_see_each_others_spending(tmp_path):
    """The limit is per-IP and capture runs a process per venue, so a bucket in
    one process's memory protects nothing at all."""
    first = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=0,
                       clock=lambda: 0.0)
    second = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=0,
                        clock=lambda: 0.0)
    assert first.try_spend(80)
    assert not second.try_spend(30), "second process did not see the first's spend"
    assert second.try_spend(20)


def test_venues_have_separate_budgets(tmp_path):
    """Limits are per-IP *per exchange*. Spending Binance's budget must not
    throttle Hyperliquid."""
    binance = RateBudget(tmp_path, "binance", capacity=10, refill_per_second=0,
                         clock=lambda: 0.0)
    hyper = RateBudget(tmp_path, "hyperliquid", capacity=10, refill_per_second=0,
                       clock=lambda: 0.0)
    assert binance.try_spend(10)
    assert not binance.try_spend(1)
    assert hyper.try_spend(10)


# --- what the venue tells us overrides what we think ------------------------

def test_a_retry_after_blocks_every_spend_until_it_expires(tmp_path):
    """`ARCHITECTURE.md`: honour `Retry-After` over your own schedule. Our bucket
    is a model of the venue's limit; the header is the venue itself speaking."""
    now = {"t": 0.0}
    b = RateBudget(tmp_path, "binance", capacity=1000, refill_per_second=100,
                   clock=lambda: now["t"])
    b.note_rate_limited(retry_after_seconds=30)
    assert not b.try_spend(1), "spent during a Retry-After window"
    now["t"] = 29.0
    assert not b.try_spend(1)
    now["t"] = 31.0
    assert b.try_spend(1)


def test_a_ban_is_respected_even_with_a_full_bucket(tmp_path):
    """A 418 means the venue has already decided. A full local bucket is our
    model being wrong, not permission to continue."""
    b = RateBudget(tmp_path, "binance", capacity=1000, refill_per_second=0,
                   clock=lambda: 0.0)
    assert b.remaining() == 1000
    b.note_rate_limited(retry_after_seconds=120)
    assert not b.try_spend(1)


def test_a_ban_survives_a_restart(tmp_path):
    """Binance's bans escalate 2 minutes to 3 days for repeat offenders, so a
    ban a restart forgets is how a process earns the next tier."""
    b = RateBudget(tmp_path, "binance", capacity=1000, refill_per_second=0,
                   clock=lambda: 0.0)
    b.note_rate_limited(retry_after_seconds=600)

    reloaded = RateBudget(tmp_path, "binance", capacity=1000,
                          refill_per_second=0, clock=lambda: 60.0)
    assert not reloaded.try_spend(1), "a restart cleared an active ban"


def test_a_zero_or_negative_weight_is_refused(tmp_path):
    """A free request is a bug in the caller, not a discount."""
    b = RateBudget(tmp_path, "binance", capacity=100, refill_per_second=0,
                   clock=lambda: 0.0)
    with pytest.raises(ValueError):
        b.try_spend(0)
