"""The Holdout Custodian — VX-003. It refuses queries, and that is its whole job.

`FEATURES.md` §8: *"Owns the untouched holdout; commit hash + not-consumed flag,
CI blocks any run touching the range before freeze."*

The closest prior art (ledger VX-049, `early-repos` row 63) is the capstone
protocol: search and select on history only, **freeze**, forward-test on unseen
holdout, and report **only the frozen score**. The ledger's assessment is that the
freeze-then-report-once pattern is exactly right, and it is what this implements.

The failure it prevents cannot be caught by a statistic. DSR, PBO and MinBTL all
correct for the number of times you looked at the data - none of them can tell
that you looked at the *holdout*. A holdout that has been peeked at is simply no
longer a holdout, and nothing downstream can detect the difference. The only
defence is a component that refuses the read.

Hence: sealed by default, one consumption ever, the code identity pinned at freeze
so the thing tested is the thing that was frozen, and all of it persisted, because
a seal a process restart clears is a seal that lifts the moment someone is
frustrated.
"""
import pytest

from validation.holdout_custodian import (
    HoldoutAlreadyConsumed,
    HoldoutCustodian,
    HoldoutSealed,
    HoldoutTampered,
)

START = 1_700_000_000_000_000_000
END = 1_800_000_000_000_000_000


def custodian(root, **kw):
    return HoldoutCustodian(root, holdout_start_ns=START, holdout_end_ns=END, **kw)


# --- sealed by default ------------------------------------------------------

def test_a_fresh_custodian_is_sealed(tmp_path):
    assert not custodian(tmp_path).is_frozen()
    assert not custodian(tmp_path).is_consumed()


def test_reading_inside_the_holdout_before_freeze_is_refused(tmp_path):
    """The core refusal. Not a warning, not a log line - an exception, because a
    peek that returns data has already destroyed the holdout."""
    with pytest.raises(HoldoutSealed):
        custodian(tmp_path).assert_readable(START + 1_000)


def test_reading_outside_the_holdout_is_always_allowed(tmp_path):
    """Research needs the rest of history freely, or the custodian becomes an
    obstacle people route around - and a defence that gets routed around is worse
    than none, because it still reads as present."""
    cust = custodian(tmp_path)
    cust.assert_readable(START - 1)
    cust.assert_readable(END)


def test_the_holdout_boundary_is_half_open(tmp_path):
    """[start, end). The end instant belongs to the readable side, so a range
    stated as 'up to T' and another starting at T do not both claim T."""
    cust = custodian(tmp_path)
    with pytest.raises(HoldoutSealed):
        cust.assert_readable(END - 1)
    cust.assert_readable(END)


def test_the_refusal_names_the_range_and_what_to_do(tmp_path):
    """A refusal nobody can act on gets disabled. It has to say why and how."""
    with pytest.raises(HoldoutSealed) as caught:
        custodian(tmp_path).assert_readable(START)
    message = str(caught.value)
    assert str(START) in message and "freeze" in message.lower()


# --- freeze -----------------------------------------------------------------

def test_freezing_records_the_commit_hash(tmp_path):
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="momentum-v3 selected on history")
    assert cust.is_frozen()
    assert cust.frozen_commit() == "abc1234"


def test_the_freeze_survives_a_restart(tmp_path):
    custodian(tmp_path).freeze(commit_hash="abc1234", reason="r")
    assert custodian(tmp_path).is_frozen()


def test_freezing_twice_is_refused(tmp_path):
    """A second freeze is a re-freeze after seeing something, which is exactly the
    manoeuvre the protocol exists to stop."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    with pytest.raises(HoldoutTampered):
        cust.freeze(commit_hash="def5678", reason="r2")


def test_freezing_does_not_by_itself_open_the_holdout(tmp_path):
    """Freeze declares the code final; consumption is the separate, counted act.
    Collapsing them would make an accidental read indistinguishable from the one
    intentional test."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    with pytest.raises(HoldoutSealed):
        cust.assert_readable(START)


# --- consumption: exactly once ----------------------------------------------

def test_consuming_after_freeze_opens_the_holdout_once(tmp_path):
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    cust.consume(commit_hash="abc1234", purpose="frozen forward test")
    cust.assert_readable(START + 5)


def test_consuming_before_freeze_is_refused(tmp_path):
    with pytest.raises(HoldoutSealed):
        custodian(tmp_path).consume(commit_hash="abc1234", purpose="p")


def test_consuming_twice_is_refused(tmp_path):
    """*"Only the frozen score is reported."* A second look is a second trial on
    data whose entire value was being untouched - and it would not appear in any
    trial count, so no statistic downstream could correct for it."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    cust.consume(commit_hash="abc1234", purpose="p")
    with pytest.raises(HoldoutAlreadyConsumed):
        cust.consume(commit_hash="abc1234", purpose="again")


def test_consuming_with_a_different_commit_than_was_frozen_is_refused(tmp_path):
    """The point of pinning the hash: without it, freeze then edit then test is a
    complete bypass, and the report still says the score came from frozen code."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    with pytest.raises(HoldoutTampered) as caught:
        cust.consume(commit_hash="def5678", purpose="p")
    assert "abc1234" in str(caught.value) and "def5678" in str(caught.value)


def test_the_consumption_survives_a_restart(tmp_path):
    """Otherwise the holdout is re-openable by restarting the process, which is
    the least deliberate act imaginable."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="r")
    cust.consume(commit_hash="abc1234", purpose="p")

    reloaded = custodian(tmp_path)
    assert reloaded.is_consumed()
    with pytest.raises(HoldoutAlreadyConsumed):
        reloaded.consume(commit_hash="abc1234", purpose="p")


# --- the record -------------------------------------------------------------

def test_every_seal_event_is_recorded_with_its_reason(tmp_path):
    """An audit trail, because 'the holdout was already consumed' is a claim
    someone will dispute months later when a promotion is on the line."""
    cust = custodian(tmp_path)
    cust.freeze(commit_hash="abc1234", reason="momentum-v3 selected")
    cust.consume(commit_hash="abc1234", purpose="frozen forward test")

    events = cust.history()
    assert [e["event"] for e in events] == ["freeze", "consume"]
    assert events[0]["reason"] == "momentum-v3 selected"
    assert events[1]["purpose"] == "frozen forward test"
    assert all(e["at_ns"] > 0 for e in events)


def test_refused_attempts_are_recorded_too(tmp_path):
    """A refusal that leaves no trace hides the most interesting fact available:
    that somebody's pipeline is repeatedly trying to read the holdout."""
    cust = custodian(tmp_path)
    with pytest.raises(HoldoutSealed):
        cust.assert_readable(START)
    assert [e["event"] for e in cust.history()] == ["refused"]


def test_an_inverted_holdout_range_is_refused(tmp_path):
    with pytest.raises(ValueError):
        HoldoutCustodian(tmp_path, holdout_start_ns=END, holdout_end_ns=START)


# --- wired into the one data path -------------------------------------------

def test_the_clock_gated_reader_refuses_a_sealed_holdout(tmp_path):
    """The custodian is worthless unwired. `CLAUDE.md` names this codebase's
    recurring trap - `tail_specs()` built, tested, called by nothing - so the
    refusal is tested through the door every market-data read actually uses."""
    from store.clock_gated_reader import ClockGatedReader

    reader = ClockGatedReader(tmp_path / "store", "bars",
                              custodian=custodian(tmp_path))
    with pytest.raises(HoldoutSealed):
        reader.read_as_of(START + 1_000)


def test_the_clock_gated_reader_is_unaffected_outside_the_holdout(tmp_path):
    from store.clock_gated_reader import ClockGatedReader

    reader = ClockGatedReader(tmp_path / "store", "bars",
                              custodian=custodian(tmp_path))
    assert reader.read_as_of(START - 1).empty      # no data, but no refusal


def test_a_reader_with_no_custodian_still_works(tmp_path):
    """Existing callers must not break. The custodian is opt-in per reader, and
    that is a stated limitation rather than a claim of full coverage - see the
    module docstring."""
    from store.clock_gated_reader import ClockGatedReader

    assert ClockGatedReader(tmp_path / "store", "bars").read_as_of(START).empty
