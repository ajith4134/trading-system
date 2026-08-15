"""Outage detection — the gap this box spent five days demonstrating.

2026-08-10 12:53 to 2026-08-15 17:13: the VM was off, capture wrote nothing,
`capture/raw/binance/` jumps straight from 08-10 to 08-15, and every board went
on serving a file dated 08-10 without one word saying so. Nothing was broken in
a way anything checked. The system had no concept of its own absence.

The awkward truth this module is built around: **a watcher running on the box
cannot report that the box is off.** So it does the one thing a process on the
machine honestly can — it records that it was alive, and when it next runs it
compares that stamp against the clock and reports the hole. Retrospective, and
that is not a weakness to hide: an outage that ended is still the fact that
explains five missing days of tape.

Two states that must never be confused:

**A cold start is not an outage.** The first run on a fresh machine has no prior
stamp, and calling that a gap of "since the epoch" would put a fictional outage
at the top of the ledger on day one.

**A bounded gap, not a guess.** All that is known is that nothing wrote between
the last stamp and now. The ledger records exactly that and never rounds it into
a claim about when the box actually died.
"""
import json

import pytest

from ops.liveness_ledger import (
    ColdStart, Outage, last_seen_ns, read_outages, record_liveness,
)

MINUTE = 60 * 1_000_000_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR


# --- the first run is not an outage -----------------------------------------

def test_a_fresh_machine_has_never_been_seen(tmp_path):
    assert last_seen_ns(tmp_path) is None


def test_the_first_run_reports_a_cold_start_not_a_gap(tmp_path):
    result = record_liveness(tmp_path, now_ns=1_000 * DAY)
    assert isinstance(result, ColdStart)
    assert read_outages(tmp_path) == []


def test_a_cold_start_still_records_that_we_are_alive(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    assert last_seen_ns(tmp_path) == 1_000 * DAY


# --- ordinary ticks are silent ----------------------------------------------

def test_a_tick_inside_the_threshold_is_not_an_outage(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    result = record_liveness(tmp_path, now_ns=1_000 * DAY + MINUTE)
    assert result is None
    assert read_outages(tmp_path) == []


def test_ordinary_ticks_keep_moving_the_stamp_forward(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    record_liveness(tmp_path, now_ns=1_000 * DAY + MINUTE)
    assert last_seen_ns(tmp_path) == 1_000 * DAY + MINUTE


# --- the gap ----------------------------------------------------------------

def test_a_gap_past_the_threshold_is_recorded_as_an_outage(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    result = record_liveness(tmp_path, now_ns=1_000 * DAY + 5 * DAY)
    assert isinstance(result, Outage)
    assert result.duration_ns == 5 * DAY


def test_the_outage_is_bounded_by_what_was_actually_observed(tmp_path):
    """All that is known is that nothing wrote between the last stamp and now.
    The ledger says exactly that and never guesses when the box really died."""
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    outage = record_liveness(tmp_path, now_ns=1_000 * DAY + 5 * DAY)
    assert outage.last_seen_ns == 1_000 * DAY
    assert outage.returned_ns == 1_000 * DAY + 5 * DAY


def test_an_outage_is_appended_to_the_ledger_and_survives(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    record_liveness(tmp_path, now_ns=1_000 * DAY + 5 * DAY)
    (row,) = read_outages(tmp_path)
    assert row.duration_ns == 5 * DAY


def test_every_outage_is_kept_not_only_the_latest(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    record_liveness(tmp_path, now_ns=1_002 * DAY)
    record_liveness(tmp_path, now_ns=1_002 * DAY + MINUTE)
    record_liveness(tmp_path, now_ns=1_009 * DAY)
    # The second gap is 7 days LESS the minute the third stamp bought, because
    # the bound is measured from the last stamp that actually exists rather than
    # rounded to the tick that was expected.
    assert [o.duration_ns for o in read_outages(tmp_path)] == [
        2 * DAY, 7 * DAY - MINUTE]


def test_the_threshold_is_a_parameter_not_a_buried_constant(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY, threshold_ns=10 * MINUTE)
    result = record_liveness(tmp_path, now_ns=1_000 * DAY + 11 * MINUTE,
                             threshold_ns=10 * MINUTE)
    assert isinstance(result, Outage)


# --- refusals over quiet defaults -------------------------------------------

def test_a_clock_that_went_backwards_is_refused_rather_than_recorded(tmp_path):
    """A negative outage is not a shorter outage. NTP stepping the clock
    backwards would otherwise write a row claiming the future came first, and
    every duration computed from this ledger afterwards would be suspect."""
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    with pytest.raises(ValueError, match="backwards"):
        record_liveness(tmp_path, now_ns=999 * DAY)


def test_an_unreadable_stamp_is_treated_as_a_cold_start_not_as_alive(tmp_path):
    """Wrong in the safe direction: an unreadable stamp read as 'seen just now'
    would erase the very outage that corrupted it."""
    (tmp_path / "liveness.json").write_text("{broken", encoding="utf-8")
    assert last_seen_ns(tmp_path) is None
    assert isinstance(record_liveness(tmp_path, now_ns=1_000 * DAY), ColdStart)


def test_a_torn_ledger_line_does_not_hide_the_outages_around_it(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    record_liveness(tmp_path, now_ns=1_005 * DAY)
    path = tmp_path / "outages.ndjson"
    path.write_text(path.read_text(encoding="utf-8") + "{torn\n",
                    encoding="utf-8")
    assert len(read_outages(tmp_path)) == 1


def test_the_stamp_is_written_whole_so_a_reader_never_sees_half_of_it(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    payload = json.loads((tmp_path / "liveness.json").read_text(encoding="utf-8"))
    assert payload["last_seen_ns"] == 1_000 * DAY


# --- a reconstruction is not an observation ---------------------------------

def test_an_observed_outage_is_not_marked_reconstructed(tmp_path):
    record_liveness(tmp_path, now_ns=1_000 * DAY)
    record_liveness(tmp_path, now_ns=1_005 * DAY)
    (row,) = read_outages(tmp_path)
    assert row.is_reconstructed is False


def test_a_reconstructed_outage_says_so_on_the_row(tmp_path):
    """Follows store.bar_backfill, which keeps reconstructed bars beside the
    observed ones and never inside them. A ledger that cannot tell them apart
    quietly upgrades inference into measurement."""
    from ops.liveness_ledger import record_reconstructed_outage
    record_reconstructed_outage(tmp_path, 1_000 * DAY, 1_005 * DAY,
                                evidence="boot.log and raw archive dates")
    (row,) = read_outages(tmp_path)
    assert row.is_reconstructed is True
    assert row.duration_ns == 5 * DAY


def test_a_reconstructed_outage_without_evidence_is_refused(tmp_path):
    from ops.liveness_ledger import record_reconstructed_outage
    with pytest.raises(ValueError, match="evidence"):
        record_reconstructed_outage(tmp_path, 1_000 * DAY, 1_005 * DAY,
                                    evidence="   ")


def test_a_reconstructed_outage_that_ends_before_it_starts_is_refused(tmp_path):
    from ops.liveness_ledger import record_reconstructed_outage
    with pytest.raises(ValueError, match="end after"):
        record_reconstructed_outage(tmp_path, 1_005 * DAY, 1_000 * DAY,
                                    evidence="x")


def test_the_evidence_survives_a_round_trip_to_disk(tmp_path):
    from ops.liveness_ledger import record_reconstructed_outage
    record_reconstructed_outage(tmp_path, 1_000 * DAY, 1_005 * DAY,
                                evidence="boot.log 2026-08-15T17:13:47Z")
    (row,) = read_outages(tmp_path)
    assert "2026-08-15T17:13:47Z" in row.evidence
