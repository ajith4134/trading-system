"""Grading what a poll COSTS, not only what it found.

The engine was OOM-killed on 2026-08-17 (exit 137, fourth restart that day)
while every heartbeat it wrote looked healthy: fills, orders and events say
nothing about how long the read took, and the read was what killed it. So the
heartbeat carries the cost and `probe_poll_scan_cost` grades it.

The field that matters most here is the absent one. A heartbeat written before
these fields existed has to read NOT MEASURED, never OK - zero footers opened is
the *healthy* answer, so defaulting an absent measurement to zero would paint
the failing case green (Rule 8).
"""
from __future__ import annotations

import json

from paper.forward_journal import ForwardJournal, read_heartbeat
from statuswall.evidence import DEGRADED, NOT_MEASURED, OK
from statuswall.ruling_conformance import probe_poll_scan_cost


def _beat(root, **overrides) -> None:
    payload = dict(now_ns=1_000, strategy="plumbing-momentum",
                   makes_edge_claim=False, events_fed=3, orders_submitted=0,
                   orders_rejected=0, fills=0, open_orders=0,
                   last_event_time_ns=900, detail="watermark note")
    payload.update(overrides)
    ForwardJournal(root).record_heartbeat(**payload)


def _dataset(root, fragments: int):
    dataset = root / "bars"
    (dataset / "symbol=BTCUSDT").mkdir(parents=True)
    for index in range(fragments):
        (dataset / "symbol=BTCUSDT" / f"part-{index}.parquet").write_bytes(b"")
    return dataset


# --- the journal round-trips the cost ---------------------------------------

def test_the_heartbeat_round_trips_what_the_poll_cost(tmp_path):
    _beat(tmp_path, fragment_schema_reads=7, poll_seconds=1.25)
    beat = read_heartbeat(tmp_path)
    assert beat.fragment_schema_reads == 7
    assert beat.poll_seconds == 1.25


def test_a_heartbeat_from_before_the_fields_still_parses(tmp_path):
    """An engine mid-upgrade must not be graded dead over a cost field."""
    stale = {"written_at_ns": 1, "strategy": "s", "makes_edge_claim": False,
             "events_fed": 0, "orders_submitted": 0, "orders_rejected": 0,
             "fills": 0, "open_orders": 0, "last_event_time_ns": None,
             "detail": ""}
    (tmp_path / "heartbeat.json").write_text(json.dumps(stale))
    beat = read_heartbeat(tmp_path)
    assert beat is not None
    assert beat.fragment_schema_reads is None and beat.poll_seconds is None


# --- and the probe grades it ------------------------------------------------

def test_a_cheap_poll_reads_ok(tmp_path):
    _beat(tmp_path, fragment_schema_reads=2, poll_seconds=1.5)
    result = probe_poll_scan_cost(tmp_path, _dataset(tmp_path, 100))
    assert result.state == OK
    assert "2/100 fragment footers" in result.detail


def test_a_poll_slower_than_the_interval_reads_degraded(tmp_path):
    """The failing state has to be reachable, or the tile was never tested."""
    _beat(tmp_path, fragment_schema_reads=1, poll_seconds=93.0)
    result = probe_poll_scan_cost(tmp_path, _dataset(tmp_path, 100))
    assert result.state == DEGRADED
    assert "falling behind the tape" in result.detail


def test_cost_scaling_with_the_archive_reads_degraded(tmp_path):
    """The regression SL-14 exists to catch: the whole store re-walked."""
    _beat(tmp_path, fragment_schema_reads=100, poll_seconds=4.0)
    result = probe_poll_scan_cost(tmp_path, _dataset(tmp_path, 100))
    assert result.state == DEGRADED
    assert "scaling with the archive" in result.detail


def test_a_heartbeat_without_the_cost_fields_is_not_measured(tmp_path):
    _beat(tmp_path)
    result = probe_poll_scan_cost(tmp_path, _dataset(tmp_path, 100))
    assert result.state == NOT_MEASURED


def test_no_heartbeat_at_all_is_not_measured(tmp_path):
    assert probe_poll_scan_cost(tmp_path, _dataset(tmp_path, 1)).state == NOT_MEASURED


def test_the_probe_is_registered_under_the_name_the_plan_uses(tmp_path):
    from statuswall.ruling_conformance import PROBES
    assert PROBES["probe_poll_scan_cost"] is probe_poll_scan_cost
