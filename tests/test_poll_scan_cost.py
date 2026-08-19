"""Grading what a poll COSTS, not only what it found.

The engine was OOM-killed on 2026-08-17 (exit 137, fourth restart that day)
while every heartbeat it wrote looked healthy: fills, orders and events say
nothing about how long the read took, and the read was what killed it. So the
heartbeat carries the cost and `probe_poll_scan_cost` grades it.

The field that matters most here is the absent one. A heartbeat written before
these fields existed has to read NOT MEASURED, never OK - zero footers opened is
the *healthy* answer, so defaulting an absent measurement to zero would paint
the failing case green (Rule 8).

**Repointed 2026-08-19.** The engine those first two tests describe -
`plumbing-momentum` - was RETIRED under RL-025, and nothing polls the store on a
loop any more: RL-024 moved every trading price to a live feed. The journal still
round-trips its cost fields and those tests still defend that. What the PROBE
grades changed with its subject: the property SL-14 and SL-15 defend is that a
read costs what arrived rather than what is archived, and the thing that now
decides that is the hour partitioning itself. So the probe grades the layout, and
the failing state is still reachable on purpose - a tile with no way to render
red has not been tested against a real failure.
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


# --- and the probe grades the layout that makes a read cheap ----------------

def _hour_partitioned(root, hours: int, per_hour: int):
    """A store partitioned by availability hour, as SL-15 built it."""
    dataset = root / "bars"
    for hour in range(hours):
        directory = dataset / f"availability_hour=2026-08-19T{hour:02d}" / "symbol=BTCUSDT"
        directory.mkdir(parents=True)
        for index in range(per_hour):
            (directory / f"part-{index}.parquet").write_bytes(b"")
    return dataset


def test_a_store_whose_recent_hours_are_a_small_share_of_it_reads_ok(tmp_path):
    dataset = _hour_partitioned(tmp_path, hours=24, per_hour=3)

    result = probe_poll_scan_cost(dataset=dataset, learn_root=tmp_path / "learn")

    assert result.state == OK
    assert "24 hour partitions" in result.detail
    assert "6/72" in result.detail


def test_recent_hours_holding_most_of_the_archive_reads_degraded(tmp_path):
    """The regression SL-14 and SL-15 exist to catch, in its layout form: if a
    two-hour read still opens most of the store, bounding the read bought
    nothing."""
    dataset = tmp_path / "bars"
    # Twenty-two quiet hours and two that hold almost the whole archive: the
    # shape a compaction into recent partitions would produce, and the one that
    # makes "read only the newest hours" stop meaning anything.
    for hour in range(24):
        directory = dataset / f"availability_hour=2026-08-19T{hour:02d}" / "symbol=BTCUSDT"
        directory.mkdir(parents=True)
        for index in range(200 if hour >= 22 else 1):
            (directory / f"part-{index}.parquet").write_bytes(b"")

    result = probe_poll_scan_cost(dataset=dataset, learn_root=tmp_path / "learn")

    assert result.state == DEGRADED
    assert "not staying bounded" in result.detail


def test_a_store_that_is_not_hour_partitioned_reads_degraded(tmp_path):
    """Every read walks the whole archive, which is the state SL-15 replaced."""
    result = probe_poll_scan_cost(dataset=_dataset(tmp_path, 5),
                                  learn_root=tmp_path / "learn")

    assert result.state == DEGRADED
    assert "not partitioned by availability hour" in result.detail


def test_no_dataset_at_all_is_not_measured(tmp_path):
    result = probe_poll_scan_cost(dataset=tmp_path / "absent",
                                  learn_root=tmp_path / "learn")

    assert result.state == NOT_MEASURED


def test_the_retrainer_read_is_reported_beside_the_layout(tmp_path):
    """What still reads the store on a cadence is the retrainer, so what it paid
    on its last pass belongs on the same tile as the layout that priced it."""
    learn_root = tmp_path / "learn"
    learn_root.mkdir()
    (learn_root / "perp-retrain.json").write_text(json.dumps(
        {"segment": "perp", "read": {"hours_read": 30, "rows": 969505}}))

    result = probe_poll_scan_cost(dataset=_hour_partitioned(tmp_path, 24, 3),
                                  learn_root=learn_root)

    assert "perp read 30h -> 969,505 rows" in result.detail


def test_the_probe_is_registered_under_the_name_the_plan_uses(tmp_path):
    from statuswall.ruling_conformance import PROBES
    assert PROBES["probe_poll_scan_cost"] is probe_poll_scan_cost
