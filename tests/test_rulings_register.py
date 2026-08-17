"""The rulings register is the record of what the user settled.

Its one job is to be complete and unambiguous, so the tests defend exactly
that: every row identifiable, every scope from the closed set, and a probe
field that is present even when it is null - because an absent probe field is
indistinguishable from an oversight, while an explicit null is a decision.

The refusal of a `status` field is the load-bearing one. A status typed into
this register would be the same fiction the status wall exists to prevent
(Rule 8): a claim about the running system written by whoever last edited a
document rather than measured by anything.
"""
import json
from pathlib import Path

import pytest

from plan.rulings import MalformedRuling, SCOPES, read_rulings


REGISTER = Path(__file__).resolve().parents[1] / "docs" / "rulings.json"


def _write(tmp_path: Path, rulings: list[dict]) -> Path:
    path = tmp_path / "rulings.json"
    path.write_text(json.dumps({"rulings": rulings}))
    return path


def _row(**over) -> dict:
    row = {"id": "RL-999", "date": "2026-08-17", "session": "test",
           "verbatim": "a thing the user said", "means": "what it means",
           "recorded_in": ["docs/x.md"], "probe": None, "scope": "shared"}
    row.update(over)
    return row


# --- the real register ----------------------------------------------------

def test_the_real_register_loads_and_every_ruling_is_identifiable():
    rulings = read_rulings(REGISTER)
    assert len(rulings) >= 21, "21 rulings were recovered on 2026-08-17"
    ids = [r.id for r in rulings]
    assert len(ids) == len(set(ids)), "ruling ids must be unique"
    assert all(r.verbatim.strip() for r in rulings), (
        "a ruling with no verbatim text is a paraphrase, which is how meaning drifts")


def test_every_real_ruling_carries_a_scope_from_the_closed_set():
    for ruling in read_rulings(REGISTER):
        assert ruling.scope in SCOPES, f"{ruling.id} has scope {ruling.scope!r}"


def test_a_superseded_ruling_names_the_ruling_that_replaced_it():
    by_id = {r.id: r for r in read_rulings(REGISTER)}
    superseded = [r for r in by_id.values() if r.superseded_by]
    assert superseded, "RL-015 was superseded by RL-017 on 2026-08-15"
    for ruling in superseded:
        assert ruling.superseded_by in by_id, (
            f"{ruling.id} points at {ruling.superseded_by}, which does not exist")


# --- refusals -------------------------------------------------------------

def test_a_missing_probe_field_is_refused_but_an_explicit_null_is_accepted(tmp_path):
    no_field = _row()
    del no_field["probe"]
    with pytest.raises(MalformedRuling, match="probe"):
        read_rulings(_write(tmp_path, [no_field]))

    explicit_null = read_rulings(_write(tmp_path, [_row(probe=None)]))
    assert explicit_null[0].probe is None


def test_an_unknown_scope_is_refused_by_name(tmp_path):
    with pytest.raises(MalformedRuling, match="per-venue"):
        read_rulings(_write(tmp_path, [_row(scope="per-venue")]))


def test_a_duplicate_id_is_refused(tmp_path):
    with pytest.raises(MalformedRuling, match="RL-999"):
        read_rulings(_write(tmp_path, [_row(), _row()]))


def test_a_status_field_is_refused_because_status_is_measured_not_typed(tmp_path):
    with pytest.raises(MalformedRuling, match="status"):
        read_rulings(_write(tmp_path, [_row(status="HONOURED")]))
