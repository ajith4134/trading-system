"""The spine holds decisions; it must never hold a measurement.

The whole written/generated split rests on one property: no state value is
ever typed into the plan document. If that erodes, the plan starts asserting
state again and is back to being the thing that went eight days stale -
`2026-08-09-full-build-master-plan.md`, whose phase J put paper trading last
while paper trading ran, and `DECISIONS.md` §13, which records the identical
defect about itself.

So the parser refuses a typed state by name, and refuses a row missing any of
its eight fields rather than accepting a half-specified contract: an unwritten
decision must be visible as a blank field before code is written, not as a
wrong module afterwards.
"""
from pathlib import Path

import pytest

from plan.master_plan import (
    MalformedRow,
    ROW_FIELDS,
    TypedState,
    read_master_plan,
)


def _plan(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "AJIT-MASTER-PLAN.md"
    path.write_text(body)
    return path


GOOD = """
## SLICE spot-bot — SPOT BOT

### SP-04
  slice:      spot-bot
  does:       compute spot-only microstructure features per symbol
  satisfies:  RL-006 RL-009
  sources:    FEATURES §2.4 · ledger FE-014
  depends on: SP-02
  probe:      probe_spot_features
  accepts:    every value carries a staleness stamp
  state:      measured by probe_spot_features
"""


def test_a_well_formed_row_parses_with_all_eight_fields(tmp_path):
    slices = read_master_plan(_plan(tmp_path, GOOD))
    assert [s.key for s in slices] == ["spot-bot"]
    row = slices[0].rows[0]
    assert row.id == "SP-04"
    assert row.satisfies == ("RL-006", "RL-009")
    assert row.depends_on == ("SP-02",)
    assert row.probe == "probe_spot_features"
    assert row.decided is None


def test_a_typed_state_is_refused_and_the_error_names_the_row(tmp_path):
    body = GOOD.replace("measured by probe_spot_features", "OK")
    with pytest.raises(TypedState, match="SP-04"):
        read_master_plan(_plan(tmp_path, body))


def test_every_state_value_from_the_wall_vocabulary_is_refused(tmp_path):
    for typed in ("OK", "BUILT", "PARTIAL", "DEGRADED", "FAILING", "NOT BUILT"):
        body = GOOD.replace("measured by probe_spot_features", typed)
        with pytest.raises(TypedState):
            read_master_plan(_plan(tmp_path, body))


def test_a_measured_word_hiding_behind_the_prefix_is_still_refused(tmp_path):
    """`measured by OK` is a typed state wearing the right words."""
    body = GOOD.replace("measured by probe_spot_features", "measured by BUILT")
    with pytest.raises(TypedState, match="BUILT"):
        read_master_plan(_plan(tmp_path, body))


def test_a_decided_state_is_allowed_because_a_decision_is_not_a_measurement(tmp_path):
    body = GOOD.replace("measured by probe_spot_features",
                        "DECLINED - no DEX venue is decided, so on-chain has no target")
    row = read_master_plan(_plan(tmp_path, body))[0].rows[0]
    assert row.decided == "DECLINED"
    assert row.probe == "probe_spot_features"


def test_a_row_missing_a_field_is_refused_and_names_the_field(tmp_path):
    for field in ROW_FIELDS:
        body = "\n".join(l for l in GOOD.splitlines()
                         if not l.strip().startswith(field + ":"))
        with pytest.raises(MalformedRow, match=field.split()[0]):
            read_master_plan(_plan(tmp_path, body))


def test_a_probe_of_none_is_allowed_and_parses_as_none(tmp_path):
    body = GOOD.replace("  probe:      probe_spot_features", "  probe:      none")
    body = body.replace("measured by probe_spot_features", "measured by none")
    row = read_master_plan(_plan(tmp_path, body))[0].rows[0]
    assert row.probe is None


def test_a_duplicate_row_id_is_refused(tmp_path):
    body = GOOD + GOOD.split("## SLICE spot-bot — SPOT BOT")[1]
    with pytest.raises(MalformedRow, match="SP-04"):
        read_master_plan(_plan(tmp_path, body))


def test_a_wrapped_accepts_line_is_joined_rather_than_truncated(tmp_path):
    body = GOOD.replace(
        "  accepts:    every value carries a staleness stamp",
        "  accepts:    every value carries a staleness stamp and no value is\n"
        "              readable before its availability time")
    row = read_master_plan(_plan(tmp_path, body))[0].rows[0]
    assert "before its availability time" in row.accepts


def test_a_slice_with_no_rows_parses_as_an_empty_slice_not_a_missing_one(tmp_path):
    """An empty slice is the honest record of work that is not planned yet."""
    body = GOOD + "\n## SLICE options-bot — OPTIONS BOT\n\n*inventory pending*\n"
    slices = read_master_plan(_plan(tmp_path, body))
    assert [s.key for s in slices] == ["spot-bot", "options-bot"]
    assert slices[1].rows == ()
