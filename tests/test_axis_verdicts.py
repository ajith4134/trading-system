"""§1a.6: no module ships without a learning / reasoning / depth verdict.

    A module may legitimately be *not applicable* on an axis - a Parquet writer
    does not reason - but that must be stated, not left blank.

    The status wall carries the verdicts. A module claimed intelligent with no
    passing test renders as `NOT MEASURED`, never as green.

The verdict is a judgement and this module does not pretend otherwise. What is
tested here is the part that can be: coverage enumerated from the code, and
named evidence that actually exists.
"""
import json
from pathlib import Path

import pytest

from integrity.axis_verdicts import (
    AXES, FAIL, NOT_APPLICABLE, PASS, VERDICT_FILE, assess, read_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo(tmp_path, modules: dict, verdicts: dict, evidence=()):
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    for name, body in modules.items():
        path = tmp_path / "src" / f"{name.replace('.', '/')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    for rel in evidence:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    out = tmp_path / VERDICT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"modules": verdicts}), encoding="utf-8")
    return tmp_path


def _full(**over):
    base = {a: f"{PASS}: because" for a in AXES}
    base.update(over)
    return base


# --------------------------------------------------------------------------
# coverage is counted from the code, not from the record
# --------------------------------------------------------------------------

def test_a_module_with_no_verdict_lowers_the_number(tmp_path):
    """Enumerated from `src/`. Reading the list out of the verdict file would
    make every omission invisible, which is the failure this exists to stop."""
    root = _repo(tmp_path, {"judged": "x = 1\n", "unjudged": "y = 2\n"},
                 {"judged": _full()})

    coverage = assess(root)

    assert coverage.verdicted == ["judged"]
    assert coverage.unverdicted == ["unjudged"]
    assert coverage.total == 2
    assert coverage.share == 0.5


def test_an_axis_left_blank_is_incomplete_not_absent(tmp_path):
    """"Not applicable" has to be SAID. The difference between deciding and
    forgetting is the whole point of the clause."""
    root = _repo(tmp_path, {"m": "x = 1\n"},
                 {"m": {"learning": f"{PASS}: fitted", "depth": f"{PASS}: yes"}})

    coverage = assess(root)

    assert coverage.incomplete == {"m": ["reasoning"]}


def test_not_applicable_is_a_real_verdict(tmp_path):
    """A Parquet writer does not reason, and should say so once rather than be
    counted as an omission forever."""
    root = _repo(tmp_path, {"m": "x = 1\n"},
                 {"m": _full(reasoning=f"{NOT_APPLICABLE}: writes bytes")})

    coverage = assess(root)

    assert coverage.verdicted == ["m"]
    assert coverage.incomplete == {}


def test_a_word_that_is_not_a_verdict_is_rejected(tmp_path):
    """"probably" is not pass, fail or n/a."""
    root = _repo(tmp_path, {"m": "x = 1\n"}, {"m": _full(depth="probably: hmm")})

    assert assess(root).invalid == {"m": ["depth"]}


def test_an_explicit_fail_is_reported_rather_than_hidden(tmp_path):
    """A fail is a legitimate verdict and must survive being recorded - a
    standard that only admits passes is a standard nobody fails."""
    root = _repo(tmp_path, {"m": "x = 1\n"},
                 {"m": _full(learning=f"{FAIL}: constants are typed")})

    coverage = assess(root)

    assert coverage.failing == {"m": ["learning"]}
    assert coverage.verdicted == ["m"], "a failing module is still judged"


# --------------------------------------------------------------------------
# the evidence has to exist
# --------------------------------------------------------------------------

def test_evidence_that_is_not_on_disk_is_reported(tmp_path):
    """A verdict citing a test nobody wrote is the same defect as a ledger row
    citing a module nobody calls. Caught for real on 2026-08-09: this module's
    own verdict named `tests/test_axis_verdicts.py` before it existed."""
    root = _repo(tmp_path, {"m": "x = 1\n"},
                 {"m": {**_full(), "evidence": ["tests/test_absent.py"]}})

    assert assess(root).missing_evidence == {"m": ["tests/test_absent.py"]}


def test_evidence_that_exists_passes(tmp_path):
    root = _repo(tmp_path, {"m": "x = 1\n"},
                 {"m": {**_full(), "evidence": ["tests/test_present.py"]}},
                 evidence=["tests/test_present.py"])

    assert assess(root).missing_evidence == {}


def test_a_verdict_for_a_module_that_no_longer_exists_is_orphaned(tmp_path):
    """A record of a judgement about deleted code reads as coverage it is not."""
    root = _repo(tmp_path, {"m": "x = 1\n"}, {"m": _full(), "deleted": _full()})

    assert assess(root).orphaned == ["deleted"]


def test_no_verdict_file_is_no_verdicts_rather_than_an_error(tmp_path):
    """The honest starting state of a repo that has never reviewed itself."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n", encoding="utf-8")

    coverage = assess(tmp_path)
    assert coverage.verdicted == []
    assert coverage.unverdicted == ["m"]
    assert read_verdicts(tmp_path / "nope.json") == {}


# --------------------------------------------------------------------------
# against the real repo
# --------------------------------------------------------------------------

def test_every_declared_verdict_in_this_repo_is_well_formed():
    """The ratchet for the record itself: incomplete axes, non-verdict words,
    evidence nobody wrote, and judgements about deleted modules all fail here.
    Unverdicted modules deliberately do NOT - the standard has to be adoptable
    module by module rather than all at once."""
    coverage = assess(REPO_ROOT)

    assert coverage.incomplete == {}, coverage.incomplete
    assert coverage.invalid == {}, coverage.invalid
    assert coverage.missing_evidence == {}, coverage.missing_evidence
    assert coverage.orphaned == [], coverage.orphaned


def test_the_repo_carries_at_least_the_verdicts_it_claims():
    """Guards against the file being emptied and the tile going quietly green
    on a denominator of zero."""
    coverage = assess(REPO_ROOT)
    assert coverage.total > 50
    assert len(coverage.verdicted) >= 9
