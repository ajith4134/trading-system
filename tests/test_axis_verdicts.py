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


def test_every_module_in_this_repo_is_judged():
    """§1a.6 admits no partial answer: NO module ships without a verdict.
    Completed 2026-08-09. A module added later without one drops this below
    100% and fails here, which is the ratchet."""
    coverage = assess(REPO_ROOT)

    assert coverage.total > 50, "denominator collapsed; the tile would go green on nothing"
    assert coverage.unverdicted == [], coverage.unverdicted
    assert coverage.share == 1.0


def test_the_record_admits_failures():
    """A standard that produces only passes is a standard nobody fails.

    Two kinds of failure are admitted, and they say different things:

    **DEPTH** - the module cannot change what the system does when it is wrong,
    because nothing calls it or no test exercises it. That is the master test.

    **LEARNING** - added 2026-08-18 with the four segment bots. Their BULL, BEAR
    and PROFIT-TAIL brains are RULE brains under RL-025: stated thresholds, no
    adaptation, the same market twice producing the same decision forever. They
    are recorded as FAILING the learning axis rather than as `n/a`, because `n/a`
    is the honest answer for a Parquet writer and a self-serving one for something
    calling itself a brain. §1a is the standard they are measured against and they
    do not meet it yet; the tile says RULE BRAIN, every fill carries
    `makes_edge_claim: false`, and this record is where that is admitted rather
    than argued about.

    The point of the exercise is that these entries exist. A verdict file where
    every brain passed learning would be the one thing §1a exists to prevent.
    """
    coverage = assess(REPO_ROOT)
    assert coverage.failing, "every verdict passed, which is not a judgement"
    allowed = {"depth", "learning"}
    for module, axes in coverage.failing.items():
        assert set(axes) <= allowed, f"{module} fails an unexpected axis: {axes}"
    learning_failures = {m for m, axes in coverage.failing.items() if "learning" in axes}
    assert learning_failures, (
        "no module admits a learning failure; the rule brains must not quietly "
        "become n/a on the axis they actually fail")
    # Every learning failure is a brain. If something else starts failing learning,
    # that is a different claim and belongs in its own verdict with its own reason.
    assert all("brains" in m or "profit_tail" in m or "arbiter" in m or "features" in m
               for m in learning_failures), sorted(learning_failures)


def test_the_depth_failures_are_exactly_the_unreachable_modules_plus_the_untested_one():
    """The two checks agree, and neither is allowed to drift from the other: a
    module the reachability audit calls unreachable cannot honestly pass a depth
    test whose question is whether it changes behaviour under error."""
    from integrity.unsupported_claims import (
        UNREACHABLE, classify_modules, invoked_modules, read_source_tree)

    verdict = classify_modules(read_source_tree(REPO_ROOT / "src"),
                               invoked_modules(REPO_ROOT))
    unreachable = {m for m, state in verdict.items() if state == UNREACHABLE}
    failing = set(assess(REPO_ROOT).failing)

    assert unreachable <= failing, (
        f"unreachable but not failing depth: {sorted(unreachable - failing)}")
