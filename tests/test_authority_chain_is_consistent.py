"""Two documents claiming to say what to build next is the original defect.

`2026-08-09-full-build-master-plan.md` put paper trading last while paper
trading ran, and `DECISIONS.md` §13 records the identical failure about itself
six days earlier. Whatever else is true, exactly one file may claim authority
over the order of work, and the superseded one must say so in its own text -
a reader who opens it must not have to know about this test to learn it is
stale.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "docs" / "superpowers" / "plans" / "2026-08-09-full-build-master-plan.md"
NEW = REPO / "docs" / "AJIT-MASTER-PLAN.md"
GOAL = (REPO / "docs" / "superpowers" / "specs"
        / "2026-08-08-final-project-goal-design.md")


def test_the_old_master_plan_declares_itself_superseded():
    body = OLD.read_text()
    assert "SUPERSEDED" in body.upper()
    assert "AJIT-MASTER-PLAN" in body, (
        "the superseded plan must name its replacement, not merely say it is old")


def test_the_new_plan_claims_the_authority_explicitly():
    assert "top of the authority chain" in NEW.read_text().lower()


def test_the_goal_document_records_the_segment_bot_ruling_verbatim():
    body = GOAL.read_text()
    assert re.search(r"^##\s+3b\.", body, re.MULTILINE), (
        "§3b must be a real section, not a passing mention of the string '3b'")
    assert "each segment are like there own bots" in body, (
        "RL-019 must appear in the user's own words. A paraphrase reads better "
        "and is a different sentence, and the difference is where a design "
        "quietly becomes something else")


def test_the_goal_document_amends_universe_scanning_to_a_per_bot_property():
    body = GOAL.read_text()
    section = body[body.find("## 5a."):body.find("## 6.")]
    assert section, "§5a must still exist"
    assert "per-segment" in section or "per segment bot" in section, (
        "§5a called scanning a property of the SYSTEM; RL-019 makes it a "
        "property of each bot, and the amendment must be visible where the "
        "original claim is")


def test_the_goal_document_still_says_what_it_said_before():
    """An amendment adds; it must not quietly delete what was settled."""
    body = GOAL.read_text()
    for kept in ("## 0. Prime directive",
                 "## 1a. The intelligence standard",
                 "## 3a. INTRADAY, on all three segments",
                 "## 5a. Universe-wide scanning"):
        assert kept in body, f"{kept!r} disappeared from the goal document"
