"""A claim of BUILT has to survive the question "what calls it?".

Four defects in this project have had the same shape, all failing in the
flattering direction: `tail_specs()`, ledger row DM-066's dollar-quote filter,
the status wall's "auto-halt armed", and five circuit breakers living only in
docstrings. Every existing guard points at code - and a function nothing calls
passes its tests forever. This one points at the claims.

The tests below are mostly about the check not crying wolf. A reachability
report with false alarms in it gets switched off, and then it protects nothing;
its first version called nine hand-run modules dead, `round_trip_cost` among
them, which the paper demo quotes every cost through.
"""
from pathlib import Path

import pytest

from integrity.unsupported_claims import (
    BY_HAND, RUNNING, UNREACHABLE, audit_claims, classify_modules,
    invoked_modules, read_ledger_claims, read_source_tree, reachable_from,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path.home() / "research" / "ledger" / "merged"
BASELINE = Path(__file__).parent / "data" / "unsupported_claims_baseline.txt"


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _tiny_repo(tmp_path: Path, modules: dict[str, str], script: str = "") -> Path:
    for name, body in modules.items():
        _write(tmp_path, f"src/{name.replace('.', '/')}.py", body)
    _write(tmp_path, "scripts/run.sh", script or "#!/bin/bash\n")
    return tmp_path


# --------------------------------------------------------------------------
# reachability
# --------------------------------------------------------------------------

def test_a_module_a_script_runs_is_running(tmp_path):
    _tiny_repo(tmp_path, {"alive": "x = 1\n"}, "python -m alive --flag\n")
    modules = read_source_tree(tmp_path / "src")

    verdict = classify_modules(modules, invoked_modules(tmp_path))

    assert verdict["alive"] == RUNNING


def test_a_module_nothing_imports_or_runs_is_unreachable(tmp_path):
    _tiny_repo(tmp_path, {"alive": "x = 1\n", "orphan": "def f():\n    return 1\n"},
               "python -m alive\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["orphan"] == UNREACHABLE


def test_reachability_is_transitive(tmp_path):
    """Two hops, because a helper of a helper is still live code."""
    _tiny_repo(tmp_path, {
        "alive": "import middle\n",
        "middle": "import leaf\n",
        "leaf": "def f():\n    return 1\n",
    }, "python -m alive\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["leaf"] == RUNNING


def test_a_dead_subsystem_does_not_vouch_for_itself(tmp_path):
    """The `validation/` shape exactly: modules importing each other, and nothing
    importing the group. Counting an importer without asking whether the importer
    is itself reachable would call all of this alive."""
    _tiny_repo(tmp_path, {
        "alive": "x = 1\n",
        "gate": "import scorer\nimport registry\n",
        "scorer": "def score():\n    return 1\n",
        "registry": "def record():\n    return 1\n",
    }, "python -m alive\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["gate"] == UNREACHABLE
    assert verdict["scorer"] == UNREACHABLE
    assert verdict["registry"] == UNREACHABLE


def test_a_hand_run_cli_and_what_it_uses_are_not_called_dead(tmp_path):
    """The false alarm that would have sunk this. `cost.round_trip_cost` is
    reached only by the paper demo and the cost CLI - both things a person types.
    Built and exercised, just not in the running system."""
    _tiny_repo(tmp_path, {
        "alive": "x = 1\n",
        "demo": 'import pricing\n\nif __name__ == "__main__":\n    pricing\n',
        "pricing": "def quote():\n    return 1\n",
    }, "python -m alive\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["demo"] == BY_HAND
    assert verdict["pricing"] == BY_HAND, "called a hand-run dependency dead"


def test_running_beats_by_hand_when_a_module_is_both(tmp_path):
    """A CLI a script also runs is running. Reporting the weaker answer would
    understate what is live."""
    _tiny_repo(tmp_path, {"tool": 'if __name__ == "__main__":\n    pass\n'},
               "python -m tool\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["tool"] == RUNNING


def test_an_empty_package_init_is_not_reported_at_all(tmp_path):
    """Namespace, not code. Listing `paper` beside `paper.fill_model` says the
    same thing twice and makes the dead list look worse than it is."""
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/orphan.py", "def f():\n    return 1\n")
    _write(tmp_path, "scripts/run.sh", "#!/bin/bash\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert "pkg" not in verdict
    assert verdict["pkg.orphan"] == UNREACHABLE


def test_a_docstring_mentioning_a_module_is_not_an_import(tmp_path):
    """The precise failure this exists to catch. `universe_tracker`'s only
    mentions of `store.quote_currency` are two docstrings explaining what it
    would do, and a text search reads those as usage."""
    _tiny_repo(tmp_path, {
        "alive": '"""See `orphan.f` for the filter."""\nx = 1\n',
        "orphan": "def f():\n    return 1\n",
    }, "python -m alive\n")
    verdict = classify_modules(read_source_tree(tmp_path / "src"),
                               invoked_modules(tmp_path))

    assert verdict["orphan"] == UNREACHABLE


def test_an_unparseable_module_raises_rather_than_being_skipped(tmp_path):
    """A module whose imports cannot be read is a module whose edges are
    invisible, and an invisible edge is what makes live code look dead."""
    _write(tmp_path, "src/broken.py", "def (:\n")
    with pytest.raises(SyntaxError):
        read_source_tree(tmp_path / "src")


# --------------------------------------------------------------------------
# the ledger cross-reference
# --------------------------------------------------------------------------

def _ledger(tmp_path: Path, status: str, evidence: str) -> Path:
    _write(tmp_path, "ledger/slice.md",
           "| # | Requirement | Category | Status | Phase | Evidence / satisfying module | Sources | Notes |\n"
           "|---|---|---|---|---|---|---|---|\n"
           f"| XX-001 | A thing | area | {status} | P0 | {evidence} | src | n |\n")
    return tmp_path / "ledger"


def test_a_built_row_naming_a_dead_module_is_unsupported(tmp_path):
    _tiny_repo(tmp_path, {"alive": "x = 1\n", "orphan": "def f():\n    return 1\n"},
               "python -m alive\n")
    ledger = _ledger(tmp_path, "BUILT", "`src/orphan.py`; tests/test_orphan.py")

    audits = audit_claims(tmp_path, ledger)

    assert [a.row_id for a in audits if a.is_unsupported] == ["XX-001"]
    assert audits[0].dead_modules == ["orphan"]


def test_a_prior_art_row_is_not_contradicted_by_this_repo(tmp_path):
    """PRIOR-ART points at another repository and PLANNED at nothing yet. Neither
    claims working code here, so neither can be refuted by reachability here."""
    _tiny_repo(tmp_path, {"alive": "x = 1\n", "orphan": "def f():\n    return 1\n"},
               "python -m alive\n")

    for status in ("PRIOR-ART", "PLANNED", "DECLINED", "UNRESOLVED"):
        ledger = _ledger(tmp_path, status, "`src/orphan.py`")
        assert audit_claims(tmp_path, ledger) == [], status


def test_a_built_row_whose_module_is_live_says_nothing(tmp_path):
    _tiny_repo(tmp_path, {"alive": "x = 1\n"}, "python -m alive\n")
    ledger = _ledger(tmp_path, "BUILT", "`src/alive.py`")

    assert audit_claims(tmp_path, ledger) == []


def test_a_qualified_status_still_counts_as_a_claim(tmp_path):
    """DM-066 now reads "BUILT (classification only) / BUILT-NOT-APPLIED". A
    prefix match keeps a hedged status inside the check rather than letting
    wording be the way out of it."""
    _tiny_repo(tmp_path, {"alive": "x = 1\n", "orphan": "def f():\n    return 1\n"},
               "python -m alive\n")
    ledger = _ledger(tmp_path, "**BUILT (classification only)**", "`src/orphan.py`")

    assert [a.row_id for a in audit_claims(tmp_path, ledger)] == ["XX-001"]


# --------------------------------------------------------------------------
# the ratchet, against the real repo and the real ledger
# --------------------------------------------------------------------------

def _baselined_ids() -> set[str]:
    return {line.strip() for line in BASELINE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def test_no_ledger_row_starts_claiming_code_nothing_can_reach():
    """The ratchet. Fails in BOTH directions on purpose: a new dead claim fails
    the suite, and so does fixing one without deleting its baseline line. A list
    that only ever grows is how a known-issues file becomes wallpaper.
    """
    if not LEDGER.is_dir():
        pytest.skip("ledger not present on this machine")

    unsupported = {a.row_id for a in audit_claims(REPO_ROOT, LEDGER) if a.is_unsupported}
    baselined = _baselined_ids()

    new = sorted(unsupported - baselined)
    fixed = sorted(baselined - unsupported)
    assert not new, (
        f"ledger rows now claim BUILT for code nothing reaches: {new}. "
        f"Wire it in, or correct the row - do not add it here without doing one.")
    assert not fixed, (
        f"these are no longer unsupported: {fixed}. Delete them from "
        f"{BASELINE.name} so the ratchet keeps its teeth.")


def test_the_baseline_explains_every_row_it_carries():
    """A bare list of IDs decays into wallpaper. Each entry has to sit under a
    comment saying why it is dead and what would clear it."""
    lines = BASELINE.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip() or line.startswith("#"):
            continue
        preceding = [l for l in lines[:index] if l.strip()]
        assert preceding and preceding[-1].startswith("#") or any(
            l.startswith("#") for l in lines[max(0, index - 12):index]), \
            f"{line.strip()} is baselined with no explanation above it"


def test_the_real_ledger_is_parseable_and_not_silently_empty():
    """A parser that quietly matches nothing reports a clean bill of health, and
    that is indistinguishable from a clean repo."""
    if not LEDGER.is_dir():
        pytest.skip("ledger not present on this machine")

    claims = read_ledger_claims(LEDGER)
    assert len(claims) > 20, f"only {len(claims)} claiming rows parsed - check the format"
    assert any(row_id == "DM-066" for row_id, _, _, _ in claims)


def test_this_repo_has_no_module_that_cannot_be_parsed():
    """Guards the audit's own input. One unreadable module and its edges vanish."""
    assert read_source_tree(REPO_ROOT / "src")
