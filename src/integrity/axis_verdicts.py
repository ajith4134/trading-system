"""Which modules carry a learning / reasoning / depth verdict, and which do not.

`§1a.6` of the goal spec:

    **No module ships without an axis verdict.** Learning / reasoning / depth,
    each pass or fail with the evidence named. A module may legitimately be *not
    applicable* on an axis - a Parquet writer does not reason - but that must be
    stated, not left blank.

    **The status wall carries the verdicts** (Rule 8). A module claimed
    intelligent with no passing test renders as `NOT MEASURED`, never as green.

The verdict itself is a judgement and this module does not pretend otherwise.
What it MEASURES is the two things that can be measured:

- **coverage** - which modules have a verdict at all, enumerated from `src/`
  rather than from the verdict file, so adding a module without a verdict makes
  the number fall rather than leaving the omission invisible;
- **evidence that exists** - every verdict names an artifact, and a named
  artifact that is not on disk is reported as unsupported. A verdict citing a
  test file nobody wrote is the same defect as a ledger row citing a module
  nobody calls, which this package already has a check for.

`n/a` is a first-class verdict and carries its reason. A Parquet writer that
does not reason should say so once, and never be counted as an omission again.
The axis it is not applicable on still has to be named, which is the difference
between deciding and forgetting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

AXES = ("learning", "reasoning", "depth")
PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "n/a"
VERDICTS = (PASS, FAIL, NOT_APPLICABLE)

# Beside the specs rather than in `src/`: it is a record about the code, not part
# of it, and it is edited by whoever reviews a module rather than by the module.
VERDICT_FILE = Path("docs") / "axis-verdicts.json"


@dataclass(frozen=True)
class ModuleVerdict:
    module: str
    axes: dict
    evidence: list[str] = field(default_factory=list)

    def missing_axes(self) -> list[str]:
        return [a for a in AXES if a not in self.axes]

    def invalid_axes(self) -> list[str]:
        return [a for a, v in self.axes.items()
                if a in AXES and str(v).split(":", 1)[0].strip() not in VERDICTS]

    def failed_axes(self) -> list[str]:
        return [a for a, v in self.axes.items()
                if str(v).split(":", 1)[0].strip() == FAIL]


@dataclass
class Coverage:
    """What the verdict file says, against what is actually in `src/`."""

    verdicted: list[str]
    unverdicted: list[str]
    incomplete: dict            # module -> axes with no verdict
    invalid: dict               # module -> axes whose verdict is not a verdict
    missing_evidence: dict      # module -> named artifacts not on disk
    failing: dict               # module -> axes explicitly marked fail
    orphaned: list[str]         # verdicts for modules that no longer exist

    @property
    def total(self) -> int:
        return len(self.verdicted) + len(self.unverdicted)

    @property
    def share(self) -> float:
        return len(self.verdicted) / self.total if self.total else 0.0


def read_verdicts(path: Path) -> dict[str, ModuleVerdict]:
    """The declared verdicts. A missing file is no verdicts, not an error.

    No verdicts is the honest starting state of a repo that has never reviewed
    itself, and raising there would make "nobody has judged this yet"
    indistinguishable from a broken check.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    entries = body.get("modules", {}) if isinstance(body, dict) else {}
    return {
        name: ModuleVerdict(
            module=name,
            axes={k: v for k, v in entry.items() if k in AXES},
            evidence=list(entry.get("evidence", [])),
        )
        for name, entry in entries.items() if isinstance(entry, dict)
    }


def assess(repo_root: Path, verdict_file: Path | None = None) -> Coverage:
    """Cross-reference the verdicts against the modules that actually exist."""
    from integrity.unsupported_claims import read_source_tree

    repo_root = Path(repo_root)
    verdicts = read_verdicts(verdict_file or repo_root / VERDICT_FILE)

    # Enumerated from src/, never from the verdict file. A module added without
    # a verdict must make the number fall; reading the file for the list would
    # make every omission invisible, which is the failure this exists to stop.
    modules = {
        name for name, facts in read_source_tree(repo_root / "src").items()
        if not (facts.path.name == "__init__.py" and not facts.defines)
    }

    verdicted = sorted(m for m in modules if m in verdicts)
    incomplete, invalid, missing_evidence, failing = {}, {}, {}, {}
    for name in verdicted:
        verdict = verdicts[name]
        if verdict.missing_axes():
            incomplete[name] = verdict.missing_axes()
        if verdict.invalid_axes():
            invalid[name] = verdict.invalid_axes()
        if verdict.failed_axes():
            failing[name] = verdict.failed_axes()
        absent = [e for e in verdict.evidence if not (repo_root / e).exists()]
        if absent:
            missing_evidence[name] = absent

    return Coverage(
        verdicted=verdicted,
        unverdicted=sorted(modules - set(verdicts)),
        incomplete=incomplete,
        invalid=invalid,
        missing_evidence=missing_evidence,
        failing=failing,
        orphaned=sorted(set(verdicts) - modules),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="axis-verdicts",
        description="§1a.6: which modules carry a learning/reasoning/depth verdict.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--list-unverdicted", action="store_true")
    args = parser.parse_args(argv)

    coverage = assess(Path(args.repo_root))
    print(f"{len(coverage.verdicted)}/{coverage.total} modules carry an axis verdict "
          f"({coverage.share:.0%})")
    for label, rows in (("incomplete", coverage.incomplete),
                        ("invalid verdict", coverage.invalid),
                        ("evidence not on disk", coverage.missing_evidence),
                        ("marked FAIL", coverage.failing)):
        for name, detail in sorted(rows.items()):
            print(f"  [{label}] {name}: {detail}")
    for name in coverage.orphaned:
        print(f"  [orphaned] {name}: a verdict for a module that no longer exists")
    if args.list_unverdicted:
        print(f"\nunverdicted ({len(coverage.unverdicted)}):")
        for name in coverage.unverdicted:
            print(f"  {name}")

    # Nonzero on a verdict that is broken - incomplete, invalid, or citing
    # evidence nobody wrote. NOT on an unverdicted module: those are counted and
    # rendered, and failing the build for them would mean the standard could
    # only be adopted all at once.
    broken = bool(coverage.incomplete or coverage.invalid
                  or coverage.missing_evidence or coverage.orphaned)
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
