"""Find claims of BUILT whose code nothing can reach.

Four defects in this project have now had the same shape, and all four failed in
the flattering direction: a claim written at the moment the code was written and
never re-checked against whether anything calls it.

- `tail_specs()` - built, tested, called by nothing. The repo's standing warning.
- Ledger row **DM-066**, "the dollar-quote filter" - marked BUILT while
  `dollar_quoted_symbols` had zero callers and `store.cli --symbols ALL` built
  every captured symbol unfiltered. 539 of 2,123 listed pairs are not
  dollar-quoted, and they went into bars for a week.
- The status wall's **auto-halt armed** - `VenueHaltRegistry.observe()` has no
  caller, so no degradation could ever halt a venue, and the tile said otherwise.
- Five circuit breakers and health checks that lived only in docstrings.

`CLAUDE.md` warns about the first. Nothing checked for the rest, because the
existing guards all point at code: tests prove a function works, and a function
nothing calls passes its tests forever. What was missing is a check pointed at
the *claims* - the ledger rows and the tiles that say a thing is real.

## What "reachable" means here

An import graph over `src/`, seeded from the modules that scripts actually
invoke, walked transitively. A module imported only by another unreachable module
is unreachable too - otherwise a dead subsystem vouches for itself, which is
exactly how `validation/` looks healthy: seven modules, every one of them reached
only through `promotion_gate`, and `promotion_gate` reached by nothing.

Three outcomes rather than two, because "has no importer" is not one situation:

- **running** - reachable from a module a script invokes. Live in production.
- **by hand** - reachable only from a module that defines `__main__` and that no
  script runs. `cost.round_trip_cost` is here: the paper demo and the cost CLI
  both use it, and both are things a person types. Built and exercised, just not
  in the running system. Not a defect, and calling it one is how a check gets
  switched off - the first version of this reported nine such modules as dead.
- **unreachable** - not reachable from anything, by hand or otherwise. The dead
  set, and the only one that fails.

## What it deliberately does not do

Symbol-level reference checking is by NAME, not by resolved binding, because
`registry.observe(...)` cannot be resolved to `VenueHaltRegistry.observe` without
type inference. So a method sharing a name with any attribute used anywhere
counts as referenced.

That direction is chosen on purpose: this check must not cry wolf. A false alarm
gets the whole thing switched off, and then it protects nothing. It will miss
some dead symbols; it will not invent dead ones. Module-level reachability is
exact and is where the hard failure lives.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# `| # | Requirement | Category | Status | Phase | Evidence | Sources | Notes |`
_LEDGER_COLUMNS = 8
_EVIDENCE_COLUMN = 5
_STATUS_COLUMN = 3
_ID_COLUMN = 0
_REQUIREMENT_COLUMN = 1

# Statuses that assert working code in THIS repo. PRIOR-ART points at another
# repository and PLANNED at nothing yet, so neither can be contradicted by what
# is or is not reachable here.
CLAIMING_STATUSES = ("BUILT", "CLAIMED")

# `src/store/quote_currency.py` -> store.quote_currency
_SRC_PATH = re.compile(r"src/([A-Za-z0-9_/]+)\.py")
# Backticked code spans are where evidence names things: `RawWriter.close_if_hour_ended`,
# `captured_symbols`, `VenueHaltRegistry.observe()`.
_CODE_SPAN = re.compile(r"`([^`]+)`")
_DOTTED_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b")


@dataclass(frozen=True)
class ModuleFacts:
    name: str
    path: Path
    imports: frozenset[str]
    defines: frozenset[str]
    references: frozenset[str]
    has_main: bool


@dataclass
class ClaimAudit:
    """One ledger row, and what its named evidence turned out to be."""
    row_id: str
    requirement: str
    status: str
    dead_modules: list[str] = field(default_factory=list)
    uncalled_symbols: list[str] = field(default_factory=list)

    @property
    def is_unsupported(self) -> bool:
        """Only a dead module fails. See the module docstring on name matching -
        an uncalled symbol is reported for a human to judge, never asserted on."""
        return bool(self.dead_modules)


def module_name_for(path: Path, src_root: Path) -> str:
    relative = path.relative_to(src_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def read_module(path: Path, src_root: Path) -> ModuleFacts:
    """Parse one module into what it imports, defines and mentions.

    A syntax error is raised rather than skipped: a module this cannot read is a
    module whose imports are invisible, and an invisible import is what would
    make something look unreachable when it is not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    defines: set[str] = set()
    references: set[str] = set()
    has_main = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module)
            # `from capture import venues` names a module through the alias, not
            # through `node.module`, so both spellings have to be recorded or a
            # package-style import reads as no edge at all.
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defines.add(node.name)
        elif isinstance(node, ast.Attribute):
            references.add(node.attr)
        elif isinstance(node, ast.Name):
            references.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Docstrings and messages are NOT references - the whole point is to
            # catch a module whose only mentions are prose about it. Skipped
            # explicitly so the omission reads as a decision.
            continue

    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"):
            has_main = True

    return ModuleFacts(module_name_for(path, src_root), path, frozenset(imports),
                       frozenset(defines), frozenset(references), has_main)


def read_source_tree(src_root: Path) -> dict[str, ModuleFacts]:
    return {facts.name: facts
            for facts in (read_module(path, src_root)
                          for path in sorted(Path(src_root).rglob("*.py"))
                          if "__pycache__" not in path.parts)}


def invoked_modules(repo_root: Path) -> set[str]:
    """Modules something actually runs: `python -m X` in a script, or a console
    entry point in `pyproject.toml`.

    These are the graph's roots. Getting them wrong in the generous direction
    hides dead code; getting them wrong in the strict direction cries wolf. They
    are read from the files that do the invoking rather than guessed from which
    modules look like entry points.
    """
    roots: set[str] = set()
    for path in list((Path(repo_root) / "scripts").rglob("*")) + [Path(repo_root) / "pyproject.toml"]:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        roots.update(re.findall(r"-m\s+([A-Za-z_][A-Za-z0-9_.]*)", text))
    return roots


def reachable_from(modules: dict[str, ModuleFacts], roots: set[str]) -> set[str]:
    """Everything the roots can reach, transitively.

    Transitive on purpose. A dead subsystem whose modules import each other would
    otherwise vouch for itself - which is precisely the shape `validation/` has.
    """
    known = set(modules)
    frontier = [r for r in roots if r in known]
    reached: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        for imported in modules[name].imports:
            # An import of `store.temporal_schema` also keeps `store` alive, and
            # `from store.cli import x` records `store.cli.x` which is not a
            # module - so every prefix is considered and only real ones count.
            parts = imported.split(".")
            for depth in range(len(parts), 0, -1):
                candidate = ".".join(parts[:depth])
                if candidate in known and candidate not in reached:
                    frontier.append(candidate)
    return reached


RUNNING = "running"
BY_HAND = "by hand"
UNREACHABLE = "unreachable"


def classify_modules(modules: dict[str, ModuleFacts], roots: set[str]) -> dict[str, str]:
    """Each module as `running`, `by hand`, or `unreachable`.

    Two passes, and the second is what stops this crying wolf. A module reached
    only from a hand-run CLI is exercised every time someone runs that CLI, and
    the first version of this called nine such modules dead - `round_trip_cost`
    among them, which the paper demo quotes every cost through.

    A package `__init__` that defines nothing is excluded entirely: it is
    namespace, and reporting `paper` alongside `paper.fill_model` says the same
    thing twice while making the list look worse than it is.
    """
    running = reachable_from(modules, roots)
    by_hand = reachable_from(
        modules, {name for name, facts in modules.items() if facts.has_main}) - running

    verdict = {}
    for name, facts in modules.items():
        if facts.path.name == "__init__.py" and not facts.defines:
            continue
        if name in running:
            verdict[name] = RUNNING
        elif name in by_hand:
            verdict[name] = BY_HAND
        else:
            verdict[name] = UNREACHABLE
    return verdict


def uncalled_public_symbols(modules: dict[str, ModuleFacts]) -> dict[str, list[str]]:
    """Public functions and classes no other module mentions by name.

    By name, not by binding - see the module docstring. Underscore-prefixed names
    are skipped: they are declared private, so having no external caller is the
    intent rather than a finding.
    """
    elsewhere: dict[str, set[str]] = {}
    for name, facts in modules.items():
        for other, other_facts in modules.items():
            if other == name:
                continue
            elsewhere.setdefault(name, set()).update(other_facts.references)
    return {name: sorted(symbol for symbol in facts.defines
                         if not symbol.startswith("_")
                         and symbol not in elsewhere.get(name, set()))
            for name, facts in modules.items()}


def _evidence_targets(cell: str) -> tuple[set[str], set[str]]:
    """Modules and symbols an evidence cell names."""
    found_modules = {match.replace("/", ".") for match in _SRC_PATH.findall(cell)}
    symbols: set[str] = set()
    for span in _CODE_SPAN.findall(cell):
        if _SRC_PATH.search(span):
            continue
        for dotted in _DOTTED_NAME.findall(span):
            # The last component of `VenueHaltRegistry.observe` is what a caller
            # writes at the call site, and the bare name covers a free function.
            symbols.add(dotted.rsplit(".", 1)[-1])
    return found_modules, symbols


def read_ledger_claims(ledger_dir: Path) -> list[tuple[str, str, str, str]]:
    """`(row_id, requirement, status, evidence)` for every row claiming this repo."""
    claims = []
    for path in sorted(Path(ledger_dir).rglob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < _LEDGER_COLUMNS:
                continue
            status = cells[_STATUS_COLUMN].replace("*", "").strip()
            if not any(status.upper().startswith(s) for s in CLAIMING_STATUSES):
                continue
            claims.append((cells[_ID_COLUMN], cells[_REQUIREMENT_COLUMN], status,
                           cells[_EVIDENCE_COLUMN]))
    return claims


def audit_claims(repo_root: Path, ledger_dir: Path) -> list[ClaimAudit]:
    """Every claiming ledger row, against what its named code can actually reach."""
    src_root = Path(repo_root) / "src"
    modules = read_source_tree(src_root)
    verdict = classify_modules(modules, invoked_modules(repo_root))
    uncalled = uncalled_public_symbols(modules)
    dead_symbols = {symbol for name, symbols in uncalled.items() for symbol in symbols}

    audits = []
    for row_id, requirement, status, evidence in read_ledger_claims(ledger_dir):
        named_modules, named_symbols = _evidence_targets(evidence)
        audit = ClaimAudit(
            row_id=row_id, requirement=requirement, status=status,
            dead_modules=sorted(m for m in named_modules
                                if verdict.get(m) == UNREACHABLE),
            uncalled_symbols=sorted(named_symbols & dead_symbols))
        if audit.dead_modules or audit.uncalled_symbols:
            audits.append(audit)
    return audits


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Find claims of BUILT whose code nothing can reach.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--ledger", default=str(Path.home() / "research" / "ledger" / "merged"))
    parser.add_argument("--show-dead-code", action="store_true",
                        help="also list unreachable modules no ledger row claims")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    modules = read_source_tree(repo_root / "src")
    verdict = classify_modules(modules, invoked_modules(repo_root))

    if args.show_dead_code:
        for state in (UNREACHABLE, BY_HAND):
            named = sorted(n for n, v in verdict.items() if v == state)
            print(f"\n{state.upper()} ({len(named)}):")
            for name in named:
                print(f"  {name}")

    audits = audit_claims(repo_root, Path(args.ledger))
    unsupported = [a for a in audits if a.is_unsupported]

    print(f"\n{len(unsupported)} unsupported claim(s); "
          f"{len(audits) - len(unsupported)} row(s) naming an uncalled symbol.")
    for audit in audits:
        marker = "UNSUPPORTED" if audit.is_unsupported else "check"
        print(f"\n[{marker}] {audit.row_id} {audit.status} - {audit.requirement}")
        for module in audit.dead_modules:
            print(f"    unreachable module: {module}")
        for symbol in audit.uncalled_symbols:
            print(f"    no caller by name : {symbol}")

    # Nonzero only on a dead module. A named symbol nothing references is
    # reported for a human to judge and never fails a build - see the module
    # docstring on why this check must not cry wolf.
    return 1 if unsupported else 0


if __name__ == "__main__":
    raise SystemExit(main())
