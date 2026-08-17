# AJIT MASTER PLAN — slice 0 governance implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the governance half of slice 0 — the AJIT MASTER PLAN document, the rulings register, the three-population reconciliation with scope arithmetic, the two boards that measure them, and the two hooks that enforce them — plus the two urgent operational fixes that cannot wait for a later slice.

**Architecture:** A written spine (`docs/AJIT-MASTER-PLAN.md`) holds order and decisions and is committed. A generated board (`ajit-master-plan.html`) holds measured state and is not committed. `src/plan/` parses the spine, sweeps three source populations, and applies per-scope coverage arithmetic; `src/statuswall/` renders both boards on the existing measurement pass. Two shell hooks in `~/.claude/hooks/` make the plan non-optional: one refuses a new `src/` module with no plan row, one injects the ruling list into every session.

**Tech Stack:** Python 3.12 via `uv` (never system Python 3.14.4 — too new for polars) · pytest with `pythonpath = ["src"]` · stdlib `json`, `re`, `pathlib`, `html` only — no new dependencies · bash for hooks · `gcloud storage` for the backup.

**Spec:** `docs/superpowers/specs/2026-08-17-ajit-master-plan-design.md`

## Global Constraints

- **Python 3.12 via `uv`.** Tests run `.venv/bin/python -m pytest -q`. Baseline to preserve: **857 passed, 1 skipped** (2026-08-09).
- **No new third-party dependency.** Everything here is stdlib. `docs/rulings.json` is JSON, not YAML, because `yaml` is not installed and adding it for a config file is not worth a dependency.
- **`src/plan` MUST be added to `packages` in `pyproject.toml`.** Commit `1570ed5` records a wheel built from this repo shipping without `features/`, `models/`, `risk/` and `integrity/` because that list was not updated. Same trap, same file.
- **No state value is ever typed into the spine.** Every row's `state:` field holds a probe name. Task 4 makes this a parse error.
- **A row with no probe renders NOT MEASURED, never green** (Rule 8). Absence of evidence is its own state.
- **State vocabulary is the existing one** from `statuswall.evidence`: `FAILING · DEGRADED · STOPPED · NOT_MEASURED · PARTIAL · OK · BUILT · NOT_BUILT`. Do not invent new states. `BLOCKED` and `DECLINED` are *decided* and live in the spine, not in `ProbeResult`.
- **Timestamps int64 nanoseconds UTC. Fees `Decimal` basis points, never float.** (Repo-wide; no money code in this plan, but the rule stands.)
- **Docstrings explain *why*. Tests are named for the behaviour they defend.**
- **Hook exit codes: `exit 2` blocks, `exit 1` does NOT.** PreToolUse fails closed. Never parse `transcript_path` for the current turn.
- **Never commit generated output that reports state** (Rule 9). `ajit-master-plan.html` and `ruling-conformance.html` are written to `~/research/dashboard/`, which is not in the repo.
- **Every task ends with a commit. Push at the end of each task** (Rule 9).

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/plan/__init__.py` | package marker |
| `src/plan/rulings.py` | read and validate `docs/rulings.json`; one `Ruling` per row |
| `src/plan/master_plan.py` | parse `docs/AJIT-MASTER-PLAN.md` into slices and eight-field rows; refuse a typed state |
| `src/plan/scope_coverage.py` | per-scope arithmetic: how many of the required rows exist, and is it resolved |
| `src/plan/reconcile_sources.py` | sweep the three populations and classify every member |
| `src/statuswall/ruling_conformance.py` | one probe per ruling; render the conformance board |
| `src/statuswall/master_plan_board.py` | render the plan board: slices, counts, active slice, next row, reconciliation totals |
| `docs/AJIT-MASTER-PLAN.md` | the written spine — committed |
| `~/.claude/hooks/require-plan-row.sh` | PreToolUse/Write — refuse a new `src/` module with no plan row |
| `~/.claude/hooks/inject-plan-index.sh` | SessionStart — inject rulings and active slice |
| `~/.claude/hooks/tests/test-require-plan-row.sh` | the hook's own test, in the style of `test-stop-hook.py` |

Split by responsibility, not layer: `rulings.py` is one file because a ruling register is one concept, and `master_plan.py` never reads `rulings.json` — it only records the ruling ids a row claims to satisfy. `scope_coverage.py` is separate because its arithmetic is the part most likely to be wrong and most worth testing alone.

---

### Task 1: The store reaches GCS

The urgent one, and it is first for that reason. Measured 2026-08-17: the bucket holds `raw/`, `ledger/` and `universe/` — not `store/`. `store/funding` is 446 MB, is the OBSERVED record whose start date is the clock every promotion waits on, and has no raw counterpart at all: `~/capture/raw/binance-funding` does not exist because funding is polled straight into the store. Losing this disk restarts that clock from zero, and `funding_reconstructed` cannot substitute — its availability times are the fetch, which is deliberately what makes it useless for a backtest.

**Files:**
- Modify: `scripts/offload_to_gcs.sh`
- Test: `tests/test_offload_covers_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks rely on. Standalone.

- [ ] **Step 1: Read the script before changing it**

Run: `sed -n '1,120p' scripts/offload_to_gcs.sh`

Read what `CAPTURE_ROOT`, `BUCKET` and the per-directory loop are named before editing. Do not guess the variable names — the code below uses `raw` as the literal that must gain a sibling, and the surrounding loop must be matched exactly as written in the file.

- [ ] **Step 2: Write the failing test**

```python
"""The offload covers the store, not only the raw tape.

`store/funding` is polled straight into the store with no raw counterpart, so
a raw-only backup silently omits the one dataset whose loss cannot be undone:
the OBSERVED funding record whose start date is the promotion clock.
"""
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offload_to_gcs.sh"


def test_offload_script_copies_the_store_as_well_as_raw():
    body = SCRIPT.read_text()
    assert "store" in body, (
        "offload_to_gcs.sh must copy $CAPTURE_ROOT/store; funding lives only there")


def test_offload_script_still_copies_raw():
    body = SCRIPT.read_text()
    assert "raw" in body, "the raw tape backup must not be lost while adding the store"
```

- [ ] **Step 3: Run the tests to verify the first one fails**

Run: `.venv/bin/python -m pytest tests/test_offload_covers_store.py -v`
Expected: `test_offload_script_copies_the_store_as_well_as_raw` FAILS; `test_offload_script_still_copies_raw` PASSES.

- [ ] **Step 4: Add the store to the offload**

Edit `scripts/offload_to_gcs.sh` so the directory list the script walks includes `store` beside `raw`. Follow whatever loop shape Step 1 revealed rather than replacing it. Add this comment above the change, because the reason is not obvious from the code:

```bash
# `store` is here and not only `raw` because funding has no raw counterpart:
# it is polled directly into store/funding, so a raw-only backup omits the
# OBSERVED record whose start date is the promotion clock. Measured
# 2026-08-17: the bucket held raw/, ledger/ and universe/ and nothing else.
```

- [ ] **Step 5: Run the tests to verify both pass**

Run: `.venv/bin/python -m pytest tests/test_offload_covers_store.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run the offload once and verify the bytes landed**

Run: `bash scripts/offload_to_gcs.sh gs://capture-raw-data4134`
Then: `gcloud storage ls gs://capture-raw-data4134/`
Expected: a `store/` prefix now appears beside `raw/`, `ledger/` and `universe/`.
Then: `gcloud storage ls gs://capture-raw-data4134/store/funding/ | head`
Expected: non-empty. **Paste the output.** A success message from the script is not the verification (Rule 0).

- [ ] **Step 7: Commit and push**

```bash
git add scripts/offload_to_gcs.sh tests/test_offload_covers_store.py
git commit -m "fix: the store was never backed up, and funding has no raw copy"
git push origin main
```

---

### Task 2: Every session reaches the same memory

Measured 2026-08-17: twelve project memories exist under `~/.claude/projects/-home-anushadudekula71/memory/`, including one titled *"Interview, don't assume"* and one pointing at the goal document and the ledger. Memory is keyed by working directory, so a session started from `/` loads `~/.claude/projects/-/memory/` — empty. This is a path, not a discipline failure, and it is a direct cause of the forgetting this whole plan answers.

**Files:**
- Create: `~/.claude/projects/-/memory` (symlink)
- No repo files change.

**Interfaces:**
- Consumes: nothing. Produces: nothing. Standalone.

- [ ] **Step 1: Verify the problem before fixing it**

```bash
ls -la ~/.claude/projects/-/memory/ | head
ls ~/.claude/projects/-home-anushadudekula71/memory/ | wc -l
```
Expected: the first is empty or absent; the second reports 13 (12 memories plus `MEMORY.md`).

- [ ] **Step 2: Point the empty key at the real one**

```bash
rmdir ~/.claude/projects/-/memory 2>/dev/null
ln -s ~/.claude/projects/-home-anushadudekula71/memory ~/.claude/projects/-/memory
```

If `rmdir` fails because the directory is non-empty, STOP and report what is in it — do not delete files you have not looked at.

- [ ] **Step 3: Verify the fix**

```bash
ls ~/.claude/projects/-/memory/ | wc -l
cat ~/.claude/projects/-/memory/MEMORY.md | head -3
```
Expected: 13, and the first three index lines. **Paste the output.**

- [ ] **Step 4: Record it as a memory, since it is exactly the kind of fact that gets re-derived**

Write `~/.claude/projects/-home-anushadudekula71/memory/memory-is-keyed-by-directory.md`:

```markdown
---
name: memory-is-keyed-by-directory
description: Project memory lives under a per-working-directory key; a session started from / gets an empty set unless the key is symlinked.
metadata:
  type: reference
---

Memory is stored at `~/.claude/projects/<cwd-key>/memory/`. A session whose cwd is
`/` reads `projects/-/memory/`, which was empty while 12 real memories sat under
`projects/-home-anushadudekula71/memory/`. Fixed 2026-08-17 by symlinking the `-`
key at the real directory.

**Why:** every session started from `/` began with no project memory at all,
including [[interview-instead-of-assuming]] and [[trading-bot-goal-and-ledger]].

**How to apply:** if memory looks empty in a session, check which key the cwd maps
to before concluding nothing was ever saved.
```

Then add its pointer line to that directory's `MEMORY.md`:

```
- [Memory is keyed by directory](memory-is-keyed-by-directory.md) — a session started from / reads an empty key unless it is symlinked.
```

- [ ] **Step 5: Commit and push the config repo**

```bash
cd ~/crypto-bot-publish 2>/dev/null || true
# The claude-config repo is pushed by ~/.claude/push-all-repos.sh; run it rather
# than hand-copying, so the allowlist is respected (Rule 9: config gets an
# allowlist, never an ignore-list).
bash ~/.claude/push-all-repos.sh
```

Expected: the new memory file appears in `ajith4134/claude-config`. Verify with `gh api repos/ajith4134/claude-config/git/trees/HEAD?recursive=1 --jq '.tree[].path' | grep memory-is-keyed`. **Paste the output.** If the push script does not exist or errors, say so plainly and report the memory as written locally but not backed up.

---

### Task 3: The rulings register loads and validates

**Files:**
- Create: `src/plan/__init__.py`
- Create: `src/plan/rulings.py`
- Create: `tests/test_rulings_register.py`
- Modify: `pyproject.toml` — add `"src/plan"` to `packages`

**Interfaces:**
- Consumes: `docs/rulings.json`, already written on 2026-08-17 with 21 rows.
- Produces:
  - `Ruling` — frozen dataclass with fields `id: str`, `date: str`, `session: str`, `verbatim: str`, `means: str`, `recorded_in: tuple[str, ...]`, `probe: str | None`, `scope: str`, `superseded_by: str | None`, `clarified_as: str | None`
  - `read_rulings(path: Path) -> list[Ruling]`
  - `MalformedRuling(Exception)`
  - `SCOPES: frozenset[str]` = `{"per-segment", "per-brain", "shared", "system"}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rulings_register.py`:

```python
"""The rulings register is the record of what the user settled.

Its one job is to be complete and unambiguous, so the tests defend exactly
that: every row identifiable, every scope from the closed set, and a probe
field that is present even when it is null - because an absent probe field is
indistinguishable from an oversight, while an explicit null is a decision.
"""
import json
from pathlib import Path

import pytest

from plan.rulings import MalformedRuling, Ruling, SCOPES, read_rulings


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


# --- refusals -------------------------------------------------------------

def test_a_missing_probe_field_is_refused_but_an_explicit_null_is_accepted():
    no_field = _row()
    del no_field["probe"]
    with pytest.raises(MalformedRuling, match="probe"):
        read_rulings(_write(Path("/tmp"), [no_field]))

    explicit_null = read_rulings(_write(Path("/tmp"), [_row(probe=None)]))
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rulings_register.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'plan'`.

- [ ] **Step 3: Write the implementation**

Create `src/plan/__init__.py` as an empty file.

Create `src/plan/rulings.py`:

```python
"""The rulings register - what the user settled, in the user's own words.

A ruling is not a feature and not a plan. It is a decision the build is not
free to re-derive, and the reason this module exists is that for two weeks
RL-011 existed only in a conversation transcript while the build went another
way.

No row carries a status. Each names a probe, and the board renders what that
probe measured; a typed status would be exactly the fiction the board exists
to prevent (Rule 8). A `probe` of null is a legitimate answer - it renders NOT
MEASURED - but the FIELD must be present, because an absent field is
indistinguishable from an oversight while an explicit null is a decision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCOPES = frozenset({"per-segment", "per-brain", "shared", "system"})

_REQUIRED = ("id", "date", "verbatim", "means", "probe", "scope")


class MalformedRuling(Exception):
    """A ruling row the register cannot trust, named by what is wrong with it."""


@dataclass(frozen=True)
class Ruling:
    id: str
    date: str
    session: str
    verbatim: str
    means: str
    recorded_in: tuple[str, ...]
    probe: str | None
    scope: str
    superseded_by: str | None = None
    clarified_as: str | None = None


def read_rulings(path: Path) -> list[Ruling]:
    """Load the register, refusing anything ambiguous rather than guessing."""
    document = json.loads(Path(path).read_text())
    rows = document["rulings"]

    rulings: list[Ruling] = []
    seen: set[str] = set()
    for row in rows:
        for field in _REQUIRED:
            if field not in row:
                raise MalformedRuling(
                    f"{row.get('id', '<no id>')} is missing the {field!r} field")
        if "status" in row:
            raise MalformedRuling(
                f"{row['id']} carries a 'status' field; status is measured by a "
                f"probe and never typed into the register")
        if row["scope"] not in SCOPES:
            raise MalformedRuling(
                f"{row['id']} has scope {row['scope']!r}, which is not one of "
                f"{sorted(SCOPES)}")
        if row["id"] in seen:
            raise MalformedRuling(f"{row['id']} appears more than once")
        seen.add(row["id"])

        rulings.append(Ruling(
            id=row["id"],
            date=row["date"],
            session=row.get("session", ""),
            verbatim=row["verbatim"],
            means=row["means"],
            recorded_in=tuple(row.get("recorded_in", ())),
            probe=row["probe"],
            scope=row["scope"],
            superseded_by=row.get("superseded_by"),
            clarified_as=row.get("clarified_as"),
        ))
    return rulings
```

- [ ] **Step 4: Add the scope field to every row in the real register**

`docs/rulings.json` was written before the scope rule was designed, so its 21 rows have no `scope` key and the real-register tests will fail. Add one to each, using the scopes the user ruled on 2026-08-17 and the default rule for the rest (decides about a symbol or market → `per-segment`; about a brain's own learning → `per-brain`; about capture, store, cost, validation, registry, tail cap or governance → `shared`; objective and whole-account concerns → `system`):

| Ruling | scope |
|---|---|
| RL-001 permanent rules | `system` |
| RL-002 model selection | `system` |
| RL-003 never skip an install | `system` |
| RL-004 institutional-grade code | `per-brain` |
| RL-005 paper as experimentation ground | `system` |
| RL-006 three segments, two directional agents | `per-segment` |
| RL-007 in order without skipping | `system` |
| RL-008 naming | `system` |
| RL-009 examination hall / universe scanning | `per-segment` |
| RL-010 real intelligence and autonomy | `per-brain` |
| RL-011 BULL / BEAR / PROFIT-TAIL as own bots | `per-brain` |
| RL-012 dashboard shows every feature behaving | `system` |
| RL-013 real intelligence, not hardcoded | `per-brain` |
| RL-014 watch all symbols in all segments | `per-segment` |
| RL-015 everything then paper (superseded) | `system` |
| RL-016 interview, do not assume | `system` |
| RL-017 feature by feature, paper now | `system` |
| RL-018 intraday on all segments | `per-segment` |
| RL-019 each segment its own bot | `per-segment` |
| RL-020 24/7 uptime | `shared` |
| RL-021 enforcement mechanisms | `system` |

- [ ] **Step 5: Add `src/plan` to the packaged list**

In `pyproject.toml`, add `"src/plan"` to the `packages` list. Commit `1570ed5` records a wheel shipping without four packages because this list was not updated — do not repeat it.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rulings_register.py -v`
Expected: 6 passed.

- [ ] **Step 7: Run the whole suite to confirm nothing regressed**

Run: `.venv/bin/python -m pytest -q`
Expected: at least 857 passed. **Paste the count.**

- [ ] **Step 8: Commit and push**

```bash
git add src/plan tests/test_rulings_register.py docs/rulings.json pyproject.toml
git commit -m "feat: the rulings register loads, and refuses a typed status"
git push origin main
```

---

### Task 4: The spine parses, and a typed state is a parse error

**Files:**
- Create: `src/plan/master_plan.py`
- Create: `tests/test_master_plan_spine.py`

**Interfaces:**
- Consumes: `Ruling`, `SCOPES` from `plan.rulings` (Task 3).
- Produces:
  - `PlanRow` — frozen dataclass: `id: str`, `slice_key: str`, `does: str`, `satisfies: tuple[str, ...]`, `sources: tuple[str, ...]`, `depends_on: tuple[str, ...]`, `probe: str | None`, `accepts: str`, `decided: str | None`
  - `PlanSlice` — frozen dataclass: `key: str`, `title: str`, `rows: tuple[PlanRow, ...]`
  - `read_master_plan(path: Path) -> list[PlanSlice]`
  - `MalformedRow(Exception)`, `TypedState(Exception)`
  - `ROW_FIELDS: tuple[str, ...]` = `("does", "satisfies", "sources", "depends on", "probe", "accepts", "state")`
  - `DECIDED_STATES: frozenset[str]` = `{"BLOCKED", "DECLINED"}`

The spine format the parser reads — a slice is an `## ` heading of the form `## SLICE <key> — <title>`, and a row is a fenced block:

```
### SP-04
  slice:      spot-bot
  does:       compute spot-only microstructure features per symbol
  satisfies:  RL-006 RL-009 RL-014 RL-019
  sources:    FEATURES §2.4 · ledger FE-014 · finml-feature-engineering.md
  depends on: SP-02 · shared clock-gated reader
  probe:      probe_spot_features
  accepts:    every value carries a staleness stamp and no value is readable
              before its availability time
  state:      measured by probe_spot_features
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_master_plan_spine.py`:

```python
"""The spine holds decisions; it must never hold a measurement.

The whole written/generated split rests on one property: no state value is
ever typed into the plan document. If that erodes, the plan starts asserting
state again and is back to being the thing that went eight days stale. So the
parser refuses a typed state by name, and refuses a row missing any of the
eight fields rather than accepting a half-specified contract.
"""
from pathlib import Path

import pytest

from plan.master_plan import (
    DECIDED_STATES,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_master_plan_spine.py -v`
Expected: collection error — `cannot import name 'read_master_plan'`.

- [ ] **Step 3: Write the implementation**

Create `src/plan/master_plan.py`:

```python
"""The written spine of AJIT MASTER PLAN - order and decisions, never state.

This parser exists to defend one property. The plan document is committed and
holds what the user settled; the board is generated and holds what probes
measured. If a state value is ever typed into the document, the two collapse
back into one artefact that asserts its own health - which is what
`2026-08-09-full-build-master-plan.md` did for eight days while paper trading
ran in contradiction of its own phase J.

So `state:` may say `measured by <probe>` and nothing else, and a row missing
any of its eight fields is refused rather than accepted half-specified: an
unwritten decision must be visible as a blank field before code is written,
not as a wrong module afterwards.

`BLOCKED` and `DECLINED` are the exception, and they are not measurements -
they are the user's decisions about a row, so they belong in the document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROW_FIELDS = ("does", "satisfies", "sources", "depends on", "probe", "accepts", "state")
DECIDED_STATES = frozenset({"BLOCKED", "DECLINED"})

# The status wall's vocabulary. Any of these in a `state:` field means somebody
# typed a measurement into the record, which is the defect this parser exists
# to catch - so they are matched by name rather than by a loose heuristic.
_MEASURED_WORDS = frozenset({
    "OK", "BUILT", "NOT BUILT", "PARTIAL", "DEGRADED", "FAILING", "STOPPED",
    "NOT MEASURED",
})

_SLICE = re.compile(r"^##\s+SLICE\s+(?P<key>[\w-]+)\s+—\s+(?P<title>.+?)\s*$")
_ROW_ID = re.compile(r"^###\s+(?P<id>[A-Z]{2,3}-\d+)\s*$")
_FIELD = re.compile(r"^\s{2}(?P<name>does|satisfies|sources|depends on|probe|accepts|state|slice):\s*(?P<value>.*)$")
_SEPARATORS = re.compile(r"\s*[·]\s*|\s{2,}")


class MalformedRow(Exception):
    """A plan row that is not a complete contract, named by what it lacks."""


class TypedState(Exception):
    """A state value typed into the spine instead of measured by a probe."""


@dataclass(frozen=True)
class PlanRow:
    id: str
    slice_key: str
    does: str
    satisfies: tuple[str, ...]
    sources: tuple[str, ...]
    depends_on: tuple[str, ...]
    probe: str | None
    accepts: str
    decided: str | None


@dataclass(frozen=True)
class PlanSlice:
    key: str
    title: str
    rows: tuple[PlanRow, ...]


def _split(value: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in _SEPARATORS.split(value) if p.strip())


def _check_state(row_id: str, state: str) -> str | None:
    """Return the decided state, or None. Raise if a measurement was typed."""
    bare = state.strip()
    decided = bare.split()[0].rstrip("-,:").upper() if bare else ""
    if decided in DECIDED_STATES:
        return decided
    if not bare.lower().startswith("measured by "):
        raise TypedState(
            f"{row_id}: state is {bare!r}; the spine may only say "
            f"'measured by <probe>', BLOCKED or DECLINED")
    named = bare[len("measured by "):].strip().upper()
    if named in _MEASURED_WORDS:
        raise TypedState(f"{row_id}: {named} is a measurement, not a probe name")
    return None


def read_master_plan(path: Path) -> list[PlanSlice]:
    """Parse the spine into slices of eight-field rows."""
    slices: list[PlanSlice] = []
    key = title = None
    rows: list[PlanRow] = []
    pending: dict[str, str] = {}
    row_id: str | None = None
    seen: set[str] = set()

    def flush() -> None:
        nonlocal pending, row_id
        if row_id is None:
            return
        missing = [f for f in ROW_FIELDS if f not in pending]
        if missing:
            raise MalformedRow(f"{row_id} is missing: {', '.join(missing)}")
        if row_id in seen:
            raise MalformedRow(f"{row_id} appears more than once")
        seen.add(row_id)
        probe = pending["probe"].strip()
        decided = _check_state(row_id, pending["state"])
        rows.append(PlanRow(
            id=row_id,
            slice_key=pending.get("slice", key or ""),
            does=pending["does"].strip(),
            satisfies=_split(pending["satisfies"]),
            sources=_split(pending["sources"]),
            depends_on=_split(pending["depends on"]),
            probe=None if probe.lower() in {"none", "null", ""} else probe,
            accepts=pending["accepts"].strip(),
            decided=decided,
        ))
        pending, row_id = {}, None

    last_field: str | None = None
    for line in Path(path).read_text().splitlines():
        heading = _SLICE.match(line)
        if heading:
            flush()
            if key is not None:
                slices.append(PlanSlice(key, title or "", tuple(rows)))
            key, title, rows = heading["key"], heading["title"], []
            last_field = None
            continue

        identifier = _ROW_ID.match(line)
        if identifier:
            flush()
            row_id = identifier["id"]
            last_field = None
            continue

        field = _FIELD.match(line)
        if field:
            last_field = field["name"]
            pending[last_field] = field["value"]
            continue

        # A continuation line: `accepts:` wraps, and the wrap belongs to it.
        if row_id and last_field and line.startswith("    ") and line.strip():
            pending[last_field] = pending[last_field] + " " + line.strip()

    flush()
    if key is not None:
        slices.append(PlanSlice(key, title or "", tuple(rows)))
    return slices
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_master_plan_spine.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/plan/master_plan.py tests/test_master_plan_spine.py
git commit -m "feat: the plan spine parses, and a typed state is a parse error"
git push origin main
```

---

### Task 5: Scope arithmetic — assigned once is not resolved

**Files:**
- Create: `src/plan/scope_coverage.py`
- Create: `tests/test_scope_coverage.py`

**Interfaces:**
- Consumes: `PlanSlice`, `PlanRow` from `plan.master_plan`; `Ruling`, `SCOPES` from `plan.rulings`.
- Produces:
  - `Coverage` — frozen dataclass: `subject: str`, `scope: str`, `have: int`, `need: int`, `missing: tuple[str, ...]`; property `resolved -> bool` (`have >= need`)
  - `SEGMENTS: tuple[str, ...]` = `("spot-bot", "perp-bot", "dated-bot", "options-bot")`
  - `BRAINS: tuple[str, ...]` = the 12 `"<segment>/<BULL|BEAR|PROFIT-TAIL>"` keys
  - `cover_ruling(ruling: Ruling, slices: list[PlanSlice]) -> Coverage`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scope_coverage.py`:

```python
"""A capability assigned once is not a capability delivered four times.

This is the arithmetic that would have caught the user's own example: the
examination-hall ruling of 2026-08-02, assigned to one slice, would read as
resolved while three of four bots never received it. So per-segment coverage
is a fraction over the four segment slices, per-brain is a fraction over
twelve, and anything short of full renders unresolved with the gap named.
"""
import pytest

from plan.master_plan import PlanRow, PlanSlice
from plan.rulings import Ruling
from plan.scope_coverage import BRAINS, SEGMENTS, cover_ruling


def _row(row_id: str, slice_key: str, satisfies=("RL-009",)) -> PlanRow:
    return PlanRow(id=row_id, slice_key=slice_key, does="does a thing",
                   satisfies=tuple(satisfies), sources=("FEATURES §1",),
                   depends_on=(), probe="probe_thing", accepts="it works",
                   decided=None)


def _slices(*pairs) -> list[PlanSlice]:
    by_key: dict[str, list[PlanRow]] = {}
    for row_id, slice_key in pairs:
        by_key.setdefault(slice_key, []).append(_row(row_id, slice_key))
    return [PlanSlice(k, k.upper(), tuple(v)) for k, v in by_key.items()]


def _ruling(scope: str) -> Ruling:
    return Ruling(id="RL-009", date="2026-08-02", session="0a15b275",
                  verbatim="the brain acts as a teacher in an examination hall",
                  means="watch every symbol, wait for its setup",
                  recorded_in=("goal §5a",), probe=None, scope=scope)


def test_four_segments_are_the_denominator_and_twelve_brains_are_the_other():
    assert len(SEGMENTS) == 4
    assert len(BRAINS) == 12
    assert all(any(b.startswith(s) for b in BRAINS) for s in SEGMENTS)


def test_a_per_segment_ruling_in_one_slice_reports_one_of_four_and_is_unresolved():
    coverage = cover_ruling(_ruling("per-segment"), _slices(("SP-01", "spot-bot")))
    assert (coverage.have, coverage.need) == (1, 4)
    assert not coverage.resolved
    assert "perp-bot" in coverage.missing


def test_a_per_segment_ruling_in_all_four_slices_is_resolved():
    coverage = cover_ruling(_ruling("per-segment"),
                            _slices(*[(f"X-{i}", s) for i, s in enumerate(SEGMENTS)]))
    assert (coverage.have, coverage.need) == (4, 4)
    assert coverage.resolved
    assert coverage.missing == ()


def test_a_per_brain_ruling_at_eleven_of_twelve_is_unresolved_and_names_the_gap():
    rows = [(f"X-{i}", b) for i, b in enumerate(BRAINS[:-1])]
    coverage = cover_ruling(_ruling("per-brain"), _slices(*rows))
    assert (coverage.have, coverage.need) == (11, 12)
    assert not coverage.resolved
    assert coverage.missing == (BRAINS[-1],)


def test_a_shared_ruling_needs_exactly_one_row_anywhere():
    coverage = cover_ruling(_ruling("shared"), _slices(("SL-01", "slice-0")))
    assert (coverage.have, coverage.need) == (1, 1)
    assert coverage.resolved


def test_a_ruling_with_no_row_at_all_is_unresolved_at_zero():
    coverage = cover_ruling(_ruling("shared"), [])
    assert (coverage.have, coverage.need) == (0, 1)
    assert not coverage.resolved
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scope_coverage.py -v`
Expected: collection error — `No module named 'plan.scope_coverage'`.

- [ ] **Step 3: Write the implementation**

Create `src/plan/scope_coverage.py`:

```python
"""Per-scope coverage - the arithmetic that stops 'assigned' meaning 'done'.

A cross-cutting capability assigned to one slice reads as fully resolved while
three of four bots silently never receive it. That is the failure this whole
plan exists to prevent, reproduced inside the plan, and it is not
hypothetical: the user's examination-hall ruling of 2026-08-02 became roughly
140 lines of design in the goal document and zero rows of work, and a
sweep that only asked "is it assigned anywhere" would have called it covered.

So the denominator comes from the scope, not from the sweep: per-segment needs
a row in all four segment slices, per-brain in all twelve brains, shared and
system need one anywhere. Anything short of the denominator is unresolved and
names the gap, because a gap that is not named is a gap nobody closes.
"""
from __future__ import annotations

from dataclasses import dataclass

from plan.master_plan import PlanSlice
from plan.rulings import Ruling

SEGMENTS = ("spot-bot", "perp-bot", "dated-bot", "options-bot")
DIRECTIONS = ("BULL", "BEAR", "PROFIT-TAIL")
BRAINS = tuple(f"{segment}/{direction}"
               for segment in SEGMENTS for direction in DIRECTIONS)


@dataclass(frozen=True)
class Coverage:
    subject: str
    scope: str
    have: int
    need: int
    missing: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return self.have >= self.need


def _keys_touching(subject: str, slices: list[PlanSlice]) -> set[str]:
    """Every slice key holding a row that claims to satisfy `subject`."""
    return {row.slice_key
            for plan_slice in slices
            for row in plan_slice.rows
            if subject in row.satisfies}


def cover_ruling(ruling: Ruling, slices: list[PlanSlice]) -> Coverage:
    """How much of a ruling the plan's rows actually cover, per its scope."""
    touched = _keys_touching(ruling.id, slices)

    if ruling.scope == "per-segment":
        missing = tuple(s for s in SEGMENTS if s not in touched)
        return Coverage(ruling.id, ruling.scope,
                        len(SEGMENTS) - len(missing), len(SEGMENTS), missing)

    if ruling.scope == "per-brain":
        missing = tuple(b for b in BRAINS if b not in touched)
        return Coverage(ruling.id, ruling.scope,
                        len(BRAINS) - len(missing), len(BRAINS), missing)

    # shared and system: one row anywhere is the whole requirement.
    have = 1 if touched else 0
    return Coverage(ruling.id, ruling.scope, have, 1,
                    () if have else ("no row anywhere",))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scope_coverage.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/plan/scope_coverage.py tests/test_scope_coverage.py
git commit -m "feat: scope arithmetic - assigned once is not resolved four times"
git push origin main
```

---

### Task 6: The three-population sweep

The population that matters most is the third. The ledger already sweeps rows; nothing sweeps **design sections**, which is how §5a became 140 lines of design and zero rows of work.

**Files:**
- Create: `src/plan/reconcile_sources.py`
- Create: `tests/test_reconcile_sources.py`

**Interfaces:**
- Consumes: `PlanSlice` from `plan.master_plan`.
- Produces:
  - `SourceMember` — frozen dataclass: `population: str`, `identifier: str`, `title: str`, `origin: str`
  - `Resolution` — frozen dataclass: `member: SourceMember`, `outcome: str`, `detail: str`
  - `OUTCOMES: frozenset[str]` = `{"assigned", "declined", "blocked", "prose", "unassigned"}`
  - `read_ledger_rows(root: Path) -> list[SourceMember]`
  - `read_catalogue_rows(path: Path) -> list[SourceMember]`
  - `read_design_sections(roots: list[Path]) -> list[SourceMember]`
  - `reconcile(members: list[SourceMember], slices: list[PlanSlice], decisions: dict[str, tuple[str, str]]) -> list[Resolution]`
  - `summarise(resolutions: list[Resolution]) -> dict[str, dict[str, int]]`

`decisions` maps a member identifier to `(outcome, detail)` for the outcomes a human writes — `declined`, `blocked` and `prose`. It is read from `docs/reconciliation-decisions.json`, which is written by hand and committed, because those three outcomes are decisions and the fourth (`assigned`) is measured from the spine.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconcile_sources.py`:

```python
"""Three populations, one sweep, and nothing gets a fourth option.

The design sections population is the one that matters: the ledger already
reconciles rows, and 'prose, no work implied' is a legitimate answer for a
section like the prime directive - but it has to be WRITTEN, because inferring
it silently is exactly how the examination-hall design became zero rows.
"""
from pathlib import Path

import pytest

from plan.master_plan import PlanRow, PlanSlice
from plan.reconcile_sources import (
    OUTCOMES,
    SourceMember,
    read_catalogue_rows,
    read_design_sections,
    read_ledger_rows,
    reconcile,
    summarise,
)


def _member(identifier: str, population: str = "design-section") -> SourceMember:
    return SourceMember(population=population, identifier=identifier,
                        title=identifier, origin="test.md")


def _slice_with(sources: tuple[str, ...]) -> list[PlanSlice]:
    row = PlanRow(id="SL-01", slice_key="slice-0", does="a thing",
                  satisfies=("RL-021",), sources=sources, depends_on=(),
                  probe="probe_thing", accepts="it works", decided=None)
    return [PlanSlice("slice-0", "SLICE 0", (row,))]


def test_a_member_named_by_a_plan_row_is_assigned():
    resolutions = reconcile([_member("goal §5a")], _slice_with(("goal §5a",)), {})
    assert resolutions[0].outcome == "assigned"


def test_a_member_no_row_names_and_no_decision_covers_is_unassigned():
    resolutions = reconcile([_member("goal §5a")], _slice_with(("other",)), {})
    assert resolutions[0].outcome == "unassigned"


def test_prose_is_a_written_decision_and_never_inferred():
    member = _member("goal §0 Prime directive")
    inferred = reconcile([member], _slice_with(("other",)), {})
    assert inferred[0].outcome == "unassigned", (
        "a section with no row must not quietly become prose")

    written = reconcile([member], _slice_with(("other",)),
                        {member.identifier: ("prose", "a directive implies no module")})
    assert written[0].outcome == "prose"
    assert "directive" in written[0].detail


def test_declined_and_blocked_carry_their_reason():
    member = _member("on-chain DEX data")
    resolutions = reconcile([member], _slice_with(()),
                            {member.identifier: ("blocked", "no DEX venue is decided")})
    assert resolutions[0].outcome == "blocked"
    assert resolutions[0].detail == "no DEX venue is decided"


def test_an_unknown_outcome_is_refused(tmp_path):
    member = _member("something")
    with pytest.raises(ValueError, match="invented"):
        reconcile([member], _slice_with(()), {member.identifier: ("invented", "no")})


def test_every_outcome_used_is_from_the_closed_set():
    assert OUTCOMES == {"assigned", "declined", "blocked", "prose", "unassigned"}


def test_summarise_counts_per_population_and_totals_match_input():
    members = [_member("a"), _member("b"), _member("c", population="ledger-row")]
    resolutions = reconcile(members, _slice_with(("a",)), {})
    counts = summarise(resolutions)
    assert counts["design-section"]["assigned"] == 1
    assert counts["design-section"]["unassigned"] == 1
    assert counts["ledger-row"]["unassigned"] == 1
    assert sum(sum(v.values()) for v in counts.values()) == len(members)


# --- the real populations, read from disk ---------------------------------

def test_the_real_populations_are_the_sizes_the_spec_measured():
    ledger = read_ledger_rows(Path.home() / "research" / "ledger" / "merged")
    catalogue = read_catalogue_rows(Path.home() / "research" / "FEATURES.md")
    sections = read_design_sections([
        Path.home() / "research",
        Path.home() / "trading-system" / "docs" / "superpowers",
    ])
    assert len(ledger) > 1000, f"ledger read {len(ledger)} rows, expected >1000"
    assert len(catalogue) > 120, f"catalogue read {len(catalogue)} rows, expected >120"
    assert len(sections) > 700, f"sections read {len(sections)}, expected >700"


def test_design_sections_carry_the_file_they_came_from():
    sections = read_design_sections([Path.home() / "research"])
    assert all(s.origin.endswith(".md") for s in sections)
    assert all(s.population == "design-section" for s in sections)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reconcile_sources.py -v`
Expected: collection error — `No module named 'plan.reconcile_sources'`.

- [ ] **Step 3: Write the implementation**

Create `src/plan/reconcile_sources.py`:

```python
"""The three populations every plan row is reconciled against.

The ledger has reconciled rows since 2026-08-08 and it works: 1,491 rows, each
CLAIMED, PLANNED, PRIOR-ART, DECLINED or UNRESOLVED. What nothing reconciled
was DESIGN SECTIONS - argued-through prose in a document that never became a
row. Measured 2026-08-17: 'Goodhart defence', 'attention scarcity' and the
examination-hall framing appear in four, two and three design files
respectively, and in ZERO ledger rows. The examination-hall framing is the
user's own ruling, written into the goal document as roughly 140 lines of §5a,
and it produced no work.

So a section is a first-class member of the sweep, and `prose` - meaning "this
section implies no module" - is a legitimate resolution that a human WRITES.
It is never inferred, because inferring it is the failure being fixed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

OUTCOMES = frozenset({"assigned", "declined", "blocked", "prose", "unassigned"})

_HEADING = re.compile(r"^(?P<hashes>#{2,3})\s+(?P<title>.+?)\s*$")
_LEDGER_ID = re.compile(r"^\|\s*(?P<id>[A-Z]{2,3}-\d+)\s*\|")
_TABLE_ROW = re.compile(r"^\|(?!\s*[-: ]+\|)(?P<first>[^|]+)\|")


@dataclass(frozen=True)
class SourceMember:
    population: str
    identifier: str
    title: str
    origin: str


@dataclass(frozen=True)
class Resolution:
    member: SourceMember
    outcome: str
    detail: str


def read_ledger_rows(root: Path) -> list[SourceMember]:
    """Every identified row across the merged ledger slices."""
    members: list[SourceMember] = []
    for path in sorted(Path(root).glob("*.md")):
        for line in path.read_text(errors="ignore").splitlines():
            found = _LEDGER_ID.match(line)
            if not found:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            members.append(SourceMember(
                population="ledger-row",
                identifier=found["id"],
                title=cells[1] if len(cells) > 1 else "",
                origin=path.name,
            ))
    return members


def read_catalogue_rows(path: Path) -> list[SourceMember]:
    """Every feature row in FEATURES.md, identified by section and name."""
    members: list[SourceMember] = []
    section = ""
    for line in Path(path).read_text(errors="ignore").splitlines():
        heading = _HEADING.match(line)
        if heading:
            section = heading["title"].split(".")[0].strip()
            continue
        row = _TABLE_ROW.match(line)
        if not row:
            continue
        name = re.sub(r"[*`]", "", row["first"]).strip()
        if not name or name.lower() in {"feature", "idea", "capability"}:
            continue
        members.append(SourceMember(
            population="catalogue-row",
            identifier=f"FEATURES §{section} {name}",
            title=name,
            origin=Path(path).name,
        ))
    return members


def read_design_sections(roots: list[Path]) -> list[SourceMember]:
    """Every `##` and `###` heading across the design corpus."""
    members: list[SourceMember] = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.md")):
            for line in path.read_text(errors="ignore").splitlines():
                heading = _HEADING.match(line)
                if not heading:
                    continue
                title = heading["title"].strip()
                members.append(SourceMember(
                    population="design-section",
                    identifier=f"{path.name}#{title}",
                    title=title,
                    origin=path.name,
                ))
    return members


def reconcile(members: list[SourceMember],
              slices: list["PlanSlice"],
              decisions: dict[str, tuple[str, str]]) -> list[Resolution]:
    """Classify every member. A member with no home is unassigned, loudly."""
    named: set[str] = {source
                       for plan_slice in slices
                       for row in plan_slice.rows
                       for source in row.sources}

    resolutions: list[Resolution] = []
    for member in members:
        decided = decisions.get(member.identifier)
        if decided is not None:
            outcome, detail = decided
            if outcome not in OUTCOMES or outcome in {"assigned", "unassigned"}:
                raise ValueError(
                    f"{member.identifier}: {outcome!r} is invented; a written "
                    f"decision may only be declined, blocked or prose")
            resolutions.append(Resolution(member, outcome, detail))
            continue

        if any(member.identifier in source or source in member.identifier
               for source in named):
            resolutions.append(Resolution(member, "assigned", ""))
            continue

        resolutions.append(Resolution(member, "unassigned", ""))
    return resolutions


def summarise(resolutions: list[Resolution]) -> dict[str, dict[str, int]]:
    """Counts per population per outcome. Every member counted exactly once."""
    counts: dict[str, dict[str, int]] = {}
    for resolution in resolutions:
        population = counts.setdefault(resolution.member.population, {})
        population[resolution.outcome] = population.get(resolution.outcome, 0) + 1
    return counts
```

Add the import that the annotation needs, at the top of the file beside the others:

```python
from plan.master_plan import PlanSlice
```

and change the `reconcile` signature's quoted annotation to the real type.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reconcile_sources.py -v`
Expected: 9 passed. If the three real-population size assertions fail, **do not weaken the assertion** — report the counts you got, because a population smaller than the spec measured means the reader is dropping members.

- [ ] **Step 5: Print the real sweep and record the honest number**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path
from plan.master_plan import read_master_plan
from plan.reconcile_sources import *
led = read_ledger_rows(Path.home()/'research'/'ledger'/'merged')
cat = read_catalogue_rows(Path.home()/'research'/'FEATURES.md')
sec = read_design_sections([Path.home()/'research',
                            Path.home()/'trading-system'/'docs'/'superpowers'])
print('ledger', len(led), 'catalogue', len(cat), 'sections', len(sec))
res = reconcile(led+cat+sec, [], {})
for pop, counts in summarise(res).items():
    print(pop, counts)
"
```

**Paste the output.** The unassigned count will be large. That is the correct first result and must not be hidden or softened.

- [ ] **Step 6: Commit and push**

```bash
git add src/plan/reconcile_sources.py tests/test_reconcile_sources.py
git commit -m "feat: sweep design sections too - the population nothing reconciled"
git push origin main
```

---

### Task 7: Write the spine

**Files:**
- Create: `docs/AJIT-MASTER-PLAN.md`
- Create: `docs/reconciliation-decisions.json`
- Create: `tests/test_real_spine_parses.py`

**Interfaces:**
- Consumes: the parser from Task 4 and the row format it defines.
- Produces: the document every later task and every future session reads.

- [ ] **Step 1: Write the failing test**

Create `tests/test_real_spine_parses.py`:

```python
"""The real spine must satisfy its own parser, or it is not a contract.

A plan document that does not parse is a plan document nobody can measure
against, which is the state `2026-08-09-full-build-master-plan.md` is in.
"""
from pathlib import Path

from plan.master_plan import read_master_plan
from plan.rulings import read_rulings
from plan.scope_coverage import cover_ruling

REPO = Path(__file__).resolve().parents[1]
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"


def test_the_real_spine_parses_into_the_six_slices():
    slices = read_master_plan(SPINE)
    keys = [s.key for s in slices]
    assert keys == ["slice-0", "spot-bot", "perp-bot", "dated-bot",
                    "options-bot", "slice-5"]


def test_every_row_in_the_real_spine_names_a_ruling_that_exists():
    known = {r.id for r in read_rulings(REPO / "docs" / "rulings.json")}
    for plan_slice in read_master_plan(SPINE):
        for row in plan_slice.rows:
            unknown = set(row.satisfies) - known
            assert not unknown, f"{row.id} cites unknown ruling(s): {unknown}"


def test_slice_zero_has_rows_and_the_segment_slices_are_honestly_empty():
    slices = {s.key: s for s in read_master_plan(SPINE)}
    assert len(slices["slice-0"].rows) >= 10
    # The segment slices carry no rows yet, deliberately: their inventory comes
    # from the reconciliation sweep and is reviewed with the user first. An
    # empty slice reporting 0/0 is the honest board for unplanned work.
    assert slices["spot-bot"].rows == ()


def test_the_per_segment_rulings_report_zero_of_four_today():
    slices = read_master_plan(SPINE)
    rulings = {r.id: r for r in read_rulings(REPO / "docs" / "rulings.json")}
    coverage = cover_ruling(rulings["RL-009"], slices)
    assert coverage.need == 4
    assert not coverage.resolved, (
        "the examination-hall ruling has no segment rows yet and must say so")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_real_spine_parses.py -v`
Expected: `FileNotFoundError` for `docs/AJIT-MASTER-PLAN.md`.

- [ ] **Step 3: Write `docs/AJIT-MASTER-PLAN.md`**

Header, verbatim:

```markdown
# AJIT MASTER PLAN

**This file is the top of the authority chain for WHAT TO BUILD NEXT.** Where the goal document,
`ARCHITECTURE.md`, `DECISIONS.md`, `FEATURES.md` or any plan disagrees with this file about order,
this file wins. They keep what they own: the goal document keeps the goal and the §1a intelligence
standard, `ARCHITECTURE.md` keeps structure, `DECISIONS.md` keeps the record of why, `FEATURES.md`
keeps the capability list, and `~/research/ledger/` keeps the research corpus.

`docs/superpowers/plans/2026-08-09-full-build-master-plan.md` is **superseded by this file** and
retained as the record of what was decided on 2026-08-09.

**Nothing in this file reports state.** Every row's `state:` field names a probe. The measured
values live at `~/research/dashboard/ajit-master-plan.html`, which is generated on every boards
pass and never committed (Rule 8, Rule 9). If you find a status value typed below, the parser is
broken — it is supposed to refuse.

Design: `docs/superpowers/specs/2026-08-17-ajit-master-plan-design.md`.
Rulings: `docs/rulings.json` — 21 rulings, verbatim, dated.

## Gates

**BUILT** — every row in the slice lit, the bot running under its own supervisor, journalling fills,
`integrity.unsupported_claims` reporting zero for it, its board tile measured. *The next slice starts
here.*

**EARNING** — a strategy of that bot promoted through the full gate stack on observed history.
Tracked per bot. **Never blocks the build**, because that clock is calendar time on a record that
started 2026-08-08 and cannot be shortened.
```

Then the six slice headings, in order, each `## SLICE <key> — <title>`:

`slice-0` — FOUNDATION & GOVERNANCE · `spot-bot` — SPOT BOT · `perp-bot` — PERP BOT · `dated-bot` — DATED FUTURES BOT · `options-bot` — OPTIONS BOT · `slice-5` — ALLOCATOR, PORTFOLIO RISK, BRAINS 2-3

Under `## SLICE slice-0 — FOUNDATION & GOVERNANCE`, write one row per task in this plan, in the eight-field format. Two worked examples to copy the shape from — write the remaining rows the same way, one per task, using that task's own files, probe and acceptance sentence:

```
### SL-01
  slice:      slice-0
  does:       copy the store to GCS beside the raw tape, on every offload pass
  satisfies:  RL-020
  sources:    spec §2.1 · DECISIONS §15 · Rule 9
  depends on: none
  probe:      probe_store_offloaded
  accepts:    gs://capture-raw-data4134/store/funding lists non-empty, and the
              offload script names store beside raw
  state:      measured by probe_store_offloaded

### SL-04
  slice:      slice-0
  does:       parse the spine into slices and eight-field rows, refusing a typed state
  satisfies:  RL-021 RL-007
  sources:    spec §4 · spec §1
  depends on: SL-03
  probe:      probe_spine_parses
  accepts:    the real spine parses into six slices and every row cites a ruling
              that exists in the register
  state:      measured by probe_spine_parses
```

Under each of the four segment slice headings and `slice-5`, write **only** the heading plus this line — no rows:

```
*Row inventory pending the reconciliation sweep, reviewed with the user before this slice starts
(spec §12). An empty slice reports 0/0, which is the honest board for unplanned work.*
```

- [ ] **Step 4: Write `docs/reconciliation-decisions.json`**

Start it with the decisions that are already true and defensible, and nothing else. Every entry is a human decision, so do not bulk-fill it:

```json
{
  "_what_this_is": "Written resolutions for source members that are NOT assigned to a plan row. Only three outcomes may appear here - declined, blocked, prose - because 'assigned' is measured from the spine and 'unassigned' is what is left. Each carries its reason, because a decline without a reason is a forget wearing better clothes.",
  "decisions": {
    "2026-08-08-final-project-goal-design.md#0. Prime directive": ["prose", "an objective, not a module"],
    "2026-08-08-final-project-goal-design.md#Provenance": ["prose", "a record of how the document was made"]
  }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_real_spine_parses.py -v`
Expected: 4 passed. If `test_every_row_in_the_real_spine_names_a_ruling_that_exists` fails, the row cites a ruling id that is not in the register — fix the row, or add the ruling if it is real.

- [ ] **Step 6: Commit and push**

```bash
git add docs/AJIT-MASTER-PLAN.md docs/reconciliation-decisions.json tests/test_real_spine_parses.py
git commit -m "feat: AJIT MASTER PLAN - the spine, and slices that admit they are empty"
git push origin main
```

---

### Task 8: One probe per ruling, and no probe means NOT MEASURED

**Files:**
- Create: `src/statuswall/ruling_conformance.py`
- Create: `tests/test_ruling_conformance.py`

**Interfaces:**
- Consumes: `Ruling`, `read_rulings` from `plan.rulings`; `Coverage`, `cover_ruling` from `plan.scope_coverage`; `ProbeResult`, `NOT_MEASURED`, `OK`, `PARTIAL`, `NOT_BUILT`, `STATE_LABEL`, `SEVERITY_ORDER` from `statuswall.evidence`; `render_staleness_banner` from `statuswall.staleness_banner`.
- Produces:
  - `probe_store_offloaded(bucket: str) -> ProbeResult`
  - `probe_memory_reachable() -> ProbeResult`
  - `probe_paper_engine_running(state_dir: Path) -> ProbeResult`
  - `probe_enforcement_live(hooks_dir: Path, settings: Path) -> ProbeResult`
  - `PROBES: dict[str, callable]` mapping probe name to the callable
  - `assess_rulings(rulings, slices) -> list[tuple[Ruling, Coverage, ProbeResult]]`
  - `render_ruling_conformance_page(assessments, generated_at_ns: int) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ruling_conformance.py`:

```python
"""A ruling with no probe is NOT MEASURED, and that must be visible.

The board's purpose is to make an unhonoured ruling loud. So the two things
worth defending are: a ruling naming no probe never renders a lit state, and a
ruling whose probe exists renders whatever that probe measured - never
whatever the ruling hoped for.
"""
from pathlib import Path

from plan.rulings import Ruling
from plan.master_plan import PlanRow, PlanSlice
from statuswall.evidence import NOT_MEASURED, OK, ProbeResult
from statuswall.ruling_conformance import (
    PROBES,
    assess_rulings,
    probe_memory_reachable,
    render_ruling_conformance_page,
)


def _ruling(rid="RL-001", probe=None, scope="system") -> Ruling:
    return Ruling(id=rid, date="2026-08-01", session="s", verbatim="said",
                  means="meant", recorded_in=(), probe=probe, scope=scope)


def _slices() -> list[PlanSlice]:
    row = PlanRow(id="SL-01", slice_key="slice-0", does="d", satisfies=("RL-001",),
                  sources=("spec §1",), depends_on=(), probe="probe_memory_reachable",
                  accepts="a", decided=None)
    return [PlanSlice("slice-0", "SLICE 0", (row,))]


def test_a_ruling_with_no_probe_is_not_measured_never_lit():
    _, _, result = assess_rulings([_ruling(probe=None)], _slices())[0]
    assert result.state == NOT_MEASURED
    assert "no probe" in result.detail.lower()


def test_a_ruling_naming_an_unknown_probe_is_not_measured_and_says_which():
    _, _, result = assess_rulings([_ruling(probe="probe_that_does_not_exist")], _slices())[0]
    assert result.state == NOT_MEASURED
    assert "probe_that_does_not_exist" in result.detail


def test_a_known_probe_supplies_the_state_rather_than_the_ruling():
    ruling = _ruling(probe="probe_memory_reachable")
    _, _, result = assess_rulings([ruling], _slices())[0]
    assert result.state in {s for s in PROBES and {OK, NOT_MEASURED}} or result.state
    assert result.proof, "every state must carry its proof (Rule 8)"


def test_coverage_travels_with_the_ruling_so_a_partial_scope_is_visible():
    ruling = _ruling(rid="RL-009", scope="per-segment")
    _, coverage, _ = assess_rulings([ruling], _slices())[0]
    assert coverage.need == 4
    assert not coverage.resolved


def test_the_page_renders_every_ruling_and_names_the_unmeasured_ones():
    assessments = assess_rulings([_ruling(), _ruling(rid="RL-002")], _slices())
    page = render_ruling_conformance_page(assessments, generated_at_ns=1)
    assert "RL-001" in page and "RL-002" in page
    assert "NOT MEASURED" in page.upper()


def test_memory_probe_reports_what_it_found_either_way():
    result = probe_memory_reachable()
    assert result.state in {OK, NOT_MEASURED} or result.state
    assert result.proof
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ruling_conformance.py -v`
Expected: collection error — `No module named 'statuswall.ruling_conformance'`.

- [ ] **Step 3: Write the implementation**

Create `src/statuswall/ruling_conformance.py`. The probes below are the four that can be honestly measured today; every other ruling names `null` and renders NOT MEASURED, which is the correct board for work that does not exist.

```python
"""One board row per human ruling, measured or explicitly not measured.

A ruling honoured in a document and not in running code is the failure this
board exists to catch. RL-011 was given 2026-08-03 and is 2/22 built; nothing
on any board said so, because no board knew the ruling existed.

Two numbers per ruling, and they answer different questions. COVERAGE asks
whether the plan has rows for it, per its scope - four of four segments,
twelve of twelve brains. STATE asks whether the running system does it. A
ruling can be fully covered and still fail its probe; that is not a
contradiction, it is the difference between planned and true.

A ruling naming no probe renders NOT MEASURED and never green (Rule 8).
"""
from __future__ import annotations

import html
import json
import subprocess
import time
from pathlib import Path

from plan.master_plan import PlanSlice
from plan.rulings import Ruling
from plan.scope_coverage import Coverage, cover_ruling
from statuswall.evidence import (
    DEGRADED, NOT_MEASURED, OK, PARTIAL, ProbeResult, STATE_LABEL,
)
from statuswall.staleness_banner import render_staleness_banner


def probe_store_offloaded(bucket: str = "gs://capture-raw-data4134") -> ProbeResult:
    """RL-020: the observed record must exist somewhere other than this disk."""
    listing = subprocess.run(
        ["gcloud", "storage", "ls", f"{bucket}/store/funding/"],
        capture_output=True, text=True, timeout=120)
    if listing.returncode != 0:
        return ProbeResult(DEGRADED,
                           "the store prefix is absent from the bucket",
                           f"gcloud storage ls {bucket}/store/funding/ -> "
                           f"exit {listing.returncode}")
    lines = [l for l in listing.stdout.splitlines() if l.strip()]
    if not lines:
        return ProbeResult(DEGRADED, "the store prefix exists and is empty",
                           f"{bucket}/store/funding/ listed 0 objects")
    return ProbeResult(OK, f"{len(lines)} object(s) under store/funding",
                       f"{bucket}/store/funding/ listed {len(lines)} object(s)")


def probe_memory_reachable() -> ProbeResult:
    """RL-016: the memory carrying 'interview, do not assume' must be loadable."""
    generic = Path.home() / ".claude" / "projects" / "-" / "memory"
    named = Path.home() / ".claude" / "projects" / "-home-anushadudekula71" / "memory"
    if not named.is_dir():
        return ProbeResult(NOT_MEASURED, "no project memory directory found",
                           str(named))
    generic_count = len(list(generic.glob("*.md"))) if generic.is_dir() else 0
    named_count = len(list(named.glob("*.md")))
    if generic_count < named_count:
        return ProbeResult(
            DEGRADED,
            f"a session started from / sees {generic_count} of {named_count} memories",
            f"{generic} vs {named}")
    return ProbeResult(OK, f"{named_count} memories reachable from either key",
                       f"{generic} -> {generic_count}, {named} -> {named_count}")


def probe_paper_engine_running(
    state_dir: Path = Path.home() / "capture" / "paper" / "forward",
) -> ProbeResult:
    """RL-005 and RL-017: paper trading is the experimentation ground."""
    heartbeat = state_dir / "heartbeat.json"
    if not heartbeat.is_file():
        return ProbeResult(NOT_MEASURED, "no heartbeat written", str(heartbeat))
    beat = json.loads(heartbeat.read_text())
    age_s = (time.time_ns() - int(beat["written_at_ns"])) / 1e9
    detail = (f"strategy {beat['strategy']!r}, edge claim "
              f"{bool(beat.get('makes_edge_claim'))}, {beat.get('fills', 0)} fill(s), "
              f"heartbeat {age_s:.0f}s old")
    proof = str(heartbeat)
    if age_s > 600:
        return ProbeResult(DEGRADED, detail, proof)
    if not beat.get("makes_edge_claim", False):
        # Running, honestly, on a signal that claims nothing. PARTIAL is the
        # correct state: the plumbing works and the strategy is not a strategy.
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def probe_enforcement_live(
    hooks_dir: Path = Path.home() / ".claude" / "hooks",
    settings: Path = Path.home() / ".claude" / "settings.json",
) -> ProbeResult:
    """RL-021: the enforcement layer is a hook that fires, not a paragraph."""
    required = ("require-plan-row.sh", "inject-plan-index.sh")
    present = [name for name in required if (hooks_dir / name).is_file()]
    registered = settings.read_text() if settings.is_file() else ""
    wired = [name for name in required if name in registered]
    detail = f"{len(present)}/2 hook scripts present, {len(wired)}/2 registered"
    proof = f"{hooks_dir} and {settings}"
    if len(wired) == len(required) == len(present):
        return ProbeResult(OK, detail, proof)
    if present or wired:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


PROBES = {
    "probe_store_offloaded": probe_store_offloaded,
    "probe_memory_reachable": probe_memory_reachable,
    "probe_paper_engine_running": probe_paper_engine_running,
    "probe_enforcement_live": probe_enforcement_live,
}


def assess_rulings(
    rulings: list[Ruling], slices: list[PlanSlice],
) -> list[tuple[Ruling, Coverage, ProbeResult]]:
    """Coverage from the plan, state from the probe, neither from the ruling."""
    assessments = []
    for ruling in rulings:
        coverage = cover_ruling(ruling, slices)
        if ruling.probe is None:
            result = ProbeResult(
                NOT_MEASURED, "no probe is named for this ruling", "docs/rulings.json")
        elif ruling.probe not in PROBES:
            result = ProbeResult(
                NOT_MEASURED,
                f"{ruling.probe} is named but not implemented",
                "statuswall/ruling_conformance.PROBES")
        else:
            try:
                result = PROBES[ruling.probe]()
            except Exception as failure:  # a probe that dies measures nothing
                result = ProbeResult(NOT_MEASURED,
                                     f"{ruling.probe} raised {failure!r}",
                                     ruling.probe)
        assessments.append((ruling, coverage, result))
    return assessments


def render_ruling_conformance_page(assessments, generated_at_ns: int) -> str:
    """One row per ruling: what was said, how covered, what is true."""
    rows = []
    for ruling, coverage, result in assessments:
        label = STATE_LABEL.get(result.state, result.state).upper()
        rows.append(
            "<tr>"
            f"<td class='id'>{html.escape(ruling.id)}</td>"
            f"<td class='date'>{html.escape(ruling.date)}</td>"
            f"<td class='said'>{html.escape(ruling.verbatim[:180])}</td>"
            f"<td class='scope'>{html.escape(ruling.scope)}</td>"
            f"<td class='cover'>{coverage.have}/{coverage.need}</td>"
            f"<td class='state s-{html.escape(result.state)}'>{html.escape(label)}</td>"
            f"<td class='detail'>{html.escape(result.detail)}</td>"
            f"<td class='proof'>{html.escape(result.proof)}</td>"
            "</tr>")

    banner = render_staleness_banner(generated_at_ns)
    return (
        "<h1>Ruling conformance</h1>"
        f"{banner}"
        "<table><thead><tr>"
        "<th>Ruling</th><th>Date</th><th>What was said</th><th>Scope</th>"
        "<th>Covered</th><th>State</th><th>Detail</th><th>Proof</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p class='foot'>Covered counts plan rows against the ruling's scope. "
        "State is measured by the named probe. A ruling with no probe reads "
        "NOT MEASURED and is never green.</p>")
```

- [ ] **Step 4: Check `render_staleness_banner`'s real signature before running**

Run: `grep -n "def render_staleness_banner" -A6 src/statuswall/staleness_banner.py`

If it takes different parameters than `(generated_at_ns)`, adapt the call — do not change the existing function, which four boards already use.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ruling_conformance.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit and push**

```bash
git add src/statuswall/ruling_conformance.py tests/test_ruling_conformance.py
git commit -m "feat: one board row per ruling, and no probe reads NOT MEASURED"
git push origin main
```

---

### Task 9: The plan board renders on the boards pass

**Files:**
- Create: `src/statuswall/master_plan_board.py`
- Create: `tests/test_master_plan_board.py`
- Modify: `src/statuswall/cli.py` — render both new pages beside `build-progress.html`

**Interfaces:**
- Consumes: everything from Tasks 3–8.
- Produces: `render_master_plan_page(slices, results, resolutions, generated_at_ns) -> str`; `active_slice(slices, results) -> str`; `next_row(slices, results) -> PlanRow | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_master_plan_board.py`:

```python
"""The board answers 'how much is done and what is next' without being asked.

The user asked that question six times between 2026-08-08 and 2026-08-17. The
board's job is to make it answer itself: counts per slice, the active slice,
the single next row, and the unassigned totals - all measured, no estimates.
"""
from plan.master_plan import PlanRow, PlanSlice
from plan.reconcile_sources import Resolution, SourceMember
from statuswall.evidence import NOT_BUILT, OK, ProbeResult
from statuswall.master_plan_board import (
    active_slice, next_row, render_master_plan_page,
)


def _row(rid, probe="probe_x") -> PlanRow:
    return PlanRow(id=rid, slice_key="slice-0", does=f"{rid} does a thing",
                   satisfies=("RL-021",), sources=("spec §1",), depends_on=(),
                   probe=probe, accepts="it works", decided=None)


def _slices() -> list[PlanSlice]:
    return [PlanSlice("slice-0", "FOUNDATION", (_row("SL-01"), _row("SL-02"))),
            PlanSlice("spot-bot", "SPOT BOT", ())]


def test_the_active_slice_is_the_first_one_not_fully_built():
    results = {"SL-01": ProbeResult(OK, "d", "p"),
               "SL-02": ProbeResult(NOT_BUILT, "d", "p")}
    assert active_slice(_slices(), results) == "slice-0"


def test_the_next_row_is_the_first_unbuilt_row_of_the_active_slice():
    results = {"SL-01": ProbeResult(OK, "d", "p"),
               "SL-02": ProbeResult(NOT_BUILT, "d", "p")}
    assert next_row(_slices(), results).id == "SL-02"


def test_an_empty_slice_reports_zero_of_zero_rather_than_complete():
    results = {"SL-01": ProbeResult(OK, "d", "p"),
               "SL-02": ProbeResult(OK, "d", "p")}
    page = render_master_plan_page(_slices(), results, [], generated_at_ns=1)
    assert "0 / 0" in page, "an empty slice must not render as finished"


def test_the_page_publishes_the_unassigned_count_rather_than_hiding_it():
    member = SourceMember("design-section", "x.md#Something", "Something", "x.md")
    resolutions = [Resolution(member, "unassigned", "")]
    page = render_master_plan_page(_slices(), {}, resolutions, generated_at_ns=1)
    assert "UNASSIGNED" in page.upper()
    assert "1" in page


def test_a_row_with_no_probe_result_renders_not_measured():
    page = render_master_plan_page(_slices(), {}, [], generated_at_ns=1)
    assert "NOT MEASURED" in page.upper()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_master_plan_board.py -v`
Expected: collection error — `No module named 'statuswall.master_plan_board'`.

- [ ] **Step 3: Write the implementation**

Create `src/statuswall/master_plan_board.py` following the structure of `build_progress.py` — read that file first for the page skeleton, the CSS block and the staleness banner call, and reuse them rather than inventing a second visual language. The module must provide:

```python
def active_slice(slices, results) -> str:
    """The first slice with any row not measured BUILT or OK. Slices are strict."""

def next_row(slices, results):
    """The first row of the active slice that is not BUILT or OK, else None."""

def render_master_plan_page(slices, results, resolutions, generated_at_ns) -> str:
    """Slices with counts, the active slice, the next row, reconciliation totals."""
```

Required page content, because these are what make the board answer the user's question:

1. One line per slice: `key · title · <built> / <total>` — and `0 / 0` for an empty slice, never a completion mark.
2. **ACTIVE SLICE** named, and **NEXT ROW** named with its `does:` text.
3. A reconciliation block: per population, the count of `assigned · declined · blocked · prose · **unassigned**`, with unassigned styled as the failing colour.
4. Each row's state from `results[row.id]`, defaulting to `NOT_MEASURED` when absent.
5. No dates, no estimates, no percentages that are not a measured ratio.

- [ ] **Step 4: Wire both boards into the CLI**

Read `src/statuswall/cli.py:60-90` first, then add the two pages beside `build-progress.html` on the same measurement pass, writing to `ajit-master-plan.html` and `ruling-conformance.html` in the same output directory.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_master_plan_board.py -v`
Expected: 5 passed.

- [ ] **Step 6: Render for real and look at it**

```bash
.venv/bin/python -m statuswall.cli --out ~/research/dashboard/status-wall.html
ls -la ~/research/dashboard/ajit-master-plan.html ~/research/dashboard/ruling-conformance.html
```
Expected: both files exist and are non-trivial in size. **Paste the listing and the unassigned counts the page reports.**

- [ ] **Step 7: Commit and push**

```bash
git add src/statuswall/master_plan_board.py src/statuswall/cli.py tests/test_master_plan_board.py
git commit -m "feat: the plan board names the active slice and the next row"
git push origin main
```

---

### Task 10: The hook that refuses an unplanned module

**Files:**
- Create: `~/.claude/hooks/require-plan-row.sh`
- Create: `~/.claude/hooks/tests/test-require-plan-row.sh`
- Modify: `~/.claude/settings.json`

**Interfaces:**
- Consumes: `docs/AJIT-MASTER-PLAN.md` (Task 7).
- Produces: nothing importable. A PreToolUse gate.

**Behaviour, exactly:** on `Write` to a path under `trading-system/src/` that **does not already exist**, the hook reads the spine. If no row's `probe:` or `does:` line mentions the file's stem, it exits 2 with the reason on stderr. Existing files, tests, docs and everything outside `src/` pass through.

- [ ] **Step 1: Write the failing test**

Create `~/.claude/hooks/tests/test-require-plan-row.sh`:

```bash
#!/usr/bin/env bash
# The hook must refuse a NEW src/ module with no plan row, and must not get in
# the way of anything else. Exit 2 blocks; exit 1 does NOT - that asymmetry is
# the easiest way to write a hook that looks like it works and silently does
# not, so both directions are asserted here.
set -uo pipefail
HOOK="$HOME/.claude/hooks/require-plan-row.sh"
REPO="$HOME/trading-system"
pass=0; fail=0

check() {
    local name=$1 want=$2 payload=$3
    printf '%s' "$payload" | "$HOOK" >/dev/null 2>&1
    local got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass+1)); echo "ok   $name (exit $got)"
    else
        fail=$((fail+1)); echo "FAIL $name (want $want, got $got)"
    fi
}

check "new src module with no plan row is blocked" 2 \
  "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$REPO/src/plan/invented_thing.py\"}}"

check "new src module named by a plan row is allowed" 0 \
  "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$REPO/src/plan/master_plan.py\"}}"

check "a test file is never blocked" 0 \
  "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$REPO/tests/test_invented.py\"}}"

check "a doc is never blocked" 0 \
  "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$REPO/docs/notes.md\"}}"

check "a file outside the repo is never blocked" 0 \
  '{"tool_name":"Write","tool_input":{"file_path":"/tmp/scratch.py"}}'

check "a malformed payload does not block" 0 'not json at all'

echo "---"; echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
```

Make it executable: `chmod +x ~/.claude/hooks/tests/test-require-plan-row.sh`

- [ ] **Step 2: Run it to verify it fails**

Run: `~/.claude/hooks/tests/test-require-plan-row.sh`
Expected: every case FAILs — the hook does not exist yet, so the shell reports 127.

- [ ] **Step 3: Write the hook**

Create `~/.claude/hooks/require-plan-row.sh`:

```bash
#!/usr/bin/env bash
# Rule 4 enforcement for RL-021: a NEW module under trading-system/src/ needs a
# row in AJIT MASTER PLAN naming it.
#
# The escape hatch is ADDING THE ROW, which is the behaviour wanted rather than
# an obstacle to route around. Editing an existing file is never blocked, so
# ordinary work is untouched; this catches exactly one thing, which is building
# something no plan row asked for.
#
# exit 2 BLOCKS. exit 1 does not. PreToolUse fails closed by design - a wrong
# block here is merely annoying, and the reason prints on stderr.
set -uo pipefail

SPINE="$HOME/trading-system/docs/AJIT-MASTER-PLAN.md"

payload=$(cat)
path=$(printf '%s' "$payload" \
  | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")' 2>/dev/null)

# Anything we cannot read is not something we block on.
[ -z "$path" ] && exit 0
case "$path" in
    */trading-system/src/*) ;;
    *) exit 0 ;;
esac
[ -e "$path" ] && exit 0          # an edit to an existing module, not a new one
[ -f "$SPINE" ] || exit 0         # no plan yet is not the writer's fault

stem=$(basename "$path"); stem=${stem%.py}
if grep -q -- "$stem" "$SPINE"; then
    exit 0
fi

cat >&2 <<EOF
BLOCKED by require-plan-row.sh (RL-021).

  new module : $path
  plan       : $SPINE

No row in AJIT MASTER PLAN names "$stem". Building something the plan does not
ask for is the failure this hook exists to catch - RL-011 was ruled on
2026-08-03 and was 2/22 built two weeks later because work and plan drifted
apart with nothing measuring the gap.

To proceed: add the row. Eight fields, no placeholders -
  does / satisfies / sources / depends on / probe / accepts / state
and 'state:' must read 'measured by <probe>', never a status value.
EOF
exit 2
```

Make it executable: `chmod +x ~/.claude/hooks/require-plan-row.sh`

- [ ] **Step 4: Run the hook's test to verify it passes**

Run: `~/.claude/hooks/tests/test-require-plan-row.sh`
Expected: `6 passed, 0 failed`. **Paste the output.**

- [ ] **Step 5: Register it in settings.json**

Add to the existing `PreToolUse` array — do not replace the two entries already there:

```json
{
  "matcher": "Write",
  "hooks": [
    {
      "type": "command",
      "command": "/home/anushadudekula71/.claude/hooks/require-plan-row.sh",
      "timeout": 10
    }
  ]
}
```

- [ ] **Step 6: Verify the existing enforcement layer still fires**

Registered is not firing. Run all three checks:

```bash
printf '{"tool_input":{"command":"ls"}}' | ~/.claude/hooks/block-dangerous-bash.sh; echo "bash hook: $?"
python3 ~/.claude/hooks/tests/test-stop-hook.py
~/.claude/hooks/tests/test-require-plan-row.sh
```
Expected: `0`, the 12-case Stop suite green, and 6 passed. **Paste all three outputs.**

- [ ] **Step 7: Push the config repo**

```bash
bash ~/.claude/push-all-repos.sh
```

---

### Task 11: The session-start injection

**Files:**
- Create: `~/.claude/hooks/inject-plan-index.sh`
- Modify: `~/.claude/settings.json`

**Interfaces:**
- Consumes: `docs/rulings.json`, `docs/AJIT-MASTER-PLAN.md`.
- Produces: text on stdout, which the harness adds to the session's context.

- [ ] **Step 1: Write the hook**

Create `~/.claude/hooks/inject-plan-index.sh`:

```bash
#!/usr/bin/env bash
# RL-021: the plan must be PRESENT in every session, not merely findable.
#
# CLAUDE.md already said "search the ledger first" and it still failed, because
# prose that has to be remembered is prose that can be skipped. This prints the
# ruling list and the active slice into every session's context so the answer
# to "what did we decide" is already there before the question is asked.
#
# SessionStart output is context, not a gate: this hook must never block, so it
# exits 0 on every path.
set -uo pipefail

REPO="$HOME/trading-system"
RULINGS="$REPO/docs/rulings.json"
SPINE="$REPO/docs/AJIT-MASTER-PLAN.md"

[ -f "$RULINGS" ] || exit 0

echo "## AJIT MASTER PLAN — standing rulings (docs/rulings.json)"
echo
echo "These are the user's own decisions, verbatim. Do not re-derive them, do not"
echo "paraphrase them from memory, and do not build against a design that"
echo "contradicts one. A new ruling is written into this file the moment it is"
echo "given, before other work continues."
echo

python3 - "$RULINGS" <<'PY' 2>/dev/null
import json, sys
rulings = json.load(open(sys.argv[1]))["rulings"]
for r in rulings:
    if r.get("superseded_by"):
        continue
    said = " ".join(r["verbatim"].split())[:150]
    print(f"- **{r['id']}** ({r['date']}, {r.get('scope','?')}) — {said}")
PY

echo
if [ -f "$SPINE" ]; then
    echo "Plan: \`docs/AJIT-MASTER-PLAN.md\` is the top authority for what to build next."
    echo "Board: \`~/research/dashboard/ajit-master-plan.html\` carries the measured state."
else
    echo "Plan: \`docs/AJIT-MASTER-PLAN.md\` does not exist yet."
fi
exit 0
```

Make it executable: `chmod +x ~/.claude/hooks/inject-plan-index.sh`

- [ ] **Step 2: Run it directly and read what it prints**

Run: `~/.claude/hooks/inject-plan-index.sh; echo "exit: $?"`
Expected: exit 0, and a bullet per non-superseded ruling with its id, date, scope and first 150 characters. **Paste the output.** RL-015 must be absent — it is superseded by RL-017.

- [ ] **Step 3: Register it**

Add a `SessionStart` entry to `~/.claude/settings.json`:

```json
"SessionStart": [
  {
    "hooks": [
      {
        "type": "command",
        "command": "/home/anushadudekula71/.claude/hooks/inject-plan-index.sh",
        "timeout": 10
      }
    ]
  }
]
```

- [ ] **Step 4: Verify it is registered and the file still parses**

Run: `python3 -c "import json; d=json.load(open('$HOME/.claude/settings.json')); print(list(d['hooks'].keys()))"`
Expected: `['PreToolUse', 'Stop', 'SessionStart']`. A `json.JSONDecodeError` here means the settings file is broken — fix it before doing anything else, because a broken settings file disables every hook.

- [ ] **Step 5: Push the config repo**

```bash
bash ~/.claude/push-all-repos.sh
```

---

### Task 12: The paper engine stops re-priming the archive

Measured 2026-08-17: after boot the engine fed ~1,975,345 archived events before its first poll — about eleven minutes with nothing trading, against a market that never closes and a user ruling that the box stays up 24/7. With four bots that becomes four cold starts.

**Files:**
- Modify: `src/paper/forward_engine.py`
- Modify: `src/paper/forward_journal.py` (only if the prime marker belongs there — read both first)
- Test: `tests/test_forward_engine_cold_start.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a persisted prime marker so a restart resumes rather than re-primes.

- [ ] **Step 1: Read `prime()` before changing it**

Run: `grep -n "def prime" -A40 src/paper/forward_engine.py`

`prime()` already exists and already marks the archive seen without trading it — that fix landed 2026-08-15 after a fresh journal replayed 3,494 archived bars into 1,074 fills. The defect now is different: the marking is **not persisted**, so every restart redoes it. Understand the existing mechanism before adding to it.

- [ ] **Step 2: Write the failing test**

Create `tests/test_forward_engine_cold_start.py`:

```python
"""A restart must not re-read the whole archive to learn what it already knew.

Measured 2026-08-17: ~1.98M events primed on every boot, about eleven minutes
of an engine that is up but not trading. Against a 24/7 market and four bots
to come, the prime has to persist.

The property being defended is narrow and important: resuming must not cause a
single archived event to be TRADED. The 2026-08-15 defect - a fresh journal
replaying 3,494 archived bars into 1,074 fills at prices days old - is the
thing a resume could quietly reintroduce.
"""
from pathlib import Path

import pytest

from paper import forward_engine


def test_a_prime_marker_is_written_and_names_the_watermark(tmp_path):
    marker = forward_engine.write_prime_marker(tmp_path, last_availability_ns=123)
    assert marker.is_file()
    assert forward_engine.read_prime_marker(tmp_path) == 123


def test_no_marker_means_a_full_prime_rather_than_trading_the_archive(tmp_path):
    assert forward_engine.read_prime_marker(tmp_path) is None


def test_a_marker_from_a_different_strategy_is_ignored(tmp_path):
    forward_engine.write_prime_marker(tmp_path, last_availability_ns=123,
                                      strategy="plumbing-momentum")
    assert forward_engine.read_prime_marker(tmp_path, strategy="other") is None


def test_a_corrupt_marker_falls_back_to_a_full_prime(tmp_path):
    (tmp_path / "prime-marker.json").write_text("{not json")
    assert forward_engine.read_prime_marker(tmp_path) is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_forward_engine_cold_start.py -v`
Expected: `AttributeError: module 'paper.forward_engine' has no attribute 'write_prime_marker'`.

- [ ] **Step 4: Implement the marker**

Add to `src/paper/forward_engine.py`:

```python
def write_prime_marker(state_dir: Path, last_availability_ns: int,
                       strategy: str = "") -> Path:
    """Record how far the archive was primed, so a restart resumes.

    The marker holds an AVAILABILITY time, not an event time: the engine's
    contract is that it never trades a row whose availability time is older
    than what it has already seen, and availability is the clock the store is
    gated on. Stamping the event time would let a late-arriving correction to
    an old bar look new.

    The strategy is part of the marker because a different strategy has not
    seen this archive - resuming its neighbour's watermark would silently skip
    every bar it should have primed on.
    """
    marker = Path(state_dir) / "prime-marker.json"
    marker.write_text(json.dumps({
        "last_availability_ns": int(last_availability_ns),
        "strategy": strategy,
    }))
    return marker


def read_prime_marker(state_dir: Path, strategy: str = "") -> int | None:
    """The watermark to resume from, or None meaning prime the whole archive.

    Every failure path returns None. A corrupt or foreign marker must cost a
    slow start, never a wrong one - the 2026-08-15 defect turned archived bars
    into 1,074 forward fills, and that is the direction this must never fail.
    """
    marker = Path(state_dir) / "prime-marker.json"
    if not marker.is_file():
        return None
    try:
        held = json.loads(marker.read_text())
    except (ValueError, OSError):
        return None
    if strategy and held.get("strategy", "") != strategy:
        return None
    value = held.get("last_availability_ns")
    return int(value) if isinstance(value, int) else None
```

Add `import json` and `from pathlib import Path` at the top if they are not already imported.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_forward_engine_cold_start.py -v`
Expected: 4 passed.

- [ ] **Step 6: Wire the marker into the poll loop**

In the engine's poll loop: read the marker at startup and pass the watermark as the lower bound of the prime read; write the marker after every successful poll with the newest availability time fed. Keep `prime()`'s existing behaviour intact for the no-marker case.

- [ ] **Step 7: Verify against the real engine**

```bash
.venv/bin/python -m pytest tests/test_forward_engine_cold_start.py tests/test_live_bars.py -q
```
Then restart the supervisor and time the first poll:

```bash
pkill -f "paper.forward_engine" || true
sleep 2
tail -f ~/capture/paper/forward/engine.log
```
Expected: the first `poll 1:` line appears in well under the ~11 minutes measured on 2026-08-17. **Paste the timestamp of the restart and of the first poll line.** If it is not faster, say so — an unverified performance fix is not a fix.

- [ ] **Step 8: Commit and push**

```bash
git add src/paper/forward_engine.py tests/test_forward_engine_cold_start.py
git commit -m "fix: a restart re-primed 1.98M events, against a 24/7 mandate"
git push origin main
```

---

### Task 13: Retire the old plan and point everything at the new one

**Files:**
- Modify: `docs/superpowers/plans/2026-08-09-full-build-master-plan.md`
- Modify: `docs/superpowers/specs/2026-08-08-final-project-goal-design.md` — new §3b, amended §5a
- Modify: `~/research/DECISIONS.md` — new §15
- Modify: `CLAUDE.md` (repo) and `~/.claude/CLAUDE.md` — index table
- Test: `tests/test_authority_chain_is_consistent.py`

**Interfaces:**
- Consumes: everything. This is the task that makes the new plan the one people read.

- [ ] **Step 1: Write the failing test**

Create `tests/test_authority_chain_is_consistent.py`:

```python
"""Two documents claiming to say what to build next is the original defect.

`2026-08-09-full-build-master-plan.md` put paper trading last while paper
trading ran. Whatever else is true, exactly one file may claim that authority,
and the superseded one must say so in its own text.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "docs" / "superpowers" / "plans" / "2026-08-09-full-build-master-plan.md"
NEW = REPO / "docs" / "AJIT-MASTER-PLAN.md"
GOAL = REPO / "docs" / "superpowers" / "specs" / "2026-08-08-final-project-goal-design.md"


def test_the_old_master_plan_declares_itself_superseded():
    body = OLD.read_text()
    assert "SUPERSEDED" in body.upper()
    assert "AJIT-MASTER-PLAN" in body


def test_the_new_plan_claims_the_authority_explicitly():
    body = NEW.read_text()
    assert "top of the authority chain" in body.lower()


def test_the_goal_document_records_the_segment_bot_ruling_verbatim():
    body = GOAL.read_text()
    assert "3b" in body
    assert "each segment are like there own bots" in body, (
        "RL-019 must appear in the user's own words, not paraphrased")


def test_the_goal_document_amends_universe_scanning_to_a_per_bot_property():
    body = GOAL.read_text()
    section = body[body.find("## 5a."):body.find("## 6.")]
    assert "per-bot" in section or "per segment bot" in section, (
        "§5a called scanning a system property; RL-019 makes it a per-bot one")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_authority_chain_is_consistent.py -v`
Expected: 4 failed.

- [ ] **Step 3: Mark the old plan superseded**

Insert immediately below its `Status:` line:

```markdown
> **SUPERSEDED 2026-08-17 by `docs/AJIT-MASTER-PLAN.md`.** Retained as the record of what was
> decided on 2026-08-09, not as a plan to follow. Its phase J puts paper trading last; paper
> trading has been running since 2026-08-15 under RL-017, so this file no longer describes the
> order of work. Read AJIT MASTER PLAN for what to build next.
```

- [ ] **Step 4: Add §3b to the goal document**

Immediately after §3a and before `## 4. Capital is a dial`:

```markdown
## 3b. Each segment is its own bot — user ruling, 2026-08-17

> **"each segment are like there own bots with its own architecture, data features etc like this
> so make sure you do not assume and build different from what we discussed and fixed before"**

Recorded verbatim. Clarified with the user the same day: **fully separate bots.** Spot, perpetual
futures, dated futures and options each get their own engine process, own feature set, own models,
own risk gate, own paper journal, own board tile and own on/off switch. They share only the store,
the clock-gated reader, the cost engine and the promotion pipeline.

**This composes with RL-011 rather than replacing it.** RL-011 (2026-08-03) splits by *direction*
— BULL, BEAR, PROFIT-TAIL, each with its own data, features and architecture. This ruling splits by
*segment*. The user chose the nested reading: **four bots, three brains inside each, twelve brains
in total.** The bot boundary is the segment.

**The intelligence standard binds per brain.** Each of the twelve carries its own verdict on all
three axes of §1a — learning, reasoning, depth. 36 verdicts. Reading 0/36 on 2026-08-17.

Order of work, gates, and reconciliation: `docs/AJIT-MASTER-PLAN.md`.
```

- [ ] **Step 5: Amend §5a**

Add immediately under the §5a heading:

```markdown
> **Amended 2026-08-17 by RL-019.** This section calls universe-wide scanning *"a property of the
> system, not a strategy."* The user ruled it is a property of **each segment bot**: the spot bot
> runs its own monitor over the whole spot universe on spot data, and does not receive candidates
> from a shared screener. Everything below about the mechanism, the gate and the risks it creates
> stands unchanged — only the level it lives at moves. Scope: `per-segment`, and the plan's
> reconciliation reports it as 0/4 until all four bots have one.
```

- [ ] **Step 6: Add §15 to `~/research/DECISIONS.md`**

```markdown
## 15. The box stays up, and the plan became a file — 2026-08-17

**24/7 uptime, ruled by the user (RL-020).** The instance is not preemptible —
`scheduling/preemptible` reads FALSE — so the gaps were deliberate stops: 124.3h from
2026-08-10 to 08-15, and 14.1h from 08-16 17:41 to 08-17 07:50. Against a market that never
closes and an intraday mandate (§3a), a bot that is off part of the week cannot be honestly
judged. The cold-start cost is cut with it: ~1.98M events re-primed on every boot, about eleven
minutes of an engine that is up and not trading.

**AJIT MASTER PLAN (RL-021).** `docs/AJIT-MASTER-PLAN.md`, named by the user, is now the top
authority for what to build next. Written spine, generated status, five vertical slices — one
segment bot at a time, each ending with a bot that trades.

**What the sweep found.** 183 HIGH-rated ideas across the six IDEAS files: at most 69 traced into
`src/`, 106 with no trace anywhere. Eight of the ten the corpus itself named *"the ten to build
first"* are unbuilt. `src/strategy/` holds two modules, and neither is an opportunity monitor —
so the examination-hall ruling of 2026-08-02 is 0/4 by the scope rule, exactly as the design
predicted.
```

- [ ] **Step 7: Point both CLAUDE.md index tables at the new plan**

In the repo `CLAUDE.md`, make `docs/AJIT-MASTER-PLAN.md` the **first** row of the "Read before doing anything here" table, described as *"**The order.** What to build next, its slices, its gates, and every ruling — top of the authority chain."* In `~/.claude/CLAUDE.md`, add one line to Rule 9's repo table area noting the plan's location. Do not restructure either file — the counterweight rule warns against growing them.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_authority_chain_is_consistent.py -v`
Expected: 4 passed.

- [ ] **Step 9: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: at least 857 passed plus the ~40 added here. **Paste the count.** If anything regressed, fix it before committing — a green count that was not run is not a result.

- [ ] **Step 10: Commit and push**

```bash
git add docs CLAUDE.md tests/test_authority_chain_is_consistent.py
git commit -m "docs: AJIT MASTER PLAN takes the authority, and §3b records RL-019"
git push origin main
cd ~/research && git add -A && git commit -m "docs: DECISIONS §15 - 24/7, and what the sweep found" && git push
bash ~/.claude/push-all-repos.sh
```

---

## Self-Review

**Spec coverage.** §1 authority → Task 13. §1 written/generated split → Tasks 4, 7, 9. §2 slices → Task 7. §2 slice-0 contents → Tasks 1, 2, 3–11, 12; the shared bot framework is deliberately deferred to its own plan and named in the scope check at the top. §3 topology → Task 13 §3b; the bots themselves are slice 1+. §3 on/off switch → deferred with the framework. §3 per-brain intelligence → recorded in §3b (Task 13), built in slice 1+. §4 row anatomy → Task 4. §4 two state kinds → Task 4. §4 two gates → Task 7 header. §5 three populations → Task 6. §5 scope rule → Task 5. §5 default scope → Task 3 Step 4. §5 eight ruled scopes → Task 3 Step 4 table. §6 four mechanisms → Tasks 3, 10, 8, 11; memory fix → Task 2. §7 amendments → process, not code; recorded in the spine header. §8 progress reporting → Task 9. §9 files → the File Structure table. §10 tests 1–8 → Tasks 4 (1, 2, 6), 5 (3), 6 (4, 5), 10 (7), 3 (8). §11 definition of done → Task 13 Step 9 plus each task's verification step.

**Placeholder scan.** No TBD, no "add error handling", no "similar to Task N". Two steps deliberately say *read the existing file first* rather than showing code — Task 1 Step 1 and Task 9 Step 3 — because inventing a loop shape or a CSS block that must match an existing file is how a plan produces code that does not fit. Both name exactly what to read and what to match.

**Type consistency.** `PlanRow.slice_key` is used in Tasks 4, 5, 9 — same name throughout. `Coverage.have/need/missing/resolved` consistent across Tasks 5, 8, 9. `ProbeResult(state, detail, proof)` matches the existing `statuswall.evidence` dataclass exactly. `read_master_plan`, `read_rulings`, `cover_ruling`, `reconcile`, `summarise` are each defined once and called by the names defined. `probe_*` names in the Task 3 rulings table match the `PROBES` dict in Task 8: `probe_store_offloaded`, `probe_memory_reachable`, `probe_paper_engine_running`, `probe_enforcement_live`. Rulings citing probe names not in that dict render NOT MEASURED by design, which Task 8's second test asserts.

**One gap found and closed while reviewing:** the spec's §9 file list named `src/plan/master_plan.py` and `src/plan/reconcile_sources.py` but not `src/plan/rulings.py` or `src/plan/scope_coverage.py`. Splitting them out is a decomposition improvement, not a scope change — each file now has one responsibility and the scope arithmetic, which is the part most likely to be wrong, is testable alone.
