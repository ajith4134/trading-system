"""Cumulative N, structurally enforced — the number every other gate depends on.

`FEATURES.md` §8 asks for a trial count *"structurally impossible to evaluate
without incrementing"*, and an experiment ledger that **includes abandoned runs**.
`DECISIONS.md` §5 is the reason: with five years of daily data and 45+ tried
variations, the best selected strategy is more likely than not to have a true
out-of-sample Sharpe of zero.

Deflated Sharpe, MinBTL, PBO and FDR are all functions of N. Every one of them
gets *weaker* as N shrinks, so every way of losing a trial biases the system
toward promoting noise. That asymmetry sets the whole design:

**The trial is written before the evaluation runs.** A post-hoc counter cannot
count a crash, a timeout, or a run someone killed because the numbers looked
wrong - and those are the trials most likely to be lost and most important to
keep. Pre-registration is ported from `nse-crypto-bot-final`'s
`validation_holdout.py`, which appends every live prediction before its outcome
exists so cherry-picking is impossible.

**Enforcement is by construction, not convention.** The only way to get a score
is `evaluate()`, which increments first. The corpus's closest prior art
(`antioverfit.py`) exposes `register_backtest()` and asks the search lanes to
remember - and the ledger's own survey found no implementation anywhere with the
structural property.

**A damaged ledger raises.** `antioverfit.py` returns `{"backtests_run": 0}` on
any exception; N=0 disables multiple-testing correction entirely. When a bare
`except` produces the flattering answer, it has to go.

A torn final line is treated differently from corruption on purpose: a process
killed mid-append is a normal crash, and the trial it was writing really did run.
It counts.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LEDGER_FILE = "trial_registry.ndjson"

OUTCOME_PENDING = "pending"
OUTCOME_COMPLETE = "complete"
OUTCOME_ABANDONED = "abandoned"


class LedgerCorrupt(RuntimeError):
    """The trial ledger cannot be read, so N is unknown.

    Deliberately fatal. Every downstream gate divides its protection by N, so an
    unknown N must stop the search rather than default to a number that happens
    to make promotion easier.
    """


@dataclass(frozen=True)
class TrialSpec:
    """What was tried. Recorded verbatim so an N can be audited afterwards.

    `family` matters beyond bookkeeping: purge and embargo horizons differ by
    orders of magnitude between strategy families (`FEATURES.md` §8, VX-004), and
    a trial whose family is unknown cannot be validated correctly later.
    """

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


class TrialRegistry:
    """The append-only ledger of every candidate ever evaluated against the data."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._path = self._root / LEDGER_FILE

    # --- reading --------------------------------------------------------------

    def trials(self) -> list[dict]:
        """Every recorded trial, in registration order.

        A torn final line is counted but not parsed - see `_read_rows`.
        """
        rows, _ = self._read_rows()
        return rows

    def cumulative_count(self) -> int:
        """N. Total candidates ever evaluated against this dataset."""
        rows, torn = self._read_rows()
        return len(rows) + (1 if torn else 0)

    def trial_sharpes(self) -> list[float]:
        """The Sharpes actually produced, for the DSR's trial variance.

        Abandoned and pending trials are absent by design: they inflate N because
        they consumed a look at the data, but they produced no number, and
        inventing one would corrupt the variance the deflation depends on.
        """
        return [float(r["result"]["sharpe"]) for r in self.trials()
                if r.get("outcome") == OUTCOME_COMPLETE
                and isinstance(r.get("result"), dict)
                and r["result"].get("sharpe") is not None]

    def _read_rows(self) -> tuple[list[dict], bool]:
        """Parse and fold the ledger. Returns (trials, a_torn_final_line_was_seen).

        The file holds two kinds of record: a registration (which creates a trial)
        and a settlement (which reports its outcome). Both are appends, because an
        append-only ledger is what stops a disappointing result from being
        negotiated out of N later. Folding them on read is what keeps **N the
        number of trials rather than the number of lines** - the two diverge the
        moment anything settles, and a line count would silently double it.

        Only the *last* line may be torn: that is a process killed mid-append, and
        the trial it was writing really did run, so it counts. A malformed line
        anywhere else means the file was damaged or edited, and N can no longer be
        established from it at all.
        """
        if not self._path.exists():
            return [], False
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LedgerCorrupt(f"cannot read {self._path}: {exc}") from exc

        lines = [ln for ln in text.split("\n") if ln.strip()]
        by_id: dict[int, dict] = {}
        order: list[int] = []
        torn = False
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except ValueError as exc:
                if index == len(lines) - 1 and not text.endswith("\n"):
                    torn = True
                    continue
                raise LedgerCorrupt(
                    f"{self._path} line {index + 1} is not valid JSON ({exc}). N "
                    f"cannot be established, and defaulting it would weaken every "
                    f"multiple-testing gate downstream") from exc

            trial_id = record.get("trial_id")
            if trial_id is None:
                raise LedgerCorrupt(
                    f"{self._path} line {index + 1} has no trial_id, so it cannot "
                    f"be attributed to a trial")
            if trial_id in by_id:
                by_id[trial_id].update(
                    {k: v for k, v in record.items() if k != "trial_id"})
            else:
                by_id[trial_id] = dict(record)
                order.append(trial_id)
        return [by_id[i] for i in order], torn

    # --- writing --------------------------------------------------------------

    def pre_register(self, spec: TrialSpec) -> int:
        """Claim a trial id and put the row on disk *before* anything is run.

        Returns the trial id. The row lands with `outcome: pending`, so a process
        killed during evaluation leaves a trial that is counted and visibly
        unfinished rather than quietly assumed successful.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        row = {
            "trial_id": None,
            "name": spec.name,
            "family": spec.family,
            "params": spec.params,
            "registered_at_ns": time.time_ns(),
            "pid": os.getpid(),
            "outcome": OUTCOME_PENDING,
            "result": None,
            "error": None,
        }
        # The id is assigned and the row appended under one exclusive lock. Two
        # search processes share a dataset and therefore share an N; if both can
        # read the same count and both claim it, the ledger loses a row.
        with self._locked_ledger() as handle:
            rows, torn = self._read_rows()
            row["trial_id"] = len(rows) + (1 if torn else 0) + 1
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row["trial_id"]

    def evaluate(self, spec: TrialSpec,
                 evaluator: Callable[[TrialSpec], dict]) -> dict:
        """The only way to score a candidate. Counts it first, then runs it.

        Re-raises whatever the evaluator raised, having already recorded the
        trial as abandoned. The exception is the caller's to handle; the count is
        not negotiable.
        """
        trial_id = self.pre_register(spec)
        try:
            result = evaluator(spec)
        except BaseException as exc:
            self._settle(trial_id, OUTCOME_ABANDONED, None,
                         f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            raise
        self._settle(trial_id, OUTCOME_COMPLETE, result, None)
        return result

    def _settle(self, trial_id: int, outcome: str, result: dict | None,
                error: str | None) -> None:
        """Append the outcome for an already-counted trial.

        An append rather than an edit: the ledger is append-only so that a
        disappointing result cannot be negotiated out of N after the fact. The
        outcome row supersedes the pending one when the ledger is read back.
        """
        with self._locked_ledger() as handle:
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps({
                "trial_id": trial_id,
                "settled_at_ns": time.time_ns(),
                "outcome": outcome,
                "result": result,
                "error": error,
                "supersedes": trial_id,
            }) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _locked_ledger(self):
        """Open the ledger append-only under an exclusive advisory lock."""
        self._root.mkdir(parents=True, exist_ok=True)
        return _LockedFile(self._path)


class _LockedFile:
    """Context manager: an append-mode handle holding LOCK_EX for its lifetime."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def __enter__(self):
        self._handle = self._path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:                       # pragma: no cover - platform
            if exc.errno not in (errno.EINVAL, errno.ENOLCK, errno.EOPNOTSUPP):
                raise
        return self._handle

    def __exit__(self, *exc_info):
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:                              # pragma: no cover - platform
            pass
        self._handle.close()
        return False
