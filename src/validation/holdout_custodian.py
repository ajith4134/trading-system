"""Owns the untouched holdout and refuses to serve it. That refusal is the feature.

`FEATURES.md` §8: *"Owns the untouched holdout; commit hash + not-consumed flag,
CI blocks any run touching the range before freeze."*

**Why a component rather than a discipline.** Every other gate in this package
corrects for how many times the data was looked at. None of them can tell that the
*holdout* was among those looks. A holdout that has been peeked at is no longer a
holdout, and there is no statistic that detects the difference afterwards - the
peeked score and the honest score are the same number with different meanings. So
the only possible defence is a thing that says no.

The design follows the corpus's capstone protocol (ledger VX-049): search and
select on history only, **freeze**, forward-test once on the unseen range, and
report only the frozen score. Four properties make that enforceable:

**Sealed by default.** Reads inside the range raise until explicitly consumed.
Outside the range, nothing is restricted - a custodian that obstructs ordinary
research gets routed around, and a defence that has been routed around is worse
than no defence, because it still reads as present.

**Freeze and consume are separate acts.** Freezing declares the code final;
consuming is the one counted read. Collapsing them would make an accidental read
indistinguishable from the intended test.

**The commit hash is pinned at freeze.** Without it, freeze → edit → test is a
complete bypass that still reports a score from "frozen" code.

**Everything persists.** A seal a restart clears is a seal that lifts the moment
someone is frustrated, which is precisely when it matters.

**Stated limitation.** `ClockGatedReader` takes the custodian as an optional
argument, so a reader constructed without one is unguarded. This is a real gap, not
a covered case: full coverage needs the custodian injected at every construction
site, and the ones in the pricing path do not have it yet. Do not read the presence
of this module as proof the holdout cannot be read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

STATE_FILE = "holdout_custodian.json"
HISTORY_FILE = "holdout_custodian.ndjson"


class HoldoutSealed(PermissionError):
    """The holdout is not open for reading. Raised instead of returning rows."""


class HoldoutAlreadyConsumed(PermissionError):
    """The one permitted read has already happened.

    A second look is a second trial on data whose entire value was being untouched,
    and it would appear in no trial count - so nothing downstream could correct
    for it.
    """


class HoldoutTampered(PermissionError):
    """The freeze was re-made, or the code under test is not the code frozen."""


class HoldoutCustodian:
    """Guards a half-open time range `[holdout_start_ns, holdout_end_ns)`."""

    def __init__(self, root: Path, holdout_start_ns: int,
                 holdout_end_ns: int) -> None:
        if holdout_end_ns <= holdout_start_ns:
            raise ValueError(
                f"holdout range is empty or inverted: "
                f"[{holdout_start_ns}, {holdout_end_ns})")
        self._root = Path(root)
        self._start = int(holdout_start_ns)
        self._end = int(holdout_end_ns)

    # --- state ---------------------------------------------------------------

    def _state(self) -> dict:
        path = self._root / STATE_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Fail closed: an unreadable seal file is treated as sealed-and-
            # tampered, never as absent. "The file is corrupt" must not be a way
            # to reopen the holdout.
            raise HoldoutTampered(
                f"cannot read the custodian state at {path} ({exc}). Treated as "
                f"tampered rather than absent - a damaged seal file must not be a "
                f"way to reopen the holdout") from exc

    def _persist(self, state: dict) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / STATE_FILE
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _record(self, event: str, **fields) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        with (self._root / HISTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"event": event, "at_ns": time.time_ns(), **fields}) + "\n")

    def history(self) -> list[dict]:
        """Every freeze, consumption and refusal, in order.

        Refusals are included deliberately: a repeated attempt to read the holdout
        is the most interesting fact the system has about a pipeline, and a refusal
        that leaves no trace hides it.
        """
        path = self._root / HISTORY_FILE
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # --- queries -------------------------------------------------------------

    def is_frozen(self) -> bool:
        return bool(self._state().get("frozen_at_ns"))

    def is_consumed(self) -> bool:
        return bool(self._state().get("consumed_at_ns"))

    def frozen_commit(self) -> str | None:
        return self._state().get("frozen_commit")

    def covers(self, event_time_ns: int) -> bool:
        """Whether an instant falls inside the guarded range."""
        return self._start <= int(event_time_ns) < self._end

    # --- the refusal ---------------------------------------------------------

    def assert_readable(self, event_time_ns: int) -> None:
        """Permit a read at this instant, or raise.

        Outside the range this is free. Inside it, the holdout must have been both
        frozen and consumed.
        """
        if not self.covers(event_time_ns):
            return
        state = self._state()
        if state.get("consumed_at_ns"):
            return

        self._record("refused", requested_ns=int(event_time_ns),
                     frozen=bool(state.get("frozen_at_ns")))
        stage = ("frozen but not consumed" if state.get("frozen_at_ns")
                 else "not frozen")
        raise HoldoutSealed(
            f"timestamp {event_time_ns} is inside the sealed holdout "
            f"[{self._start}, {self._end}) and the custodian is {stage}. Select on "
            f"history only, then freeze(commit_hash=...) and consume(...) exactly "
            f"once - the holdout's only value is never having been seen, and a "
            f"peeked score is indistinguishable from an honest one")

    # --- the two state changes ----------------------------------------------

    def freeze(self, commit_hash: str, reason: str) -> None:
        """Declare the code final. Does not open the holdout."""
        if not commit_hash:
            raise ValueError("a freeze needs a commit hash to pin the code identity")
        state = self._state()
        if state.get("frozen_at_ns"):
            raise HoldoutTampered(
                f"already frozen at commit {state.get('frozen_commit')}. A second "
                f"freeze is a re-freeze after seeing something, which is the "
                f"manoeuvre this protocol exists to prevent")
        state.update({"frozen_at_ns": time.time_ns(),
                      "frozen_commit": commit_hash,
                      "freeze_reason": reason})
        self._persist(state)
        self._record("freeze", commit_hash=commit_hash, reason=reason)

    def consume(self, commit_hash: str, purpose: str) -> None:
        """Spend the single permitted read of the holdout."""
        state = self._state()
        if not state.get("frozen_at_ns"):
            raise HoldoutSealed(
                "cannot consume a holdout that was never frozen - freeze the "
                "selected code first, so the thing tested is pinned before the "
                "data is seen")
        if state.get("consumed_at_ns"):
            raise HoldoutAlreadyConsumed(
                f"the holdout was consumed at {state['consumed_at_ns']} for "
                f"{state.get('consume_purpose')!r}. Only the frozen score is "
                f"reported; a second look is an uncounted trial on data whose "
                f"value was being untouched")
        if commit_hash != state["frozen_commit"]:
            raise HoldoutTampered(
                f"frozen at commit {state['frozen_commit']} but consuming at "
                f"{commit_hash}. Freeze-then-edit-then-test is a complete bypass "
                f"that still reports a score from frozen code")
        state.update({"consumed_at_ns": time.time_ns(),
                      "consume_purpose": purpose})
        self._persist(state)
        self._record("consume", commit_hash=commit_hash, purpose=purpose)
