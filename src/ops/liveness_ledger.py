"""The system's record of its own absence.

Written after 2026-08-10 12:53 to 2026-08-15 17:13, during which this VM was off:
capture wrote nothing, `capture/raw/binance/` skips from 08-10 straight to 08-15,
the last bar build was five days stale, and every board went on serving a file
dated 08-10 without one word saying so. Nothing was broken in any way that
anything checked, because the system had no concept of its own absence.

**A watcher running on the box cannot report that the box is off.** That is not a
limitation to engineer around with more processes on the same box - it is the
shape of the problem. What a process here honestly can do is stamp that it was
alive, and on its next run compare that stamp against the clock and record the
hole. Retrospective, and stated as such: an outage that has ended is still the
fact that explains five missing days of tape, and without it the gap gets
rediscovered months later as a mysterious hole in a training set.

Two states this refuses to confuse:

**A cold start is not an outage.** A first run has no prior stamp. Treating that
as a gap "since the epoch" would put a fictional 56-year outage at the top of the
ledger on day one, and a ledger whose first row is fiction is one nobody trusts
by the second.

**A bounded gap, never a guess.** All that is known is that nothing wrote between
the last stamp and now. The row records exactly those two instants. The box may
have died one second after the last stamp or one second before the next; the
ledger does not pretend to know which, because a duration nobody measured is how
a five-day outage becomes "about a day" in someone's memory.

The threshold is a parameter. A supervisor stamping every 60s and a daily batch
job have different ideas of what silence means, and burying one number here would
make the ledger wrong for whichever caller did not write it.
"""
from __future__ import annotations

import json
import sys
import os
from dataclasses import dataclass
from pathlib import Path

STAMP_FILE = "liveness.json"
LEDGER_FILE = "outages.ndjson"

# Fifteen minutes. The health supervisor ticks every 60s, so this is fifteen
# missed ticks - long enough that a slow universe-wide read or a restart storm
# does not enter the ledger as an outage, short enough that a real one is caught
# on the first tick back.
DEFAULT_THRESHOLD_NS = 15 * 60 * 1_000_000_000


@dataclass(frozen=True)
class ColdStart:
    """First run on this machine. Not an outage, and must never be filed as one."""

    at_ns: int


@dataclass(frozen=True)
class Outage:
    """A hole in the record, bounded by the two instants that were observed.

    `is_reconstructed` marks a gap this module did NOT watch happen - one worked
    out afterwards from other evidence, such as boot logs and the dates present
    in the raw archive. It follows `store.bar_backfill`, which keeps
    reconstructed bars beside the observed ones and never inside them: a
    reconstruction is usable and it is not an observation, and a ledger that
    cannot tell them apart quietly upgrades inference into measurement.
    """

    last_seen_ns: int
    returned_ns: int
    is_reconstructed: bool = False
    evidence: str = ""

    @property
    def duration_ns(self) -> int:
        return self.returned_ns - self.last_seen_ns

    @property
    def duration_hours(self) -> float:
        return self.duration_ns / 3_600_000_000_000

    def describe(self) -> str:
        return (f"{self.duration_hours:.1f}h with nothing written, between the "
                f"last stamp and the return")


def last_seen_ns(root: Path) -> int | None:
    """When this system last recorded being alive, or `None` if never.

    An unreadable stamp reads as `None` - a cold start - rather than as "seen
    just now". Wrong in the safe direction: treating a corrupt stamp as alive
    would erase the very outage that corrupted it.
    """
    try:
        payload = json.loads(
            (Path(root) / STAMP_FILE).read_text(encoding="utf-8"))
        return int(payload["last_seen_ns"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_outages(root: Path) -> list[Outage]:
    """Every gap recorded, oldest first.

    A torn line is skipped rather than fatal. The ledger is append-only evidence
    about the past, and refusing to read all of it because one line was cut short
    by the very crash it was recording would be the wrong trade.
    """
    path = Path(root) / LEDGER_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[Outage] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out.append(Outage(
                last_seen_ns=int(row["last_seen_ns"]),
                returned_ns=int(row["returned_ns"]),
                is_reconstructed=bool(row.get("is_reconstructed", False)),
                evidence=row.get("evidence", "")))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def record_reconstructed_outage(root: Path, last_seen_ns_: int,
                                returned_ns: int, evidence: str) -> Outage:
    """File a gap worked out after the fact, labelled as reconstructed.

    For outages that predate this module - the honest way to get a known hole
    onto the record without pretending anything watched it happen. `evidence` is
    mandatory and must be non-empty: a reconstructed row whose basis is not
    written down is indistinguishable from a guess by the time anyone reads it.
    """
    if not evidence.strip():
        raise ValueError(
            "a reconstructed outage must carry the evidence it was derived "
            "from - without it the row is a guess wearing a measurement's "
            "clothes, and nobody can check it later")
    if returned_ns <= last_seen_ns_:
        raise ValueError(
            f"a reconstructed outage must end after it began: "
            f"{last_seen_ns_} -> {returned_ns}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    outage = Outage(last_seen_ns=last_seen_ns_, returned_ns=returned_ns,
                    is_reconstructed=True, evidence=evidence)
    with (root / LEDGER_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "last_seen_ns": last_seen_ns_, "returned_ns": returned_ns,
            "duration_ns": returned_ns - last_seen_ns_,
            "is_reconstructed": True, "evidence": evidence}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return outage


def record_liveness(root: Path, now_ns: int,
                    threshold_ns: int = DEFAULT_THRESHOLD_NS
                    ) -> Outage | ColdStart | None:
    """Stamp that we are alive, and report any hole since the last stamp.

    Returns an `Outage` when one was found and recorded, a `ColdStart` on the
    first run, and `None` on an ordinary tick. Raises `ValueError` when the clock
    has gone backwards.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    now_ns = int(now_ns)
    previous = last_seen_ns(root)

    result: Outage | ColdStart | None
    if previous is None:
        result = ColdStart(at_ns=now_ns)
    elif now_ns < previous:
        # A negative outage is not a shorter outage. NTP stepping the clock back
        # would otherwise write a row claiming the return preceded the departure,
        # and every duration derived from this ledger afterwards is suspect.
        raise ValueError(
            f"the clock went backwards: last seen at {previous}, now {now_ns}. "
            f"Refusing to record a negative outage - a ledger holding one cannot "
            f"be used to reason about any of the others")
    elif now_ns - previous > threshold_ns:
        result = Outage(last_seen_ns=previous, returned_ns=now_ns)
        with (root / LEDGER_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "last_seen_ns": previous, "returned_ns": now_ns,
                "duration_ns": now_ns - previous,
                "threshold_ns": threshold_ns}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    else:
        result = None

    # Written whole and renamed: a reader catching a half-written stamp would
    # parse nothing, read it as a cold start, and forget every outage recorded.
    tmp = root / (STAMP_FILE + ".tmp")
    tmp.write_text(json.dumps({"last_seen_ns": now_ns}) + "\n", encoding="utf-8")
    os.replace(tmp, root / STAMP_FILE)
    return result


def main(argv: list[str] | None = None) -> int:
    """One tick. Called from a supervisor loop, not run as a daemon itself.

    Deliberately a single pass: a daemon that owned its own sleep would be one
    more process able to die quietly, and the thing being defended against is
    exactly a process dying quietly.
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(
        prog="ops.liveness_ledger",
        description="Stamp that this system is alive, and report any gap since "
                    "the last stamp.")
    parser.add_argument("--capture-root", default=str(Path.home() / "capture"))
    parser.add_argument("--threshold-seconds", type=float,
                        default=DEFAULT_THRESHOLD_NS / 1_000_000_000)
    args = parser.parse_args(argv)

    root = Path(args.capture_root) / "liveness"
    result = record_liveness(
        root, now_ns=time.time_ns(),
        threshold_ns=int(args.threshold_seconds * 1_000_000_000))

    if isinstance(result, Outage):
        # Printed loudly and to stderr: this is the one line in the log that
        # explains a hole in the tape, and a supervisor log only records
        # failures, so an outage written at INFO would be the quietest possible
        # report of the loudest possible event.
        print(f"OUTAGE: {result.describe()}", file=sys.stderr, flush=True)
    elif isinstance(result, ColdStart):
        print("cold start: no prior liveness stamp on this machine", flush=True)
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
