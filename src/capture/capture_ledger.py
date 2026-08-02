"""Records capture anomalies as first-class, queryable events.

Plain NDJSON, uncompressed: this file is read during incidents, and volume is
tiny compared to market data.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from capture.raw_writer import utc_date_of

SEVERITY_INFO = "info"
SEVERITY_OBSERVATION_LOSS = "observation_loss"
SEVERITY_CORRUPTING = "corrupting"


@dataclass(frozen=True)
class LedgerEvent:
    ts_ns: int
    venue: str
    stream: str
    kind: str
    severity: str
    detail: dict


def _path_for(root: Path, venue: str, date: str) -> Path:
    return Path(root) / "ledger" / venue / date / "events.ndjson"


class CaptureLedger:
    def __init__(self, root: Path, venue: str) -> None:
        self._root, self._venue = Path(root), venue
        self._fh = None
        self._date: str | None = None

    def record(self, event: LedgerEvent) -> None:
        """Append one event and make it durable before returning.

        Fsynced, not merely flushed. `fh.flush()` moves the line from a Python
        buffer into the page cache, which survives `kill -9` and does not
        survive a power loss or a hypervisor reset. This file is the incident
        record and the sole evidence of every anomaly the capture noticed -
        losing its tail is losing exactly the events that describe the crash
        that lost them, and it defeats the guarantees that nothing is dropped
        silently, that recovery discards nothing, and that gap detection cannot
        silently fail to detect.

        The cost is affordable because the volume is: measured 2026-08-02 on
        this disk (GCE ext4, fsync median 1.85 ms), 1000 events take 1.54 s
        against 13.9 ms unsynced. The ledger records anomalies, not frames -
        the alarm rate on a healthy stream is bounded below 5% by
        `StalenessTracker`, so this is single-digit seconds per hour per stream
        in the worst healthy case, and nothing at all in the ordinary one.
        """
        date = utc_date_of(event.ts_ns)
        if date != self._date:
            self.close()
            path = _path_for(self._root, self._venue, date)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._date = date
        self._fh.write(json.dumps(asdict(event), separators=(",", ":"), sort_keys=True, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._date = None


@dataclass(frozen=True)
class DamagedLedgerLine:
    """A ledger line that could not be turned back into an event."""
    line_number: int
    text: str
    error: str


class LedgerReadResult(list):
    """The intact events, carrying the lines that could not be parsed.

    Subclasses `list` so it reads exactly like the event list callers already
    iterate and `len()`, while `damaged` makes the unparseable lines reachable
    instead of either lost or fatal.
    """

    def __init__(self, events: list[LedgerEvent], damaged: list[DamagedLedgerLine]) -> None:
        super().__init__(events)
        self.damaged = damaged

    @property
    def events(self) -> list[LedgerEvent]:
        return list(self)


def read_all(root: Path, venue: str, date: str) -> LedgerReadResult:
    """Read a day's events, keeping the intact ones when a line is damaged.

    The ledger is the anomaly record - the last line of defence during an
    incident - and its final line is the one a crash truncates. Parsing the file
    as a single list comprehension made one torn line fatal for the whole day:
    every intact event before it became unreadable at the exact moment it was
    needed. Damaged lines are collected on the result instead, so they are
    surfaced rather than silently skipped or allowed to destroy the rest.
    """
    path = _path_for(root, venue, date)
    events: list[LedgerEvent] = []
    damaged: list[DamagedLedgerLine] = []
    if not path.exists():
        return LedgerReadResult(events, damaged)
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                # ValueError covers json.JSONDecodeError; TypeError covers a line
                # that is valid JSON but not a LedgerEvent's fields.
                events.append(LedgerEvent(**json.loads(line)))
            except (ValueError, TypeError) as exc:
                damaged.append(DamagedLedgerLine(
                    line_number=line_number,
                    text=line.rstrip("\n"),
                    error=f"{type(exc).__name__}: {exc}",
                ))
    return LedgerReadResult(events, damaged)
