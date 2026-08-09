"""Auto-halt per venue. The gate no order may pass while a venue is degraded.

`ARCHITECTURE.md` Layer 3 puts this in Phase 2, before any capital, and gives
the reason: **every major volatility spike since 2020 has a matching degradation
on a top-5 venue.** Downtime is a base rate, not a tail. Oct 10 2025 was the
largest cascade on record - ~$19.13B liquidated, ~1.6M accounts - and Binance
degraded for roughly an hour while valuing collateral on internal prices,
depegging USDe to $0.65 *only there*. Hyperliquid held 100% uptime through it.

Three properties, each chosen against a specific way this goes wrong.

**Fail closed.** A venue nothing has been measured about is not tradeable. An
empty registry means no health signal has arrived, which is not the same as
health, and the two must never collapse into each other. Refusing to trade is
recoverable; trading into a degraded matching engine is not.

**Per venue.** Binance and Hyperliquid were chosen precisely because they fail
differently. A global halt throws away the only benefit of running both.

**Sticky.** A halt does not lift when the signal recovers - it lifts after
health has been *sustained*. The instant an error rate dips is not the instant a
matching engine is sound, and `ARCHITECTURE.md` is explicit that the correct
response to a degraded venue is never to retry into it.

Deliberately not here: what to do with open positions. `DECISIONS.md` §6 settles
that separately - *flatten* for system-integrity faults, *hold* for market-wide
halts because forcing liquidation into a halted book is worse, *hedge* only as a
stopgap - and it needs an execution layer that does not exist yet. This module
answers one question, "may this venue be traded", and the corpus is emphatic that
halt logic must be simple enough to execute correctly while degraded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

# Why a venue is halted. Strings rather than an enum so a persisted state file
# stays readable by a human at 3am, which is when it will be read.
HALT_UNKNOWN = "no_health_measured"
HALT_CORRUPTING = "corrupting_events"
HALT_SILENCE = "sustained_silence"

# A single corrupting event halts. That is deliberate and not tunable: a
# corrupting event means the archive itself could not be trusted for some
# stream, and there is no threshold of "some corruption is fine" that survives
# contact with a matching engine.
_CORRUPTING_LIMIT = 0

# Silence is NOT a halt signal, and finding that out cost a false positive worth
# recording. A first version halted above 10 silence events, calibrated when
# capture ran 3 symbols. Run against the real archive at 2,123 symbols it halted
# all three venues immediately - binance reported 1,845 silence events while
# perfectly healthy.
#
# The count cannot be rescued by a threshold or by converting it to a rate,
# because two things inflate it that have nothing to do with venue health:
#
#   * The broad tail is deliberately full of thin symbols. A symbol that trades
#     twice an hour is quiet, not broken.
#   * `forceOrder` is withheld by the venue and subscribed anyway on purpose, so
#     it is silent forever by design - roughly half the tail's streams.
#
# `ARCHITECTURE.md` Layer 3 names the signals that do belong here: API error
# rate, price deviation against a reference, and staleness *of a specific
# stream*. None is a count of silence events across a whole venue. Until a
# per-stream staleness view exists for the core symbols, this module halts on
# corruption alone and says so rather than firing on a proxy it cannot justify.
_SILENCE_LIMIT = None

# How long a venue must look healthy before a halt lifts. Thirty minutes is
# scaled to the Oct 2025 precedent - Binance degraded for about an hour - not to
# how quickly a dashboard turns green.
_RECOVERY_DWELL_NS = 30 * 60 * 1_000_000_000

_STATE_FILE = "venue_halts.json"
_HISTORY_FILE = "venue_halts.ndjson"


def assess_venue(report: dict) -> tuple[str, str] | None:
    """`(reason, detail)` if this health report warrants a halt, else `None`.

    Pure, and separable from the bookkeeping on purpose: the decision is the part
    worth reasoning about, and it can be tested without a filesystem.
    """
    corrupting = int(report.get("corrupting_non_gap") or 0)
    if corrupting > _CORRUPTING_LIMIT:
        return (HALT_CORRUPTING,
                f"{corrupting} corrupting event(s) - the archive could not be "
                f"trusted for at least one stream")

    # No silence check. See _SILENCE_LIMIT: the count is inflated by thin tail
    # symbols and by a feed the venue withholds, so it cannot distinguish a sick
    # venue from a healthy one with a wide universe.
    return None


class VenueHaltRegistry:
    """Which venues may be traded, and why not, persisted across restarts.

    Persistence is the point rather than a convenience: a halt a process restart
    clears is a halt an unhealthy venue escapes by killing the watcher.
    """

    def __init__(self, root: Path, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._root = Path(root)
        self._clock_ns = clock_ns
        self._state_path = self._root / _STATE_FILE
        self._history_path = self._root / _HISTORY_FILE
        self._state: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # No state is not an error - it is the first run, and every venue
            # is correctly not-tradeable until something is measured.
            return {}

    def _persist(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2) + "\n", encoding="utf-8")
        # Renamed into place so a crash mid-write cannot leave a state file that
        # parses as "everything is fine".
        tmp.replace(self._state_path)

    def _record(self, venue: str, event: str, reason: str, detail: str) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "ts_ns": self._clock_ns(), "venue": venue, "event": event,
                "reason": reason, "detail": detail}) + "\n")

    def observe(self, venue: str, report: dict) -> None:
        """Feed one health report in and update the venue's halt state."""
        now = self._clock_ns()
        verdict = assess_venue(report)
        entry = self._state.setdefault(
            venue, {"halted": False, "reason": None, "detail": None,
                    "since_ns": None, "healthy_since_ns": None,
                    "last_observed_ns": None})
        # Stamped on every observation, halted or healthy, because "not halted"
        # is only meaningful alongside when it was last checked. Without this the
        # state file is a verdict with no date on it, and a reader cannot tell a
        # venue that is fine from a venue nothing has looked at since yesterday -
        # which is exactly what the status wall was doing with it, reporting a
        # 19-hour-old judgement as current on 2026-08-09.
        entry["last_observed_ns"] = now

        if verdict is not None:
            reason, detail = verdict
            if not entry["halted"]:
                self._record(venue, "halted", reason, detail)
            entry.update(halted=True, reason=reason, detail=detail,
                         since_ns=entry["since_ns"] or now, healthy_since_ns=None)
            self._persist()
            return

        # Healthy sample. Start the dwell clock if it is not already running,
        # and only lift once health has been sustained - a halt that lifts on
        # the first clean sample is a halt that flaps through a degradation.
        if entry["healthy_since_ns"] is None:
            entry["healthy_since_ns"] = now
        if entry["halted"] and now - entry["healthy_since_ns"] >= _RECOVERY_DWELL_NS:
            self._record(venue, "cleared", entry["reason"] or "",
                         f"healthy for {(now - entry['healthy_since_ns']) / 1e9:.0f}s")
            entry.update(halted=False, reason=None, detail=None, since_ns=None)
        self._persist()

    def is_tradeable(self, venue: str) -> bool:
        """The one question this module answers. Unknown means no."""
        entry = self._state.get(venue)
        if entry is None:
            return False
        return not entry["halted"]

    def halt_reason(self, venue: str) -> str | None:
        """Why a venue is not tradeable, or `None` when it is."""
        entry = self._state.get(venue)
        if entry is None:
            return HALT_UNKNOWN
        return entry["reason"] if entry["halted"] else None

    def last_observed_ns(self, venue: str) -> int | None:
        """When a health report was last fed in for this venue, or None.

        None covers both "never" and "written by a build that predated this
        field", and both mean the same thing to a caller: nothing here can be
        treated as a current statement about the venue.
        """
        entry = self._state.get(venue)
        if entry is None:
            return None
        value = entry.get("last_observed_ns")
        return value if isinstance(value, int) else None

    def history(self) -> list[dict]:
        """Every halt and clear, in order. A halt nobody can explain afterwards
        is indistinguishable from a bug, and this one blocks trading."""
        try:
            lines = self._history_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [json.loads(line) for line in lines if line.strip()]
