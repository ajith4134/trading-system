"""Consumes venue frames and routes them to writers, ledger and gap trackers.

Governing principle (see spec): data is never dropped silently, and never
modified to "fix" it. A malformed frame is still written verbatim and flagged.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

from capture.capture_ledger import (
    CaptureLedger, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO,
    SEVERITY_OBSERVATION_LOSS,
)
from capture.raw_writer import RawCaptureError, RawWriter, hour_key, utc_date_of
from capture.sequencing import BinanceDepthTracker, StalenessTracker, quantile_ns

# stream/symbol come from `extract()`, which reads them out of the wire
# payload (event name, "s"/"coin" fields, ...). They end up as filename
# components in RawWriter's path (see raw_writer.paths_for). A value
# containing "/" or ".." would be treated as extra path segments, letting a
# malformed or hostile frame steer writes outside the intended directory.
# Anything that isn't a plain token falls back to "unknown", the same bucket
# already used for frames whose routing fields could not be determined.
_SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")

# How long a stream may go quiet before the silence is reported, as a function
# of what that stream routinely does.
#
# The estimator is a quantile over a bounded window of recent gaps, and both
# properties are load-bearing. A high-water mark was tried first and is the
# same self-reinforcing failure `StalenessTracker` documents: one genuine 600s
# stall on a settled 1s stream raised the threshold to 1800s, so the stream
# could then die for good and five minutes of true silence reported nothing -
# the stall taught the check that was meant to catch it.
#
# Excluding gaps that were themselves flagged as stalls does not work here,
# which is why this is a quantile instead. On first sight a 200s gap on a 200s
# stream is indistinguishable from a 200s stall on a 1s stream, so exclusion
# throws away exactly the evidence a slow stream needs and then reports it dead
# every session (test_a_routinely_slow_stream_still_learns_its_own_cadence).
# What actually separates them is repetition: a stall is one-off and sits above
# the quantile, a slow cadence repeats and becomes the quantile.
_SILENCE_STALL_MULTIPLE = 3.0
_SILENCE_CADENCE_QUANTILE = 0.90
# Frames, matching `StalenessTracker`'s window for the same reason: an old
# outlier has to age out rather than bind the threshold for the whole session.
_SILENCE_WINDOW_FRAMES = 200
# However slow a stream claims to be, silence becomes reportable eventually -
# an hour, one file rotation. Without it a stream can talk its way into never
# being checked again, and detection has to be bounded regardless of history.
_SILENCE_CEILING_SECONDS = 3600.0


def _safe_path_token(value: str) -> str:
    return value if _SAFE_PATH_TOKEN.match(value) else "unknown"


@dataclass
class _QuarantinedHour:
    """One (stream, symbol, hour) whose files cannot be written, and its cost.

    Carries the display casing of the stream and symbol because the quarantine
    is keyed on the casefolded form - the same key `_writers` uses, so a stream
    reported with two casings cannot quarantine one and keep writing the other.
    """

    stream: str
    symbol: str
    hour: str
    frames_lost: int = 0


class VenueRecorder:
    """Routes one venue's frames to per-stream writers, the ledger and gap trackers.

    `queue_size` is accepted for forward compatibility with a future bounded-queue
    task and is currently unused - see the note on `dropped` in `stats()`.
    """

    def __init__(self, venue, specs, root: Path, queue_size: int = 10_000,
                 clock_ns: Callable[[], int] = time.time_ns,
                 silence_grace_seconds: float = 60.0) -> None:
        self._venue = venue
        self._specs = specs
        self._root = Path(root)
        self._queue_size = queue_size
        self._clock_ns = clock_ns
        self._ledger = CaptureLedger(root, venue.name)
        # Subscribed streams, keyed the way writers are keyed, so "did this one
        # ever speak?" is a lookup in `_writers`. See `_record_silent_streams`.
        self._expected_streams = {
            (spec.stream.casefold(), spec.symbol.casefold()): (spec.stream, spec.symbol)
            for spec in specs
        }
        self._session_start_ns = clock_ns()
        self._silence_grace_ns = int(silence_grace_seconds * 1e9)
        # Per stream: when it last spoke, how many frames it has sent, and its
        # recent frame-to-frame gaps - the evidence `_silence_threshold_ns`
        # judges silence against. `_recorded_silent_days` keeps reporting to one
        # event per stream per UTC day; the key carries the day for that reason.
        self._last_frame_ns: dict[tuple[str, str], int] = {}
        self._frames_seen: dict[tuple[str, str], int] = {}
        self._recent_gaps_ns: dict[tuple[str, str], deque[int]] = {}
        self._recorded_silent_days: set[tuple[tuple[str, str], str]] = set()
        self._writers: dict[tuple[str, str], RawWriter] = {}
        self._trackers: dict[tuple[str, str], object] = {}
        # (stream, symbol, hour) triples whose files cannot be written, and what
        # each has cost so far. Keyed by HOUR, not by stream: see
        # `_append_or_quarantine_stream`.
        self._unwritable_hours: dict[tuple[str, str, str], _QuarantinedHour] = {}
        self._stats = {"written": 0, "dropped": 0, "control": 0, "malformed": 0,
                       "unwritable": 0}

    def _writer_for(self, stream: str, symbol: str) -> RawWriter:
        # Keyed on casefolded (stream, symbol) so the same logical stream reported
        # with different casing across frames (e.g. "BTCUSDT" vs "btcusdt") still
        # lands in one writer/file rather than silently splitting across two.
        # The first-seen casing is kept as the writer's on-disk name.
        key = (stream.casefold(), symbol.casefold())
        if key not in self._writers:
            self._writers[key] = RawWriter(self._root, self._venue.name, stream, symbol)
        return self._writers[key]

    def _tracker_for(self, stream: str, symbol: str):
        """The gap owner for one stream: a sequence chain where one exists,
        cadence everywhere else.

        Binance depth carries a U/u/pu chain, and a break in it corrupts the
        book, so that is what is checked (spec 5.4). Every other stream carries
        no sequence at all, and used to get no tracker whatsoever - `trade`,
        `markPrice` and `forceOrder` had no staleness owner, so a stream that
        slowed to a crawl was invisible. `StalenessTracker` works from receipt
        times alone, so it owns all of them.
        """
        key = (stream.casefold(), symbol.casefold())
        if key not in self._trackers:
            if self._venue.name == "binance" and stream == "depth":
                self._trackers[key] = BinanceDepthTracker()
            else:
                self._trackers[key] = StalenessTracker()
        return self._trackers[key]

    def _append_or_quarantine_stream(self, stream: str, symbol: str, payload: str,
                                     t_recv_ns: int, t_exch_ms: int | None,
                                     seq: dict | None, kind: str) -> None:
        """Write one frame, isolating a stream whose hour cannot be written.

        A damaged hour makes `RawWriter.append` raise `HourFileNotAppendable`
        every time, for that one (stream, symbol). Letting it out of the loop
        unwound `consume` entirely and took the venue down with it: on restart
        the first frame killed the session, so an undamaged sibling stream in the
        same session never even had a file created. A single torn hour became a
        permanent crash loop across every stream of the venue - strictly worse
        than the silent data loss the refusal replaced.

        So the damage is contained to the stream that owns it, FOR THE HOUR THAT
        OWNS IT. The quarantine is keyed on (stream, symbol, hour) rather than on
        the stream alone, because that is the scope of the condition: every
        `RawCaptureError` names one hour's pair of files, and the next hour is a
        different pair that has not been written yet. Keying it per stream made
        one torn hour left behind by a prior crash cost that stream every
        remaining hour of the run - on a 24/7 recorder, the stream itself.

        Within an hour the refusal is not retried per frame. The condition IS
        persistent at that scope - the damaged bytes do not heal while the
        process runs - and retrying would put one ledger event and one decompress
        of the damaged hour behind every frame that arrives. Across rotation it
        is not persistent at all, which is what the previous version of this
        docstring got wrong.

        Only `RawCaptureError` is isolated. It names damage to, or contention
        over, one hour's files, which is inherently per-hour. Anything else -
        ENOSPC, a bad descriptor, MemoryError - is a whole-recorder condition,
        and pretending it is per-stream would spin instead of surfacing it.
        """
        key = (stream.casefold(), symbol.casefold())
        # Liveness is recorded before anything can go wrong with the write: a
        # stream whose hour is quarantined is still speaking, and reporting it
        # dead as well would be false.
        self._note_stream_spoke(key, t_recv_ns)

        hour = hour_key(t_recv_ns)
        # This stream has moved on to another hour, so whatever it lost in the
        # hours it left behind is final and can be reported now. Waiting for
        # `close()` means an unattended run never reports the size of the loss.
        self._record_unwritable_hour_totals(
            [self._unwritable_hours.pop(k) for k in list(self._unwritable_hours)
             if k[:2] == key and k[2] != hour],
            ts_ns=t_recv_ns)

        quarantine_key = key + (hour,)
        quarantined = self._unwritable_hours.get(quarantine_key)
        if quarantined is not None:
            quarantined.frames_lost += 1
            self._stats["unwritable"] += 1
            return
        try:
            self._writer_for(stream, symbol).append(
                payload, t_recv_ns, t_exch_ms, seq, kind=kind)
        except RawCaptureError as exc:
            self._unwritable_hours[quarantine_key] = _QuarantinedHour(
                stream=stream, symbol=symbol, hour=hour, frames_lost=1)
            self._stats["unwritable"] += 1
            self._ledger.record(LedgerEvent(
                ts_ns=t_recv_ns, venue=self._venue.name, stream=stream,
                kind="unwritable_stream", severity=SEVERITY_CORRUPTING,
                detail={"symbol": symbol, "hour": hour,
                        "error": type(exc).__name__, "message": str(exc)},
            ))
            return
        self._stats["written"] += 1

    def _record_unwritable_hour_totals(self, quarantined: list[_QuarantinedHour],
                                       ts_ns: int) -> None:
        """Put each quarantined hour's frame count in the ledger, not just RAM.

        The in-memory counter dies with the process; the ledger is the record an
        incident is reconstructed from, so the size of the loss has to reach it.

        The caller removes each record from `_unwritable_hours` before passing it
        here, so nothing can be reported twice - `consume` closes in a `finally`
        and callers close explicitly.
        """
        for record in quarantined:
            self._ledger.record(LedgerEvent(
                ts_ns=ts_ns, venue=self._venue.name, stream=record.stream,
                kind="unwritable_stream_total", severity=SEVERITY_CORRUPTING,
                detail={"symbol": record.symbol, "hour": record.hour,
                        "frames_lost": record.frames_lost},
            ))

    def _record_unwritable_stream_totals(self) -> None:
        """Report every hour still quarantined when the session ends.

        Hours the recorder rotated past have already been reported from the
        frame loop; what is left here is whatever was still quarantined at the
        moment the session stopped.

        Cleared as it is recorded, so a second `close()` does not double-count.
        """
        if not self._unwritable_hours:
            # Deliberately does not read the clock: `close()` is the only caller
            # and every clock read there is one a test has to supply.
            return
        recorded, self._unwritable_hours = self._unwritable_hours, {}
        self._record_unwritable_hour_totals(list(recorded.values()),
                                            ts_ns=self._clock_ns())

    def _note_stream_spoke(self, key: tuple[str, str], t_recv_ns: int) -> None:
        """Remember that this stream is alive, and how far apart its frames come.

        The gap window is kept here rather than taken from a `StalenessTracker`
        because that tracker cannot supply it for the streams that need it most:
        a stream whose ordinary cadence exceeds `stall_multiple` x its floor
        never learns a baseline at all, so its threshold stays pinned at a few
        seconds forever (see the learning-rule note in
        `sequencing.StalenessTracker`). Judging a liquidation feed by that would
        report it dead every session.

        Every gap is recorded, stalls included. They are not excluded but
        outvoted - see the note on `_SILENCE_CADENCE_QUANTILE`.
        """
        last = self._last_frame_ns.get(key)
        if last is not None:
            gaps = self._recent_gaps_ns.setdefault(
                key, deque(maxlen=_SILENCE_WINDOW_FRAMES))
            gaps.append(t_recv_ns - last)
        self._last_frame_ns[key] = t_recv_ns
        self._frames_seen[key] = self._frames_seen.get(key, 0) + 1

    def _silence_threshold_ns(self, key: tuple[str, str]) -> int:
        """How long this stream may be quiet before that is worth recording.

        Never below the grace period - every stream is quiet at startup and a
        stream that has said nothing has no cadence to be judged against - and
        never above the ceiling, so death is always detected in bounded time.
        """
        gaps = self._recent_gaps_ns.get(key)
        routine_ns = quantile_ns(sorted(gaps), _SILENCE_CADENCE_QUANTILE) if gaps else 0
        return min(int(_SILENCE_CEILING_SECONDS * 1e9),
                   max(self._silence_grace_ns,
                       int(_SILENCE_STALL_MULTIPLE * routine_ns)))

    def _record_silent_streams(self, now_ns: int) -> None:
        """Record every subscribed stream that is not producing frames.

        Two conditions, one event, because they are indistinguishable on disk
        and identical to an operator: a stream that never answered at all, and a
        stream that answered and then died. Neither leaves anything to notice -
        the first creates no file, the second leaves a file that simply stops
        growing, which looks exactly like a healthy stream in a quiet market.
        Both are measured facts here: on 2026-08-02 Binance delivered zero
        `aggTrade`, `markPrice@1s` and `forceOrder` frames while `depth` flowed,
        and an earlier version of this check excluded any stream the moment it
        produced one frame - so a stream that died mid-session was invisible.

        Nothing frame-driven can catch the second condition. A dead stream sends
        no frame, so its tracker never runs; only the venue clock, ticking on
        every other stream's frames and again at close, can ask on its behalf.
        That is why this is not delegated to the gap trackers - and the trackers
        genuinely do not own it, which the previous version of this docstring
        wrongly claimed they did.

        How long is too long is per stream, and never shorter than the grace
        period. A stream that has shown its cadence is judged against
        `_SILENCE_STALL_MULTIPLE` x a high quantile of its recent gaps, so a
        liquidation feed minutes between frames is not called dead while a 100ms
        depth stream is - and capped, so no history buys a stream permanent
        exemption. A stream that has shown nothing (never spoke, or spoke
        exactly once) has only the grace period to go on.

        Recorded at most once per stream per UTC DAY, and the day is what makes
        this useful rather than merely quiet. `consume` calls this as frames
        arrive and `close` calls it again, so an event per frame would drown the
        ledger in the anomaly it exists to surface - but once per SESSION, which
        this was, does not line up with anything that reads it. `build_report`
        reports one UTC day, and a `--seconds 0` capture is one session for
        weeks: a stream that died on day one left an event in day one's ledger
        only, and every later day read back `silent_streams=0`. Verified over
        four days with three of four streams dead - days two, three and four all
        reported a clean venue.

        The day is therefore both the dedup scope and the ledger partition the
        event lands in, so the two cannot drift apart again. The cost is one
        extra event per still-dead stream per day, which is the smallest signal
        that can honestly say "this is still dead" to a reader who only ever
        looks at one day.

        A stream that recovers is not re-armed within its day - it already had
        its incident - and is simply not silent on the days after.

        Severity is observation loss, not corruption: what was captured is
        intact, there is simply less of it than was asked for.
        """
        day = utc_date_of(now_ns)
        for key, (stream, symbol) in self._expected_streams.items():
            if (key, day) in self._recorded_silent_days:
                continue
            last_ns = self._last_frame_ns.get(key, self._session_start_ns)
            quiet_ns = now_ns - last_ns
            # The threshold is never below the grace, so a stream well inside it
            # is settled without sorting its window - this runs on every frame.
            if quiet_ns < self._silence_grace_ns:
                continue
            threshold_ns = self._silence_threshold_ns(key)
            if quiet_ns < threshold_ns:
                continue
            self._recorded_silent_days.add((key, day))
            self._ledger.record(LedgerEvent(
                ts_ns=now_ns, venue=self._venue.name, stream=stream,
                kind="silent_stream", severity=SEVERITY_OBSERVATION_LOSS,
                detail={"symbol": symbol,
                        "frames_received": self._frames_seen.get(key, 0),
                        "silent_for_seconds": round(quiet_ns / 1e9, 3),
                        "threshold_seconds": round(threshold_ns / 1e9, 3)},
            ))

    def _record_gap(self, stream: str, symbol: str, report, t_recv_ns: int) -> None:
        self._ledger.record(LedgerEvent(
            ts_ns=t_recv_ns, venue=self._venue.name, stream=stream,
            kind="gap", severity=report.severity,
            detail={"symbol": symbol, **report.detail},
        ))

    async def consume(self, frames: AsyncIterator[str]) -> None:
        # Every RawWriter and the CaptureLedger hold open file handles with
        # buffered zstandard writers. If a frame handler or the frame iterator
        # itself raises, those handles must still be closed (flushing what was
        # already appended) rather than leaked - hence the try/finally around
        # the whole loop rather than just around the happy path.
        try:
            async for payload in frames:
                t_recv_ns = self._clock_ns()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    self._record_malformed_frame(payload, t_recv_ns)
                else:
                    self._route_frame(parsed, payload, t_recv_ns)
                # After the frame is routed, never before: a stream whose first
                # frame is this one has already been counted as having spoken,
                # so it cannot be reported silent in the same breath.
                self._record_silent_streams(t_recv_ns)
        finally:
            self.close()

    def _record_malformed_frame(self, payload: str, t_recv_ns: int) -> None:
        """Flag a frame that is not JSON - and store it verbatim anyway."""
        self._stats["malformed"] += 1
        self._ledger.record(LedgerEvent(
            ts_ns=t_recv_ns, venue=self._venue.name, stream="unknown",
            kind="malformed", severity=SEVERITY_INFO,
            detail={"bytes": len(payload)},
        ))
        self._append_or_quarantine_stream(
            "unknown", "unknown", payload, t_recv_ns, None, None,
            kind="malformed")

    def _route_frame(self, parsed, payload: str, t_recv_ns: int) -> None:
        """Send one parsed frame to its writer, tracker and - on a gap - the ledger."""
        meta = self._venue.extract(parsed)
        if meta.kind == "control":
            self._stats["control"] += 1

        stream = _safe_path_token(meta.stream)
        symbol = _safe_path_token(meta.symbol)

        tracker = self._tracker_for(stream, symbol)
        if tracker is not None and meta.kind == "data":
            body = parsed.get("data", parsed)
            if isinstance(tracker, BinanceDepthTracker):
                report = tracker.check(body)
            else:
                report = tracker.check(t_recv_ns)
                # Before a baseline exists, a staleness report says only that
                # the gap beat a fixed floor - it has not been compared against
                # this stream at all. For a stream whose ordinary cadence is
                # slower than that floor, and which can therefore never acquire
                # a baseline, that is a ledger event on every frame it will ever
                # receive. Whether such a stream has died is answered by
                # `_record_silent_streams` from the venue clock instead.
                if not tracker.has_baseline():
                    report = None
            if report is not None:
                self._record_gap(stream, symbol, report, t_recv_ns)

        self._append_or_quarantine_stream(
            stream, symbol, payload, t_recv_ns, meta.t_exch_ms, meta.seq,
            kind=meta.kind)

    def stats(self) -> dict:
        # `dropped` stays at 0 forever in this task: nothing here has anywhere to
        # drop a frame *to*. It only becomes reachable once a bounded queue sits
        # in front of consume() (a later, unwritten task) and can overflow. The
        # key and the zero-assertion are kept now, deliberately, so that task
        # only has to wire in the increment rather than invent the contract.
        #
        # `unwritable` is a different loss and deliberately a different key:
        # frames that reached the recorder and could not be stored because their
        # hour's files are damaged. It counts frames NOT in `written`.
        return dict(self._stats)

    def close(self) -> None:
        # Every writer (and the ledger) must get a close attempt regardless of
        # whether an earlier one raised - e.g. a disk-full during one writer's
        # zstd footer write must not orphan the rest with unflushed buffers.
        # Errors are collected and re-raised after every close was attempted,
        # rather than propagating from the first failure and abandoning the loop.
        errors: list[Exception] = []
        # Before the ledger is closed, and inside the same error collection: a
        # failure to record the totals must not cost the closes below.
        try:
            self._record_unwritable_stream_totals()
        except Exception as exc:
            errors.append(exc)
        # A venue that sent nothing at all never entered the frame loop, so this
        # is the only place that condition can be caught - and it is the worst
        # one, because it leaves no file anywhere to notice the absence of.
        try:
            self._record_silent_streams(self._clock_ns())
        except Exception as exc:
            errors.append(exc)
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        try:
            self._ledger.close()
        except Exception as exc:
            errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("errors while closing venue recorder resources", errors)
