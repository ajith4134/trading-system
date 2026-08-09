"""Consumes venue frames and routes them to writers, ledger and gap trackers.

Governing principle (see spec): data is never dropped silently, and never
modified to "fix" it. A malformed frame is still written verbatim and flagged.
"""
from __future__ import annotations

import json
import base64
import re
import resource
import time
from collections import deque, OrderedDict
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

# What actually makes a token dangerous, as opposed to merely not-ASCII: path
# separators, and control characters (NUL ends a filename early in the syscall
# layer even when Python is happy with it). A value that is only dots - "." or
# ".." - is a path component in its own right and is caught separately.
_PATH_HOSTILE = re.compile(r"[/\\\x00-\x1f\x7f]")

# base32 is A-Z2-7, already inside _SAFE_PATH_TOKEN, so an encoded symbol needs
# no second escaping pass. The prefix marks it as encoded and is chosen to be
# something no exchange would list: symbols do not begin with an underscore.
_ENCODED_PREFIX = "_b32_"

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
# UTC hours are exact multiples of this since the epoch - Unix time carries no
# leap seconds - so an hour boundary is reachable by integer arithmetic, with no
# string formatted to find it. See `_settle_writers_whose_hour_ended`.
_NS_PER_HOUR = 3_600_000_000_000
# A writer holds one raw and one index descriptor for as long as its hour is
# open, so descriptors scale with the universe: 2,115 symbols is over 4,000.
_FDS_PER_OPEN_WRITER = 2
# Held back from the descriptor budget for everything that is not a writer -
# websockets, the ledger, the ops files, stdio, and whatever a library opens
# without asking. Generous on purpose: the cost of over-reserving is a few more
# evictions, and the cost of under-reserving is the crash this bounds.
_RESERVED_FDS = 256
# Below this the pool would thrash harder than it protects, so a limit too small
# to work with is treated as a limit to ignore. A recorder that evicts on almost
# every frame is not capturing.
_MIN_OPEN_WRITERS = 64
# How often the pool reports itself to the ledger, in stream time. Five minutes
# is fine enough that a wall tile is never quoting a number from a different
# regime, and coarse enough to cost 288 lines per venue per day against a ledger
# that already carries hundreds of thousands of gap events.
_POOL_REPORT_INTERVAL_NS = 300_000_000_000
# How long a session waits before its first report. Not zero: a recorder opens
# its hours as symbols first trade, so a report from the first frame describes a
# pool holding one hour and nothing else. Measured live 2026-08-09 - a recorder
# four minutes into a run, holding 1,158 descriptors, had told the ledger
# "peak 1 of 32640 open hours (0%)", which is a true statement about the first
# millisecond and a misleading one about the tile it renders.
#
# Short enough that a healthy recorder is on the board within half a minute, and
# a session that dies before reaching it is covered by the close-time report.
_POOL_WARMUP_NS = 30_000_000_000


def max_open_writers_for(soft_limit: int) -> int:
    """How many hours may stay open at once under this descriptor limit.

    Derived rather than configured, because the number that matters is the one
    the process was actually given. Measured 2026-08-09: the supervisor ran under
    `sudo -H bash -lc` and the recorder inherited a soft limit of 1024 against a
    universe needing 4,000 - it died on `OSError: [Errno 24]` with exactly 1024
    descriptors open, was restarted, and walked into the same wall twenty times.

    Raising the limit stopped that, and this makes it survivable rather than
    merely unlikely: at 65536 the budget is far above what any venue here opens,
    so nothing is ever evicted, and at the inherited 1024 the recorder evicts
    instead of dying. The failure mode moves from data loss to compression.
    """
    return max(_MIN_OPEN_WRITERS, (soft_limit - _RESERVED_FDS) // _FDS_PER_OPEN_WRITER)


def _descriptor_soft_limit() -> int:
    """The process's own soft NOFILE, with "unlimited" read as a large finite number.

    `RLIM_INFINITY` is -1, and arithmetic on it produces a budget below the
    floor - which would cap a process with no limit at all to 64 open hours, the
    exact opposite of what it says.
    """
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    return 1 << 30 if soft == resource.RLIM_INFINITY else soft
# How often the silence check may run, in stream time. It walks every subscribed
# stream, and it can report a given stream at most once per UTC day, so running it
# per frame buys nothing and costs O(streams) on the hot path: measured 136 us per
# frame at binance's 1,141 expected streams, 27% of one core at 2,000 frames/s.
# One second is far below the smallest threshold it can act on - the 60 s grace -
# so nothing it reports changes.
_SILENCE_CHECK_INTERVAL_NS = 1_000_000_000
# How many hour files the boundary sweep may close on one frame. Each close writes
# a zstd footer and fsyncs the raw file, the index and the directory: measured
# 4.2-4.4 ms on this box's ext4, at a median fsync of 1.40 ms. Closing binance's
# ~600 writers in one pass therefore blocks the event loop for 2.65 s and spot's
# ~1,000 for 4.17 s, which pauses the socket and backdates `t_recv_ns` for every
# frame behind the stall. 25 keeps one slice near 110 ms while still draining a
# thousand writers within a second of frames.
_MAX_HOUR_CLOSES_PER_FRAME = 25
# However slow a stream claims to be, silence becomes reportable eventually -
# an hour, one file rotation. Without it a stream can talk its way into never
# being checked again, and detection has to be bounded regardless of history.
_SILENCE_CEILING_SECONDS = 3600.0


def _safe_path_token(value: str) -> str:
    """Turn a routing field into a filename component without losing it.

    Three outcomes, and the middle one exists because the first live run of the
    broad tail on 2026-08-08 found the rule was throwing real data away.

    A plain ASCII token passes through untouched, so every file written before
    this function grew a second branch keeps exactly the name it had.

    A symbol the venue really lists but which is not a plain ASCII token -
    Binance carries perpetuals with CJK symbols, `币安人生USDT` among them - is
    **encoded, not discarded**. Filing it as "unknown" merged it with every
    other non-ASCII symbol into one bucket, made it unreadable by symbol, and
    reported nothing: 0 dropped, 0 malformed, one quietly wrong file. The
    encoding is base32 of the UTF-8 bytes, which is `A-Z2-7` and therefore
    already inside the safe set, behind a prefix no exchange symbol would use.
    Padding is stripped because `=` is not path-safe on every filesystem.

    Anything that could steer a write out of the archive still falls back to
    "unknown", which is the whole reason this function exists. That is decided
    on the dangerous characters themselves rather than on "not ASCII", so
    widening the rule for real symbols does not widen the attack surface.
    """
    # Hostile is decided first, and that ordering is load-bearing rather than
    # stylistic. `.` is inside _SAFE_PATH_TOKEN, so a value of exactly ".." used
    # to match it and pass through untouched - a path component in its own
    # right. The traversal test that existed only used "../../evil", which the
    # separator caught, so the bare case was never exercised.
    if not value or _PATH_HOSTILE.search(value) or value.strip(".") == "":
        return "unknown"
    if _SAFE_PATH_TOKEN.match(value):
        return value
    return _ENCODED_PREFIX + base64.b32encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def decode_path_token(token: str) -> str:
    """Recover the symbol a path token was built from.

    The archive has to be able to name what it captured; an encoding that
    cannot be reversed is the "unknown" bucket with extra steps.
    """
    if not token.startswith(_ENCODED_PREFIX):
        return token
    body = token[len(_ENCODED_PREFIX):]
    return base64.b32decode(body + "=" * (-len(body) % 8)).decode("utf-8")


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
                 silence_grace_seconds: float = 60.0,
                 max_open_writers: int | None = None) -> None:
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
        # Every writer this recorder has ever built, open or not. Entries are
        # never dropped: a `RawWriter` with no hour open holds no descriptors,
        # and keeping the object is what lets an evicted stream reopen its hour
        # from remembered state instead of decompressing the file again.
        self._writers: dict[tuple[str, str], RawWriter] = {}
        # The descriptor pool: the subset believed to be holding an hour open,
        # in least-recently-written-first order. `OrderedDict.move_to_end` is
        # O(1), which is the only reason this can live on the per-frame path -
        # see `_settle_writers_whose_hour_ended` for what happens to this hot
        # path when something O(writers) is put on it.
        self._open_writers: OrderedDict[tuple[str, str], RawWriter] = OrderedDict()
        self._max_open_writers = (max_open_writers if max_open_writers is not None
                                  else max_open_writers_for(_descriptor_soft_limit()))
        self._peak_open_writers = 0
        # None until the first frame arms it, because the warm-up is measured
        # from stream time and there is none before then.
        self._next_pool_report_ns: int | None = None
        self._pool_reported_at_close = False
        self._last_frame_recv_ns: int | None = None
        # Zero so the first frame sweeps once and sets the real boundary. At that
        # point there are no writers yet, so the sweep costs nothing.
        self._next_hour_starts_ns = 0
        # Zero so the first frame runs the check and sets the real deadline.
        self._next_silence_check_ns = 0
        # Writers whose hour has ended and which have not been closed yet. Drained
        # a slice at a time so no single frame pays for all of them.
        self._hour_close_backlog: list[RawWriter] = []
        self._trackers: dict[tuple[str, str], object] = {}
        # (stream, symbol, hour) triples whose files cannot be written, and what
        # each has cost so far. Keyed by HOUR, not by stream: see
        # `_append_or_quarantine_stream`.
        self._unwritable_hours: dict[tuple[str, str, str], _QuarantinedHour] = {}
        self._stats = {"written": 0, "dropped": 0, "control": 0, "malformed": 0,
                       "unwritable": 0, "writers_evicted": 0}

    def _writer_for(self, stream: str, symbol: str) -> RawWriter:
        # Keyed on casefolded (stream, symbol) so the same logical stream reported
        # with different casing across frames (e.g. "BTCUSDT" vs "btcusdt") still
        # lands in one writer/file rather than silently splitting across two.
        # The first-seen casing is kept as the writer's on-disk name.
        key = (stream.casefold(), symbol.casefold())
        writer = self._writers.get(key)
        if writer is None:
            writer = RawWriter(self._root, self._venue.name, stream, symbol)
            self._writers[key] = writer
        # Marked as open before it is, because the caller appends immediately and
        # the alternative is a second dictionary operation on the hot path. A
        # writer that is tracked but never actually opened - an append that
        # quarantined instead - costs nothing: eviction drops it for free.
        if key in self._open_writers:
            self._open_writers.move_to_end(key)
        else:
            self._open_writers[key] = writer
            self._evict_until_within_budget()
            # After eviction, not before. The insert above only marks the key -
            # `append` is what opens the files - so a pool momentarily holding
            # budget+1 entries never held budget+1 descriptors, and a peak taken
            # before the eviction reported "9 of 8 open hours (112%)" on the
            # tile. A number over 100% of its own budget reads as a bug in the
            # board, which is a good way to have a real one ignored.
            #
            # Only an insert can grow the pool, so this is here rather than on
            # every frame. It is what says how close the budget came to biting on
            # a run that evicted nothing - the difference between headroom and luck.
            if len(self._open_writers) > self._peak_open_writers:
                self._peak_open_writers = len(self._open_writers)
        return writer

    def _evict_until_within_budget(self) -> None:
        """Close the least recently written hours until the pool fits.

        Eviction is not a loss. The hour file is finished properly - zstd footers
        written, both files fsynced, the `.writing` marker released - and the next
        frame for that stream reopens it and appends a new frame to the same pair.
        Concatenated frames read back as one stream, which is already how every
        restart writes.

        What it costs is compression ratio, because a shorter zstd frame has less
        history to reference, and that is the trade this exists to make: worse
        compression on the quietest streams instead of `OSError: [Errno 24]` and a
        dead recorder.

        LRU is what keeps the cost small, and not by accident. The evicted writer
        is the one that has gone longest without a frame, so it is the one whose
        hour file is smallest and whose reopen is cheapest - and the one least
        likely to need reopening at all.
        """
        while len(self._open_writers) > self._max_open_writers:
            _, writer = self._open_writers.popitem(last=False)
            if not writer.is_open:
                # Already closed by hour rotation or by the end-of-hour sweep.
                # It holds no descriptors, so dropping the entry is the whole job.
                continue
            try:
                writer.close()
            except Exception:
                # A close that fails has still released what it could, and the
                # writer marks itself unresumable so its next open re-reads. The
                # frame that triggered this eviction is not the place to raise:
                # it belongs to a different stream and has done nothing wrong.
                pass
            self._stats["writers_evicted"] += 1

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
            # Asked of the venue rather than matched on its name. The name test
            # this replaced was correct only while exactly one Binance venue
            # existed; the moment spot arrived it silently downgraded spot depth
            # to plain staleness tracking, and a gap the archive does not report
            # is one nobody can find later.
            if getattr(self._venue, "depth_is_binance_chained", False) and stream == "depth":
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

    def _check_stream_health(self, now_ns: int) -> None:
        """Run the silence check, at most once per `_SILENCE_CHECK_INTERVAL_NS`.

        The throttle lives here rather than inside `_record_silent_streams` so that
        every test driving that method directly still exercises it unconditionally,
        and so the hot path pays one integer compare.

        Why it has to be throttled at all: the check walks every subscribed stream,
        which is 136 us per frame at binance's 1,141 streams - 27% of one core at
        2,000 frames/s. That is what made this recorder a slow consumer, and
        websockets kills a slow consumer. Its message queue defaults to
        `max_queue=16` and is built with `pause=transport.pause_reading`
        (`websockets/asyncio/connection.py`), so a full queue stops the transport
        reading the socket at all - Pong frames included. The keepalive task then
        waits `ping_timeout=20` for a pong that cannot arrive and closes the
        connection with `sent 1011 (internal error) keepalive ping timeout`. That is
        the exit recorded for binance in the supervisor log, and binance is the
        venue with the highest frame rate and the most streams.
        """
        if now_ns < self._next_silence_check_ns:
            return
        self._next_silence_check_ns = now_ns + _SILENCE_CHECK_INTERVAL_NS
        self._record_silent_streams(now_ns)

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

    def _settle_writers_whose_hour_ended(self, now_ns: int) -> None:
        """Finish every hour file whose hour is over, on behalf of a quiet symbol.

        The same argument as `_record_silent_streams`, one layer down: nothing
        frame-driven can catch this, because the symbol holding the stale hour open
        is precisely the one sending no frames. `RawWriter.append` rotates only
        when that symbol speaks again, so a thin pair's completed hour stays open -
        claimed by its `.writing` marker, with its last frames inside the zstd
        compressor and, at worst, nothing at all on disk.

        Measured on the live archive 2026-08-08 at 17:28, three venues captured
        broad: 117 binance-spot hour files, 1 binance and 2 hyperliquid were still
        held open on hours that had already ended.
        `trade_ARBIDR_2026-08-08T11.ndjson.zst` was **0 bytes** six and a half
        hours after hour 11 closed, and `read_pair` returned zero frames for it
        without raising - so a build of that day reads a captured hour as a market
        with no trades and records the result as complete. The loss is silent at
        every layer that could have noticed.

        Driven by the arriving frame's own receive time, the same clock `append`
        keys the hour on. No wall clock enters this path, so a replayed stream
        still produces byte-identical files.

        **One integer compare per frame, and the writers are reached only when an
        hour actually ends.** The first version of this called
        `close_if_hour_ended` on every writer on every frame, and that method
        computes `hour_key`, which is a `strftime`. Measured on this box: 2.35 us
        per call, so binance's ~600 writers cost 1.41 ms per frame - 282% of one
        core at 2,000 frames/s, and 644% for spot's 1,372. That does not merely
        waste CPU: it blocks the asyncio loop long enough for the websocket
        keepalive to time out, and the recorder dies with `sent 1011 (internal
        error) keepalive ping timeout`. The archive's own supervisor log shows that
        exit for binance, which is why the hot path here has to stay this cheap.

        The boundary is held in nanoseconds because UTC hours are exact multiples
        of 3600 s since the epoch - Unix time carries no leap seconds - so the
        modulo below is the same answer `hour_key` gives, without formatting a
        string to get it.

        Residual, accepted: a frame whose receive time falls in an already-swept
        hour reopens that hour in `append`, and this sweep will not revisit it
        until the next boundary. That needs the receive clock to move backwards
        across an hour edge, and `store.cli` skips and names a claimed hour of the
        day it is building, so the cost is a rebuild rather than a silent loss.
        """
        if now_ns >= self._next_hour_starts_ns:
            self._next_hour_starts_ns = now_ns - (now_ns % _NS_PER_HOUR) + _NS_PER_HOUR
            # Snapshotted, because `_writers` gains entries as new hours open and a
            # live view would hand this loop writers on the current hour forever.
            self._hour_close_backlog = list(self._writers.values())

        # A writer already on the current hour still consumes a slot and returns
        # False. That keeps the bound honest: the slice is a cap on work attempted,
        # not on work that happened to be needed.
        for _ in range(min(_MAX_HOUR_CLOSES_PER_FRAME, len(self._hour_close_backlog))):
            self._hour_close_backlog.pop().close_if_hour_ended(now_ns)

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
            async for frame in frames:
                t_recv_ns = self._clock_ns()
                # A polled frame travels with the spec that requested it,
                # because some REST bodies name no symbol. The payload itself
                # is untouched either way.
                route_hint = getattr(frame, "spec", None)
                payload = getattr(frame, "payload", frame)
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    self._record_malformed_frame(payload, t_recv_ns)
                else:
                    self._route_frame(parsed, payload, t_recv_ns, route_hint)
                # After the frame is routed, never before: a stream whose first
                # frame is this one has already been counted as having spoken,
                # so it cannot be reported silent in the same breath.
                self._check_stream_health(t_recv_ns)
                self._settle_writers_whose_hour_ended(t_recv_ns)
                self._report_writer_pool_if_due(t_recv_ns)
        finally:
            self.close()

    def _report_writer_pool_if_due(self, t_recv_ns: int) -> None:
        """Put the descriptor pool's state in the ledger on a fixed cadence.

        Emitted whether or not anything was evicted, and that is the point. A
        pool reported only when it evicts leaves the wall unable to tell "nothing
        was evicted" from "nothing ever looked", and Rule 8 says those must not
        render the same. This event is what makes the tile's OK a measurement
        rather than an absence of bad news.

        The cadence also makes staleness legible: the tile reads the newest event
        and can say how old it is, so a recorder that died hours ago stops
        reporting a healthy pool on its behalf.

        One integer compare per frame, for the same reason `_check_stream_health`
        is guarded that way - see `_settle_writers_whose_hour_ended` for what an
        unguarded per-frame walk did to the event loop here.
        """
        # Kept so `close()` can stamp its final report without reading the clock.
        # `_record_unwritable_stream_totals` states the constraint: close is
        # called from a `finally` that tests reach with a finite clock iterator
        # sized to the frames they feed, so a clock read there is one every such
        # test has to be rewritten to supply.
        self._last_frame_recv_ns = t_recv_ns
        if self._next_pool_report_ns is None:
            self._next_pool_report_ns = t_recv_ns + _POOL_WARMUP_NS
            return
        if t_recv_ns < self._next_pool_report_ns:
            return
        self._next_pool_report_ns = t_recv_ns + _POOL_REPORT_INTERVAL_NS
        self._record_writer_pool(t_recv_ns)

    def _record_writer_pool(self, ts_ns: int) -> None:
        self._ledger.record(LedgerEvent(
            ts_ns=ts_ns, venue=self._venue.name, stream="writer_pool",
            kind="writer_pool", severity=SEVERITY_INFO,
            detail={"open_hours": len(self._open_writers),
                    "peak_open_hours": self._peak_open_writers,
                    "budget": self._max_open_writers,
                    "evicted": self._stats["writers_evicted"]},
        ))

    def _record_writer_pool_totals(self) -> None:
        """Report the pool one last time as the session ends.

        The cadence alone is not enough, and the gap is worst exactly where this
        tile matters. A recorder dying every forty seconds - which is what
        `Errno 24` did to this one on 2026-08-09 - never reaches the five-minute
        interval, so the only line it ever writes is the one from its first
        frame: zero evicted, one hour open. The tile would have read OK straight
        through the crash loop it exists to catch.

        Measured while building it, before this existed: a run that evicted 72
        times reported `evicted: 0` to the ledger, because the first frame was
        the only report it ever made.

        Stamped with the last frame's receive time rather than the clock, for the
        same reason `_record_unwritable_stream_totals` reads no clock here. A
        session that saw no frames has no such time and reports nothing - which
        is the honest answer: nothing measured the pool, so the tile must say
        nothing measured the pool.
        """
        if self._pool_reported_at_close or self._last_frame_recv_ns is None:
            return
        self._pool_reported_at_close = True
        self._record_writer_pool(self._last_frame_recv_ns)

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

    def _route_frame(self, parsed, payload: str, t_recv_ns: int,
                     route_hint=None) -> None:
        """Send one parsed frame to its writer, tracker and - on a gap - the ledger.

        `route_hint` is the PollSpec that requested a polled frame. It wins over
        whatever `extract` could work out, because the subscription knows what
        it asked for and some REST bodies do not say.
        """
        meta = self._venue.extract(parsed)
        if meta.kind == "control":
            self._stats["control"] += 1

        if route_hint is not None:
            stream = _safe_path_token(route_hint.stream)
            symbol = _safe_path_token(route_hint.symbol)
        else:
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
        # Before the writers are closed, so `open_hours` describes the session
        # rather than the shutdown - a pool reported after everything is closed
        # says zero open hours on every run, which is true and useless.
        try:
            self._record_writer_pool_totals()
        except Exception as exc:
            errors.append(exc)
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        # Nothing is open any more, and the pool must agree: a stale entry here
        # would have a reopened writer evicted before it had written a frame.
        self._open_writers.clear()
        try:
            self._ledger.close()
        except Exception as exc:
            errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("errors while closing venue recorder resources", errors)
