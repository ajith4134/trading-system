import json
import random
from pathlib import Path

import pytest

from capture.venue_recorder import VenueRecorder
from capture.sequencing import StalenessTracker
from capture.venues.binance import BinanceVenue
from capture.capture_ledger import (
    read_all, LedgerEvent, SEVERITY_CORRUPTING, SEVERITY_INFO,
    SEVERITY_OBSERVATION_LOSS,
)


async def _frames(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_writes_frames_and_counts_them(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    payloads = [json.dumps({"stream": "btcusdt@depth@100ms", "data": {
        "e": "depthUpdate", "E": 1785650606214, "s": "BTCUSDT",
        "U": 1 + i * 10, "u": 10 + i * 10, "pu": i * 10}}) for i in range(3)]

    await rec.consume(_frames(payloads))
    assert rec.stats()["written"] == 3
    assert rec.stats()["dropped"] == 0


@pytest.mark.asyncio
async def test_broken_chain_records_corrupting_ledger_event(tmp_path: Path):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    rec = VenueRecorder(venue, specs, tmp_path, clock_ns=lambda: 1785648600_000_000_000)

    good = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                "U": 1, "u": 10, "pu": 0}})
    broken = json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "BTCUSDT",
                                  "U": 50, "u": 60, "pu": 49}})
    await rec.consume(_frames([good, broken]))

    events = read_all(tmp_path, "binance", "2026-08-02")
    gaps = [e for e in events if e.kind == "gap"]
    assert len(gaps) == 1
    assert gaps[0].severity == SEVERITY_CORRUPTING


@pytest.mark.asyncio
async def test_malformed_frame_is_still_written(tmp_path: Path):
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    await rec.consume(_frames(["this is not json"]))

    assert rec.stats()["malformed"] == 1
    assert rec.stats()["written"] == 1          # written anyway, never discarded
    events = read_all(tmp_path, "binance", "2026-08-02")
    assert any(e.kind == "malformed" for e in events)


@pytest.mark.asyncio
async def test_unsafe_symbol_does_not_escape_output_directory(tmp_path: Path):
    """extract() reads `symbol` straight off the wire (e.g. body["s"]). A frame
    carrying "/" or ".." there must not be able to steer RawWriter's path into
    an unintended directory - it should fall back to the same "unknown" bucket
    already used when routing fields can't be determined."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    hostile = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "../../evil",
                                    "U": 1, "u": 10, "pu": 0}})
    await rec.consume(_frames([hostile]))

    assert rec.stats()["written"] == 1
    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    names = [p.name for p in raw_dir.iterdir()]
    assert names, "expected the frame to land somewhere under the venue's raw dir"
    assert all("evil" not in n for n in names)
    assert any(n.startswith("depth_unknown_") for n in names)
    # No directory traversal: nothing was created outside tmp_path's raw tree.
    assert not (tmp_path.parent / "evil").exists()


@pytest.mark.asyncio
async def test_casing_variants_of_same_symbol_share_one_writer(tmp_path: Path):
    """Two frames for the same logical (stream, symbol) but reported with
    different casing must collapse into a single writer/file, not silently
    split the stream across two files keyed by casing alone."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)
    upper = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                  "U": 1, "u": 10, "pu": 0}})
    lower = json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "btcusdt",
                                  "U": 11, "u": 20, "pu": 10}})
    await rec.consume(_frames([upper, lower]))

    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    depth_raw_files = [p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.suffix == ".zst" and p.name.endswith("ndjson.zst")]
    assert len(depth_raw_files) == 1, f"expected one depth file, got {[p.name for p in depth_raw_files]}"


@pytest.mark.asyncio
async def test_frame_iterator_failure_still_closes_writers_and_ledger(tmp_path: Path):
    """If the frame source itself raises mid-stream (e.g. a websocket error),
    everything already appended must still be flushed and closed rather than
    left in buffered zstandard writers that were never finalized."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    good = json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                                "U": 1, "u": 10, "pu": 0}})

    async def _flaky_frames():
        yield good
        raise ConnectionError("socket dropped")

    with pytest.raises(ConnectionError):
        await rec.consume(_flaky_frames())

    assert rec.stats()["written"] == 1
    raw_dir = tmp_path / "raw" / "binance" / "2026-08-02"
    raw_file = next(p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.name.endswith("ndjson.zst"))
    idx_file = next(p for p in raw_dir.iterdir() if p.name.startswith("depth_") and p.name.endswith("idx.zst"))

    from capture.raw_writer import read_pair
    pairs = read_pair(raw_file, idx_file)   # raises if the zstd frame was never finalized
    assert len(pairs) == 1
    assert pairs[0][0] == good


@pytest.mark.asyncio
async def test_close_still_closes_remaining_writers_and_ledger_when_one_fails(tmp_path: Path):
    """close() must attempt every writer (and the ledger) even if an earlier
    one raises - e.g. a disk-full during one writer's zstd footer write must
    not orphan every writer after it in iteration order, plus the ledger,
    with unflushed buffered data. The failure must still surface to the
    caller, just not at the cost of abandoning the rest of the cleanup."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    # Two distinct writers with open, buffered handles.
    w1 = rec._writer_for("depth", "BTCUSDT")
    w1.append('{"a":1}', 1785648600_000_000_000, None, None)
    w2 = rec._writer_for("aggTrade", "BTCUSDT")
    w2.append('{"a":2}', 1785648600_000_000_000, None, None)

    # An open ledger handle too.
    rec._ledger.record(LedgerEvent(
        ts_ns=1785648600_000_000_000, venue="binance", stream="depth",
        kind="info", severity=SEVERITY_INFO, detail={}))

    assert w1._raw_z is not None and w2._raw_z is not None
    assert rec._ledger._fh is not None

    def _boom() -> None:
        raise OSError("disk full")
    w1.close = _boom

    with pytest.raises(OSError, match="disk full"):
        rec.close()

    # w1 (first in iteration order) is the one that raised - w2 and the
    # ledger must still have been closed rather than left open/unflushed.
    assert w2._raw_z is None
    assert rec._ledger._fh is None


# --------------------------------------------------------------------------
# CRITICAL - one damaged hour must not take the whole venue down
# --------------------------------------------------------------------------

def _depth_frame(symbol: str, i: int) -> str:
    return json.dumps({"data": {"e": "depthUpdate", "E": i, "s": symbol,
                                "U": 1 + i * 10, "u": 10 + i * 10, "pu": i * 10}})


def _trade_frame(symbol: str, i: int) -> str:
    return json.dumps({"data": {"e": "trade", "E": i, "s": symbol, "t": i,
                                "p": "1", "q": "1"}})


def _chop_last_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-1])


@pytest.mark.asyncio
async def test_a_damaged_hour_does_not_stop_a_sibling_stream_recording(tmp_path: Path):
    """The blast radius that made the refusal worse than the bug it replaced.

    `append` refuses a damaged hour with HourFileNotAppendable. Unguarded, that
    unwound `consume` entirely: the recorder died on the FIRST frame of the next
    session, so an undamaged trades/ETHUSDT stream in the same session never had
    a file created at all, and every restart died identically. One torn hour on
    one stream became a permanent crash loop across every stream of the venue.
    """
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    assert first.stats()["written"] == 400

    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await second.consume(_frames(
        [_depth_frame("BTCUSDT", 500)] + [_trade_frame("ETHUSDT", i) for i in range(3)]))

    # The damaged stream is isolated and its loss counted, not swallowed.
    assert second.stats()["unwritable"] == 1
    assert second.stats()["written"] == 3

    # The sibling stream recorded normally - this is the assertion that used to
    # be impossible, because the session died before it ever opened a file.
    trade_raw, trade_idx = paths_for(tmp_path, "binance", "trade", "ETHUSDT",
                                     "2026-08-02T05")
    from capture.raw_writer import read_pair
    assert len(read_pair(trade_raw, trade_idx)) == 3

    # And the loss is in the ledger, not only in memory.
    events = read_all(tmp_path, "binance", "2026-08-02")
    unwritable = [e for e in events if e.kind == "unwritable_stream"]
    assert len(unwritable) == 1
    assert unwritable[0].severity == SEVERITY_CORRUPTING
    assert unwritable[0].detail["error"] == "HourFileNotAppendable"
    totals = [e for e in events if e.kind == "unwritable_stream_total"]
    assert totals and totals[-1].detail["frames_lost"] == 1


@pytest.mark.asyncio
async def test_a_quarantined_stream_is_not_retried_per_frame(tmp_path: Path):
    """The condition is persistent, so retrying costs a decompress per frame."""
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await second.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(20)]))

    assert second.stats()["unwritable"] == 20
    assert second.stats()["written"] == 0
    events = read_all(tmp_path, "binance", "2026-08-02")
    assert len([e for e in events if e.kind == "unwritable_stream"]) == 1, (
        "one event per quarantined stream, not one per frame")
    totals = [e for e in events if e.kind == "unwritable_stream_total"]
    assert totals[-1].detail["frames_lost"] == 20


@pytest.mark.asyncio
async def test_recording_resumes_after_the_hour_is_repaired(tmp_path: Path):
    """refuse -> repair -> resume, end to end through the recorder."""
    from capture.raw_writer import paths_for, reconcile_pair, read_pair

    venue = BinanceVenue()
    clock = lambda: 1785648600_000_000_000

    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(400)]))
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    blocked = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await blocked.consume(_frames([_depth_frame("BTCUSDT", 1)]))
    assert blocked.stats()["unwritable"] == 1

    outcome = reconcile_pair(raw, idx)
    assert outcome.raw_was_salvaged

    resumed = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path, clock_ns=clock)
    await resumed.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(5)]))
    assert resumed.stats()["written"] == 5
    assert resumed.stats()["unwritable"] == 0
    assert len(read_pair(raw, idx)) == 5


@pytest.mark.asyncio
async def test_a_non_capture_error_still_unwinds_the_loop(tmp_path: Path):
    """Isolation is for per-hour damage only.

    ENOSPC, a bad descriptor or MemoryError are whole-recorder conditions;
    treating them as per-stream would spin quietly instead of surfacing them.
    """
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        clock_ns=lambda: 1785648600_000_000_000)

    def _boom(*args, **kwargs):
        raise OSError("disk full")
    rec._writer_for("depth", "BTCUSDT").append = _boom

    with pytest.raises(OSError, match="disk full"):
        await rec.consume(_frames([_depth_frame("BTCUSDT", 1)]))


# --------------------------------------------------------------------------
# subscribed but silent
# --------------------------------------------------------------------------

def clock_advancing_by(start_ns: int, step_ns: int):
    """A clock that moves on every read, so a session can outrun a grace period
    without the test waiting out real time."""
    state = {"now": start_ns}

    def clock_ns() -> int:
        now = state["now"]
        state["now"] = now + step_ns
        return now

    return clock_ns


def silent_stream_events(root: Path) -> list:
    return [e for e in read_all(root, "binance", "2026-08-02")
            if e.kind == "silent_stream"]


@pytest.mark.asyncio
async def test_a_subscribed_stream_that_never_speaks_reaches_the_ledger(tmp_path: Path):
    """A stream we asked for and never heard from looks exactly like a healthy
    stream in a quiet market - an empty directory nobody notices. Measured
    2026-08-02: Binance delivered depth and nothing else, and only the frames
    that did arrive left any trace at all."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])      # depth, trade, markPrice, forceOrder
    rec = VenueRecorder(venue, specs, tmp_path, silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))

    await rec.consume(_frames([
        json.dumps({"data": {"e": "depthUpdate", "E": 1, "s": "BTCUSDT",
                             "U": 1, "u": 10, "pu": 0}}),
        json.dumps({"data": {"e": "depthUpdate", "E": 2, "s": "BTCUSDT",
                             "U": 11, "u": 20, "pu": 10}}),
    ]))

    events = silent_stream_events(tmp_path)
    assert {e.stream for e in events} == {"trade", "markPrice", "forceOrder"}
    assert all(e.severity == SEVERITY_OBSERVATION_LOSS for e in events)
    assert all(e.detail["symbol"] == "BTCUSDT" for e in events)
    assert all(e.detail["frames_received"] == 0 for e in events)


@pytest.mark.asyncio
async def test_a_stream_is_not_called_silent_during_the_startup_grace(tmp_path: Path):
    """Every stream is silent for the first moments of a session. Flagging that
    would put an event in the ledger on every single start."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    1_000_000_000))

    await rec.consume(_frames([_depth_frame("BTCUSDT", 1)]))

    assert silent_stream_events(tmp_path) == []


@pytest.mark.asyncio
async def test_a_venue_that_sends_nothing_at_all_is_still_recorded(tmp_path: Path):
    """The frame loop cannot notice this one: it never runs. Without a check on
    the way out, a session that captured absolutely nothing leaves an empty
    ledger - indistinguishable from a session that captured everything."""
    venue = BinanceVenue()
    # One tick at construction, one when close() looks at the time: the second
    # read has to be past the grace period for the check to have anything to say.
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    70_000_000_000))

    await rec.consume(_frames([]))

    assert {e.stream for e in silent_stream_events(tmp_path)} == {
        "depth", "trade", "markPrice", "forceOrder"}


@pytest.mark.asyncio
async def test_silence_is_recorded_once_not_on_every_frame(tmp_path: Path):
    """`consume` closes in a finally and callers close explicitly, and frames
    keep arriving after the grace expires. One event per stream per session, or
    the ledger drowns in the anomaly it is meant to surface."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))

    await rec.consume(_frames([_depth_frame("BTCUSDT", n) for n in range(1, 6)]))
    rec.close()

    assert len(silent_stream_events(tmp_path)) == 3


@pytest.mark.asyncio
async def test_silence_is_reported_during_the_run_not_only_at_shutdown(tmp_path: Path):
    """A `--seconds 0` capture runs for days. Learning at shutdown that a stream
    never spoke is learning far too late, so the check runs as frames arrive and
    the ledger is written before the session ends."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    40_000_000_000))
    recorded_mid_run = []

    async def frames_and_a_look_at_the_ledger():
        yield _depth_frame("BTCUSDT", 1)          # t0 + 40s: inside the grace
        yield _depth_frame("BTCUSDT", 2)          # t0 + 80s: grace has passed
        recorded_mid_run.extend(silent_stream_events(tmp_path))
        yield _depth_frame("BTCUSDT", 3)

    await rec.consume(frames_and_a_look_at_the_ledger())

    assert {e.stream for e in recorded_mid_run} == {"trade", "markPrice", "forceOrder"}


def gap_events(root: Path) -> list:
    return [e for e in read_all(root, "binance", "2026-08-02") if e.kind == "gap"]


@pytest.mark.asyncio
async def test_a_stream_that_speaks_once_and_then_dies_reaches_the_ledger(tmp_path: Path):
    """The condition the first version of this check could not see: `trade`
    fires once, and is thereafter permanently excluded from the silence check
    because it has a writer. A 24/7 recorder would never notice it died.

    Nothing frame-driven can catch this - a dead stream sends no frame for a
    tracker to run on - so the venue clock has to ask on its behalf.
    """
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                    5_000_000_000))

    await rec.consume(_frames(
        [json.dumps({"data": {"e": "trade", "E": 1, "s": "BTCUSDT", "t": 1}})]
        + [_depth_frame("BTCUSDT", i) for i in range(1, 40)]))

    dead = [e for e in silent_stream_events(tmp_path) if e.stream == "trade"]
    assert len(dead) == 1
    assert dead[0].detail["frames_received"] == 1        # it spoke, then died
    assert dead[0].detail["silent_for_seconds"] >= 60
    assert dead[0].severity == SEVERITY_OBSERVATION_LOSS


@pytest.mark.asyncio
async def test_a_healthy_bursty_trade_stream_does_not_alarm(tmp_path: Path):
    """Trades arrive in bursts with quiet stretches between them. An earlier
    round of this project turned exactly that shape into an alarm on 57% of
    frames; a silence check that repeats the mistake is worse than none."""
    venue = BinanceVenue()
    rng = random.Random(7)
    gaps_ns = [int((6.0 if rng.random() < 0.6 else 0.5) * 1e9) for _ in range(300)]
    # One tick at construction, one per frame, one when close() checks silence.
    ticks = [1785648600_000_000_000]
    for gap in gaps_ns + [1_000_000_000] * 3:
        ticks.append(ticks[-1] + gap)
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60, clock_ns=iter(ticks).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "trade", "E": i, "s": "BTCUSDT", "t": i}})
        for i in range(len(gaps_ns))]))

    # Without this the test is vacuous: a stream with no tracker at all raises
    # no alarms either, and an upper bound alone cannot tell the two apart.
    assert isinstance(rec._trackers[("trade", "btcusdt")], StalenessTracker)
    assert [e for e in silent_stream_events(tmp_path) if e.stream == "trade"] == []
    trade_gaps = [e for e in gap_events(tmp_path) if e.stream == "trade"]
    assert len(trade_gaps) <= len(gaps_ns) * 0.05, (
        f"{len(trade_gaps)} alarms on {len(gaps_ns)} healthy bursty frames")


@pytest.mark.asyncio
async def test_a_sparse_stream_is_judged_against_its_own_cadence(tmp_path: Path):
    """Liquidations arrive minutes apart. Judging that against the grace period
    would report a dead stream on a healthy one every session."""
    venue = BinanceVenue()
    every_180s = [1785648600_000_000_000 + i * 180_000_000_000 for i in range(6)]
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(every_180s + [every_180s[-1]]).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "forceOrder", "E": i,
                             "o": {"s": "BTCUSDT", "q": "1"}}})
        for i in range(len(every_180s) - 1)]))

    assert [e for e in silent_stream_events(tmp_path)
            if e.stream == "forceOrder"] == []


@pytest.mark.asyncio
async def test_a_stream_too_slow_to_learn_a_cadence_does_not_alarm_on_every_frame(
        tmp_path: Path):
    """A liquidation feed minutes between frames never builds a baseline (see
    test_a_stream_slower_than_the_stall_rule_never_learns_a_baseline), so its
    tracker measures every gap against the 5s floor and flags all of them,
    forever. One ledger event per frame for the life of the stream is the alarm
    storm this project has already been bitten by. A staleness report is only
    worth recording once the tracker has something to compare against; whether
    such a stream has died is answered by the silence check instead.
    """
    venue = BinanceVenue()
    every_180s = [1785648600_000_000_000 + i * 180_000_000_000 for i in range(9)]
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(every_180s + [every_180s[-1]]).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "forceOrder", "E": i,
                             "o": {"s": "BTCUSDT", "q": "1"}}})
        for i in range(len(every_180s) - 1)]))

    assert [e for e in gap_events(tmp_path) if e.stream == "forceOrder"] == []


@pytest.mark.asyncio
async def test_a_real_stall_on_a_settled_stream_is_still_recorded(tmp_path: Path):
    """The other half of that trade-off: once a stream has shown what its
    cadence is, a stall against it must still reach the ledger."""
    venue = BinanceVenue()
    ticks = [1785648600_000_000_000 + i * 1_000_000_000 for i in range(30)]
    ticks.append(ticks[-1] + 600_000_000_000)          # a 10 minute hole
    ticks.append(ticks[-1])
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60, clock_ns=iter(ticks).__next__)

    await rec.consume(_frames([
        json.dumps({"data": {"e": "trade", "E": i, "s": "BTCUSDT", "t": i}})
        for i in range(30)]))

    stalls = [e for e in gap_events(tmp_path) if e.stream == "trade"]
    assert len(stalls) == 1
    assert stalls[0].severity == SEVERITY_OBSERVATION_LOSS
    assert stalls[0].detail["gap_seconds"] == 600.0


@pytest.mark.asyncio
async def test_a_quarantined_stream_is_not_also_reported_dead(tmp_path: Path):
    """A stream whose hour cannot be written is still speaking. Reporting it
    silent as well would send whoever reads the ledger looking for a venue
    problem that does not exist - the frames are arriving, they just cannot be
    stored, which the unwritable events already say."""
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: 1785648600_000_000_000)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(20)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           silence_grace_seconds=60,
                           clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                       5_000_000_000))
    await second.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(30, 60)]))

    assert second.stats()["unwritable"] == 30
    assert [e for e in silent_stream_events(tmp_path) if e.stream == "depth"] == []


def ticks_from_gaps(start_ns: int, gaps_seconds: list[float],
                    trailing_silence_seconds: float) -> list[int]:
    """Clock reads for a session: construction, one per frame, one at close."""
    ticks = [start_ns, start_ns]
    for gap in gaps_seconds:
        ticks.append(ticks[-1] + int(gap * 1e9))
    ticks.append(ticks[-1] + int(trailing_silence_seconds * 1e9))
    return ticks


def trade_frames(count: int) -> list[str]:
    return [json.dumps({"data": {"e": "trade", "E": i, "s": "BTCUSDT", "t": i}})
            for i in range(count)]


@pytest.mark.asyncio
async def test_a_stall_does_not_teach_the_silence_threshold(tmp_path: Path):
    """The stall poisoning bug again, in the silence threshold this time.

    A settled 1s stream takes one genuine 600s stall and then dies for good.
    Folding that stall into the estimator made the threshold 1800s, so five
    minutes of true silence - five times the grace - reported nothing. The
    estimator has to describe what the stream routinely does, not what it did
    on its worst frame.
    """
    venue = BinanceVenue()
    gaps = [1.0] * 29 + [600.0]
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(ticks_from_gaps(1785648600_000_000_000,
                                                      gaps, 300.0)).__next__)

    await rec.consume(_frames(trade_frames(len(gaps) + 1)))

    dead = [e for e in silent_stream_events(tmp_path) if e.stream == "trade"]
    assert len(dead) == 1, "a settled stream that died was not reported"
    assert dead[0].detail["threshold_seconds"] == 60.0     # the stall taught nothing
    assert dead[0].detail["silent_for_seconds"] == 300.0
    assert dead[0].detail["frames_received"] == 31
    # the stall itself is still reported as a gap
    assert len([e for e in gap_events(tmp_path) if e.stream == "trade"]) == 1


@pytest.mark.asyncio
async def test_a_routinely_slow_stream_still_learns_its_own_cadence(tmp_path: Path):
    """The case that rules out simply excluding flagged gaps: a stream whose
    every gap is 200s would have every one of them excluded as a stall and
    would then be reported dead on a 60s grace, every session. A stall is a
    one-off; a slow cadence repeats, and a quantile is what tells them apart."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(ticks_from_gaps(1785648600_000_000_000,
                                                      [200.0] * 5, 100.0)).__next__)

    await rec.consume(_frames(trade_frames(6)))

    assert [e for e in silent_stream_events(tmp_path) if e.stream == "trade"] == []


@pytest.mark.asyncio
async def test_silence_is_reported_within_a_bounded_time_however_slow_the_stream(
        tmp_path: Path):
    """However long a stream's history says it may sleep, silence has to be
    reportable eventually - otherwise a stream can talk its way into never
    being checked again."""
    venue = BinanceVenue()
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(ticks_from_gaps(1785648600_000_000_000,
                                                      [3000.0] * 5, 4000.0)).__next__)

    await rec.consume(_frames(trade_frames(6)))

    dead = [e for e in silent_stream_events(tmp_path) if e.stream == "trade"]
    assert len(dead) == 1
    assert dead[0].detail["threshold_seconds"] == 3600.0        # the ceiling


@pytest.mark.asyncio
async def test_a_stream_that_speeds_up_is_judged_on_its_recent_cadence(tmp_path: Path):
    """Why the gap window is bounded rather than a running history.

    A quantile alone already outvotes a one-off stall, so a single old outlier
    proves nothing about the window. What the window is for is a stream that
    changes regime: 250 frames at 100s apart and then 200 at 1s apart is a
    stream now capable of being judged in seconds, and an unbounded history
    keeps it judged in minutes long after that stopped being true.
    """
    venue = BinanceVenue()
    gaps = [100.0] * 250 + [1.0] * 200
    rec = VenueRecorder(venue, venue.tail_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=iter(ticks_from_gaps(1785648600_000_000_000,
                                                      gaps, 61.0)).__next__)

    await rec.consume(_frames(trade_frames(len(gaps) + 1)))

    dead = [e for e in silent_stream_events(tmp_path) if e.stream == "trade"]
    assert len(dead) == 1, "judged on a cadence the stream has long outgrown"
    assert dead[0].detail["threshold_seconds"] == 60.0


# --------------------------------------------------------------------------
# CRITICAL - the quarantine is the damaged HOUR, not the stream forever
# --------------------------------------------------------------------------

HOUR_NS = 3600 * 10**9


def clock_from(ticks: list[int]):
    """A clock over a fixed script that holds its last value forever after.

    `close()` reads the clock an unbounded number of times (once per quarantined
    hour's total, once for the silence sweep), so a bare iterator turns a missing
    tick into StopIteration instead of a test failure.
    """
    remaining = iter(ticks)
    state = {"last": ticks[0]}

    def clock_ns() -> int:
        state["last"] = next(remaining, state["last"])
        return state["last"]

    return clock_ns


@pytest.mark.asyncio
async def test_a_quarantined_stream_is_re_armed_at_the_next_hour(tmp_path: Path):
    """`HourFileNotAppendable` names ONE hour's files. The quarantine was keyed
    per stream for the life of the process, so a single torn hour left behind by
    a prior crash cost that stream every remaining hour of the run - hours that
    were undamaged and would have opened cleanly. On a 24/7 recorder that is the
    whole stream, indefinitely, from one bad byte.
    """
    from capture.raw_writer import paths_for, read_pair

    venue = BinanceVenue()
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: 1785648600_000_000_000)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(2)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    t05 = 1785648600_000_000_000
    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           clock_ns=clock_from([t05, t05, t05 + HOUR_NS,
                                                t05 + 2 * HOUR_NS]))
    await second.consume(_frames([_depth_frame("BTCUSDT", i) for i in (9, 10, 11)]))

    # Only the damaged hour is lost; the two clean hours recorded.
    assert second.stats()["unwritable"] == 1
    assert second.stats()["written"] == 2
    for hour in ("2026-08-02T06", "2026-08-02T07"):
        r, i = paths_for(tmp_path, "binance", "depth", "BTCUSDT", hour)
        assert r.exists(), f"{hour} was refused although it was undamaged"
        assert len(read_pair(r, i)) == 1


@pytest.mark.asyncio
async def test_the_cost_of_a_quarantined_hour_reaches_the_ledger_at_rotation(
        tmp_path: Path):
    """A `--seconds 0` capture runs for weeks. Reporting the size of the loss
    only from `close()` means an unattended process never reports it at all, so
    the count for a quarantined hour is recorded when that hour rotates away.
    """
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: 1785648600_000_000_000)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(2)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    t05 = 1785648600_000_000_000
    totals_seen_mid_run = []

    async def frames_then_a_look_at_the_ledger():
        yield _depth_frame("BTCUSDT", 1)          # T05: refused
        yield _depth_frame("BTCUSDT", 2)          # T05: refused
        yield _depth_frame("BTCUSDT", 3)          # T06: rotation drains the total
        totals_seen_mid_run.extend(
            e for e in read_all(tmp_path, "binance", "2026-08-02")
            if e.kind == "unwritable_stream_total")

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           clock_ns=clock_from([t05, t05, t05, t05 + HOUR_NS]))
    await second.consume(frames_then_a_look_at_the_ledger())

    assert len(totals_seen_mid_run) == 1, "the loss was invisible until close()"
    assert totals_seen_mid_run[0].detail["frames_lost"] == 2
    assert totals_seen_mid_run[0].detail["hour"] == "2026-08-02T05"


@pytest.mark.asyncio
async def test_each_damaged_hour_is_refused_once_not_once_per_frame(tmp_path: Path):
    """Re-arming per hour must not become re-trying per frame: retrying puts one
    ledger event and one decompress of the damaged hour behind every frame.
    """
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    t05 = 1785648600_000_000_000
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: t05)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(2)]))
    raw05, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw05)

    hour_06 = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                            clock_ns=lambda: t05 + HOUR_NS)
    await hour_06.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(2)]))
    raw06, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T06")
    _chop_last_byte(raw06)

    # Ten frames into hour 05, ten into hour 06: two damaged hours, no more.
    ticks = [t05] + [t05] * 10 + [t05 + HOUR_NS] * 10
    both = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                         clock_ns=clock_from(ticks))
    await both.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(20)]))

    assert both.stats()["unwritable"] == 20
    events = read_all(tmp_path, "binance", "2026-08-02")
    refusals = [e for e in events if e.kind == "unwritable_stream"]
    assert len(refusals) == 2, "one refusal per damaged hour, not per frame"
    assert {e.detail["hour"] for e in refusals} == {"2026-08-02T05", "2026-08-02T06"}
    totals = [e for e in events if e.kind == "unwritable_stream_total"]
    assert {e.detail["hour"]: e.detail["frames_lost"] for e in totals} == {
        "2026-08-02T05": 10, "2026-08-02T06": 10}


@pytest.mark.asyncio
async def test_closing_twice_does_not_double_count_a_quarantined_hour(tmp_path: Path):
    """`consume` closes in a `finally` and callers close explicitly. The totals
    dict has to be cleared as it is recorded, or the second close reports the
    same lost frames again and the ledger overstates the incident.
    """
    from capture.raw_writer import paths_for

    venue = BinanceVenue()
    t05 = 1785648600_000_000_000
    first = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                          clock_ns=lambda: t05)
    await first.consume(_frames([_depth_frame("BTCUSDT", i) for i in range(2)]))
    raw, _ = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    _chop_last_byte(raw)

    second = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           clock_ns=lambda: t05)
    await second.consume(_frames([_depth_frame("BTCUSDT", 1)]))
    second.close()
    second.close()

    totals = [e for e in read_all(tmp_path, "binance", "2026-08-02")
              if e.kind == "unwritable_stream_total"]
    assert len(totals) == 1, f"{len(totals)} totals for one quarantined hour"
    assert totals[0].detail["frames_lost"] == 1


# --------------------------------------------------------------------------
# CRITICAL - silence detection and health reporting must share a scope
# --------------------------------------------------------------------------

def silent_streams_on(root: Path, date: str) -> list:
    return [e for e in read_all(root, "binance", date) if e.kind == "silent_stream"]


def a_four_day_session_with_three_dead_streams(root: Path):
    """depth speaks every 6h for four UTC days; the other three never speak."""
    venue = BinanceVenue()
    start = 1785648600_000_000_000                      # 2026-08-02T05:30:00Z
    ticks = [start + i * 6 * 3600 * 10**9 for i in range(16)]
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), root,
                        silence_grace_seconds=60,
                        clock_ns=clock_from([ticks[0]] + ticks))
    return rec, [_depth_frame("BTCUSDT", i) for i in range(len(ticks))]


FOUR_DAYS = ["2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"]


@pytest.mark.asyncio
async def test_a_stream_still_dead_the_next_day_is_reported_again(tmp_path: Path):
    """The scope mismatch that produced a clean bill of health over dead streams.

    Silence was recorded once per stream per SESSION; `build_report` reports one
    UTC DAY. A `--seconds 0` capture is one session for weeks, so a stream that
    died on day one left an event only in day one's ledger and every later day
    read `silent_streams=0`. Verified over four days with three of four streams
    dead: days two, three and four all reported healthy.

    The two scopes are made to agree by re-arming the check each UTC day, which
    is exactly the granularity the report reads at - one event per stream per
    day, not one per frame.
    """
    rec, frames = a_four_day_session_with_three_dead_streams(tmp_path)
    await rec.consume(_frames(frames))

    for date in FOUR_DAYS:
        assert {e.stream for e in silent_streams_on(tmp_path, date)} == {
            "trade", "markPrice", "forceOrder"}, f"{date} reported a clean venue"


@pytest.mark.asyncio
async def test_a_dead_stream_is_reported_once_a_day_not_once_a_frame(tmp_path: Path):
    """Re-arming per day must not become re-arming per frame: an event per frame
    drowns the ledger in the anomaly it exists to surface."""
    rec, frames = a_four_day_session_with_three_dead_streams(tmp_path)
    await rec.consume(_frames(frames))

    for date in FOUR_DAYS:
        assert len(silent_streams_on(tmp_path, date)) == 3, date


@pytest.mark.asyncio
async def test_a_stream_that_comes_back_is_not_reported_dead_the_next_day(
        tmp_path: Path):
    """Re-arming reports a stream that is STILL dead, not one that recovered."""
    venue = BinanceVenue()
    start = 1785648600_000_000_000                      # 2026-08-02T05:30:00Z
    day = 86400 * 10**9
    ticks = [start, start + 12 * 3600 * 10**9,          # day 1: depth only
             start + day, start + day + 3600 * 10**9]   # day 2: trade speaks again
    rec = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                        silence_grace_seconds=60,
                        clock_ns=clock_from([ticks[0]] + ticks))
    await rec.consume(_frames([
        _depth_frame("BTCUSDT", 0),
        _depth_frame("BTCUSDT", 1),
        _trade_frame("BTCUSDT", 2),
        _trade_frame("BTCUSDT", 3),
    ]))

    assert "trade" in {e.stream for e in silent_streams_on(tmp_path, "2026-08-02")}
    assert "trade" not in {e.stream for e in silent_streams_on(tmp_path, "2026-08-03")}


@pytest.mark.asyncio
async def test_the_health_report_still_sees_the_dead_streams_days_later(
        tmp_path: Path):
    """The end the operator actually reads. Verified before the fix: days two
    through four reported `status=present silent_streams=0 names=[]` while three
    of four streams had been dead since day one, and no alert was ever raised
    for them again.
    """
    from capture.capture_health import build_report, write_alerts

    rec, frames = a_four_day_session_with_three_dead_streams(tmp_path)
    await rec.consume(_frames(frames))

    for date in FOUR_DAYS:
        report = build_report(tmp_path, "binance", date,
                              free_bytes=10**12, daily_bytes=1.0)
        assert report["silent_streams"] == 3, date
        assert report["silent_stream_names"] == ["forceOrder", "markPrice", "trade"]
        # Only depth wrote bytes, so the venue-day total alone says "present".
        assert list(report["raw_bytes_by_stream"]) == ["depth_BTCUSDT"]
        assert write_alerts(tmp_path, report) >= 1, f"{date} raised no alert"

    reasons = [json.loads(line)["reason"]
               for line in (tmp_path / "health" / "alerts.ndjson")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert reasons.count("silent_streams") == len(FOUR_DAYS)


@pytest.mark.asyncio
async def test_a_double_start_costs_the_contended_hour_not_the_venue(tmp_path: Path):
    """A second `capture --venue binance` must not corrupt the first's files,
    and must not take itself down either: contention is a `RawCaptureError`
    over one hour's pair, so the hour is quarantined like any other damage to
    it and every other stream keeps recording.
    """
    from capture.raw_writer import paths_for, read_pair

    venue = BinanceVenue()
    t05 = 1785648600_000_000_000
    holder = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                           clock_ns=lambda: t05)
    holder._writer_for("depth", "BTCUSDT").append(
        '{"held":0}', t05, None, None)          # holds depth's hour 05 open

    intruder = VenueRecorder(venue, venue.core_specs(["BTCUSDT"]), tmp_path,
                             clock_ns=lambda: t05)
    await intruder.consume(_frames(
        [_depth_frame("BTCUSDT", 1)] + [_trade_frame("ETHUSDT", i) for i in range(3)]))
    holder.close()

    assert intruder.stats()["unwritable"] == 1
    assert intruder.stats()["written"] == 3, "a sibling stream was taken down too"

    refusals = [e for e in read_all(tmp_path, "binance", "2026-08-02")
                if e.kind == "unwritable_stream"]
    assert [e.detail["error"] for e in refusals] == ["HourHeldByAnotherWriter"]

    # The holder's hour is exactly what the holder wrote - no interleaving.
    raw, idx = paths_for(tmp_path, "binance", "depth", "BTCUSDT", "2026-08-02T05")
    assert [p[0] for p in read_pair(raw, idx)] == ['{"held":0}']
