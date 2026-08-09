"""Descriptors must not scale with the universe.

Found 2026-08-09. A `RawWriter` holds one raw and one index descriptor for as
long as its hour is open, so a broad tail needs descriptors in proportion to the
number of symbols: 2,115 is over 4,000. The recorder inherited a soft `NOFILE`
of 1024 from `sudo -H bash -lc`, died on `OSError: [Errno 24] Too many open
files` with exactly 1024 open, was restarted by its supervisor and walked into
the same wall - twenty times, at roughly forty seconds of capture per cycle.

Raising the limit stopped the bleeding. This is what makes it survivable: the
open hours are a bounded pool, evicted least-recently-written first, and the
budget is derived from the descriptor limit the process was actually given.

Eviction is not loss. The hour file is finished properly and the next frame for
that stream reopens it and appends. What it costs is compression ratio, because
a shorter zstd frame has less history to reference. The tests below are mostly
about that claim: that the bound holds, and that nothing is lost when it bites.
"""
import json
from pathlib import Path

import pytest

from capture.raw_writer import RawWriter, read_pair, paths_for
from capture.venue_recorder import VenueRecorder, max_open_writers_for
from capture.venues.binance_spot import BinanceSpotVenue

# Inside one UTC hour, so nothing below is testing hour rotation by accident.
_BASE_NS = 1785648600_000_000_000


async def _frames(items):
    for item in items:
        yield item


async def _frames_probed(items, probe):
    """Yield each frame, then run `probe` once the recorder has handled it.

    `consume` closes every writer in a `finally`, so a test that calls it once
    per frame observes a recorder that has just shut down - it would see an empty
    pool and pass against a pool that never bounded anything. The probe runs
    inside the loop, which is the only place the pool is alive.
    """
    for item in items:
        yield item
        probe()


def trade(symbol: str, i: int) -> str:
    return json.dumps({"data": {"e": "trade", "E": i, "s": symbol, "t": i,
                                "p": "1", "q": "1"}})


def recorder(tmp_path: Path, symbols: list[str], cap: int) -> VenueRecorder:
    venue = BinanceSpotVenue()
    return VenueRecorder(venue, venue.tail_specs(symbols), tmp_path,
                         clock_ns=lambda: _BASE_NS, max_open_writers=cap)


def frames_on_disk(root: Path, symbol: str, stream: str = "trade") -> int:
    raw, idx = paths_for(root, "binance-spot", stream, symbol, "2026-08-02T05")
    return len(read_pair(raw, idx))


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------

def test_the_budget_comes_from_the_limit_the_process_was_actually_given():
    """The number that matters is the process's own soft limit. The crash
    happened because nothing had ever asked what it was."""
    # The inherited limit that killed the recorder: 384 hours, not 4,000.
    assert max_open_writers_for(1024) == (1024 - 256) // 2
    # The raised limit: far above what any venue here opens, so nothing evicts.
    assert max_open_writers_for(65536) > 30_000


def test_a_limit_too_small_to_work_with_is_floored_rather_than_honoured():
    """Below the floor the pool would evict on nearly every frame, which is not
    capturing. Better to sit above a tiny limit and fail loudly on a descriptor
    than to quietly grind."""
    assert max_open_writers_for(64) == 64
    assert max_open_writers_for(0) == 64


# --------------------------------------------------------------------------
# the bound
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_hours_never_exceed_the_budget(tmp_path: Path):
    """The whole claim. 50 symbols through a pool of 8."""
    symbols = [f"SYM{i}USDT" for i in range(50)]
    rec = recorder(tmp_path, symbols, cap=8)
    seen = []

    def probe():
        seen.append(sum(1 for writer in rec._writers.values() if writer.is_open))

    await rec.consume(_frames_probed(
        [trade(symbol, i) for i, symbol in enumerate(symbols)], probe))

    assert max(seen) == 8, "the pool never filled, so the bound was never tested"
    assert all(count <= 8 for count in seen), f"high water {max(seen)} against a budget of 8"


@pytest.mark.asyncio
async def test_every_frame_survives_being_evicted_and_reopened(tmp_path: Path):
    """Eviction finishes the hour and the next frame appends to the same pair.
    If that is wrong, it is wrong silently - the files still exist and still
    parse, they are just short."""
    symbols = [f"SYM{i}USDT" for i in range(20)]
    rec = recorder(tmp_path, symbols, cap=4)

    # Three full rounds in ONE consume, so eviction is what closes these hours -
    # across separate consume calls the shutdown in `finally` would do it, and
    # the test would pass without the pool existing at all.
    await rec.consume(_frames([
        trade(symbol, round_number * 100 + i)
        for round_number in range(3)
        for i, symbol in enumerate(symbols)]))

    assert rec.stats()["writers_evicted"] > 0, "nothing was evicted, so nothing was proven"
    assert rec.stats()["written"] == 60
    for symbol in symbols:
        assert frames_on_disk(tmp_path, symbol) == 3, symbol


@pytest.mark.asyncio
async def test_index_numbering_stays_in_step_across_an_eviction(tmp_path: Path):
    """`n` has to keep matching each frame's position in the raw file. A reopen
    that resumed from the wrong count would mislabel every later entry, and
    `read_pair` would not complain - it checks lengths, not that the numbering
    means anything."""
    symbols = [f"SYM{i}USDT" for i in range(6)]
    rec = recorder(tmp_path, symbols, cap=2)

    await rec.consume(_frames([
        trade(symbol, round_number * 100 + i)
        for round_number in range(4)
        for i, symbol in enumerate(symbols)]))

    raw, idx = paths_for(tmp_path, "binance-spot", "trade", "SYM0USDT", "2026-08-02T05")
    entries = read_pair(raw, idx)
    assert [entry.n for _, entry in entries] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_eviction_is_counted_so_thrash_is_visible(tmp_path: Path):
    """A pool that is quietly evicting on every frame looks exactly like a pool
    that is not evicting at all - same files, same counts, more CPU. The counter
    is the only difference."""
    symbols = [f"SYM{i}USDT" for i in range(10)]
    rec = recorder(tmp_path, symbols, cap=3)
    await rec.consume(_frames([trade(symbol, i) for i, symbol in enumerate(symbols)]))

    assert rec.stats()["writers_evicted"] == 7


@pytest.mark.asyncio
async def test_a_pool_wide_enough_evicts_nothing(tmp_path: Path):
    """The raised descriptor limit puts every venue here in this case, so this is
    the path that actually runs in production. It must cost nothing."""
    symbols = [f"SYM{i}USDT" for i in range(10)]
    rec = recorder(tmp_path, symbols, cap=64)
    await rec.consume(_frames([trade(symbol, i) for i, symbol in enumerate(symbols)]))

    assert rec.stats()["writers_evicted"] == 0


@pytest.mark.asyncio
async def test_the_least_recently_written_hour_is_the_one_evicted(tmp_path: Path):
    """Not an arbitrary one. LRU is what keeps the cost small: the evicted writer
    is the one that has gone longest without a frame, so its file is the smallest
    and its reopen the cheapest."""
    rec = recorder(tmp_path, ["AUSDT", "BUSDT", "CUSDT"], cap=2)
    open_after_each = []

    def probe():
        open_after_each.append(
            {key[1] for key, writer in rec._writers.items() if writer.is_open})

    await rec.consume(_frames_probed([
        trade("AUSDT", 1),
        trade("BUSDT", 2),
        trade("AUSDT", 3),      # A speaks again, so B is now the least recent
        trade("CUSDT", 4),      # ...and B is what C displaces
    ], probe))

    assert open_after_each[-1] == {"ausdt", "cusdt"}, "evicted the wrong hour"


# --------------------------------------------------------------------------
# the reopen, which is where the cost would hide
# --------------------------------------------------------------------------

def test_a_reopen_of_a_cleanly_closed_hour_does_not_re_read_the_file(tmp_path: Path):
    """`_count_frames_already_written` decompresses both files end to end. Paying
    that on every reopen would turn eviction into a read amplifier that gets
    worse as the hour fills - the opposite of a fix.
    """
    writer = RawWriter(tmp_path, "binance-spot", "trade", "BTCUSDT")
    for i in range(5):
        writer.append('{"e":"trade"}', t_recv_ns=_BASE_NS + i, t_exch_ms=None, seq=None)
    writer.close()

    def refuse(*args, **kwargs):
        raise AssertionError("re-read a file this writer had just closed itself")

    writer._count_frames_already_written = refuse
    writer.append('{"e":"trade"}', t_recv_ns=_BASE_NS + 99, t_exch_ms=None, seq=None)
    writer.close()

    raw, idx = paths_for(tmp_path, "binance-spot", "trade", "BTCUSDT", "2026-08-02T05")
    assert [entry.n for _, entry in read_pair(raw, idx)] == [0, 1, 2, 3, 4, 5]


def test_a_file_that_changed_size_underneath_is_read_rather_than_trusted(tmp_path: Path):
    """The remembered count is an optimisation, and the `stat` is what keeps it
    honest. Nothing should touch a live hour - so a size that moved means an
    assumption is wrong somewhere, and the answer to that is to go and read.
    """
    writer = RawWriter(tmp_path, "binance-spot", "trade", "ETHUSDT")
    writer.append('{"e":"trade"}', t_recv_ns=_BASE_NS, t_exch_ms=None, seq=None)
    writer.close()

    raw, _ = paths_for(tmp_path, "binance-spot", "trade", "ETHUSDT", "2026-08-02T05")
    raw.write_bytes(raw.read_bytes() + b"\x00")

    read_instead = []
    original = writer._count_frames_already_written

    def counting(*args, **kwargs):
        read_instead.append(True)
        return original(*args, **kwargs)

    writer._count_frames_already_written = counting
    # The appended byte makes the pair unreadable, which is the correct outcome:
    # the point is that it was READ rather than assumed intact.
    with pytest.raises(Exception):
        writer.append('{"e":"trade"}', t_recv_ns=_BASE_NS + 1, t_exch_ms=None, seq=None)
    assert read_instead, "trusted a remembered count against a file that had changed"


def test_a_fresh_writer_meeting_an_existing_hour_still_reads_it(tmp_path: Path):
    """The fast path must not reach across process restarts. After a crash the
    count on disk is the only truth, and a new writer has no memory to trust."""
    first = RawWriter(tmp_path, "binance-spot", "trade", "SOLUSDT")
    for i in range(3):
        first.append('{"e":"trade"}', t_recv_ns=_BASE_NS + i, t_exch_ms=None, seq=None)
    first.close()

    second = RawWriter(tmp_path, "binance-spot", "trade", "SOLUSDT")
    second.append('{"e":"trade"}', t_recv_ns=_BASE_NS + 9, t_exch_ms=None, seq=None)
    second.close()

    raw, idx = paths_for(tmp_path, "binance-spot", "trade", "SOLUSDT", "2026-08-02T05")
    assert [entry.n for _, entry in read_pair(raw, idx)] == [0, 1, 2, 3]


def test_a_writer_that_closed_with_an_error_refuses_to_resume_from_memory(tmp_path: Path):
    """After a partial close, what is on disk is exactly the unknown. Trusting a
    remembered count there is trusting the one number the failure calls into
    question."""
    writer = RawWriter(tmp_path, "binance-spot", "trade", "ADAUSDT")
    writer.append('{"e":"trade"}', t_recv_ns=_BASE_NS, t_exch_ms=None, seq=None)

    class FailingStream:
        def __init__(self, real):
            self._real = real

        def write(self, data):
            return self._real.write(data)

        def close(self):
            raise OSError("disk went away mid-footer")

    writer._idx_z = FailingStream(writer._idx_z)
    with pytest.raises(OSError):
        writer.close()

    assert writer._closed_hour is None, "resumed from memory after a failed close"
