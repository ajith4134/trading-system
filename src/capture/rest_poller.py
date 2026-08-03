"""A polled frame source, for feeds the venue will not push to this host.

Layer 0 records what a venue sends. This module exists because Binance does not
send some of what it was asked for. Measured 2026-08-03 from this box, across
three separate `fstream.binance.com` edge IPs, with `trade` flowing on the very
same sockets as a control:

    btcusdt@markPrice@1s      0 frames / 35s
    !markPrice@arr@1s         0 frames / 35s   (a guaranteed 1 Hz stream)
    btcusdt@forceOrder        0 frames / 35s
    btcusd_perp@markPrice@1s  25 frames / 25s  (COIN-M, same stream type)
    btcusdt@trade             2001 frames / 35s

The names are not the problem - they match Binance's own connector source. Nor
is it a bad edge: six connections landed on three distinct IPs and behaved
identically. `SUBSCRIBE` is no help either, because the endpoint accepts any
string at all: `btcusdt@thisIsNotAStream` returned `{"result": null}` and then
appeared in `LIST_SUBSCRIPTIONS`. So a subscription acknowledgement proves
nothing, and silence is the only symptom a wrong name or a withheld stream has.

REST answers with the same data (`/fapi/v1/premiumIndex`, 200, live values), so
the feed is recovered by asking on a cadence. What arrives is written verbatim
under the endpoint's own name, never re-labelled as the websocket stream it
replaces - the payload shape differs, and a single filename holding two shapes
cannot be decoded later without knowing which day produced which.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
from typing import AsyncIterator

_REQUEST_TIMEOUT_SECONDS = 15

# The sentinel a finished source pushes onto the merge queue. A module-level
# object() is used rather than None so that a source legitimately yielding None
# could never be mistaken for the end of that source.
_SOURCE_EXHAUSTED = object()


async def _fetch_text(url: str) -> str:
    """GET a URL and return its body verbatim.

    Imported lazily so that the poller can be exercised - and the rest of the
    capture service can run - without aiohttp being importable at module load.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()


async def _fetch_or_none(fetch, url: str) -> str | None:
    """One tick of one endpoint. A failure yields no frame, and no exception.

    Deliberately swallowing here: this source runs alongside the websocket in a
    single process, and letting a 500 or a reset connection propagate would take
    that venue's depth and trade capture down with it. The failure is not
    invisible - a poll that keeps failing produces no frames, and a stream
    producing no frames is exactly what `VenueRecorder._record_silent_streams`
    reports to the ledger and the status wall.
    """
    try:
        return await fetch(url)
    except Exception:
        return None


async def poll_frames(venue, specs, interval_seconds: float,
                      duration_seconds: float, fetch=None) -> AsyncIterator[str]:
    """Sample every spec once per interval and yield the bodies, verbatim.

    Every spec is polled on every tick, concurrently. Round-robin would halve
    the resolution of each symbol's series relative to the cadence the caller
    asked for, and mark price is a series whose whole value is its regularity.

    `duration_seconds` may be infinite, which is the run-until-interrupted mode.
    The deadline is re-checked before each tick and the sleep is clipped to it,
    so a long interval cannot run past the end of a bounded capture.
    """
    fetch = fetch or _fetch_text
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_seconds

    while True:
        tick_started = loop.time()
        if tick_started >= deadline:
            return

        bodies = await asyncio.gather(
            *(_fetch_or_none(fetch, spec.url) for spec in specs))
        for body in bodies:
            if body is not None:
                yield body

        # Measured from the start of the tick, not its end, so the cadence does
        # not drift by the request latency on every single poll.
        elapsed = loop.time() - tick_started
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        nap = max(0.0, interval_seconds - elapsed)
        if not math.isinf(remaining):
            nap = min(nap, remaining)
        await asyncio.sleep(nap)


async def merge_frame_sources(*sources) -> AsyncIterator[str]:
    """Interleave several frame sources into one, in arrival order.

    Arrival order, not round-robin: the websocket delivers thousands of frames a
    second and the poller one, and any scheme that waited for the slow source
    would pace depth capture off the REST tick.

    A source that raises propagates. That is the point for the websocket - the
    supervisor restarts a process whose socket died, and a merge that swallowed
    the error would leave a live process capturing nothing. The polled source
    never raises for an endpoint failure; see `_fetch_or_none`.
    """
    if not sources:
        return

    queue: asyncio.Queue = asyncio.Queue()

    async def pump(source) -> None:
        # `aclosing` rather than a bare `async for`: when this task is cancelled
        # - which is how the merge shuts down - the source generator is left
        # suspended inside whatever context it holds, and for the websocket
        # source that is an open connection that would survive until the
        # interpreter finalised it. Closing it here makes that deterministic.
        try:
            async with contextlib.aclosing(source):
                async for frame in source:
                    await queue.put(("frame", frame))
        except asyncio.CancelledError:
            raise
        except Exception as error:              # noqa: BLE001 - re-raised below
            await queue.put(("error", error))
        finally:
            # The queue is unbounded, so this cannot block even while the task
            # is being cancelled.
            queue.put_nowait(("done", _SOURCE_EXHAUSTED))

    tasks = [asyncio.create_task(pump(source)) for source in sources]
    live = len(tasks)
    try:
        while live:
            kind, value = await queue.get()
            if kind == "frame":
                yield value
            elif kind == "error":
                raise value
            else:
                live -= 1
    finally:
        # Reached on the error path and on early close by the consumer - a
        # bounded capture stops reading the moment its duration elapses, and a
        # pump left running would hold its websocket open until interpreter
        # shutdown.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
