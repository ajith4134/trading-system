"""Entry point that connects to a venue and records what it sends.

This is the only place in the service that touches a network. Everything it
receives is passed to `VenueRecorder` exactly as it arrived: no decoding,
re-encoding or normalising happens here, because a frame that is altered on the
way in can never be recovered later.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import AsyncIterator

import websockets

from capture.rest_poller import merge_frame_sources, poll_frames
from capture.universe_tracker import UniverseTracker
from ops.rate_budget import RateBudget
from capture.venue_recorder import VenueRecorder
from capture.venues import shard_by_url_budget
from capture.venues.binance import BinanceVenue
from capture.venues.binance_spot import BinanceSpotVenue
from capture.venues.hyperliquid import HyperliquidVenue

_VENUES = {
    "binance": BinanceVenue,                # USDs-M perpetual futures
    "binance-spot": BinanceSpotVenue,       # spot; the other leg of the basis
    "hyperliquid": HyperliquidVenue,
}

_OPEN_TIMEOUT_SECONDS = 20
_UNIVERSE_TIMEOUT_SECONDS = 20
_INTERRUPTED_EXIT_CODE = 130       # 128 + SIGINT, the shell convention


async def _stream_frames(venue, specs, duration_seconds: float) -> AsyncIterator[str]:
    """Yield raw text frames for a bounded duration, then stop.

    Two venue facts, both established by probing the real endpoints, are
    reconciled here: Binance encodes the subscription in the URL and returns no
    subscribe messages at all, while Hyperliquid needs them sent over the socket
    after connecting and answers with a control frame before any data. Sending
    whatever the venue asks for - including nothing - covers both.

    The deadline is enforced on each receive rather than by an outer timeout, so
    a stream that has gone quiet still ends the run at its duration instead of
    hanging until the next frame arrives. Cancelling `recv()` is documented as
    safe in websockets 17: no message is lost by timing it out.

    The duration is counted from the moment the subscription is in place, not
    from the call: a handshake is allowed to take up to `_OPEN_TIMEOUT_SECONDS`,
    which would otherwise be able to consume the entire capture window and
    return a run that captured nothing while the venue was working fine.

    `duration_seconds` may be infinite, which is how the run-until-interrupted
    mode is expressed; the receive then simply has no timeout.
    """
    loop = asyncio.get_running_loop()
    async with websockets.connect(venue.ws_url(specs),
                                  open_timeout=_OPEN_TIMEOUT_SECONDS) as socket:
        for message in venue.subscribe_messages(specs):
            await socket.send(json.dumps(message))
        deadline = loop.time() + duration_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                yield await asyncio.wait_for(
                    socket.recv(),
                    timeout=None if math.isinf(remaining) else remaining)
            except TimeoutError:
                return


TAIL_ALL = "ALL"


def fetch_instruments(venue) -> dict:
    """The venue's raw instrument listing, unparsed.

    Returns the payload rather than the parsed symbols so that one request can
    answer more than one question about the same listing. The symbols and the
    currency each is quoted in arrive together, and fetching twice could get two
    different answers with a listing landing between them - leaving a recorded
    quote map that covers symbols the recorded universe does not.

    This replaced a `fetch_universe` helper that returned only the parsed
    symbols. It was left in place at first and then deleted: with the caller
    moved over, nothing called it, and a plausible unused helper beside the one
    that is used is how `tail_specs` came to be built, tested and never run.

    Synchronous and blocking on purpose: this runs once, before the capture loop
    starts, and a tail built from a stale list is the thing it exists to prevent.
    """
    import urllib.request

    method, url, payload = venue.instruments_request()
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(request, timeout=_UNIVERSE_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())




async def run_capture(venue, specs, root: Path, duration_seconds: float,
                      silence_grace_seconds: float = 60.0,
                      poll_specs=(), poll_interval_seconds: float = 1.0,
                      stream_shards=None) -> dict:
    """Record one venue for `duration_seconds` and report what was captured.

    `silence_grace_seconds` is how long a subscribed stream may deliver nothing
    before the recorder writes that fact to the ledger. It is exposed because a
    run shorter than the grace can never report a silent stream, which makes a
    short capture look clean when it is not.

    `poll_specs` are the feeds this venue will not push to us and which are
    fetched over REST instead - see `capture.rest_poller`. They are handed to the
    recorder alongside the subscribed streams because the recorder tracks
    silence, gaps and writers by (stream, symbol) alone: a polled feed that dies
    is then reported by the same machinery that reports a dead websocket, rather
    than by a second copy of it.

    `stream_shards` is how the subscription is split across sockets. It exists
    because fstream refuses a request line past roughly 16.3KB with HTTP 414 -
    measured, see `capture.venues.shard_by_url_budget` - so the broad tail
    cannot be one connection. The recorder still receives the flat `specs`,
    because it tracks silence, gaps and writers by (stream, symbol) and does not
    care which socket a frame arrived on. Defaulting to a single shard keeps a
    core-only run byte-identical to what it was before the tail existed.

    A shard that dies propagates rather than being swallowed, and the supervisor
    restarts the process. That is deliberate: a live process capturing three
    quarters of the universe is worse than one that visibly failed, because
    nothing downstream can tell the difference from a quiet market.
    """
    shards = list(stream_shards) if stream_shards is not None else [specs]
    recorder = VenueRecorder(venue, [*specs, *poll_specs], root,
                             silence_grace_seconds=silence_grace_seconds)
    sources = [_stream_frames(venue, shard, duration_seconds)
               for shard in shards if shard]
    if poll_specs:
        # One budget per venue, on disk, so the capture process for binance and
        # the one for binance-spot cannot together spend more weight than the
        # per-IP limit allows. Sharing it is the whole point - a bucket in this
        # process's memory would protect nothing.
        sources.append(
            poll_frames(venue, poll_specs, poll_interval_seconds, duration_seconds,
                        budget=RateBudget(Path(root) / "ops", venue.name)))
    frames = sources[0] if len(sources) == 1 else merge_frame_sources(*sources)
    # `aclosing` matters on the failure path: if consume() raises, the async
    # generator is left suspended inside its `async with websockets.connect(...)`
    # and the socket stays open until the interpreter finalises it. Closing it
    # here makes that deterministic.
    async with contextlib.aclosing(frames):
        await recorder.consume(frames)
    return recorder.stats()


def _stop_gracefully_on_sigterm() -> None:
    """Make SIGTERM a clean shutdown rather than a torn hour.

    SIGINT alone is not enough, and the reason is not obvious. A non-interactive
    shell sets SIGINT to SIG_IGN for the children it starts in the background,
    and the disposition survives exec - so a recorder launched by a supervisor
    script cannot be interrupted by SIGINT at all, no matter what Python
    installs afterwards. Measured on this box: the supervisor's `kill -INT` was
    a no-op and it hung waiting for a child that never noticed.

    SIGTERM does land, and by default it kills the process where it stands.
    That leaves the zstd frame of every stream mid-write torn, which
    `RawWriter` then correctly refuses to append to - costing each affected
    stream the remainder of its hour. Also measured: the two busiest depth
    streams, seventeen minutes each.

    Raising KeyboardInterrupt routes SIGTERM into the shutdown path that already
    exists and is already tested, so `close()` runs, the zstd footers are
    written, and the `.writing` markers are removed.
    """
    def raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture", description="Record a venue's raw frames to disk.")
    parser.add_argument("--venue", choices=sorted(_VENUES), required=True)
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--tail-symbols", default="",
                        help="comma-separated symbols captured on the cheap "
                             "channels only (no depth). This is the broad tail: "
                             "widening it is cheap today and impossible to "
                             "backfill later")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 means run until interrupted")
    parser.add_argument("--silence-grace-seconds", type=float, default=60.0,
                        help="how long a subscribed stream may deliver nothing "
                             "before that is recorded in the ledger")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0,
                        help="cadence for feeds fetched over REST because the "
                             "venue will not push them (see capture.rest_poller)")
    args = parser.parse_args(argv)

    # A symbol carrying whitespace builds a channel name the venue does not
    # know, and a venue answers an unknown channel by sending nothing - an empty
    # capture that looks exactly like a quiet market. An empty list does the
    # same, so it is refused rather than run.
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        parser.error("--symbols must name at least one symbol")
    if args.seconds < 0:
        parser.error("--seconds cannot be negative (0 means run until interrupted)")
    if args.silence_grace_seconds < 0:
        # A negative grace makes every threshold collapse, so every stream is
        # reported silent on its first check - an alert flood from a typo.
        parser.error("--silence-grace-seconds cannot be negative")

    if args.poll_interval_seconds <= 0:
        # A zero or negative cadence is an unthrottled request loop against the
        # venue's REST API, which is a ban rather than a capture.
        parser.error("--poll-interval-seconds must be greater than zero")

    venue = _VENUES[args.venue]()
    specs = venue.core_specs(symbols)
    poll_specs = venue.poll_specs(symbols)

    # The core and the tail overlap on `trade` and `forceOrder`, so a symbol
    # named in both would be subscribed twice and every one of its frames
    # written twice. That corrupts the archive rather than enriching it, and it
    # is silent - two identical frames a millisecond apart look like a busy
    # market. `dict.fromkeys` drops repeats within the tail list too, and keeps
    # the order the operator gave.
    core_symbols = set(symbols)
    if args.tail_symbols.strip().upper() == TAIL_ALL:
        # A hand-maintained tail list goes stale the first time a symbol lists
        # or delists, and the stale day cannot be backfilled. Refusing beats
        # falling back to the core: six symbols out of several hundred,
        # captured silently, would leave the archive looking like a healthy run.
        try:
            # One fetch, both answers. The listing carries the symbols and the
            # currency each is priced in, and asking twice could get two answers
            # with a listing in between - a quote map covering symbols the
            # recorded universe does not.
            listing = fetch_instruments(venue)
            discovered = venue.parse_instruments(listing)
            quote_assets = venue.parse_quote_assets(listing)
        except Exception as exc:
            parser.error(f"--tail-symbols ALL could not read {args.venue}'s "
                         f"universe, so the tail would silently shrink to the "
                         f"core: {exc}")
        if not discovered:
            parser.error(f"{args.venue} reported an empty universe; refusing "
                         f"rather than capturing only the core")
        # Point-in-time membership, recorded before a single frame is captured.
        # Backtesting "watch every symbol" against today's list conditions on
        # survival, and no purge or embargo scheme catches it. Trivial now,
        # impossible to reconstruct later.
        #
        # The quote currencies go down with it and for the same reason. Which
        # pairs exist changes daily, so a map fetched later describes a different
        # universe - and without one, a dollar P&L cannot tell a lira-quoted pair
        # from a dollar-quoted one by any means that is not a guess.
        UniverseTracker(Path(args.root), venue.name).record_snapshot(
            sorted(discovered), time.time_ns(), quote_assets=quote_assets)
        tail_source = discovered
    else:
        tail_source = [s.strip() for s in args.tail_symbols.split(",") if s.strip()]

    tail_symbols = [symbol for symbol in dict.fromkeys(tail_source)
                    if symbol not in core_symbols]

    # The core gets a socket to itself. Depth is the expensive feed and the one
    # least recoverable if it drops, and sharing a connection with several
    # hundred tail streams would let a tail disconnect take it down too.
    stream_shards = [specs]
    if tail_symbols:
        tail_specs = venue.tail_specs(tail_symbols)
        stream_shards.extend(shard_by_url_budget(venue, tail_specs))
        specs = [*specs, *tail_specs]

    duration = args.seconds if args.seconds > 0 else float("inf")

    _stop_gracefully_on_sigterm()

    try:
        stats = asyncio.run(run_capture(venue, specs, Path(args.root), duration,
                                        args.silence_grace_seconds,
                                        poll_specs=poll_specs,
                                        poll_interval_seconds=args.poll_interval_seconds,
                                        stream_shards=stream_shards))
    except KeyboardInterrupt:
        # asyncio.run cancels the capture before re-raising, which unwinds
        # VenueRecorder.consume through its own `finally` and flushes every
        # writer. The frames are on disk; only the counters are lost.
        print("interrupted; captured frames were flushed to disk", file=sys.stderr)
        return _INTERRUPTED_EXIT_CODE
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
