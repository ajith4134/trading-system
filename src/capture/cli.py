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
import sys
from pathlib import Path
from typing import AsyncIterator

import websockets

from capture.venue_recorder import VenueRecorder
from capture.venues.binance import BinanceVenue
from capture.venues.hyperliquid import HyperliquidVenue

_VENUES = {"binance": BinanceVenue, "hyperliquid": HyperliquidVenue}

_OPEN_TIMEOUT_SECONDS = 20
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


async def run_capture(venue, specs, root: Path, duration_seconds: float,
                      silence_grace_seconds: float = 60.0) -> dict:
    """Record one venue for `duration_seconds` and report what was captured.

    `silence_grace_seconds` is how long a subscribed stream may deliver nothing
    before the recorder writes that fact to the ledger. It is exposed because a
    run shorter than the grace can never report a silent stream, which makes a
    short capture look clean when it is not.
    """
    recorder = VenueRecorder(venue, specs, root,
                             silence_grace_seconds=silence_grace_seconds)
    frames = _stream_frames(venue, specs, duration_seconds)
    # `aclosing` matters on the failure path: if consume() raises, the async
    # generator is left suspended inside its `async with websockets.connect(...)`
    # and the socket stays open until the interpreter finalises it. Closing it
    # here makes that deterministic.
    async with contextlib.aclosing(frames):
        await recorder.consume(frames)
    return recorder.stats()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture", description="Record a venue's raw frames to disk.")
    parser.add_argument("--venue", choices=sorted(_VENUES), required=True)
    parser.add_argument("--symbols", required=True,
                        help="comma-separated, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--root", default=str(Path.home() / "capture"))
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 means run until interrupted")
    parser.add_argument("--silence-grace-seconds", type=float, default=60.0,
                        help="how long a subscribed stream may deliver nothing "
                             "before that is recorded in the ledger")
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

    venue = _VENUES[args.venue]()
    specs = venue.core_specs(symbols)
    duration = args.seconds if args.seconds > 0 else float("inf")

    try:
        stats = asyncio.run(run_capture(venue, specs, Path(args.root), duration,
                                        args.silence_grace_seconds))
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
