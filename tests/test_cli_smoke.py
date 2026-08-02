"""Tests for the capture entry point - the only component that touches a venue.

Everything below the CLI is proven against synthetic frames. The test that
matters here is `test_captures_real_binance_frames`: it opens a real socket to a
real exchange and checks that what came off the wire is what landed on disk,
byte for byte. It is opt-in (CAPTURE_LIVE=1) because it needs a network and a
live market, and it fails loudly rather than passing quietly when it captured
nothing.

The offline tests drive the same `run_capture` path through a fake socket, so
the wiring - URL, subscribe messages, deadline, byte-exact storage - is
constrained without a network.
"""
import asyncio
import contextlib
import json
import os
import time
import types
from pathlib import Path

import pytest

from capture import cli
from capture.cli import main, run_capture
from capture.capture_ledger import read_all
from capture.raw_writer import hour_key, paths_for, read_pair
from capture.venues.binance import BinanceVenue
from capture.venues.hyperliquid import HyperliquidVenue


class FakeWebSocket:
    """One venue connection: hands back queued frames, then goes quiet.

    Going quiet (rather than closing) is what a real socket does between
    frames, and it is the condition the capture deadline has to break out of.
    """

    def __init__(self, frames: list[str]) -> None:
        self._frames = list(frames)
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        await asyncio.Event().wait()      # quiet stream: block until cancelled


class FakeConnect:
    """Stands in for `websockets.connect`, recording how it was called."""

    def __init__(self, socket: FakeWebSocket, handshake_seconds: float = 0.0) -> None:
        self.socket = socket
        self.handshake_seconds = handshake_seconds
        self.url: str | None = None
        self.kwargs: dict | None = None
        self.exited = False

    def __call__(self, url: str, **kwargs):
        self.url, self.kwargs = url, kwargs
        return self

    async def __aenter__(self) -> FakeWebSocket:
        await asyncio.sleep(self.handshake_seconds)
        return self.socket

    async def __aexit__(self, *exc_info) -> bool:
        self.exited = True
        return False


def install_fake_socket(monkeypatch, frames: list[str],
                        handshake_seconds: float = 0.0) -> FakeConnect:
    """Point the CLI at a fake connection and return the recorder for it."""
    connect = FakeConnect(FakeWebSocket(frames), handshake_seconds)
    monkeypatch.setattr(cli, "websockets", types.SimpleNamespace(connect=connect))
    return connect


def depth_frame(update_id: int, extra: str = "") -> str:
    """A Binance combined-stream depth frame, shaped like the real one."""
    return json.dumps({
        "stream": "btcusdt@depth@100ms",
        "data": {"e": "depthUpdate", "E": 1785650606214 + update_id, "s": "BTCUSDT",
                 "U": update_id * 10 + 1, "u": update_id * 10 + 10,
                 "pu": update_id * 10, "note": extra,
                 "b": [["63110.20", "3.551"]], "a": [["63110.30", "18.931"]]},
    })


async def finish_within(awaitable, seconds: float = 10):
    """Fail rather than hang when a bounded run never ends.

    Every offline test drives a socket that goes quiet once its frames run out -
    the exact condition the capture deadline exists to break out of. Without
    this bound, losing the deadline hangs the suite instead of failing it.
    """
    return await asyncio.wait_for(awaitable, timeout=seconds)


def read_stored_frames(root: Path, stream: str, symbol: str,
                       hours: set[str], venue: str = "binance") -> list[tuple[str, object]]:
    """Every (payload, entry) stored for one stream, across the hours it spans.

    A capture that straddles an hour boundary rotates files mid-run, so the
    caller passes the hour at the start and the hour at the end rather than
    guessing one.
    """
    pairs: list[tuple[str, object]] = []
    for hour in sorted(hours):
        raw, idx = paths_for(root, venue, stream, symbol, hour)
        if raw.exists():
            pairs.extend(read_pair(raw, idx))
    return pairs


# --------------------------------------------------------------------------
# the live test - the reason this file exists
# --------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("CAPTURE_LIVE") != "1",
                    reason="live venue test; set CAPTURE_LIVE=1 to run")
@pytest.mark.asyncio
async def test_captures_real_binance_frames(tmp_path: Path, monkeypatch):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])

    # Tee the socket so the test holds the exact text Binance sent and can
    # compare it against what came back off disk. Without this the test could
    # only prove "some JSON was stored", not that it was stored unaltered.
    on_the_wire: list[str] = []
    stream_frames = cli._stream_frames

    async def tee_stream_frames(*args, **kwargs):
        async for frame in stream_frames(*args, **kwargs):
            on_the_wire.append(frame)
            yield frame

    monkeypatch.setattr(cli, "_stream_frames", tee_stream_frames)

    started_ns = time.time_ns()
    stats = await run_capture(venue, specs, tmp_path, duration_seconds=15)
    finished_ns = time.time_ns()

    assert stats["written"] > 0
    assert stats["dropped"] == 0
    assert stats["malformed"] == 0
    assert stats["unwritable"] == 0
    # nothing may vanish between the socket and the disk
    assert stats["written"] == len(on_the_wire)

    hours = {hour_key(started_ns), hour_key(finished_ns)}
    raw_files = [paths_for(tmp_path, "binance", "depth", "BTCUSDT", hour)[0]
                 for hour in sorted(hours)]
    stored_bytes = sum(path.stat().st_size for path in raw_files if path.exists())
    pairs = read_stored_frames(tmp_path, "depth", "BTCUSDT", hours)
    assert len(pairs) > 0, "connected but captured no depth frames"
    assert stored_bytes > 0

    # byte-exactness: every stored line is the exact text the venue sent
    sent = set(on_the_wire)
    unaltered = [payload for payload, _ in pairs if payload in sent]
    assert len(unaltered) == len(pairs)

    first_payload, first_entry = pairs[0]
    body = json.loads(first_payload)["data"]
    assert body["e"] == "depthUpdate" and body["s"] == "BTCUSDT"
    assert first_entry.n == 0
    assert first_entry.t_exch_ms == body["E"]
    assert first_entry.seq == {k: body[k] for k in ("U", "u", "pu", "T") if k in body}
    assert started_ns <= first_entry.t_recv_ns <= finished_ns


# --------------------------------------------------------------------------
# offline: the same code path, driven through a fake socket
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_frames_connects_to_the_venue_url(monkeypatch):
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    connect = install_fake_socket(monkeypatch, [depth_frame(1), depth_frame(2)])

    frames = await finish_within(_collect(cli._stream_frames(venue, specs, 0.25)))

    assert connect.url == venue.ws_url(specs)
    assert frames == [depth_frame(1), depth_frame(2)]
    # Binance encodes the subscription in the URL; sending anything is a protocol error
    assert connect.socket.sent == []
    assert connect.exited


@pytest.mark.asyncio
async def test_stream_frames_sends_subscribe_messages_after_connecting(monkeypatch):
    """Hyperliquid subscribes over the socket, not in the URL - established by
    live probing. A capture that skips the sends connects and receives nothing."""
    venue = HyperliquidVenue()
    specs = venue.core_specs(["BTC"])
    control = json.dumps({"channel": "subscriptionResponse", "data": {}})
    connect = install_fake_socket(monkeypatch, [control])

    frames = await finish_within(_collect(cli._stream_frames(venue, specs, 0.25)))

    assert connect.url == venue.ws_url(specs)
    assert connect.socket.sent == [json.dumps(m) for m in venue.subscribe_messages(specs)]
    assert connect.socket.sent, "hyperliquid must subscribe over the socket"
    assert frames == [control]


@pytest.mark.asyncio
async def test_stream_frames_stops_when_the_duration_is_up(monkeypatch):
    """A quiet stream must still end the run. Without a deadline on the receive
    the capture would hang past its duration and never return stats."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    connect = install_fake_socket(monkeypatch, [depth_frame(1)])

    started = time.monotonic()
    frames = await finish_within(_collect(cli._stream_frames(venue, specs, 0.3)))
    elapsed = time.monotonic() - started

    assert frames == [depth_frame(1)]
    assert 0.3 <= elapsed < 5
    assert connect.exited, "the connection must be closed when the run ends"


async def _collect(frames) -> list[str]:
    return [frame async for frame in frames]


@pytest.mark.asyncio
async def test_stream_frames_does_not_spend_its_duration_connecting(monkeypatch):
    """The handshake is allowed 20 seconds, so a deadline started before it can
    be spent entirely on connecting - returning an empty capture from a venue
    that was working. The duration is the capture window, not the call."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    install_fake_socket(monkeypatch, [depth_frame(1)], handshake_seconds=0.4)

    frames = await finish_within(_collect(cli._stream_frames(venue, specs, 0.3)))

    assert frames == [depth_frame(1)]


@pytest.mark.asyncio
async def test_stream_frames_closes_the_connection_when_the_consumer_stops(monkeypatch):
    """A capture abandoned mid-run must not leak the socket."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    connect = install_fake_socket(monkeypatch, [depth_frame(1), depth_frame(2)])

    frames = cli._stream_frames(venue, specs, 30.0)
    async with contextlib.aclosing(frames):
        assert await frames.__anext__() == depth_frame(1)

    assert connect.exited


@pytest.mark.asyncio
async def test_run_capture_stores_frames_byte_exactly(tmp_path: Path, monkeypatch):
    """The whole point of the service: what the venue sent is what is on disk."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    sent = [
        depth_frame(1),
        depth_frame(2, extra="näïve € \\ \" quoted"),
        depth_frame(3).replace('{"stream"', '{\n  "stream"'),   # literal newline
    ]
    install_fake_socket(monkeypatch, sent)

    started_ns = time.time_ns()
    stats = await finish_within(run_capture(venue, specs, tmp_path, duration_seconds=0.3))
    hours = {hour_key(started_ns), hour_key(time.time_ns())}

    assert stats["written"] == 3
    assert stats["dropped"] == 0 and stats["malformed"] == 0

    pairs = read_stored_frames(tmp_path, "depth", "BTCUSDT", hours)
    assert [payload for payload, _ in pairs] == sent
    assert [entry.n for _, entry in pairs] == [0, 1, 2]


@pytest.mark.asyncio
async def test_run_capture_returns_the_recorders_own_stats(tmp_path: Path, monkeypatch):
    """The returned dict has to be the recorder's counters, not a hopeful shape."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    install_fake_socket(monkeypatch, [depth_frame(1), "not json at all"])

    stats = await finish_within(run_capture(venue, specs, tmp_path, duration_seconds=0.3))

    assert stats == {"written": 2, "dropped": 0, "control": 0,
                     "malformed": 1, "unwritable": 0}


@pytest.mark.asyncio
async def test_run_capture_closes_the_socket_when_recording_fails(tmp_path: Path,
                                                                  monkeypatch):
    """A recorder that raises leaves the frame stream suspended mid-connection.
    The socket has to be closed on the way out, or a supervisor restarting the
    capture stacks a dead connection per attempt against the venue's limit."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    connect = install_fake_socket(monkeypatch, [depth_frame(1), depth_frame(2)])

    class FailingRecorder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def consume(self, frames) -> None:
            async for _ in frames:
                raise RuntimeError("no space left on device")

        def stats(self) -> dict:
            return {}

    monkeypatch.setattr(cli, "VenueRecorder", FailingRecorder)

    with pytest.raises(RuntimeError):
        await finish_within(run_capture(venue, specs, tmp_path, duration_seconds=30.0))

    assert connect.exited, "the socket must be closed when the recorder fails"


@pytest.mark.asyncio
async def test_run_capture_keeps_what_it_captured_when_cancelled(tmp_path: Path,
                                                                 monkeypatch):
    """`--seconds 0` ends by Ctrl-C, which reaches a running capture as a
    cancellation. Everything already received must be flushed to disk and the
    socket closed, rather than left in a buffer that dies with the process."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    connect = install_fake_socket(monkeypatch, [depth_frame(1), depth_frame(2)])
    started_ns = time.time_ns()

    capture = asyncio.create_task(
        run_capture(venue, specs, tmp_path, duration_seconds=float("inf")))
    await asyncio.sleep(0.1)      # both frames land, then the stream goes quiet
    capture.cancel()
    with pytest.raises(asyncio.CancelledError):
        await capture

    hours = {hour_key(started_ns), hour_key(time.time_ns())}
    pairs = read_stored_frames(tmp_path, "depth", "BTCUSDT", hours)
    assert [payload for payload, _ in pairs] == [depth_frame(1), depth_frame(2)]
    assert connect.exited, "the socket must be closed when the capture is cancelled"


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------

def record_run_capture_calls(monkeypatch, stats: dict | None = None) -> list[dict]:
    """Replace run_capture with a recorder of how main called it."""
    calls: list[dict] = []

    async def record_call(venue, specs, root, duration_seconds,
                          silence_grace_seconds=60.0):
        calls.append({"venue": venue, "specs": specs, "root": root,
                      "duration_seconds": duration_seconds,
                      "silence_grace_seconds": silence_grace_seconds})
        return stats if stats is not None else {"written": 0}

    monkeypatch.setattr(cli, "run_capture", record_call)
    return calls


def test_main_captures_the_requested_venue_and_symbols(tmp_path: Path, monkeypatch, capsys):
    stats = {"written": 7, "dropped": 0, "control": 1, "malformed": 0, "unwritable": 0}
    calls = record_run_capture_calls(monkeypatch, stats)

    exit_code = main(["--venue", "binance", "--symbols", "BTCUSDT,ETHUSDT",
                      "--root", str(tmp_path), "--seconds", "12"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == stats
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["venue"], BinanceVenue)
    assert call["specs"] == BinanceVenue().core_specs(["BTCUSDT", "ETHUSDT"])
    assert call["root"] == Path(tmp_path)
    assert call["duration_seconds"] == 12.0


def test_main_selects_hyperliquid_by_name(tmp_path: Path, monkeypatch, capsys):
    calls = record_run_capture_calls(monkeypatch)

    assert main(["--venue", "hyperliquid", "--symbols", "BTC",
                 "--root", str(tmp_path), "--seconds", "1"]) == 0

    assert isinstance(calls[0]["venue"], HyperliquidVenue)
    assert calls[0]["specs"] == HyperliquidVenue().core_specs(["BTC"])


def test_main_runs_until_interrupted_when_seconds_is_zero(tmp_path: Path, monkeypatch, capsys):
    calls = record_run_capture_calls(monkeypatch)

    assert main(["--venue", "binance", "--symbols", "BTCUSDT",
                 "--root", str(tmp_path), "--seconds", "0"]) == 0

    assert calls[0]["duration_seconds"] == float("inf")


def test_main_ignores_padding_around_symbols(tmp_path: Path, monkeypatch, capsys):
    """A symbol with a stray space builds a channel the venue does not know, and
    the venue answers by sending nothing at all - a silent empty capture."""
    calls = record_run_capture_calls(monkeypatch)

    assert main(["--venue", "binance", "--symbols", " BTCUSDT , ETHUSDT ",
                 "--root", str(tmp_path), "--seconds", "1"]) == 0

    assert calls[0]["specs"] == BinanceVenue().core_specs(["BTCUSDT", "ETHUSDT"])


def test_main_refuses_a_symbol_list_with_nothing_in_it(tmp_path: Path, monkeypatch):
    """Subscribing to no streams connects and captures nothing forever."""
    record_run_capture_calls(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(["--venue", "binance", "--symbols", " , ", "--root", str(tmp_path)])

    assert exit_info.value.code == 2


def test_main_refuses_a_negative_duration(tmp_path: Path, monkeypatch):
    """`--seconds -5` is a typo, not a request to capture forever."""
    record_run_capture_calls(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(["--venue", "binance", "--symbols", "BTCUSDT",
              "--root", str(tmp_path), "--seconds", "-5"])

    assert exit_info.value.code == 2


def test_main_refuses_a_negative_silence_grace(tmp_path: Path, monkeypatch):
    """A negative grace collapses the threshold and reports every stream silent
    on its first check - the alert flood, from a typo."""
    record_run_capture_calls(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(["--venue", "binance", "--symbols", "BTCUSDT", "--root", str(tmp_path),
              "--silence-grace-seconds", "-1"])

    assert exit_info.value.code == 2


def test_main_exits_on_interrupt_without_a_traceback(tmp_path: Path, monkeypatch):
    """Ctrl-C is how the run-until-interrupted mode is meant to end. asyncio
    cancels the capture (flushing it) and re-raises KeyboardInterrupt here."""
    async def interrupt_the_capture(venue, specs, root, duration_seconds,
                                    silence_grace_seconds=60.0):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_capture", interrupt_the_capture)

    assert main(["--venue", "binance", "--symbols", "BTCUSDT",
                 "--root", str(tmp_path), "--seconds", "0"]) == 130


def test_main_refuses_an_unknown_venue(tmp_path: Path, monkeypatch):
    record_run_capture_calls(monkeypatch)

    with pytest.raises(SystemExit) as exit_info:
        main(["--venue", "coinbase", "--symbols", "BTCUSDT", "--root", str(tmp_path)])

    assert exit_info.value.code == 2


def test_main_passes_the_silence_grace_through(tmp_path: Path, monkeypatch, capsys):
    """A short run needs a short grace, or the silence check has nothing to say
    before the run is over."""
    calls = record_run_capture_calls(monkeypatch)

    assert main(["--venue", "binance", "--symbols", "BTCUSDT", "--root", str(tmp_path),
                 "--seconds", "20", "--silence-grace-seconds", "5"]) == 0

    assert calls[0]["silence_grace_seconds"] == 5.0


@pytest.mark.asyncio
async def test_run_capture_reports_a_silent_stream_within_the_run(tmp_path: Path,
                                                                  monkeypatch):
    """End to end through the real recorder: a venue that delivers only one of
    its subscribed streams must say so in the ledger by the time it returns."""
    venue = BinanceVenue()
    specs = venue.core_specs(["BTCUSDT"])
    install_fake_socket(monkeypatch, [depth_frame(1), depth_frame(2)])

    await finish_within(run_capture(venue, specs, tmp_path, duration_seconds=0.4,
                                    silence_grace_seconds=0.05))

    events = read_all(tmp_path, "binance", hour_key(time.time_ns()).split("T")[0])
    # Streams that delivered nothing at all. `depth` may also appear once the
    # socket goes quiet at the end of the run - it went quiet for many times the
    # 50ms grace this test uses - but it is reported with the frames it did
    # send, which is the distinction that matters to whoever reads this.
    never_spoke = {e.stream for e in events
                   if e.kind == "silent_stream" and e.detail["frames_received"] == 0}
    assert never_spoke == {"trade", "markPrice", "forceOrder"}
    assert "depth" not in never_spoke
