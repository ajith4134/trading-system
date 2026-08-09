"""The polled source: what it yields, and how it joins the pushed one.

Polling exists because Binance withholds streams this host is subscribed to.
Measured 2026-08-03 across three edge IPs: `markPrice@1s`, `!markPrice@arr@1s`,
`forceOrder` and `aggTrade` delivered zero frames on `fstream.binance.com` while
`trade` and `depth` flowed on the same connections, and COIN-M served the
identical `markPrice@1s` stream normally. REST `/fapi/v1/premiumIndex` answers
200 with the same data, so the feed is recovered by asking for it instead.
"""
import asyncio
import json

import pytest

from capture.rest_poller import merge_frame_sources, poll_frames
from capture.venues import PollSpec
from capture.venues.binance import BinanceVenue

SPEC = PollSpec("binance", "premiumIndex", "BTCUSDT",
                "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT")
OTHER = PollSpec("binance", "premiumIndex", "ETHUSDT",
                 "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=ETHUSDT")


async def _drain(source, limit=None):
    out = []
    async for frame in source:
        out.append(frame)
        if limit is not None and len(out) >= limit:
            break
    return out


async def _emit(*frames, delay=0.0):
    for frame in frames:
        if delay:
            await asyncio.sleep(delay)
        yield frame


async def test_poll_frames_yields_the_response_body_verbatim():
    """The archive stores what the venue said. A body that is re-encoded on the
    way in - even to tidy it - can never be compared against the venue again."""
    body = '{"symbol":"BTCUSDT",  "markPrice":"63856.20000000" ,"time":1785778577000}'

    async def fetch(spec):

        url = spec.url
        return body

    frames = await _drain(poll_frames(object(), [SPEC], 0.01, 0.05, fetch=fetch), limit=1)
    # The frame now travels with the spec that requested it, because some REST
    # bodies name no symbol. The payload itself is still byte-for-byte the
    # venue's - that is what this test defends.
    assert [f.payload for f in frames] == [body]
    assert frames[0].spec is SPEC


async def test_poll_frames_polls_every_spec_on_each_tick():
    seen = []

    async def fetch(spec):

        url = spec.url
        seen.append(url)
        return "{}"

    await _drain(poll_frames(object(), [SPEC, OTHER], 0.01, 0.055, fetch=fetch))
    # Both endpoints on every tick, not one per tick round-robin: a symbol that
    # is only sampled every other interval has a mark price series with half the
    # resolution of the one the cadence promises.
    assert seen.count(SPEC.url) == seen.count(OTHER.url)
    assert seen.count(SPEC.url) >= 2


async def test_poll_frames_keeps_polling_after_an_endpoint_fails():
    """A 500 or a dropped connection must not end the run.

    The websocket source dying is a restart signal - the supervisor exists for
    that. A REST tick failing is ordinary, and if it propagated it would take
    the venue's depth and trade capture down with it. Prolonged failure is not
    swallowed: no frames means the stream goes silent, which the recorder
    already records against the ledger.
    """
    calls = []

    async def fetch(spec):

        url = spec.url
        calls.append(url)
        if len(calls) == 1:
            raise ConnectionError("first tick fails")
        return '{"symbol":"BTCUSDT"}'

    frames = [f.payload for f in
              await _drain(poll_frames(object(), [SPEC], 0.01, 0.06, fetch=fetch))]
    assert frames, "the poller gave up after one failure"
    assert all(frame == '{"symbol":"BTCUSDT"}' for frame in frames)


async def test_poll_frames_yields_nothing_for_the_tick_that_failed():
    """A failed fetch has no body. Yielding a placeholder would put a frame in
    the archive that the venue never sent."""
    async def fetch(spec):
        url = spec.url
        raise TimeoutError("endpoint down")

    assert await _drain(poll_frames(object(), [SPEC], 0.01, 0.05, fetch=fetch)) == []


async def test_poll_frames_stops_at_its_duration():
    async def fetch(spec):
        url = spec.url
        return "{}"

    started = asyncio.get_running_loop().time()
    await _drain(poll_frames(object(), [SPEC], 0.01, 0.05, fetch=fetch))
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_merge_frame_sources_yields_from_every_source():
    merged = await _drain(merge_frame_sources(_emit("a", "b"), _emit("c")))
    assert sorted(merged) == ["a", "b", "c"]


async def test_merge_frame_sources_ends_only_when_all_sources_end():
    merged = await _drain(merge_frame_sources(_emit("fast"), _emit("slow", delay=0.02)))
    assert sorted(merged) == ["fast", "slow"]


async def test_merge_frame_sources_does_not_let_a_slow_source_hold_up_a_fast_one():
    """The websocket carries thousands of frames a second and the poller one a
    second. If the merge alternated between them, depth capture would be paced
    by the REST tick."""
    fast = _emit(*[f"f{i}" for i in range(20)])
    slow = _emit("s0", "s1", delay=0.05)
    merged = await _drain(merge_frame_sources(fast, slow), limit=20)
    assert sum(1 for frame in merged if frame.startswith("f")) == 20


async def test_merge_frame_sources_propagates_a_source_failure():
    """The websocket dying must reach the supervisor. Swallowing it here would
    leave a process alive and capturing nothing."""
    async def failing():
        yield "one"
        raise ConnectionError("socket closed")

    with pytest.raises(ConnectionError):
        await _drain(merge_frame_sources(failing(), _emit("other")))


async def test_merge_frame_sources_with_no_sources_ends_immediately():
    assert await _drain(merge_frame_sources()) == []


# --- per-spec cadence --------------------------------------------------------
# A depth snapshot costs 20 request-weight against a 2400/minute budget; the
# premiumIndex poll costs 1. Running both at one shared cadence means either
# the funding feed is needlessly slow or the snapshot feed bans the IP.

@pytest.mark.asyncio
async def test_a_slow_spec_polls_less_often_than_a_fast_one():
    fast = PollSpec("binance", "premiumIndex", "BTCUSDT", "http://fast")
    slow = PollSpec("binance", "depthSnapshot", "BTCUSDT", "http://slow",
                    interval_seconds=0.20)
    calls: list[str] = []

    async def fetch(spec) -> str:

        url = spec.url
        calls.append(url)
        return json.dumps({"url": url})

    frames = [f async for f in poll_frames(
        BinanceVenue(), [fast, slow], interval_seconds=0.05,
        duration_seconds=0.42, fetch=fetch)]

    assert frames
    fast_calls = calls.count("http://fast")
    slow_calls = calls.count("http://slow")
    assert slow_calls >= 1, "the slow spec never polled at all"
    assert fast_calls > slow_calls, (fast_calls, slow_calls)


@pytest.mark.asyncio
async def test_a_spec_without_its_own_cadence_uses_the_shared_one():
    """Adding the field must not change how every existing spec behaves."""
    spec = PollSpec("binance", "premiumIndex", "BTCUSDT", "http://x")
    calls: list[str] = []

    async def fetch(spec) -> str:

        url = spec.url
        calls.append(url)
        return "{}"

    _ = [f async for f in poll_frames(BinanceVenue(), [spec],
                                      interval_seconds=0.05,
                                      duration_seconds=0.22, fetch=fetch)]
    assert len(calls) >= 3


@pytest.mark.asyncio
async def test_a_poll_that_cannot_afford_its_weight_is_skipped_not_delayed():
    """The limit is per-IP and exchange-wide, so one process's burst bans every
    process. A dropped poll is cheaper than a ban - and it is retried next tick
    rather than waiting a whole cadence."""
    from ops.rate_budget import RateBudget
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # Enough for the cheap spec, nowhere near the expensive one.
        budget = RateBudget(tmp, "binance", capacity=5, refill_per_second=0,
                            clock=lambda: 0.0)
        cheap = PollSpec("binance", "premiumIndex", "BTCUSDT", "http://cheap",
                         weight=1)
        dear = PollSpec("binance", "depthSnapshot", "BTCUSDT", "http://dear",
                        weight=50)
        seen = []

        async def fetch(spec):

            url = spec.url
            seen.append(url)
            return "{}"

        _ = [f async for f in poll_frames(
            BinanceVenue(), [cheap, dear], interval_seconds=0.02,
            duration_seconds=0.12, fetch=fetch, budget=budget)]

    assert "http://cheap" in seen
    assert "http://dear" not in seen, "spent 50 weight from a 5-weight budget"


def test_the_measured_weights_reach_the_specs():
    """Measured from x-mbx-used-weight: a limit=1000 snapshot is 50, the
    all-market funding poll is 10. A spec carrying the wrong weight makes the
    budget a decoration.

    The all-market form is why broad funding is affordable at all: per symbol it
    is weight 1, so 857 perps would be 857 a tick against a 2,400/minute budget.
    One request at 10 covers the universe for less than three symbols cost.
    """
    specs = {s.stream: s for s in BinanceVenue().poll_specs(["BTCUSDT"])}
    assert specs["premiumIndex"].weight == 10
    assert specs["depthSnapshot"].weight == 50
