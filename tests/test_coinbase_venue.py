"""Coinbase spot, and the blocked premise that turned out to be wrong.

The build plan carried this venue as blocked on a missing API key. Probed from
this host 2026-08-10, the market data needs none - only the private API does.
Every frame below is real, taken off `wss://ws-feed.exchange.coinbase.com` and
`https://api.exchange.coinbase.com/products` on that date.
"""
import json

import pytest

from capture.frame_codec import IndexEntry
from capture.venues.coinbase import CoinbaseVenue
from store.trade_bars import build_bars, extract_trades, is_tradeable

MATCH = {
    "type": "match", "trade_id": 833226786,
    "maker_order_id": "2c8b571b-82d9-4631-9494-84622b6f890d",
    "taker_order_id": "54595b54-1008-43a4-954c-2377fac52dd4",
    "side": "sell", "size": "0.01028384", "price": "1925.31",
    "product_id": "ETH-USD", "sequence": 101414083044,
    "time": "2026-08-10T06:52:33.774893Z",
}
LAST_MATCH = dict(MATCH, type="last_match", product_id="BTC-USD",
                  price="65193.93", size="0.0049402",
                  time="2026-08-10T06:52:32.913802Z")
L2UPDATE = {
    "type": "l2update", "product_id": "BTC-USD",
    "changes": [["buy", "65184.00", "1.00000000"],
                ["buy", "65189.60", "0.00000000"]],
    "time": "2026-08-10T06:52:33.900000Z",
}
HEARTBEAT = {
    "type": "heartbeat", "last_trade_id": 833226786, "product_id": "ETH-USD",
    "sequence": 101414083160, "time": "2026-08-10T06:52:34.000000Z",
}
SUBSCRIPTIONS = {
    "type": "subscriptions",
    "channels": [{"name": "matches", "product_ids": ["BTC-USD"],
                  "account_ids": None}],
}

PRODUCTS = [
    {"id": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD",
     "status": "online", "trading_disabled": False},
    {"id": "ETH-BTC", "base_currency": "ETH", "quote_currency": "BTC",
     "status": "online", "trading_disabled": False},
    {"id": "OLD-USD", "base_currency": "OLD", "quote_currency": "USD",
     "status": "delisted", "trading_disabled": True},
    {"id": "HALT-USD", "base_currency": "HALT", "quote_currency": "USD",
     "status": "online", "trading_disabled": True},
]


@pytest.fixture
def venue():
    return CoinbaseVenue()


# --- subscribing ----------------------------------------------------------

def test_the_whole_universe_rides_one_subscribe_message(venue):
    """The subscription goes over the socket, not in the URL. Measured
    2026-08-10: all 517 online product ids together are 5,578 characters, so
    this venue declares no shard budget and takes one connection."""
    specs = venue.core_specs(["BTC-USD", "ETH-USD"])
    messages = venue.subscribe_messages(specs)

    assert len(messages) == 1
    assert messages[0]["type"] == "subscribe"
    assert messages[0]["product_ids"] == ["BTC-USD", "ETH-USD"]
    assert set(messages[0]["channels"]) == {"matches", "level2_batch", "heartbeat"}


def test_the_url_carries_no_subscription(venue):
    specs = venue.core_specs(["BTC-USD"])
    assert venue.ws_url(specs) == "wss://ws-feed.exchange.coinbase.com"
    assert "BTC-USD" not in venue.ws_url(specs)


def test_nothing_to_subscribe_sends_nothing(venue):
    assert venue.subscribe_messages([]) == []


def test_the_heartbeat_is_subscribed_because_silence_is_ambiguous(venue):
    """One frame per product per second carrying the current sequence. Without
    it a product that simply has not traded and a product whose feed died are
    the same silence."""
    channels = {spec.channel for spec in venue.core_specs(["BTC-USD"])}
    assert "heartbeat" in channels


def test_the_broad_tail_takes_trades_only(venue):
    """Depth on 517 products is a different order of write volume, and there is
    no all-market form of it."""
    assert {s.channel for s in venue.tail_specs(["BTC-USD", "ETH-USD"])} == {"matches"}


def test_the_book_is_polled_because_the_socket_sends_it_only_once(venue):
    """The websocket sends one `snapshot` at subscribe and diffs after, and
    `store.book_snapshots` deliberately does not replay diffs - so without a
    poll the book dataset would hold one row per recorder restart."""
    specs = venue.poll_specs(["BTC-USD", "ETH-USD"])

    assert [s.symbol for s in specs] == ["BTC-USD", "ETH-USD"]
    assert all(s.stream == "depthSnapshot" for s in specs)
    assert "products/BTC-USD/book?level=2" in specs[0].url


def test_the_book_poll_is_five_minutes_not_one(venue):
    """Reference-price only by ARCHITECTURE §3b, and the cost is the archive
    rather than the request: measured 2026-08-10, one BTC-USD book is 355,923
    bytes compressed, so three symbols at 60s is 1.5 GB/day against 308 MB/day
    at 300s. `consolidated_price` judges staleness against each venue's own
    routine gap, so a five-minute book is not read as a dead one."""
    assert venue.poll_specs(["BTC-USD"])[0].interval_seconds == 300.0


def test_nothing_else_is_polled(venue):
    """Spot has no funding to poll for."""
    streams = {s.stream for s in venue.poll_specs(["BTC-USD"])}
    assert streams == {"depthSnapshot"}


# --- reading frames -------------------------------------------------------

def test_a_trade_carries_its_sequence_and_venue_clock(venue):
    meta = venue.extract(MATCH)
    assert meta.stream == "matches"
    assert meta.symbol == "ETH-USD"
    assert meta.kind == "data"
    assert meta.seq == {"sequence": 101414083044}
    # 2026-08-10T06:52:33.774893Z in milliseconds.
    assert meta.t_exch_ms == 1786344753774


def test_the_subscribe_time_trade_is_filed_with_the_trades(venue):
    """`last_match` is the most recent trade, sent once at subscribe. Filed
    elsewhere it would be lost, and it is the only trade some quiet products
    publish in an hour."""
    assert venue.extract(LAST_MATCH).stream == "matches"


def test_depth_carries_no_sequence_and_that_absence_is_recorded(venue):
    """An `l2update` has `product_id` and `changes` and nothing to chain on, so
    depth here is staleness-only - the hyperliquid shape, not the binance one.
    Recording a fabricated sequence would make the gap detector believe it had
    continuity it never had."""
    meta = venue.extract(L2UPDATE)
    assert meta.stream == "level2"
    assert meta.seq is None
    assert meta.symbol == "BTC-USD"


def test_the_venue_is_not_declared_binance_chained(venue):
    """The recorder picks a gap tracker off this capability rather than off a
    venue name. Claiming it here would chain depth updates that carry no
    sequence to chain."""
    assert getattr(venue, "depth_is_binance_chained", False) is False


def test_the_heartbeat_is_data_and_keyed_to_its_product(venue):
    meta = venue.extract(HEARTBEAT)
    assert (meta.stream, meta.symbol, meta.kind) == ("heartbeat", "ETH-USD", "data")
    assert meta.seq == {"sequence": 101414083160}


def test_a_control_frame_keeps_the_venues_own_word_for_it(venue):
    """So a message type the venue adds later shows up in the archive rather
    than being silently renamed to something it is not."""
    meta = venue.extract(SUBSCRIPTIONS)
    assert meta.kind == "control"
    assert meta.stream == "subscriptions"
    assert meta.symbol == "unknown"


def test_an_unparseable_frame_does_not_raise(venue):
    for junk in ("not a dict", None, {}, {"type": 7}):
        meta = venue.extract(junk)
        assert meta.kind == "control"


def test_an_unreadable_timestamp_is_absent_rather_than_guessed(venue):
    """The archive records receipt time regardless. A fabricated venue clock
    would silently reorder this tape against the other three."""
    assert venue.extract(dict(MATCH, time="not a time")).t_exch_ms is None
    assert venue.extract(dict(MATCH, time=None)).t_exch_ms is None


# --- the instrument list --------------------------------------------------

def test_only_markets_that_are_online_and_enabled_are_listed(venue):
    """Both conditions, not either: a market can be halted without being
    delisted, and subscribing to it would report a feed silent forever.
    Measured live - 832 listings, 517 online, 315 delisted."""
    assert venue.parse_instruments(PRODUCTS) == ["BTC-USD", "ETH-BTC"]


def test_the_quote_currency_comes_from_the_venues_field_never_the_symbol(venue):
    """`ETH-BTC` is not a dollar pair, and anything matching on a suffix has to
    be told that twice."""
    quotes = venue.parse_quote_assets(PRODUCTS)
    assert quotes == {"BTC-USD": "USD", "ETH-BTC": "BTC"}


def test_a_malformed_products_payload_lists_nothing_rather_than_raising(venue):
    assert venue.parse_instruments({"products": []}) == []
    assert venue.parse_quote_assets("nonsense") == {}


# --- the tape becomes bars ------------------------------------------------

def _entry(recv_ns):
    return IndexEntry(n=0, t_recv_ns=recv_ns, t_exch_ms=None, seq=None,
                      kind="data", esc=False)


def test_a_match_becomes_a_trade_the_bar_builder_can_use():
    """Archived and never read is the shape this project keeps paying for. The
    store needs its own extractor for this venue or the tape is bytes."""
    trades = extract_trades(json.dumps(MATCH), _entry(1_786_344_753_800_000_000),
                            venue="coinbase", symbol="ETH-USD")

    assert len(trades) == 1
    assert trades[0].price == 1925.31
    assert trades[0].size == 0.01028384
    assert trades[0].symbol == "ETH-USD"
    assert trades[0].event_time_ns == 1786344753774 * 1_000_000
    assert is_tradeable(trades[0])


def test_a_trade_missing_its_price_or_clock_is_refused():
    """A zero-price fill would price everything that reads the bar it landed
    in - the defect that put `low <= 0` into 746 of the store's first bars."""
    for broken in ({**MATCH, "price": None}, {**MATCH, "size": None},
                   {**MATCH, "time": "nonsense"}):
        assert extract_trades(json.dumps(broken), _entry(1), venue="coinbase",
                              symbol="ETH-USD") == []


def test_depth_frames_are_not_mistaken_for_trades():
    for frame in (L2UPDATE, HEARTBEAT, SUBSCRIPTIONS):
        assert extract_trades(json.dumps(frame), _entry(1), venue="coinbase",
                              symbol="BTC-USD") == []


def test_coinbase_trades_build_bars_keyed_to_their_own_venue():
    """BTC-USD here and BTCUSDT on binance are different instruments at
    different prices. `build_bars` keys on (symbol, venue) so they stay
    distinct."""
    trades = extract_trades(json.dumps(MATCH), _entry(1_786_344_753_800_000_000),
                            venue="coinbase", symbol="ETH-USD")
    bars = build_bars(trades, interval_ns=60_000_000_000)

    assert len(bars) == 1
    assert bars.iloc[0]["venue"] == "coinbase"
    assert bars.iloc[0]["symbol"] == "ETH-USD"
    assert bars.iloc[0]["close"] == 1925.31


@pytest.mark.asyncio
async def test_a_channel_archived_under_another_name_is_not_recorded_silent(tmp_path):
    """`level2_batch` is subscribed; its frames arrive filed as `level2`.

    The recorder judges silence on the name it SUBSCRIBED and records liveness
    on the name a frame ARRIVES under, and for this channel those differ - the
    archive stream comes from the venue's own `type` field, so the string
    `level2_batch` never appears on a frame. Before `silence_stream_aliases`,
    that made the channel silent by construction: measured 2026-08-10,
    coinbase/level2 scored 0.000 delivery with 3 of 3 symbols silent while the
    feed was writing 565 KB an hour and its newest file had been touched two
    seconds earlier.

    Worse than a wrong tile, it could not heal. `capture_health`'s recovery rule
    clears a silence event that a later write postdates, but it looks the write
    up by the archive name - so it searched for `level2_batch_BTC-USD`, found
    nothing, and read the default 0 as "no write has ever postdated this".
    """
    from capture.capture_ledger import read_all
    from capture.venue_recorder import VenueRecorder

    def clock_advancing_by(start_ns: int, step_ns: int):
        state = {"now": start_ns}

        def clock_ns() -> int:
            now = state["now"]
            state["now"] = now + step_ns
            return now
        return clock_ns

    async def frames(items):
        for item in items:
            yield item

    venue = CoinbaseVenue()
    # One second per clock read, against a five-second grace. The step has to be
    # small enough that no stream outruns the grace before the first frame is
    # processed: silence is recorded once per stream per UTC day and is never
    # revised, so a coarse clock records every stream silent during startup and
    # the run afterwards can no longer say anything. That is a property of the
    # test harness, not of the recorder - with a 40-second step this test
    # reported the defect it was written to prove absent.
    recorder = VenueRecorder(venue, venue.core_specs(["BTC-USD"]), tmp_path,
                             silence_grace_seconds=5,
                             clock_ns=clock_advancing_by(1785648600_000_000_000,
                                                         1_000_000_000))

    await recorder.consume(frames([json.dumps(L2UPDATE) for _ in range(5)]))
    recorder.close()

    silent = {event.stream for event in read_all(tmp_path, "coinbase", "2026-08-02")
              if event.kind == "silent_stream"}

    assert "level2_batch" not in silent, (
        "the channel spoke - its frames were archived as level2 - so recording "
        "it silent is the false-silence defect this alias exists to prevent")
    # Not a vacuous assertion: the two channels that genuinely received nothing
    # ARE recorded, which is what makes the absence above meaningful.
    assert silent == {"matches", "heartbeat"}
