from dataclasses import dataclass

# The denomination a venue reports when it reports one per venue rather than per
# listing - hyperliquid's perps, all marked in USD. Named here rather than in the
# store's classifier so Layer 0 does not import Layer 1 to describe its own
# archive; the classifier holds the same string in its own dollar set.
QUOTE_USD = "USD"


class UrlBudgetTooSmall(Exception):
    """One subscription cannot fit the venue's URL budget even on its own.

    Raised rather than emitting an over-budget shard, because the venue answers
    an over-long request line with HTTP 414 and that failure is indistinguishable
    from a quiet market: no frames arrive, nothing is written, and no error
    surfaces unless someone was watching the connect. Refusing names the symbol.
    """


def shard_for_connection(venue, specs) -> list[list]:
    """Split `specs` into connections that fit whatever budget this venue has.

    Two budget shapes exist and a venue carries at most one. Binance's limit is
    the URL request line (`max_url_bytes` — streams ride in the URL), Bybit's is
    the cumulative characters of subscribe `args` sent over the socket
    (`max_subscribe_chars` — the URL never grows). A venue declaring neither
    gets one connection, which is a statement about Hyperliquid: it subscribes
    over the socket and documents no ceiling.

    This is the function call sites use; the two shard functions below are its
    mechanisms. Keeping the choice here means a new venue's budget shape is a
    venue declaration, not a CLI edit.
    """
    if getattr(venue, "max_subscribe_chars", None) is not None:
        return shard_by_subscribe_budget(venue, specs)
    return shard_by_url_budget(venue, specs)


def shard_by_subscribe_budget(venue, specs) -> list[list]:
    """Split `specs` so each connection's subscribe args fit the venue ceiling.

    Bybit documents at most 21,000 characters of `args` per public connection.
    The measure below is what the venue itself counts — the character length of
    every topic string plus the JSON list's own punctuation (a quote pair and a
    separating comma per element) — computed from the channel names rather than
    by rendering messages, so the answer cannot drift from what
    `subscribe_messages` later sends.

    Measured 2026-08-09: the full linear universe (805 topics) is 21,244
    characters, already past the line, and one probe connection over it did
    work. Sharding to the documented budget anyway costs one extra socket and
    removes the bet that the venue never starts enforcing its own ceiling.
    """
    specs = list(specs)
    if not specs:
        return []
    budget = venue.max_subscribe_chars

    def arg_chars(spec) -> int:
        return len(spec.channel) + 3      # "topic", -> 2 quotes + 1 comma

    shards: list[list] = []
    current: list = []
    current_chars = 0
    for spec in specs:
        cost = arg_chars(spec)
        if cost > budget:
            raise UrlBudgetTooSmall(
                f"{venue.name} subscription {spec.channel!r} needs {cost} "
                f"characters of args on its own, over the {budget}-character budget")
        if current and current_chars + cost > budget:
            shards.append(current)
            current, current_chars = [], 0
        current.append(spec)
        current_chars += cost
    shards.append(current)
    return shards


def shard_by_url_budget(venue, specs, max_url_bytes: int | None = None) -> list[list]:
    """Split `specs` into the fewest connections whose URLs each fit the budget.

    Measured on 2026-08-08 against the live venue and recorded in
    `~/research/binance-fstream-connection-limits.md`: fstream's binding limit is
    the length of the request line, **not** the 1024-stream cap Binance documents
    for spot. 928 streams (16,338 bytes) connect; 960 (16,886) return HTTP 414.
    The full perpetual tail is 1,138 streams, so it cannot be one socket.

    Packing is by measured bytes rather than by a symbol count on purpose. Symbol
    names vary in length and the universe changes daily, so a shard sized in
    symbols silently crosses the ceiling on the day a batch of long-named tokens
    lists - and discovers it as an outage rather than as a refusal.

    A venue that declares no `max_url_bytes` is returned as a single shard. That
    is not a default so much as a statement about Hyperliquid: it carries no
    channels in its URL and subscribes over the socket instead, so splitting
    would spend connections against a constraint it does not have.

    Order is preserved and every spec appears exactly once. A dropped spec is a
    symbol that is never captured, and Layer 0 cannot backfill.
    """
    specs = list(specs)
    if not specs:
        return []

    budget = max_url_bytes if max_url_bytes is not None else getattr(venue, "max_url_bytes", None)
    if budget is None:
        return [specs]

    def refuse(spec):
        raise UrlBudgetTooSmall(
            f"{venue.name} subscription {spec.channel!r} needs "
            f"{len(venue.ws_url([spec]))} bytes of URL on its own, over the "
            f"{budget}-byte budget")

    shards: list[list] = []
    current: list = []
    for spec in specs:
        if len(venue.ws_url([*current, spec])) <= budget:
            current.append(spec)
            continue
        if not current:
            refuse(spec)
        shards.append(current)
        current = [spec]
        if len(venue.ws_url(current)) > budget:
            refuse(spec)
    shards.append(current)
    return shards


@dataclass(frozen=True)
class StreamSpec:
    venue: str
    stream: str
    symbol: str
    channel: str


@dataclass(frozen=True)
class PollSpec:
    """One REST endpoint sampled on a fixed cadence, recorded like a stream.

    It carries `stream` and `symbol` for the same reason `StreamSpec` does: the
    recorder keys writers, gap trackers and silence detection off those two
    fields alone, so a polled source is tracked by the machinery that already
    exists rather than a parallel copy of it.

    A polled feed is not a pushed one and is never filed as though it were - the
    `stream` names the endpoint, not the websocket channel it stands in for.
    """
    venue: str
    stream: str
    symbol: str
    url: str
    method: str = "GET"
    # How often to fetch this one, when it should not share the run's cadence.
    # A depth snapshot costs 20 request-weight against a 2400/minute budget and
    # the funding poll costs 1; one shared cadence makes the cheap feed slow or
    # the expensive feed a ban. None means "use whatever the run was given".
    interval_seconds: float | None = None
    # Request weight this endpoint costs, measured from the venue's own
    # x-mbx-used-weight header rather than assumed. Binance charges weight, not
    # requests: a limit=1000 depth snapshot is 50 while premiumIndex is 1, so a
    # budget counting requests would let fifty snapshots through as cheaply as
    # fifty polls and earn the ban it exists to prevent.
    weight: int = 1
    # Request body, for endpoints that will not answer a GET. Hyperliquid's
    # `/info` is one POST with a JSON type discriminator and no query string at
    # all, so a poller that can only GET cannot reach its funding.
    body: str | None = None
    # True when one response covers many instruments and the venue adapter knows
    # how to split it - see `fan_out_poll`. The symbol on a fan-out spec names
    # the request, not an instrument, and no file is ever written under it.
    #
    # This is what makes broad funding affordable. Per symbol, `premiumIndex`
    # costs weight 1 and 857 perps would be 857 per tick; the all-market form is
    # one request at weight 10, so the whole universe costs less than three
    # symbols did.
    fan_out: bool = False


@dataclass(frozen=True)
class ExtractedMeta:
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    stream: str
    symbol: str
