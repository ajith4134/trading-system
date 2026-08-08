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


@dataclass(frozen=True)
class ExtractedMeta:
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    stream: str
    symbol: str
