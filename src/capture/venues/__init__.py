from dataclasses import dataclass


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


@dataclass(frozen=True)
class ExtractedMeta:
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    stream: str
    symbol: str
