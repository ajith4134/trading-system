from dataclasses import dataclass


@dataclass(frozen=True)
class StreamSpec:
    venue: str
    stream: str
    symbol: str
    channel: str


@dataclass(frozen=True)
class ExtractedMeta:
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    stream: str
    symbol: str
