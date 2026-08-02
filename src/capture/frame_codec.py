"""Byte-exact framing helpers.

Raw payloads are stored one per line. A payload containing a literal newline
would break line alignment, so newlines are escaped and the index records that
it happened. Backslash is escaped first so unescaping is unambiguous.
"""
import json
from dataclasses import dataclass, asdict

_ESCAPES = (("\\", "\\\\"), ("\n", "\\n"), ("\r", "\\r"))


def escape_payload(payload: str) -> tuple[str, bool]:
    if "\n" not in payload and "\r" not in payload:
        return payload, False
    out = payload
    for raw, esc in _ESCAPES:
        out = out.replace(raw, esc)
    return out, True


def unescape_payload(payload: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(payload):
        ch = payload[i]
        if ch == "\\" and i + 1 < len(payload):
            nxt = payload[i + 1]
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == "r":
                out.append("\r"); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
        out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class IndexEntry:
    n: int
    t_recv_ns: int
    t_exch_ms: int | None
    seq: dict | None
    kind: str
    esc: bool


def encode_index_entry(entry: IndexEntry) -> str:
    return json.dumps(asdict(entry), separators=(",", ":"), sort_keys=True)


def decode_index_entry(line: str) -> IndexEntry:
    return IndexEntry(**json.loads(line))
