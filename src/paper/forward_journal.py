"""The forward journal: what the paper engine did, durably, and how long ago.

Phase J's definition of done is *"paper engine running as a supervised process,
journalling fills under both accountings, visible on the wall"* — so this file is
the thing the status wall probes. It exists to make two different silences
distinguishable:

    an engine that is running and has found nothing to trade
    an engine that stopped three days ago

A board cannot tell those apart from fills alone, because both produce no fills.
So the heartbeat is written on **every poll**, whether or not anything happened,
and it carries the clock it was written at. The wall grades on that age. This is
Rule 8 applied to the one process whose failure looks exactly like quiet.

The box was off from 2026-08-10 to 2026-08-15 and the board went on looking
healthy the whole time, because serving a file and refreshing it are different
jobs. A heartbeat that is only written when something interesting happens repeats
that failure exactly.

**Fills are appended and fsynced, never buffered.** A journal that loses its tail
to a page cache on a reboot is a record of paper trading that disagrees with the
WAL beside it, and the disagreement would be read as a bug in the engine rather
than in the journal.

**Both accountings on every row, never one.** A fill row carrying a single price
has already made the choice the promotion gate is supposed to make.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

HEARTBEAT_FILE = "heartbeat.json"
FILLS_PREFIX = "fills-"


def _fsynced_append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class Heartbeat:
    """The engine's last sign of life, as read back off disk."""

    written_at_ns: int
    strategy: str
    makes_edge_claim: bool
    events_fed: int
    orders_submitted: int
    orders_rejected: int
    fills: int
    open_orders: int
    last_event_time_ns: int | None
    detail: str
    # What the poll COST, as opposed to what it found. `None` on a heartbeat
    # written before these were recorded, and None is not zero: zero means the
    # poll opened no fragment footers, which is the healthy answer, and reading
    # an absent measurement as the healthy answer is the exact failure Rule 8
    # exists to prevent.
    fragment_schema_reads: int | None = None
    poll_seconds: float | None = None

    def age_ns(self, now_ns: int) -> int:
        return int(now_ns) - self.written_at_ns


class ForwardJournal:
    """Writes what happened; never decides whether it was good."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def fills_path(self, now_ns: int) -> Path:
        """One file per UTC day, so a day can be read without the whole history."""
        day = dt.datetime.fromtimestamp(
            now_ns / 1_000_000_000, dt.timezone.utc).strftime("%Y-%m-%d")
        return self._root / f"{FILLS_PREFIX}{day}.ndjson"

    def record_fill(self, fill, *, now_ns: int, strategy: str,
                    makes_edge_claim: bool) -> None:
        """Append one fill, priced under both accountings.

        `makes_edge_claim` rides on the row rather than sitting in a header: rows
        get grepped, concatenated and loaded into frames one file at a time, and a
        header is the first thing that gets separated from its data.
        """
        _fsynced_append(self.fills_path(now_ns), {
            "at_ns": now_ns,
            "strategy": strategy,
            "makes_edge_claim": makes_edge_claim,
            "client_order_id": fill.client_order_id,
            "symbol": fill.symbol,
            "venue": fill.venue,
            "side": fill.side,
            "quantity": str(fill.quantity),
            "optimistic_price": str(fill.optimistic_price),
            "optimistic_liquidity": fill.optimistic_liquidity,
            "pessimistic_price": str(fill.pessimistic_price),
            "pessimistic_liquidity": fill.pessimistic_liquidity,
            "participation": str(fill.participation),
            "uncalibrated": fill.uncalibrated,
        })

    def record_heartbeat(self, *, now_ns: int, strategy: str,
                         makes_edge_claim: bool, events_fed: int,
                         orders_submitted: int, orders_rejected: int,
                         fills: int, open_orders: int,
                         last_event_time_ns: int | None, detail: str,
                         fragment_schema_reads: int | None = None,
                         poll_seconds: float | None = None) -> None:
        """Write the engine's liveness. Called on EVERY poll, findings or not.

        Written whole to a temp file and renamed, so a reader never sees a
        half-written heartbeat and grades the engine dead on a torn parse.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at_ns": now_ns, "strategy": strategy,
            "makes_edge_claim": makes_edge_claim, "events_fed": events_fed,
            "orders_submitted": orders_submitted,
            "orders_rejected": orders_rejected, "fills": fills,
            "open_orders": open_orders,
            "last_event_time_ns": last_event_time_ns, "detail": detail,
            "fragment_schema_reads": fragment_schema_reads,
            "poll_seconds": poll_seconds,
        }
        target = self._root / HEARTBEAT_FILE
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)


def read_heartbeat(root: Path) -> Heartbeat | None:
    """The last heartbeat, or `None` when the engine has never run.

    `None` rather than a zeroed Heartbeat: a zeroed one reads as "ran, did
    nothing", and the wall must be able to say NOT BUILT instead of OK-but-quiet.
    """
    try:
        payload = json.loads(
            (Path(root) / HEARTBEAT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Heartbeat(
            written_at_ns=int(payload["written_at_ns"]),
            strategy=payload["strategy"],
            makes_edge_claim=bool(payload["makes_edge_claim"]),
            events_fed=int(payload["events_fed"]),
            orders_submitted=int(payload["orders_submitted"]),
            orders_rejected=int(payload["orders_rejected"]),
            fills=int(payload["fills"]),
            open_orders=int(payload["open_orders"]),
            last_event_time_ns=(None if payload.get("last_event_time_ns") is None
                                else int(payload["last_event_time_ns"])),
            detail=payload.get("detail", ""),
            # `.get` rather than `[...]`: a heartbeat written before these
            # existed is still a valid heartbeat, and refusing it would grade a
            # running engine as dead over a field about its cost.
            fragment_schema_reads=(
                None if payload.get("fragment_schema_reads") is None
                else int(payload["fragment_schema_reads"])),
            poll_seconds=(None if payload.get("poll_seconds") is None
                          else float(payload["poll_seconds"])))
    except (KeyError, TypeError, ValueError):
        # A heartbeat we cannot parse is not a heartbeat. Reporting it as absent
        # makes the wall say NOT MEASURED, which is true; inventing defaults for
        # the missing fields would make it say something green.
        return None


def count_fills(root: Path) -> int:
    """Every fill journalled, across every day file present."""
    total = 0
    for path in sorted(Path(root).glob(f"{FILLS_PREFIX}*.ndjson")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        total += sum(1 for line in text.split("\n") if line.strip())
    return total
