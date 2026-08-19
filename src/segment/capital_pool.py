"""The shared margin pool two bot processes reserve against — CL-04, RL-040.

`portfolio_usdt` in the declaration is ONE budget and there are TWO bots spending
it. perp and spot are independent OS processes on independent poll loops, and that
is the whole difficulty: a budget shared by two processes is not arithmetic, it is
concurrency.

## The failure this module exists to prevent

A lost update. Each bot polls every six seconds across hundreds of symbols. If each
computes headroom from its own view of the world:

    perp reads   "800 USDT free"          spot reads   "800 USDT free"
    perp opens   50 USDT                  spot opens   50 USDT
    both believe 750 remains, and 700 does.

Nothing here is a rare interleaving. At this cadence, across this many symbols, it
is the normal case, and it drifts in the direction that always looks fine: the pool
reports more headroom than it has, so the bots keep trading and the breach is only
visible in a total nobody computes.

## The reservation, and why it comes before the fill

    take the exclusive lock
    replay the ledger to current held margin
    check pool headroom, then the per-bot cap
    append RESERVE, or refuse with a NAMED reason
    release the lock
    only then journal the fill

Reserving BEFORE journalling the fill is the ordering that makes the invariant
hold. The reverse - fill, then reserve - can journal a trade the pool cannot fund,
and there is no honest thing to do with it afterwards: unwinding it is a trade that
never happened, keeping it is a budget that is not one.

## Why flock and not a lock file

`fcntl.flock` is released by the kernel when the holder dies, however it dies. A
lock file has to be cleaned up by the process that made it, so a bot killed between
creating it and removing it leaves a lock nobody holds and nothing can take - and
the recovery for that is a human deleting a file at 3am. Both bots are on one box
and one filesystem, so flock is a real mutex here and costs microseconds.

## Orphans, and why a startup sweep is not optional

A RESERVE whose bot dies before its RELEASE holds margin forever. Nothing times it
out, because a timeout would also release live positions that are merely long-held.
So on startup each bot names the positions it actually has, and any reservation of
its own that no position matches is reclaimed. Without this, every crash
permanently shrinks the pool, and the shrinking looks exactly like a bot that has
grown cautious.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from segment.capital_accounting import DEFAULT_STATE_ROOT

LEDGER_FILENAME = "pool.ndjson"

RESERVE = "RESERVE"
RELEASE = "RELEASE"

# Named refusals. A bot that cannot open because the pool is empty must render as
# BROKE, never as a bot that found nothing - an abstention and an exhausted budget
# look identical on a tile and mean opposite things (Rule 8).
POOL_EXHAUSTED = "POOL_EXHAUSTED"
BOT_CAP_REACHED = "BOT_CAP_REACHED"
BELOW_MIN_MARGIN = "BELOW_MIN_MARGIN"
ALREADY_RESERVED = "ALREADY_RESERVED"

# Rewrite the ledger once it holds this many rows. It is append-only within a run,
# so a long-lived bot would otherwise replay a file that grows without bound on
# every single reservation. Compaction happens UNDER THE SAME LOCK as a reservation
# and writes only the live rows, so a reader can never see a partial ledger.
COMPACT_AFTER_ROWS = 20_000


@dataclass(frozen=True)
class PoolDecision:
    """The answer to one reservation request, and the pool state it RESULTS IN.

    Resulting, not prior. On a refusal nothing changed so the two are the same; on
    a grant these are the figures including this reservation, which is what a tile
    or a journal line wants - the state a reader would see if they looked now.
    """

    granted: bool
    reason: str
    margin_usdt: Decimal
    held_by_segment: dict
    headroom_usdt: Decimal

    @property
    def total_held_usdt(self) -> Decimal:
        return sum(self.held_by_segment.values(), Decimal(0))

    def as_dict(self) -> dict:
        return {
            "granted": self.granted,
            "reason": self.reason,
            "margin_usdt": str(self.margin_usdt),
            "held_by_segment": {k: str(v) for k, v in self.held_by_segment.items()},
            "headroom_usdt": str(self.headroom_usdt),
            "total_held_usdt": str(self.total_held_usdt),
        }


def ledger_path(state_root: Path | None = None) -> Path:
    return Path(state_root or DEFAULT_STATE_ROOT) / LEDGER_FILENAME


class _ExclusiveLedger:
    """The ledger open for read-modify-write, with the lock held throughout.

    Opened `a+` so the file is created if absent and the handle can both read the
    whole ledger and append to it. The lock is taken on the same descriptor that
    does the writing - locking one handle and writing through another is a lock
    that protects nothing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = False
        return False

    def rows(self) -> list:
        self.handle.seek(0)
        parsed = []
        for line in self.handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except ValueError:
                # A torn final line from a kill mid-write. Skipping it is right:
                # the reservation it would have described was never completed, so
                # no position exists against it.
                continue
        return parsed

    def append(self, row: dict) -> None:
        self.handle.seek(0, os.SEEK_END)
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def rewrite(self, rows: list) -> None:
        """Replace the ledger with `rows`, under the lock already held."""
        self.handle.seek(0)
        self.handle.truncate()
        for row in rows:
            self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())


def _replay(rows: list) -> dict:
    """Live reservations, keyed by (segment, venue, symbol).

    A RELEASE with no matching RESERVE is ignored rather than treated as an error:
    it is what a double close or a replayed journal produces, and it cannot make
    the total wrong in the dangerous direction.
    """
    live: dict = {}
    for row in rows:
        key = (row.get("segment"), row.get("venue"), row.get("symbol"))
        if row.get("event") == RESERVE:
            try:
                live[key] = Decimal(str(row["margin_usdt"]))
            except (KeyError, ValueError, TypeError):
                continue
        elif row.get("event") == RELEASE:
            live.pop(key, None)
    return live


def _held_by_segment(live: dict) -> dict:
    held: dict = {}
    for (segment, _venue, _symbol), margin in live.items():
        held[segment] = held.get(segment, Decimal(0)) + margin
    return held


class CapitalPool:
    """One shared budget, reserved against by whichever bot asks first.

    The pool holds no state of its own between calls. Everything is replayed from
    the ledger inside the lock, because a cached balance in one process is exactly
    the stale view this module exists to eliminate.
    """

    def __init__(self, path: Path | None = None, *,
                 state_root: Path | None = None) -> None:
        self.path = Path(path) if path is not None else ledger_path(state_root)

    # ------------------------------------------------------------- reading

    def held_by_segment(self) -> dict:
        """Live reserved margin per bot. Cheap, and safe to call for a display."""
        with _ExclusiveLedger(self.path) as ledger:
            return _held_by_segment(_replay(ledger.rows()))

    def headroom(self, declaration) -> Decimal:
        held = sum(self.held_by_segment().values(), Decimal(0))
        return declaration.portfolio_usdt - held

    # ------------------------------------------------------------- writing

    def reserve(self, *, segment: str, venue: str, symbol: str,
                margin_usdt: Decimal, declaration, at_ns: int) -> PoolDecision:
        """Claim margin for one intended position, or refuse and say why.

        Called BEFORE the fill is journalled. Every check runs against state
        replayed inside the lock, so two bots asking at the same instant are
        serialised rather than both reading the same headroom.
        """
        margin_usdt = Decimal(str(margin_usdt))
        with _ExclusiveLedger(self.path) as ledger:
            rows = ledger.rows()
            live = _replay(rows)
            held = _held_by_segment(live)
            total = sum(held.values(), Decimal(0))
            headroom = declaration.portfolio_usdt - total

            def decision(granted, reason):
                return PoolDecision(granted=granted, reason=reason,
                                    margin_usdt=margin_usdt,
                                    held_by_segment=dict(held),
                                    headroom_usdt=headroom)

            key = (segment, venue, symbol)
            if key in live:
                # One position per instrument per bot, so a second reserve for the
                # same key would overwrite the first and lose its margin from the
                # total while the position stayed open.
                return decision(False, ALREADY_RESERVED)

            if margin_usdt < declaration.min_margin_per_trade_usdt:
                return decision(False, BELOW_MIN_MARGIN)

            if margin_usdt > headroom:
                return decision(False, POOL_EXHAUSTED)

            cap = declaration.cap_for(segment)
            if held.get(segment, Decimal(0)) + margin_usdt > cap:
                return decision(False, BOT_CAP_REACHED)

            ledger.append({"at_ns": at_ns, "event": RESERVE, "segment": segment,
                           "venue": venue, "symbol": symbol,
                           "margin_usdt": str(margin_usdt)})
            if len(rows) + 1 >= COMPACT_AFTER_ROWS:
                self._compact(ledger)

            held[segment] = held.get(segment, Decimal(0)) + margin_usdt
            headroom -= margin_usdt
            return decision(True, "RESERVED")

    def release(self, *, segment: str, venue: str, symbol: str, at_ns: int) -> None:
        """Give the margin back. Idempotent, because a double close must be safe."""
        with _ExclusiveLedger(self.path) as ledger:
            if (segment, venue, symbol) not in _replay(ledger.rows()):
                return
            ledger.append({"at_ns": at_ns, "event": RELEASE, "segment": segment,
                           "venue": venue, "symbol": symbol})

    def reclaim_orphans(self, *, segment: str, open_keys, at_ns: int) -> list:
        """Release this bot's reservations that no open position matches.

        Run at startup, and ONLY against the calling bot's own rows. Sweeping
        another bot's reservations would be one process deciding what a running
        process does not own - the two bots do not share a view of each other's
        positions, and guessing is how a live position loses its margin.
        """
        wanted = {(venue, symbol) for venue, symbol in open_keys}
        reclaimed = []
        with _ExclusiveLedger(self.path) as ledger:
            for (held_segment, venue, symbol) in _replay(ledger.rows()):
                if held_segment != segment or (venue, symbol) in wanted:
                    continue
                ledger.append({"at_ns": at_ns, "event": RELEASE,
                               "segment": segment, "venue": venue,
                               "symbol": symbol, "detail": "orphan reclaimed"})
                reclaimed.append((venue, symbol))
        return reclaimed

    @staticmethod
    def _compact(ledger: _ExclusiveLedger) -> None:
        """Rewrite the ledger as its live reservations only.

        Under the lock the caller already holds. The rewritten rows keep their
        original `at_ns` so a reservation's age survives compaction - a compaction
        that restamped them would make every position look newly opened.
        """
        live = _replay(ledger.rows())
        originals = {}
        for row in ledger.rows():
            if row.get("event") == RESERVE:
                originals[(row.get("segment"), row.get("venue"),
                           row.get("symbol"))] = row.get("at_ns")
        ledger.rewrite([
            {"at_ns": originals.get(key, 0), "event": RESERVE, "segment": key[0],
             "venue": key[1], "symbol": key[2], "margin_usdt": str(margin)}
            for key, margin in live.items()])
