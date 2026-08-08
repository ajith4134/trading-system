"""Records point-in-time universe membership.

Without this, any backtest over "all symbols" silently conditions on survival.
Exchanges do not reliably publish historical membership, so it must be captured
as it happens.

The record is append-only and permanent, so this module is biased hard in one
direction: a missing entry is recoverable, a wrong entry is not. Every
uncertainty below therefore resolves to "refuse and raise" rather than "write
something plausible". The caller decides whether a refusal is an incident - it
should be recorded in the capture ledger.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from capture.raw_writer import utc_date_of

KIND_LISTED = "listed"
KIND_DELISTED = "delisted"
KIND_SNAPSHOT = "snapshot"

# `parse_instruments` returns [] both for a malformed payload and for a
# genuinely empty universe, and no venue has ever been observed empty:
# Hyperliquid carries 232 perps and Binance several hundred (verified live
# 2026-08-02). Below this size the delisted fraction carries no signal, so the
# fraction guard does not apply - a 3-symbol dev universe losing 2 symbols is
# ordinary, a 300-symbol venue losing 200 is not.
_MIN_UNIVERSE_FOR_MASS_DELISTING_GUARD = 10
_DEFAULT_MAX_DELISTED_FRACTION = 0.5


class UniverseTrackerError(Exception):
    """Base for every condition that stops this module writing to the record."""


class ImplausibleUniverseSnapshot(UniverseTrackerError):
    """The observed universe cannot be believed, so it was not recorded."""


class OutOfOrderSnapshot(UniverseTrackerError):
    """A snapshot older than the recorded state would rewrite history."""


class UnreadableUniverseState(UniverseTrackerError):
    """The last-snapshot file is missing fields, mistyped, or not JSON."""


@dataclass(frozen=True)
class UniverseEvent:
    ts_ns: int
    venue: str
    symbol: str
    kind: str
    detail: dict


def diff_universe(previous: list[str], current: list[str],
                  venue: str, ts_ns: int) -> list[UniverseEvent]:
    before, after = set(previous), set(current)
    events = [UniverseEvent(ts_ns, venue, s, KIND_LISTED, {}) for s in sorted(after - before)]
    events += [UniverseEvent(ts_ns, venue, s, KIND_DELISTED, {}) for s in sorted(before - after)]
    return events


def _write_state_atomically(path: Path, payload: dict) -> None:
    """Replace the state file in one step, or leave the old one untouched.

    A truncate-in-place write that is interrupted leaves a state file that is
    either unreadable or - worse - readable and short, and the next run then
    emits a spurious mass listing or delisting into a permanent record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".last_snapshot.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class UniverseTracker:
    def __init__(self, root: Path, venue_name: str,
                 max_delisted_fraction: float = _DEFAULT_MAX_DELISTED_FRACTION) -> None:
        self._root = Path(root)
        self._venue = venue_name
        self._max_delisted_fraction = max_delisted_fraction

    def _dir_for(self, ts_ns: int) -> Path:
        return self._root / "universe" / self._venue / utc_date_of(ts_ns)

    def _state_path(self) -> Path:
        return self._root / "universe" / self._venue / "last_snapshot.json"

    def _read_state(self) -> tuple[int, list[str], dict[str, str]] | None:
        """Return (ts_ns, symbols, quote_assets) of the last recorded snapshot, or None.

        Anything unexpected raises rather than degrading to "no previous
        universe": that degradation would turn a damaged file into a full mass
        listing on the next snapshot. Note especially that a `symbols` field
        holding a string would pass a truthiness check and then be diffed
        character by character.
        """
        path = self._state_path()
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise UnreadableUniverseState(f"{path}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(state, dict):
            raise UnreadableUniverseState(f"{path}: expected a JSON object, got {type(state).__name__}")

        ts_ns, symbols = state.get("ts_ns"), state.get("symbols")
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise UnreadableUniverseState(f"{path}: 'ts_ns' is not an integer")
        if not isinstance(symbols, list) or not all(isinstance(s, str) and s for s in symbols):
            raise UnreadableUniverseState(f"{path}: 'symbols' is not a list of non-empty strings")

        # Absent is legitimate - every snapshot recorded before 2026-08-08 has no
        # quote map - but present-and-malformed is not, and the two must not
        # collapse into the same empty dict. A quote currency read off a damaged
        # state file decides which symbols reach a dollar P&L.
        quote_assets = state.get("quote_assets", {})
        if not isinstance(quote_assets, dict) or not all(
                isinstance(k, str) and k and isinstance(v, str) and v
                for k, v in quote_assets.items()):
            raise UnreadableUniverseState(
                f"{path}: 'quote_assets' is not a mapping of non-empty strings")
        return ts_ns, symbols, quote_assets

    def load_last(self, ts_ns: int) -> list[str]:
        """Membership as it was known at `ts_ns`.

        A snapshot recorded after `ts_ns` is not knowledge available at `ts_ns`;
        returning it would be exactly the lookahead this module exists to
        prevent.
        """
        state = self._read_state()
        if state is None or state[0] > ts_ns:
            return []
        return list(state[1])

    def _refuse_if_implausible(self, observed: list[str], previous: list[str]) -> None:
        if not observed:
            raise ImplausibleUniverseSnapshot(
                f"{self._venue}: empty universe. A failed fetch or a payload whose shape "
                f"changed parses to [] identically to a genuinely empty venue, and no venue "
                f"has been observed empty. Recording it would write "
                f"{len(previous)} delistings that cannot later be told from real ones.")

        delisted_count = len(set(previous) - set(observed))
        if (len(previous) >= _MIN_UNIVERSE_FOR_MASS_DELISTING_GUARD
                and delisted_count > self._max_delisted_fraction * len(previous)):
            raise ImplausibleUniverseSnapshot(
                f"{self._venue}: {delisted_count} of {len(previous)} symbols would be "
                f"delisted at once, above max_delisted_fraction="
                f"{self._max_delisted_fraction}. A partially parsed payload looks exactly "
                f"like this.")

    def load_last_quote_assets(self, ts_ns: int) -> dict[str, str]:
        """The symbol-to-quote-currency map as it was known at `ts_ns`.

        Empty for a snapshot recorded before quote assets were captured at all,
        and empty is the caller's problem to name rather than this module's to
        paper over: a filter that reads {} as "nothing is dollar-quoted" excludes
        the entire universe, and one that reads it as "everything is" admits 312
        lira-quoted pairs into a dollar P&L. `store.quote_currency` refuses on the
        empty map for exactly that reason.
        """
        state = self._read_state()
        if state is None or state[0] > ts_ns:
            return {}
        return dict(state[2])

    def record_snapshot(self, symbols: list[str], ts_ns: int,
                        quote_assets: dict[str, str] | None = None) -> list[UniverseEvent]:
        """Record membership at `ts_ns` and return what changed since the last one.

        Raises rather than recording anything when the observation cannot be
        believed. Nothing is written on a refusal, so the next believable
        snapshot still diffs against the last good universe.

        `quote_assets` maps symbol to the currency it is priced in, as the venue
        itself reports it. Recorded here because it is point-in-time in the same
        way membership is: a pair's quote currency is fixed, but which pairs exist
        is not, and a map fetched later covers a different set of symbols than the
        day being read. It is optional so that a venue adapter without one keeps
        working - a missing map is recoverable, a wrong one is not. Recording a
        snapshot without one CLEARS the map rather than carrying the previous one
        forward: a map kept beside a universe it was not fetched with describes a
        different set of symbols, and that is the mismatch this method exists to
        prevent. The loss is loud - `store.quote_currency` refuses an empty map
        instead of filtering with it.
        """
        # Written straight into the state file, so a non-integer here would
        # leave behind a state file this module itself refuses to read.
        if not isinstance(ts_ns, int) or isinstance(ts_ns, bool):
            raise TypeError(f"ts_ns must be an integer number of nanoseconds, got {ts_ns!r}")

        # A bare string passes an element-wise check - every character is a
        # non-empty string - and would be diffed character by character.
        if isinstance(symbols, str) or not isinstance(symbols, (list, tuple)):
            raise ImplausibleUniverseSnapshot(
                f"{self._venue}: symbols must be a list of strings, got {type(symbols).__name__}")
        if not all(isinstance(s, str) and s for s in symbols):
            raise ImplausibleUniverseSnapshot(
                f"{self._venue}: symbols must all be non-empty strings, got {symbols!r}")
        observed = sorted(set(symbols))

        # Restricted to the universe being recorded, not merged with it. A quote
        # map naming symbols this snapshot does not is a fetch that raced a
        # listing, and keeping the extras would let a caller iterate the map and
        # request a symbol no capture subscribed. A symbol in the universe and
        # absent from the map stays absent - `unknown` downstream, and counted.
        if quote_assets is None:
            quotes: dict[str, str] = {}
        else:
            if not isinstance(quote_assets, dict):
                raise ImplausibleUniverseSnapshot(
                    f"{self._venue}: quote_assets must be a mapping of symbol to "
                    f"quote currency, got {type(quote_assets).__name__}")
            if not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in quote_assets.items()):
                raise ImplausibleUniverseSnapshot(
                    f"{self._venue}: quote_assets must map non-empty strings to "
                    f"non-empty strings")
            in_universe = set(observed)
            quotes = {s: q for s, q in quote_assets.items() if s in in_universe}

        state = self._read_state()
        if state is not None and ts_ns < state[0]:
            raise OutOfOrderSnapshot(
                f"{self._venue}: snapshot at {ts_ns} is older than the recorded state at "
                f"{state[0]}; diffing against a later universe would invent listings.")
        previous = list(state[1]) if state is not None else []

        self._refuse_if_implausible(observed, previous)
        events = diff_universe(previous, observed, self._venue, ts_ns)

        # The events go down first and are fsynced before the state advances.
        # If the order were reversed and the append were then lost, the state
        # would already claim the new universe and no later diff would ever
        # re-emit the transition - it would be gone from the record entirely.
        # This way round the worst case is a repeated snapshot line and
        # duplicate events on the next run, which is visible and recoverable.
        snapshot_line = {"ts_ns": ts_ns, "kind": KIND_SNAPSHOT, "symbols": observed}
        # Only when there is one. An empty `quote_assets` on every historical
        # line would read as "the venue reported no quote for any symbol", which
        # is a different fact from "nobody asked yet".
        if quotes:
            snapshot_line["quote_assets"] = quotes
        lines = [snapshot_line]
        lines += [asdict(event) for event in events]
        blob = "".join(json.dumps(line, separators=(",", ":"), sort_keys=True) + "\n"
                       for line in lines)

        folder = self._dir_for(ts_ns)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "instruments.ndjson", "a", encoding="utf-8") as fh:
            fh.write(blob)          # one write call: no interleaving with a concurrent appender
            fh.flush()
            os.fsync(fh.fileno())

        payload = {"ts_ns": ts_ns, "symbols": observed}
        if quotes:
            payload["quote_assets"] = quotes
        _write_state_atomically(self._state_path(), payload)
        return events
