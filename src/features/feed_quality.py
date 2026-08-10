"""A quality score per feed, so feed health is a number rather than a habit.

`FEATURES.md` §1 marks this `[MISSED]`: **feed health as a first-class metric,
not an assumption.** The assumption it names is the one every system makes by
default - that data which arrived is data that can be traded on.

**Control channels are not feeds and are not scored.** A subscribe-ack
stream writes once at connect and never again, which is correct behaviour and
reads as total staleness against any freshness test. They are recognised by
the archive rather than by a list of names: the recorder files every control
frame under the symbol `unknown`, so a stream whose only symbol is `unknown`
is a control channel on the evidence of its own files.

## What a feed is here, and why it is not a venue

A feed is one `(venue, stream)` pair - binance's trades, bybit's liquidations,
hyperliquid's asset contexts. Not a venue, because a venue is several feeds
with independent failure modes, and a venue-level number would let a healthy
trade tape carry a dead depth stream. Not a `(venue, stream, symbol)` triple
either: at 2,001 stream-symbols on one venue that is a number nobody reads,
and the recorder already judges silence per symbol underneath.

## The four measured components

Each is a fraction of the feed's own opportunity to fail, so none needs a
scale factor and all four are comparable:

- **delivery** - the share of this feed's subscribed symbols that were not
  recorded silent today.
- **integrity** - the share of the feed's own ledger events that were not
  corrupting. Corruption is rarer than silence and worse, so it is measured
  apart from it.
- **continuity** - the feed's own observation-loss gaps per event, subtracted
  from one. A feed whose every event is a gap report is not delivering a
  series. Per stream since 2026-08-09: charging every feed the venue's total
  gave all five binance feeds an identical 0.049, which is a number about the
  busiest stream wearing five labels.
- **freshness** - whether the feed is still writing while its siblings are.
  Measured against the newest write anywhere on that venue rather than against
  a wall clock, within the archive's own hour - the rotation unit the files
  are already cut on, rather than a duration chosen here. A feed that stopped
  an hour ago holds exactly the same bytes as one still running, so nothing in
  a byte count can tell them apart.

The score is their **minimum, not their mean.** A mean lets three healthy
components hide a dead one, which is the arithmetic version of the failure
this module exists to name. The components ride alongside the score for the
same reason: a single number that cannot be decomposed is a number nobody can
act on.

## What changes when it is wrong

`worst_component` names which of the four dragged the score down, so the
answer to a bad feed is a next step rather than a shrug. The wall reads this
today - it is the consumer, and unlike the sizing-dependent features it is not
waiting on anything to exist.

Nothing here invents a passing mark. There is no threshold in this module: it
reports scores and lets the board rank them, because what counts as too low
differs between a liquidation feed that is legitimately quiet for hours and a
depth stream that must never stop.
"""
from __future__ import annotations

from dataclasses import dataclass

_COMPONENTS = ("delivery", "integrity", "continuity", "freshness")


@dataclass(frozen=True)
class FeedScore:
    """One feed's score, decomposed so it can be argued with."""
    venue: str
    stream: str
    score: float
    delivery: float
    integrity: float
    continuity: float
    freshness: float
    symbols: int
    silent_symbols: int

    @property
    def worst_component(self) -> str:
        """Which component holds the score down - the actionable half."""
        return min(_COMPONENTS, key=lambda name: getattr(self, name))


def score_feeds(reports: dict[str, dict]) -> list[FeedScore]:
    """Score every feed present in a set of `capture_health` venue reports.

    Takes the reports rather than a capture root: they are already measured
    once per board pass, and re-reading the archive here would let one tile
    describe a different instant from the rest of the wall.
    """
    scores: list[FeedScore] = []
    for venue, report in sorted(reports.items()):
        by_stream = _streams_of(report)
        if not by_stream:
            continue

        events_by_stream = report.get("events_by_stream", {}) or {}
        gaps_by_stream = report.get("gaps_by_stream", {}) or {}
        # Only used where a stream has no events of its own recorded - a feed
        # that has never appeared in the ledger has nothing to divide by.
        venue_events = max(1, int(report.get("events_total", 0)))

        silent_symbols = _silent_by_stream(report)
        wrote_this_hour = _streams_still_writing(report)

        for stream, symbols in sorted(by_stream.items()):
            if symbols == {"unknown"}:
                continue          # a control channel, not a feed

            events = int(events_by_stream.get(stream, 0)) or venue_events
            stream_gaps = gaps_by_stream.get(stream, {})
            observation_loss = int(stream_gaps.get("observation_loss", 0))
            corrupting = int(stream_gaps.get("corrupting", 0))
            integrity = _fraction_left(corrupting, events)
            continuity = _fraction_left(observation_loss, events)

            silent = len(silent_symbols.get(stream, ()))
            delivery = _fraction_left(silent, max(1, len(symbols)))
            freshness = 1.0 if stream in wrote_this_hour else 0.0
            parts = {
                "delivery": delivery, "integrity": integrity,
                "continuity": continuity, "freshness": freshness,
            }
            scores.append(FeedScore(
                venue=venue, stream=stream,
                # The minimum, never the mean: a mean lets three healthy
                # components hide a dead one.
                score=min(parts.values()),
                symbols=len(symbols), silent_symbols=silent, **parts))
    return scores


def _fraction_left(bad: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, 1.0 - bad / total)


def _streams_of(report: dict) -> dict[str, set[str]]:
    """Stream -> symbols, from the per-stream byte map the report already has."""
    found: dict[str, set[str]] = {}
    for stream_symbol in report.get("raw_bytes_by_stream", {}):
        stream, _, symbol = stream_symbol.partition("_")
        if stream:
            found.setdefault(stream, set()).add(symbol)
    return found


def _silent_by_stream(report: dict) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for entry in report.get("silent_stream_symbols", ()):
        stream, _, symbol = entry.partition("_")
        if stream:
            found.setdefault(stream, set()).add(symbol)
    return found


_HOUR_NS = 3_600_000_000_000


def _streams_still_writing(report: dict) -> set[str]:
    """Streams whose newest write is within an hour of the venue's newest.

    Relative to the venue rather than to a wall clock, so a capture that is
    entirely stopped does not report every one of its feeds as individually
    stale - that is one fact about the venue, and the venue's own tiles say
    it. What this catches is the asymmetric case: one feed dead while its
    siblings on the same connection keep writing.

    An hour because that is the archive's rotation unit, not a duration
    chosen here. A report carrying no write times at all yields the empty
    set, so freshness reads 0 and says so rather than defaulting to healthy.
    """
    newest_by_stream: dict[str, int] = {}
    for stream_symbol, mtime_ns in report.get("newest_mtime_ns_by_stream", {}).items():
        stream, _, _symbol = stream_symbol.partition("_")
        if stream:
            newest_by_stream[stream] = max(newest_by_stream.get(stream, 0),
                                           int(mtime_ns))
    if not newest_by_stream:
        return set()
    venue_newest = max(newest_by_stream.values())
    return {stream for stream, newest in newest_by_stream.items()
            if venue_newest - newest <= _HOUR_NS}
