"""Gap detection. Deliberately different per venue - see spec 5.4.

Binance depth is a stateful diff stream: a break in the chain corrupts the book
until a REST resync. Hyperliquid l2Book is stateless snapshots: a gap loses an
observation but nothing is corrupted, so only staleness can be detected.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS


def quantile_ns(ordered: list[int], quantile: float) -> int:
    """The value at `quantile` of an already-sorted list of gaps.

    Shared with `VenueRecorder`, which needs the same "what does this stream
    routinely do?" question answered about silence. One definition, so the two
    answers cannot drift apart.
    """
    return ordered[min(int(len(ordered) * quantile), len(ordered) - 1)]


@dataclass(frozen=True)
class GapReport:
    severity: str
    detail: dict


class BinanceDepthTracker:
    """Validates the U/u/pu chain. Uses pu when present (futures), else u (spot)."""

    def __init__(self) -> None:
        self._last_u: int | None = None

    def check(self, parsed: dict) -> GapReport | None:
        first_id, final_id = parsed.get("U"), parsed.get("u")
        prev_final = parsed.get("pu")

        # If u is missing or None, this message is malformed.
        # Don't touch state, just skip it.
        if final_id is None:
            return None

        last_u, self._last_u = self._last_u, final_id
        if last_u is None:
            return None

        if prev_final is not None:
            if prev_final == last_u:
                return None
            return GapReport(SEVERITY_CORRUPTING,
                             {"expected_pu": last_u, "got_pu": prev_final})

        if first_id == last_u + 1:
            return None
        return GapReport(SEVERITY_CORRUPTING,
                         {"expected_U": last_u + 1, "got_U": first_id})


class StalenessTracker:
    """Learns a stream's cadence and infers a stall from it. Venue-agnostic.

    Written for Hyperliquid l2Book, which carries no sequence numbers, and named
    for it until 2026-08-02. It works from `t_recv_ns` alone, so it now owns
    cadence for every stream that has no sequence chain to check - Binance
    `trade`, `markPrice` and `forceOrder` included, which had no staleness owner
    at all and so could die mid-session unnoticed.

    The threshold is the widest of three terms, and each answers a different
    question. Getting any one of them wrong reintroduces the alarm storm this
    class exists to prevent, so all three are load-bearing:

    `floor_seconds` - "how fast is too fast to bother alarming on?" Without it a
    10ms stream alarms on ordinary jitter.

    `multiple` x a low quantile of the window - "how much slower than typical is
    suspicious?" The quantile is low, not the median, because stalls only ever
    push gaps upward: the fast end of the distribution is where the true cadence
    lives, and a low quantile survives a window that warmup filled with stalls
    where a median would be dragged up by them.

    A high quantile of the window - "what does this stream do routinely?" This is
    the term a low quantile alone cannot supply, and its absence is what made the
    previous version alarm on 57% of frames forever on a healthy *bursty* stream.
    On a stream that is 40% fast bursts and 60% ~6s cadence, the low quantile
    lands inside the burst and puts the threshold under the stream's own ordinary
    cadence. Every routine 6s gap then alarms - permanently, not just during
    warmup - which is exactly the "a genuine outage is indistinguishable from the
    noise" condition the alarming is supposed to prevent. Anchoring the threshold
    at (by default) the 99th percentile of the window caps the false-alarm rate
    at roughly 1% of frames by construction, whatever shape the distribution has.

    Learning rule: a gap is learned from unless it exceeds `stall_multiple` x the
    threshold. Being flagged is deliberately NOT enough to be excluded. Excluding
    every flagged gap is what let a wrong baseline become self-reinforcing - the
    window refilled only with the gaps that already agreed with it, so no amount
    of evidence could correct it. A gap that is merely over the line is treated
    as evidence the baseline may be wrong; a gap far beyond it is treated as a
    stall and kept out, which is what keeps a real outage from teaching the
    tracker to ignore outages.

    This also removes the warmup special case entirely. A stream slower than the
    floor (an illiquid l2Book updating every 8s against a 5s floor) learns its
    cadence because an 8s gap is nowhere near 3x the 5s floor, so it is learned
    despite being flagged.

    That headroom is bounded, and the bound is the limit of what this class can
    honestly claim: a stream whose ordinary cadence exceeds `stall_multiple` x
    `floor_seconds` (15s by default) never learns a baseline at all, and flags
    every frame forever. See the note on the learning rule in `check`, and
    `test_a_stream_slower_than_the_stall_rule_never_learns_a_baseline`. Whether
    a stream has *died* is therefore not a question for this class - a dead
    stream sends no frame for `check` to run on - and `VenueRecorder` answers it
    from the venue clock instead.
    """

    def __init__(self, floor_seconds: float = 5.0, multiple: float = 10.0,
                 window: int = 200, min_samples: int = 10,
                 cadence_quantile: float = 0.25,
                 routine_ceiling_quantile: float = 0.99,
                 stall_multiple: float = 3.0) -> None:
        self._floor_ns = int(floor_seconds * 1e9)
        self._multiple = multiple
        self._min_samples = min_samples
        self._cadence_quantile = cadence_quantile
        self._routine_ceiling_quantile = routine_ceiling_quantile
        self._stall_multiple = stall_multiple
        self._gaps: deque[int] = deque(maxlen=window)
        self._last_ns: int | None = None

    def _estimate_cadence_ns(self) -> int:
        return quantile_ns(sorted(self._gaps), self._cadence_quantile)

    def _estimate_routine_ceiling_ns(self) -> int:
        """The gap this stream does not routinely exceed, from the window itself."""
        return quantile_ns(sorted(self._gaps), self._routine_ceiling_quantile)

    def has_baseline(self) -> bool:
        return len(self._gaps) >= self._min_samples

    def _threshold_ns(self) -> int:
        # Before a baseline exists there is nothing to compare against, so the
        # floor is the only usable threshold.
        if not self.has_baseline():
            return self._floor_ns
        ordered = sorted(self._gaps)
        return max(
            self._floor_ns,
            int(quantile_ns(ordered, self._cadence_quantile) * self._multiple),
            quantile_ns(ordered, self._routine_ceiling_quantile),
        )

    def check(self, t_recv_ns: int) -> GapReport | None:
        last, self._last_ns = self._last_ns, t_recv_ns
        if last is None:
            return None
        gap = t_recv_ns - last
        threshold = self._threshold_ns()

        report = None
        if gap > threshold:
            report = GapReport(SEVERITY_OBSERVATION_LOSS, {
                "gap_seconds": gap / 1e9,
                "threshold_seconds": threshold / 1e9,
            })

        # A flagged gap is still learned from: it may be evidence the baseline is
        # wrong rather than evidence the stream is sick. Only a gap far beyond
        # the threshold is treated as a stall and kept out of the window.
        #
        # Note the consequence, which is deliberate: a stream whose true cadence
        # is slower than `stall_multiple` x floor (180s against a 5s floor - a
        # liquidation feed) can never learn a baseline, because its ordinary
        # cadence is indistinguishable from a stall on the evidence available.
        # Learning those gaps instead was tried on 2026-08-02 and is worse: six
        # genuine 60s stalls during warmup then set the routine ceiling and hid
        # every later stall (see test_hyperliquid_stalls_do_not_poison_median).
        # `VenueRecorder` handles the slow-stream case from the venue clock
        # instead, and does not ask this tracker a question it cannot answer.
        if gap <= self._stall_multiple * threshold:
            self._gaps.append(gap)

        return report
