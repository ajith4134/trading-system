import random

from capture.sequencing import BinanceDepthTracker, StalenessTracker
from capture.capture_ledger import SEVERITY_CORRUPTING, SEVERITY_OBSERVATION_LOSS

S = 1_000_000_000  # one second in ns


def _bursty_gaps_seconds(count: int, seed: int = 7) -> list[float]:
    """A healthy but bursty stream: 60% of gaps near 6s, 40% near 0.5s.

    Deliberately deterministic. This is the shape a low quantile alone gets
    wrong - the quantile lands inside the burst and puts the alarm threshold
    under the stream's own ordinary cadence.
    """
    rng = random.Random(seed)
    return [(6.0 if rng.random() < 0.6 else 0.5) * (1 + rng.uniform(-0.05, 0.05))
            for _ in range(count)]


def _replay_gaps(tracker: StalenessTracker, gaps_seconds: list[float],
                 start_ns: int = 1785648600 * S) -> tuple[list, int]:
    now = start_ns
    tracker.check(now)
    reports = []
    for gap in gaps_seconds:
        now += int(gap * S)
        reports.append(tracker.check(now))
    return reports, now


def test_binance_first_frame_is_not_a_gap():
    t = BinanceDepthTracker()
    assert t.check({"U": 10, "u": 20, "pu": 9}) is None


def test_binance_contiguous_chain_has_no_gap():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    assert t.check({"U": 21, "u": 30, "pu": 20}) is None


def test_binance_broken_chain_is_corrupting():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})
    report = t.check({"U": 40, "u": 50, "pu": 39})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING
    assert report.detail["expected_pu"] == 20
    assert report.detail["got_pu"] == 39


def test_binance_spot_without_pu_uses_u_chain():
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20})
    assert t.check({"U": 21, "u": 30}) is None
    report = t.check({"U": 99, "u": 120})
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING


def test_hyperliquid_learns_cadence_then_flags_stall():
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        assert t.check(base + i * S) is None          # steady 1s cadence
    report = t.check(base + 20 * S + 60 * S)          # 60s later
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 60


def test_hyperliquid_floor_prevents_false_alarm_on_fast_streams():
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(20):
        t.check(base + int(i * 0.01 * S))             # 10ms cadence
    # 1s gap: 100x the median, but under the 5s floor -> not an alarm
    assert t.check(base + int(20 * 0.01 * S) + S) is None


def test_binance_malformed_message_does_not_disarm_detection():
    """A malformed message missing 'u' should not reset state."""
    t = BinanceDepthTracker()
    t.check({"U": 10, "u": 20, "pu": 9})  # Good message
    t.check({"U": 21})  # Malformed - missing u
    report = t.check({"U": 999, "u": 1010, "pu": 998})  # Gap should be detected
    assert report is not None
    assert report.severity == SEVERITY_CORRUPTING


def test_hyperliquid_long_gap_during_warmup_is_detected():
    """A long gap as the second interval should be detected, not absorbed."""
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    t.check(base)  # First timestamp
    # 500s gap as the very next interval (well above floor)
    report = t.check(base + 500 * S)
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 500


def test_hyperliquid_stream_slower_than_the_floor_learns_its_cadence():
    """An illiquid l2Book updating every 8s against a 5s floor must not alarm forever.

    Refusing to learn from any gap above the floor meant the window could never
    fill on such a stream: the threshold stayed pinned at the floor and every
    single frame became an observation_loss event. That floods the ledger and,
    worse, makes a genuine outage arrive with the same severity and shape as the
    hundreds of false ones around it.
    """
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    cadence_seconds = 8

    reports = [t.check(base + i * cadence_seconds * S) for i in range(200)]
    alarms = [r for r in reports if r is not None]

    assert len(alarms) <= 10, f"{len(alarms)} false alarms on a steady 8s stream"
    assert reports[-1] is None, "still alarming on ordinary cadence after 200 frames"
    assert t.has_baseline(), "the tracker never learned a cadence"


def test_hyperliquid_slow_stream_still_flags_a_real_outage():
    """Learning a slow cadence must not cost the detection it exists for."""
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S
    for i in range(200):
        t.check(base + i * 8 * S)

    last = base + 199 * 8 * S
    report = t.check(last + 30 * 60 * S)          # a 30-minute outage
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 1800
    # The threshold is derived from the learned 8s cadence, not the 5s floor.
    assert report.detail["threshold_seconds"] > 5.0


def test_hyperliquid_stalls_do_not_poison_median():
    """Stalls should not be included in the cadence learning window."""
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    base = 1785648600 * S

    # Start with first timestamp
    t.check(base)

    # Six genuine 50s stalls during initial learning
    for i in range(1, 7):
        report = t.check(base + i * 60 * S)  # 60s apart
        assert report is not None, f"Stall {i} should be flagged"

    # Six normal 1s intervals to establish cadence
    for i in range(6):
        t.check(base + 7 * 60 * S + i * S)

    # Another stall should still be detected despite the prior burst
    report = t.check(base + 7 * 60 * S + 6 * S + 40 * S)  # 40s gap
    assert report is not None, "Stall after burst should be detected"
    assert report.severity == SEVERITY_OBSERVATION_LOSS


# --------------------------------------------------------------------------
# A bursty-but-healthy stream must not sit in permanent alarm
# --------------------------------------------------------------------------

def test_bursty_healthy_stream_does_not_alarm_forever():
    """The regression a low cadence quantile alone reintroduced.

    On a stream that is 40% fast bursts and 60% ~6s cadence, a low quantile lands
    inside the burst (p25 = 0.48s), pins the threshold at the 5s floor, and flags
    every routine 6s gap. Because flagged gaps were then excluded from learning,
    the window refilled with burst gaps only and the baseline could never
    correct itself: 57% of frames flagged as observation_loss, permanently.

    That is the same failure the slow-stream fix exists to prevent - a genuine
    outage arriving with the same severity and shape as hundreds of false ones.
    """
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    reports, _ = _replay_gaps(t, _bursty_gaps_seconds(600))

    per_hundred = [sum(r is not None for r in reports[i:i + 100])
                   for i in range(0, 600, 100)]
    # Warmup may alarm; steady state must not. Frames 100 onward are steady state.
    assert max(per_hundred[1:]) <= 10, (
        f"alarms per 100 frames after warmup: {per_hundred} - a healthy 6s stream "
        f"is being reported as a permanent outage")
    assert sum(per_hundred) <= 60, f"{sum(per_hundred)}/600 frames flagged"


def test_bursty_stream_still_flags_a_real_outage():
    """Tolerating the burst must not cost the detection the tracker exists for."""
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    _, now = _replay_gaps(t, _bursty_gaps_seconds(600))

    report = t.check(now + 30 * 60 * S)
    assert report is not None
    assert report.severity == SEVERITY_OBSERVATION_LOSS
    assert report.detail["gap_seconds"] >= 1800
    # The threshold tracks the stream's own 6s cadence, not the 0.5s burst.
    assert report.detail["threshold_seconds"] > 5.0


def test_a_stall_far_beyond_the_threshold_stays_out_of_the_window():
    """Flagged-but-close gaps teach the baseline; genuine stalls must not.

    Without this, a real 10-minute outage would widen the threshold and teach the
    tracker to ignore the next one.
    """
    t = StalenessTracker(floor_seconds=5.0, multiple=10.0)
    _, now = _replay_gaps(t, [1.0] * 199)
    widest_before = max(t._gaps)

    t.check(now + 600 * S)                       # a 10-minute outage
    assert max(t._gaps) == widest_before, "a stall was learned as ordinary cadence"

    # And the very next outage is still detected at the same threshold.
    report = t.check(now + 600 * S + 600 * S)
    assert report is not None
    assert report.detail["threshold_seconds"] <= 10.0


def test_a_stream_slower_than_the_stall_rule_never_learns_a_baseline():
    """The documented limit of this tracker, and the reason `VenueRecorder` does
    not ask it whether a stream has died.

    A liquidation feed at one frame every few minutes is indistinguishable from
    a stalling 1s stream on the evidence available here: gaps beyond
    `stall_multiple` x the threshold are kept out of the window, and before a
    baseline the threshold is only the floor - so nothing is ever learned and
    every frame is flagged. Learning them instead poisons the baseline with
    warmup stalls (test_hyperliquid_stalls_do_not_poison_median). The recorder
    therefore judges silence from the venue clock and the widest gap a stream
    has actually shown, and records a gap event from this tracker only once it
    has a baseline to speak from.
    """
    t = StalenessTracker(floor_seconds=5.0, min_samples=10, stall_multiple=3.0)
    reports, _ = _replay_gaps(t, [180.0] * 40)

    assert not t.has_baseline()
    assert all(r is not None for r in reports)
