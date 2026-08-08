"""Auto-halt per venue — the Phase 2 gap `DECISIONS.md` §13 carried forward.

`ARCHITECTURE.md` Layer 3: *"Tracks API error rate, price deviation vs
reference, staleness. **Auto-halts per venue.** Never retry into a degraded
matching engine."* The evidence for why it is Phase 2 rather than later is in
the same section: every major volatility spike since 2020 has a matching
degradation on a top-5 venue, so downtime is a base rate rather than a tail.

Three properties the tests below exist to pin down:

  * **Fail closed.** A venue nothing is known about is not tradeable. Refusing
    to trade is recoverable; trading into a degraded engine is not.
  * **Per venue.** Binance degrading must not halt Hyperliquid. The two were
    chosen because they fail differently, and a global halt throws that away.
  * **Sticky.** A halt does not lift the instant the signal recovers.
    `ARCHITECTURE.md` is explicit that Binance degraded for ~1 hour during the
    Oct 2025 cascade while valuing collateral on internal prices - the moment
    the error rate dips is not the moment the matching engine is sound.
"""
import pytest

from ops.venue_halt import (
    HALT_CORRUPTING,
    HALT_UNKNOWN,
    VenueHaltRegistry,
    assess_venue,
)

HEALTHY = {"silent_streams": 0, "corrupting_non_gap": 0}


def test_a_healthy_venue_is_tradeable(tmp_path):
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", HEALTHY)
    assert reg.is_tradeable("binance")


def test_a_venue_nothing_is_known_about_is_not_tradeable(tmp_path):
    """Fail closed. An empty registry means no health has been measured, which
    is not the same as health."""
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    assert not reg.is_tradeable("binance")
    assert reg.halt_reason("binance") == HALT_UNKNOWN


def test_a_corrupting_event_halts_the_venue(tmp_path):
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 1})
    assert not reg.is_tradeable("binance")
    assert reg.halt_reason("binance") == HALT_CORRUPTING


def test_silence_alone_does_not_halt_a_venue(tmp_path):
    """Measured 2026-08-08: at 2,123 symbols binance reported 1,845 silence
    events while completely healthy. The broad tail is deliberately full of thin
    symbols, and `forceOrder` is withheld by the venue and subscribed anyway, so
    it is silent by design. A halt on this count fires forever on a sound venue -
    which is worse than no halt, because it trains everyone to ignore it."""
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", {"silent_streams": 1845, "corrupting_non_gap": 0})
    assert reg.is_tradeable("binance")


def test_halting_one_venue_leaves_the_other_tradeable(tmp_path):
    """Binance and Hyperliquid were chosen because they fail differently. A
    global halt throws away the only benefit of running both."""
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 3})
    reg.observe("hyperliquid", HEALTHY)
    assert not reg.is_tradeable("binance")
    assert reg.is_tradeable("hyperliquid")


def test_a_halt_does_not_lift_the_moment_the_signal_recovers(tmp_path):
    """The Oct 2025 cascade: Binance degraded ~1 hour while valuing collateral
    on internal prices. The instant an error rate dips is not the instant the
    matching engine is sound."""
    now = {"t": 0}
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: now["t"])
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 1})
    assert not reg.is_tradeable("binance")

    now["t"] = 60 * 10**9              # one minute of quiet
    reg.observe("binance", HEALTHY)
    assert not reg.is_tradeable("binance"), "lifted the halt on one clean sample"


def test_a_halt_lifts_only_after_a_sustained_healthy_dwell(tmp_path):
    now = {"t": 0}
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: now["t"])
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 1})

    for minutes in (5, 10, 20, 40):
        now["t"] = minutes * 60 * 10**9
        reg.observe("binance", HEALTHY)
    assert reg.is_tradeable("binance"), "never recovered despite sustained health"


def test_every_transition_is_recorded_with_its_reason(tmp_path):
    """A halt nobody can explain afterwards is indistinguishable from a bug,
    and this one blocks trading."""
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 2})

    events = reg.history()
    assert events
    assert events[-1]["venue"] == "binance"
    assert events[-1]["reason"] == HALT_CORRUPTING
    assert "corrupting" in str(events[-1]["detail"]).lower()


def test_the_halt_survives_a_restart(tmp_path):
    """A halt that a process restart clears is a halt an unhealthy venue can
    escape by killing the watcher."""
    reg = VenueHaltRegistry(tmp_path, clock_ns=lambda: 1_000)
    reg.observe("binance", {"silent_streams": 0, "corrupting_non_gap": 1})

    reloaded = VenueHaltRegistry(tmp_path, clock_ns=lambda: 2_000)
    assert not reloaded.is_tradeable("binance")
    assert reloaded.halt_reason("binance") == HALT_CORRUPTING


def test_assess_is_pure_and_needs_no_registry():
    """The decision is separable from the bookkeeping, so it can be tested and
    reasoned about on its own."""
    assert assess_venue(HEALTHY) is None
    assert assess_venue({"silent_streams": 0, "corrupting_non_gap": 1})[0] == HALT_CORRUPTING
    assert assess_venue({"silent_streams": 1845, "corrupting_non_gap": 0}) is None
