"""Binance funding, in its own process, away from the websocket it was killing.

Split out 2026-08-09, and the reason is a measurement rather than a preference.

After the funding poll went all-market, the binance recorder died at every hour
boundary with `ConnectionClosedError: no close frame received or sent`, roughly
forty seconds past the hour, every hour:

    09:00:37  ran 3414s
    10:00:40  ran 3602s

**The comparison that identified it:** `binance-spot` carries ~1,369 trade
writers and crossed four hour boundaries alive in the same period. `binance`
carried the same trade load PLUS 857 funding writers - and those 857 all rotate
inside ONE poll, because the fan-out writes every instrument in a single
synchronous tick. Trade writers rotate spread across thousands of frames as each
symbol happens to trade. **The burst is what mattered, not the total.**

Measured on this filesystem, idle: 857 closes-and-reopens is 5.64 s in one tick.
That is under websockets' 20 s keepalive budget, so the threshold was never
proven - under real load, with three recorders and the store builder competing
for fsync, it evidently is not. What IS established is the coupling: heavy
synchronous IO sharing an event loop with a keepalive-sensitive socket. Removing
the coupling does not depend on knowing the exact threshold, which is why this is
the fix rather than a longer ping timeout.

`bybit` has run this shape since it was added - poll-only, no websocket, 805
instruments - without an incident.

## It is still the binance venue

`name` is `"binance"`, deliberately. Files land in `raw/binance/` beside the
trades, so the archive layout is unchanged and `store.build_polled` finds
`premiumIndex` where it always did. The rate budget is keyed on the same name, so
this process and the recorder spend from one bucket - which is the point, since
the venue's limits are per-IP and exchange-wide.

Two processes write into the same venue-day directory and never the same file:
the stream name is part of every filename, and the `.writing` marker is per-file.
"""
from __future__ import annotations

from capture.venues import PollSpec, StreamSpec
from capture.venues.binance import BinanceVenue


class BinanceFundingPushesNothing(RuntimeError):
    """Something asked this venue for a websocket. The recorder owns that."""


class BinanceFundingVenue(BinanceVenue):
    """Binance, polled for funding only. Inherits the parsing, drops the socket."""

    def core_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def tail_specs(self, symbols: list[str]) -> list[StreamSpec]:
        return []

    def poll_specs(self, symbols: list[str]) -> list[PollSpec]:
        """The funding fan-out, and nothing else.

        The depth snapshot stays with the recorder: it is three symbols on a
        60-second cadence, it is not a fan-out, and it belongs beside the depth
        diffs it exists to let anyone replay.
        """
        return self.funding_poll_specs()

    def ws_url(self, specs: list[StreamSpec]) -> str:
        raise BinanceFundingPushesNothing(
            "funding is polled in its own process; the recorder owns the socket")

    def subscribe_messages(self, specs: list[StreamSpec]) -> list[dict]:
        raise BinanceFundingPushesNothing(
            "funding is polled in its own process; the recorder owns the socket")
