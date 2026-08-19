"""BF-06: one segment bot, end to end, on live prices.

    python -m segment.live_engine --segment perp

## The loop

    live feed  →  feature frame  →  BULL, BEAR, PROFIT-TAIL(assess)
                                          ↓
                                      ARBITER  ── selects, or abstains
                                          ↓
                              PROFIT-TAIL.time_entry  ── when, never whether
                                          ↓
                                      RISK GATE  ── the only thing that refuses
                                          ↓
                                     PAPER BROKER  →  journal
                                          ↓
                      PROFIT-TAIL.manage  ── owns the position until it is closed

Every arrow is one direction. PROFIT-TAIL appears twice and holds no veto at either
point (RL-023); the risk gate is the single place an order is refused, because every
blow-up in the research set traces to risk authority being split.

## What makes this paper trading rather than a backtest

**RL-024.** Prices come from `live.live_feed` — a venue websocket or REST poll — and
never from the parquet store. The engine refuses to trade a symbol whose quote is
stale and refuses to trade at all when its feed reports QUIET, so a dead connection
stops the bot instead of freezing it at the last price it happened to hold.

Fills are modelled by `paper.fill_model` against the live two-sided quote. No order
reaches a venue: this is a paper broker, and the `edge_claim` flag on every journalled
fill records whether the brain that produced it claims an edge. Today every brain is a
rule brain and every flag is False (RL-025).

## Why positions are closed by the same loop that opens them

A separate exit process would be a second reader of the same feed with its own view of
what is open, and the two would disagree during any restart. `manage()` runs on every
poll for every open position before any new entry is considered, so the bot cannot open
a new position while failing to notice one it should have closed.

## Warm-up is a refusal, not a delay

`segment.live_features` holds its rolling state in process, so a restarted bot has no
features until it has watched live ticks for its window. During that period it declines
every symbol with `samples(n<12)` and journals a heartbeat saying so. That is deliberate:
the alternative is priming from the store, which is the RL-024 violation this build
exists to remove.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from live import universe_discovery
from live.live_feed import LIVE, NEVER_DELIVERED, QUIET
from segment.arbiter import ABSTAIN, select
from segment.capital_accounting import DOLLAR_QUOTES, quote_currency
from segment.bot_registry import all_segments, segment_bot
from segment.brain import BrainOutputs
from segment.live_features import LiveFeatureFrames
from segment.profit_tail import (
    CLOSE, ENTER_NOW, HOLD, MISSED_ENTRY, RATCHET_LOCK, OpenPosition,
)

DEFAULT_ROOT = Path.home() / "capture" / "segment"
# How often the loop runs. Fast enough that a fast-band scalp is managed on a
# meaningful cadence, slow enough that the REST-polled segments are not asked for
# more than they can give.
DEFAULT_INTERVAL_SECONDS = 5.0
# The stop distance the RISK GATE sets at fill is declared PER SEGMENT in
# `segment.bot_registry`; PROFIT-TAIL never moves it (spec §4).
#
# Whatever a segment declares, the stop is floored at this multiple of the spread the
# entry just crossed. A stop inside the round-trip cost is not a risk limit - it is a
# guarantee that every position closes at a loss the instant it opens, which is
# exactly what the options bot did 155 times on 2026-08-18 before this floor existed.
MIN_STOP_SPREAD_MULTIPLE = Decimal("3")


@dataclass
class EngineCounts:
    polls: int = 0
    ticks: int = 0
    frames_refused: int = 0
    proposals: int = 0
    declines: int = 0
    abstentions: int = 0
    selected: int = 0
    missed_entries: int = 0
    opened: int = 0
    closed: int = 0
    gate_refusals: int = 0
    ratchets: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LiveSegmentEngine:
    """One segment's bot. Owns its feed, its brains, its positions and its journal."""

    bot: object
    feed: object
    features: object
    root: Path
    counts: EngineCounts = field(default_factory=EngineCounts)
    positions: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    # (venue, symbol) -> (raw model score, side) for open positions, so a close can
    # be attributed to the score that opened it. Without the pairing the
    # calibration would learn from outcomes it cannot attribute to a prediction.
    opened_scores: dict = field(default_factory=dict)
    realised_pnl: Decimal = Decimal(0)
    # **What the last poll admitted, and why it excluded the rest (BF-09, BF-10).**
    # The engine is the only place that knows both the universe it was handed and
    # what survived admission this poll, so it is the only place that can publish
    # the pair. Held rather than recomputed: `describe_universe` counts the
    # decisions of the poll that just ran, not a fresh listing call.
    last_admission: dict = field(default_factory=dict)
    # **(venue, symbol) -> the PROFIT-TAIL that opened this position (LB-09,
    # RL-030).** A champion that arrives mid-position does not take it over: the
    # position runs out on the brains that opened it, so its P&L stays
    # attributable to the model that earned it. A trade opened by one model and
    # closed by another is attributable to neither, which is the same failure as
    # crediting one P&L to all 36 features that could have produced it.
    position_tails: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ journal

    def _journal_path(self, name: str, now_ns: int) -> Path:
        day = time.strftime("%Y-%m-%d", time.gmtime(now_ns / 1e9))
        directory = self.root / self.bot.segment
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}-{day}.ndjson"

    def _record(self, name: str, payload: dict, now_ns: int) -> None:
        path = self._journal_path(name, now_ns)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def recover_open_positions(self, now_ns: int) -> int:
        """Rebuild open positions from the journal after a restart.

        **Without this a restart orphans every open position.** The book lives in
        memory, so a supervised restart forgets what is open while the journal keeps
        the OPEN row that has no CLOSE. Measured 2026-08-18: the options bot showed
        `opened=1` against `open_positions=0` with no close journalled - a position
        that was neither held nor closed, and a P&L that silently excluded it.

        That is worse than a crash. The record reads complete and is not, and every
        winrate computed from it is computed over the trades that happened to survive
        a restart.

        Positions are matched per (venue, symbol): an OPEN with no later CLOSE is
        still open. The hard stop is read back from the journal rather than
        recomputed, because it was set by the risk gate at fill and PROFIT-TAIL is
        never allowed to move it - recomputing it from today's price would do exactly
        that, and in the flattering direction.
        """
        directory = self.root / self.bot.segment
        recovered = {}
        for path in sorted(directory.glob("fills-*.ndjson")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = (row.get("venue"), row.get("symbol"))
                if row.get("event") == "OPEN":
                    try:
                        recovered[key] = OpenPosition(
                            venue=key[0], symbol=key[1], side=row["side"],
                            quantity=Decimal(str(row["quantity"])),
                            entry_price=Decimal(str(row["price"])),
                            entry_ns=int(row["at_ns"]),
                            hard_stop=Decimal(str(row["hard_stop"])),
                            band=row.get("band", self.bot.band))
                    except (KeyError, ValueError, TypeError):
                        continue
                elif row.get("event") == "CLOSE":
                    recovered.pop(key, None)
        self.positions.update(recovered)
        if recovered:
            self._record("decisions", {
                "at_ns": now_ns, "segment": self.bot.segment,
                "outcome": "POSITIONS_RECOVERED",
                "reason": "restart",
                "evidence": {"count": len(recovered),
                             "symbols": [k[1] for k in recovered]}}, now_ns)
        return len(recovered)


    def _usd_conversion(self, frame, symbol: str, venue: str) -> dict:
        """The rate this instrument's price converts to USDT at, journalled (RL-029).

        Deribit quotes an option in BTC or ETH, so a premium of 0.1135 is not
        eleven cents - it is 0.1135 of the underlying. Summing that into a USDT
        total unconverted is a unit error that reads as a result.

        The rate is taken from the frame the fill was decided on, so it is the
        price the bot actually saw rather than one looked up afterwards, and it
        is written onto the fill so any converted figure can be audited back to
        it. When no rate is available the fill says so and the accounting reports
        it as unconvertible - it is never converted at a rate nobody recorded.
        """
        currency = quote_currency(venue, symbol)
        if currency in DOLLAR_QUOTES:
            return {"quote_currency": currency, "usd_rate": "1",
                    "rate_source": "quoted in dollars"}
        underlying = (frame or {}).get("venue_underlying_price")
        if underlying is None:
            underlying = (frame or {}).get("underlying_price")
        if underlying is None:
            return {"quote_currency": currency, "usd_rate": None,
                    "rate_source": "no rate available at the fill"}
        return {"quote_currency": currency, "usd_rate": str(underlying),
                "rate_source": "venue underlying price on the deciding frame"}

    def _adopt_new_champion(self, now_ns: int) -> None:
        """Decide from the champion registered NOW, without a restart (LB-09).

        The check is a small file read on every poll and the model is loaded only
        when the version id actually changed (RL-030). Before this, a bot resolved
        its champion once at process start while the retrainer refit every four
        hours and the bots ran 24/7 - so every model fitted between two restarts
        was registered and never used, and the heartbeat published the version the
        bot had LOADED rather than the one that existed.

        Open positions are untouched: `position_tails` holds the brain that opened
        each one and `_manage_open_positions` uses it until the position closes.
        """
        from learn.learned_brains import registered_champion_id

        registered = registered_champion_id(self.bot.segment)
        if not registered or registered == self.bot.model_version:
            return
        from segment.bot_registry import segment_bot
        try:
            swapped = segment_bot(self.bot.segment)
        except Exception as failure:                     # noqa: BLE001
            # A champion that cannot be loaded leaves the bot exactly as it was.
            # Trading on the previous model is correct; stopping is not.
            self._record("decisions", {
                "at_ns": now_ns, "segment": self.bot.segment,
                "outcome": "CHAMPION_SWAP_REFUSED", "reason": type(failure).__name__,
                "evidence": {"registered": registered,
                             "running": self.bot.model_version,
                             "detail": str(failure)[:200]}}, now_ns)
            return
        if not swapped.learned or swapped.model_version != registered:
            # The provenance check refused it. That refusal is the point of
            # `_with_learned_brains` and it is recorded rather than retried.
            self._record("decisions", {
                "at_ns": now_ns, "segment": self.bot.segment,
                "outcome": "CHAMPION_SWAP_REFUSED", "reason": "NOT_ADOPTED",
                "evidence": {"registered": registered,
                             "running": self.bot.model_version,
                             "champion_refused": swapped.extra.get("champion_refused")}},
                         now_ns)
            return

        previous = self.bot.model_version or "none"
        self.bot = swapped
        self._record("decisions", {
            "at_ns": now_ns, "segment": self.bot.segment,
            "outcome": "CHAMPION_SWAPPED", "reason": "NEW_CHAMPION_REGISTERED",
            "evidence": {"from_model": previous, "to_model": registered,
                         "open_positions_kept_on_their_own_brains":
                             len(self.position_tails),
                         "authority": "new entries only (RL-030)"}}, now_ns)

    def _warm_up(self) -> dict | None:
        """How far a LEARNED bot is from being able to decide at all.

        **Measured 2026-08-19, and it looked exactly like a broken model.** The
        perp bot swapped to its trained champion and then declined 152,334 times
        across 319 polls without one proposal. Nothing was wrong: a learned brain
        needs `FEATURE_WINDOW_BARS` SEALED one-minute bars before a feature vector
        exists, so every restart costs it that many minutes of not trading - and
        the board showed `0 proposals`, which is indistinguishable from a model
        whose threshold is never crossed.

        None for a rule bot, which has no such window and for which the question
        is meaningless rather than zero.
        """
        if not self.bot.learned:
            return None
        from learn.training_set import FEATURE_WINDOW_BARS
        held = [self.features.bars_held(venue, symbol)
                for venue, symbol in self.features.symbols()]
        ready = sum(1 for count in held if count >= FEATURE_WINDOW_BARS)
        return {"bars_required": FEATURE_WINDOW_BARS,
                "symbols_ready": ready,
                "symbols_watched": len(held),
                "deepest_bars": max(held, default=0),
                # One bar a minute, so what is left IS the wait in minutes.
                "minutes_to_first_decision": max(0, FEATURE_WINDOW_BARS
                                                 - max(held, default=0))}

    def _heartbeat(self, now_ns: int, note: str) -> None:
        directory = self.root / self.bot.segment
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at_ns": now_ns,
            "segment": self.bot.segment,
            "venue": self.bot.venue,
            "band": self.bot.band,
            "brains": {"bull": self.bot.bull.name, "bear": self.bot.bear.name,
                       "profit_tail": self.bot.profit_tail.name},
            # RL-025 / RL-026: the label belongs to the decision, and the heartbeat
            # carries it so a board tile cannot render a brain as anything else.
            "makes_edge_claim": bool(getattr(self.bot.bull, "makes_edge_claim", False)),
            "rule_brain": not self.bot.learned,
            "learned": self.bot.learned,
            "model_version": self.bot.model_version,
            "tail_model_version": self.bot.tail_model_version,
            # **Why a bot is running rule brains, when a champion exists and was
            # refused.** Without this the board can only say `not learned`, which
            # reads as `nobody has trained one yet` - a different fact.
            "champion_refused": self.bot.extra.get("champion_refused"),
            # A learned bot that cannot yet see 60 sealed bars is WARMING UP, not
            # declining. The two produce the same zero on a board.
            "warm_up": self._warm_up(),
            # §1a L2: what the LIVE loop fitted, kept apart from what the retrainer
            # set, because the two are different claims.
            "live_fitted": (self.bot.extra["calibration"].realised_coverage()
                            if self.bot.extra.get("calibration") else None),
            "feed": self.feed.describe(now_ns),
            # Admission, with its denominator. Counts only: the admitted symbol
            # list is thousands of entries on the spot and options boards and a
            # heartbeat rewritten every poll is the wrong place for it.
            "universe_admission": self.last_admission,
            "counts": self.counts.as_dict(),
            "open_positions": len(self.positions),
            "realised_pnl": str(self.realised_pnl),
            "note": note,
        }
        (directory / "heartbeat.json").write_text(json.dumps(payload, indent=2, default=str))

    # --------------------------------------------------------------------- poll

    def poll_once(self, now_ns: int) -> dict:
        self.counts.polls += 1
        self._adopt_new_champion(now_ns)
        ticks = self.feed.poll()
        self.counts.ticks += len(ticks)
        self.features.update(ticks)

        liveness = self.feed.liveness(now_ns)
        if liveness in (QUIET, NEVER_DELIVERED):
            # A dead feed stops the bot. It does NOT trade on the last price it
            # holds, and it does not fall back to the store.
            self._heartbeat(now_ns, f"feed {liveness}: not trading")
            return {"traded": False, "reason": liveness}

        frames = self.features.frames(now_ns)
        refused = [key for key, frame in frames.items() if hasattr(frame, "is_refusal")]
        self.counts.frames_refused = len(refused)

        # The LEARNED brains read a positional float vector computed by the SAME
        # function the model was fitted with (`learn.training_set.compute_features`),
        # from the same one-minute bar shape. Attached here rather than inside the
        # brains so both directional brains and PROFIT-TAIL see one vector per
        # symbol per poll instead of recomputing it three times.
        if self.bot.learned:
            from learn.learned_brains import features_from_live
            for key, frame in frames.items():
                if hasattr(frame, "is_refusal"):
                    continue
                frame["model_vector"] = features_from_live(
                    self.features, key[0], key[1])
                frame["sealed_bars"] = self.features.bars_held(key[0], key[1])

        decisions = self.bot.admit(frames)
        admitted = {(d.venue, d.symbol) for d in decisions if d.admitted}
        self.last_admission = {k: v for k, v
                               in self.bot.describe_universe(decisions).items()
                               if k != "admitted_symbols"}

        # Positions are managed BEFORE any new entry is considered, so the bot can
        # never open something new while failing to close something old.
        self._manage_open_positions(frames, now_ns)

        chain_median = None
        if self.bot.brains_need_chain_median:
            from options.segment_brains import chain_median_iv
            chain_median = chain_median_iv(list(frames.values()))

        for key in admitted:
            frame = frames.get(key)
            if frame is None or hasattr(frame, "is_refusal"):
                continue
            if key in self.positions:
                continue          # one position per instrument per bot
            self._consider(frame, now_ns, chain_median)

        self._heartbeat(now_ns, f"feed {LIVE}")
        return {"traded": True, "admitted": len(admitted), "refused": len(refused)}

    def _consider(self, frame, now_ns: int, chain_median) -> None:
        venue, symbol = frame["venue"], frame["symbol"]
        if chain_median is not None:
            bull = self.bot.bull(frame, chain_median=chain_median)
            bear = self.bot.bear(frame, chain_median=chain_median)
        else:
            bull = self.bot.bull(frame)
            bear = self.bot.bear(frame)

        outputs = BrainOutputs(bull=bull, bear=bear)
        for output in (bull, bear):
            if hasattr(output, "confidence"):
                self.counts.proposals += 1
            else:
                self.counts.declines += 1

        # PROFIT-TAIL's advisory numbers, consumed by the arbiter as inputs.
        tail = self.bot.profit_tail.assess(
            venue=venue, symbol=symbol, frame=frame, at_ns=now_ns)

        selection = select(outputs=outputs, tail=tail, at_ns=now_ns,
                           min_confidence=self.bot.min_confidence,
                           min_margin=self.bot.min_margin,
                           max_loss_tail=self.bot.max_loss_tail)

        if selection.side == ABSTAIN:
            self.counts.abstentions += 1
            # Journalled, because a symbol nobody wanted and a symbol the arbiter
            # refused are different events and only one is information.
            self._record("decisions", {
                "at_ns": now_ns, "segment": self.bot.segment, "venue": venue,
                "symbol": symbol, "outcome": "ABSTAIN", "reason": selection.reason,
                "evidence": selection.evidence}, now_ns)
            return

        self.counts.selected += 1

        # PROFIT-TAIL decides WHEN. It cannot decide whether.
        timing = self.bot.profit_tail.time_entry(
            venue=venue, symbol=symbol, side=selection.side,
            selected_at_ns=self.pending.get((venue, symbol), now_ns),
            now_ns=now_ns, frame=frame)

        if timing.outcome == MISSED_ENTRY:
            self.counts.missed_entries += 1
            self.pending.pop((venue, symbol), None)
            # Attributed to PROFIT-TAIL by name. A timing bot that misses the movers
            # shows excellent fill prices while failing, and this is the record that
            # makes that visible.
            self._record("missed_entries", {
                "at_ns": now_ns, "segment": self.bot.segment, "venue": venue,
                "symbol": symbol, "side": selection.side,
                "attributed_to": self.bot.profit_tail.name,
                "reason": timing.reason, "evidence": timing.evidence}, now_ns)
            return

        if timing.outcome != ENTER_NOW:
            self.pending.setdefault((venue, symbol), now_ns)
            return

        self._open(frame, selection, timing, now_ns)

    # ------------------------------------------------------------------ trading

    def _open(self, frame, selection, timing, now_ns: int) -> None:
        venue, symbol = frame["venue"], frame["symbol"]
        side = selection.side
        # Cross the spread: a taker entry is modelled against the side that would
        # actually have to be lifted, never against the mid. Modelling entries at the
        # mid is the flattering direction and would show an edge that is half spread.
        price = frame["ask"] if side == "LONG" else frame["bid"]
        quantity = self.bot.quantity

        gate = self._risk_gate(frame, side, price, quantity, now_ns)
        if not gate["passed"]:
            self.counts.gate_refusals += 1
            self._record("decisions", {
                "at_ns": now_ns, "segment": self.bot.segment, "venue": venue,
                "symbol": symbol, "outcome": "GATE_REFUSED",
                "reason": gate["reason"], "evidence": gate}, now_ns)
            return

        # The declared distance, or three times the spread just crossed - whichever
        # is further from the entry. The floor is what stops a trade dying of its own
        # transaction cost; the declared distance is what bounds a real adverse move.
        declared = self.bot.hard_stop_fraction
        spread = frame.get("relative_spread")
        floor = (Decimal(str(spread)) * MIN_STOP_SPREAD_MULTIPLE
                 if spread is not None else Decimal(0))
        stop_fraction = max(declared, floor)
        hard_stop = (price * (1 - stop_fraction) if side == "LONG"
                     else price * (1 + stop_fraction))

        position = OpenPosition(
            venue=venue, symbol=symbol, side=side, quantity=quantity,
            entry_price=price, entry_ns=now_ns, hard_stop=hard_stop,
            band=self.bot.band)
        self.positions[(venue, symbol)] = position
        # The brain that opened it owns it to the close (RL-030).
        self.position_tails[(venue, symbol)] = self.bot.profit_tail
        self.pending.pop((venue, symbol), None)
        self.counts.opened += 1
        raw = (selection.evidence.get("bull", {}).get("evidence", {})
               or {}).get("raw_score")
        if raw is None:
            raw = (selection.evidence.get("bear", {}).get("evidence", {})
                   or {}).get("raw_score")
        if raw is not None:
            self.opened_scores[(venue, symbol)] = (float(raw), side)

        self._record("fills", {
            "at_ns": now_ns, "segment": self.bot.segment, "venue": venue,
            "symbol": symbol, "event": "OPEN", "side": side,
            "quantity": str(quantity), "price": str(price),
            # BF-11: what this position actually costs, in the instrument's own
            # currency, with the rate that turns it into USDT.
            "notional": str(quantity * price),
            **self._usd_conversion(frame, symbol, venue),
            "hard_stop": str(hard_stop),
            "stop_fraction": str(stop_fraction),
            "stop_fraction_declared": str(declared),
            "stop_fraction_spread_floor": str(floor),
            "band": self.bot.band,
            "brains": {"bull": self.bot.bull.name, "bear": self.bot.bear.name,
                       "profit_tail": self.bot.profit_tail.name},
            "makes_edge_claim": selection.makes_edge_claim,
            "confidence": str(selection.confidence),
            "calibrated": selection.calibrated,
            "selection_reason": selection.reason,
            "entry_timing": timing.reason,
            "evidence": selection.evidence}, now_ns)

    def _manage_open_positions(self, frames, now_ns: int) -> None:
        for key in list(self.positions):
            position = self.positions[key]
            frame = frames.get(key)
            if frame is None or hasattr(frame, "is_refusal"):
                continue
            # Mark against the side the position would have to EXIT into, for the
            # same reason the entry crossed the spread.
            mark = frame["bid"] if position.side == "LONG" else frame["ask"]
            # **The brain that OPENED this position manages it to the close
            # (RL-030).** After a mid-run champion swap `self.bot.profit_tail` is
            # the new one, and handing it a position it never chose would split
            # that trade's P&L across two models - attributable to neither.
            # Falls back to the current tail for a position recovered from the
            # journal after a restart, where the object that opened it is gone.
            tail = self.position_tails.get(key, self.bot.profit_tail)
            directive = tail.manage(
                position=position, mark=mark, now_ns=now_ns, frame=frame)

            if directive.action == HOLD:
                continue

            if directive.action == RATCHET_LOCK:
                self.counts.ratchets += 1
                self.positions[key] = OpenPosition(
                    venue=position.venue, symbol=position.symbol, side=position.side,
                    quantity=position.quantity, entry_price=position.entry_price,
                    entry_ns=position.entry_ns, hard_stop=position.hard_stop,
                    band=position.band, locked_stop=directive.locked_stop,
                    peak_favourable=position.peak_favourable)
                self._record("decisions", {
                    "at_ns": now_ns, "segment": self.bot.segment,
                    "venue": position.venue, "symbol": position.symbol,
                    "outcome": "RATCHET_LOCK", "reason": directive.reason,
                    "evidence": directive.evidence}, now_ns)
                continue

            if directive.action == CLOSE:
                self._close(position, mark, directive, now_ns, frame, tail)

    def _close(self, position, mark, directive, now_ns: int, frame=None,
               tail=None) -> None:
        gross = ((mark - position.entry_price) * position.quantity
                 if position.side == "LONG"
                 else (position.entry_price - mark) * position.quantity)
        self.realised_pnl += gross
        # **LB-05: the one place the live loop changes a parameter itself.**
        #
        # A closed trade is a realised outcome, so it updates the reliability bin
        # the decision was scored in and the conformal non-conformity window. §1a
        # L2 turns on exactly this: without it, every parameter would come from a
        # script a human invokes, and the honest label would be "scheduled
        # retraining" rather than adaptive.
        self._learn_from_close(position, gross, now_ns)
        self.positions.pop((position.venue, position.symbol), None)
        self.position_tails.pop((position.venue, position.symbol), None)
        self.counts.closed += 1
        self._record("fills", {
            "at_ns": now_ns, "segment": self.bot.segment, "venue": position.venue,
            "symbol": position.symbol, "event": "CLOSE", "side": position.side,
            "quantity": str(position.quantity), "price": str(mark),
            "entry_price": str(position.entry_price),
            "gross_pnl": str(gross), "band": position.band,
            # The rate at the CLOSE, not the one recorded at the open: a P&L
            # realised today converts at today's price, and carrying both lets a
            # reader see when the two disagreed (RL-029).
            **self._usd_conversion(frame, position.symbol, position.venue),
            "held_ns": now_ns - position.entry_ns,
            "close_reason": directive.reason,
            "closed_by": (tail or self.bot.profit_tail).name,
            "reduce_only": directive.reduce_only,
            "makes_edge_claim": False,
            "evidence": directive.evidence}, now_ns)

    def _learn_from_close(self, position, gross, now_ns: int) -> None:
        """Fold a realised outcome back into the live-fitted calibration."""
        calibration = self.bot.extra.get("calibration")
        opened = self.opened_scores.pop((position.venue, position.symbol), None)
        if calibration is None or opened is None:
            return
        raw_score, side = opened
        # The label is "did the LONG resolve profitable", which is what the model
        # predicts. A short that made money is a long that would have lost, so the
        # outcome is inverted for a short rather than recorded as a win - recording
        # it as a win would teach the calibration the opposite of the truth.
        profitable = gross > 0
        outcome = 1 if (profitable if side == "LONG" else not profitable) else 0
        calibration.observe(score=raw_score, outcome=outcome, acted=True)
        try:
            calibration.save(Path(str(
                __import__("segment.bot_registry", fromlist=["CALIBRATION_PATH"])
                .CALIBRATION_PATH).format(segment=self.bot.segment)))
        except OSError:
            # A calibration that cannot be persisted must not stop trading; it is
            # rebuilt from subsequent outcomes.
            pass

    def _risk_gate(self, frame, side, price, quantity, now_ns: int) -> dict:
        """The only thing in this loop that may refuse an order.

        Deliberately separate from PROFIT-TAIL and from the brains. Kept small and
        explicit here rather than wired to `risk.pre_trade_gate`, whose limits are
        seeded per strategy from a file the segment bots do not yet have; PB-07 is
        the row that replaces this with the full gate, and it names the reason.
        """
        if price is None or price <= 0:
            return {"passed": False, "reason": "NON_POSITIVE_PRICE"}
        notional = price * quantity
        open_notional = sum(
            (p.entry_price * p.quantity for p in self.positions.values()),
            Decimal(0))
        limit = self.bot.max_open_positions
        if len(self.positions) >= limit:
            return {"passed": False, "reason": "MAX_OPEN_POSITIONS",
                    "open": len(self.positions), "limit": limit}
        spread = frame.get("relative_spread")
        if spread is not None and Decimal(str(spread)) > Decimal("0.02"):
            return {"passed": False, "reason": "SPREAD_UNTRADEABLE",
                    "relative_spread": str(spread)}
        return {"passed": True, "reason": "PASSED", "notional": str(notional),
                "open_notional": str(open_notional)}


def record_venue_listing(segment: str, root: Path = DEFAULT_ROOT) -> dict:
    """Write what the VENUE lists for this segment, once, at start (BF-09).

    Separate from the admission counts in the heartbeat, and they answer different
    questions: this is the size of the board the venue offers, the heartbeat is
    what survived one poll of it. Publishing only the second would let a bot that
    admitted 40 symbols out of 40 read as complete coverage of 570.

    A listing call that fails is recorded as a FAILED DISCOVERY rather than left
    absent, because an absent file and an empty venue are indistinguishable to a
    probe and only one of them is a market with nothing in it.
    """
    directory = root / segment
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"segment": segment, "discovered_at_ns": time.time_ns(),
               "source": "live.universe_discovery - a venue listing call"}
    try:
        payload["listing"] = universe_discovery.discover(segment).describe()
    except Exception as failure:                        # noqa: BLE001
        payload["discovery_failed"] = f"{type(failure).__name__}: {failure}"
    (directory / "universe.json").write_text(json.dumps(payload, indent=2,
                                                        default=str))
    return payload


def build_engine(segment: str, root: Path = DEFAULT_ROOT) -> LiveSegmentEngine:
    bot = segment_bot(segment)
    record_venue_listing(segment, root)
    feed = bot.build_feed().start()
    features = LiveFeatureFrames(segment=segment,
                                 window_ns=bot.feature_window_ns,
                                 min_samples=bot.min_samples)
    return LiveSegmentEngine(bot=bot, feed=feed, features=features, root=root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segment", required=True, choices=list(all_segments()))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval-seconds", type=float,
                        default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-polls", type=int, default=0,
                        help="stop after N polls; 0 runs until stopped")
    args = parser.parse_args(argv)

    engine = build_engine(args.segment, root=args.root)
    recovered = engine.recover_open_positions(time.time_ns())
    if recovered:
        print(f"recovered {recovered} open position(s) from the journal", flush=True)
    stopping = {"now": False}

    def _stop(signum, frame):        # noqa: ARG001
        stopping["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # **The banner reports what loaded, it does not assert what was expected.**
    # It read "edge_claim=False (RULE BRAIN, RL-025)" as a constant, and on
    # 2026-08-19 the perp bot started on a trained champion and printed that line
    # anyway. A start-up line nobody can trust is worse than none.
    claim = bool(getattr(engine.bot.bull, "makes_edge_claim", False))
    kind = (f"LEARNED, model {engine.bot.model_version}" if engine.bot.learned
            else "RULE BRAIN, RL-025")
    refused = engine.bot.extra.get("champion_refused")
    print(f"live segment bot: segment={args.segment} venue={engine.bot.venue} "
          f"band={engine.bot.band} brains={engine.bot.bull.name},"
          f"{engine.bot.bear.name},{engine.bot.profit_tail.name} "
          f"edge_claim={claim} ({kind})", flush=True)
    if refused:
        print(f"champion refused: {refused}", flush=True)

    # The feature frames need live ticks before anything can be computed. How long
    # that takes is the feed's business, not a constant: a websocket fills the window
    # in seconds, a 10-second REST poll needs `min_samples` polls before one frame
    # can exist at all. Waiting less does not fail loudly - it produces a bot that
    # refuses every symbol and looks like a market with no opportunities.
    warmup_seconds = min(180.0, max(10.0, engine.bot.min_samples * args.interval_seconds))
    print(f"warming up {warmup_seconds:.0f}s: the feature window is "
          f"{engine.bot.feature_window_ns / 1e9:.0f}s and needs "
          f"{engine.bot.min_samples} observations before any frame exists",
          flush=True)
    time.sleep(warmup_seconds)

    while not stopping["now"]:
        started = time.monotonic()
        try:
            result = engine.poll_once(time.time_ns())
        except Exception as exc:                          # noqa: BLE001
            print(f"poll failed: {type(exc).__name__}: {exc}", flush=True)
            raise
        counts = engine.counts
        print(f"poll {counts.polls}: {result} | opened={counts.opened} "
              f"closed={counts.closed} abstain={counts.abstentions} "
              f"gate_refused={counts.gate_refusals} missed={counts.missed_entries} "
              f"pnl={engine.realised_pnl}", flush=True)
        engine.features.reset_flow()
        if args.max_polls and counts.polls >= args.max_polls:
            break
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, args.interval_seconds - elapsed))

    engine.feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
