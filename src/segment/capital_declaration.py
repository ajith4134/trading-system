"""The declared capital budget the perp and spot bots may spend — CL-01, RL-040.

One JSON file, `~/capture/segment/capital.json`, is the only place capital is
declared. This module parses it, validates it, and reloads it live. **Nothing else
in the tree may read the raw file** — a second reader is a second interpretation,
and two interpretations of a budget is how a limit stops being one.

## What changed, and why this is not part of `capital_accounting`

`capital_accounting.SEGMENT_BANKROLLS_USDT` says of its own number:

> It is a measurement base, not a limit - the risk gate owns limits and does not
> read it.

RL-040 overturns exactly that sentence. The declared figure becomes a BUDGET that
is actually spent and can actually be exhausted, and the gate now reads it. The two
ideas keep separate homes because they answer different questions: `capital_accounting`
owns the DENOMINATOR a display divides by, and must stay importable without starting
a bot; this module owns the LIMIT a bot spends against.

## Capital per trade means MARGIN POSTED

RL-040, chosen deliberately over position notional. At 10x leverage a 50 USDT
maximum means positions up to 500 USDT of notional. So:

    notional = margin x leverage      what P&L and excursions are computed on
    margin                            what the pool is debited

Mixing the two is the unit error this file exists to prevent, so the two are never
carried in one field and never named alike.

## Why a rejected file stops the bots instead of falling back

There is no default. A missing or malformed declaration means the bots STOP OPENING
new positions and say `NO_CAPITAL_DECLARATION`; positions already open are still
managed and closed normally, because refusing to manage a live position would be a
worse failure than any configuration error.

A fallback default would be capital chosen by nobody, sized by whoever last edited a
constant, and it would look exactly like a working system. The same reasoning gives
`segment_bot_supervisor.sh` no default segment: "a supervisor that guessed would be
starting a bot nobody chose."

## Floats are refused, not rounded

Every numeric field is given as a STRING and parsed to `Decimal`. A JSON float has
already lost precision before this module sees it, so accepting one would be
accepting a number nobody typed. `0.1` is not `0.1`, and a budget is the last place
to discover that.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from segment.capital_accounting import DEFAULT_STATE_ROOT

# The bots this declaration governs. Dated and options are switched off (RL-039),
# so a ceiling for them is neither required nor read; naming the tuple here keeps
# validation and readers from disagreeing about who is in scope.
GOVERNED_SEGMENTS = ("perp", "spot")

DECLARATION_FILENAME = "capital.json"

# The leverage rules this code actually implements. RL-041 requires the choosing
# rule to be DECLARED, and a declared name nothing implements is worse than no name
# at all - it reads as a decision that was made.
#
#   volatility_targeted  leverage = target annual vol / that instrument's realised
#                        vol, clamped. Quiet instrument, more leverage; violent one,
#                        less. Every position then carries about the same RISK,
#                        which is the point. The RL-041 default.
#   fixed                every trade takes the segment's ceiling. Predictable, and
#                        the honest choice when you want to pin 10x and read the
#                        result without a second moving part.
#
# Deliberately NOT offered: confidence-scaled. Both bots journal `calibrated=False`,
# and scaling leverage by an uncalibrated confidence compounds the miscalibration
# exactly where it costs most (RL-041).
LEVERAGE_RULES = ("volatility_targeted", "fixed")

# The refusal a caller publishes when there is no usable declaration. It is a named
# refusal and never an abstention: a bot that is broke must not render as a bot that
# found nothing (Rule 8).
NO_DECLARATION = "NO_CAPITAL_DECLARATION"


class DeclarationRejected(ValueError):
    """The declaration could not be used, with the reason needed to fix it.

    Carries the field name so the board can say WHICH line of the file is wrong. A
    validation error that only says "invalid" sends the reader back to the file to
    guess, which is where a wrong number gets typed a second time.
    """

    def __init__(self, field: str, detail: str) -> None:
        self.field = field
        self.detail = detail
        super().__init__(f"{field}: {detail}")


@dataclass(frozen=True)
class CapitalDeclaration:
    """What the two bots may spend, and how they may lever it."""

    declared_at: str
    portfolio_usdt: Decimal
    per_bot_cap_fraction: Decimal
    min_margin_per_trade_usdt: Decimal
    max_margin_per_trade_usdt: Decimal
    leverage_rule: str
    leverage_floor: Decimal
    leverage_ceiling: dict[str, Decimal]
    target_annual_vol_pct: Decimal
    spot_borrow_annual_pct: Decimal
    maintenance_margin_rate: Decimal
    source_path: str = ""
    source_mtime_ns: int = 0

    def cap_for(self, segment: str) -> Decimal:
        """The most USDT of margin this one bot may hold at once.

        The per-bot cap is what stops a shared pool becoming a race the faster bot
        always wins. perp polls the same 6-second interval as spot but scans a
        larger universe, so without this it would routinely reach a setup first and
        hold capital spot never gets to see.
        """
        if segment not in GOVERNED_SEGMENTS:
            raise DeclarationRejected(
                "segment",
                f"{segment!r} is not governed by this declaration; governed "
                f"segments are {', '.join(GOVERNED_SEGMENTS)}")
        return self.portfolio_usdt * self.per_bot_cap_fraction

    def ceiling_for(self, segment: str) -> Decimal:
        """The most leverage this bot may take on any single trade."""
        try:
            return self.leverage_ceiling[segment]
        except KeyError:
            raise DeclarationRejected(
                f"leverage.ceiling.{segment}",
                f"no ceiling declared; governed segments are "
                f"{', '.join(GOVERNED_SEGMENTS)}") from None

    def as_dict(self) -> dict:
        """For the board and the journal. Decimals as strings, as they were given."""
        return {
            "declared_at": self.declared_at,
            "portfolio_usdt": str(self.portfolio_usdt),
            "per_bot_cap_fraction": str(self.per_bot_cap_fraction),
            "min_margin_per_trade_usdt": str(self.min_margin_per_trade_usdt),
            "max_margin_per_trade_usdt": str(self.max_margin_per_trade_usdt),
            "leverage_rule": self.leverage_rule,
            "leverage_floor": str(self.leverage_floor),
            "leverage_ceiling": {k: str(v) for k, v in self.leverage_ceiling.items()},
            "target_annual_vol_pct": str(self.target_annual_vol_pct),
            "spot_borrow_annual_pct": str(self.spot_borrow_annual_pct),
            "maintenance_margin_rate": str(self.maintenance_margin_rate),
            "source_path": self.source_path,
            "source_mtime_ns": self.source_mtime_ns,
        }


def _decimal_field(payload: dict, field: str) -> Decimal:
    """One numeric field, given as a string, to `Decimal`. Floats are refused."""
    if field not in payload:
        raise DeclarationRejected(field, "missing")
    raw = payload[field]
    if isinstance(raw, float):
        raise DeclarationRejected(
            field,
            f"given as a JSON float ({raw!r}); write it as a string so the value "
            f"that was typed is the value that is used")
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise DeclarationRejected(
            field, f"expected a numeric string, got {type(raw).__name__}")
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        raise DeclarationRejected(field, f"not a number: {raw!r}") from None


def parse_declaration(payload: dict, *, source_path: str = "",
                      source_mtime_ns: int = 0) -> CapitalDeclaration:
    """Validate one declaration whole, or reject it whole.

    There is no partial acceptance. A file with a good portfolio total and a bad
    margin band is not a file with one good number in it - it is a file whose
    author's intent is unknown, and guessing which half they meant is how a budget
    ends up being half of one.
    """
    if not isinstance(payload, dict):
        raise DeclarationRejected("<file>", "top level is not a JSON object")

    declared_at = payload.get("declared_at")
    if not isinstance(declared_at, str) or not declared_at:
        raise DeclarationRejected(
            "declared_at",
            "missing; every declaration is stamped so a stale one is visible")

    portfolio = _decimal_field(payload, "portfolio_usdt")
    if portfolio <= 0:
        raise DeclarationRejected("portfolio_usdt", f"must be > 0, got {portfolio}")

    cap_fraction = _decimal_field(payload, "per_bot_cap_fraction")
    if not 0 < cap_fraction <= 1:
        raise DeclarationRejected(
            "per_bot_cap_fraction",
            f"must be in (0, 1]; got {cap_fraction}. Above 1 would let one bot "
            f"reserve capital the portfolio does not hold")

    min_margin = _decimal_field(payload, "min_margin_per_trade_usdt")
    max_margin = _decimal_field(payload, "max_margin_per_trade_usdt")
    if min_margin <= 0:
        raise DeclarationRejected(
            "min_margin_per_trade_usdt",
            f"must be > 0, got {min_margin}. A floor of zero is what the system "
            f"does today, and it is why the median trade used 0.00014 USDT")
    if max_margin < min_margin:
        raise DeclarationRejected(
            "max_margin_per_trade_usdt",
            f"is below the minimum ({max_margin} < {min_margin}); no trade could "
            f"ever be sized")

    bot_cap = portfolio * cap_fraction
    if max_margin > bot_cap:
        raise DeclarationRejected(
            "max_margin_per_trade_usdt",
            f"{max_margin} exceeds the per-bot cap of {bot_cap} "
            f"({portfolio} x {cap_fraction}); one trade could never be opened")

    leverage = payload.get("leverage")
    if not isinstance(leverage, dict):
        raise DeclarationRejected("leverage", "missing or not an object")

    rule = leverage.get("rule")
    if rule not in LEVERAGE_RULES:
        raise DeclarationRejected(
            "leverage.rule",
            f"{rule!r} is not implemented; known rules are {', '.join(LEVERAGE_RULES)}")

    floor = _decimal_field(leverage, "floor")
    if floor < 1:
        raise DeclarationRejected(
            "leverage.floor",
            f"must be at least 1, got {floor}. Below 1 is not deleveraging, it is "
            f"posting more margin than the position is worth")

    ceilings_raw = leverage.get("ceiling")
    if not isinstance(ceilings_raw, dict):
        raise DeclarationRejected("leverage.ceiling", "missing or not an object")
    ceilings: dict[str, Decimal] = {}
    for segment in GOVERNED_SEGMENTS:
        value = _decimal_field(ceilings_raw, segment)
        if value < floor:
            raise DeclarationRejected(
                f"leverage.ceiling.{segment}",
                f"ceiling {value} is below the floor {floor}")
        ceilings[segment] = value

    target_vol = _decimal_field(leverage, "target_annual_vol_pct")
    if target_vol <= 0:
        raise DeclarationRejected(
            "leverage.target_annual_vol_pct", f"must be > 0, got {target_vol}")

    borrow = _decimal_field(payload, "spot_borrow_annual_pct")
    if borrow < 0:
        raise DeclarationRejected(
            "spot_borrow_annual_pct",
            f"must be >= 0, got {borrow}. Negative borrow is being paid to borrow")
    if borrow == 0 and ceilings["spot"] > 1:
        raise DeclarationRejected(
            "spot_borrow_annual_pct",
            "is zero while spot leverage is above 1x. Leveraged spot is borrowed "
            "money, and a zero rate makes it a free loan that inflates spot's "
            "returns MORE the more leverage it takes (RL-041)")

    maintenance = _decimal_field(payload, "maintenance_margin_rate")
    if not 0 <= maintenance < 1:
        raise DeclarationRejected(
            "maintenance_margin_rate", f"must be in [0, 1), got {maintenance}")

    # At the ceiling, an isolated position liquidates on roughly a 1/L adverse move
    # less the maintenance margin. A ceiling so high that the distance is zero or
    # negative means a position that is liquidated at the moment it opens, and no
    # stop the risk gate sets could ever fire first (RL-041 §4a).
    for segment, ceiling in ceilings.items():
        distance = Decimal(1) / ceiling - maintenance
        if distance <= 0:
            raise DeclarationRejected(
                f"leverage.ceiling.{segment}",
                f"at {ceiling}x with maintenance {maintenance} the liquidation "
                f"distance is {distance}; the position would be liquidated on "
                f"open and no hard stop could ever fire first")

    return CapitalDeclaration(
        declared_at=declared_at,
        portfolio_usdt=portfolio,
        per_bot_cap_fraction=cap_fraction,
        min_margin_per_trade_usdt=min_margin,
        max_margin_per_trade_usdt=max_margin,
        leverage_rule=rule,
        leverage_floor=floor,
        leverage_ceiling=ceilings,
        target_annual_vol_pct=target_vol,
        spot_borrow_annual_pct=borrow,
        maintenance_margin_rate=maintenance,
        source_path=source_path,
        source_mtime_ns=source_mtime_ns)


def declaration_path(state_root: Path | None = None) -> Path:
    """Where the declaration lives, beside the per-segment state it governs."""
    return Path(state_root or DEFAULT_STATE_ROOT) / DECLARATION_FILENAME


class DeclarationReloader:
    """Reads the declaration once, then only when the file actually changes.

    **Reload is by mtime and applies to NEW ENTRIES ONLY (RL-030, RL-040).** An open
    position keeps the margin and leverage it was opened under. Silently
    re-margining a live position would make the journal describe a trade that was
    never taken, and would move a hard stop the risk gate already sized against a
    liquidation distance.

    A rejected file does NOT leave the previous declaration in force. That is the
    less obvious of the two choices and it is deliberate: an edit that fails to
    parse is an edit someone MEANT, and continuing to trade the superseded numbers
    would be trading against a stated intention while the board showed no problem.
    Stopping is loud, and loud is the point.
    """

    def __init__(self, path: Path | None = None, *,
                 state_root: Path | None = None) -> None:
        self.path = Path(path) if path is not None else declaration_path(state_root)
        self.current: CapitalDeclaration | None = None
        self.rejection: str | None = None
        self.loaded_mtime_ns: int | None = None
        self.reloads = 0

    def poll(self) -> CapitalDeclaration | None:
        """The declaration in force, reloaded if the file changed.

        `None` means the caller must refuse to open with `NO_CAPITAL_DECLARATION`,
        and `self.rejection` says why in words a person can act on.
        """
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            self._reject(
                f"no declaration at {self.path}; write one with "
                f"scripts/write_capital_declaration.sh", mtime_ns=None)
            return None
        except OSError as error:
            self._reject(f"cannot read {self.path}: {error}", mtime_ns=None)
            return None

        # The mtime alone decides, NOT `current is not None`. A rejected file leaves
        # `current` as None, so guarding on it would re-read and re-parse a broken
        # declaration on every six-second poll for as long as it stayed broken - and
        # the verdict would be identical every time.
        if self.loaded_mtime_ns == stat.st_mtime_ns:
            return self.current

        self.reloads += 1
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            self._reject(f"{self.path} is not readable JSON: {error}",
                         mtime_ns=stat.st_mtime_ns)
            return None

        try:
            declaration = parse_declaration(
                payload, source_path=str(self.path),
                source_mtime_ns=stat.st_mtime_ns)
        except DeclarationRejected as error:
            self._reject(str(error), mtime_ns=stat.st_mtime_ns)
            return None

        self.current = declaration
        self.loaded_mtime_ns = stat.st_mtime_ns
        self.rejection = None
        return declaration

    def _reject(self, reason: str, *, mtime_ns: int | None) -> None:
        """Drop the declaration and record why, without losing the mtime.

        The mtime is kept on a parse failure so a file that is broken and NOT being
        edited is not re-parsed every six seconds; it is cleared when the file is
        gone, because the next file to appear must always be read.
        """
        self.current = None
        self.loaded_mtime_ns = mtime_ns
        self.rejection = reason
