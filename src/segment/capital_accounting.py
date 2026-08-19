"""BF-11: what a bot put at risk, and what it made on it, in USDT.

RL-028, in the user's words: *"i need te usdt profit or loss for te captal used
or ow muc capital orusdt by bot tradin ow muc profit it ot in usdt"*. Asked how
capital should be counted they chose all three, side by side, and the reason the
question was worth asking is that the three differ by orders of magnitude on the
same trades:

  PEAK AT RISK        the most USDT open at one moment. What the bot needed.
  TURNOVER            every entry's notional added up. What it traded through.
  EQUITY vs BANKROLL  a declared allocation plus realised P&L. What an account
                      holder would have seen.

A return quoted against an unnamed denominator is a number chosen to flatter, so
nothing here returns one number: a `CapitalReport` carries all three or the bot
has not traded.

## Units, and the error this module exists to refuse (RL-029)

Deribit options are quoted in BTC and ETH, not USDT. Three bots are already in
USDT and summing the fourth into a total without conversion is a unit error that
reads as a result - the same shape as the $837 loss in `DECISIONS.md`, where one
P&L was credited to all 36 features that could have produced it.

So every fill declares the currency it was priced in, and a fill priced in
anything but a dollar stablecoin needs a rate JOURNALLED WITH IT to be counted.
A fill with no rate is counted as UNCONVERTIBLE and named, never converted at a
rate nobody recorded and never quietly dropped - a dropped fill would make the
P&L smaller and the board would look tidier for it.

## Gross, and saying so

`gross_pnl` is what the engine journals today: it does not carry fees. Every
figure here is therefore labelled gross, because a gross P&L presented as net is
wrong in the flattering direction on every single trade.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

DEFAULT_STATE_ROOT = Path.home() / "capture" / "segment"

# Quote assets that ARE a dollar. A fill priced in one of these needs no rate.
# Deliberately a closed list: treating an unknown quote as a dollar is exactly
# the silent unit error this module refuses.
DOLLAR_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD")
# Quote assets that are NOT dollars and therefore need a journalled rate.
CRYPTO_QUOTES = ("BTC", "ETH", "BNB", "SOL")

UNCONVERTIBLE = "UNCONVERTIBLE"

# **The declared bankroll per bot (RL-028), and it lives HERE rather than in the
# bot declaration on purpose.**
#
# It is the denominator a display divides by, and a display must be able to read
# it without starting a trading bot. Measured 2026-08-19: reading it from
# `segment.bot_registry` cost the segment wall 32 seconds of import before it
# rendered a single tile, because that module pulls in the model registry, the
# feeds and the universes. `bot_registry` imports this value rather than owning
# it, so there is still exactly one number.
#
# The same figure for every bot on purpose: four bots on one allocation is what
# makes their returns comparable, and none has earned a larger share yet. It is a
# measurement base, not a limit - the risk gate owns limits and does not read it.
SEGMENT_BANKROLLS_USDT = {
    "perp": Decimal("1000"),
    "spot": Decimal("1000"),
    "dated": Decimal("1000"),
    "options": Decimal("1000"),
}


def bankroll_of(segment: str) -> Decimal:
    """The USDT allocation this bot's return is measured against."""
    return SEGMENT_BANKROLLS_USDT.get(segment, Decimal("1000"))


def quote_currency(venue: str, symbol: str) -> str:
    """What currency this instrument's PRICE is expressed in.

    Deribit names an option `BTC-26MAR27-68000-C` and quotes it in BTC - the
    settlement currency is the prefix, not a suffix, so the venue decides how the
    name is read rather than one shared rule guessing at both.
    """
    symbol = (symbol or "").upper()
    if venue == "deribit":
        # `BTC-26MAR27-68000-C`: the settlement currency is the PREFIX, and the
        # premium is quoted in it.
        head = symbol.split("-", 1)[0]
        return head if head in CRYPTO_QUOTES + DOLLAR_QUOTES else UNCONVERTIBLE
    # A dated contract carries its expiry after a separator - `BTCUSDT-28AUG26`,
    # `BTCUSDT_261225` - and the currency is in the part before it.
    base = symbol.split("-", 1)[0].split("_", 1)[0]
    for quote in sorted(DOLLAR_QUOTES + CRYPTO_QUOTES, key=len, reverse=True):
        if base.endswith(quote):
            return quote
    # **Everything else is unconvertible ON PURPOSE, and two live cases show why.**
    # `BTCUSDU26` is a bybit INVERSE contract: quoted in USD, settled in BTC, and
    # its P&L is not linear in the price - reading the `USD` off the end and
    # multiplying would be arithmetic that does not apply to the instrument.
    # `ACETRY` is quoted in Turkish lira, for which nothing here holds a rate.
    # Both are named rather than guessed at.
    return UNCONVERTIBLE


def _decimal(value) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def usd_rate_of(fill: dict) -> Decimal | None:
    """The rate this fill's price converts to USDT at, or None if there is none.

    A dollar quote is 1 by definition. Anything else must carry `usd_rate`,
    written by the engine at the moment of the fill from the price it actually
    saw. None means unconvertible, which is a reported state and not a zero.
    """
    currency = fill.get("quote_currency") or quote_currency(
        fill.get("venue", ""), fill.get("symbol", ""))
    if currency in DOLLAR_QUOTES:
        return Decimal(1)
    return _decimal(fill.get("usd_rate"))


@dataclass(frozen=True)
class CapitalReport:
    """One bot's capital and P&L. Every field was computed from journalled fills."""

    segment: str
    bankroll_usdt: Decimal
    peak_at_risk_usdt: Decimal
    open_now_usdt: Decimal
    turnover_usdt: Decimal
    realised_pnl_usdt: Decimal
    opens: int
    closes: int
    wins: int
    # Fills whose currency could not be converted, and the currencies involved.
    # Published rather than dropped: a dropped fill makes P&L look smaller and
    # the board look tidier, which is the wrong direction to be wrong in.
    unconvertible_fills: int = 0
    unconvertible_currencies: tuple = ()
    converted_fills: int = 0
    basis: str = "gross of fees"

    @property
    def equity_usdt(self) -> Decimal:
        return self.bankroll_usdt + self.realised_pnl_usdt

    def _ratio(self, denominator: Decimal) -> float | None:
        """A percentage, or None when there is nothing to divide by.

        None renders NOT MEASURED. A bot that has closed nothing has not made 0%
        - it has made no measurement, and those are different claims.
        """
        if self.closes == 0 or denominator <= 0:
            return None
        return float(self.realised_pnl_usdt / denominator) * 100.0

    @property
    def return_on_peak_pct(self) -> float | None:
        return self._ratio(self.peak_at_risk_usdt)

    @property
    def return_on_turnover_pct(self) -> float | None:
        return self._ratio(self.turnover_usdt)

    @property
    def return_on_bankroll_pct(self) -> float | None:
        return self._ratio(self.bankroll_usdt)

    @property
    def winrate_pct(self) -> float | None:
        # The same floor the segment wall uses: a rate over a handful of trades
        # is not a rate (PB-10).
        if self.closes < 20:
            return None
        return 100.0 * self.wins / self.closes

    def as_dict(self) -> dict:
        return {
            "segment": self.segment,
            "bankroll_usdt": str(self.bankroll_usdt),
            "peak_at_risk_usdt": str(self.peak_at_risk_usdt),
            "open_now_usdt": str(self.open_now_usdt),
            "turnover_usdt": str(self.turnover_usdt),
            "realised_pnl_usdt": str(self.realised_pnl_usdt),
            "equity_usdt": str(self.equity_usdt),
            "return_on_peak_pct": self.return_on_peak_pct,
            "return_on_turnover_pct": self.return_on_turnover_pct,
            "return_on_bankroll_pct": self.return_on_bankroll_pct,
            "opens": self.opens, "closes": self.closes, "wins": self.wins,
            "winrate_pct": self.winrate_pct,
            "converted_fills": self.converted_fills,
            "unconvertible_fills": self.unconvertible_fills,
            "unconvertible_currencies": list(self.unconvertible_currencies),
            "basis": self.basis,
        }


def account_for_fills(fills, *, segment: str,
                      bankroll_usdt: Decimal) -> CapitalReport:
    """Walk one bot's fills in time order and account for them.

    Peak at risk needs the walk: it is the largest the sum of open notionals ever
    reached, which no per-trade total can recover. A CLOSE whose OPEN is not in
    the window still counts its P&L - the journals roll daily and a position held
    across midnight would otherwise vanish from the record that matters most.
    """
    open_notionals: dict[tuple, Decimal] = {}
    peak = Decimal(0)
    turnover = Decimal(0)
    realised = Decimal(0)
    opens = closes = wins = converted = unconvertible = 0
    unconvertible_currencies: set[str] = set()

    for fill in sorted(fills, key=lambda f: f.get("at_ns") or 0):
        rate = usd_rate_of(fill)
        currency = fill.get("quote_currency") or quote_currency(
            fill.get("venue", ""), fill.get("symbol", ""))
        if rate is None or rate <= 0:
            unconvertible += 1
            unconvertible_currencies.add(currency)
            continue
        converted += 1
        key = (fill.get("venue"), fill.get("symbol"))
        quantity = _decimal(fill.get("quantity")) or Decimal(0)
        price = _decimal(fill.get("price")) or Decimal(0)

        if fill.get("event") == "OPEN":
            opens += 1
            notional = quantity * price * rate
            turnover += notional
            open_notionals[key] = open_notionals.get(key, Decimal(0)) + notional
            total = sum(open_notionals.values())
            peak = max(peak, total)
        elif fill.get("event") == "CLOSE":
            closes += 1
            # `net_pnl` when the engine ever journals one, `gross_pnl` today. The
            # basis is stated on the report rather than assumed by the reader.
            pnl = _decimal(fill.get("net_pnl"))
            if pnl is None:
                pnl = _decimal(fill.get("gross_pnl")) or Decimal(0)
            realised += pnl * rate
            if pnl > 0:
                wins += 1
            open_notionals.pop(key, None)

    return CapitalReport(
        segment=segment,
        bankroll_usdt=bankroll_usdt,
        peak_at_risk_usdt=peak,
        open_now_usdt=sum(open_notionals.values()) or Decimal(0),
        turnover_usdt=turnover,
        realised_pnl_usdt=realised,
        opens=opens, closes=closes, wins=wins,
        converted_fills=converted,
        unconvertible_fills=unconvertible,
        unconvertible_currencies=tuple(sorted(unconvertible_currencies)),
    )


def read_fills(segment: str, state_root: Path = DEFAULT_STATE_ROOT,
               days: int = 0) -> list:
    """Every journalled fill for one bot, oldest first.

    `days` limits the read to the newest N day files; 0 reads them all. The
    accounting is over a bot's whole life by default because peak at risk and
    turnover are life-to-date quantities, and a window would silently redefine
    both.
    """
    paths = sorted((state_root / segment).glob("fills-*.ndjson"))
    if days:
        paths = paths[-days:]
    rows = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            continue
    return rows


def account_for_segment(segment: str, *, bankroll_usdt: Decimal,
                        state_root: Path = DEFAULT_STATE_ROOT,
                        days: int = 0) -> CapitalReport:
    return account_for_fills(read_fills(segment, state_root, days),
                             segment=segment, bankroll_usdt=bankroll_usdt)
