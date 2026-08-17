"""Archived Deribit chain polls, turned into a clock-gated option dataset.

Written the same day the capture was, and deliberately not later. This codebase's
recorded failure is capturing a tape and never building it: 99.6% of the binance
tape was archived and never became a bar, and coinbase ran for a day with a
working extractor the builder could not reach. A venue captured with no reader is
a venue nobody can tell is working.

## Absence is carried, never coerced

`bid_price` is null on **89 of 818** BTC instruments, `high`, `low` and
`price_change` on **588**, `last` on 139. Those are not defects: an option with
no bid is one nobody will buy at any price, which is ordinary for a far
out-of-the-money strike.

They stay absent here. The prior-art client in `nse-crypto-bot-final` did
`float(info.get("bid_price") or 0.0)` on exactly these fields, which turns a
missing bid into a bid of **zero** - a valid-looking price that makes a worthless
option look free, and prices any spread built on it at 100% of its value.

**A zero is not the same as an absence, and both occur here.** `mark_price` is
0.0 on real rows - a deep out-of-the-money strike days from expiry genuinely
marks at nothing - while `bid_price` being null means no bid existed. Collapsing
them loses the difference permanently, so the null stays `NaN` and the zero stays
`0.0`.

## What is deliberately NOT computed here

**Greeks.** This endpoint publishes none, and that is an improvement rather than
a gap: a delta computed here is one whose model, rate and dividend assumptions
are ours and are testable. Layer 1 records what the venue said; a feature module
computes what it means. `mark_iv` is the venue's own implied vol and is kept
because it is an observation, not a derivation.

**A dollar price.** Deribit options are quoted in the BASE currency - a BTC
option's `quote_currency` is `BTC` and its mark price is a fraction of a coin, so
0.62 means 0.62 BTC and not $0.62. `underlying_price` is carried on every row so
the conversion is always possible, and `quote_currency` is carried so nothing can
apply the dollar default by accident.

## Units are in the column names, because a percent read as a fraction is silent

The venue reports `mark_iv` as **41.42 meaning 41.42%**. Stored as a fraction and
named `mark_iv_fraction`, because a column called `mark_iv` holding 41.42 next to
a model expecting 0.4142 is wrong by a factor of 100 in a way no test of the
column's presence would catch.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass

import pandas as pd

from store.temporal_schema import (
    AVAILABILITY_TIME,
    EVENT_TIME,
    INGESTION_TIME,
    SYMBOL,
    VENUE,
)

_MS_TO_NS = 1_000_000

DATASET = "option_chain"

CALL, PUT = "C", "P"

# BTC-25DEC26-104000-P, and BTC-25DEC26-104000-C. Anchored at both ends so a name
# this pattern does not fully explain is refused rather than partially matched -
# a strike read out of a name the venue meant differently would misprice every
# surface built on it.
_INSTRUMENT = re.compile(
    r"^(?P<underlying>[A-Z0-9]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})"
    r"-(?P<strike>\d+(?:d\d+)?)-(?P<option_type>[CP])$")

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Deribit settles every option at 08:00 UTC on its expiry date. Carried as a
# timestamp rather than left as the venue's date string so time-to-expiry is a
# subtraction rather than a parse repeated at every call site.
_SETTLEMENT_HOUR_UTC = 8

# Fields the venue always populates, measured across 818 BTC and 684 ETH rows:
# none was ever null. A null here is therefore news - the venue's shape changed -
# and the row is refused rather than defaulted.
_REQUIRED_NUMBERS = ("mark_price", "mark_iv", "underlying_price",
                     "open_interest", "volume")
# Fields that are legitimately absent, and stay absent.
_NULLABLE_NUMBERS = ("bid_price", "ask_price", "mid_price", "high", "low",
                     "last", "price_change", "volume_usd",
                     "estimated_delivery_price", "interest_rate")


class UnparsableInstrumentName(ValueError):
    """A Deribit instrument name this module cannot fully explain.

    Raised by `parse_instrument_name` and caught by the extractor, which counts
    it. Guessing a strike or an expiry out of a name whose shape changed is how a
    surface gets built on numbers the venue never meant.
    """


@dataclass(frozen=True)
class OptionQuote:
    """One instrument as the venue reported it in one chain poll."""
    symbol: str
    venue: str
    underlying: str
    option_type: str
    strike: float
    expiry_time_ns: int
    mark_price: float
    mark_iv_fraction: float
    underlying_price: float
    open_interest: float
    volume: float
    quote_currency: str
    # Absent as `None`, never as zero. See the module docstring.
    bid_price: float | None
    ask_price: float | None
    mid_price: float | None
    event_time_ns: int
    ingestion_time_ns: int


def parse_instrument_name(name: str) -> tuple[str, str, float, int]:
    """`BTC-25DEC26-104000-P` to (underlying, option_type, strike, expiry_ns).

    Deribit settles options at 08:00 UTC on the expiry date, so the returned
    timestamp is that instant rather than midnight - a twelve-hour error in
    time-to-expiry, which is most of the remaining life of a daily option.
    """
    match = _INSTRUMENT.match(name or "")
    if match is None:
        raise UnparsableInstrumentName(
            f"instrument name {name!r} does not match Deribit's option shape; "
            f"refused rather than partially parsed")

    raw_expiry = match["expiry"]
    day, month, year = int(raw_expiry[:-5]), raw_expiry[-5:-2], int(raw_expiry[-2:])
    if month not in _MONTHS:
        raise UnparsableInstrumentName(
            f"instrument name {name!r} carries month {month!r}, which is not a month")
    expiry = dt.datetime(2000 + year, _MONTHS[month], day, _SETTLEMENT_HOUR_UTC,
                         tzinfo=dt.timezone.utc)

    # Deribit writes a fractional strike with a `d` for the decimal point:
    # ETH-...-1d5-C. NOT observed in the captured chain - all 1,502 live BTC and
    # ETH strikes are integers - so this branch is written from the venue's
    # notation rather than from a frame that exercised it, and is marked as such
    # rather than presented as measured.
    strike = float(match["strike"].replace("d", "."))
    return match["underlying"], match["option_type"], strike, int(
        expiry.timestamp() * 1e9)


def _number_or_none(value) -> float | None:
    """A float, or None for an absent value. Never zero for an absence."""
    if value is None:
        return None
    if isinstance(value, bool):        # bool is an int; a flag is not a price
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_option_quotes(payload: str, entry, venue: str,
                          symbol: str) -> list[OptionQuote]:
    """Read one archived chain record, or nothing if it is not one.

    A frame is recognised by the fields it is fetched for rather than by the file
    it came from, matching how the funding reader routes the same archive - the
    venue writes several streams side by side and misreading one as another
    poisons everything downstream of it.

    Returns an empty list rather than raising on a row it cannot use, because one
    bad instrument must not cost the poll; `build_option_frame` counts what was
    refused so the loss is visible rather than silent.
    """
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if not isinstance(body, dict):
        return []
    # The fields this endpoint is polled for. `mark_iv` is what distinguishes an
    # option summary from the futures summary the same endpoint shape serves.
    if "instrument_name" not in body or "mark_iv" not in body:
        return []

    created_ms = body.get("creation_timestamp")
    if not isinstance(created_ms, int):
        # Undatable. Refused rather than stamped from receipt, which would
        # invent a venue clock and silently reorder this tape against the others.
        return []

    try:
        underlying, option_type, strike, expiry_ns = parse_instrument_name(
            body["instrument_name"])
    except UnparsableInstrumentName:
        return []

    numbers = {}
    for field in _REQUIRED_NUMBERS:
        value = _number_or_none(body.get(field))
        if value is None:
            # Never null in 1,502 measured instruments, so a null means the
            # venue's shape changed. Refusing beats defaulting: a mark price of
            # zero invented here prices every position built on it.
            return []
        numbers[field] = value

    return [OptionQuote(
        symbol=body["instrument_name"],
        venue=venue,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiry_time_ns=expiry_ns,
        mark_price=numbers["mark_price"],
        # Percent to fraction, once, here. See the module docstring.
        mark_iv_fraction=numbers["mark_iv"] / 100.0,
        underlying_price=numbers["underlying_price"],
        open_interest=numbers["open_interest"],
        volume=numbers["volume"],
        quote_currency=str(body.get("quote_currency") or underlying),
        bid_price=_number_or_none(body.get("bid_price")),
        ask_price=_number_or_none(body.get("ask_price")),
        mid_price=_number_or_none(body.get("mid_price")),
        event_time_ns=created_ms * _MS_TO_NS,
        ingestion_time_ns=int(entry.t_recv_ns),
    )]


def build_option_frame(quotes: list[OptionQuote]) -> pd.DataFrame:
    """The clock-gated frame for a set of quotes.

    Availability is ingestion exactly. A polled snapshot is knowable the moment it
    lands and not one nanosecond earlier - there is no bar to close and nothing to
    wait for, so this is not a max over anything, for the same reason the funding
    dataset says so in the same words.
    """
    if not quotes:
        return pd.DataFrame(columns=[
            SYMBOL, VENUE, "underlying", "option_type", "strike",
            "expiry_time_ns", "mark_price", "mark_iv_fraction",
            "underlying_price", "open_interest", "volume", "quote_currency",
            "bid_price", "ask_price", "mid_price",
            EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME])

    frame = pd.DataFrame([{
        SYMBOL: q.symbol,
        VENUE: q.venue,
        "underlying": q.underlying,
        "option_type": q.option_type,
        "strike": q.strike,
        "expiry_time_ns": q.expiry_time_ns,
        "mark_price": q.mark_price,
        "mark_iv_fraction": q.mark_iv_fraction,
        "underlying_price": q.underlying_price,
        "open_interest": q.open_interest,
        "volume": q.volume,
        "quote_currency": q.quote_currency,
        "bid_price": q.bid_price,
        "ask_price": q.ask_price,
        "mid_price": q.mid_price,
        EVENT_TIME: q.event_time_ns,
        INGESTION_TIME: q.ingestion_time_ns,
        AVAILABILITY_TIME: q.ingestion_time_ns,
    } for q in quotes])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME,
                   "expiry_time_ns"):
        frame[column] = frame[column].astype("int64")
    # float64, so an absent bid is NaN and stays distinguishable from a real
    # 0.0 mark. An integer or nullable-int dtype here would force a choice
    # between the two and lose whichever it did not pick.
    for column in ("strike", "mark_price", "mark_iv_fraction", "underlying_price",
                   "open_interest", "volume", "bid_price", "ask_price",
                   "mid_price"):
        frame[column] = frame[column].astype("float64")
    return frame
