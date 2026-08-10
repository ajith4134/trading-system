"""The calendar curve: what the market charges to hold a contract until a date.

`FEATURES.md` §1 pairs this with the spot-perp basis as one P0 line, and only
half of it existed. `features.spot_perp_basis` prices the perpetual against the
reference its venue funds on; nothing read the 40 expiring contracts bybit
publishes across 9 underlyings, so the term half of "basis / term structure" was
a raw archive nobody opened. This is that half.

Three refusals carry the weight here, and each one exists because the number it
would otherwise produce looks perfectly reasonable.

**An annualised rate near expiry is mostly leverage on noise.** A 4-day contract
annualises at 91x, a 12-hour one at 730x, and the second number sorts to the top
of any table ranked by carry. Under `MIN_DAYS_TO_ANNUALISE` the raw basis is
still reported - it is a real price - and the annualised column is REFUSED
rather than filled. A curve that quietly annualises its front point is how a
tail-chasing strategy gets discovered.

**The underlying is established by evidence, not by splitting the symbol.**
`BTCUSDT-14AUG26` obviously means BTCUSDT to a human, but a suffix convention is
a convention and this repo has already been bitten by naming heuristics. The
split is a *candidate*, and the row is kept only when that candidate exists as a
perpetual on the same venue at the same clock. So the perp anchor is doing two
jobs: it is the near end of the curve, and it is the proof that the name means
what it looks like.

**The venue's own basis fields are stored and compared, never substituted.**
Bybit publishes `basis`, `basisRate` and `basisRateYear`, and measured on a real
frame they are not all against the same price - `basis` is against `lastPrice`,
`basisRateYear` annualises something closer to the mark. Ours is computed from
mark against index, the same pair `funding` is keyed on, and the venue's
annualised figure rides alongside as a corroborating witness with its
disagreement stated. A consumer can then see when the two diverge instead of
inheriting whichever one happened to be read.

Both datasets are read through `ClockGatedReader`, so a backtest asking for the
curve as of a past clock sees the contracts that had been polled by then and no
others.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from features.staleness import measure_staleness, stamp
from store.clock_gated_reader import ClockGatedReader

_DATED_DATASET = "dated_futures"
_FUNDING_DATASET = "funding"
_BPS = Decimal(10_000)
_NS_PER_DAY = Decimal(86_400_000_000_000)
# Days in a year for annualising a calendar basis. 365, not 252: a perpetual
# and a dated contract both carry over weekends, so a trading-day count would
# understate the holding period of every position this curve describes.
_DAYS_PER_YEAR = Decimal(365)

# Below this, the annualised column is refused rather than computed. One day
# multiplies the observed basis by 365; anything shorter multiplies it by more
# than the basis itself is measured to.
MIN_DAYS_TO_ANNUALISE = Decimal(1)

_COLUMNS = ("venue", "underlying", "symbol", "event_time_ns", "delivery_time_ns",
            "days_to_delivery", "mark_price", "index_price", "basis_bps",
            "annualised_basis_bps", "venue_annualised_bps",
            "venue_disagreement_bps", "perp_mark_price", "spread_to_perp_bps")


@dataclass(frozen=True)
class TermStructure:
    """The curve points knowable at one clock, and an account of what was refused.

    `rows` carries one point per (venue, symbol), sorted by underlying then
    delivery. `refused` counts what the datasets offered and this could not
    price, by reason - visible, because a consumer that cannot see the refusals
    cannot tell a thin curve from a quiet one.
    """
    rows: pd.DataFrame
    refused: dict[str, int]

    def curve(self, underlying: str, venue: str | None = None) -> pd.DataFrame:
        """The points of one underlying, nearest delivery first."""
        if self.rows.empty:
            return self.rows
        selected = self.rows[self.rows["underlying"] == underlying]
        if venue is not None:
            selected = selected[selected["venue"] == venue]
        return selected.sort_values("delivery_time_ns")


def _empty_rows() -> pd.DataFrame:
    return stamp(pd.DataFrame({column: [] for column in _COLUMNS}), {},
                 ["venue", "symbol"])


def _decimal_or_none(value) -> Decimal | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _newest_per_symbol(frame: pd.DataFrame) -> pd.DataFrame:
    """The latest row per (venue, symbol). The datasets are append-only, so
    the newest row of a key is that key's current state."""
    return (frame.sort_values("event_time_ns")
                 .groupby(["venue", "symbol"], as_index=False).last())


def _perp_marks(store_root: Path, as_of_ns: int, custodian) -> dict[tuple[str, str], Decimal]:
    """Mark price per (venue, perpetual symbol), through the same clock gate.

    Read from `funding` rather than from a symbol list: what makes a name a
    perpetual is that the venue was publishing a funding rate for it at this
    clock, which is exactly what a row in this dataset is.
    """
    reader = ClockGatedReader(Path(store_root), _FUNDING_DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))
    if frame.empty:
        return {}
    marks: dict[tuple[str, str], Decimal] = {}
    for row in _newest_per_symbol(frame).itertuples(index=False):
        mark = _decimal_or_none(row.mark_price)
        if mark is not None and mark > 0:
            marks[(row.venue, row.symbol)] = mark
    return marks


def compute_term_structure(store_root: Path, as_of_ns: int,
                           custodian=None) -> TermStructure:
    """Every dated contract knowable at `as_of_ns`, priced onto its curve.

    Tenor is measured from the as-of clock rather than from the poll's event
    time. The question a curve answers is "how long from now until this
    settles", and a stale poll of a contract does not make that contract
    further away.
    """
    as_of_ns = int(as_of_ns)
    refused = {"no_index_price": 0, "no_mark_price": 0, "unparseable_price": 0,
               "already_delivered": 0, "no_perp_anchor": 0,
               "too_near_expiry_to_annualise": 0}

    reader = ClockGatedReader(Path(store_root), _DATED_DATASET, custodian=custodian)
    frame = reader.read_as_of(as_of_ns)
    if frame.empty:
        return TermStructure(rows=_empty_rows(), refused=refused)

    perp_marks = _perp_marks(store_root, as_of_ns, custodian)

    out: dict[str, list] = {column: [] for column in _COLUMNS}
    for row in _newest_per_symbol(frame).itertuples(index=False):
        mark = _decimal_or_none(row.mark_price)
        index = _decimal_or_none(row.index_price)
        if mark is None or mark <= 0:
            refused["no_mark_price"] += 1
            continue
        if index is None:
            refused["no_index_price"] += 1
            continue
        if index <= 0:
            # A zero index is a placeholder wearing a price's clothes, and
            # dividing by it produces either an exception or an absurd carry.
            refused["unparseable_price"] += 1
            continue

        delivery_ns = int(row.delivery_time_ns)
        days = Decimal(delivery_ns - as_of_ns) / _NS_PER_DAY
        if days <= 0:
            # Settled, or settling this instant. Not a curve point any more -
            # and a negative tenor would annualise to a rate with the sign
            # flipped, which is worse than no number at all.
            refused["already_delivered"] += 1
            continue

        # The candidate underlying, and the evidence for it. Kept only if the
        # venue is publishing funding for that name right now.
        underlying = str(row.symbol).split("-", 1)[0]
        perp_mark = perp_marks.get((row.venue, underlying))
        if perp_mark is None:
            refused["no_perp_anchor"] += 1
            continue

        basis_bps = (mark - index) / index * _BPS
        if days >= MIN_DAYS_TO_ANNUALISE:
            annualised_bps = basis_bps * _DAYS_PER_YEAR / days
        else:
            annualised_bps = None
            refused["too_near_expiry_to_annualise"] += 1

        venue_year_rate = _decimal_or_none(getattr(row, "venue_basis_rate_year", None))
        venue_annualised_bps = (None if venue_year_rate is None
                                else venue_year_rate * _BPS)
        disagreement = (None if (venue_annualised_bps is None or annualised_bps is None)
                        else annualised_bps - venue_annualised_bps)

        out["venue"].append(row.venue)
        out["underlying"].append(underlying)
        out["symbol"].append(row.symbol)
        out["event_time_ns"].append(int(row.event_time_ns))
        out["delivery_time_ns"].append(delivery_ns)
        out["days_to_delivery"].append(days)
        out["mark_price"].append(mark)
        out["index_price"].append(index)
        out["basis_bps"].append(basis_bps)
        out["annualised_basis_bps"].append(annualised_bps)
        out["venue_annualised_bps"].append(venue_annualised_bps)
        out["venue_disagreement_bps"].append(disagreement)
        out["perp_mark_price"].append(perp_mark)
        out["spread_to_perp_bps"].append((mark - perp_mark) / perp_mark * _BPS)

    rows = pd.DataFrame(out)
    if not rows.empty:
        rows = rows.sort_values(["underlying", "delivery_time_ns"],
                                ignore_index=True)
    # FE-001. A curve is a statement about now, and its tenors are measured
    # from the as-of clock - so a contract priced off a poll that stopped
    # arriving keeps producing a confident annualised carry while the days to
    # delivery tick down around a frozen price.
    ages = {
        (venue, symbol): measure_staleness(group["event_time_ns"].astype("int64"),
                                           as_of_ns)
        for (venue, symbol), group in frame.groupby(["venue", "symbol"], sort=False)
    }
    return TermStructure(rows=stamp(rows, ages, ["venue", "symbol"]),
                         refused=refused)


def summarise_curves(structure: TermStructure) -> pd.DataFrame:
    """One line per (venue, underlying): how many tenors, and how the curve leans.

    `slope_bps_per_year` is the annualised basis of the LAST point minus the
    FIRST, over the points that could be annualised. It is stated per curve
    rather than as a single market-wide number because a term structure that is
    in contango on BTC and backwardated on DOGE has no meaningful average -
    averaging them produces a number that describes neither.

    A curve with fewer than two annualisable points reports `slope_bps_per_year`
    as None. One point is a price, not a structure.
    """
    if structure.rows.empty:
        return pd.DataFrame({"venue": [], "underlying": [], "tenors": [],
                             "nearest_days": [], "farthest_days": [],
                             "slope_bps_per_year": []})

    records = []
    for (venue, underlying), group in structure.rows.groupby(["venue", "underlying"]):
        ordered = group.sort_values("delivery_time_ns")
        annualisable = ordered[ordered["annualised_basis_bps"].notna()]
        slope = None
        if len(annualisable) >= 2:
            slope = (annualisable["annualised_basis_bps"].iloc[-1]
                     - annualisable["annualised_basis_bps"].iloc[0])
        records.append({
            "venue": venue,
            "underlying": underlying,
            "tenors": len(ordered),
            "nearest_days": ordered["days_to_delivery"].iloc[0],
            "farthest_days": ordered["days_to_delivery"].iloc[-1],
            "slope_bps_per_year": slope,
        })
    return pd.DataFrame(records).sort_values(["venue", "underlying"],
                                             ignore_index=True)
