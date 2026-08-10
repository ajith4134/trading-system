"""Spot-perp basis, priced against the reference each venue actually funds on.

The catalogue calls this a core carry input (§1, P0), and the number itself is
one subtraction - the work of this module is refusing to take that subtraction
against the wrong reference. Binance and Bybit fund on **mark against index**;
Hyperliquid funds on **oracle**, hourly, capped - the same `[MISSED]` asymmetry
`cost.funding_carry` encodes for settlements. A basis computed venue-blind
against one shared reference is wrong at one end or the other, and wrong in the
flattering direction whenever the wrong reference sits nearer the mark.

Reads the Layer 1 `funding` dataset through `ClockGatedReader` - the same door
every consumer uses - so a backtest asking for the basis as of a past clock
sees exactly what was knowable then. No live REST, no raw-file reads.

A row missing its venue's reference price is REFUSED and counted, never
defaulted (the repo rule: no quote from a default). The refusals ride the
result so a consumer sees how much of the market it was not shown - a basis
table that silently dropped half its rows reads identically to a calm market
otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from store.clock_gated_reader import ClockGatedReader

_DATASET = "funding"
_BPS = Decimal(10_000)

# The price each venue settles funding against, by the venue's own `funds_on`
# declaration carried in the dataset. The mapping is from that declared string
# to the column holding the reference - not from the venue name, so a fourth
# venue arriving with `funds_on=oracle` is priced correctly without an edit
# here, and a venue declaring something unrecognised is refused rather than
# guessed at.
#
# Public because `features.price_divergence` judges the same gap this module
# measures, and two copies of this mapping would drift: the day a venue's
# declaration changed, one module would price it and the other would refuse it,
# and the two would disagree about the same instrument without either being
# obviously wrong.
REFERENCE_COLUMN_BY_FUNDS_ON = {
    "mark": "index_price",
    "oracle": "oracle_price",
}


@dataclass(frozen=True)
class BasisTable:
    """The computed basis and an account of what was refused.

    `rows` carries one basis per (venue, symbol) at the as-of clock:
    `basis_bps = (mark - reference) / reference`, in basis points, with the
    reference named per row. `refused` counts rows the dataset offered that
    could not be priced, by reason - visible, because a consumer who cannot
    see the refusals cannot tell sparse data from a quiet market.
    """
    rows: pd.DataFrame
    refused: dict[str, int]


def compute_spot_perp_basis(store_root: Path, as_of_ns: int,
                            custodian=None) -> BasisTable:
    """The newest basis per (venue, symbol) knowable at `as_of_ns`.

    Newest-per-key rather than the whole history: the basis is a state, and a
    consumer wanting its history can call this at the clocks it cares about -
    each read is clock-gated, so the history assembled that way is leak-free
    by construction.
    """
    reader = ClockGatedReader(Path(store_root), _DATASET, custodian=custodian)
    frame = reader.read_as_of(int(as_of_ns))

    refused = {"no_mark_price": 0, "no_reference_price": 0,
               "unrecognised_funds_on": 0, "unparseable_price": 0}
    if frame.empty:
        return BasisTable(rows=_empty_rows(), refused=refused)

    # Newest row per (venue, symbol) at the clock - order by event time and
    # keep the last. The dataset is append-only, so "newest" is well-defined.
    frame = (frame.sort_values("event_time_ns")
                  .groupby(["venue", "symbol"], as_index=False).last())

    out = {"venue": [], "symbol": [], "event_time_ns": [],
           "basis_bps": [], "reference": []}
    for row in frame.itertuples(index=False):
        reference_column = REFERENCE_COLUMN_BY_FUNDS_ON.get(row.funds_on)
        if reference_column is None:
            refused["unrecognised_funds_on"] += 1
            continue
        mark_raw = row.mark_price
        reference_raw = getattr(row, reference_column)
        if mark_raw is None or (isinstance(mark_raw, float) and pd.isna(mark_raw)):
            refused["no_mark_price"] += 1
            continue
        if reference_raw is None or (isinstance(reference_raw, float) and pd.isna(reference_raw)):
            refused["no_reference_price"] += 1
            continue
        try:
            mark = Decimal(str(mark_raw))
            reference = Decimal(str(reference_raw))
        except InvalidOperation:
            refused["unparseable_price"] += 1
            continue
        if reference <= 0:
            # A zero reference is the placeholder-price defect wearing a new
            # coat, and dividing by it is either an exception or an absurd
            # number depending on sign. Both are refusals.
            refused["unparseable_price"] += 1
            continue

        out["venue"].append(row.venue)
        out["symbol"].append(row.symbol)
        out["event_time_ns"].append(int(row.event_time_ns))
        out["basis_bps"].append((mark - reference) / reference * _BPS)
        out["reference"].append(reference_column)

    return BasisTable(rows=pd.DataFrame(out), refused=refused)


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame({"venue": [], "symbol": [], "event_time_ns": [],
                         "basis_bps": [], "reference": []})
