"""What the exchanges we trade on are holding, and which way it is moving.

`FEATURES.md` §1 P2, ledger DM-025. Two numbers per exchange: the reserve it
holds and the net flow in or out of it. Large inflows to exchange wallets ahead
of a fall are a known pattern, and a reserve that is quietly leaving is the
shape every venue failure has had.

## The prior art was not reusable, and the reason is worth keeping

The ledger points DM-025 at `analysis/onchain_advanced.py` in the early repos.
Read rather than trusted (2026-08-10), its `get_exchange_reserves` needs a
Coinglass key, and its keyless fallback returns:

    "total_reserve": circ * 0.15,   # rough estimate: ~15% of supply on exchanges
    "change_24h":    0,

A constant times circulating supply, under the name `total_reserve`, and a
hardcoded zero under `change_24h`. Any consumer reading that netflow saw "no
flow" forever, and the only tell was a `source` field nobody was required to
read. The Coinglass endpoint it preferred is also dead - that exact URL returns
HTTP 500 today, and the current API answers 401 without a key. This module
shares nothing with it.

## The source, and what it is not

DefiLlama's CEX transparency dataset - free, keyless, 91 exchanges, of which 79
carry values. It works by labelling on-chain addresses as belonging to an
exchange and summing them, so **it is a third party's measurement of somebody
else's wallets, not the venue's own statement.** That is carried per row in
`source` rather than left implicit, because it is the difference between a
number the venue is accountable for and a number an aggregator inferred.

Twelve of the 91 are listed with no values at all - Coinbase among them, which
publishes no wallet set. Those are REFUSED and counted. A zero reserve for an
exchange holding billions is the single most dangerous number this module could
produce, and it is exactly what the prior art produced.

## Own-token share, which is why both reserve numbers are kept

DefiLlama publishes total assets and "clean" assets - the second excludes the
exchange's own token. The gap between them is collateral the exchange printed,
and pricing it at market is the arithmetic that ended FTX. Measured on the first
live poll (2026-08-10): Binance 15.0% of $139.7bn, Bybit 8.9% of $13.1bn. Both
are venues this system trades on, so this is not a market-colour statistic.

## Why the poll is the observation

There is no free history for this. A reading not taken today cannot be taken
later at all, which is the same argument the raw archive rests on (DM-030). So
availability time is the fetch and that is honest - unlike a backfill, each row
here IS an observation we made, and the dataset is legitimately readable by a
backtest whose clock is after the poll.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from store.parquet_partition import append_partition
from store.temporal_schema import (
    AVAILABILITY_TIME, EVENT_TIME, INGESTION_TIME, SYMBOL, VENUE,
)

DATASET = "exchange_reserves"
_CEX_URL = "https://api.llama.fi/cexs"
SOURCE = "defillama"
# The reserve is the whole exchange rather than one instrument, and this is the
# word the rest of the pipeline already uses for a market-wide reading.
_ALL_MARKET_SYMBOL = "ALL"


@dataclass(frozen=True)
class ExchangeReserve:
    """One exchange's holdings and flows, as one poll saw them.

    Amounts are US dollars. `netflow_*_usd` is signed with the venue as the
    destination: **positive means value moved TO the exchange**. The sign is
    the entire content of the number and a convention nobody wrote down is a
    convention that gets read backwards.
    """

    exchange: str
    slug: str
    total_reserve_usd: float
    clean_reserve_usd: float | None      # excludes the exchange's own token
    netflow_24h_usd: float | None
    netflow_7d_usd: float | None
    netflow_30d_usd: float | None
    spot_volume_usd: float | None
    open_interest_usd: float | None
    fetched_at_ns: int

    @property
    def own_token_share(self) -> float | None:
        """Fraction of reserves that is the exchange's own token.

        None when the clean figure is missing - not zero. Zero here would read
        as an exchange holding no self-issued collateral, which is the most
        reassuring thing this module could say and the least safe thing to
        guess.
        """
        if self.clean_reserve_usd is None or self.total_reserve_usd <= 0:
            return None
        return (self.total_reserve_usd - self.clean_reserve_usd) / self.total_reserve_usd


def _number_or_none(value) -> float | None:
    """A field the source may omit entirely, and does for 12 of its 91 rows."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None      # NaN is absent, not zero


def parse_cex_transparency(payload: str, fetched_at_ns: int) -> tuple[list[ExchangeReserve], dict]:
    """Read one CEX transparency response into rows, and count what was refused.

    Returns the rows AND the refusals, because an exchange dropped in silence
    is indistinguishable from an exchange that was never listed.
    """
    refused = {"no_total_reserve": 0, "no_name": 0, "malformed": 0}
    try:
        body = json.loads(payload)
    except (TypeError, ValueError):
        return [], {**refused, "malformed": 1}
    if not isinstance(body, dict) or not isinstance(body.get("cexs"), list):
        return [], {**refused, "malformed": 1}

    rows: list[ExchangeReserve] = []
    for entry in body["cexs"]:
        if not isinstance(entry, dict):
            refused["malformed"] += 1
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            refused["no_name"] += 1
            continue
        total = _number_or_none(entry.get("currentTvl"))
        if total is None or total <= 0:
            # Listed with nothing measured - Coinbase publishes no wallet set,
            # and eleven others are the same. Counted, never zero-filled.
            refused["no_total_reserve"] += 1
            continue
        rows.append(ExchangeReserve(
            exchange=name,
            slug=str(entry.get("slug") or name),
            total_reserve_usd=total,
            clean_reserve_usd=_number_or_none(entry.get("cleanAssetsTvl")),
            netflow_24h_usd=_number_or_none(entry.get("inflows_24h")),
            netflow_7d_usd=_number_or_none(entry.get("inflows_1w")),
            netflow_30d_usd=_number_or_none(entry.get("inflows_1m")),
            spot_volume_usd=_number_or_none(entry.get("spotVolume")),
            open_interest_usd=_number_or_none(entry.get("oi")),
            fetched_at_ns=fetched_at_ns))
    return rows, refused


def build_reserves_frame(rows: list[ExchangeReserve]) -> pd.DataFrame:
    """One bitemporal row per exchange per poll.

    Event time is the fetch as well as availability, and that is not laziness:
    the source publishes no timestamp of its own, so the only instant this
    reading can honestly be attached to is the one we asked at. Claiming an
    event time we were not given is the same error as stamping a backfill at
    bar close.
    """
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame([{
        SYMBOL: _ALL_MARKET_SYMBOL,
        VENUE: r.slug,
        "exchange": r.exchange,
        "total_reserve_usd": r.total_reserve_usd,
        "clean_reserve_usd": r.clean_reserve_usd,
        "own_token_share": r.own_token_share,
        "netflow_24h_usd": r.netflow_24h_usd,
        "netflow_7d_usd": r.netflow_7d_usd,
        "netflow_30d_usd": r.netflow_30d_usd,
        "spot_volume_usd": r.spot_volume_usd,
        "open_interest_usd": r.open_interest_usd,
        # A third party's measurement of somebody else's wallets. Carried per
        # row so a frame that is filtered or joined elsewhere still says whose
        # arithmetic it is.
        "source": SOURCE,
        EVENT_TIME: r.fetched_at_ns,
        INGESTION_TIME: r.fetched_at_ns,
        AVAILABILITY_TIME: r.fetched_at_ns,
    } for r in rows])

    for column in (EVENT_TIME, INGESTION_TIME, AVAILABILITY_TIME):
        frame[column] = frame[column].astype("int64")
    # Pinned so a poll where every exchange omitted a field still unifies with
    # one where they did not - the all-null column defect the funding dataset
    # was broken by once already.
    frame["source"] = frame["source"].astype("string")
    frame["exchange"] = frame["exchange"].astype("string")
    for column in ("clean_reserve_usd", "own_token_share", "netflow_24h_usd",
                   "netflow_7d_usd", "netflow_30d_usd", "spot_volume_usd",
                   "open_interest_usd"):
        frame[column] = frame[column].astype("float64")
    return frame


def _fetch(url: str, timeout: float = 25.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-system/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def poll_exchange_reserves(store_root: Path, fetch=_fetch,
                           now_ns=time.time_ns) -> dict:
    """Fetch one snapshot and append it. One partition per poll.

    A failed fetch is reported, not raised: this runs on a loop beside builds
    that must not stop because a third-party endpoint had a bad minute.
    """
    fetched_at = now_ns()
    try:
        payload = fetch(_CEX_URL)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return {"dataset": DATASET, "source": SOURCE, "exchanges": 0,
                "appended": False, "error": str(error)}

    rows, refused = parse_cex_transparency(payload, fetched_at)
    frame = build_reserves_frame(rows)
    if frame.empty:
        return {"dataset": DATASET, "source": SOURCE, "exchanges": 0,
                "appended": False, "refused": refused}

    append_partition(store_root, DATASET, frame,
                     snapshot_id=f"{DATASET}-{SOURCE}-{fetched_at}")
    return {"dataset": DATASET, "source": SOURCE, "exchanges": int(len(frame)),
            "appended": True, "refused": refused,
            "fetched_at_ns": int(fetched_at)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Poll exchange reserves and netflow into the store")
    parser.add_argument("--store-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(poll_exchange_reserves(Path(args.store_root))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
