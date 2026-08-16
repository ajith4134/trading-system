"""The highest-carry rows of any naive screen are the ones you cannot hedge.

That is not a metaphor here - it is what the first live run produced. Without the
dollar-quoted universe wired in, the top four proposals were BTWUSDT, ESPORTSUSDT,
SPORTFUNUSDT and STARUSDT at 79% to 137% annualised, and 173 of the candidates
had no spot leg at all. The funding was paying for exactly the risk the hedge was
supposed to remove.

So the universe is a required argument, not an optional filter, and
`test_the_universe_is_required` is the test that keeps it that way.
"""
from decimal import Decimal

import pandas as pd
import pytest

from strategy.funding_carry import (
    DEFAULT_ORDER_TYPE,
    HEDGE_VENUES,
    CarrySelection,
    UniverseNotSupplied,
    flow_threshold,
    select,
)
from validation.trial_registry import TrialRegistry

_MINUTE = 60_000_000_000
_START = 1_700_000_000_000_000_000
_UNIVERSE_VENUES = ("binance", "binance-spot")


def _registry(tmp_path):
    return TrialRegistry(tmp_path / "trials")


def _funding_rows(symbol, venue, n, rate, mark="100.000", index="100.000"):
    return [{
        "venue": venue, "symbol": symbol, "funds_on": "mark",
        "funding_rate": Decimal(rate),
        "mark_price": Decimal(mark), "index_price": Decimal(index),
        "oracle_price": None,
        "event_time_is_receipt": False, "next_funding_time_unknown": False,
        "next_funding_time_ns": _START + 1, "funding_interval_hours": 8,
        "event_time_ns": _START + i * _MINUTE,
        "ingestion_time_ns": _START + i * _MINUTE,
        "availability_time_ns": _START + i * _MINUTE,
    } for i in range(n)]


def _store(tmp_path, rows):
    from store.parquet_partition import append_partition
    append_partition(tmp_path, "funding", pd.DataFrame(rows),
                     snapshot_id="carry-test")
    return tmp_path


def _universe(*symbols, spot=None):
    """Dollar-quoted sets for both feeds. `spot` defaults to the same symbols."""
    spot_symbols = symbols if spot is None else spot
    return {"binance": frozenset(symbols),
            "binance-spot": frozenset(spot_symbols)}


def _select(tmp_path, rows, universe, capacity=10, **kwargs):
    return select(_store(tmp_path, rows), _START + 200 * _MINUTE,
                  capacity=capacity, registry=_registry(tmp_path),
                  trial_name=kwargs.pop("trial_name", "test"),
                  dollar_quoted=universe, **kwargs)


# --- the universe is not optional ----------------------------------------

def test_the_universe_is_required():
    """It defaulted to None once, and skipping the filter skipped the check that
    a hedge leg exists at all - so the setup proposed shorting microcap perps
    against a spot market that may not list them, and nothing in the output said
    so."""
    with pytest.raises(TypeError):
        select(".", 0, capacity=10, registry=None, trial_name="x")   # noqa


def test_a_missing_venue_universe_is_refused_not_treated_as_everything(tmp_path):
    with pytest.raises(UniverseNotSupplied, match="naked short"):
        select(tmp_path, _START, capacity=10, registry=_registry(tmp_path),
               trial_name="partial", dollar_quoted={"binance": frozenset()})


def test_a_perp_with_no_dollar_quoted_spot_leg_is_declined(tmp_path):
    """The one that matters. A dollar-quoted perp with no dollar-quoted spot is
    a naked short with a hedge's name, and it is exactly what sits at the top of
    an unfiltered carry screen - because that is what the funding is paying for.
    """
    rows = _funding_rows("HOTUSDT", "binance", 60, "0.0012")
    selection = _select(tmp_path, rows,
                        _universe("HOTUSDT", spot=()))

    assert selection.proposals == []
    assert selection.declined["no_spot_leg"] == 1
    assert "STAND ASIDE" in selection.describe()


def test_a_symbol_not_dollar_quoted_on_the_perp_venue_is_declined(tmp_path):
    rows = _funding_rows("BTCTRY", "binance", 60, "0.0012")
    selection = _select(tmp_path, rows, _universe())

    assert selection.declined["not_dollar_quoted"] == 1


# --- both legs are priced -------------------------------------------------

def test_a_hedged_carry_pays_two_round_trips(tmp_path):
    """Pricing one leg is the flattering error and it is roughly a factor of
    two: the trade is short the perp and long the spot, and it pays to enter and
    exit both."""
    rows = _funding_rows("BTCUSDT", "binance", 60, "0.0012")
    selection = _select(tmp_path, rows, _universe("BTCUSDT"))

    assert selection.proposals, selection.declined
    proposal = selection.proposals[0]
    assert proposal.perp_cost_bps > 0 and proposal.spot_cost_bps > 0
    assert proposal.total_cost_bps == (proposal.perp_cost_bps
                                       + proposal.spot_cost_bps)
    assert proposal.net_carry_bps == (proposal.expected_carry_bps
                                      - proposal.total_cost_bps)


def test_a_carry_below_the_two_leg_cost_is_declined(tmp_path):
    """It would clear a one-leg gate. That is the whole point of pricing both."""
    # 0.00001 -> 0.1 bps a settlement -> ~110 bps a year, under a 24 bps
    # round trip only once both legs are charged... so make it smaller still.
    rows = _funding_rows("BTCUSDT", "binance", 60, "0.0000001")
    selection = _select(tmp_path, rows, _universe("BTCUSDT"))

    assert selection.proposals == []
    assert selection.declined["below_cost_gate"] == 1


def test_an_unknown_perp_venue_is_declined_rather_than_paired_by_guess(tmp_path):
    """A hedge on the wrong exchange is not a hedge, and it would look like an
    ordinary position until the two legs moved apart."""
    rows = _funding_rows("BTCUSDT", "hyperliquid", 60, "0.0012")
    selection = _select(tmp_path, rows, _universe("BTCUSDT"))

    assert selection.declined["unknown_perp_venue"] == 1


def test_the_fee_identity_is_declared_separately_from_the_feed(tmp_path):
    """The vocabulary collision that made every spot quote refuse: the store's
    venue is a FEED (`binance-spot`) and the fee table's is an EXCHANGE
    (`("binance", "spot")`). 868 of 868 candidates declined `cost_refused` until
    the pairing declared both."""
    spot_feed, perp_fee, spot_fee = HEDGE_VENUES["binance"]

    assert spot_feed == "binance-spot"
    assert perp_fee == ("binance", "perp")
    assert spot_fee == ("binance", "spot")


# --- the threshold rises with flow ---------------------------------------

def test_flow_below_capacity_leaves_the_cost_gate_as_the_only_threshold():
    """The correct answer on a quiet day, and it is reported as None rather than
    as a threshold of zero."""
    assert flow_threshold([Decimal(5), Decimal(3)], capacity=10) is None


def test_flow_above_capacity_raises_the_bar_to_the_capacity_th_best():
    carries = [Decimal(n) for n in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)]

    assert flow_threshold(carries, capacity=3) == Decimal(8)


def test_a_book_that_can_hold_nothing_is_refused():
    """It declines everything, and calling that a threshold hides it."""
    with pytest.raises(ValueError, match="declines everything"):
        flow_threshold([Decimal(1)], capacity=0)


def test_the_same_carry_is_taken_when_quiet_and_declined_when_busy(tmp_path):
    """§5a.5's rule, end to end: a fixed threshold takes the same trade whether
    it is one of three opportunities or one of three hundred."""
    modest = _funding_rows("MIDUSDT", "binance", 60, "0.0004")
    quiet = _select(tmp_path / "quiet", modest, _universe("MIDUSDT"),
                    capacity=2)

    busy_rows = list(modest)
    for i in range(6):
        busy_rows += _funding_rows(f"HOT{i}USDT", "binance", 60, "0.0020")
    busy = _select(tmp_path / "busy", busy_rows,
                   _universe("MIDUSDT", *[f"HOT{i}USDT" for i in range(6)]),
                   capacity=2)

    assert [p.symbol for p in quiet.proposals] == ["MIDUSDT"]
    assert "MIDUSDT" not in {p.symbol for p in busy.proposals}
    assert busy.declined["below_flow_threshold"] > 0


def test_ties_at_the_threshold_are_reported_rather_than_broken(tmp_path):
    """Choosing between identical opportunities would invent a preference in a
    module whose contract is that it proposes and the arbiter selects. On the
    first live run 48 symbols tied at binance's 1 bp default rate."""
    rows = []
    for i in range(6):
        rows += _funding_rows(f"SAME{i}USDT", "binance", 60, "0.0010")
    selection = _select(tmp_path, rows,
                        _universe(*[f"SAME{i}USDT" for i in range(6)]),
                        capacity=2)

    assert len(selection.proposals) == 6
    assert selection.over_capacity == 4
    assert "TIE at the threshold" in selection.describe()


# --- one hypothesis, not N ------------------------------------------------

def test_no_per_symbol_parameter_exists():
    """§5a.5: "one setup on 1,290 symbols is one hypothesis with 1,290 samples.
    It becomes 1,290 hypotheses the moment per-symbol tuning is allowed."

    Pinned by checking that no module-level mapping is keyed by symbol. The
    declared maps here are keyed by VENUE, which is a fact about an exchange
    rather than a parameter fitted to an instrument.
    """
    import strategy.funding_carry as module

    for name, value in vars(module).items():
        if name.isupper() and isinstance(value, dict):
            assert all(key in ("binance", "binance-spot", "hyperliquid", "bybit")
                       or isinstance(key, tuple) for key in value), (
                f"{name} is keyed by something that is not a venue")


def test_the_carry_is_labelled_a_forecast_not_a_measurement(tmp_path):
    """The expectation inside it is "the next settlement's rate equals the last
    one", which is a random walk on funding. Funding is persistent, so it is
    usually nearly right - which is the property that stops anyone checking."""
    rows = _funding_rows("BTCUSDT", "binance", 60, "0.0012")
    selection = _select(tmp_path, rows, _universe("BTCUSDT"))
    proposal = selection.proposals[0]

    assert "forecast carry" in proposal.describe()
    assert proposal.rate_observed_at_ns > 0, (
        "the age of the assumption must be visible, not implied")


# --- counting and the empty case -----------------------------------------

def test_a_selection_pass_is_one_trial(tmp_path):
    """§5a.5: every scan counts. A pass is a look at the data whatever it
    proposes."""
    registry = _registry(tmp_path)
    rows = _funding_rows("BTCUSDT", "binance", 60, "0.0012")
    select(_store(tmp_path, rows), _START + 200 * _MINUTE, capacity=10,
           registry=registry, trial_name="counted",
           dollar_quoted=_universe("BTCUSDT"))

    assert registry.cumulative_count() == 1
    assert registry.trials()[0]["family"] == "carry"
    assert registry.trials()[0]["result"]["sharpe"] is None


def test_the_declines_separate_a_quiet_market_from_a_broken_feed(tmp_path):
    """An empty proposal list is the same empty list either way."""
    # Both rates carry seven decimal places on purpose: pyarrow infers decimal
    # precision per part, and per-symbol parts with different precisions refuse
    # to merge on read - the same fixture rule `test_spot_perp_basis` records.
    rows = (_funding_rows("NOSPOTUSDT", "binance", 60, "0.0012000")
            + _funding_rows("TINYUSDT", "binance", 60, "0.0000001"))
    selection = _select(tmp_path, rows,
                        _universe("NOSPOTUSDT", "TINYUSDT", spot=("TINYUSDT",)))

    assert selection.proposals == []
    assert selection.declined["no_spot_leg"] == 1
    assert selection.declined["below_cost_gate"] == 1
    assert "largest loss" in selection.describe()


def test_an_empty_store_stands_aside(tmp_path):
    (tmp_path / "funding").mkdir(parents=True)
    selection = select(tmp_path, _START, capacity=10,
                       registry=_registry(tmp_path), trial_name="empty",
                       dollar_quoted=_universe("BTCUSDT"))

    assert isinstance(selection, CarrySelection)
    assert selection.proposals == []
    assert selection.candidates == 0


def test_the_order_type_is_maker_on_both_legs():
    """Carry is latency-immune by construction - there is no race to be in it -
    so resting is the correct assumption. Taking would price a trade nobody has
    to do in a hurry as though they did."""
    assert DEFAULT_ORDER_TYPE == "maker"
