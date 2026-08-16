"""An empty closed-trades table looks exactly like a quiet day.

It is not one. On 2026-08-16 the journal held 12,576 fills and every single one
was a BUY, so there were zero closed round trips - not because nothing closed,
but because a round trip was impossible. `test_a_journal_of_only_buys_reports_
that_a_round_trip_is_impossible` is the test that keeps those two apart, and it
is the reason this module exists rather than a `SELECT * FROM fills`.
"""
import json
from decimal import Decimal

import pytest

from paper.blotter import (
    BUY,
    OPTIMISTIC,
    PESSIMISTIC,
    SELL,
    build_blotter,
    read_blotter,
    read_fills,
)

_START = 1_700_000_000_000_000_000
_SECOND = 1_000_000_000


def _fill_row(symbol, side, quantity, optimistic, pessimistic, at_ns,
              venue="binance", strategy="test", edge_claim=False,
              uncalibrated=False):
    return {
        "at_ns": at_ns, "strategy": strategy, "makes_edge_claim": edge_claim,
        "client_order_id": f"c{at_ns}{symbol}{side}", "symbol": symbol,
        "venue": venue, "side": side, "quantity": str(quantity),
        "optimistic_price": str(optimistic), "optimistic_liquidity": "maker",
        "pessimistic_price": str(pessimistic), "pessimistic_liquidity": "taker",
        "participation": "0.1", "uncalibrated": uncalibrated,
    }


def _write(capture_root, rows, day="2026-08-16"):
    forward = capture_root / "paper" / "forward"
    forward.mkdir(parents=True, exist_ok=True)
    with open(forward / f"fills-{day}.ndjson", "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return capture_root


def _fills(capture_root):
    return read_fills(capture_root / "paper" / "forward")


# --- the distinction the module exists for -------------------------------

def test_a_journal_of_only_buys_reports_that_a_round_trip_is_impossible(tmp_path):
    """The live case: 12,576 fills, every one a BUY, zero closed trades. An
    empty table would read as 'nothing closed today'; the truth is that nothing
    CAN close until the strategy sells."""
    rows = [_fill_row("BTCUSDT", BUY, "0.001", "60000", "60001",
                      _START + i * _SECOND) for i in range(5)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert view.closed_trades == []
    assert not view.closed_trades_possible
    assert "impossible rather than absent" in view.describe()


def test_a_journal_with_both_sides_can_produce_a_round_trip(tmp_path):
    """The other half - a verdict that fires on everything says nothing."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "101", _START),
            _fill_row("BTCUSDT", SELL, "1", "110", "109", _START + _SECOND)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert view.closed_trades_possible
    assert len(view.closed_trades) == 1


# --- both accountings -----------------------------------------------------

def test_every_closed_trade_is_priced_both_ways(tmp_path):
    """A blotter reporting one figure is choosing which, and the one that gets
    chosen is the flattering one."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "101", _START),
            _fill_row("BTCUSDT", SELL, "1", "110", "109", _START + _SECOND)]
    trade = build_blotter(_fills(_write(tmp_path, rows))).closed_trades[0]

    # Optimistic: bought at 100, sold at 110. Pessimistic: bought at 101, sold
    # at 109 - the same trade, crossing the spread at both ends.
    assert trade.pnl(OPTIMISTIC) == Decimal(10)
    assert trade.pnl(PESSIMISTIC) == Decimal(8)
    assert trade.pnl(OPTIMISTIC) > trade.pnl(PESSIMISTIC)


def test_the_gap_between_the_accountings_is_reported(tmp_path):
    """On a strategy whose participation is uncalibrated it is the first number
    worth reading - how much of the result is an execution assumption."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "101", _START),
            _fill_row("BTCUSDT", SELL, "1", "110", "109", _START + _SECOND)]
    trade = build_blotter(_fills(_write(tmp_path, rows))).closed_trades[0]

    # 2 of P&L on a 100 notional = 200 bps.
    assert trade.spread_bps == pytest.approx(Decimal(200))


def test_a_short_round_trip_makes_money_when_price_falls(tmp_path):
    """The sign, which is silent when it is wrong."""
    rows = [_fill_row("BTCUSDT", SELL, "1", "110", "109", _START),
            _fill_row("BTCUSDT", BUY, "1", "100", "101", _START + _SECOND)]
    trade = build_blotter(_fills(_write(tmp_path, rows))).closed_trades[0]

    assert trade.side == SELL
    assert trade.pnl(OPTIMISTIC) == Decimal(10)


# --- FIFO -----------------------------------------------------------------

def test_lots_are_closed_first_in_first_out(tmp_path):
    """FIFO is a convention rather than a fact, and it changes the reported P&L
    of every partially-closed position - LIFO on a rising market books different
    trades. Declared, and tested."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START),
            _fill_row("BTCUSDT", BUY, "1", "200", "200", _START + _SECOND),
            _fill_row("BTCUSDT", SELL, "1", "300", "300", _START + 2 * _SECOND)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert len(view.closed_trades) == 1
    assert view.closed_trades[0].entry_optimistic == Decimal(100), (
        "the FIRST lot closed, not the most recent")
    assert view.open_positions[0].average_optimistic == Decimal(200)


def test_a_partial_close_leaves_the_remainder_open(tmp_path):
    rows = [_fill_row("BTCUSDT", BUY, "3", "100", "100", _START),
            _fill_row("BTCUSDT", SELL, "1", "110", "110", _START + _SECOND)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert view.closed_trades[0].quantity == Decimal(1)
    assert view.open_positions[0].quantity == Decimal(2)


def test_an_exit_larger_than_the_position_opens_the_other_way(tmp_path):
    """A reversal is a new position, never a flipped one - the same rule the
    PROFIT-TAIL spec states about never flipping a position."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START),
            _fill_row("BTCUSDT", SELL, "3", "110", "110", _START + _SECOND)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert len(view.closed_trades) == 1
    assert view.closed_trades[0].quantity == Decimal(1)
    position = view.open_positions[0]
    assert position.side == SELL and position.quantity == Decimal(2)


def test_positions_are_kept_apart_by_venue_and_symbol(tmp_path):
    """A sell on one venue must not close a position on another."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START,
                      venue="binance"),
            _fill_row("BTCUSDT", SELL, "1", "110", "110", _START + _SECOND,
                      venue="bybit")]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert view.closed_trades == []
    assert len(view.open_positions) == 2


# --- marks ----------------------------------------------------------------

def test_an_unmarked_position_has_no_unrealised_pnl_rather_than_zero(tmp_path):
    """Zero reads as flat and the truth is unknown - the distinction
    `features.staleness` insists on for the same reason."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START)]
    view = build_blotter(_fills(_write(tmp_path, rows)))
    position = view.open_positions[0]

    assert position.unrealised_pnl(OPTIMISTIC) is None
    assert view.unmarked_positions == 1


def test_a_marked_position_reports_both_accountings(tmp_path):
    rows = [_fill_row("BTCUSDT", BUY, "2", "100", "101", _START)]
    view = build_blotter(_fills(_write(tmp_path, rows)),
                         mark_prices={("binance", "BTCUSDT"): Decimal(110)})
    position = view.open_positions[0]

    assert position.unrealised_pnl(OPTIMISTIC) == Decimal(20)
    assert position.unrealised_pnl(PESSIMISTIC) == Decimal(18)
    assert view.unmarked_positions == 0


# --- the journal ----------------------------------------------------------

def test_a_torn_final_line_is_skipped(tmp_path):
    """The engine appends while this reads, so a partial last row is the
    ordinary case rather than corruption."""
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START)]
    capture = _write(tmp_path, rows)
    path = capture / "paper" / "forward" / "fills-2026-08-16.ndjson"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"at_ns": 1700')

    assert len(read_fills(capture / "paper" / "forward")) == 1


def test_fills_from_several_days_are_read_oldest_first(tmp_path):
    _write(tmp_path, [_fill_row("BTCUSDT", BUY, "1", "100", "100",
                                _START + 5 * _SECOND)], day="2026-08-16")
    _write(tmp_path, [_fill_row("BTCUSDT", BUY, "1", "90", "90", _START)],
           day="2026-08-15")

    fills = read_fills(tmp_path / "paper" / "forward")

    assert [f.optimistic_price for f in fills] == [Decimal(90), Decimal(100)]


def test_the_edge_claim_rides_the_view(tmp_path):
    """A blotter showing P&L for a strategy that claims no edge, without saying
    so, invites reading it as performance."""
    quiet = build_blotter(_fills(_write(
        tmp_path / "a", [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START)])))
    claiming = build_blotter(_fills(_write(
        tmp_path / "b", [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START,
                                   edge_claim=True)])))

    assert not quiet.makes_edge_claim
    assert "NO EDGE CLAIM" in quiet.describe()
    assert claiming.makes_edge_claim
    assert "NO EDGE CLAIM" not in claiming.describe()


def test_uncalibrated_fills_are_counted(tmp_path):
    rows = [_fill_row("BTCUSDT", BUY, "1", "100", "100", _START,
                      uncalibrated=True),
            _fill_row("BTCUSDT", BUY, "1", "100", "100", _START + _SECOND)]
    view = build_blotter(_fills(_write(tmp_path, rows)))

    assert view.uncalibrated_fills == 1


def test_no_journal_at_all_reports_that_nothing_traded(tmp_path):
    view = read_blotter(tmp_path)

    assert view.fills_read == 0
    assert "has not traded" in view.describe()
