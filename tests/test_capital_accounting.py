"""A P&L with no capital beside it is not a result (BF-11, RL-028, RL-029).

The user asked for the USDT profit or loss against the capital each bot used,
and when asked how capital should be counted chose all three denominators side
by side. These tests pin the two things that would otherwise go wrong quietly:

  A return computed against the wrong denominator. Peak-at-risk, turnover and
  bankroll differ by orders of magnitude on the same trades, and whichever is
  largest makes the return look smallest. Each is computed separately here and
  none of them is allowed to stand in for another.

  A unit error. Deribit quotes options in BTC. A BTC premium summed into a USDT
  total is wrong by five orders of magnitude and looks like a spectacular result,
  which is the shape of every number this project has had to throw away.
"""
from decimal import Decimal

from segment.capital_accounting import (
    DOLLAR_QUOTES, UNCONVERTIBLE, account_for_fills, quote_currency, usd_rate_of,
)


def _open(symbol="BTCUSDT", venue="binance-futures", qty="1", price="100",
          at_ns=1, **extra):
    row = {"at_ns": at_ns, "event": "OPEN", "venue": venue, "symbol": symbol,
           "quantity": qty, "price": price}
    row.update(extra)
    return row


def _close(symbol="BTCUSDT", venue="binance-futures", qty="1", price="110",
           pnl="10", at_ns=2, **extra):
    row = {"at_ns": at_ns, "event": "CLOSE", "venue": venue, "symbol": symbol,
           "quantity": qty, "price": price, "gross_pnl": pnl}
    row.update(extra)
    return row


def _report(fills, bankroll="1000"):
    return account_for_fills(fills, segment="perp",
                             bankroll_usdt=Decimal(bankroll))


# --- what currency is this priced in --------------------------------------

def test_a_usdt_pair_is_a_dollar_quote():
    assert quote_currency("binance-futures", "BTCUSDT") == "USDT"


def test_a_deribit_option_is_quoted_in_its_underlying_not_in_dollars():
    """`BTC-26MAR27-68000-C` at 0.1135 is 0.1135 BTC, not eleven cents."""
    assert quote_currency("deribit", "BTC-26MAR27-68000-C") == "BTC"
    assert quote_currency("deribit", "ETH-26MAR27-3500-C") == "ETH"


def test_a_spot_pair_quoted_in_btc_is_not_treated_as_dollars():
    assert quote_currency("binance-spot", "ETHBTC") == "BTC"


def test_an_unrecognised_quote_asset_is_unconvertible_rather_than_assumed():
    """Treating an unknown quote as a dollar is the silent unit error."""
    assert quote_currency("binance-spot", "FOOWEIRD") == UNCONVERTIBLE


def test_a_dollar_quote_needs_no_journalled_rate():
    assert usd_rate_of(_open()) == Decimal(1)


def test_a_crypto_quote_without_a_journalled_rate_has_no_rate():
    assert usd_rate_of(_open(venue="deribit", symbol="BTC-26MAR27-68000-C")) is None


def test_a_crypto_quote_uses_the_rate_journalled_on_the_fill():
    fill = _open(venue="deribit", symbol="BTC-26MAR27-68000-C",
                 usd_rate="61240")
    assert usd_rate_of(fill) == Decimal("61240")


# --- the three denominators, each measured separately ----------------------

def test_peak_at_risk_is_the_most_open_at_one_moment_not_the_sum_of_trades():
    """Two positions open together are 200; opened and closed in turn they are 100.
    No per-trade total can recover that, which is why the walk is in time order."""
    together = _report([_open(symbol="AUSDT", at_ns=1),
                        _open(symbol="BUSDT", at_ns=2),
                        _close(symbol="AUSDT", at_ns=3),
                        _close(symbol="BUSDT", at_ns=4)])
    in_turn = _report([_open(symbol="AUSDT", at_ns=1),
                       _close(symbol="AUSDT", at_ns=2),
                       _open(symbol="BUSDT", at_ns=3),
                       _close(symbol="BUSDT", at_ns=4)])

    assert together.peak_at_risk_usdt == Decimal(200)
    assert in_turn.peak_at_risk_usdt == Decimal(100)
    assert together.turnover_usdt == in_turn.turnover_usdt == Decimal(200)


def test_turnover_counts_the_same_dollar_every_round_trip():
    fills = []
    for i in range(5):
        fills += [_open(at_ns=2 * i + 1), _close(at_ns=2 * i + 2)]
    report = _report(fills)

    assert report.turnover_usdt == Decimal(500)
    assert report.peak_at_risk_usdt == Decimal(100)


def test_equity_is_the_declared_bankroll_plus_what_was_realised():
    report = _report([_open(), _close(pnl="25")], bankroll="1000")

    assert report.realised_pnl_usdt == Decimal(25)
    assert report.equity_usdt == Decimal(1025)
    assert report.return_on_bankroll_pct == 2.5


def test_the_three_returns_are_different_numbers_on_the_same_trades():
    """The point of RL-028: whichever denominator is largest makes the return
    look smallest, so publishing one alone is publishing whichever flatters."""
    fills = []
    for i in range(5):
        fills += [_open(at_ns=2 * i + 1), _close(at_ns=2 * i + 2, pnl="10")]
    report = _report(fills, bankroll="1000")

    assert report.return_on_peak_pct == 50.0        # 50 on 100 ever at risk
    assert report.return_on_turnover_pct == 10.0    # 50 on 500 turned over
    assert report.return_on_bankroll_pct == 5.0     # 50 on a 1000 allocation


# --- absence is not zero ---------------------------------------------------

def test_a_bot_that_has_closed_nothing_reports_no_return_rather_than_zero():
    report = _report([_open()])

    assert report.closes == 0
    assert report.return_on_peak_pct is None
    assert report.return_on_bankroll_pct is None
    assert report.peak_at_risk_usdt == Decimal(100), "it still put capital at risk"


def test_a_winrate_under_twenty_closes_is_not_reported():
    fills = []
    for i in range(10):
        fills += [_open(at_ns=2 * i + 1), _close(at_ns=2 * i + 2)]

    assert _report(fills).winrate_pct is None


# --- units ------------------------------------------------------------------

def test_a_btc_priced_option_is_converted_at_the_rate_journalled_with_it():
    fills = [_open(venue="deribit", symbol="BTC-26MAR27-68000-C", qty="0.1",
                   price="0.1135", usd_rate="61240", at_ns=1),
             _close(venue="deribit", symbol="BTC-26MAR27-68000-C", qty="0.1",
                    price="0.1200", pnl="0.00065", usd_rate="61240", at_ns=2)]
    report = _report(fills)

    assert report.peak_at_risk_usdt == Decimal("0.1") * Decimal("0.1135") * 61240
    assert report.realised_pnl_usdt == Decimal("0.00065") * 61240
    assert report.unconvertible_fills == 0


def test_a_fill_with_no_rate_is_reported_unconvertible_never_converted_or_dropped():
    """RL-029. Dropping it silently would make P&L smaller and the board tidier,
    which is the wrong direction to be wrong in."""
    fills = [_open(venue="deribit", symbol="BTC-26MAR27-68000-C", qty="0.1",
                   price="0.1135", at_ns=1),
             _open(at_ns=2)]
    report = _report(fills)

    assert report.unconvertible_fills == 1
    assert report.unconvertible_currencies == ("BTC",)
    assert report.converted_fills == 1
    assert report.peak_at_risk_usdt == Decimal(100), "only the dollar fill counted"


def test_a_btc_premium_is_never_summed_into_usdt_unconverted():
    """The unit error this refuses: 0.1135 BTC is 6,950 USDT, not 0.1135."""
    unrated = _report([_open(venue="deribit", symbol="BTC-26MAR27-68000-C",
                             qty="1", price="0.1135")])

    assert unrated.peak_at_risk_usdt == Decimal(0)
    assert unrated.unconvertible_fills == 1


# --- journals roll, positions do not ---------------------------------------

def test_a_close_whose_open_is_outside_the_window_still_counts_its_pnl():
    """Journals roll daily and a position held across midnight would otherwise
    vanish from the record that matters most."""
    report = _report([_close(pnl="7")])

    assert report.realised_pnl_usdt == Decimal(7)
    assert report.closes == 1
    assert report.peak_at_risk_usdt == Decimal(0)


def test_open_now_is_what_is_still_open_after_the_walk():
    report = _report([_open(symbol="AUSDT", at_ns=1),
                      _open(symbol="BUSDT", at_ns=2),
                      _close(symbol="AUSDT", at_ns=3)])

    assert report.open_now_usdt == Decimal(100)
    assert report.peak_at_risk_usdt == Decimal(200)


def test_the_basis_says_the_figures_are_gross_of_fees():
    """A gross P&L presented as net is wrong in the flattering direction on every
    single trade."""
    assert "gross" in _report([_open(), _close()]).basis


def test_a_net_pnl_is_preferred_over_gross_when_the_engine_journals_one():
    report = _report([_open(), _close(pnl="10", net_pnl="8")])

    assert report.realised_pnl_usdt == Decimal(8)


def test_fills_are_accounted_in_time_order_whatever_order_they_arrive_in():
    scrambled = [_close(symbol="AUSDT", at_ns=3), _open(symbol="BUSDT", at_ns=2),
                 _open(symbol="AUSDT", at_ns=1)]

    assert _report(scrambled).peak_at_risk_usdt == Decimal(200)


def test_every_dollar_quote_is_a_recognised_currency():
    for quote in DOLLAR_QUOTES:
        assert usd_rate_of({"quote_currency": quote}) == Decimal(1)


# --- venue naming, measured against symbols the bots actually traded -------

def test_a_dated_contract_reads_the_currency_before_its_expiry():
    """Measured 2026-08-19: all 193 dated fills read UNCONVERTIBLE because the
    expiry suffix was being taken for the currency."""
    assert quote_currency("bybit", "BTCUSDT-28AUG26") == "USDT"
    assert quote_currency("bybit", "BTCUSDT_261225") == "USDT"


def test_an_inverse_contract_is_unconvertible_rather_than_multiplied_linearly():
    """`BTCUSDU26` is quoted in USD and settled in BTC: its P&L is not linear in
    the price, so reading USD off the end would be arithmetic that does not apply
    to the instrument."""
    assert quote_currency("bybit", "BTCUSDU26") == UNCONVERTIBLE
    assert quote_currency("bybit", "BTCUSDZ26") == UNCONVERTIBLE


def test_a_lira_quoted_spot_pair_is_unconvertible_not_read_as_a_dollar():
    assert quote_currency("binance-spot", "ACETRY") == UNCONVERTIBLE


# --- the engine writes the rate, and that is what makes the sum auditable ---

def test_the_engine_journals_the_underlying_price_as_the_rate_for_an_option():
    """RL-029: the rate comes from the frame the fill was decided on, so it is the
    price the bot actually saw rather than one looked up afterwards."""
    from segment.live_engine import LiveSegmentEngine

    conversion = LiveSegmentEngine._usd_conversion(
        object(), {"venue_underlying_price": 61240.5},
        "BTC-26MAR27-68000-C", "deribit")

    assert conversion["quote_currency"] == "BTC"
    assert conversion["usd_rate"] == "61240.5"
    assert "underlying" in conversion["rate_source"]


def test_a_dollar_quoted_fill_journals_a_rate_of_one_and_says_why():
    from segment.live_engine import LiveSegmentEngine

    conversion = LiveSegmentEngine._usd_conversion(
        object(), {}, "BTCUSDT", "binance-futures")

    assert conversion["usd_rate"] == "1"
    assert conversion["quote_currency"] == "USDT"


def test_an_option_with_no_underlying_on_the_frame_journals_no_rate():
    """Never converted at a rate nobody recorded."""
    from segment.live_engine import LiveSegmentEngine

    conversion = LiveSegmentEngine._usd_conversion(
        object(), {}, "BTC-26MAR27-68000-C", "deribit")

    assert conversion["usd_rate"] is None
    assert "no rate available" in conversion["rate_source"]
