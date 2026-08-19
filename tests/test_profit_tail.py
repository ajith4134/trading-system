"""CL-02 / RL-042: what the best and the worst of a trade are, and when.

`peak_favourable` existed on `OpenPosition` before this and was assigned NOWHERE
in the tree - it was `None` for every position that has ever existed. These tests
exist so it cannot silently go back to being a decoy.
"""

from decimal import Decimal

from segment.profit_tail import OpenPosition, sample_excursions


# ---------------------------------------------------------------- CL-02 / RL-042


def _position(**overrides):
    base = dict(venue="binance-futures", symbol="BTCUSDT", side="LONG",
                quantity=Decimal("1"), entry_price=Decimal("100"),
                entry_ns=0, hard_stop=Decimal("90"), band="fast")
    base.update(overrides)
    return OpenPosition(**base)


def test_a_position_starts_with_no_excursion_and_no_samples():
    # The state that must be distinguishable from "sampled and never moved".
    position = _position()

    assert position.peak_favourable is None
    assert position.peak_adverse is None
    assert position.excursion_samples == 0


def test_the_first_sample_sets_both_peaks():
    sampled = sample_excursions(_position(), Decimal("103"))

    assert sampled.peak_favourable == Decimal("0.03")
    assert sampled.peak_adverse == Decimal("0")
    assert sampled.excursion_samples == 1


def test_both_peaks_only_ever_advance():
    position = _position()
    for mark in ("103", "108", "94", "101", "97"):
        position = sample_excursions(position, Decimal(mark))

    # Best it reached was +8%, worst was -6%, and neither retreats when the
    # price comes back - a peak that can retreat is not a peak.
    assert position.peak_favourable == Decimal("0.08")
    assert position.peak_adverse == Decimal("0.06")
    assert position.excursion_samples == 5


def test_a_short_measures_excursion_in_its_own_direction():
    position = sample_excursions(_position(side="SHORT"), Decimal("95"))

    # Price fell, so a SHORT is up 5%. The sign convention is the position's,
    # not the market's.
    assert position.peak_favourable == Decimal("0.05")
    assert position.peak_adverse == Decimal("0")


def test_a_trade_that_never_went_green_reports_zero_not_a_negative():
    position = sample_excursions(_position(), Decimal("98"))

    # Both columns are magnitudes, so they never have to be read against each
    # other's sign on the board.
    assert position.peak_favourable == Decimal("0")
    assert position.peak_adverse == Decimal("0.02")


def test_sampling_does_not_mutate_the_position_it_was_given():
    original = _position()
    sample_excursions(original, Decimal("110"))

    # OpenPosition is frozen so a brain cannot rewrite the trade it is judging.
    assert original.peak_favourable is None
    assert original.excursion_samples == 0


def test_sampling_preserves_every_other_field():
    # The bug this defends: RATCHET_LOCK used to rebuild the position field by
    # field and dropped everything it did not name, which is how peak_favourable
    # stayed None for the life of the system.
    position = sample_excursions(
        _position(locked_stop=Decimal("99")), Decimal("105"))

    assert position.locked_stop == Decimal("99")
    assert position.hard_stop == Decimal("90")
    assert position.entry_ns == 0
    assert position.band == "fast"


def test_the_peaks_bracket_the_realised_return_of_the_closing_mark():
    # The invariant probe_excursions_journalled_on_close asserts on live data:
    # peak_favourable >= realised >= -peak_adverse. It only holds because the
    # engine samples BEFORE manage can close on that same mark.
    position = _position()
    for mark in ("104", "96", "102"):
        position = sample_excursions(position, Decimal(mark))
    realised = (Decimal("102") - position.entry_price) / position.entry_price

    assert position.peak_favourable >= realised >= -position.peak_adverse
