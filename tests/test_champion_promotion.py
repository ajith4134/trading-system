"""CG-01: a worse model does not get to serve — RL-044.

The case these defend is real and dated. On 2026-08-19 perp's champion moved from
`976ee2399a5c` to `9fa37695dae0`: accuracy 0.54856 to 0.54862, base rate 0.53161
to 0.53635, so the edge fell from 0.01695 to 0.01227 on 42% fewer out-of-fold
rows. The outgoing model had opened 3,352 positions; the incoming one selected
nothing. Nothing compared them because nothing was asked to.
"""

import pytest

from validation.champion_promotion import (
    ChampionVerdict,
    evaluate_champion,
    evaluate_tail_champion,
    out_of_fold_edge,
    tail_pinball_loss,
)

# The two real perp models, verbatim from the registry.
REPLACED = {"accuracy": 0.5485574892686278, "base_rate": 0.5316079093837262,
            "beats_majority_class": True, "n_out_of_fold": 158647,
            "effective_sample_fraction": 0.0022281545473968553}
REPLACED_BY = {"accuracy": 0.5486204795683015, "base_rate": 0.5363484050654945,
               "beats_majority_class": True, "n_out_of_fold": 91916,
               "effective_sample_fraction": 0.0038335272137672376}


# ------------------------------------------------------------------- the edge


def test_edge_is_accuracy_above_the_base_rate():
    assert out_of_fold_edge(REPLACED) == pytest.approx(0.016949, abs=1e-6)
    assert out_of_fold_edge(REPLACED_BY) == pytest.approx(0.012272, abs=1e-6)


def test_an_unmeasured_model_has_no_edge_rather_than_an_edge_of_zero():
    # Zero would let an unmeasured challenger tie a measured incumbent and lose
    # only on the tie-break, when the honest answer is that it cannot be compared.
    assert out_of_fold_edge({}) is None
    assert out_of_fold_edge({"accuracy": 0.55}) is None
    assert out_of_fold_edge(None) is None


# -------------------------------------------------- the regression that happened


def test_the_real_perp_swap_would_have_been_refused():
    """**The whole reason this module exists.**

    Identical accuracy to four decimal places, a higher base rate, and therefore
    less skill. It took over the bot anyway.
    """
    verdict = evaluate_champion(challenger_metrics=REPLACED_BY,
                                incumbent_metrics=REPLACED,
                                incumbent_version_id="976ee2399a5cd426")

    assert not verdict.promote
    assert verdict.challenger_edge < verdict.incumbent_edge
    assert any("no_regression" == c.name and not c.passed for c in verdict.checks)


def test_the_refusal_names_both_models_and_the_size_of_the_regression():
    # A refusal without its numbers sends the reader to guess which dial to turn.
    verdict = evaluate_champion(challenger_metrics=REPLACED_BY,
                                incumbent_metrics=REPLACED,
                                incumbent_version_id="976ee2399a5cd426")

    assert "976ee2399a5cd426" in verdict.describe()
    assert "-0.004" in verdict.describe()


def test_the_reverse_swap_would_have_been_allowed():
    # The same two models the other way round: more skill, so it serves.
    verdict = evaluate_champion(challenger_metrics=REPLACED,
                                incumbent_metrics=REPLACED_BY,
                                incumbent_version_id="9fa37695dae0d091")

    assert verdict.promote
    assert "promoted" in verdict.describe()


# ------------------------------------------------------------------- the rules


def test_a_tie_goes_to_the_model_already_serving():
    # A swap costs a warm-up the incumbent has already paid - measured today at
    # 60 sealed one-minute bars, roughly an hour of not trading.
    verdict = evaluate_champion(challenger_metrics=dict(REPLACED),
                                incumbent_metrics=dict(REPLACED),
                                incumbent_version_id="976ee2399a5cd426")

    assert not verdict.promote


def test_the_first_champion_promotes_on_its_own_merits():
    # Refusing it would leave the bot with no model at all, which is worse than
    # an unproven one.
    verdict = evaluate_champion(challenger_metrics=REPLACED_BY)

    assert verdict.promote
    assert "no incumbent" in verdict.describe()


def test_a_challenger_that_cannot_be_measured_is_refused():
    verdict = evaluate_champion(challenger_metrics={"accuracy": 0.6},
                                incumbent_metrics=REPLACED,
                                incumbent_version_id="976ee2399a5cd426")

    assert not verdict.promote
    assert any(c.name == "challenger_measured" and not c.passed
               for c in verdict.checks)


def test_a_challenger_that_does_not_beat_the_majority_class_is_refused():
    verdict = evaluate_champion(
        challenger_metrics={**REPLACED_BY, "beats_majority_class": False},
        incumbent_metrics=REPLACED, incumbent_version_id="976ee2399a5cd426")

    assert not verdict.promote


def test_replacing_an_unmeasurable_incumbent_says_so_rather_than_claiming_a_win():
    # "We replaced a model we could not measure" must never read later as
    # "we beat it".
    verdict = evaluate_champion(challenger_metrics=REPLACED_BY,
                                incumbent_metrics={"accuracy": 0.55},
                                incumbent_version_id="unmeasured")

    assert verdict.promote
    detail = " ".join(c.detail for c in verdict.checks)
    assert "replacing an unmeasured model" in detail


def test_a_higher_bar_can_be_demanded_without_changing_the_code():
    # Equal-and-a-bit is still a regression if the bar is set above it.
    better = {**REPLACED, "accuracy": REPLACED["accuracy"] + 0.0001}

    lenient = evaluate_champion(challenger_metrics=better,
                                incumbent_metrics=REPLACED,
                                incumbent_version_id="v")
    strict = evaluate_champion(challenger_metrics=better,
                               incumbent_metrics=REPLACED,
                               incumbent_version_id="v",
                               min_edge_improvement=0.005)

    assert lenient.promote
    assert not strict.promote


# --------------------------------------------------- what is measured, not gated


def test_effective_sample_is_reported_and_never_decides():
    """A floor here would have blocked the BETTER model.

    0.0022 for the one that was replaced against 0.0038 for the one that replaced
    it. A threshold with the power to stop all promotion is a human's to set.
    """
    verdict = evaluate_champion(challenger_metrics=REPLACED,
                                incumbent_metrics=REPLACED_BY,
                                incumbent_version_id="9fa37695dae0d091")

    reported = [c for c in verdict.checks if c.name == "effective_sample_reported"]
    assert reported, "the effective sample was not reported at all"
    assert reported[0].passed, "it must never fail a promotion on its own"
    assert reported[0].measured == pytest.approx(0.0022281, abs=1e-6)
    assert reported[0].threshold is None
    assert verdict.promote, "the lower effective sample did not block the better model"


def test_every_check_carries_its_measured_value_whether_it_passed_or_not():
    verdict = evaluate_champion(challenger_metrics=REPLACED_BY,
                                incumbent_metrics=REPLACED,
                                incumbent_version_id="976ee2399a5cd426")

    assert isinstance(verdict, ChampionVerdict)
    payload = verdict.as_dict()
    assert payload["challenger_edge"] is not None
    assert payload["incumbent_edge"] is not None
    assert all("measured" in c for c in payload["checks"])


# --------------------------------------------- the tail model, scored by LOSS

# The two real perp profit-tail models, verbatim from the alias history.
TAIL_REPLACED = {"pinball_loss": {"0.1": 0.00069, "0.5": 0.00160, "0.9": 0.00075}}
TAIL_REPLACED_BY = {"pinball_loss": {"0.1": 0.0006909218647235398,
                                     "0.5": 0.0016751511147484368,
                                     "0.9": 0.0007520874190338046}}


def test_the_tail_model_is_judged_on_loss_because_it_has_no_accuracy():
    # It forecasts quantiles of forward P&L. Running it through the accuracy gate
    # would refuse every promotion and freeze the tail brains on whatever they
    # happened to be running.
    assert tail_pinball_loss(TAIL_REPLACED) == pytest.approx(0.00160)
    assert out_of_fold_edge(TAIL_REPLACED) is None


def test_a_tail_model_with_higher_loss_does_not_take_the_alias():
    verdict = evaluate_tail_champion(challenger_metrics=TAIL_REPLACED_BY,
                                     incumbent_metrics=TAIL_REPLACED,
                                     incumbent_version_id="c744358979f633d4")

    assert not verdict.promote


def test_a_tail_model_with_lower_loss_serves():
    verdict = evaluate_tail_champion(challenger_metrics=TAIL_REPLACED,
                                     incumbent_metrics=TAIL_REPLACED_BY,
                                     incumbent_version_id="d658b08dea4331f9")

    assert verdict.promote


def test_the_tail_verdict_negates_loss_so_larger_always_means_better():
    # A field meaning "higher is worse" on one gate and "higher is better" on the
    # other is the sign error waiting to happen in whatever reads them next.
    verdict = evaluate_tail_champion(challenger_metrics=TAIL_REPLACED,
                                     incumbent_metrics=TAIL_REPLACED_BY,
                                     incumbent_version_id="d658b08dea4331f9")

    assert verdict.challenger_edge < 0
    assert verdict.challenger_edge > verdict.incumbent_edge


def test_a_tail_model_with_no_loss_recorded_is_refused():
    verdict = evaluate_tail_champion(challenger_metrics={},
                                     incumbent_metrics=TAIL_REPLACED,
                                     incumbent_version_id="c744358979f633d4")

    assert not verdict.promote


def test_the_first_tail_champion_promotes_on_its_own_merits():
    assert evaluate_tail_champion(challenger_metrics=TAIL_REPLACED).promote
