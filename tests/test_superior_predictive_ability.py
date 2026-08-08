"""Hansen's SPA at the promotion gate — VX-010.

`FEATURES.md` §8: *"Does the challenger beat the **incumbent**, corrected for
variants tried."* The corpus also settles the choice of test: *"White's Reality
Check explicitly superseded - Hansen's SPA strictly dominates it; running both is
ceremony."*

What makes SPA dominate RC is **recentering**. White's Reality Check builds its null
distribution from every candidate, so a pile of hopeless variants inflates the
critical value and buries a genuinely good challenger. Hansen recenters: candidates
whose mean performance is far enough below the benchmark are held at zero in the
null, so adding junk to the search no longer protects the incumbent. That property
is tested below directly, because it is the entire reason to prefer this test.

The prior art (VX-039, `nse-crypto-bot-final/trading/strategy/generators/
stats_gate.py`) was fetched raw and is **not a donor**, for two reasons:

*It answers a different question.* Its benchmark is `pd.Series(np.zeros(T))` - a
zero-return do-nothing. That asks "did this beat cash", which nearly every
candidate passes in a rising market. VX-010 asks whether the challenger beats the
**incumbent**, which is the comparison a promotion actually makes.

*It fails open, by declared design.* `except Exception: return all_ids`, documented
as *"so it can only ever ADD strictness when statistically meaningful, never
silently block the whole portfolio."* The reasoning is coherent for a filter layered
on other gates, but the consequence is that a missing `arch` install, a NaN, or a
pandas upgrade makes every candidate pass family-wise error control silently and
permanently. `arch` is not installed in this project's environment, which is exactly
the condition under which that branch reports a full pass having tested nothing.
This is the fourth defect found in this corpus that fails in the flattering
direction.

So: benchmarked against the incumbent, and it raises rather than passing when it
cannot test.

Cross-validated against `arch.bootstrap.SPA` (Hansen's own reference
implementation) in the final test, since a bootstrap p-value has no closed form to
check against.
"""
import random

import pytest

from validation.superior_predictive_ability import (
    NotEnoughPaths,
    SpaResult,
    superior_predictive_ability,
)


def returns(n, mean_, sd=0.01, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(mean_, sd) for _ in range(n)]


# --- the basic verdict ------------------------------------------------------

def test_a_challenger_no_better_than_the_incumbent_is_not_significant():
    """Same distribution, different draws. The p-value must not reject."""
    result = superior_predictive_ability(
        incumbent=returns(500, 0.0005, seed=1),
        challengers={"a": returns(500, 0.0005, seed=2)},
        n_bootstrap=500, seed=7)
    assert result.p_value > 0.10
    assert result.superior == ()


def test_a_clearly_better_challenger_is_significant():
    result = superior_predictive_ability(
        incumbent=returns(500, 0.0000, seed=1),
        challengers={"good": returns(500, 0.0030, seed=2)},
        n_bootstrap=500, seed=7)
    assert result.p_value < 0.05
    assert "good" in result.superior


def test_a_worse_challenger_is_never_reported_superior():
    result = superior_predictive_ability(
        incumbent=returns(500, 0.0030, seed=1),
        challengers={"bad": returns(500, -0.0010, seed=2)},
        n_bootstrap=500, seed=7)
    assert result.p_value > 0.10
    assert result.superior == ()


def test_the_result_carries_its_statistic_and_the_count_it_corrected_for():
    """Rule 8 on a promotion decision. "Not significant" with no visible N is
    indistinguishable from "we did not correct for anything"."""
    result = superior_predictive_ability(
        incumbent=returns(300, 0.0, seed=1),
        challengers={f"c{i}": returns(300, 0.0, seed=i + 10) for i in range(5)},
        n_bootstrap=300, seed=7)
    assert isinstance(result, SpaResult)
    assert result.n_challengers == 5
    assert result.n_bootstrap == 300
    assert result.test_statistic >= 0.0
    assert result.detail and "5" in result.detail


# --- correcting for how many were tried -------------------------------------

def test_adding_more_challengers_makes_a_marginal_winner_harder_to_promote():
    """The correction actually correcting. A challenger that squeaks through on its
    own must face a higher bar once it is one of fifty things tried."""
    marginal = returns(400, 0.0012, seed=99)
    alone = superior_predictive_ability(
        incumbent=returns(400, 0.0, seed=1), challengers={"m": marginal},
        n_bootstrap=500, seed=7)
    crowded = superior_predictive_ability(
        incumbent=returns(400, 0.0, seed=1),
        challengers={"m": marginal,
                     **{f"j{i}": returns(400, 0.0, seed=200 + i) for i in range(49)}},
        n_bootstrap=500, seed=7)
    assert crowded.p_value >= alone.p_value, (
        "trying 50 variants did not raise the bar relative to trying one")


def test_recentering_stops_hopeless_variants_from_shielding_the_incumbent():
    """**Why SPA rather than White's Reality Check.** RC builds its null from every
    candidate, so a pile of terrible variants inflates the critical value and buries
    a real winner. Hansen holds far-below-benchmark candidates at zero in the null,
    so adding junk cannot rescue the incumbent.

    Tested by comparing SPA's own consistent p-value against its lower/RC-style
    variant on the same data: with many hopeless candidates present, the RC-style
    p-value is the more conservative one."""
    good = returns(400, 0.0035, seed=99)
    junk = {f"j{i}": returns(400, -0.0080, seed=300 + i) for i in range(40)}
    result = superior_predictive_ability(
        incumbent=returns(400, 0.0, seed=1),
        challengers={"good": good, **junk},
        n_bootstrap=500, seed=7)

    assert result.p_value <= result.p_value_reality_check, (
        f"SPA ({result.p_value:.4f}) was not at least as powerful as the "
        f"RC-style p-value ({result.p_value_reality_check:.4f}) despite 40 "
        f"hopeless candidates - the recentering is not doing its work")
    assert "good" in result.superior


def test_only_the_challengers_that_beat_the_incumbent_are_named():
    result = superior_predictive_ability(
        incumbent=returns(400, 0.0, seed=1),
        challengers={"good": returns(400, 0.0040, seed=2),
                     "bad": returns(400, -0.0040, seed=3)},
        n_bootstrap=500, seed=7)
    assert "good" in result.superior and "bad" not in result.superior


# --- it refuses instead of failing open -------------------------------------

def test_too_few_paths_is_refused_not_passed(tmp_path):
    """The donor's fail-open branch, inverted. A gate that cannot test must not
    report a pass - "we could not check" and "it passed" are different facts, and
    only one of them should let capital move."""
    with pytest.raises(NotEnoughPaths):
        superior_predictive_ability(
            incumbent=returns(5, 0.0, seed=1),
            challengers={"a": returns(5, 0.0, seed=2)}, n_bootstrap=100)


def test_no_challengers_is_refused():
    with pytest.raises(ValueError):
        superior_predictive_ability(incumbent=returns(300, 0.0), challengers={},
                                    n_bootstrap=100)


def test_misaligned_series_are_refused_rather_than_truncated():
    """The donor silently truncates to the shortest series. Two candidates scored
    over different periods are not comparable, and quietly trimming hides that the
    comparison was never valid."""
    with pytest.raises(ValueError):
        superior_predictive_ability(
            incumbent=returns(300, 0.0, seed=1),
            challengers={"a": returns(250, 0.0, seed=2)}, n_bootstrap=100)


def test_a_zero_variance_performance_difference_is_refused():
    """Identical series give a zero studentising denominator. Returning a
    significant result there would promote a challenger for being indistinguishable
    from the incumbent."""
    same = returns(300, 0.001, seed=1)
    with pytest.raises(ValueError):
        superior_predictive_ability(incumbent=same, challengers={"clone": same},
                                    n_bootstrap=100)


def test_the_test_is_deterministic_for_a_given_seed():
    kw = dict(incumbent=returns(300, 0.0, seed=1),
              challengers={"a": returns(300, 0.001, seed=2)}, n_bootstrap=200)
    assert (superior_predictive_ability(**kw, seed=5).p_value
            == superior_predictive_ability(**kw, seed=5).p_value)


# --- checked against Hansen's own reference implementation ------------------

def test_agrees_with_arch_reference_spa_on_the_same_data():
    """A bootstrap p-value has no closed form, so the only real check is another
    implementation. `arch.bootstrap.SPA` is Hansen's reference.

    Skipped rather than silently passing when `arch` is absent - `arch` is a
    verification tool here, not a runtime dependency, and this project's own
    validation modules take no third-party statistics dependency.
    """
    arch_bootstrap = pytest.importorskip("arch.bootstrap")
    numpy = pytest.importorskip("numpy")

    gaps = []
    for data_seed in (11, 21, 31):
        incumbent = returns(500, 0.0005, seed=data_seed)
        challengers = {f"c{i}": returns(500, 0.0005 + 0.0004 * i,
                                       seed=40 + i + data_seed)
                       for i in range(4)}
        mine = superior_predictive_ability(incumbent, challengers,
                                          n_bootstrap=1000, seed=3)

        # arch works in losses, where lower is better; returns negate to losses.
        reference = arch_bootstrap.SPA(
            -numpy.asarray(incumbent, dtype=float),
            numpy.column_stack([-numpy.asarray(v, dtype=float)
                                for v in challengers.values()]),
            reps=1000, block_size=50, seed=3)
        reference.compute()
        theirs = float(reference.pvalues["consistent"])

        assert mine.p_value == pytest.approx(theirs, abs=0.03), (
            f"data seed {data_seed}: mine {mine.p_value:.4f} vs arch {theirs:.4f}")
        gaps.append(mine.p_value - theirs)

    # Two independent bootstraps disagree by noise, which lands on both sides. A
    # one-sided gap across every dataset is a bias, not sampling error - and that is
    # exactly how the missing √n in omega showed up: mine was systematically ~2.6x
    # the reference before the fix, because under-recentring degrades the consistent
    # SPA toward White's Reality Check.
    assert not (all(g > 0 for g in gaps) or all(g < 0 for g in gaps)), (
        f"the gap against the reference is one-sided across all datasets ({gaps}), "
        f"which indicates a systematic bias rather than bootstrap noise")
