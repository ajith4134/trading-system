"""The Trial Registry — cumulative N, structurally enforced.

`FEATURES.md` §8 requires two things the corpus has never built together:
an experiment ledger that **includes abandoned runs** (VX-001), and a trial count
that is *"structurally impossible to evaluate without incrementing"* (VX-002).

`DECISIONS.md` §5 is why: with five years of daily data and 45+ tried variations,
the best selected strategy is more likely than not to have a true out-of-sample
Sharpe of zero. Every statistic downstream - Deflated Sharpe, MinBTL, PBO, FDR -
is a function of N. **A registry that stores only winners cannot produce an
honest N**, and an N that is too small makes every one of those gates weaker in
exactly the flattering direction.

The prior art was surveyed before writing this (ledger row VX-002 says no prior
implementation has the enforcement property, and the two that come closest were
read):

  * `nse-crypto-bot-final/trading/antioverfit.py` counts backtests via a
    `register_backtest()` the search lanes are *expected* to call. Convention,
    not structure - and its `_read_counter` returns `{"backtests_run": 0}` on any
    exception, so a corrupt counter file silently resets N to zero. N=0 means no
    deflation at all. The failure direction flatters.
  * `nse-crypto-bot-final/trading/validation_holdout.py` appends every live
    prediction *before its outcome exists*, which makes cherry-picking
    impossible. That property is worth porting here even though the module
    itself solves a different problem.

So the two rules the tests below exist to defend:

**A trial is recorded before it is evaluated.** Not after. A crash, a timeout, a
result someone dislikes and re-runs - all of them already counted. This is the
only way "including abandoned runs" can be true, because an abandoned run is
precisely the one that never reaches the code that would have counted it.

**Fail closed on a damaged ledger.** If N cannot be established, the registry
raises rather than reporting a small number. Refusing to evaluate is recoverable;
promoting a strategy against N=0 is not.
"""
import json

import pytest

from validation.trial_registry import (
    LedgerCorrupt,
    TrialRegistry,
    TrialSpec,
)


def spec(name="momentum-v1", family="trend", **params):
    return TrialSpec(name=name, family=family, params=params or {"lookback": 20})


# --- the count is honest ----------------------------------------------------

def test_a_fresh_registry_has_seen_no_trials(tmp_path):
    assert TrialRegistry(tmp_path).cumulative_count() == 0


def test_evaluating_increments_the_count(tmp_path):
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec(), lambda s: {"sharpe": 1.2})
    assert reg.cumulative_count() == 1


def test_the_count_is_cumulative_across_processes(tmp_path):
    """N is the total number of candidates *ever* evaluated against this data,
    not the number in the current search. `DECISIONS.md` §5's 45-variant finding
    is about the cumulative figure; a per-run counter understates it forever."""
    TrialRegistry(tmp_path).evaluate(spec("a"), lambda s: {"sharpe": 1.0})
    TrialRegistry(tmp_path).evaluate(spec("b"), lambda s: {"sharpe": 1.0})
    assert TrialRegistry(tmp_path).cumulative_count() == 2


def test_the_evaluator_result_is_returned_to_the_caller(tmp_path):
    """Enforcement must not make the registry annoying enough to route around."""
    reg = TrialRegistry(tmp_path)
    assert reg.evaluate(spec(), lambda s: {"sharpe": 1.7})["sharpe"] == 1.7


def test_the_evaluator_receives_the_spec(tmp_path):
    seen = []
    TrialRegistry(tmp_path).evaluate(spec(name="x", family="carry"),
                                     lambda s: seen.append(s) or {"sharpe": 0.0})
    assert seen[0].name == "x" and seen[0].family == "carry"


# --- abandoned runs count, which is the whole point -------------------------

def test_a_crashing_evaluation_still_counts(tmp_path):
    """The trial that blew up is still a trial. It consumed a look at the data,
    and the next candidate benefits from what was learned - so N must include it.
    Counting only completions is how a 500-candidate search reports N=40."""
    reg = TrialRegistry(tmp_path)

    def explodes(s):
        raise RuntimeError("optimiser diverged")

    with pytest.raises(RuntimeError):
        reg.evaluate(spec(), explodes)

    assert reg.cumulative_count() == 1


def test_a_crashing_evaluation_is_recorded_as_abandoned(tmp_path):
    reg = TrialRegistry(tmp_path)
    with pytest.raises(ZeroDivisionError):
        reg.evaluate(spec(), lambda s: 1 / 0)

    trial = reg.trials()[-1]
    assert trial["outcome"] == "abandoned"
    assert "ZeroDivisionError" in trial["error"]


def test_the_trial_is_on_disk_before_the_evaluator_runs(tmp_path):
    """Pre-registration, ported from `validation_holdout.py`'s append-before-the-
    outcome-exists pattern. If the process is killed mid-evaluation the trial is
    already counted - which is exactly the case a post-hoc counter loses."""
    reg = TrialRegistry(tmp_path)
    observed = {}

    def look_at_the_ledger_from_inside(s):
        observed["count"] = TrialRegistry(tmp_path).cumulative_count()
        return {"sharpe": 1.0}

    reg.evaluate(spec(), look_at_the_ledger_from_inside)
    assert observed["count"] == 1, "the trial was not counted until after it finished"


def test_a_killed_process_leaves_the_trial_counted_and_pending(tmp_path):
    """Simulates SIGKILL mid-evaluation: the pre-registered row is on disk with
    no outcome. It must still count toward N, and must be visibly unfinished
    rather than quietly assumed successful."""
    reg = TrialRegistry(tmp_path)
    reg.pre_register(spec())        # and then the process dies

    reloaded = TrialRegistry(tmp_path)
    assert reloaded.cumulative_count() == 1
    assert reloaded.trials()[-1]["outcome"] == "pending"


# --- what N is used for -----------------------------------------------------

def test_the_registry_reports_the_trial_sharpes_for_deflation(tmp_path):
    """The Deflated Sharpe needs the cross-sectional variance of the trials
    actually run, not an assumed variance. `cpcv_botonly.py` got this wrong in
    the other direction - it deflated against the number of CPCV resample paths
    of a single strategy, so a candidate picked from thousands was deflated by
    N=15 and the gate read as rigorous while being toothless."""
    reg = TrialRegistry(tmp_path)
    for i, s in enumerate([0.4, 1.1, 0.9]):
        reg.evaluate(spec(f"s{i}"), lambda _s, s=s: {"sharpe": s})
    assert sorted(reg.trial_sharpes()) == [0.4, 0.9, 1.1]


def test_abandoned_trials_count_toward_n_but_contribute_no_sharpe(tmp_path):
    """They consumed a look at the data, so they inflate N. They produced no
    number, so inventing one for them would corrupt the variance estimate."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec("ok"), lambda s: {"sharpe": 1.0})
    with pytest.raises(RuntimeError):
        reg.evaluate(spec("bad"), lambda s: (_ for _ in ()).throw(RuntimeError("x")))

    assert reg.cumulative_count() == 2
    assert reg.trial_sharpes() == [1.0]


# --- fail closed ------------------------------------------------------------

def test_a_corrupt_ledger_raises_rather_than_reporting_zero(tmp_path):
    """The opposite of `antioverfit.py`, which returns 0 on any exception. N=0
    means no multiple-testing correction at all, so the flattering answer is the
    one a bare `except` produces. Refusing is recoverable; a silent N=0 is not."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec(), lambda s: {"sharpe": 1.0})
    (tmp_path / "trial_registry.ndjson").write_text("{not json at all\n", encoding="utf-8")

    with pytest.raises(LedgerCorrupt):
        TrialRegistry(tmp_path).cumulative_count()


def test_a_torn_final_line_is_tolerated_but_still_counted(tmp_path):
    """A process killed mid-append leaves a partial last line. That is a normal
    crash, not corruption - but the trial it represents was still run, so it
    counts. Dropping it would understate N in precisely the flattering
    direction."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec(), lambda s: {"sharpe": 1.0})
    path = tmp_path / "trial_registry.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"trial_id": 2, "name": "half-writ')

    assert TrialRegistry(tmp_path).cumulative_count() == 2


def test_the_ledger_is_append_only(tmp_path):
    """A registry whose history can be rewritten is a registry whose N can be
    negotiated down after a disappointing result."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec("first"), lambda s: {"sharpe": 1.0})
    before = (tmp_path / "trial_registry.ndjson").read_text(encoding="utf-8")

    reg.evaluate(spec("second"), lambda s: {"sharpe": 2.0})
    after = (tmp_path / "trial_registry.ndjson").read_text(encoding="utf-8")

    assert after.startswith(before), "an earlier trial's record was rewritten"


def test_every_trial_records_what_was_tried(tmp_path):
    """A count with no record of what produced it cannot be audited, and an
    unauditable N is one nobody will trust enough to act on."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(TrialSpec(name="m", family="trend", params={"lookback": 55}),
                 lambda s: {"sharpe": 1.0})

    row = reg.trials()[-1]
    assert row["name"] == "m"
    assert row["family"] == "trend"
    assert row["params"] == {"lookback": 55}
    assert row["registered_at_ns"] > 0


def test_identical_specs_are_two_trials_not_one(tmp_path):
    """Re-running the same candidate is another look at the same data. Treating
    it as free is how a search that tried 200 things reports 30 - and the whole
    point of `DECISIONS.md` §5 is that the *number of looks* is what harms you."""
    reg = TrialRegistry(tmp_path)
    reg.evaluate(spec("same"), lambda s: {"sharpe": 1.0})
    reg.evaluate(spec("same"), lambda s: {"sharpe": 1.4})
    assert reg.cumulative_count() == 2


def test_trial_ids_are_dense_and_monotonic(tmp_path):
    reg = TrialRegistry(tmp_path)
    for i in range(3):
        reg.evaluate(spec(f"s{i}"), lambda s: {"sharpe": 1.0})
    assert [t["trial_id"] for t in reg.trials()] == [1, 2, 3]


def test_concurrent_registries_do_not_reuse_a_trial_id(tmp_path):
    """Two search processes sharing a dataset share an N. If they can both claim
    trial 7, the ledger loses a row and N undercounts."""
    a, b = TrialRegistry(tmp_path), TrialRegistry(tmp_path)
    a.pre_register(spec("a"))
    b.pre_register(spec("b"))
    ids = [t["trial_id"] for t in TrialRegistry(tmp_path).trials()]
    assert sorted(ids) == [1, 2], f"trial ids collided: {ids}"
