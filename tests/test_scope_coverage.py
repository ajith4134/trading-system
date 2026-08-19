"""A capability assigned once is not a capability delivered four times.

This is the arithmetic that would have caught the user's own example. The
examination-hall ruling of 2026-08-02 - watch every symbol in every segment,
wait for each one's setup - was argued through, written into the goal document
as roughly 140 lines of §5a, and produced zero rows of work. A sweep that only
asked "is it assigned anywhere" would have called it covered the moment one
slice mentioned it, while three of four bots never received it.

So the denominator comes from the SCOPE, not from the sweep: per-segment needs
a row in all four segment slices, per-brain in all twelve brains, shared and
system need one anywhere. Anything short of the denominator is unresolved and
names the gap, because a gap nobody names is a gap nobody closes.
"""
from plan.master_plan import PlanRow, PlanSlice
from plan.rulings import Ruling
from plan.scope_coverage import BRAINS, SEGMENTS, cover_ruling


def _row(row_id: str, slice_key: str, satisfies=("RL-009",)) -> PlanRow:
    return PlanRow(id=row_id, slice_key=slice_key, does="does a thing",
                   satisfies=tuple(satisfies), sources=("FEATURES §1",),
                   depends_on=(), probe="probe_thing", accepts="it works",
                   decided=None)


def _slices(*pairs) -> list[PlanSlice]:
    by_key: dict[str, list[PlanRow]] = {}
    for row_id, slice_key in pairs:
        by_key.setdefault(slice_key, []).append(_row(row_id, slice_key))
    return [PlanSlice(k, k.upper(), tuple(v)) for k, v in by_key.items()]


def _ruling(scope: str) -> Ruling:
    return Ruling(id="RL-009", date="2026-08-02", session="0a15b275",
                  verbatim="the brain acts as a teacher in an examination hall",
                  means="watch every symbol, wait for its setup",
                  recorded_in=("goal §5a",), probe=None, scope=scope)


def test_four_segments_are_the_denominator_and_twelve_brains_are_the_other():
    assert len(SEGMENTS) == 4
    assert len(BRAINS) == 12
    assert all(any(b.startswith(s) for b in BRAINS) for s in SEGMENTS)


def test_a_per_segment_ruling_in_one_slice_reports_one_of_four_and_is_unresolved():
    coverage = cover_ruling(_ruling("per-segment"), _slices(("SP-01", "spot-bot")))
    assert (coverage.have, coverage.need) == (1, 4)
    assert not coverage.resolved
    assert "perp-bot" in coverage.missing


def test_a_per_segment_ruling_in_all_four_slices_is_resolved():
    coverage = cover_ruling(_ruling("per-segment"),
                            _slices(*[(f"X-{i}", s) for i, s in enumerate(SEGMENTS)]))
    assert (coverage.have, coverage.need) == (4, 4)
    assert coverage.resolved
    assert coverage.missing == ()


# --- per-brain, under RL-031 -----------------------------------------------
#
# **The test that used to stand here keyed its slices BY BRAIN NAME**, and that
# is a shape the real plan cannot produce: a slice key is `spot-bot` or
# `learned-brains`, never `spot-bot/BULL`. So the fixture proved an arithmetic
# that could never run, and underneath it all nine per-brain rulings read 0/12
# and always would have - RL-026 "make the brains real ai" among them. Measured
# 2026-08-19 against the real spine.
#
# RL-031 settles how a row becomes attributable to a brain: a row in a SEGMENT
# slice covers that segment's three brains together, because that is how the
# bots are actually built - a capability lands in a segment bot and BULL, BEAR
# and PROFIT-TAIL receive it at once.


def test_a_row_in_a_segment_slice_covers_that_segments_three_brains():
    coverage = cover_ruling(_ruling("per-brain"), _slices(("SP-01", "spot-bot")))
    assert (coverage.have, coverage.need) == (3, 12)
    assert not coverage.resolved
    assert set(coverage.missing) == set(BRAINS) - {
        "spot-bot/BULL", "spot-bot/BEAR", "spot-bot/PROFIT-TAIL"}


def test_a_per_brain_ruling_in_all_four_segment_slices_is_resolved():
    coverage = cover_ruling(_ruling("per-brain"),
                            _slices(*[(f"X-{i}", s) for i, s in enumerate(SEGMENTS)]))
    assert (coverage.have, coverage.need) == (12, 12)
    assert coverage.resolved
    assert coverage.missing == ()


def test_a_row_in_a_shared_slice_covers_no_brain():
    """**A capability built once is not thereby delivered to four bots.**

    This is the whole doctrine of the scope arithmetic, and letting a shared
    slice count for all twelve would reproduce inside the plan the exact failure
    the plan exists to catch. LB-09 sits in `learned-brains` and satisfies the
    per-brain RL-030; it must read as covering no brain until rows exist in the
    segment slices that receive it.
    """
    for shared in ("learned-brains", "bot-framework", "slice-0", "slice-5"):
        coverage = cover_ruling(_ruling("per-brain"), _slices(("X-01", shared)))
        assert coverage.have == 0, f"{shared} must attribute to no brain"
        assert set(coverage.missing) == set(BRAINS)


def test_three_segments_of_four_reports_nine_of_twelve_and_names_the_missing_bot():
    rows = [(f"X-{i}", s) for i, s in enumerate(SEGMENTS[:-1])]
    coverage = cover_ruling(_ruling("per-brain"), _slices(*rows))
    assert (coverage.have, coverage.need) == (9, 12)
    assert all(m.startswith(SEGMENTS[-1]) for m in coverage.missing)
    assert len(coverage.missing) == 3


def test_a_shared_ruling_needs_exactly_one_row_anywhere():
    coverage = cover_ruling(_ruling("shared"), _slices(("SL-01", "slice-0")))
    assert (coverage.have, coverage.need) == (1, 1)
    assert coverage.resolved


def test_a_ruling_with_no_row_at_all_is_unresolved_at_zero():
    coverage = cover_ruling(_ruling("shared"), [])
    assert (coverage.have, coverage.need) == (0, 1)
    assert not coverage.resolved


def test_a_row_that_does_not_cite_the_ruling_does_not_count_toward_it():
    """Coverage is what a row CLAIMS to satisfy, never what it sits near."""
    other = PlanSlice("spot-bot", "SPOT", (_row("SP-01", "spot-bot",
                                                satisfies=("RL-018",)),))
    coverage = cover_ruling(_ruling("per-segment"), [other])
    assert coverage.have == 0
    assert set(coverage.missing) == set(SEGMENTS)


def test_two_rows_in_the_same_slice_count_once_not_twice():
    """Four bots is the denominator; two rows for one bot is still one bot."""
    coverage = cover_ruling(_ruling("per-segment"),
                            _slices(("SP-01", "spot-bot"), ("SP-02", "spot-bot")))
    assert (coverage.have, coverage.need) == (1, 4)
    assert not coverage.resolved
