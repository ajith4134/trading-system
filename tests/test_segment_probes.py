"""A probe measures the running system, or it says it measured nothing.

BF-10 exists because 29 of the spine's 43 named probes had no implementation, so
rows describing running, journalling, trading bots rendered NOT MEASURED. The
defect these tests defend against is the opposite one and it is worse: a probe
that returns OK without having read anything. Every case below either feeds a
probe a fabricated state root and checks it reports what is in it, or feeds it an
empty one and checks it refuses to say anything good.

The other property under test is cost. The options bot wrote 4.1 GB of decisions
in one day; a probe that read a journal end to end would take the boards process
down with it, which is exactly how the wall generator was OOM-killed three times
on 2026-08-19. `tail_rows` is therefore tested on a file larger than its own read
window.
"""
import json

import pytest

from statuswall import segment_probes as probes
from statuswall.evidence import DEGRADED, FAILING, NOT_MEASURED, OK, PARTIAL


# --- fixtures: a state root that looks like the real one -------------------

def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _heartbeat(now_ns, **overrides) -> dict:
    payload = {
        "written_at_ns": now_ns,
        "segment": "perp",
        "venue": "binance-futures",
        "band": "fast",
        "brains": {"bull": "perp-bull-rule", "bear": "perp-bear-rule",
                   "profit_tail": "perp-profit-tail-fast"},
        "learned": False,
        "model_version": "",
        "tail_model_version": "",
        "live_fitted": None,
        "feed": {"liveness": "LIVE", "newest_age_seconds": 0.4},
        "universe_admission": {"considered": 300, "admitted": 120, "excluded": 180,
                               "excluded_by_reason": {"NOT_ENOUGH_SAMPLES": 180}},
        "counts": {"polls": 9, "opened": 3, "closed": 2, "gate_refusals": 7,
                   "frames_refused": 11},
        "open_positions": 1,
    }
    payload.update(overrides)
    return payload


def _decision(outcome="ABSTAIN", bull_reason="FLOW_NOT_BUY_SIDE",
              bear_reason="MOMENTUM_NOT_DOWN", tail_authority="advisory-input-only",
              **extra) -> dict:
    row = {
        "at_ns": 1, "segment": "perp", "symbol": "BTCUSDT", "outcome": outcome,
        "reason": extra.pop("reason", "NO_PROPOSAL"),
        "evidence": {
            "bull": {"brain": "perp-bull-rule", "outcome": "DECLINE",
                     "reason": bull_reason, "evidence": {"samples": 19}},
            "bear": {"brain": "perp-bear-rule", "outcome": "DECLINE",
                     "reason": bear_reason, "evidence": {"samples": 19}},
            "tail": {"net_expectancy": "0.001", "authority": tail_authority},
        },
    }
    row.update(extra)
    return row


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """An empty capture/segment tree the probes read instead of the real one."""
    root = tmp_path / "segment"
    root.mkdir()
    monkeypatch.setattr(probes, "STATE_ROOT", root)
    return root


@pytest.fixture
def live_root(state_root):
    """All four bots polling, journalling decisions and fills."""
    now_ns = probes.time.time_ns()
    for segment in probes.SEGMENTS:
        _write(state_root / segment / "heartbeat.json",
               _heartbeat(now_ns, segment=segment, venue=f"venue-{segment}",
                          brains={"bull": f"{segment}-bull", "bear": f"{segment}-bear",
                                  "profit_tail": f"{segment}-tail"}))
        _write_rows(state_root / segment / "decisions-2026-08-19.ndjson",
                    [_decision(), _decision(outcome="SELECTED")])
        _write_rows(state_root / segment / "fills-2026-08-19.ndjson",
                    [{"event": "OPEN", "band": "fast", "hard_stop": "1.0",
                      "entry_timing": "IMMEDIATE_ENTRY_BASELINE"}])
    return state_root


# --- the bounded read ------------------------------------------------------

def test_a_journal_larger_than_the_read_window_is_still_read_from_its_end(tmp_path):
    path = tmp_path / "decisions-2026-08-19.ndjson"
    rows = [{"i": i, "pad": "x" * 200} for i in range(20_000)]
    _write_rows(path, rows)
    assert path.stat().st_size > probes.TAIL_BYTES

    read = probes.tail_rows(path, rows=5, max_bytes=probes.TAIL_BYTES)

    assert [r["i"] for r in read] == [19_995, 19_996, 19_997, 19_998, 19_999]


def test_the_half_line_a_seek_lands_in_is_dropped_rather_than_parsed(tmp_path):
    path = tmp_path / "fills-2026-08-19.ndjson"
    _write_rows(path, [{"i": i, "pad": "y" * 100} for i in range(500)])

    read = probes.tail_rows(path, rows=400, max_bytes=1_000)

    assert read, "a small window must still return the rows it can read"
    assert all("i" in row for row in read)


def test_a_journal_that_does_not_exist_reads_as_no_rows_rather_than_raising(tmp_path):
    assert probes.tail_rows(tmp_path / "absent.ndjson") == []


# --- absence is its own state ---------------------------------------------

JOURNAL_BACKED = [
    "probe_segment_engine_running", "probe_perp_engine_running",
    "probe_live_feed_fresh", "probe_intraday_data_lag",
    "probe_segment_features_current", "probe_perp_features_current",
    "probe_segment_brains_reason", "probe_perp_bull_agent_reasons",
    "probe_perp_bear_agent_reasons", "probe_segment_arbiter_decides",
    "probe_perp_arbiter_decides", "probe_perp_bands_journalled",
    "probe_perp_risk_gate_refuses", "probe_perp_band_performance",
    "probe_spot_universe_measured", "probe_dated_universe_measured",
    "probe_options_universe_measured", "probe_universe_is_the_venues",
    "probe_universe_breadth", "probe_segments_captured", "probe_segment_bots",
    "probe_three_bots", "probe_uptime_continuous", "probe_beliefs_carry_provenance",
]


@pytest.mark.parametrize("name", JOURNAL_BACKED)
def test_no_probe_reports_ok_against_a_state_root_with_nothing_in_it(state_root, name):
    result = probes.SEGMENT_PROBES[name]()

    assert result.state != OK, f"{name} claimed OK having read nothing"
    assert result.proof, f"{name} returned a state with no provenance"


def test_a_bot_that_never_wrote_a_heartbeat_is_not_measured_rather_than_stopped(state_root):
    result = probes.probe_perp_engine_running()

    assert result.state == NOT_MEASURED
    assert "never written a heartbeat" in result.detail


def test_a_heartbeat_that_stopped_advancing_is_degraded_not_absent(state_root):
    stale_ns = probes.time.time_ns() - 3_600_000_000_000
    _write(state_root / "perp" / "heartbeat.json", _heartbeat(stale_ns))

    result = probes.probe_perp_engine_running()

    assert result.state == DEGRADED
    assert "stopped advancing" in result.detail


def test_a_bot_switched_off_is_named_as_off_and_never_counted_as_running(live_root):
    (live_root / "spot" / "OFF").touch()

    result = probes.probe_segment_engine_running()

    assert result.state == PARTIAL
    assert "spot: deliberately OFF" in result.detail


# --- the fraction, because assigned once is not resolved four times --------

def test_four_of_four_is_ok_and_two_of_four_names_the_two_that_are_missing(live_root):
    assert probes.probe_segment_engine_running().state == OK

    for segment in ("dated", "options"):
        (live_root / segment / "heartbeat.json").unlink()
    result = probes.probe_segment_engine_running()

    assert result.state == PARTIAL
    assert "2/4" in result.detail
    assert "dated" in result.detail and "options" in result.detail


# --- feeds -----------------------------------------------------------------

def test_a_quiet_feed_is_reported_quiet_rather_than_as_an_empty_market(live_root):
    payload = _heartbeat(probes.time.time_ns())
    payload["feed"] = {"liveness": "QUIET", "newest_age_seconds": 900}
    _write(live_root / "perp" / "heartbeat.json", payload)

    result = probes.probe_live_feed_fresh()

    assert result.state == PARTIAL
    assert "perp: feed QUIET" in result.detail


def test_a_tick_measured_in_minutes_fails_the_intraday_data_rule(live_root):
    for segment in probes.SEGMENTS:
        payload = _heartbeat(probes.time.time_ns(), segment=segment)
        payload["feed"] = {"liveness": "LIVE", "newest_age_seconds": 4000}
        _write(live_root / segment / "heartbeat.json", payload)

    assert probes.probe_intraday_data_lag().state == FAILING


# --- brains and the arbiter ------------------------------------------------

def test_a_bear_that_declines_for_exactly_the_bulls_reasons_reads_as_a_negation(live_root):
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_decision(bull_reason="SAME", bear_reason="SAME")])

    result = probes.probe_perp_bear_agent_reasons()

    assert result.state == DEGRADED
    assert "negation" in result.detail


def test_a_decision_selecting_both_sides_fails_the_arbiter_probe(live_root):
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_decision(outcome="SELECTED", side="BOTH")])

    assert probes.probe_perp_arbiter_decides().state == FAILING


def test_a_profit_tail_claiming_more_than_advisory_authority_fails(live_root):
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_decision(tail_authority="veto")])

    assert probes.probe_perp_arbiter_decides().state == FAILING


def test_an_open_journalled_without_a_hard_stop_is_not_counted_as_managed(live_root):
    _write_rows(live_root / "perp" / "fills-2026-08-19.ndjson",
                [{"event": "OPEN", "band": "fast",
                  "entry_timing": "IMMEDIATE_ENTRY_BASELINE"}])

    result = probes.probe_profit_tail_authority()

    assert result.state == PARTIAL
    assert "without a hard stop" in result.detail


# --- bands and performance -------------------------------------------------

def test_a_fill_with_no_band_is_a_failure_because_it_cannot_be_attributed(live_root):
    _write_rows(live_root / "perp" / "fills-2026-08-19.ndjson",
                [{"event": "OPEN", "hard_stop": "1.0"}])

    result = probes.probe_perp_bands_journalled()

    assert result.state == FAILING
    assert "no band" in result.detail


def test_one_band_running_reads_as_partial_and_says_the_second_is_not_running(live_root):
    result = probes.probe_perp_bands_journalled()

    assert result.state == PARTIAL
    assert "second band" in result.detail


def test_two_bands_in_the_journal_are_separable(live_root):
    _write_rows(live_root / "perp" / "fills-2026-08-19.ndjson",
                [{"event": "OPEN", "band": "fast", "hard_stop": "1", "entry_timing": "x"},
                 {"event": "OPEN", "band": "slow", "hard_stop": "1", "entry_timing": "x"}])

    result = probes.probe_perp_bands_journalled()

    assert result.state == OK
    assert "fast 1" in result.detail and "slow 1" in result.detail


def test_a_band_with_too_few_closes_says_so_instead_of_reporting_a_winrate(live_root):
    _write_rows(live_root / "perp" / "fills-2026-08-19.ndjson",
                [{"event": "CLOSE", "band": "fast", "net_pnl": "1.0"} for _ in range(5)])

    result = probes.probe_perp_band_performance()

    assert result.state == PARTIAL
    assert "too few for a winrate" in result.detail
    assert "%" not in result.detail


def test_a_band_with_enough_closes_reports_the_rate_it_earned(live_root):
    closes = [{"event": "CLOSE", "band": "fast", "net_pnl": "1.0"} for _ in range(15)]
    closes += [{"event": "CLOSE", "band": "fast", "net_pnl": "-1.0"} for _ in range(15)]
    _write_rows(live_root / "perp" / "fills-2026-08-19.ndjson", closes)

    result = probes.probe_perp_band_performance()

    assert result.state == OK
    assert "winrate 50.0%" in result.detail


# --- universes -------------------------------------------------------------

def _listing(state_root, segment, listed):
    _write(state_root / segment / "universe.json",
           {"segment": segment, "discovered_at_ns": 1,
            "source": "live.universe_discovery - a venue listing call",
            "listing": {"segment": segment, "listed": listed,
                        "admitted_at_discovery": listed, "dropped_at_discovery": 0,
                        "dropped_by_reason": {}}})


def test_a_venue_that_listed_nothing_is_a_failed_discovery_not_an_empty_market(live_root):
    _listing(live_root, "spot", 0)

    result = probes.probe_spot_universe_measured()

    assert result.state == FAILING
    assert "failed discovery" in result.detail


def test_a_discovery_that_raised_is_recorded_and_read_as_failing(live_root):
    _write(live_root / "dated" / "universe.json",
           {"segment": "dated", "discovery_failed": "HTTPError: 451"})

    result = probes.probe_dated_universe_measured()

    assert result.state == FAILING
    assert "451" in result.detail


def test_the_universe_probe_publishes_the_listing_and_what_admission_excluded(live_root):
    _listing(live_root, "options", 1442)

    result = probes.probe_options_universe_measured()

    assert result.state == OK
    assert "1442 listed by the venue" in result.detail
    assert "180 NOT_ENOUGH_SAMPLES" in result.detail


def test_breadth_reads_as_partial_when_most_of_the_listed_board_is_not_scanned(live_root):
    for segment in probes.SEGMENTS:
        _listing(live_root, segment, 10_000)

    result = probes.probe_universe_breadth()

    assert result.state == PARTIAL
    assert "40000 instruments listed" in result.detail


# --- learned brains --------------------------------------------------------

def test_a_bot_on_rule_brains_is_not_measured_as_learned_and_says_why(live_root):
    payload = _heartbeat(probes.time.time_ns())
    payload["champion_refused"] = {"why": "fitted on other venues (RL-019)"}
    _write(live_root / "perp" / "heartbeat.json", payload)

    result = probes.probe_brains_are_learned()

    assert result.state == NOT_MEASURED
    assert "champion refused" in result.detail


def test_a_bot_deciding_from_a_registered_model_counts_as_learned(live_root):
    for segment in probes.SEGMENTS:
        _write(live_root / segment / "heartbeat.json",
               _heartbeat(probes.time.time_ns(), segment=segment, learned=True,
                          model_version="bbccd8cf0493d613",
                          brains={"bull": f"{segment}-bull-learned",
                                  "bear": f"{segment}-bear-learned",
                                  "profit_tail": f"{segment}-tail"},
                          venue=f"venue-{segment}"))

    assert probes.probe_brains_are_learned().state == OK


def test_a_belief_without_provenance_or_a_half_life_is_not_counted(live_root):
    row = _decision()
    row["evidence"]["bull"]["belief"] = {"claim": "up"}
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson", [row])

    result = probes.probe_beliefs_carry_provenance()

    assert result.state == NOT_MEASURED
    assert "carry no provenance" in result.detail


# --- the self-probe --------------------------------------------------------

def test_every_probe_the_plan_names_is_implemented_somewhere():
    """BF-10's own acceptance, run against the real spine and the real register."""
    result = probes.probe_probes_implemented()

    assert result.state == OK, result.detail


def test_the_registry_keys_match_the_functions_they_point_at():
    for name, probe in probes.SEGMENT_PROBES.items():
        assert probe.__name__ == name, f"{name} is registered against {probe.__name__}"


# --- LB-09: the champion a bot RUNS versus the one that is registered -------

def test_a_bot_running_a_superseded_model_reads_as_superseded_not_as_learned(
        live_root, tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "MODEL_ROOT", tmp_path)
    _write(tmp_path / "aliases.json",
           {f"{s}-direction-champion": "newmodel00000000" for s in probes.SEGMENTS})
    for segment in probes.SEGMENTS:
        _write(live_root / segment / "heartbeat.json",
               _heartbeat(probes.time.time_ns(), segment=segment, learned=True,
                          model_version="oldmodel00000000"))

    result = probes.probe_champion_reload_current()

    assert result.state == NOT_MEASURED
    assert "superseded" in result.detail


def test_a_bot_running_the_registered_champion_is_current(live_root, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(probes, "MODEL_ROOT", tmp_path)
    _write(tmp_path / "aliases.json",
           {f"{s}-direction-champion": "current000000000" for s in probes.SEGMENTS})
    for segment in probes.SEGMENTS:
        _write(live_root / segment / "heartbeat.json",
               _heartbeat(probes.time.time_ns(), segment=segment, learned=True,
                          model_version="current000000000"))

    assert probes.probe_champion_reload_current().state == OK


def test_a_registered_champion_that_no_bot_runs_is_reported_as_unused(live_root,
                                                                     tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(probes, "MODEL_ROOT", tmp_path)
    _write(tmp_path / "aliases.json", {"perp-direction-champion": "registered000000"})

    result = probes.probe_champion_reload_current()

    assert result.state == NOT_MEASURED
    assert "the bot runs no model" in result.detail


# --- a probe must read BOTH brain shapes, not only the one it was written on

def _learned_decision(reason="MODEL_BELOW_THRESHOLD", same_evidence=False):
    bull = {"brain": "perp-bull-learned", "outcome": "DECLINE", "reason": reason,
            "evidence": {"model_version": "f2a4adfb", "trial_id": 36,
                         "rule_brain": False, "sealed_bars": 2}}
    bear = {"brain": "perp-bear-learned", "outcome": "DECLINE", "reason": reason,
            "evidence": dict(bull["evidence"]) if same_evidence else
            {"model_version": "f2a4adfb", "trial_id": 36, "rule_brain": False,
             "reversed_reading_penalty": 0.05, "sealed_bars": 2}}
    return {"at_ns": 1, "segment": "perp", "symbol": "BTCUSDT", "outcome": "ABSTAIN",
            "reason": "NO_PROPOSAL",
            "evidence": {"bull": bull, "bear": bear,
                         "tail": {"authority": "advisory-input-only"}}}


def test_a_learned_brain_names_its_window_as_sealed_bars_and_that_counts(live_root):
    """Measured 2026-08-19: the moment a trained brain was deployed this probe read
    FAILING, because it knew only the rule brain's word for the window."""
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_learned_decision()])

    assert probes.probe_perp_features_current().state == OK


def test_two_brains_sharing_a_warm_up_reason_are_not_called_a_negation(live_root):
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_learned_decision(reason="WARMING_UP_NO_FEATURE_VECTOR")])

    result = probes.probe_perp_bear_agent_reasons()

    assert result.state == PARTIAL, "a warm-up is not a negated bull"
    assert "warm-up" in result.detail


def test_a_bear_carrying_the_bulls_own_evidence_is_a_negation(live_root):
    _write_rows(live_root / "perp" / "decisions-2026-08-19.ndjson",
                [_learned_decision(same_evidence=True)])

    result = probes.probe_perp_bear_agent_reasons()

    assert result.state == DEGRADED
    assert "negation" in result.detail


# --- BF-11 / BF-12: capital and P&L, in USDT -------------------------------

def _usdt_fill(event, price="100", qty="1", at_ns=1, **extra):
    row = {"at_ns": at_ns, "event": event, "venue": "binance-futures",
           "symbol": "BTCUSDT", "quantity": qty, "price": price, "band": "fast",
           "hard_stop": "1", "entry_timing": "IMMEDIATE_ENTRY_BASELINE"}
    if event == "CLOSE":
        row["gross_pnl"] = extra.pop("pnl", "5")
    row.update(extra)
    return row


def test_a_bot_that_has_closed_a_trade_reports_its_capital_and_pnl(live_root):
    for segment in probes.SEGMENTS:
        _write_rows(live_root / segment / "fills-2026-08-19.ndjson",
                    [_usdt_fill("OPEN", at_ns=1), _usdt_fill("CLOSE", at_ns=2)])

    result = probes.probe_capital_and_pnl_reported()

    assert result.state == OK
    assert "peak" in result.detail and "turnover" in result.detail
    assert "USDT" in result.detail


def test_a_bot_whose_fills_are_all_unconvertible_says_so_rather_than_reporting_zero(
        live_root):
    """A zero P&L from nothing counted looks exactly like a zero P&L from a flat
    bot, and RL-029 is what separates them."""
    for segment in probes.SEGMENTS:
        _write_rows(live_root / segment / "fills-2026-08-19.ndjson",
                    [{"at_ns": 1, "event": "OPEN", "venue": "deribit",
                      "symbol": "BTC-26MAR27-68000-C", "quantity": "0.1",
                      "price": "0.1135"}])

    result = probes.probe_capital_and_pnl_reported()

    assert result.state == NOT_MEASURED
    assert "unconvertible" in result.detail


def test_a_bot_that_has_opened_but_closed_nothing_has_no_return_yet(live_root):
    for segment in probes.SEGMENTS:
        _write_rows(live_root / segment / "fills-2026-08-19.ndjson",
                    [_usdt_fill("OPEN", at_ns=1)])

    result = probes.probe_capital_and_pnl_reported()

    assert result.state == NOT_MEASURED
    assert "closed nothing" in result.detail
