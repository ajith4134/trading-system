"""Feed health as a number, and a number that cannot hide a dead component."""
from features.feed_quality import score_feeds

_HOUR_NS = 3_600_000_000_000
_NOW = 1_786_000_000_000_000_000


def _report(streams, events=100, observation_loss=0, corrupting=0,
            silent=(), mtimes=None, per_stream=None):
    """A capture_health-shaped report. Keys match the real one exactly, so a
    schema drift breaks these tests rather than silently zeroing a component.

    `per_stream` names which stream owns the gaps; without it they are charged
    to every stream in `streams`, which is the fixture equivalent of the
    venue-wide attribution this module stopped doing.
    """
    owners = per_stream if per_stream is not None else [
        key.partition("_")[0] for key in streams]
    return {
        "events_total": events,
        "gaps": {"observation_loss": observation_loss, "corrupting": corrupting,
                 "info": 0},
        "corrupting_non_gap": 0,
        "raw_bytes_by_stream": {key: 1 for key in streams},
        "silent_stream_symbols": list(silent),
        "newest_mtime_ns_by_stream": mtimes if mtimes is not None
        else {key: _NOW for key in streams},
        "events_by_stream": {owner: events for owner in owners},
        "gaps_by_stream": {owner: {"observation_loss": observation_loss,
                                   "corrupting": corrupting}
                           for owner in owners},
    }


def _by_stream(scores):
    return {s.stream: s for s in scores}


# --- the point of the module ----------------------------------------------

def test_one_dead_component_drags_the_score_even_when_the_rest_are_perfect():
    """The minimum, not the mean. A mean of 1, 1, 1 and 0 is 0.75, which reads
    like a healthy feed with a rounding error."""
    reports = {"binance": _report(
        ["trade_BTCUSDT", "trade_ETHUSDT"],
        silent=["trade_BTCUSDT", "trade_ETHUSDT"])}
    score = _by_stream(score_feeds(reports))["trade"]
    assert score.delivery == 0.0
    assert score.integrity == 1.0
    assert score.score == 0.0, "a mean would have said 0.75"
    assert score.worst_component == "delivery"


def test_a_healthy_feed_scores_one():
    reports = {"binance": _report(["trade_BTCUSDT"])}
    score = _by_stream(score_feeds(reports))["trade"]
    assert score.score == 1.0


def test_feeds_are_scored_apart_so_one_cannot_carry_another():
    """A venue-level number lets a healthy trade tape hide a dead depth stream
    - the archive-of-order-book-with-no-trades failure, inverted."""
    reports = {"binance": _report(
        ["trade_BTCUSDT", "depth_BTCUSDT"], silent=["depth_BTCUSDT"])}
    scores = _by_stream(score_feeds(reports))
    assert scores["trade"].score == 1.0
    assert scores["depth"].score == 0.0


def test_the_worst_component_is_named_so_a_bad_score_has_a_next_step():
    reports = {"binance": _report(["trade_BTCUSDT"], events=100,
                                  observation_loss=90)}
    score = _by_stream(score_feeds(reports))["trade"]
    assert score.worst_component == "continuity"
    assert abs(score.continuity - 0.1) < 1e-9


# --- freshness, and why it is relative to the venue -----------------------

def test_a_feed_that_stopped_while_its_siblings_kept_writing_is_stale():
    reports = {"binance": _report(
        ["trade_BTCUSDT", "depth_BTCUSDT"],
        mtimes={"trade_BTCUSDT": _NOW,
                "depth_BTCUSDT": _NOW - 3 * _HOUR_NS})}
    scores = _by_stream(score_feeds(reports))
    assert scores["trade"].freshness == 1.0
    assert scores["depth"].freshness == 0.0
    assert scores["depth"].worst_component == "freshness"


def test_a_wholly_stopped_venue_does_not_report_every_feed_individually_stale():
    """That is one fact about the venue, and the venue's own tiles say it.
    Reporting it once per feed would bury the asymmetric case that matters."""
    old = _NOW - 50 * _HOUR_NS
    reports = {"binance": _report(
        ["trade_BTCUSDT", "depth_BTCUSDT"],
        mtimes={"trade_BTCUSDT": old, "depth_BTCUSDT": old})}
    scores = _by_stream(score_feeds(reports))
    assert all(s.freshness == 1.0 for s in scores.values())


def test_a_report_with_no_write_times_scores_freshness_zero_not_healthy():
    reports = {"binance": _report(["trade_BTCUSDT"], mtimes={})}
    score = _by_stream(score_feeds(reports))["trade"]
    assert score.freshness == 0.0, "unmeasured must not read as fresh"


# --- shape ----------------------------------------------------------------

def test_every_venue_is_scored_separately():
    reports = {
        "binance": _report(["trade_BTCUSDT"]),
        "bybit": _report(["allLiquidation_BTCUSDT"],
                         silent=["allLiquidation_BTCUSDT"]),
    }
    scores = score_feeds(reports)
    assert {(s.venue, s.stream) for s in scores} == {
        ("binance", "trade"), ("bybit", "allLiquidation")}
    assert {s.venue: s.score for s in scores} == {"binance": 1.0, "bybit": 0.0}


def test_a_venue_with_nothing_captured_yields_no_feeds():
    assert score_feeds({"binance": _report([])}) == []


def test_no_reports_yields_no_scores():
    assert score_feeds({}) == []


# --- attribution ----------------------------------------------------------

def test_one_streams_gaps_are_not_charged_to_its_siblings():
    """Measured 2026-08-09: every binance feed scored an identical 0.049
    continuity because one busy stream's 251,558 gaps were charged venue-wide.
    A component that is the same for every feed says nothing about any."""
    reports = {"binance": _report(
        ["trade_BTCUSDT", "depth_BTCUSDT"], events=100, observation_loss=90,
        per_stream=["trade"])}
    scores = _by_stream(score_feeds(reports))
    assert abs(scores["trade"].continuity - 0.1) < 1e-9
    assert scores["depth"].continuity == 1.0
    assert scores["depth"].score == 1.0


def test_a_control_channel_is_not_scored_as_a_feed():
    """A subscribe-ack stream writes once at connect and never again - correct
    behaviour that reads as total staleness. Recognised by the archive rather
    than a name list: control frames are filed under the symbol `unknown`."""
    reports = {"bybit-liq": _report(
        ["allLiquidation_BTCUSDT", "subscribe_unknown"],
        mtimes={"allLiquidation_BTCUSDT": _NOW,
                "subscribe_unknown": _NOW - 5 * _HOUR_NS})}
    streams = {s.stream for s in score_feeds(reports)}
    assert streams == {"allLiquidation"}
