"""BF-10: the probes the segment, framework and learned-brain rows name.

Measured 2026-08-18: the spine named 43 distinct probes and 29 of them existed
nowhere, so every row of the bot framework and of the learned brains rendered NOT
MEASURED while the bots were running, journalling and trading. That is the plan
working rather than a reporting bug - Rule 8 says an unmeasured thing renders
unmeasured - and what was missing was the measurement. This module supplies it.

## What a probe here is allowed to do

**It reads the running system and nothing else.** Not a test result, not a
constant, not a document that claims something. The heartbeat each engine writes
on every poll, the journals it appends, the model registry the retrainer writes,
the universe listing the engine records at start. If a file is absent the probe
says NOT MEASURED and names the file it looked for; there is no path in here that
can produce OK without having read something.

**It never loads a whole journal.** The options bot wrote 4.1 GB of decisions on
2026-08-18 alone. Every journal read here is a bounded tail seek from the end of
the file, so a probe costs the same on day one as on day ninety.

## Why this is not in `ruling_conformance`

That module is imported by `plan.cli`, and probes that walk registries and
journals have no business inflating any boards process further - the wall
generator measured 4.8 GB resident on 2026-08-17 and 11.7 GB on 2026-08-18, and
was OOM-killed three times on 2026-08-19. These live in their own module and are
merged into the registry by name at the point of use.

## The states, and what each one means here

`OK` the acceptance is met on live state · `PARTIAL` met for some segments and
the missing ones are named · `DEGRADED` measured and wrong, e.g. a heartbeat that
stopped advancing · `FAILING` measured and breached · `NOT_MEASURED` nothing on
disk to read. A probe that raises is caught by the caller and also reads NOT
MEASURED, which is why nothing here defends itself with a bare except.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from statuswall.evidence import (
    DEGRADED, FAILING, NOT_MEASURED, OK, PARTIAL, ProbeResult,
)

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
STATE_ROOT = HOME / "capture" / "segment"
LEARN_ROOT = HOME / "capture" / "learn"
MODEL_ROOT = HOME / "capture" / "models"
BOARDS_DIR = HOME / "research" / "dashboard"
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"
REGISTER = REPO / "docs" / "rulings.json"
AXIS_VERDICTS = REPO / "docs" / "axis-verdicts.json"
PERMANENT_RULES = HOME / ".claude" / "CLAUDE.md"

SEGMENTS = ("perp", "spot", "dated", "options")

# A heartbeat is written every poll and the loop is 5-6 seconds. Two minutes is
# generous enough that a slow poll is not reported as a death, and short enough
# that a bot killed at boot does not read LIVE for an hour.
STALE_AFTER_S = 120.0
# A journal tail. Enough rows to see both outcomes of a decision and both exit
# bands; small enough that the read is bounded on a 4 GB file.
TAIL_ROWS = 400
TAIL_BYTES = 4_000_000
# A winrate under this many closes is not a winrate (PB-10's acceptance).
MIN_CLOSES_FOR_RATE = 20
# **What "the window this frame was computed from" is called, per brain kind.**
# A rule brain counts raw observations (`samples`); a learned brain counts the
# sealed one-minute bars its feature vector was built from (`sealed_bars`) and
# names the vector it fed the model. BF-02's acceptance is that the row NAMES its
# window - not that it uses one particular word for it, and a probe that knew
# only the rule brain's word reported FAILING the moment a trained brain was
# deployed, which is what happened on 2026-08-19.
WINDOW_KEYS = ("samples", "sealed_bars", "feature_vector_length")


# --- reading, always bounded ----------------------------------------------

def read_json(path: Path):
    """The parsed file, or None. An unreadable file is not an exception here."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_rows(path: Path, rows: int = TAIL_ROWS,
              max_bytes: int = TAIL_BYTES) -> list:
    """The last `rows` JSON objects of an ndjson file, read from the end.

    Seeks rather than iterates. The first line of the window is dropped because a
    seek lands mid-line, and a half-line is not a record.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - max_bytes))
            window = handle.read()
    except OSError:
        return []
    lines = window.split(b"\n")
    if size > max_bytes and lines:
        lines = lines[1:]
    parsed = []
    for line in lines[-(rows + 1):]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except ValueError:
            continue
    return parsed[-rows:]


# The state root is resolved at CALL time, never bound as a default argument. A
# default is evaluated once at import, so a probe carrying one would read the real
# capture tree no matter what it was pointed at - which is a probe that cannot be
# tested against a fabricated state, and an untestable probe is how a board starts
# reporting the wrong machine.

def journal_paths(segment: str, name: str, root: Path | None = None) -> list:
    """Every day file of one journal, oldest first."""
    root = STATE_ROOT if root is None else root
    return sorted((root / segment).glob(f"{name}-*.ndjson"))


def latest_journal(segment: str, name: str, root: Path | None = None):
    paths = journal_paths(segment, name, root)
    return paths[-1] if paths else None


def heartbeat(segment: str, root: Path | None = None):
    """(payload, age_seconds) for one bot. (None, None) if it never wrote one."""
    root = STATE_ROOT if root is None else root
    payload = read_json(root / segment / "heartbeat.json")
    if not isinstance(payload, dict):
        return None, None
    written = payload.get("written_at_ns")
    try:
        age_s = (time.time_ns() - int(written)) / 1e9
    except (TypeError, ValueError):
        return payload, None
    return payload, age_s


def live_heartbeats(root: Path | None = None) -> dict:
    """Every segment's heartbeat and age, in one pass, so tiles agree on the instant."""
    return {segment: heartbeat(segment, root) for segment in SEGMENTS}


def _fraction(met: list, missing: dict, subject: str, proof: str) -> ProbeResult:
    """Four segments, and a partial delivery reads as a fraction that names the gap.

    RL-019 makes each segment its own bot, so `some of them do it` is the answer
    the board must show rather than a green tile earned by one.
    """
    detail = f"{len(met)}/{len(SEGMENTS)} {subject}"
    if missing:
        detail += " - " + "; ".join(f"{k}: {v}" for k, v in sorted(missing.items()))
    if len(met) == len(SEGMENTS):
        return ProbeResult(OK, detail, proof)
    if met:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


def _process_running(pattern: str) -> tuple[bool, str]:
    """Whether a supervisor is up, by pattern, without shelling through a pipe."""
    try:
        found = subprocess.run(["pgrep", "-f", pattern],
                               capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as failure:
        return False, f"pgrep failed: {failure!r}"
    pids = [line for line in found.stdout.split() if line.isdigit()]
    return bool(pids), (f"pids {','.join(pids)}" if pids else "no process matches")


# --- the engines, their feeds and their journals --------------------------

def probe_segment_engine_running() -> ProbeResult:
    """BF-06 / BF-07: four bots, each on its own venue, each writing its own journal."""
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no heartbeat on disk"
            continue
        if (STATE_ROOT / segment / "OFF").exists():
            missing[segment] = "deliberately OFF"
            continue
        if age_s is None or age_s > STALE_AFTER_S:
            missing[segment] = f"heartbeat {age_s:.0f}s old" if age_s else "no timestamp"
            continue
        if not payload.get("venue"):
            missing[segment] = "heartbeat names no venue"
            continue
        met.append(segment)
    return _fraction(met, missing, "bots polling live, each naming its own venue",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_perp_engine_running() -> ProbeResult:
    """PB-08: the perp bot supervised, journalling fills, with its poll cost carried."""
    payload, age_s = heartbeat("perp")
    proof = str(STATE_ROOT / "perp" / "heartbeat.json")
    if payload is None:
        return ProbeResult(NOT_MEASURED, "the perp bot has never written a heartbeat",
                           proof)
    if (STATE_ROOT / "perp" / "OFF").exists():
        return ProbeResult(DEGRADED, "the off switch is set: deliberately not trading",
                           str(STATE_ROOT / "perp" / "OFF"))
    counts = payload.get("counts") or {}
    detail = (f"heartbeat {age_s:.0f}s old, polls {counts.get('polls')}, "
              f"opened {counts.get('opened')}, closed {counts.get('closed')}, "
              f"open now {payload.get('open_positions')}")
    if age_s is None or age_s > STALE_AFTER_S:
        return ProbeResult(DEGRADED, f"heartbeat has stopped advancing - {detail}", proof)
    fills = latest_journal("perp", "fills")
    if fills is None:
        return ProbeResult(PARTIAL, f"{detail}, and no fill journal exists yet", proof)
    return ProbeResult(OK, f"{detail}, fills at {fills.name}", proof)


def probe_live_feed_fresh() -> ProbeResult:
    """BF-01 / PB-16 / RL-024: the price is a live tick, not a stored bar."""
    met, missing = [], {}
    ages = []
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no heartbeat"
            continue
        feed = payload.get("feed") or {}
        liveness = feed.get("liveness")
        tick_age = feed.get("newest_age_seconds")
        if liveness != "LIVE":
            missing[segment] = f"feed {liveness or 'unknown'}"
            continue
        if age_s is not None and age_s > STALE_AFTER_S:
            missing[segment] = f"heartbeat {age_s:.0f}s old, so this feed age is stale too"
            continue
        if tick_age is None:
            missing[segment] = "feed reports no tick age"
            continue
        ages.append((segment, float(tick_age)))
        # Seconds rather than hours is the acceptance. A minute of tolerance covers
        # the REST-polled segments, which deliver every ten seconds by design.
        if float(tick_age) > 60:
            missing[segment] = f"newest tick {float(tick_age):.0f}s old"
            continue
        met.append(segment)
    shown = ", ".join(f"{s} {a:.0f}s" for s, a in ages) or "no feed ages read"
    return _fraction(met, missing, f"feeds live ({shown})",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_intraday_data_lag() -> ProbeResult:
    """RL-018: intraday means the decision is made on data measured in seconds."""
    worst, ages = None, {}
    for segment, (payload, age_s) in live_heartbeats().items():
        feed = (payload or {}).get("feed") or {}
        tick_age = feed.get("newest_age_seconds")
        if tick_age is None or age_s is None or age_s > STALE_AFTER_S:
            continue
        ages[segment] = float(tick_age)
        worst = max(worst or 0.0, float(tick_age))
    proof = str(STATE_ROOT / "<segment>" / "heartbeat.json")
    if not ages:
        return ProbeResult(NOT_MEASURED, "no live bot is reporting a tick age", proof)
    detail = ("newest tick age " +
              ", ".join(f"{k} {v:.0f}s" for k, v in sorted(ages.items())))
    if worst > 300:
        return ProbeResult(FAILING, f"{detail} - minutes is not intraday data", proof)
    if len(ages) < len(SEGMENTS):
        return ProbeResult(PARTIAL, f"{detail} ({len(ages)}/4 bots live)", proof)
    return ProbeResult(OK, detail, proof)


def probe_segment_features_current() -> ProbeResult:
    """BF-02: a frame names the window it came from, and a missing input refuses."""
    met, missing = [], {}
    for segment in SEGMENTS:
        payload, age_s = heartbeat(segment)
        if payload is None or age_s is None or age_s > STALE_AFTER_S:
            missing[segment] = "bot not live"
            continue
        rows = tail_rows(latest_journal(segment, "decisions") or Path("/nonexistent"))
        if not rows:
            missing[segment] = "no decisions journalled"
            continue
        evidenced = 0
        for row in rows:
            evidence = ((row.get("evidence") or {}).get("bull") or {}).get("evidence") or {}
            if any(key in evidence for key in WINDOW_KEYS):
                evidenced += 1
        if not evidenced:
            missing[segment] = "no decision carries the sample window it was computed from"
            continue
        met.append(segment)
    return _fraction(met, missing,
                     "bots whose decisions carry their feature window",
                     str(STATE_ROOT / "<segment>" / "decisions-*.ndjson"))


def probe_perp_features_current() -> ProbeResult:
    """PB-02: the perp frame is current and a stale input is a named refusal."""
    payload, age_s = heartbeat("perp")
    path = latest_journal("perp", "decisions")
    proof = str(path or STATE_ROOT / "perp")
    if payload is None or path is None:
        return ProbeResult(NOT_MEASURED, "the perp bot has not journalled a decision",
                           proof)
    rows = tail_rows(path)
    if not rows:
        return ProbeResult(NOT_MEASURED, "the decision journal tail is empty", proof)
    with_window = [r for r in rows
                   if any(key in ((((r.get("evidence") or {}).get("bull") or {})
                                   .get("evidence")) or {})
                          for key in WINDOW_KEYS)]
    refused = (payload.get("counts") or {}).get("frames_refused")
    detail = (f"{len(with_window)}/{len(rows)} recent decisions name their sample "
              f"count, {refused} frames refused on the last poll")
    if age_s is not None and age_s > STALE_AFTER_S:
        return ProbeResult(DEGRADED, f"the bot is not polling - {detail}", proof)
    if not with_window:
        return ProbeResult(FAILING,
                           "no perp decision names the window it was computed from",
                           proof)
    return ProbeResult(OK, detail, proof)


# --- brains, arbiter, profit tail -----------------------------------------

def _brain_outcomes(segment: str) -> dict:
    """What the last decisions of one bot actually contain. Counts, never claims."""
    path = latest_journal(segment, "decisions")
    rows = tail_rows(path) if path else []
    summary = {"rows": len(rows), "path": str(path or ""),
               "bull_proposals": 0, "bull_declines": 0, "bull_reasons": set(),
               "bear_proposals": 0, "bear_declines": 0, "bear_reasons": set(),
               "with_evidence": 0, "abstain": 0, "selected": 0, "both_sides": 0,
               "identical_evidence": 0,
               "tail_advisory": 0, "tail_present": 0}
    for row in rows:
        evidence = row.get("evidence") or {}
        for side in ("bull", "bear"):
            block = evidence.get(side) or {}
            outcome = block.get("outcome")
            if outcome == "DECLINE":
                summary[f"{side}_declines"] += 1
            elif outcome:
                summary[f"{side}_proposals"] += 1
            if block.get("reason"):
                summary[f"{side}_reasons"].add(block["reason"])
            if block.get("evidence"):
                summary["with_evidence"] += 1
        bull_block = (evidence.get("bull") or {}).get("evidence")
        bear_block = (evidence.get("bear") or {}).get("evidence")
        if bull_block and bull_block == bear_block:
            summary["identical_evidence"] += 1
        tail = evidence.get("tail") or {}
        if tail:
            summary["tail_present"] += 1
            if tail.get("authority") == "advisory-input-only":
                summary["tail_advisory"] += 1
        outcome = row.get("outcome")
        if outcome == "ABSTAIN":
            summary["abstain"] += 1
        elif outcome:
            summary["selected"] += 1
        if row.get("side") in ("BOTH", "LONG_AND_SHORT"):
            summary["both_sides"] += 1
    return summary


def probe_segment_brains_reason() -> ProbeResult:
    """BF-03 / SB-02 / DB-02 / OB-02 / RL-025: a proposal or a first-class decline."""
    met, missing = [], {}
    for segment in SEGMENTS:
        summary = _brain_outcomes(segment)
        if not summary["rows"]:
            missing[segment] = "no decisions journalled"
            continue
        if not summary["with_evidence"]:
            missing[segment] = "decisions carry no brain evidence"
            continue
        if not (summary["bull_declines"] or summary["bull_proposals"]):
            missing[segment] = "no bull outcome in the journal tail"
            continue
        if not summary["bull_reasons"] and not summary["bear_reasons"]:
            missing[segment] = "declines name no reason"
            continue
        met.append(segment)
    return _fraction(met, missing,
                     "bots whose brains journal a proposal or a reasoned decline",
                     str(STATE_ROOT / "<segment>" / "decisions-*.ndjson"))


def probe_perp_bull_agent_reasons() -> ProbeResult:
    """PB-03: a long proposal names its features, and a decline is an outcome."""
    summary = _brain_outcomes("perp")
    proof = summary["path"] or str(STATE_ROOT / "perp")
    if not summary["rows"]:
        return ProbeResult(NOT_MEASURED, "no perp decision to read", proof)
    detail = (f"{summary['bull_proposals']} proposals and {summary['bull_declines']} "
              f"declines over {summary['rows']} journalled decisions, "
              f"{len(summary['bull_reasons'])} distinct decline reasons")
    if not summary["with_evidence"]:
        return ProbeResult(FAILING, "no bull outcome carries its features", proof)
    if not summary["bull_declines"]:
        return ProbeResult(PARTIAL, f"{detail} - no decline seen in this window", proof)
    return ProbeResult(OK, detail, proof)


def probe_perp_bear_agent_reasons() -> ProbeResult:
    """PB-04: the same, and the bear is a separate decision rather than a negation."""
    summary = _brain_outcomes("perp")
    proof = summary["path"] or str(STATE_ROOT / "perp")
    if not summary["rows"]:
        return ProbeResult(NOT_MEASURED, "no perp decision to read", proof)
    detail = (f"{summary['bear_proposals']} proposals and {summary['bear_declines']} "
              f"declines, reasons {sorted(summary['bear_reasons'])[:3]}")
    if not summary["bear_reasons"]:
        return ProbeResult(FAILING, "no bear outcome names a reason", proof)
    # **Separate decision, measured on the EVIDENCE rather than on the reason.**
    # Sharing a reason is not sharing a decision: both brains legitimately decline
    # with the same word while warming up, and on 2026-08-19 a freshly restarted
    # bot was reported as running a negated bull for exactly that reason. What a
    # negation actually looks like is the bear reasoning from the bull's evidence,
    # byte for byte, which is what this compares.
    if summary["rows"] and summary["identical_evidence"] == summary["rows"]:
        return ProbeResult(DEGRADED,
                           f"{detail} - every bear outcome carries the bull's own "
                           f"evidence, which is what a negation looks like", proof)
    if summary["bear_reasons"] == summary["bull_reasons"]:
        return ProbeResult(PARTIAL,
                           f"{detail} - both brains are declining for the same "
                           f"reason in this window, which a warm-up also produces",
                           proof)
    return ProbeResult(OK, detail, proof)


def probe_segment_arbiter_decides() -> ProbeResult:
    """BF-05: one side or an abstention, never both, and the tail never vetoes."""
    met, missing = [], {}
    for segment in SEGMENTS:
        summary = _brain_outcomes(segment)
        if not summary["rows"]:
            missing[segment] = "no decisions journalled"
            continue
        if summary["both_sides"]:
            missing[segment] = f"{summary['both_sides']} decisions selected both sides"
            continue
        if summary["tail_present"] and not summary["tail_advisory"]:
            missing[segment] = "PROFIT-TAIL is journalled without advisory-only authority"
            continue
        if not (summary["abstain"] or summary["selected"]):
            missing[segment] = "no arbiter outcome journalled"
            continue
        met.append(segment)
    return _fraction(met, missing,
                     "arbiters yielding one side or an abstention with the tail advisory",
                     str(STATE_ROOT / "<segment>" / "decisions-*.ndjson"))


def probe_perp_arbiter_decides() -> ProbeResult:
    """PB-05: the perp arbiter, measured on its own journal."""
    summary = _brain_outcomes("perp")
    proof = summary["path"] or str(STATE_ROOT / "perp")
    if not summary["rows"]:
        return ProbeResult(NOT_MEASURED, "no perp decision to read", proof)
    detail = (f"{summary['abstain']} abstentions and {summary['selected']} selections "
              f"over {summary['rows']} decisions, PROFIT-TAIL advisory on "
              f"{summary['tail_advisory']}/{summary['tail_present']}")
    if summary["both_sides"]:
        return ProbeResult(FAILING,
                           f"{summary['both_sides']} decisions selected both sides", proof)
    if summary["tail_present"] and summary["tail_advisory"] < summary["tail_present"]:
        return ProbeResult(FAILING,
                           "a PROFIT-TAIL block claims authority beyond advisory", proof)
    return ProbeResult(OK, detail, proof)


def probe_profit_tail_authority() -> ProbeResult:
    """BF-04 / PB-15: entry timing and the position, never a veto and never past the stop."""
    met, missing = [], {}
    for segment in SEGMENTS:
        summary = _brain_outcomes(segment)
        fills_path = latest_journal(segment, "fills")
        fills = tail_rows(fills_path) if fills_path else []
        opens = [f for f in fills if f.get("event") == "OPEN"]
        if not summary["rows"] and not fills:
            missing[segment] = "nothing journalled"
            continue
        if summary["tail_present"] and not summary["tail_advisory"]:
            missing[segment] = "the tail block does not declare advisory-only authority"
            continue
        if opens and not all(o.get("hard_stop") for o in opens):
            missing[segment] = "an open was journalled without a hard stop"
            continue
        if opens and not all(o.get("entry_timing") for o in opens):
            missing[segment] = "an open was journalled without entry timing"
            continue
        met.append(segment)
    missed = sum(len(journal_paths(s, "missed_entries")) for s in SEGMENTS)
    return _fraction(met, missing,
                     f"bots whose tail times entries under a hard stop "
                     f"({missed} missed-entry journals exist)",
                     str(STATE_ROOT / "<segment>" / "fills-*.ndjson"))


def probe_perp_bands_journalled() -> ProbeResult:
    """PB-06: fast and slow separable in the journal, one trade never in both."""
    path = latest_journal("perp", "fills")
    proof = str(path or STATE_ROOT / "perp")
    if path is None:
        return ProbeResult(NOT_MEASURED, "the perp bot has journalled no fills", proof)
    fills = tail_rows(path)
    if not fills:
        return ProbeResult(NOT_MEASURED, "the fill journal tail is empty", proof)
    bands: dict = {}
    unbanded = 0
    for fill in fills:
        band = fill.get("band")
        if not band:
            unbanded += 1
            continue
        bands[band] = bands.get(band, 0) + 1
    detail = ", ".join(f"{k} {v}" for k, v in sorted(bands.items())) or "no band named"
    if unbanded:
        return ProbeResult(FAILING,
                           f"{unbanded} fills carry no band and are unattributable", proof)
    if len(bands) < 2:
        # One band running is the true state today, and saying so is the point.
        return ProbeResult(PARTIAL,
                           f"one band is journalling ({detail}); the second band is "
                           f"declared but not running", proof)
    return ProbeResult(OK, f"bands separable in the journal: {detail}", proof)


def probe_perp_risk_gate_refuses() -> ProbeResult:
    """PB-07: a refusal names the limit and the measured value that breached it."""
    payload, _ = heartbeat("perp")
    path = latest_journal("perp", "decisions")
    proof = str(path or STATE_ROOT / "perp")
    refusals = (payload or {}).get("counts", {}).get("gate_refusals")
    rows = tail_rows(path) if path else []
    named = [r for r in rows
             if str(r.get("reason") or "").startswith(("EXPOSURE", "SPREAD", "NOTIONAL",
                                                       "MAX_OPEN", "PRICE"))]
    if payload is None:
        return ProbeResult(NOT_MEASURED, "the perp bot has written no heartbeat", proof)
    detail = (f"{refusals} gate refusals counted this run, "
              f"{len(named)} refusals in the journal tail name their limit")
    if refusals is None:
        return ProbeResult(NOT_MEASURED, "the heartbeat does not count gate refusals",
                           proof)
    if refusals and not named:
        return ProbeResult(PARTIAL,
                           f"{detail} - refusals are counted but the tail window "
                           f"holds none of them", proof)
    return ProbeResult(OK, detail, proof)


def probe_perp_band_performance() -> ProbeResult:
    """PB-10: trades, winrate and net profit per band, and too few trades says so."""
    paths = journal_paths("perp", "fills")
    proof = str(paths[-1] if paths else STATE_ROOT / "perp")
    if not paths:
        return ProbeResult(NOT_MEASURED, "no perp fills to measure", proof)
    per_band: dict = {}
    for path in paths[-2:]:
        for fill in tail_rows(path, rows=5000, max_bytes=TAIL_BYTES):
            if fill.get("event") != "CLOSE":
                continue
            band = fill.get("band") or "unbanded"
            stats = per_band.setdefault(band, {"closes": 0, "wins": 0, "pnl": 0.0})
            stats["closes"] += 1
            try:
                pnl = float(fill.get("net_pnl") or fill.get("gross_pnl") or 0)
            except (TypeError, ValueError):
                pnl = 0.0
            stats["pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
    if not per_band:
        return ProbeResult(NOT_MEASURED, "no closed perp trade in the journal window",
                           proof)
    parts = []
    thin = []
    for band, stats in sorted(per_band.items()):
        if stats["closes"] < MIN_CLOSES_FOR_RATE:
            thin.append(band)
            parts.append(f"{band}: {stats['closes']} closes, too few for a winrate, "
                         f"P&L {stats['pnl']:.2f}")
        else:
            rate = 100.0 * stats["wins"] / stats["closes"]
            parts.append(f"{band}: {stats['closes']} closes, winrate {rate:.1f}%, "
                         f"P&L {stats['pnl']:.2f}")
    detail = " · ".join(parts)
    if thin and len(thin) == len(per_band):
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def probe_perp_promotion_gate() -> ProbeResult:
    """PB-11: promotion on observed history, deflated by the trials it came from."""
    path = HOME / "capture" / "promotion.json"
    proof = str(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ProbeResult(NOT_MEASURED, "no promotion record exists", proof)
    # The file is written with NaN, which json.loads accepts and json.dumps emits.
    try:
        record = json.loads(raw)
    except ValueError:
        return ProbeResult(DEGRADED, "the promotion record does not parse", proof)
    age_days = (time.time() - path.stat().st_mtime) / 86400
    detail = (f"promoted={record.get('promoted')}, trials={record.get('n_trials')}, "
              f"rows={record.get('rows')}, holdout frozen="
              f"{(record.get('holdout') or {}).get('frozen')}, "
              f"measured {age_days:.0f} days ago")
    if record.get("promoted"):
        return ProbeResult(OK, detail, proof)
    # Nothing promoted is the correct state on this record, not a failure.
    return ProbeResult(PARTIAL, f"{detail} - nothing has passed the gate", proof)


# --- the boards themselves ------------------------------------------------

def _board_fresh(name: str):
    path = BOARDS_DIR / name
    if not path.is_file():
        return None, None, path
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return path.read_text(encoding="utf-8", errors="replace"), age_h, path


def probe_segment_tiles_measured() -> ProbeResult:
    """BF-08: no tile green without a probe, and an untraded bot reads NOT MEASURED."""
    page, age_h, path = _board_fresh("segment-bots.html")
    proof = str(path)
    if page is None:
        return ProbeResult(NOT_MEASURED, "the segment wall has never been generated",
                           proof)
    has_unmeasured_state = "NOT MEASURED" in page
    names_all = all(segment in page for segment in SEGMENTS)
    detail = (f"generated {age_h:.1f}h ago, names {'all four' if names_all else 'not all'} "
              f"segments, renders NOT MEASURED={has_unmeasured_state}")
    if not names_all:
        return ProbeResult(DEGRADED, f"{detail} - a bot is missing from the wall", proof)
    if age_h > 6:
        return ProbeResult(DEGRADED, f"{detail} - the wall has stopped regenerating",
                           proof)
    return ProbeResult(OK, detail, proof)


def probe_perp_tile_measured() -> ProbeResult:
    """PB-09: the perp tile specifically, and its numbers come from the heartbeat."""
    page, age_h, path = _board_fresh("segment-bots.html")
    proof = str(path)
    if page is None:
        return ProbeResult(NOT_MEASURED, "the segment wall has never been generated",
                           proof)
    payload, _ = heartbeat("perp")
    if "perp" not in page:
        return ProbeResult(DEGRADED, "the wall does not carry a perp tile", proof)
    if payload is None:
        return ProbeResult(PARTIAL,
                           "the perp tile exists and the bot has written no heartbeat, "
                           "so it can only read NOT MEASURED", proof)
    return ProbeResult(OK, f"perp tile present, wall generated {age_h:.1f}h ago, "
                           f"heartbeat behind it", proof)


def probe_board_generated() -> ProbeResult:
    """RL-012: the dashboard exists, is regenerated, and is not a snapshot."""
    boards = ["status-wall.html", "segment-bots.html", "ajit-master-plan.html",
              "ruling-conformance.html", "blotter.html"]
    fresh, stale, absent = [], [], []
    for name in boards:
        page, age_h, _ = _board_fresh(name)
        if page is None:
            absent.append(name)
        elif age_h > 6:
            stale.append(f"{name} {age_h:.0f}h")
        else:
            fresh.append(name)
    detail = (f"{len(fresh)}/{len(boards)} boards regenerated within 6h"
              + (f", stale: {', '.join(stale)}" if stale else "")
              + (f", absent: {', '.join(absent)}" if absent else ""))
    proof = str(BOARDS_DIR)
    if absent and not fresh:
        return ProbeResult(NOT_MEASURED, detail, proof)
    if stale or absent:
        return ProbeResult(DEGRADED, detail, proof)
    return ProbeResult(OK, detail, proof)


# --- universes ------------------------------------------------------------

def _listing(segment: str) -> dict | None:
    return read_json(STATE_ROOT / segment / "universe.json")


def _universe_state(segment: str) -> tuple[str, str]:
    """(state, detail) for one segment's universe, from the venue listing and the poll."""
    listing = _listing(segment)
    payload, age_s = heartbeat(segment)
    admission = (payload or {}).get("universe_admission") or {}
    if listing is None:
        return NOT_MEASURED, "no venue listing has been recorded by the engine"
    if listing.get("discovery_failed"):
        return FAILING, f"discovery failed: {listing['discovery_failed']}"
    counts = listing.get("listing") or {}
    listed = counts.get("listed")
    if not listed:
        return FAILING, "the venue listed nothing - a failed discovery, not an empty market"
    reasons = admission.get("excluded_by_reason") or {}
    named = ", ".join(f"{v} {k}" for k, v in sorted(reasons.items())) or "none excluded"
    if not admission:
        return PARTIAL, (f"{listed} instruments listed by the venue; the bot has not "
                         f"published an admission pass yet")
    detail = (f"{listed} listed by the venue, {admission.get('considered')} considered "
              f"this poll, {admission.get('admitted')} admitted, "
              f"{admission.get('excluded')} excluded - {named}")
    if age_s is not None and age_s > STALE_AFTER_S:
        return DEGRADED, f"{detail} (from a heartbeat {age_s:.0f}s old)"
    return OK, detail


def _one_universe_probe(segment: str) -> ProbeResult:
    state, detail = _universe_state(segment)
    return ProbeResult(state, detail, str(STATE_ROOT / segment / "universe.json"))


def probe_spot_universe_measured() -> ProbeResult:
    """SB-01: every symbol the spot feed carries resolves, with its reason."""
    return _one_universe_probe("spot")


def probe_dated_universe_measured() -> ProbeResult:
    """DB-01: the same for the dated contracts."""
    return _one_universe_probe("dated")


def probe_options_universe_measured() -> ProbeResult:
    """OB-01: the same for the options chain."""
    return _one_universe_probe("options")


def probe_universe_is_the_venues() -> ProbeResult:
    """BF-09: the universe comes from a venue listing call, never from a typed list."""
    met, missing = [], {}
    totals = {}
    for segment in SEGMENTS:
        listing = _listing(segment)
        if listing is None:
            missing[segment] = "no listing recorded"
            continue
        if listing.get("discovery_failed"):
            missing[segment] = "discovery failed"
            continue
        counts = (listing.get("listing") or {})
        totals[segment] = counts.get("listed") or 0
        if not totals[segment]:
            missing[segment] = "the venue listed nothing"
            continue
        met.append(segment)
    shown = ", ".join(f"{k} {v}" for k, v in sorted(totals.items())) or "nothing listed"
    return _fraction(met, missing, f"segments whose universe is the venue's ({shown})",
                     str(STATE_ROOT / "<segment>" / "universe.json"))


def probe_universe_breadth() -> ProbeResult:
    """RL-009 / RL-014: everything the venues list is scanned, not a chosen few."""
    listed = 0
    considered = 0
    per_segment = {}
    for segment in SEGMENTS:
        listing = _listing(segment) or {}
        counts = listing.get("listing") or {}
        payload, age_s = heartbeat(segment)
        admission = (payload or {}).get("universe_admission") or {}
        per_segment[segment] = (counts.get("listed") or 0,
                                admission.get("considered") or 0)
        listed += counts.get("listed") or 0
        considered += admission.get("considered") or 0
    proof = str(STATE_ROOT / "<segment>" / "universe.json")
    if not listed:
        return ProbeResult(NOT_MEASURED,
                           "no venue listing has been recorded by any bot", proof)
    detail = (f"{listed} instruments listed across four venues, {considered} carried a "
              f"frame on the last poll - "
              + ", ".join(f"{k} {l}/{c}" for k, (l, c) in sorted(per_segment.items())))
    if considered == 0:
        return ProbeResult(DEGRADED,
                           f"{detail} - the board is listed and nothing is being scanned",
                           proof)
    if considered < listed * 0.25:
        # Scanning a quarter of the board is the measured truth, not a pass. The
        # warm-up window and the sample floor are what hold the tail back.
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def probe_segments_captured() -> ProbeResult:
    """RL-006: spot, futures and options each have a bot on their own feed."""
    met, missing = [], {}
    venues = {}
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no bot"
            continue
        venues[segment] = payload.get("venue")
        if age_s is None or age_s > STALE_AFTER_S:
            missing[segment] = f"not polling ({age_s:.0f}s)" if age_s else "no timestamp"
            continue
        met.append(segment)
    shown = ", ".join(f"{k}={v}" for k, v in sorted(venues.items()))
    return _fraction(met, missing, f"segments live ({shown})",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_segment_bots() -> ProbeResult:
    """RL-019: four bots, each its own venue, brains and features - not one bot four times."""
    seen_venues, seen_brains = {}, {}
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no heartbeat"
            continue
        venue = payload.get("venue")
        brains = tuple(sorted((payload.get("brains") or {}).values()))
        if not venue or not brains:
            missing[segment] = "declares no venue or no brains"
            continue
        if venue in seen_venues.values() and segment != "perp":
            missing[segment] = f"shares venue {venue} with {seen_venues}"
            continue
        if brains in seen_brains.values():
            missing[segment] = "runs another segment's brains"
            continue
        seen_venues[segment] = venue
        seen_brains[segment] = brains
        met.append(segment)
    shown = ", ".join(f"{k}:{v}" for k, v in sorted(seen_venues.items()))
    return _fraction(met, missing, f"bots with their own venue and brains ({shown})",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_three_bots() -> ProbeResult:
    """RL-011 as amended by RL-023: three brains per bot - bull, bear, profit tail."""
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        brains = (payload or {}).get("brains") or {}
        if not brains:
            missing[segment] = "no brains declared"
            continue
        if sorted(brains) != ["bear", "bull", "profit_tail"]:
            missing[segment] = f"declares {sorted(brains)}"
            continue
        if len(set(brains.values())) != 3:
            missing[segment] = "two of its three brains are the same object"
            continue
        met.append(segment)
    return _fraction(met, missing, "bots carrying three distinct brains",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_uptime_continuous() -> ProbeResult:
    """RL-020: 24/7, and a restart is recorded rather than silently absorbed."""
    met, missing = [], {}
    restarts = {}
    for segment, (payload, age_s) in live_heartbeats().items():
        ledger = STATE_ROOT / segment / "restarts.ndjson"
        rows = tail_rows(ledger, rows=50) if ledger.is_file() else []
        restarts[segment] = sum(1 for r in rows
                                if r.get("event") == "supervisor_started")
        if payload is None:
            missing[segment] = "never started"
            continue
        if (STATE_ROOT / segment / "OFF").exists():
            missing[segment] = "deliberately OFF"
            continue
        if age_s is None or age_s > STALE_AFTER_S:
            missing[segment] = f"stopped {age_s / 60:.0f} min ago" if age_s else "no timestamp"
            continue
        met.append(segment)
    shown = ", ".join(f"{k} {v} starts" for k, v in sorted(restarts.items()))
    return _fraction(met, missing, f"bots up now (recent supervisor starts: {shown})",
                     str(STATE_ROOT / "<segment>" / "restarts.ndjson"))


# --- the learned brains ---------------------------------------------------

def _retrain_reports() -> dict:
    return {segment: read_json(LEARN_ROOT / f"{segment}-retrain.json")
            for segment in SEGMENTS}


def probe_training_set_built() -> ProbeResult:
    """LB-01: rows, day coverage and the label span, published rather than implied."""
    reports = {k: v for k, v in _retrain_reports().items() if v}
    proof = str(LEARN_ROOT / "<segment>-retrain.json")
    if not reports:
        return ProbeResult(NOT_MEASURED, "no retrain report exists", proof)
    parts, complete = [], []
    for segment, report in sorted(reports.items()):
        dataset = report.get("dataset") or {}
        if report.get("skipped"):
            parts.append(f"{segment}: {report['skipped']}")
            continue
        rows = dataset.get("rows")
        days = dataset.get("n_days")
        if rows and days:
            complete.append(segment)
        parts.append(f"{segment}: {rows} rows over {days} days, "
                     f"{dataset.get('symbols')} symbols")
    detail = " · ".join(parts)
    if not complete:
        return ProbeResult(NOT_MEASURED,
                           f"no segment has a built training set - {detail}", proof)
    if len(complete) < 2:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def _model_records() -> list:
    directory = MODEL_ROOT / "models"
    if not directory.is_dir():
        return []
    return [read_json(path) for path in sorted(directory.glob("*.json"))]


def probe_model_registered_with_loss() -> ProbeResult:
    """LB-02: every registered model names its trial and the score it earned."""
    records = [r for r in _model_records() if r]
    proof = str(MODEL_ROOT / "models")
    if not records:
        return ProbeResult(NOT_MEASURED, "the model registry is empty", proof)
    with_trial = [r for r in records if r.get("trial_id") is not None]
    with_score = [r for r in records
                  if (r.get("metrics") or {}).get("accuracy") is not None
                  or (r.get("metrics") or {}).get("sharpe") is not None
                  or (r.get("metrics") or {}).get("pinball_loss") is not None]
    majority = [r for r in records
                if (r.get("metrics") or {}).get("reproduced_majority_class")]
    aliases = read_json(MODEL_ROOT / "aliases.json") or {}
    detail = (f"{len(records)} registered, {len(with_trial)} name their trial, "
              f"{len(with_score)} carry an out-of-fold score, {len(majority)} are "
              f"reported as reproducing the majority class, {len(aliases)} champion "
              f"aliases")
    if len(with_trial) < len(records) or len(with_score) < len(records):
        return ProbeResult(DEGRADED,
                           f"{detail} - a model without its trial and its loss is an "
                           f"artefact nobody can hold to account", proof)
    return ProbeResult(OK, detail, proof)


def probe_brains_are_learned() -> ProbeResult:
    """LB-04 / RL-026: the bot decides from a trained model, or it says it does not."""
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no heartbeat"
            continue
        if payload.get("learned") and payload.get("model_version"):
            warm_up = payload.get("warm_up") or {}
            if warm_up and not warm_up.get("symbols_ready"):
                # Deciding from a model and being ABLE to decide are different
                # facts. A learned bot inside its bar window has not failed - it
                # is counted apart so nobody reads "0 proposals" as a verdict on
                # the model.
                missing[segment] = (
                    f"learned on {payload['model_version'][:8]} and warming up: "
                    f"{warm_up.get('deepest_bars', 0)}/{warm_up.get('bars_required', 0)}"
                    f" bars, ~{warm_up.get('minutes_to_first_decision', 0)} min to its "
                    f"first decision")
                continue
            met.append(segment)
            continue
        refused = (payload.get("champion_refused")
                   or (payload.get("extra") or {}).get("champion_refused"))
        if refused:
            missing[segment] = f"champion refused: {refused.get('why', '')[:60]}"
        else:
            missing[segment] = "rule brains, no champion loaded"
    return _fraction(met, missing, "bots deciding from a registered model",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_beliefs_carry_provenance() -> ProbeResult:
    """LB-03 / RL-027: a claim carries what produced it and when it expires.

    Measured on the journal rather than on the class, because a belief that is
    constructible in a test and never emitted by the live loop is a type, not a
    capability.

    **A belief rides a PROPOSAL, not a decline**, so an absence of beliefs in a
    window of declines says nothing about whether the capability works. That
    distinction is the whole point of this probe: the first version reported
    NOT MEASURED across all four bots while the perp bot was emitting beliefs
    correctly, because its tail happened to hold nothing but declines - and
    "the brains emit no belief" and "no brain proposed anything recently" are
    different facts about the system.
    """
    # **The denominator is the LEARNED bots, not all four.** A rule brain makes
    # no edge claim (RL-025) and emits no belief, so counting it as a bot that
    # failed to emit one would report a design decision as a defect - and would
    # keep reporting it after the capability was working everywhere it applies.
    learned = [segment for segment, (payload, age_s) in live_heartbeats().items()
               if (payload or {}).get("learned")]
    if not learned:
        return ProbeResult(
            NOT_MEASURED,
            "no learned brain is deployed, and a rule brain emits no belief by "
            "design (RL-025) - there is nothing here to measure yet",
            str(STATE_ROOT / "<segment>" / "heartbeat.json"))

    met, missing = [], {}
    for segment in learned:
        path = latest_journal(segment, "decisions")
        rows = tail_rows(path, rows=4000, max_bytes=16_000_000) if path else []
        if not rows:
            missing[segment] = "no decisions journalled"
            continue
        proposals = 0
        with_belief = 0
        with_provenance = 0
        for row in rows:
            for side in ("bull", "bear"):
                block = ((row.get("evidence") or {}).get(side) or {})
                if block.get("outcome") not in ("PROPOSAL", "PROPOSE"):
                    continue
                proposals += 1
                belief = (block.get("evidence") or {}).get("belief") or {}
                if not belief:
                    continue
                with_belief += 1
                # The expiry contract is the half-life plus when it was held;
                # `describe()` publishes both, and a belief may not be
                # constructed without provenance at all.
                if (belief.get("provenance") and belief.get("half_life_ns")
                        and belief.get("held_at_ns") is not None):
                    with_provenance += 1
        if not proposals:
            missing[segment] = "no proposal in the journal window to carry one"
            continue
        if not with_belief:
            missing[segment] = f"{proposals} proposals and none carries a belief"
            continue
        if with_provenance < with_belief:
            missing[segment] = (f"{with_belief - with_provenance} beliefs carry no "
                                f"provenance or no half-life")
            continue
        met.append(f"{segment} ({with_provenance}/{proposals})")
    shown = ", ".join(met) or "none"
    detail = (f"{len(met)}/{len(learned)} LEARNED bots whose proposals carry a "
              f"belief with provenance and a half-life ({shown})"
              + ("; " + "; ".join(f"{k}: {v}" for k, v in sorted(missing.items()))
                 if missing else "")
              + f"; {len(SEGMENTS) - len(learned)} bot(s) run rule brains, which "
                f"emit no belief by design (RL-025)")
    proof = str(STATE_ROOT / "<segment>" / "decisions-*.ndjson")
    if len(met) == len(learned):
        return ProbeResult(OK, detail, proof)
    if met:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


def probe_calibration_updates_live() -> ProbeResult:
    """LB-05: what the live loop fitted, kept apart from what the retrainer set."""
    met, missing = [], {}
    for segment in SEGMENTS:
        path = STATE_ROOT / segment / "calibration.json"
        record = read_json(path)
        payload, _ = heartbeat(segment)
        if record is None:
            missing[segment] = "no live calibration file"
            continue
        bins = record.get("bins") or {}
        observed = sum(int(v[0]) for v in bins.values() if isinstance(v, list) and v)
        live_fitted = (payload or {}).get("live_fitted")
        if not observed:
            missing[segment] = "the calibration file holds no observed outcome"
            continue
        if live_fitted is None:
            missing[segment] = (f"{observed} outcomes recorded, and the heartbeat "
                                f"reports no realised coverage")
            continue
        met.append(segment)
    return _fraction(met, missing,
                     "bots whose live loop reports its own realised coverage",
                     str(STATE_ROOT / "<segment>" / "calibration.json"))


def probe_profit_tail_is_learned() -> ProbeResult:
    """LB-06: expectancy from fitted quantiles rather than a volatility proxy."""
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        if payload is None:
            missing[segment] = "no heartbeat"
            continue
        version = payload.get("tail_model_version")
        name = (payload.get("brains") or {}).get("profit_tail", "")
        if version:
            met.append(segment)
            continue
        missing[segment] = f"deterministic policy in use ({name or 'unnamed'})"
    aliases = read_json(MODEL_ROOT / "aliases.json") or {}
    tails = [k for k in aliases if "profit-tail" in k]
    return _fraction(met, missing,
                     f"bots timing from a fitted tail model "
                     f"({len(tails)} tail champions registered)",
                     str(STATE_ROOT / "<segment>" / "heartbeat.json"))


def probe_capital_and_pnl_reported() -> ProbeResult:
    """BF-11 / BF-12 / RL-028 / RL-029: capital used and P&L made, in USDT.

    All three denominators or none: a return quoted against an unnamed capital
    base is a number chosen to flatter, and on the same trades peak-at-risk,
    turnover and bankroll differ by orders of magnitude.

    Unconvertible fills are counted and named rather than dropped (RL-029). A
    bot whose fills are ALL unconvertible reports that, because a zero P&L from
    nothing counted looks exactly like a zero P&L from a flat bot.
    """
    from segment.capital_accounting import account_for_fills, bankroll_of

    met, missing = [], {}
    lines = []
    for segment in SEGMENTS:
        fills = []
        for path in journal_paths(segment, "fills"):
            fills.extend(tail_rows(path, rows=200_000, max_bytes=64_000_000))
        if not fills:
            missing[segment] = "no fills journalled"
            continue
        report = account_for_fills(fills, segment=segment,
                                   bankroll_usdt=bankroll_of(segment))
        lines.append(f"{segment} peak {float(report.peak_at_risk_usdt):,.0f} / "
                     f"turnover {float(report.turnover_usdt):,.0f} / "
                     f"P&L {float(report.realised_pnl_usdt):+,.2f} USDT")
        if report.converted_fills == 0:
            missing[segment] = (f"all {report.unconvertible_fills} fills "
                                f"unconvertible ({', '.join(report.unconvertible_currencies)})")
            continue
        if report.closes == 0:
            missing[segment] = "has opened but closed nothing, so no return exists yet"
            continue
        met.append(segment)

    page, _, path = _board_fresh("segment-bots.html")
    if page is not None and "peak at risk USDT" not in page:
        return ProbeResult(DEGRADED,
                           "the accounting exists and the wall does not render it",
                           str(path))
    return _fraction(met, missing,
                     "bots reporting capital and P&L in USDT (" + " · ".join(lines) + ")",
                     str(STATE_ROOT / "<segment>" / "fills-*.ndjson"))


def probe_champion_reload_current() -> ProbeResult:
    """LB-09: the bot decides from the champion registered NOW, not at its start.

    `bot_registry` resolves the champion once, at process start, while the
    retrainer refits every four hours and the bots run 24/7. Every model
    registered between two restarts is therefore registered and unused, and the
    heartbeat publishes the version the bot LOADED - which is why nothing said
    so. This compares that against the alias on disk.
    """
    aliases = read_json(MODEL_ROOT / "aliases.json") or {}
    proof = str(MODEL_ROOT / "aliases.json")
    if not aliases:
        return ProbeResult(NOT_MEASURED, "no champion alias is registered", proof)
    met, missing = [], {}
    for segment, (payload, age_s) in live_heartbeats().items():
        registered = aliases.get(f"{segment}-direction-champion")
        running = (payload or {}).get("model_version")
        if payload is None:
            missing[segment] = "no heartbeat"
            continue
        if not registered:
            missing[segment] = "no champion registered for this segment"
            continue
        if not running:
            # Refused or never loaded. `probe_brains_are_learned` owns that
            # distinction; here the fact is that the registered model is unused.
            missing[segment] = f"registered {registered[:8]} and the bot runs no model"
            continue
        if running != registered:
            missing[segment] = (f"running {running[:8]}, registered "
                                f"{registered[:8]} - superseded")
            continue
        met.append(segment)
    return _fraction(met, missing, "bots deciding from the currently registered champion",
                     proof)


def probe_axis_tests_run() -> ProbeResult:
    """LB-07: L3, L4, R9 and realised coverage run against the live brains."""
    reports = {k: v for k, v in _retrain_reports().items() if v}
    proof = str(LEARN_ROOT / "<segment>-retrain.json")
    if not reports:
        return ProbeResult(NOT_MEASURED, "no retrain report exists to carry axis probes",
                           proof)
    ran, verdicts, failing = [], {}, []
    for segment, report in sorted(reports.items()):
        probes = ((report.get("axis_probes") or {}).get("probes")) or []
        if not probes:
            continue
        ran.append(segment)
        for probe in probes:
            key = f"{probe.get('axis_test')}"
            verdicts[key] = probe.get("verdict")
            if probe.get("verdict") == "FAIL":
                failing.append(f"{segment} {key}")
    if not ran:
        return ProbeResult(NOT_MEASURED,
                           "retrain reports exist and none carries an axis probe run",
                           proof)
    detail = (f"{', '.join(ran)} ran the axis probes: "
              + ", ".join(f"{k}={v}" for k, v in sorted(verdicts.items())))
    # L6 out-of-regime is expected to FAIL while the record is one regime, and it is
    # reported rather than omitted. That is the design, so a FAIL here is not a
    # broken probe - it is the honest verdict the row asked for.
    if failing:
        return ProbeResult(PARTIAL, f"{detail} - failing: {', '.join(failing)}", proof)
    if len(ran) < 2:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def probe_retrainer_running() -> ProbeResult:
    """LB-08: a refit happens with nothing started by hand, and the board names it."""
    running, pids = _process_running("retrain_supervisor.sh")
    reports = {k: v for k, v in _retrain_reports().items() if v}
    proof = str(LEARN_ROOT)
    if not reports:
        return ProbeResult(NOT_MEASURED,
                           f"no retrain report on disk (supervisor: {pids})", proof)
    ages = {}
    for segment in reports:
        path = LEARN_ROOT / f"{segment}-retrain.json"
        ages[segment] = (time.time() - path.stat().st_mtime) / 3600
    newest = min(ages.values())
    detail = (f"supervisor {'running' if running else 'NOT running'} ({pids}), "
              f"newest refit {newest:.1f}h ago, "
              + ", ".join(f"{k} {v:.0f}h" for k, v in sorted(ages.items())))
    if not running:
        return ProbeResult(DEGRADED,
                           f"{detail} - refits happened and nothing is scheduled now",
                           proof)
    if newest > 8:
        return ProbeResult(DEGRADED, f"{detail} - the cadence has stopped", proof)
    return ProbeResult(OK, detail, proof)


def probe_axis_verdicts_present() -> ProbeResult:
    """RL-004: every module answers the three axes, and a cited artifact exists."""
    proof = str(AXIS_VERDICTS)
    record = read_json(AXIS_VERDICTS)
    if record is None:
        return ProbeResult(NOT_MEASURED, "no axis verdict file exists", proof)
    from integrity.axis_verdicts import assess as assess_axis_verdicts
    coverage = assess_axis_verdicts(REPO, AXIS_VERDICTS)
    detail = (f"{len(coverage.verdicted)}/{coverage.total} modules carry a verdict, "
              f"{len(coverage.unverdicted)} do not, {len(coverage.incomplete)} answer "
              f"only some axes, {len(coverage.missing_evidence)} cite an artifact that "
              f"is not on disk, {len(coverage.failing)} carry an explicit FAIL, "
              f"{len(coverage.orphaned)} name a module that no longer exists")
    # A FAIL is an answer and does not degrade this probe - an unanswered axis and a
    # verdict citing a test nobody wrote are what it exists to catch.
    if coverage.unverdicted or coverage.incomplete or coverage.missing_evidence:
        return ProbeResult(DEGRADED, detail, proof)
    return ProbeResult(OK, detail, proof)


def probe_intelligence_axes() -> ProbeResult:
    """RL-010 / RL-013: real learning, reasoning and depth, judged on running state.

    Three measurements, not an opinion: are the deployed brains trained, did the
    axis probes run against them, and does the live loop fit anything itself.
    """
    learned = probe_brains_are_learned()
    axes = probe_axis_tests_run()
    calibration = probe_calibration_updates_live()
    parts = [f"learned brains: {learned.state} ({learned.detail})",
             f"axis probes: {axes.state} ({axes.detail[:80]})",
             f"live fitting: {calibration.state} ({calibration.detail[:80]})"]
    detail = " · ".join(parts)
    proof = "statuswall.segment_probes: brains, axis probes and calibration"
    states = {learned.state, axes.state, calibration.state}
    if states == {OK}:
        return ProbeResult(OK, detail, proof)
    if OK in states or PARTIAL in states:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


# --- governance -----------------------------------------------------------

def probe_permanent_rules_present() -> ProbeResult:
    """RL-001: the permanent rules exist where every session reads them."""
    proof = str(PERMANENT_RULES)
    try:
        text = PERMANENT_RULES.read_text(encoding="utf-8")
    except OSError:
        return ProbeResult(NOT_MEASURED, "no permanent rules file on disk", proof)
    numbered = [line for line in text.splitlines()
                if line.startswith("## Rule ")]
    if not numbered:
        return ProbeResult(DEGRADED, "the file exists and states no numbered rule", proof)
    return ProbeResult(OK, f"{len(numbered)} numbered rules, "
                           f"{len(text.splitlines())} lines", proof)


def probe_build_order_current() -> ProbeResult:
    """RL-007 / RL-015: the order is followed - no slice starts before the one above."""
    from plan.master_plan import read_master_plan
    slices = read_master_plan(SPINE)
    proof = str(SPINE)
    empties = [s.key for s in slices if not s.rows]
    populated = [s.key for s in slices if s.rows]
    detail = (f"{len(slices)} slices, {len(populated)} carry rows "
              f"({', '.join(populated)}), {len(empties)} await inventory review "
              f"({', '.join(empties) or 'none'})")
    # The order is a property of the document: a populated slice below an empty one
    # means work started out of order. That is a real state and it is reported.
    first_empty = next((i for i, s in enumerate(slices) if not s.rows), None)
    if first_empty is not None:
        after = [s.key for s in slices[first_empty + 1:] if s.rows]
        if after:
            return ProbeResult(PARTIAL,
                               f"{detail} - rows exist below an empty slice: "
                               f"{', '.join(after)}", proof)
    return ProbeResult(OK, detail, proof)


def probe_probes_implemented() -> ProbeResult:
    """BF-10, measuring itself: every probe a row names exists, or is reported absent."""
    from plan.master_plan import read_master_plan
    from plan.rulings import read_rulings
    from statuswall.ruling_conformance import PROBES

    named: dict = {}
    for plan_slice in read_master_plan(SPINE):
        for row in plan_slice.rows:
            if row.probe:
                named.setdefault(row.probe, []).append(row.id)
    for ruling in read_rulings(REGISTER):
        if ruling.probe:
            named.setdefault(ruling.probe, []).append(ruling.id)

    registry = {**PROBES, **SEGMENT_PROBES}
    missing = sorted(k for k in named if k not in registry)
    proof = "statuswall.ruling_conformance.PROBES + statuswall.segment_probes.SEGMENT_PROBES"
    detail = (f"{len(named) - len(missing)}/{len(named)} named probes implemented"
              + (f"; absent: {', '.join(missing[:6])}" if missing else ""))
    if missing:
        return ProbeResult(DEGRADED, detail, proof)
    return ProbeResult(OK, detail, proof)


SEGMENT_PROBES = {
    "probe_segment_engine_running": probe_segment_engine_running,
    "probe_perp_engine_running": probe_perp_engine_running,
    "probe_live_feed_fresh": probe_live_feed_fresh,
    "probe_intraday_data_lag": probe_intraday_data_lag,
    "probe_segment_features_current": probe_segment_features_current,
    "probe_perp_features_current": probe_perp_features_current,
    "probe_segment_brains_reason": probe_segment_brains_reason,
    "probe_perp_bull_agent_reasons": probe_perp_bull_agent_reasons,
    "probe_perp_bear_agent_reasons": probe_perp_bear_agent_reasons,
    "probe_segment_arbiter_decides": probe_segment_arbiter_decides,
    "probe_perp_arbiter_decides": probe_perp_arbiter_decides,
    "probe_profit_tail_authority": probe_profit_tail_authority,
    "probe_perp_bands_journalled": probe_perp_bands_journalled,
    "probe_perp_risk_gate_refuses": probe_perp_risk_gate_refuses,
    "probe_perp_band_performance": probe_perp_band_performance,
    "probe_perp_promotion_gate": probe_perp_promotion_gate,
    "probe_segment_tiles_measured": probe_segment_tiles_measured,
    "probe_perp_tile_measured": probe_perp_tile_measured,
    "probe_board_generated": probe_board_generated,
    "probe_spot_universe_measured": probe_spot_universe_measured,
    "probe_dated_universe_measured": probe_dated_universe_measured,
    "probe_options_universe_measured": probe_options_universe_measured,
    "probe_universe_is_the_venues": probe_universe_is_the_venues,
    "probe_universe_breadth": probe_universe_breadth,
    "probe_segments_captured": probe_segments_captured,
    "probe_segment_bots": probe_segment_bots,
    "probe_three_bots": probe_three_bots,
    "probe_uptime_continuous": probe_uptime_continuous,
    "probe_training_set_built": probe_training_set_built,
    "probe_model_registered_with_loss": probe_model_registered_with_loss,
    "probe_brains_are_learned": probe_brains_are_learned,
    "probe_beliefs_carry_provenance": probe_beliefs_carry_provenance,
    "probe_calibration_updates_live": probe_calibration_updates_live,
    "probe_profit_tail_is_learned": probe_profit_tail_is_learned,
    "probe_capital_and_pnl_reported": probe_capital_and_pnl_reported,
    "probe_champion_reload_current": probe_champion_reload_current,
    "probe_axis_tests_run": probe_axis_tests_run,
    "probe_retrainer_running": probe_retrainer_running,
    "probe_axis_verdicts_present": probe_axis_verdicts_present,
    "probe_intelligence_axes": probe_intelligence_axes,
    "probe_permanent_rules_present": probe_permanent_rules_present,
    "probe_build_order_current": probe_build_order_current,
    "probe_probes_implemented": probe_probes_implemented,
}


def main(argv=None) -> int:
    """Run every probe here and print what it measured. No page, no state cached."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="substring filter on probe name")
    args = parser.parse_args(argv)
    for name, probe in sorted(SEGMENT_PROBES.items()):
        if args.only and args.only not in name:
            continue
        try:
            result = probe()
        except Exception as failure:                     # noqa: BLE001
            result = ProbeResult(NOT_MEASURED,
                                 f"raised {type(failure).__name__}: {failure}", name)
        print(json.dumps({"probe": name, "state": result.state,
                          "detail": result.detail, "proof": result.proof}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
