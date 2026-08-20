"""One board row per human ruling, measured or explicitly not measured.

A ruling honoured in a document and not in running code is the failure this
board exists to catch. RL-011 was given 2026-08-03 - three independent bots with
their own data, features and architecture - and measured at 2 of 22 built on
2026-08-17. Nothing on any board said so, because no board knew the ruling
existed.

Two numbers per ruling, answering different questions:

  COVERAGE  does the PLAN have rows for this, per its scope? Four of four
            segments, twelve of twelve brains, one row anywhere for shared.
  STATE     does the RUNNING SYSTEM do it? Measured by the named probe.

A ruling can be fully covered and still fail its probe. That is not a
contradiction - it is the difference between planned and true, and collapsing
the two is how a plan starts vouching for a system.

A ruling naming no probe renders NOT MEASURED and never green (Rule 8). A probe
that raises measures nothing rather than taking the board down: a board that
fails to render tells the reader less than a board with one dark row.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import time
from pathlib import Path

from paper.forward_journal import read_heartbeat
from plan.master_plan import PlanSlice, read_master_plan
from plan.rulings import Ruling
from plan.scope_coverage import Coverage, cover_ruling
from perp.tradable_universe import REPORT_PATH as PERP_UNIVERSE_REPORT
from statuswall.evidence import (
    DEGRADED, FAILING, NOT_MEASURED, OK, PARTIAL, ProbeResult, STATE_LABEL,
)
from statuswall.staleness_banner import render_staleness_banner

REPO = Path(__file__).resolve().parents[2]
SPINE = REPO / "docs" / "AJIT-MASTER-PLAN.md"
REGISTER = REPO / "docs" / "rulings.json"


# --- probes ---------------------------------------------------------------
#
# Each measures the RUNNING SYSTEM, never whether a test passed. A test proves
# the logic; a probe answers whether the thing is true on this box right now.

def probe_store_offloaded(bucket: str = "gs://capture-raw-data4134") -> ProbeResult:
    """RL-020: the observed record must exist somewhere other than this disk."""
    try:
        listing = subprocess.run(
            ["gcloud", "storage", "ls", f"{bucket}/store/funding/"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as failure:
        return ProbeResult(NOT_MEASURED, f"could not reach the bucket: {failure!r}",
                           f"gcloud storage ls {bucket}/store/funding/")
    if listing.returncode != 0:
        return ProbeResult(
            DEGRADED,
            "the store prefix is absent from the bucket - the observed funding "
            "record exists on one disk only",
            f"gcloud storage ls {bucket}/store/funding/ -> exit {listing.returncode}")
    lines = [l for l in listing.stdout.splitlines() if l.strip()]
    if not lines:
        return ProbeResult(DEGRADED, "the store prefix exists and is empty",
                           f"{bucket}/store/funding/ listed 0 objects")
    return ProbeResult(OK, f"{len(lines)} object(s) under store/funding",
                       f"{bucket}/store/funding/ listed {len(lines)} object(s)")


def probe_memory_reachable() -> ProbeResult:
    """RL-016: the memory carrying 'interview, do not assume' must be loadable.

    Memory is keyed by working directory. A session started from `/` reads a
    different key from one started at home, and on 2026-08-17 that key was
    empty while twelve real memories sat under the other one.
    """
    generic = Path.home() / ".claude" / "projects" / "-" / "memory"
    named = Path.home() / ".claude" / "projects" / "-home-anushadudekula71" / "memory"
    if not named.is_dir():
        return ProbeResult(NOT_MEASURED, "no project memory directory found", str(named))
    generic_count = len(list(generic.glob("*.md"))) if generic.is_dir() else 0
    named_count = len(list(named.glob("*.md")))
    if generic_count < named_count:
        return ProbeResult(
            DEGRADED,
            f"a session started from / sees {generic_count} of {named_count} memories",
            f"{generic} vs {named}")
    return ProbeResult(OK, f"{named_count} memories reachable from either key",
                       f"{generic} -> {generic_count}, {named} -> {named_count}")


SEGMENT_STATE_ROOT = Path.home() / "capture" / "segment"
SEGMENTS = ("perp", "spot", "dated", "options")
RETIRED_PLUMBING = Path.home() / "capture" / "paper" / "forward"


def probe_paper_engine_running(
    state_root: Path = SEGMENT_STATE_ROOT,
) -> ProbeResult:
    """RL-005, RL-017, RL-020: paper trading is the experimentation ground.

    **Repointed 2026-08-19 at the engines that are actually running.** This
    measured `capture/paper/forward` - the `plumbing-momentum` engine RETIRED
    under RL-025 on 2026-08-18 - and so reported DEGRADED on a heartbeat that
    was 38 hours old and deliberately never coming back. A tile that is
    permanently red about something switched off on purpose is one everybody
    learns to skip, which is how the next real failure gets missed.

    The four segment bots are the paper engines now, and they satisfy the row's
    acceptance more strictly than the engine that was retired: they never read
    the parquet archive at all (RL-024), so "no archived event is traded on
    resume" holds by construction, and a restart is observable - each one
    journals what it recovered from its own fill journal before it polls.
    """
    live, recovered, missing = [], [], {}
    for segment in SEGMENTS:
        beat_path = state_root / segment / "heartbeat.json"
        try:
            beat = json.loads(beat_path.read_text(encoding="utf-8"))
            age_s = (time.time_ns() - int(beat["written_at_ns"])) / 1e9
        except (OSError, ValueError, KeyError, TypeError):
            missing[segment] = "no heartbeat"
            continue
        if age_s > 300:
            missing[segment] = f"heartbeat {age_s / 60:.0f} min old"
            continue
        live.append(segment)
        # A restart that re-primed from the archive would show up as a bot that
        # traded before it had watched any live ticks. What it does instead is
        # journal the positions it rebuilt from its own fills.
        if _recovered_on_restart(state_root / segment / "engine.log"):
            recovered.append(segment)

    detail = (f"{len(live)}/4 segment paper engines polling"
              + (f" ({', '.join(live)})" if live else "")
              + f", {len(recovered)} journalled a position recovery on restart"
              + ("; " + "; ".join(f"{k}: {v}" for k, v in sorted(missing.items()))
                 if missing else "")
              + ("; the retired plumbing-momentum engine is not counted (RL-025)"
                 if (RETIRED_PLUMBING / "heartbeat.json").is_file() else ""))
    proof = str(state_root / "<segment>" / "heartbeat.json")
    if not live:
        return ProbeResult(NOT_MEASURED, detail, proof)
    if len(live) < len(SEGMENTS):
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(OK, detail, proof)


def _recovered_on_restart(log: Path, window_bytes: int = 2_000_000) -> bool:
    """Whether this bot rebuilt open positions from its journal on its last start.

    Read from the engine log rather than the decision journal, and bounded: the
    decision journals reached 4.1 GB in a day, and a recovery line sits at a
    restart point rather than at the end, so a tail of the journal would report
    a recovery that happened as though it had not.
    """
    try:
        size = log.stat().st_size
        with log.open("rb") as handle:
            handle.seek(max(0, size - window_bytes))
            return b"recovered" in handle.read()
    except OSError:
        return False


# **The archive walk, budgeted and taken once.** Measured 2026-08-19: a plain
# `rglob("*.parquet")` over the bars store did not finish in 120 seconds, and the
# plan board calls this probe once per row that names it - so a single pass spent
# its whole cadence counting files, in uninterruptible IO, and the plan board sat
# 43 hours stale while the loop above it looked healthy.
#
# Two changes, both of which keep the number honest. The walk is CACHED for the
# life of the process, because every row in one pass is describing the same
# archive at the same instant. And it is BUDGETED: past the deadline it returns
# what it counted with `complete=False`, and the caller renders that as `N+`
# rather than as N. A lower bound is a measurement; a count that never returns is
# not, and it takes the board with it.
_FRAGMENT_COUNT: dict = {}


def count_store_fragments(dataset: Path,
                          budget_s: float = 20.0) -> tuple[int, bool]:
    """(fragments seen, whether the walk finished). Cached per process."""
    key = str(dataset)
    if key in _FRAGMENT_COUNT:
        return _FRAGMENT_COUNT[key]
    if not dataset.is_dir():
        _FRAGMENT_COUNT[key] = (0, True)
        return _FRAGMENT_COUNT[key]
    deadline = time.monotonic() + budget_s
    seen, complete = 0, True
    for _ in dataset.rglob("*.parquet"):
        seen += 1
        # Checked per 500 rather than per file: the clock read is cheap, but not
        # cheaper than the stat it would be guarding on a fast directory.
        if seen % 500 == 0 and time.monotonic() > deadline:
            complete = False
            break
    _FRAGMENT_COUNT[key] = (seen, complete)
    return _FRAGMENT_COUNT[key]


STORE_ROOT = Path.home() / "capture" / "store"

# The migration renames the old copy aside rather than deleting it, so a retired
# copy is EVIDENCE the migration finished and must never be counted against the
# live dataset. `quarantine` holds rows the builders refused and is not a dataset
# anything reads.
_NOT_A_LIVE_DATASET = (".legacy-symbol-layout", ".hourly-building")


def _live_datasets(store_root: Path) -> list[Path]:
    """The dataset directories a reader would actually open."""
    if not store_root.is_dir():
        return []
    return sorted(
        d for d in store_root.iterdir()
        if d.is_dir() and d.name != "quarantine"
        and not any(d.name.endswith(suffix) for suffix in _NOT_A_LIVE_DATASET))


# The five probes wrapped in `measured_periodically`. Each writes its cache entry
# on its FIRST success, so the set of entries is a record of which ones have ever
# returned - which is why an absent cache directory was the proof that no
# expensive probe had completed since 2026-08-17.
EXPENSIVE_PROBES = (
    "probe_bitemporal_store",
    "probe_bar_price_validity",
    "probe_clock_gated_access",
    "probe_promotion_readiness",
    "probe_consolidated_price",
)

PROBE_CACHE = Path.home() / "capture" / "boards" / "probe-cache"
WALL_PAGE = Path.home() / "research" / "dashboard" / "status-wall.html"


def probe_wall_pass_completes(cache_dir: Path = PROBE_CACHE,
                              page: Path = WALL_PAGE) -> ProbeResult:
    """SL-18, RL-033: the wall finishes a pass, rather than never returning.

    **A board that never returns reports nothing**, and that is what this
    measures rather than how fast the pass was. `status-wall.html` last completed
    2026-08-17 13:38 while the generator started a fresh pass every five minutes;
    each one walked 208,880 bars fragments, reached 7 GB, and was killed. Nothing
    said so, because a stale page looks exactly like a fresh one.

    The evidence is the probe cache. `measured_periodically` writes an entry on a
    probe's FIRST success and never on a failure, so the entries are a record of
    which expensive probes have ever completed - and the absence of the directory
    is a stronger statement than any timing.
    """
    cached = sorted(p.stem for p in cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    met = [name for name in EXPENSIVE_PROBES if name in cached]
    missing = {name: "has never completed a pass"
               for name in EXPENSIVE_PROBES if name not in cached}
    proof = str(cache_dir)
    if not met:
        return ProbeResult(
            NOT_MEASURED,
            f"no expensive probe has ever completed a pass - the cache at "
            f"{cache_dir.name} holds nothing, and it is written on first success",
            proof)
    if not page.is_file():
        return ProbeResult(PARTIAL,
                           f"{len(met)}/{len(EXPENSIVE_PROBES)} expensive probes "
                           f"cached, but {page.name} has never been written", proof)
    detail = (f"{len(met)}/{len(EXPENSIVE_PROBES)} expensive probes have completed "
              f"a pass")
    if missing:
        detail += " - " + "; ".join(f"{k}: {v}" for k, v in list(missing.items())[:3])
        return ProbeResult(PARTIAL, detail, proof)
    age_h = (time.time() - page.stat().st_mtime) / 3600
    detail += f"; {page.name} written {age_h:.1f}h ago"
    return ProbeResult(OK if age_h < 6 else DEGRADED, detail, proof)


def probe_hour_pruning_enabled(store_root: Path = STORE_ROOT) -> ProbeResult:
    """SL-16, RL-032: hour pruning is off for any dataset still part-migrated.

    `_is_partitioned_by_hour` disables pruning for a WHOLE dataset while a
    top-level `symbol=` directory remains, and that is deliberate: hive
    partitioning gives a legacy part a NULL hour, `NULL >= '<hour>'` is null
    rather than true, and every legacy row would vanish from a bounded read with
    no error. Correct and slow beats fast and wrong.

    So a dataset in both layouts is a MIGRATION THAT HAS NOT FINISHED, not a
    dataset that is merely slow, and this says which ones and how much of the old
    layout is left. Measured 2026-08-19: `funding` held 16 hour directories
    beside 1,271 legacy symbol ones and `option_chain` 16 beside 1,520, so
    SL-15's speed-up was switched off for the two largest datasets while only
    `bars_60000000000ns` had ever been migrated.
    """
    datasets = _live_datasets(store_root)
    if not datasets:
        return ProbeResult(NOT_MEASURED, "no dataset under the store to measure",
                           str(store_root))
    pruned, missing = [], {}
    for dataset in datasets:
        legacy = sum(1 for _ in dataset.glob("symbol=*"))
        if legacy:
            plural = "y remains" if legacy == 1 else "ies remain"
            missing[dataset.name] = f"{legacy} legacy symbol director{plural}"
        else:
            pruned.append(dataset.name)
    return _fraction_of(pruned, missing, len(datasets),
                        "datasets with hour pruning enabled", str(store_root))


def probe_sealed_hour_compacted(
    dataset: Path = STORE_ROOT / "funding",
) -> ProbeResult:
    """SL-17, RL-032: a sealed hour holds one part per venue, not one per symbol.

    Measured 2026-08-19: one sealed funding hour held **1,892 fragments for
    26,015 rows and 16.2 MiB** - 13.8 rows and 8.8 KiB per file - at 29 ms each
    to open, so the whole 850 MB dataset cost 30.6 minutes to read and grew by
    ~1,021 files an hour. Column pushdown filters rows; it cannot prune files.

    **The newest hour is never judged.** It is the one still being written and
    compaction only touches sealed hours, so measuring it would report every
    healthy store as failing, once an hour.
    """
    if not dataset.is_dir():
        return ProbeResult(NOT_MEASURED, f"no {dataset.name} dataset to measure",
                           str(dataset))
    hours = sorted(d.name for d in dataset.iterdir()
                   if d.is_dir() and d.name.startswith("availability_hour="))
    if len(hours) < 2:
        return ProbeResult(NOT_MEASURED,
                           "no sealed hour yet - the only hour is still being written",
                           str(dataset))

    compacted, missing = [], {}
    for name in hours[:-1]:
        hour = dataset / name
        parts = sorted(hour.rglob("*.parquet"))
        venues: set[str] = set()
        for part in parts[:8]:      # a compacted hour has a handful; do not walk 1,892
            venues |= _venues_in(part)
        label = name.split("=", 1)[1]
        if not parts:
            continue
        if len(parts) <= max(len(venues), 1):
            compacted.append(f"{label}: {len(parts)} part(s), {len(venues)} venue(s)")
        else:
            missing[label] = (f"{len(parts)} parts for {len(venues)} venue(s) - "
                              f"still one per symbol")
    if not compacted and not missing:
        return ProbeResult(NOT_MEASURED, "no sealed hour holds any part",
                           str(dataset))
    return _fraction_of(compacted, missing, len(compacted) + len(missing),
                        f"sealed {dataset.name} hours compacted to one part per venue",
                        str(dataset))


def _venues_in(part: Path) -> set[str]:
    """Which venues a part holds, from its BODY rather than its name.

    **A converted part is named by content hash** - `part-08ab0c8032ed701f` -
    because its snapshot id is derived from the parts that fed it. Only the
    per-venue parts the builders write carry `part-<dataset>-<venue>-...`, so
    reading the venue off the filename works on exactly the layout being
    replaced and fails on the one replacing it.

    Measured 2026-08-19, minutes after the funding swap: fifteen healthy hours
    reported `2 parts for 0 venue(s) - still one per symbol`, which is the fix
    being reported as the fault it had just removed. One column of one small
    file answers it properly.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(part, columns=["venue"])
    except (OSError, ValueError, KeyError):
        # No venue column, or an unreadable part. Fall back to the name, which
        # is where the builders' own parts carry it.
        pieces = part.stem.split("-")
        return {pieces[2]} if len(pieces) > 2 else set()
    return {v for v in table.column("venue").to_pylist() if v}


def _fraction_of(met: list[str], missing: dict[str, str], total: int,
                 subject: str, proof: str) -> ProbeResult:
    """A fraction that names its gap, the shape every probe here reports in."""
    detail = f"{len(met)}/{total} {subject}"
    if missing:
        named = "; ".join(f"{k}: {v}" for k, v in list(missing.items())[:4])
        more = len(missing) - 4
        detail += f" - {named}" + (f"; and {more} more" if more > 0 else "")
        return ProbeResult(PARTIAL if met else DEGRADED, detail, proof)
    if met:
        detail += " - " + "; ".join(met[:3])
    return ProbeResult(OK, detail, proof)


def probe_poll_scan_cost(
    dataset: Path = Path.home() / "capture" / "store" / "bars_60000000000ns",
    learn_root: Path = Path.home() / "capture" / "learn",
) -> ProbeResult:
    """SL-14, SL-15, RL-020: a read must cost what arrived, not what is archived.

    **Repointed 2026-08-19, for the same reason as SL-12's probe.** It measured
    the poll cost of `plumbing-momentum`, retired under RL-025, and reported
    DEGRADED forever on a heartbeat 38 hours old. Nothing polls the store on a
    loop any more - RL-024 moved every trading price to a live feed - so the
    subject of the measurement changed and the property being defended did not.

    What still reads the store on a cadence is the RETRAINER, and what makes that
    read cheap or ruinous is the hour partitioning SL-15 built. So this measures
    the layout directly: how many fragments a reader bounded to the newest hours
    has to open, against how many exist. Measured here on 2026-08-19: **168,639
    fragments across 73 hour directories**, and a reader asking for the newest
    hour opens the fragments of that hour alone.

    The number is a lower bound when the walk hits its budget, and it says so.
    """
    if not dataset.is_dir():
        return ProbeResult(NOT_MEASURED, "no bars dataset to measure",
                           str(dataset))
    hours = sorted(d.name for d in dataset.iterdir()
                   if d.is_dir() and d.name.startswith("availability_hour="))
    if not hours:
        return ProbeResult(
            DEGRADED,
            "the bars dataset is not partitioned by availability hour, so every "
            "read walks the whole archive (SL-15)", str(dataset))

    newest = [dataset / name for name in hours[-2:]]
    local = sum(1 for directory in newest for _ in directory.rglob("*.parquet"))
    total, complete = count_store_fragments(dataset)
    share = f"{local}/{total}{'' if complete else '+'}"

    # What the retrainer actually paid on its last pass, when it wrote one.
    read_costs = []
    for report_path in sorted(learn_root.glob("*-retrain.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        read = report.get("read") or {}
        if read.get("hours_read"):
            read_costs.append(f"{report.get('segment')} read "
                              f"{read['hours_read']}h -> {read.get('rows', 0):,} rows")

    detail = (f"{len(hours)} hour partitions, newest two hold {share} fragments"
              + (f"; {', '.join(read_costs)}" if read_costs else
                 "; no retrain has recorded a read"))
    proof = str(dataset)

    # The point of the partitioning: bounding a read to recent hours must bound
    # the fragments it opens, and that bound must not grow with the archive.
    #
    # **The comparison is only made against a COMPLETE count.** The archive walk
    # is budgeted and returns a lower bound when it runs out of time, and
    # comparing an exact numerator against a lower-bound denominator is how a
    # healthy store gets reported as broken - which it was, on the first run of
    # this probe, at `4288/2000+`.
    if not complete:
        return ProbeResult(
            PARTIAL,
            f"{detail} - the archive total is a lower bound, so the share of it "
            f"a two-hour read opens cannot be computed this pass", proof)
    expected = 2 * (total / len(hours)) if hours else 0
    if expected and local > 4 * expected:
        return ProbeResult(
            DEGRADED,
            f"{detail} - a two-hour read opens {local} fragments where two hours' "
            f"worth is about {expected:.0f}, so recent hours are not staying "
            f"bounded", proof)
    return ProbeResult(OK, detail, proof)


def probe_enforcement_live(
    hooks_dir: Path = Path.home() / ".claude" / "hooks",
    settings: Path = Path.home() / ".claude" / "settings.json",
) -> ProbeResult:
    """RL-021: the enforcement layer is a hook that fires, not a paragraph."""
    required = ("require-plan-row.sh", "inject-plan-index.sh")
    present = [n for n in required if (hooks_dir / n).is_file()]
    registered = settings.read_text() if settings.is_file() else ""
    wired = [n for n in required if n in registered]
    detail = f"{len(present)}/2 hook scripts present, {len(wired)}/2 registered"
    proof = f"{hooks_dir} and {settings}"
    if len(wired) == len(present) == len(required):
        return ProbeResult(OK, detail, proof)
    if present or wired:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(NOT_MEASURED, detail, proof)


def probe_rulings_register_loads(register: Path = REGISTER) -> ProbeResult:
    """RL-021: the register is only a record if it can be read mechanically."""
    from plan.rulings import read_rulings
    if not register.is_file():
        return ProbeResult(NOT_MEASURED, "no register on disk", str(register))
    try:
        rulings = read_rulings(register)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the register does not parse: {failure!r}",
                           str(register))
    scoped = sum(1 for r in rulings if r.scope)
    return ProbeResult(OK, f"{len(rulings)} rulings parse, {scoped} scoped",
                       str(register))


def probe_spine_parses(spine: Path = SPINE) -> ProbeResult:
    """RL-007: an unparseable plan is a plan nothing can measure against."""
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document on disk", str(spine))
    try:
        slices = read_master_plan(spine)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the spine does not parse: {failure!r}",
                           str(spine))
    rows = sum(len(s.rows) for s in slices)
    empty = [s.key for s in slices if not s.rows]
    return ProbeResult(
        OK,
        f"{len(slices)} slices, {rows} rows, {len(empty)} slice(s) with no rows yet",
        str(spine))


def probe_spine_present(spine: Path = SPINE) -> ProbeResult:
    """RL-007: the plan states its own authority, or it has none."""
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document on disk", str(spine))
    body = spine.read_text()
    claims = "top of the authority chain" in body.lower()
    supersedes = "2026-08-09-full-build-master-plan.md" in body
    if claims and supersedes:
        return ProbeResult(OK, "claims authority and names what it supersedes",
                           str(spine))
    return ProbeResult(PARTIAL,
                       f"authority claimed: {claims}, supersession named: {supersedes}",
                       str(spine))


def _name_gap(missing: tuple[str, ...], most: int = 4) -> str:
    """Which subjects a ruling is short of, named rather than counted.

    SL-05 accepts a fraction that NAMES the missing segment, and the probe
    reported `RL-006 2/4` without ever saying which two bots were short. A gap
    nobody names is a gap nobody closes - the same reason the plan counts
    distinct slices instead of rows.

    A per-brain ruling can be short of all twelve, so a long tail is counted
    rather than listed: the first few name the gap, and the remainder says how
    much more of it there is instead of filling the tile with a list.
    """
    if not missing:
        return "nothing missing"
    # `shared` and `system` carry a sentence rather than a list of subjects,
    # because "one row anywhere" has no subject to name.
    if missing == ("no row anywhere",):
        return "no row anywhere"
    if len(missing) <= most:
        return "missing " + ", ".join(missing)
    return (f"missing {', '.join(missing[:most])} "
            f"and {len(missing) - most} more")


def probe_scope_arithmetic(spine: Path = SPINE,
                           register: Path = REGISTER) -> ProbeResult:
    """RL-021: coverage must read as a fraction, not as a yes.

    Reports how many rulings the plan actually covers. A low number is the
    correct result today and is the whole reason the arithmetic exists.
    """
    from plan.rulings import read_rulings
    if not (spine.is_file() and register.is_file()):
        return ProbeResult(NOT_MEASURED, "plan or register missing", str(spine))
    try:
        slices = read_master_plan(spine)
        rulings = read_rulings(register)
    except Exception as failure:
        return ProbeResult(DEGRADED, f"{failure!r}", str(spine))
    covers = [cover_ruling(r, slices) for r in rulings]
    resolved = sum(1 for c in covers if c.resolved)
    partial = [f"{c.subject} {c.have}/{c.need} ({_name_gap(c.missing)})"
               for c in covers if not c.resolved]
    detail = f"{resolved}/{len(covers)} rulings covered; short: {', '.join(partial[:6])}"
    if resolved == len(covers):
        return ProbeResult(OK, detail, str(spine))
    return ProbeResult(PARTIAL, detail, str(spine))


def probe_authority_chain_consistent(repo: Path = REPO) -> ProbeResult:
    """RL-007, RL-019: exactly one document may claim the order of work."""
    old = repo / "docs" / "superpowers" / "plans" / "2026-08-09-full-build-master-plan.md"
    goal = (repo / "docs" / "superpowers" / "specs"
            / "2026-08-08-final-project-goal-design.md")
    superseded = old.is_file() and "SUPERSEDED" in old.read_text().upper()
    # The HEADING, not the string. A bare "3b" appears in unrelated prose - the
    # goal document cites `ARCHITECTURE.md §3b` about coinbase - and the loose
    # version of this check reported the section present before it was written.
    records_ruling = goal.is_file() and bool(
        re.search(r"^##\s+3b\.", goal.read_text(), re.MULTILINE))
    detail = (f"old plan marked superseded: {superseded}, "
              f"goal records §3b: {records_ruling}")
    proof = f"{old.name}, {goal.name}"
    if superseded and records_ruling:
        return ProbeResult(OK, detail, proof)
    if superseded or records_ruling:
        return ProbeResult(PARTIAL, detail, proof)
    return ProbeResult(
        DEGRADED,
        "two documents still claim authority over the order of work: " + detail,
        proof)


def probe_reconciliation_sweeps(spine: Path = SPINE) -> ProbeResult:
    """RL-021: a sweep that does not run cannot report anything skipped.

    PARTIAL while members remain unassigned, which is the honest reading today
    and for a long time: 2,738 corpus members against 13 plan rows. OK would
    require every member assigned, declined or written off as prose.
    """
    from plan.cli import read_decisions
    from plan.reconcile_sources import (
        read_catalogue_rows, read_design_sections, read_ledger_rows, reconcile,
    )
    if not spine.is_file():
        return ProbeResult(NOT_MEASURED, "no plan document to sweep against",
                           str(spine))
    try:
        members = (read_ledger_rows(Path.home() / "research" / "ledger" / "merged")
                   + read_catalogue_rows(Path.home() / "research" / "FEATURES.md")
                   + read_design_sections([Path.home() / "research",
                                           REPO / "docs" / "superpowers"]))
        resolutions = reconcile(members, read_master_plan(spine), read_decisions())
    except Exception as failure:
        return ProbeResult(DEGRADED, f"the sweep failed: {failure!r}", str(spine))
    unassigned = sum(1 for r in resolutions if r.outcome == "unassigned")
    detail = f"{len(resolutions)} members swept, {unassigned} unassigned"
    proof = "plan.reconcile_sources over ledger, catalogue and design sections"
    return ProbeResult(OK if unassigned == 0 else PARTIAL, detail, proof)


def _board_freshness(name: str, out_dir: Path) -> ProbeResult:
    """A generated board is only evidence while it is recent."""
    page = out_dir / name
    if not page.is_file():
        return ProbeResult(NOT_MEASURED, f"{name} has never been generated", str(page))
    age_s = time.time() - page.stat().st_mtime
    detail = f"{name} generated {age_s / 60:.0f} min ago, {page.stat().st_size} bytes"
    if age_s > 24 * 3600:
        return ProbeResult(DEGRADED, detail + " — the generator has stopped", str(page))
    return ProbeResult(OK, detail, str(page))


def probe_plan_board_rendered(
        out_dir: Path = Path.home() / "research" / "dashboard") -> ProbeResult:
    """RL-012: the board is a display of measured state, so it must exist."""
    return _board_freshness("ajit-master-plan.html", out_dir)


def probe_ruling_conformance(
        out_dir: Path = Path.home() / "research" / "dashboard") -> ProbeResult:
    """RL-021: a conformance board nobody renders measures nothing."""
    return _board_freshness("ruling-conformance.html", out_dir)


def probe_perp_universe_measured(
        report_path: Path = PERP_UNIVERSE_REPORT,
        stale_after_s: float = 6 * 3600.0) -> ProbeResult:
    """PB-01, RL-009, RL-014: what the perp bot may trade, and what it refused.

    Reads what `scripts/record_perp_universe.sh` published rather than running
    the selection here. The selection reads the whole store, which is minutes of
    IO, and a board pass that already takes ten to nineteen minutes must not grow
    another multi-minute probe. The cost of that choice is that a stopped
    recorder shows up as a stale report, which is why age is graded.

    The states are ordered by what a reader would do about them:

    * NOT MEASURED - nothing published. Absence is its own state (Rule 8), and
      it must never render as "nothing tradable", which is a measurement.
    * FAILING - the report contradicts itself, or every symbol was refused.
      Admitted plus excluded not equalling considered means the accounting the
      row's acceptance rests on is broken, and every number below it is suspect.
    * PARTIAL - symbols are admitted but none has depth. Book-based scalping
      features cannot run on a single one of them, so a green tile here would
      say the perp bot is ready when its edge has no inputs.
    * DEGRADED - the report is older than the recorder's cadence allows.
    """
    report_path = Path(report_path)
    if not report_path.is_file():
        return ProbeResult(NOT_MEASURED,
                           "no perp universe has been recorded - run "
                           "scripts/record_perp_universe.sh",
                           str(report_path))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        considered = int(report["considered"])
        admitted = int(report["admitted"])
        excluded = int(report["excluded"])
        deep = int(report["deep"])
        written_at_ns = int(report["written_at_ns"])
    except (OSError, ValueError, KeyError, TypeError) as failure:
        return ProbeResult(NOT_MEASURED,
                           f"unreadable perp universe report: {failure!r}",
                           str(report_path))

    age_s = (time.time_ns() - written_at_ns) / 1e9
    breakdown = report.get("excluded_by_reason") or {}
    detail = (f"{considered} considered, {admitted} admitted "
              f"({deep} with depth), {excluded} excluded"
              + (f" - {', '.join(f'{count} {reason}' for reason, count in sorted(breakdown.items()))}"
                 if breakdown else "")
              + f"; recorded {age_s / 60:.0f}min ago")

    if admitted + excluded != considered:
        return ProbeResult(FAILING,
                           f"the report does not add up: {admitted} admitted + "
                           f"{excluded} excluded is not {considered} considered",
                           str(report_path))
    if considered and not admitted:
        return ProbeResult(FAILING, detail, str(report_path))
    if not considered:
        return ProbeResult(NOT_MEASURED,
                           "0 considered - no perpetual has been seen through "
                           "the clock gate",
                           str(report_path))
    if age_s > stale_after_s:
        return ProbeResult(DEGRADED, detail, str(report_path))
    if not deep:
        return ProbeResult(PARTIAL, detail, str(report_path))
    return ProbeResult(OK, detail, str(report_path))


PROBES = {
    "probe_perp_universe_measured": probe_perp_universe_measured,
    "probe_store_offloaded": probe_store_offloaded,
    "probe_reconciliation_sweeps": probe_reconciliation_sweeps,
    "probe_plan_board_rendered": probe_plan_board_rendered,
    "probe_ruling_conformance": probe_ruling_conformance,
    "probe_memory_reachable": probe_memory_reachable,
    "probe_paper_engine_running": probe_paper_engine_running,
    "probe_poll_scan_cost": probe_poll_scan_cost,
    "probe_hour_pruning_enabled": probe_hour_pruning_enabled,
    "probe_wall_pass_completes": probe_wall_pass_completes,
    "probe_sealed_hour_compacted": probe_sealed_hour_compacted,
    "probe_enforcement_live": probe_enforcement_live,
    "probe_rulings_register_loads": probe_rulings_register_loads,
    "probe_spine_parses": probe_spine_parses,
    "probe_spine_present": probe_spine_present,
    "probe_scope_arithmetic": probe_scope_arithmetic,
    "probe_authority_chain_consistent": probe_authority_chain_consistent,
}


def assess_rulings(
    rulings: list[Ruling], slices: list[PlanSlice], probes: dict | None = None,
) -> list[tuple[Ruling, Coverage, ProbeResult]]:
    """Coverage from the plan, state from the probe, neither from the ruling.

    `probes` is passed in so the segment and learned-brain probes can be merged by
    the caller (BF-10) without this module importing the registries and journals
    they open. A caller that passes nothing measures with what is defined here.
    """
    probes = PROBES if probes is None else probes
    assessments: list[tuple[Ruling, Coverage, ProbeResult]] = []
    for ruling in rulings:
        coverage = cover_ruling(ruling, slices)
        if ruling.probe is None:
            result = ProbeResult(NOT_MEASURED, "no probe is named for this ruling",
                                 "docs/rulings.json")
        elif ruling.probe not in probes:
            result = ProbeResult(
                NOT_MEASURED, f"{ruling.probe} is named but not implemented",
                "statuswall.ruling_conformance.PROBES")
        else:
            try:
                result = probes[ruling.probe]()
            except Exception as failure:
                # A broken probe measures nothing. It must not take the board
                # down: a board that fails to render tells the reader less than
                # a board carrying one dark row.
                result = ProbeResult(
                    NOT_MEASURED,
                    f"{ruling.probe} raised {type(failure).__name__}: {failure}",
                    ruling.probe)
        assessments.append((ruling, coverage, result))
    return assessments


def render_ruling_conformance_page(assessments, generated_at_epoch_s: int) -> str:
    """One row per ruling: what was said, how covered, and what is true."""
    rows = []
    for ruling, coverage, result in assessments:
        label = STATE_LABEL.get(result.state, result.state).upper()
        gap = ", ".join(coverage.missing[:4]) if coverage.missing else ""
        rows.append(
            "<tr>"
            f"<td class='id'>{html.escape(ruling.id)}</td>"
            f"<td class='date'>{html.escape(ruling.date)}</td>"
            f"<td class='said'>{html.escape(ruling.verbatim[:180])}</td>"
            f"<td class='scope'>{html.escape(ruling.scope)}</td>"
            f"<td class='cover'>{coverage.have} / {coverage.need}</td>"
            f"<td class='gap'>{html.escape(gap)}</td>"
            f"<td class='state s-{html.escape(result.state)}'>{html.escape(label)}</td>"
            f"<td class='detail'>{html.escape(result.detail)}</td>"
            f"<td class='proof'>{html.escape(result.proof)}</td>"
            "</tr>")

    banner = render_staleness_banner(
        generated_at_epoch_s,
        time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(generated_at_epoch_s)))

    return (
        "<h1>Ruling conformance</h1>"
        f"{banner}"
        "<table><thead><tr>"
        "<th>Ruling</th><th>Date</th><th>What the user said</th><th>Scope</th>"
        "<th>Covered</th><th>Missing</th><th>State</th><th>Detail</th><th>Proof</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p class='foot'>Covered counts plan rows against the ruling's scope - "
        "four segment slices for per-segment, twelve brains for per-brain. State "
        "is measured by the named probe and answers a different question: whether "
        "the running system does it. A ruling with no probe reads NOT MEASURED "
        "and is never green.</p>")
