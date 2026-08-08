"""Evicting the local copy of an archive that is already somewhere else.

The raw archive is the one thing on this box that cannot be rebuilt - exchanges
do not serve history back, and `offload_to_gcs.sh` says so in its own header. So
nothing here deletes data. It deletes the **local copy** of a file already proven
to be in the bucket, which is cache eviction wearing a frightening verb.

Every test below is a way of being wrong about "already in the bucket". Each keep
rule exists because deleting under that condition is unrecoverable, and the module
is written so that any question it cannot answer resolves to keep.
"""
import datetime as dt
import json
from pathlib import Path

from ops.raw_eviction import (
    evict_planned_files,
    main,
    object_names_from_listing,
    plan_raw_eviction,
)

TODAY = dt.date(2026, 8, 8)


def a_raw_file(raw_root, venue="binance", date="2026-08-01",
               name="trade_BTCUSDT_2026-08-01T09.ndjson.zst", body=b"x" * 100):
    path = Path(raw_root) / venue / date / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def object_name(path, raw_root):
    """The bucket key offload_to_gcs.sh would have written this file under."""
    return f"raw/{Path(path).relative_to(raw_root).as_posix()}"


def test_a_file_inside_the_retention_window_is_kept(tmp_path):
    """Seven days of local archive is what the build pipeline reads from.

    The bar builder reads yesterday and reaches into the next day's first hours
    for late trades, so even a two-day window is load-bearing. Seven is the slack
    for a failed supervisor pass or a rebuild after a code fix.
    """
    raw_root = tmp_path / "raw"
    recent = a_raw_file(raw_root, date="2026-08-06")

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={object_name(recent, raw_root)})

    assert plan.evictable == ()
    assert plan.kept == ((recent, "within retention"),)


def test_a_file_missing_from_the_bucket_is_kept_however_old_it_is(tmp_path):
    """The single rule the whole module exists to enforce.

    Age is not evidence of anything. A file that never uploaded - a failed batch,
    a bucket outage, a file written while the offload was between passes - is the
    only copy that exists, and deleting it is the one mistake here that no
    re-download can undo.
    """
    raw_root = tmp_path / "raw"
    never_uploaded = a_raw_file(raw_root, date="2026-07-01")

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects=set())

    assert plan.evictable == ()
    assert plan.kept == ((never_uploaded, "not in bucket"),)


def test_an_old_file_present_in_the_bucket_is_evictable(tmp_path):
    """Old, uploaded, closed - the only combination that permits removal."""
    raw_root = tmp_path / "raw"
    uploaded = a_raw_file(raw_root, date="2026-07-01", body=b"y" * 4096)

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={object_name(uploaded, raw_root)})

    assert plan.evictable == (uploaded,)
    assert plan.kept == ()
    assert plan.bytes_reclaimable == 4096


def test_an_hour_a_writer_still_holds_is_kept_even_when_old_and_uploaded(tmp_path):
    """A live marker on an old hour is not a contradiction.

    `offload_to_gcs.sh` records finding markers from three different hours at
    once, one of them stale from a session that had crashed six days earlier. So
    "old" does not imply "closed", and the bucket copy of a half-written hour is
    a truncated file that reads as complete - exactly the copy you must not treat
    as the survivor.
    """
    raw_root = tmp_path / "raw"
    open_hour = a_raw_file(raw_root, date="2026-07-01")
    stem = str(open_hour).removesuffix(".ndjson.zst")
    Path(stem + ".writing").write_text("")

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={object_name(open_hour, raw_root)})

    assert plan.evictable == ()
    assert plan.kept == ((open_hour, "still being written"),)


def test_seven_days_means_seven_distinct_days_including_today(tmp_path):
    """Off by one here silently shortens the window the builder relies on.

    With today at Aug 8 and keep_days 7, Aug 2 is the oldest day kept and Aug 1
    is the first evicted - seven dates, Aug 2 through Aug 8.
    """
    raw_root = tmp_path / "raw"
    oldest_kept = a_raw_file(raw_root, date="2026-08-02", name="trade_A_x.ndjson.zst")
    first_evicted = a_raw_file(raw_root, date="2026-08-01", name="trade_A_y.ndjson.zst")
    inventory = {object_name(oldest_kept, raw_root),
                 object_name(first_evicted, raw_root)}

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects=inventory)

    assert plan.evictable == (first_evicted,)
    assert plan.kept == ((oldest_kept, "within retention"),)


def test_a_path_that_is_not_venue_slash_date_is_never_touched(tmp_path):
    """Anything whose day cannot be read is a file this has no business removing.

    Quarantine files, a stray directory, a layout that changes later - all resolve
    to keep, because the alternative is a deletion rule whose behaviour on
    unfamiliar input is "delete".
    """
    raw_root = tmp_path / "raw"
    stray = raw_root / "quarantine" / "stranded_2026-07-01.ndjson.zst"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"z" * 10)

    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={"raw/quarantine/stranded_2026-07-01.ndjson.zst"})

    assert plan.evictable == ()
    assert plan.kept == ((stray, "unrecognised path"),)


# --- doing it ----------------------------------------------------------------

def test_eviction_removes_the_planned_files_and_nothing_else(tmp_path):
    raw_root = tmp_path / "raw"
    old = a_raw_file(raw_root, date="2026-07-01", name="trade_A_old.ndjson.zst")
    recent = a_raw_file(raw_root, date="2026-08-08", name="trade_A_new.ndjson.zst")
    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={object_name(old, raw_root),
                                             object_name(recent, raw_root)})

    removed, freed = evict_planned_files(plan)

    assert removed == 1
    assert freed == 100
    assert not old.exists()
    assert recent.exists()


def test_eviction_rechecks_the_writing_marker_before_each_delete(tmp_path):
    """The plan is a snapshot; the writers never stopped running.

    A capture restart between planning and deleting can reopen an hour, and the
    plan would still name it. Re-checking costs one stat and removes the whole
    class of races where a live file is deleted on stale evidence.
    """
    raw_root = tmp_path / "raw"
    old = a_raw_file(raw_root, date="2026-07-01")
    plan = plan_raw_eviction(raw_root, today=TODAY, keep_days=7,
                             bucket_objects={object_name(old, raw_root)})
    assert plan.evictable == (old,)

    Path(str(old).removesuffix(".ndjson.zst") + ".writing").write_text("")
    removed, freed = evict_planned_files(plan)

    assert (removed, freed) == (0, 0)
    assert old.exists()


# --- reading the bucket's own listing ----------------------------------------

def test_the_listing_parses_into_the_keys_the_plan_compares_against():
    """The one place a wrong answer could delete something.

    A parse that drops the prefix incorrectly yields keys that match nothing,
    which is merely useless. A parse that matches too loosely would call a file
    uploaded when it is not, which is the unrecoverable direction - so the bucket
    name is stripped exactly and directory placeholders are discarded.
    """
    listing = [
        "gs://capture-raw-data4134/raw/binance/2026-08-08/trade_BTCUSDT_2026-08-08T10.ndjson.zst",
        "gs://capture-raw-data4134/raw/binance/2026-08-08/",
        "",
        "gs://capture-raw-data4134/raw/hyperliquid/2026-08-03/trades_BTC_2026-08-03T00.idx.zst",
    ]

    keys = object_names_from_listing(listing, bucket="gs://capture-raw-data4134")

    assert keys == {
        "raw/binance/2026-08-08/trade_BTCUSDT_2026-08-08T10.ndjson.zst",
        "raw/hyperliquid/2026-08-03/trades_BTC_2026-08-03T00.idx.zst",
    }


def test_a_line_from_another_bucket_is_refused_rather_than_stripped():
    """Belt and braces: a key from the wrong bucket is not evidence of anything.

    It cannot happen through the normal call path, and that is exactly why it
    would go unnoticed if it ever did.
    """
    listing = ["gs://someone-elses-bucket/raw/binance/2026-08-08/trade_A.ndjson.zst"]

    assert object_names_from_listing(listing, bucket="gs://capture-raw-data4134") == set()


# --- the command ------------------------------------------------------------

def test_the_command_deletes_nothing_without_apply(tmp_path, capsys):
    """Dry run is the default because the mistake is one-way.

    An operator who runs this to see what it would do must not discover the
    answer by having it done.
    """
    raw_root = tmp_path / "raw"
    old = a_raw_file(raw_root, date="2026-07-01")

    code = main(["--bucket", "gs://b", "--capture-root", str(tmp_path),
                 "--keep-days", "7", "--today", "2026-08-08"],
                list_objects=lambda bucket: [f"gs://b/{object_name(old, raw_root)}"])

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["applied"] is False
    assert report["evictable"] == 1
    assert old.exists()


def test_the_command_removes_the_files_when_applied(tmp_path, capsys):
    raw_root = tmp_path / "raw"
    old = a_raw_file(raw_root, date="2026-07-01")

    code = main(["--bucket", "gs://b", "--capture-root", str(tmp_path),
                 "--keep-days", "7", "--today", "2026-08-08", "--apply"],
                list_objects=lambda bucket: [f"gs://b/{object_name(old, raw_root)}"])

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (report["applied"], report["removed"], report["freed_bytes"]) == (True, 1, 100)
    assert not old.exists()


def test_an_empty_listing_refuses_to_apply(tmp_path, capsys):
    """An empty bucket listing is far likelier to be a broken call than a fact.

    The plan would keep everything either way, so nothing is at risk - but
    reporting "0 evicted, success" would let a silently broken credential look
    like a healthy eviction run for as long as nobody checked the disk.
    """
    raw_root = tmp_path / "raw"
    a_raw_file(raw_root, date="2026-07-01")

    code = main(["--bucket", "gs://b", "--capture-root", str(tmp_path),
                 "--keep-days", "7", "--today", "2026-08-08", "--apply"],
                list_objects=lambda bucket: [])

    report = json.loads(capsys.readouterr().out)
    assert code != 0
    assert report["applied"] is False
    assert "empty" in report["refused"]
