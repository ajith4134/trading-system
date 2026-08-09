"""Building the polled datasets from the archive, and refusing to read a live hour."""
import json
from pathlib import Path

from capture.raw_writer import RawWriter, writing_marker_path
from store.build_polled import build_for_day
from store.clock_gated_reader import ClockGatedReader

PREMIUM_INDEX = json.dumps({
    "symbol": "BTCUSDT", "markPrice": "64971.30", "indexPrice": "64980.11",
    "lastFundingRate": "0.00006847", "nextFundingTime": 1786204800000,
    "time": 1785650606214,
})


def archived(root: Path, stream: str, symbol: str, payload: str) -> Path:
    writer = RawWriter(root, "binance", stream, symbol)
    writer.append(payload, t_recv_ns=1785650606_500_000_000,
                  t_exch_ms=1785650606214, seq=None)
    writer.close()
    return next((root / "raw").rglob(f"{stream}_{symbol}_*.ndjson.zst"))


def test_a_funding_poll_reaches_the_dataset(tmp_path: Path):
    archived(tmp_path, "premiumIndex", "BTCUSDT", PREMIUM_INDEX)
    store = tmp_path / "store"

    result = build_for_day(tmp_path, store, "binance", "2026-08-02",
                           ["BTCUSDT"], "funding")

    assert result["rows"] == 1 and result["appended"]
    served = ClockGatedReader(store, "funding").read_as_of(2 * 10**18)
    assert len(served) == 1
    assert served.iloc[0]["funding_rate"] == "0.00006847"


def test_an_hour_a_writer_still_holds_is_not_read(tmp_path: Path):
    """Reading one gets a zstd frame mid-write. Skipping it up front means the
    build is not counted as failed for meeting a file that is simply open."""
    raw = archived(tmp_path, "premiumIndex", "BTCUSDT", PREMIUM_INDEX)
    writing_marker_path(raw).write_text("1", encoding="utf-8")

    result = build_for_day(tmp_path, tmp_path / "store", "binance",
                           "2026-08-02", ["BTCUSDT"], "funding")

    assert result["files_read"] == 0
    assert result["rows"] == 0 and not result["appended"]


def test_rebuilding_the_same_day_does_not_double_the_rows(tmp_path: Path):
    """A timer fires repeatedly over the same days. Appending twice would
    double every rate, and a doubled funding series prices carry at twice its
    true cost without anything looking wrong."""
    archived(tmp_path, "premiumIndex", "BTCUSDT", PREMIUM_INDEX)
    store = tmp_path / "store"

    first = build_for_day(tmp_path, store, "binance", "2026-08-02",
                          ["BTCUSDT"], "funding")
    second = build_for_day(tmp_path, store, "binance", "2026-08-02",
                           ["BTCUSDT"], "funding")

    assert first["appended"] and not second["appended"]
    assert len(ClockGatedReader(store, "funding").read_as_of(2 * 10**18)) == 1


def test_a_day_with_no_polls_writes_nothing(tmp_path: Path):
    result = build_for_day(tmp_path, tmp_path / "store", "binance",
                           "2026-08-02", ["BTCUSDT"], "funding")
    assert result["rows"] == 0 and not result["appended"]


def test_hours_that_close_after_the_first_pass_still_reach_the_dataset(tmp_path: Path):
    """The defect this defends against was live for a full day: the snapshot id
    was one-per-DAY, so the first pass after midnight froze the day at whatever
    hour it had reached, and every later pass collided and appended nothing.
    Measured 2026-08-09: binance funding newest row 12:00 at 18:00."""
    later_poll = json.dumps({
        "symbol": "BTCUSDT", "markPrice": "64999.10", "indexPrice": "65001.00",
        "lastFundingRate": "0.00007000", "nextFundingTime": 1786204800000,
        "time": 1785654206214,
    })
    archived(tmp_path, "premiumIndex", "BTCUSDT", PREMIUM_INDEX)
    store = tmp_path / "store"
    first = build_for_day(tmp_path, store, "binance", "2026-08-02",
                          ["BTCUSDT"], "funding")
    assert first["appended"] and first["rows"] == 1

    # An hour closes after the first pass: same day, one more file.
    writer = RawWriter(tmp_path, "binance", "premiumIndex", "BTCUSDT")
    writer.append(later_poll, t_recv_ns=1785654206_500_000_000,
                  t_exch_ms=1785654206214, seq=None)
    writer.close()

    second = build_for_day(tmp_path, store, "binance", "2026-08-02",
                           ["BTCUSDT"], "funding")
    assert second["appended"], "the later hour must append, not collide"
    assert second["rows"] == 1, (
        "only the NEW row appends - re-appending the first would double it")

    served = ClockGatedReader(store, "funding").read_as_of(2 * 10**18)
    assert len(served) == 2
    assert sorted(served["funding_rate"]) == ["0.00006847", "0.00007000"]
