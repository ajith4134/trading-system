"""Anything that would trade has to ask whether a kill is in force.

`ops.watchdog` calls its kill file "checked by anything that would trade", and
until 2026-08-09 nothing checked it: `is_killed` had zero callers anywhere in
src/, so the most consequential control in the system was a function with no
reader. The same audit found the halt registry uncalled and the status wall
claiming it was armed.

The order path here is paper and journals to a WAL rather than to a venue, so
the gate costs nothing today. That is the argument for it being here now: a kill
switch first wired in on the day it is needed is a kill switch first tested on
that day.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from ops.watchdog import KILL_FILE, is_killed
from paper.prove_plumbing import KillSwitchEngaged, main, run


def _engage(root: Path, reason: str = "drawdown hard cap") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / KILL_FILE).write_text(
        f'{{"reason": "{reason}", "detail": "test", "ts_ns": 1}}\n', encoding="utf-8")


def test_the_run_refuses_to_start_while_a_kill_is_in_force(tmp_path):
    wal_root = tmp_path / "wal"
    _engage(wal_root)

    with pytest.raises(KillSwitchEngaged) as refusal:
        run(tmp_path / "store", wal_root, "BTCUSDT", "binance",
            Decimal("0.1"), Decimal("50"), Decimal("1000"))

    assert "drawdown hard cap" in str(refusal.value)


def test_the_refusal_happens_before_anything_is_journalled(tmp_path):
    """Refusing after the first intent is written is not refusing. The WAL is
    the durable record of what we meant to do, and a killed run must not leave
    one."""
    wal_root = tmp_path / "wal"
    _engage(wal_root)

    with pytest.raises(KillSwitchEngaged):
        run(tmp_path / "store", wal_root, "BTCUSDT", "binance",
            Decimal("0.1"), Decimal("50"), Decimal("1000"))

    written = [p for p in wal_root.rglob("*") if p.is_file() and p.name != KILL_FILE]
    assert written == [], f"journalled despite the kill: {written}"


def test_a_run_with_no_kill_in_force_proceeds(tmp_path):
    """The gate must not be the reason nothing runs. No store here, so the run
    reports it has no bars - which is it getting past the gate and stopping for
    its own reason."""
    tally = run(tmp_path / "store", tmp_path / "wal", "BTCUSDT", "binance",
                Decimal("0.1"), Decimal("50"), Decimal("1000"))

    assert tally.bars_read == 0
    assert not is_killed(tmp_path / "wal")


def test_the_cli_exits_nonzero_and_says_so(tmp_path):
    """A refusal that reads like a quiet no-op is one an operator assumes did
    not fire."""
    wal_root = tmp_path / "wal"
    _engage(wal_root)

    code = main(["--symbol", "BTCUSDT", "--venue", "binance", "--participation", "0.1",
                 "--store-root", str(tmp_path / "store"), "--wal-root", str(wal_root)])

    assert code == 2


def test_the_kill_file_can_live_apart_from_the_journal(tmp_path):
    """One watchdog guards more than one journal. Defaulting the kill root to the
    WAL root is a convenience, not the contract."""
    kill_root = tmp_path / "ops"
    _engage(kill_root)

    with pytest.raises(KillSwitchEngaged):
        run(tmp_path / "store", tmp_path / "wal", "BTCUSDT", "binance",
            Decimal("0.1"), Decimal("50"), Decimal("1000"), kill_root=kill_root)

    # And the same run is allowed when the kill is somewhere it does not guard.
    run(tmp_path / "store", tmp_path / "wal", "BTCUSDT", "binance",
        Decimal("0.1"), Decimal("50"), Decimal("1000"), kill_root=tmp_path / "elsewhere")
