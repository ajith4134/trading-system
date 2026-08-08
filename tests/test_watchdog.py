"""The kill switch, and the honest substitute for the one we cannot have.

`ARCHITECTURE.md` Layer 3 specifies a watchdog in a **separate OS process** that
kills the bot *and drops outbound network at the firewall* on a hard-cap breach,
and is explicit about why the firewall carries the weight: *"No major exchange
appears to expose programmatic self-revocation of your own key. Do not design a
kill switch assuming it - the firewall is the reliable mechanism."*

Measured on this host 2026-08-08: **there is no sudo at all** (denied outright,
not password-prompted) and neither `iptables` nor `nft` is installed. So the
reliable mechanism is unavailable, and pretending otherwise would leave a kill
switch that cannot kill.

What remains without privilege is credential denial. Moving the age identity
aside makes `secret_store` unable to decrypt, so no signed request can be
constructed at all. That is weaker than a packet filter - an already-open socket
survives it, and an unauthenticated endpoint is still reachable - and the tests
below pin down that the difference is recorded rather than glossed.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ops.watchdog import (
    KILL_FILE,
    Watchdog,
    is_killed,
    kill_reason,
)


def test_nothing_is_killed_before_a_breach(tmp_path):
    Watchdog(tmp_path, identity_path=tmp_path / "keys.txt")
    assert not is_killed(tmp_path)
    assert kill_reason(tmp_path) is None


def test_a_breach_writes_a_kill_file_that_survives_a_restart(tmp_path):
    """A kill a process restart clears is a kill the supervisor undoes on its
    next loop, which is worse than none - it looks like it fired and didn't."""
    identity = tmp_path / "keys.txt"
    identity.write_text("AGE-SECRET-KEY-placeholder", encoding="utf-8")
    dog = Watchdog(tmp_path, identity_path=identity)

    dog.trip("drawdown_hard_cap", "portfolio -18% against a -15% ceiling")

    assert is_killed(tmp_path)
    assert (tmp_path / KILL_FILE).exists()
    assert kill_reason(tmp_path) == "drawdown_hard_cap"


def test_tripping_denies_the_credential(tmp_path):
    """The substitute for the firewall we cannot install. Without the age
    identity, secret_store cannot decrypt and no signed request can be built."""
    identity = tmp_path / "keys.txt"
    identity.write_text("AGE-SECRET-KEY-placeholder", encoding="utf-8")
    dog = Watchdog(tmp_path, identity_path=identity)

    dog.trip("drawdown_hard_cap", "breach")

    assert not identity.exists(), "the identity is still readable after a kill"
    assert list(tmp_path.glob("revoked-keys.txt*")), "identity was destroyed, not set aside"


def test_the_credential_is_set_aside_not_destroyed(tmp_path):
    """Recoverable by a human, because a kill switch that loses the only copy of
    a key turns a drawdown into a permanent outage."""
    identity = tmp_path / "keys.txt"
    identity.write_text("AGE-SECRET-KEY-real-content", encoding="utf-8")
    Watchdog(tmp_path, identity_path=identity).trip("test", "breach")

    aside = next(iter(tmp_path.glob("revoked-keys.txt*")))
    assert aside.read_text(encoding="utf-8") == "AGE-SECRET-KEY-real-content"


def test_a_missing_identity_does_not_stop_the_kill(tmp_path):
    """The kill must complete even if the thing it wanted to move is already
    gone. A watchdog that raises halfway through has not killed anything."""
    dog = Watchdog(tmp_path, identity_path=tmp_path / "absent.txt")
    dog.trip("drawdown_hard_cap", "breach")
    assert is_killed(tmp_path)


def test_the_trip_records_what_it_could_and_could_not_do(tmp_path):
    """Rule 8 on the most consequential control in the system: the record says
    which mechanisms actually fired, so nobody later assumes the network was cut
    when only the credential was."""
    identity = tmp_path / "keys.txt"
    identity.write_text("k", encoding="utf-8")
    dog = Watchdog(tmp_path, identity_path=identity)

    outcome = dog.trip("drawdown_hard_cap", "breach")

    assert outcome["credential_denied"] is True
    assert outcome["network_dropped"] is False
    assert "sudo" in outcome["network_detail"].lower() or \
           "iptables" in outcome["network_detail"].lower()


def test_a_second_trip_does_not_overwrite_the_first_reason(tmp_path):
    """The first breach is the one worth diagnosing. A later trip must not
    rewrite history into whatever fired last."""
    dog = Watchdog(tmp_path, identity_path=tmp_path / "k")
    dog.trip("drawdown_hard_cap", "the real cause")
    dog.trip("something_else", "a later symptom")
    assert kill_reason(tmp_path) == "drawdown_hard_cap"


def test_the_watchdog_kills_a_process_it_is_watching(tmp_path):
    """It runs out-of-process on purpose: a watchdog inside the thing it guards
    dies with it, exactly when it is needed."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        dog = Watchdog(tmp_path, identity_path=tmp_path / "k")
        dog.trip("test", "breach", pids=[child.pid])
        for _ in range(50):
            if child.poll() is not None:
                break
            time.sleep(0.1)
        assert child.poll() is not None, "the watched process is still running"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_the_kill_file_records_what_actually_fired(tmp_path):
    """The kill file is what a human reads at 3am after a halt. Written before
    the mechanisms run, it says `credential_denied: false` on a trip that did
    deny the credential - understating the kill in the only record most people
    will look at. The truth lived in the ndjson and nowhere else."""
    import json
    identity = tmp_path / "keys.txt"
    identity.write_text("k", encoding="utf-8")

    Watchdog(tmp_path, identity_path=identity).trip("drawdown_hard_cap", "breach")

    on_disk = json.loads((tmp_path / KILL_FILE).read_text(encoding="utf-8"))
    assert on_disk["credential_denied"] is True, "kill file understates the kill"
    assert on_disk["network_dropped"] is False
    assert on_disk["network_detail"], "no record of why the network stayed up"


def test_the_kill_file_keeps_the_first_reason_when_rewritten(tmp_path):
    """The accurate-record fix must not reopen the immutability hole: a later
    trip still must not rewrite the first breach's reason."""
    import json
    dog = Watchdog(tmp_path, identity_path=tmp_path / "k")
    dog.trip("drawdown_hard_cap", "the real cause")
    dog.trip("something_else", "a later symptom")

    on_disk = json.loads((tmp_path / KILL_FILE).read_text(encoding="utf-8"))
    assert on_disk["reason"] == "drawdown_hard_cap"
    assert on_disk["detail"] == "the real cause"
