"""The kill switch — and an honest account of the one this host cannot have.

`ARCHITECTURE.md` Layer 3 specifies a watchdog in a **separate OS process** that
kills the bot *and drops outbound network at the firewall* on a hard-cap breach,
and says plainly why the firewall carries the weight:

> No major exchange appears to expose programmatic self-revocation of your own
> key. Do not design a kill switch assuming it — the firewall is the reliable
> mechanism.

**Measured on this host, 2026-08-08: the reliable mechanism is unavailable.**
`sudo` is denied outright rather than password-prompted, and neither `iptables`
nor `nft` is installed. There is no way for this process to drop a packet.

So this module does what it can and **records what it cannot**, because a kill
switch that reports success while leaving the network open is worse than none —
it is the reads-as-built failure applied to the most consequential control in the
system.

Three mechanisms, in descending order of strength:

**Credential denial.** The age identity is moved aside, so `cost.secret_store`
can no longer decrypt and no signed request can be constructed at all. This is
the substitute for the firewall and it is genuinely weaker: an already-open
socket survives it, and an unauthenticated endpoint stays reachable. It is set
aside rather than destroyed — a kill switch that loses the only copy of a key
turns a drawdown into a permanent outage.

**Process kill.** SIGKILL, not SIGTERM. A wedged process is exactly the case the
watchdog exists for, and a wedged process may not service a handler.

**A persistent kill file.** Checked by anything that would trade, and it survives
a restart on purpose: a kill the supervisor undoes on its next loop looks like it
fired and did not.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

KILL_FILE = "KILLED.json"
_HISTORY_FILE = "watchdog.ndjson"

# Present on this host? Measured, not assumed - and re-checked at trip time
# rather than trusted from a constant, because the answer would change the
# moment someone installs a packet filter and grants sudo.
_FIREWALL_TOOLS = ("iptables", "nft")


def is_killed(root: Path) -> bool:
    """Whether a kill is in force. The one question a trading path must ask."""
    return (Path(root) / KILL_FILE).exists()


def kill_reason(root: Path) -> str | None:
    """Why trading is stopped, or `None` when it is not."""
    try:
        return json.loads((Path(root) / KILL_FILE).read_text(encoding="utf-8"))["reason"]
    except (OSError, ValueError, KeyError):
        return None


def _can_drop_network() -> tuple[bool, str]:
    """Whether this process could actually cut egress, and why not.

    Re-measured on every trip. `ARCHITECTURE.md` names the firewall as the
    reliable mechanism, so whether it is reachable is the single most important
    fact about a kill and must never be a stale constant.
    """
    if os.geteuid() == 0:
        for tool in _FIREWALL_TOOLS:
            if shutil.which(tool):
                return True, f"running as root with {tool} available"
        return False, "running as root but no iptables/nft on PATH"

    if shutil.which("sudo") is None:
        return False, "no sudo on PATH and not root - cannot touch the firewall"
    # `-n` so a sudo prompt cannot block: a kill path must never stop waiting
    # for a password nobody is present to type. Run through subprocess with an
    # argument list rather than a shell, so there is no shell in a kill path at
    # all - the argument vector is fixed here, but the habit matters more in the
    # one control that has to work while everything else is failing.
    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True,
                           timeout=5, check=False)
    if probe.returncode != 0:
        return False, ("no passwordless sudo, so no firewall rule can be added - "
                       "measured on this host 2026-08-08, where sudo is denied "
                       "outright and iptables/nft are not installed")
    for tool in _FIREWALL_TOOLS:
        if shutil.which(tool):
            return True, f"passwordless sudo with {tool} available"
    return False, "passwordless sudo but no iptables/nft on PATH"


class Watchdog:
    """Trips a kill and reports honestly which mechanisms fired.

    Intended to run out-of-process. A watchdog living inside the thing it guards
    dies with it, precisely when it is needed.
    """

    def __init__(self, root: Path, identity_path: Path | None = None,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._root = Path(root)
        self._identity = Path(
            identity_path if identity_path is not None
            else Path.home() / ".config" / "sops" / "age" / "keys.txt")
        self._clock_ns = clock_ns

    def trip(self, reason: str, detail: str,
             pids: Sequence[int] = ()) -> dict:
        """Stop trading by every means available, and record what worked.

        Ordered deliberately: the kill file first, so that even if everything
        after it fails, anything consulting `is_killed` refuses to trade.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        outcome = {
            "reason": reason, "detail": detail, "at_ns": self._clock_ns(),
            "kill_file_written": False,
            "credential_denied": False, "credential_detail": "",
            "network_dropped": False, "network_detail": "",
            "killed_pids": [], "failed_pids": [],
        }

        # 1. The kill file, first and unconditionally - before any mechanism that
        #    could fail, so that even a half-completed trip stops trading. Every
        #    trading path reads it and refuses.
        kill_path = self._root / KILL_FILE
        is_first_trip = not kill_path.exists()
        if is_first_trip:
            kill_path.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        # A later trip must not rewrite the first breach's reason: the first is
        # the one worth diagnosing, and whatever fired last is usually a symptom.
        outcome["kill_file_written"] = True

        # 2. Credential denial - the substitute for the firewall.
        try:
            if self._identity.exists():
                aside = self._root / f"revoked-{self._identity.name}.{outcome['at_ns']}"
                shutil.move(str(self._identity), str(aside))
                outcome["credential_denied"] = True
                outcome["credential_detail"] = f"age identity moved to {aside.name}"
            else:
                outcome["credential_detail"] = (
                    f"no identity at {self._identity} - nothing to deny")
        except OSError as exc:
            # Recorded, never raised: a watchdog that stops halfway through has
            # not killed anything.
            outcome["credential_detail"] = f"{type(exc).__name__}: {exc}"

        # 3. The network, if this host can. Measured every time.
        can_drop, why = _can_drop_network()
        outcome["network_dropped"] = False
        outcome["network_detail"] = why
        if can_drop:
            # Left unimplemented rather than guessed: the rule shape depends on
            # the filter present, and an untested firewall command in a kill path
            # is a kill that fails when it matters. This host cannot reach here.
            outcome["network_detail"] = (
                f"{why}; no rule applied - firewall drop is unimplemented and "
                f"untested on this host, see ARCHITECTURE.md Layer 3")

        # 4. SIGKILL, not SIGTERM: a wedged process is the case this exists for,
        #    and a wedged process may never service a handler.
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                outcome["killed_pids"].append(pid)
            except OSError as exc:
                outcome["failed_pids"].append({"pid": pid, "error": str(exc)})

        # Rewrite the kill file now that the mechanisms have actually run. The
        # first pass was a placeholder written before any of them, so it claims
        # nothing fired - and this file is what a human reads after a halt. A
        # record that understates the kill is the Rule 8 failure applied to the
        # most consequential control here: it invites someone to assume the
        # network was cut when only the credential was.
        #
        # Only on the first trip: a later breach must not overwrite the original
        # reason. Its own outcome still lands in the history below.
        if is_first_trip:
            kill_path.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")

        with (self._root / _HISTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome) + "\n")
        return outcome
