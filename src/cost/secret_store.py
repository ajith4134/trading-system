"""Reads exchange credentials from the sops+age store, into memory only.

Two rules, both from `research/trading-secrets-management.md`:

**Never decrypt to a file.** `sops -d > secrets.env` leaves plaintext on disk
outliving the command unless cleanup is explicit - and unattended, nothing
notices when that cleanup silently fails. Here sops writes to a pipe and the
plaintext exists only as a Python string in this process.

**A placeholder is not a credential.** The store ships with well-formed dummy
values so the signing path can be tested before a real key exists. Signing a
request with those would produce a confusing venue error rather than an obvious
local one, so they are rejected here by name.

Nothing in this module logs, prints, or reprs a secret. `VenueCredentials`
overrides `__repr__` because the default dataclass repr would put the secret
into any traceback that happens to include the object - and tracebacks get
pasted into issues.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STORE = Path.home() / ".config" / "trading" / "secrets.enc.yaml"

# The shipped dummy values. Matched as a prefix so a placeholder cannot be
# smuggled past by appending to it.
_PLACEHOLDER_PREFIX = "PLACEHOLDER_"

_DECRYPT_TIMEOUT_SECONDS = 30


class CredentialsUnavailable(Exception):
    """No usable credential for this venue, and why.

    A typed outcome rather than a bare failure: callers are expected to degrade
    to an unverified fee schedule, and they must be able to say which of "no
    store", "no entry" or "still a placeholder" they degraded because of.
    """


@dataclass(frozen=True)
class VenueCredentials:
    api_key: str
    api_secret: str

    def __repr__(self) -> str:                 # never let a secret reach a log
        return f"VenueCredentials(api_key='***', api_secret='***')"

    __str__ = __repr__


def _decrypt_store(path: Path) -> dict:
    if not path.is_file():
        raise CredentialsUnavailable(f"no secret store at {path}")
    try:
        finished = subprocess.run(
            ["sops", "--decrypt", "--output-type", "json", str(path)],
            capture_output=True, text=True, timeout=_DECRYPT_TIMEOUT_SECONDS)
    except FileNotFoundError as error:
        raise CredentialsUnavailable("sops is not installed or not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise CredentialsUnavailable("sops did not respond") from error

    if finished.returncode != 0:
        # stderr only. stdout on a partial failure could contain plaintext, and
        # this message is going somewhere it can be read.
        raise CredentialsUnavailable(
            f"sops could not decrypt {path}: {finished.stderr.strip()[:200]}")
    return json.loads(finished.stdout)


def read_venue_credentials(venue: str, path: Path | None = None) -> VenueCredentials:
    """The credential for one venue, or a typed refusal naming what is missing."""
    store = _decrypt_store(path or DEFAULT_STORE)

    entry = store.get(venue)
    if not isinstance(entry, dict):
        raise CredentialsUnavailable(f"no entry for '{venue}' in the secret store")

    key, secret = entry.get("api_key"), entry.get("api_secret")
    if not isinstance(key, str) or not isinstance(secret, str) or not key or not secret:
        raise CredentialsUnavailable(f"'{venue}' entry is missing api_key or api_secret")

    if key.startswith(_PLACEHOLDER_PREFIX) or secret.startswith(_PLACEHOLDER_PREFIX):
        raise CredentialsUnavailable(
            f"'{venue}' still holds the shipped placeholder - "
            "run `sops ~/.config/trading/secrets.enc.yaml` and enter a real key")

    return VenueCredentials(key, secret)
