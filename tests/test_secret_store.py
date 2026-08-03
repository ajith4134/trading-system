"""Reading credentials, and the two ways that goes wrong quietly."""
import json
import subprocess

import pytest

from cost import secret_store
from cost.secret_store import CredentialsUnavailable, VenueCredentials, read_venue_credentials


def _store_returning(monkeypatch, payload: dict, returncode: int = 0, stderr: str = ""):
    """Stand in for the sops subprocess without needing a real encrypted file."""
    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=json.dumps(payload), stderr=stderr)

    monkeypatch.setattr(secret_store.subprocess, "run", fake_run)


def _existing_path(monkeypatch, tmp_path):
    path = tmp_path / "secrets.enc.yaml"
    path.write_text("placeholder ciphertext")
    return path


def test_a_real_credential_is_returned(monkeypatch, tmp_path):
    _store_returning(monkeypatch, {"binance": {"api_key": "abc", "api_secret": "xyz"}})
    creds = read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))
    assert creds == VenueCredentials("abc", "xyz")


def test_the_shipped_placeholder_is_refused(monkeypatch, tmp_path):
    """Signing with the dummy value produces a confusing error from the venue
    instead of an obvious one here, and 'the fetch failed' would then get blamed
    on the endpoint rather than on nobody having entered a key yet."""
    _store_returning(monkeypatch, {"binance": {
        "api_key": "PLACEHOLDER_NOT_A_REAL_KEY_000",
        "api_secret": "PLACEHOLDER_NOT_A_REAL_SECRET_000"}})

    with pytest.raises(CredentialsUnavailable) as raised:
        read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))
    assert "placeholder" in str(raised.value)


def test_a_placeholder_cannot_be_smuggled_past_by_appending_to_it(monkeypatch, tmp_path):
    _store_returning(monkeypatch, {"binance": {
        "api_key": "PLACEHOLDER_NOT_A_REAL_KEY_000_but_i_added_this",
        "api_secret": "PLACEHOLDER_NOT_A_REAL_SECRET_000_too"}})
    with pytest.raises(CredentialsUnavailable):
        read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))


def test_a_missing_store_is_a_typed_refusal_not_a_crash(tmp_path):
    with pytest.raises(CredentialsUnavailable) as raised:
        read_venue_credentials("binance", tmp_path / "does-not-exist.yaml")
    assert "no secret store" in str(raised.value)


def test_a_venue_with_no_entry_is_named(monkeypatch, tmp_path):
    _store_returning(monkeypatch, {"binance": {"api_key": "a", "api_secret": "b"}})
    with pytest.raises(CredentialsUnavailable) as raised:
        read_venue_credentials("kraken", _existing_path(monkeypatch, tmp_path))
    assert "kraken" in str(raised.value)


def test_a_half_filled_entry_is_refused(monkeypatch, tmp_path):
    _store_returning(monkeypatch, {"binance": {"api_key": "a", "api_secret": ""}})
    with pytest.raises(CredentialsUnavailable):
        read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))


def test_a_failed_decrypt_reports_stderr_and_never_stdout(monkeypatch, tmp_path):
    """stdout on a partial failure can contain plaintext, and this message is
    going into a log."""
    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="SUPER_SECRET_LEAKED_VALUE", stderr="age: no identity matched")

    monkeypatch.setattr(secret_store.subprocess, "run", fake_run)

    with pytest.raises(CredentialsUnavailable) as raised:
        read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))
    assert "no identity matched" in str(raised.value)
    assert "SUPER_SECRET_LEAKED_VALUE" not in str(raised.value)


def test_sops_not_installed_is_a_typed_refusal(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        raise FileNotFoundError("sops")

    monkeypatch.setattr(secret_store.subprocess, "run", fake_run)
    with pytest.raises(CredentialsUnavailable) as raised:
        read_venue_credentials("binance", _existing_path(monkeypatch, tmp_path))
    assert "sops" in str(raised.value)


def test_credentials_never_appear_in_a_repr_or_a_traceback():
    """The default dataclass repr would put the secret into any traceback that
    happens to include the object, and tracebacks get pasted into issues."""
    creds = VenueCredentials("REAL_KEY_VALUE", "REAL_SECRET_VALUE")
    assert "REAL_KEY_VALUE" not in repr(creds)
    assert "REAL_SECRET_VALUE" not in repr(creds)
    assert "REAL_SECRET_VALUE" not in str(creds)
    assert "REAL_SECRET_VALUE" not in f"{creds}"
