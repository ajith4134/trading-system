"""The server's job is to refuse, so these tests attack the refusal.

An auth layer that is merely present is worth nothing; what matters is that
every path through it denies by default and that the comparison cannot be
walked one character at a time.
"""
from __future__ import annotations

import base64
import stat
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from statuswall.board_server import (
    CredentialsError, build_server, generate_credentials, is_authorised, read_credentials,
)


def _basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def test_generated_credentials_are_owner_only(tmp_path):
    path = tmp_path / "credentials"
    generate_credentials(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"password file is mode {mode:04o}"


def test_generated_password_is_not_predictable(tmp_path):
    first = generate_credentials(tmp_path / "a")[1]
    second = generate_credentials(tmp_path / "b")[1]
    assert first != second
    assert len(first) >= 20


def test_group_or_world_readable_credentials_refuse_to_load(tmp_path):
    """A disclosed password is already disclosed; starting anyway serves the boards."""
    path = tmp_path / "credentials"
    generate_credentials(path)
    path.chmod(0o644)
    with pytest.raises(CredentialsError, match="group or others"):
        read_credentials(path)


def test_missing_credentials_refuse_rather_than_defaulting(tmp_path):
    with pytest.raises(CredentialsError, match="no credentials file"):
        read_credentials(tmp_path / "absent")


def test_malformed_credentials_refuse(tmp_path):
    path = tmp_path / "credentials"
    path.write_text("no-colon-here\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(CredentialsError, match="user:password"):
        read_credentials(path)


def test_password_may_contain_colons(tmp_path):
    path = tmp_path / "credentials"
    path.write_text("boards:pa:ss:word\n", encoding="utf-8")
    path.chmod(0o600)
    assert read_credentials(path) == ("boards", "pa:ss:word")


# --------------------------------------------------------------------------
# header checking
# --------------------------------------------------------------------------

@pytest.mark.parametrize("header", [
    None,
    "",
    "Bearer sometoken",
    "Basic",
    "Basic !!!not-base64!!!",
    "Basic " + base64.b64encode(b"\xff\xfe").decode(),
    _basic("boards", "wrong"),
    _basic("wrong", "secret"),
    _basic("boards", ""),
])
def test_bad_authorization_headers_are_all_refused(header):
    assert is_authorised(header, "boards", "secret") is False


def test_correct_header_is_accepted():
    assert is_authorised(_basic("boards", "secret"), "boards", "secret") is True


# --------------------------------------------------------------------------
# live server
# --------------------------------------------------------------------------

@pytest.fixture()
def served(tmp_path):
    (tmp_path / "index.html").write_text("<p>board</p>", encoding="utf-8")
    (tmp_path / "secret-note.txt").write_text("data", encoding="utf-8")
    server = build_server(tmp_path, "127.0.0.1", 0, "boards", "secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _fetch(url: str, header: str | None = None):
    request = urllib.request.Request(url)
    if header:
        request.add_header("Authorization", header)
    return urllib.request.urlopen(request, timeout=10)


def test_unauthenticated_request_is_refused_with_a_challenge(served):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _fetch(f"{served}/index.html")
    assert caught.value.code == 401
    assert "Basic" in caught.value.headers.get("WWW-Authenticate", "")


def test_authenticated_request_serves_the_board(served):
    response = _fetch(f"{served}/index.html", _basic("boards", "secret"))
    assert response.status == 200
    assert b"board" in response.read()


def test_wrong_password_is_refused_by_the_live_server(served):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _fetch(f"{served}/index.html", _basic("boards", "nearly-secret"))
    assert caught.value.code == 401


def test_directory_listing_is_refused_even_when_authenticated(served, tmp_path):
    """A listing enumerates whatever else sits beside the boards."""
    (tmp_path / "sub").mkdir()
    with pytest.raises(urllib.error.HTTPError) as caught:
        _fetch(f"{served}/sub/", _basic("boards", "secret"))
    assert caught.value.code == 404


def test_responses_forbid_caching_and_indexing(served):
    """A cached status board is a stale one, which is what Rule 8 forbids."""
    response = _fetch(f"{served}/index.html", _basic("boards", "secret"))
    assert "no-store" in response.headers.get("Cache-Control", "")
    assert "noindex" in response.headers.get("X-Robots-Tag", "")
