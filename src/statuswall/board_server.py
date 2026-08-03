"""Serves the boards over HTTP Basic auth, bound to loopback only.

The boards carry the system's architecture and its live operational state, and
the tunnel in front of them is reachable by anyone holding the URL. An
unguessable URL is obscurity, not a control - it leaks the moment it is pasted
anywhere, and it cannot be revoked without changing the address. A password can
be rotated without the address changing.

Bound to 127.0.0.1 because the tunnel connects locally: binding to 0.0.0.0 would
additionally expose the port to anything that can reach the VM's network, which
is a second door nobody asked for.

Directory listings are refused. The boards are served by name; an index of the
directory would enumerate whatever else happens to sit beside them.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import http.server
import os
import secrets
import socketserver
import stat
from pathlib import Path

CREDENTIALS_ENV = "BOARDS_CREDENTIALS_FILE"
DEFAULT_CREDENTIALS = Path.home() / ".config" / "boards" / "credentials"
REALM = "Trading system boards"


class CredentialsError(RuntimeError):
    """The password file is missing, malformed, or readable by others."""


def read_credentials(path: Path) -> tuple[str, str]:
    """Load `user:password`, refusing a file other accounts can read.

    The permission check is part of loading rather than a separate lint: a
    world-readable password on a shared host is already disclosed, and failing
    to start is the only response that does not serve the boards anyway.
    """
    path = Path(path)
    if not path.is_file():
        raise CredentialsError(f"no credentials file at {path}")

    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise CredentialsError(
            f"{path} is accessible to group or others (mode {stat.S_IMODE(mode):04o}); "
            f"run: chmod 600 {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if ":" not in raw:
        raise CredentialsError(f"{path} must contain exactly 'user:password'")
    user, password = raw.split(":", 1)
    if not user or not password:
        raise CredentialsError(f"{path} has an empty user or password")
    return user, password


def generate_credentials(path: Path, user: str = "boards") -> tuple[str, str]:
    """Write a fresh random password, owner-readable only. Returns it once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    password = secrets.token_urlsafe(18)
    # Opened with 0600 rather than written and chmod'ed after: between those two
    # steps the password would exist on disk world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"{user}:{password}\n")
    return user, password


def is_authorised(header: str | None, user: str, password: str) -> bool:
    """Constant-time check of an Authorization header against the credentials.

    `hmac.compare_digest` on the whole `user:password` pair: a plain `==` leaks
    how many leading characters were right through its timing, which over many
    attempts recovers the secret one character at a time.
    """
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return hmac.compare_digest(decoded, f"{user}:{password}")


class BoardHandler(http.server.SimpleHTTPRequestHandler):
    """Static files under `directory`, behind Basic auth, no listings."""

    user = ""
    password = ""

    def do_GET(self) -> None:
        if not self._demand_auth():
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if not self._demand_auth():
            return
        super().do_HEAD()

    def _demand_auth(self) -> bool:
        if is_authorised(self.headers.get("Authorization"), self.user, self.password):
            return True
        body = b"Authentication required.\n"
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{REALM}", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def list_directory(self, path):  # noqa: D102 - overriding to refuse, not to document
        self.send_error(404, "Not found")
        return None

    def end_headers(self) -> None:
        # The boards are regenerated in place, so a cached copy is a stale copy -
        # and a stale status board is the specific thing Rule 8 forbids.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("X-Robots-Tag", "noindex, nofollow")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Default logs the client address, which for a tunnelled request is always
        # the local tunnel. Keep the request line; drop the misleading address.
        print(f"{self.log_date_time_string()} {fmt % args}", flush=True)


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_server(directory: Path, host: str, port: int,
                 user: str, password: str) -> _ThreadingServer:
    handler = type("BoundBoardHandler", (BoardHandler,), {
        "user": user,
        "password": password,
        "directory": str(directory),
    })

    def factory(*args):
        return handler(*args, directory=str(directory))

    return _ThreadingServer((host, port), factory)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="board-server",
        description="Serve the boards on loopback behind HTTP Basic auth.")
    parser.add_argument("--directory", default=str(Path.home() / "research" / "dashboard"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--credentials",
                        default=os.environ.get(CREDENTIALS_ENV, str(DEFAULT_CREDENTIALS)))
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")

    user, password = read_credentials(Path(args.credentials))
    server = build_server(directory, args.host, args.port, user, password)
    print(f"serving {directory} on http://{args.host}:{args.port} as user '{user}'", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
