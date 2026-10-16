"""JSON-lines client for the control mod (control-mod/PROTOCOL.md).

One request per line, one reply per line, both UTF-8 JSON objects. The client
keeps no state between calls: each ``request`` opens a connection.
"""

from __future__ import annotations

import itertools
import json
import socket
from typing import Any

from gdctl import paths

_ids = itertools.count(1)


class ControlError(Exception):
    """The control mod could not be reached or did not answer."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def encode_request(cmd: str, params: dict[str, Any] | None = None, *, req_id: int | None = None,
                   timeout: float | None = None) -> bytes:
    """One request line. ``params`` keys go at the top level next to ``cmd``.

    ``seq`` is the correlation number echoed in the reply; None params are dropped.
    """
    msg: dict[str, Any] = {"cmd": cmd}
    if req_id is not None:
        msg["seq"] = req_id
    if timeout is not None:
        msg["timeout"] = timeout
    for key, value in (params or {}).items():
        if value is None:
            continue
        if key in ("cmd", "seq", "timeout"):
            raise ValueError(f"reserved parameter name: {key}")
        msg[key] = value
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


class LineReader:
    """Splits a byte stream into lines (without the newline)."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        lines = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl]).rstrip(b"\r")
            del self._buf[: nl + 1]
            if line:
                lines.append(line)
        return lines

    @property
    def pending(self) -> int:
        return len(self._buf)


def decode_reply(line: bytes) -> dict[str, Any]:
    try:
        reply = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlError("bad_reply", f"control mod sent invalid JSON: {exc}") from exc
    if not isinstance(reply, dict):
        raise ControlError("bad_reply", "control mod reply is not an object")
    return reply


def request(cmd: str, params: dict[str, Any] | None = None, *, timeout: float = 30.0,
            port: int | None = None, connect_timeout: float = 2.0) -> dict[str, Any]:
    """Send one command, return the reply object (``ok`` may be false).

    Raises ControlError when the mod is unreachable or silent. ``timeout`` is
    how long the mod may take on the main thread; the socket waits a bit longer.
    """
    port = port or paths.CONTROL_PORT
    req_id = next(_ids)
    data = encode_request(cmd, params, req_id=req_id, timeout=timeout)
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=connect_timeout)
    except OSError as exc:
        raise ControlError("unreachable", f"control mod not listening on 127.0.0.1:{port} ({exc})") from exc
    with sock:
        sock.settimeout(timeout + 5.0)
        try:
            sock.sendall(data)
            reader = LineReader()
            while True:
                chunk = sock.recv(1 << 16)
                if not chunk:
                    raise ControlError("disconnected", "control mod closed the connection without replying")
                for line in reader.feed(chunk):
                    reply = decode_reply(line)
                    if reply.get("seq") in (req_id, None):
                        return reply
        except socket.timeout as exc:
            raise ControlError("timeout", f"no reply to {cmd} within {timeout + 5.0:.0f} s") from exc
        except OSError as exc:
            raise ControlError("disconnected", f"connection error: {exc}") from exc


def reachable(port: int | None = None, timeout: float = 2.0) -> bool:
    """True if the control mod's network thread answers (main thread may be stuck)."""
    try:
        return bool(request("net_ping", timeout=timeout, port=port, connect_timeout=timeout).get("ok"))
    except ControlError:
        return False
