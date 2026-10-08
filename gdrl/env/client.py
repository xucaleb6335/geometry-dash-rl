"""Socket client for the bridge: speaks PROTOCOL.md to the mod or the mock server."""

from __future__ import annotations

import socket
from dataclasses import dataclass

from gdrl.env.protocol import (
    DEFAULT_PORT,
    HEADER,
    HELLO,
    HELLO_ACK,
    PROTOCOL_VERSION,
    RESET,
    STEP,
    GameState,
    MsgType,
    ProtocolError,
    decode_error,
    decode_header,
    encode_frame,
)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[MsgType, bytes]:
    msg_type, length = decode_header(recv_exact(sock, HEADER.size))
    return msg_type, recv_exact(sock, length) if length else b""


@dataclass(frozen=True)
class ServerInfo:
    protocol_version: int
    tick_rate: int
    max_objects: int


class GDClient:
    """Blocking request/reply client. One instance per game connection.

    >>> with GDClient(port=5555) as c:
    ...     state = c.reset()
    ...     state = c.step(button=True, ticks=4)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float | None = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self.info: ServerInfo | None = None

    def connect(self) -> ServerInfo:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self.info = self._hello()
        return self.info

    def _hello(self) -> ServerInfo:
        reply = self._request(MsgType.HELLO, HELLO.pack(PROTOCOL_VERSION, 0), MsgType.HELLO_ACK)
        version, tick_rate, max_objects, _ = HELLO_ACK.unpack(reply)
        return ServerInfo(version, tick_rate, max_objects)

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState:
        reply = self._request(MsgType.RESET, RESET.pack(level_id, start_pos), MsgType.STATE)
        return GameState.from_bytes(reply)

    def step(self, button: bool, ticks: int = 1) -> GameState:
        if not 0 <= ticks <= 0xFFFF:
            raise ValueError("ticks must fit in a u16")
        reply = self._request(MsgType.STEP, STEP.pack(1 if button else 0, 0, ticks), MsgType.STATE)
        return GameState.from_bytes(reply)

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.sendall(encode_frame(MsgType.BYE))
        except OSError:
            pass
        self._sock.close()
        self._sock = None

    def _request(self, msg_type: MsgType, payload: bytes, expect: MsgType) -> bytes:
        if self._sock is None:
            raise ConnectionError("not connected; call connect() first")
        self._sock.sendall(encode_frame(msg_type, payload))
        reply_type, reply = read_frame(self._sock)
        if reply_type == MsgType.ERROR:
            raise decode_error(reply)
        if reply_type != expect:
            raise ProtocolError(f"expected {expect.name}, got {reply_type.name}")
        return reply

    def __enter__(self) -> GDClient:
        if self._sock is None:
            self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
