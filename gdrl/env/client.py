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
    SNAPSHOT,
    STEP,
    ErrorCode,
    GameState,
    MsgType,
    ProtocolError,
    ServerError,
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
    max_snapshots: int = 0  # v2: SAVE_STATE slots 0 .. max_snapshots - 1; 0 = no snapshots


class GDClient:
    """Blocking request/reply client. One instance per game connection.

    >>> with GDClient(port=5555) as c:
    ...     state = c.reset()
    ...     state = c.step(button=True, ticks=4)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float | None = 10.0,
                 version: int = PROTOCOL_VERSION):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.version = version  # asked for; lowered to 1 if the server rejects it
        self._sock: socket.socket | None = None
        self.info: ServerInfo | None = None

    def connect(self) -> ServerInfo:
        """Handshake at ``version``; if the server says VERSION_MISMATCH, retry at 1."""
        try:
            self.info = self._connect_at(self.version)
        except ServerError as exc:
            if exc.code != ErrorCode.VERSION_MISMATCH or self.version == 1:
                raise
            self._drop()
            self.version = 1
            self.info = self._connect_at(1)
        return self.info

    def _connect_at(self, version: int) -> ServerInfo:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self.version = version
        reply = self._request(MsgType.HELLO, HELLO.pack(version, 0), MsgType.HELLO_ACK)
        server_version, tick_rate, max_objects, last = HELLO_ACK.unpack(reply)
        return ServerInfo(server_version, tick_rate, max_objects, last if server_version >= 2 else 0)

    def _drop(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def supports_snapshots(self) -> bool:
        return self.info is not None and self.info.protocol_version >= 2 and self.info.max_snapshots > 0

    def save_state(self, slot: int) -> GameState:
        """v2: snapshot the current game state into ``slot`` (overwriting it). Replies the current STATE."""
        return self._snapshot(MsgType.SAVE_STATE, slot)

    def load_state(self, slot: int) -> GameState:
        """v2: restore the snapshot in ``slot``. Replies the restored STATE (its tick is the saved tick)."""
        return self._snapshot(MsgType.LOAD_STATE, slot)

    def drop_state(self, slot: int) -> GameState:
        """v2: free ``slot``. Replies the current STATE unchanged."""
        return self._snapshot(MsgType.DROP_STATE, slot)

    def _snapshot(self, msg_type: MsgType, slot: int) -> GameState:
        if not self.supports_snapshots:
            raise ServerError(ErrorCode.UNSUPPORTED, "the server does not support snapshots (protocol v1?)")
        return GameState.from_bytes(self._request(msg_type, SNAPSHOT.pack(slot, 0), MsgType.STATE))

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
            self._sock.sendall(encode_frame(MsgType.BYE, version=self.version))
        except OSError:
            pass
        self._sock.close()
        self._sock = None

    def _request(self, msg_type: MsgType, payload: bytes, expect: MsgType) -> bytes:
        if self._sock is None:
            raise ConnectionError("not connected; call connect() first")
        self._sock.sendall(encode_frame(msg_type, payload, version=self.version))
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
