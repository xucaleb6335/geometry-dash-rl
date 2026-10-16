"""The mock game behind a real socket, speaking PROTOCOL.md exactly like the mod.

Use it to test the client and env over TCP before the mod's lockstep loop
exists, and as the reference for how the server side should behave.

    python -m gdrl.env.mock_server --port 5555 --level 3
"""

from __future__ import annotations

import argparse
import socket
import threading

from gdrl.env.client import read_frame
from gdrl.env.mock_game import MockGame, make_mock_level
from gdrl.env.protocol import (
    HELLO,
    HELLO_ACK,
    RESET,
    SNAPSHOT,
    STEP,
    SUPPORTED_VERSIONS,
    ErrorCode,
    MsgType,
    ProtocolError,
    encode_frame,
    error_payload,
    max_objects,
)


class MockServer:
    """Serves one client at a time on 127.0.0.1. ``port=0`` picks a free port."""

    MAX_SNAPSHOTS = 1024

    def __init__(self, port: int = 0, level_seed: int = 1, host: str = "127.0.0.1", **level_kwargs):
        self.default_seed = level_seed
        self._level_kwargs = level_kwargs
        self._listener = socket.create_server((host, port))
        self.port: int = self._listener.getsockname()[1]
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.game = MockGame(make_mock_level(level_seed, **level_kwargs))
        self._slots: dict[int, tuple] = {}

    def start(self) -> MockServer:
        self._thread = threading.Thread(target=self.serve_forever, name="mock-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopping.set()
        try:
            self._listener.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def serve_forever(self) -> None:
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return  # listener closed by stop()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._serve_client(conn)

    def _serve_client(self, conn: socket.socket) -> None:
        handshaken = False
        version = 1  # the version agreed in HELLO; replies use its layouts

        def send(msg_type: MsgType, payload: bytes) -> None:
            conn.sendall(encode_frame(msg_type, payload, version=version))

        def send_state(state) -> None:
            send(MsgType.STATE, state.to_bytes(version))

        while not self._stopping.is_set():
            try:
                msg_type, payload = read_frame(conn)
            except ProtocolError as exc:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, str(exc), version)
                return
            except (ConnectionError, OSError):
                return

            if msg_type == MsgType.HELLO:
                client_version, _ = HELLO.unpack(payload)
                if client_version not in SUPPORTED_VERSIONS:
                    self._send_error(conn, ErrorCode.VERSION_MISMATCH, "protocol version mismatch", version)
                    return
                version = client_version
                last = self.MAX_SNAPSHOTS if version >= 2 else 0
                send(MsgType.HELLO_ACK, HELLO_ACK.pack(version, self.game.tick_rate, max_objects(version), last))
                handshaken = True
            elif msg_type == MsgType.BYE:
                return
            elif not handshaken:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, "send HELLO first", version)
                return
            elif msg_type == MsgType.RESET:
                level_id, start_pos = RESET.unpack(payload)
                # Mock convention: level_id 0 keeps the current level, any other id is a seed.
                if level_id != 0 and level_id != self.game.level.seed:
                    self.game = MockGame(make_mock_level(level_id, **self._level_kwargs))
                    self._slots.clear()
                if start_pos >= len(self.game.level.start_positions):
                    self._send_error(conn, ErrorCode.UNSUPPORTED, f"no Start Pos {start_pos}", version)
                    continue
                send_state(self.game.reset(start_pos))
            elif msg_type == MsgType.STEP:
                button, _, ticks = STEP.unpack(payload)
                send_state(self.game.step(bool(button), ticks))
            elif msg_type in (MsgType.SAVE_STATE, MsgType.LOAD_STATE, MsgType.DROP_STATE):
                slot, _ = SNAPSHOT.unpack(payload)
                if not 0 <= slot < self.MAX_SNAPSHOTS:
                    self._send_error(conn, ErrorCode.NO_SNAPSHOT, f"slot {slot} out of range", version)
                elif msg_type == MsgType.SAVE_STATE:
                    self._slots[slot] = self.game.snapshot()
                    send_state(self.game.state())
                elif msg_type == MsgType.DROP_STATE:
                    self._slots.pop(slot, None)
                    send_state(self.game.state())
                elif slot not in self._slots:
                    self._send_error(conn, ErrorCode.NO_SNAPSHOT, f"slot {slot} is empty", version)
                else:
                    send_state(self.game.restore(self._slots[slot]))
            else:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, f"unexpected {msg_type.name}", version)
                return

    @staticmethod
    def _send_error(conn: socket.socket, code: ErrorCode, message: str, version: int = 1) -> None:
        try:
            conn.sendall(encode_frame(MsgType.ERROR, error_payload(code, message), version=version))
        except OSError:
            pass

    def __enter__(self) -> MockServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock GD server.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--level", type=int, default=1, help="level seed (>= 1)")
    args = parser.parse_args()
    server = MockServer(port=args.port, level_seed=args.level)
    print(f"mock server on 127.0.0.1:{server.port}, level seed {args.level}. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
