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
from gdrl.env.mock_game import MockGame, generate_level
from gdrl.env.protocol import (
    HELLO,
    HELLO_ACK,
    MAX_OBJECTS,
    PROTOCOL_VERSION,
    RESET,
    STEP,
    ErrorCode,
    MsgType,
    ProtocolError,
    encode_frame,
    error_payload,
)


class MockServer:
    """Serves one client at a time on 127.0.0.1. ``port=0`` picks a free port."""

    def __init__(self, port: int = 0, level_seed: int = 1, host: str = "127.0.0.1"):
        self.default_seed = level_seed
        self._listener = socket.create_server((host, port))
        self.port: int = self._listener.getsockname()[1]
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.game = MockGame(generate_level(level_seed))

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
        while not self._stopping.is_set():
            try:
                msg_type, payload = read_frame(conn)
            except ProtocolError as exc:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, str(exc))
                return
            except (ConnectionError, OSError):
                return

            if msg_type == MsgType.HELLO:
                client_version, _ = HELLO.unpack(payload)
                if client_version != PROTOCOL_VERSION:
                    self._send_error(conn, ErrorCode.VERSION_MISMATCH, "protocol version mismatch")
                    return
                ack = HELLO_ACK.pack(PROTOCOL_VERSION, self.game.tick_rate, MAX_OBJECTS, 0)
                conn.sendall(encode_frame(MsgType.HELLO_ACK, ack))
                handshaken = True
            elif msg_type == MsgType.BYE:
                return
            elif not handshaken:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, "send HELLO first")
                return
            elif msg_type == MsgType.RESET:
                level_id, start_pos = RESET.unpack(payload)
                if start_pos != -1:
                    self._send_error(conn, ErrorCode.UNSUPPORTED, "mock has no Start Pos objects")
                    continue
                # Mock convention: level_id 0 keeps the current level, any other id is a seed.
                if level_id != 0 and level_id != self.game.level.seed:
                    self.game = MockGame(generate_level(level_id))
                state = self.game.reset()
                conn.sendall(encode_frame(MsgType.STATE, state.to_bytes()))
            elif msg_type == MsgType.STEP:
                button, _, ticks = STEP.unpack(payload)
                state = self.game.step(bool(button), ticks)
                conn.sendall(encode_frame(MsgType.STATE, state.to_bytes()))
            else:
                self._send_error(conn, ErrorCode.BAD_MESSAGE, f"unexpected {msg_type.name}")
                return

    @staticmethod
    def _send_error(conn: socket.socket, code: ErrorCode, message: str) -> None:
        try:
            conn.sendall(encode_frame(MsgType.ERROR, error_payload(code, message)))
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
