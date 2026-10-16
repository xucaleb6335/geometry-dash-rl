"""Talk to the RL bridge (port 5555, protocol/PROTOCOL.md) just enough to
check it is ready: HELLO, RESET(0, -1), BYE. Standalone on purpose (no gdrl
import), so gdctl works without the training stack.
"""

from __future__ import annotations

import socket
import struct

MAGIC = 0x4C524447
VERSION = 1
HEADER = struct.Struct("<IHHI")
T_HELLO, T_HELLO_ACK, T_RESET, T_STEP, T_STATE, T_ERROR, T_BYE = 1, 2, 3, 4, 5, 6, 7
ERROR_NAMES = {1: "VERSION_MISMATCH", 2: "BAD_MESSAGE", 3: "NOT_IN_LEVEL", 4: "UNSUPPORTED"}


def frame(msg_type: int, payload: bytes = b"") -> bytes:
    return HEADER.pack(MAGIC, VERSION, msg_type, len(payload)) + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[int, bytes]:
    magic, _version, msg_type, length = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ConnectionError(f"bad magic 0x{magic:08x}")
    return msg_type, _recv_exact(sock, length)


def parse_error(payload: bytes) -> dict:
    code = struct.unpack_from("<H", payload, 0)[0]
    message = payload[4:68].split(b"\0", 1)[0].decode("utf-8", "replace")
    return {"code": code, "name": ERROR_NAMES.get(code, "?"), "message": message}


def parse_state(payload: bytes) -> dict:
    tick, flags, px, py, vy, rot, xs, length = struct.unpack_from("<II6f", payload, 0)
    mode, _res, n_obj = struct.unpack_from("<BBH", payload, 32)
    return {"tick": tick, "flags": flags, "dead": bool(flags & 1), "completed": bool(flags & 2),
            "player_x": round(px, 2), "player_y": round(py, 2), "x_speed": round(xs, 2),
            "level_length": round(length, 1), "mode": mode, "n_objects": n_obj}


def probe(port: int = 5555, timeout: float = 5.0, reset: bool = True) -> dict:
    """HELLO (+ RESET) + BYE. Returns {"ok": bool, ...}; never raises."""
    out: dict = {"port": port}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(frame(T_HELLO, struct.pack("<HH", VERSION, 0)))
            t, payload = read_message(sock)
            if t == T_ERROR:
                out.update(ok=False, stage="hello", error=parse_error(payload))
                return out
            if t != T_HELLO_ACK:
                out.update(ok=False, stage="hello", error=f"unexpected message type {t}")
                return out
            version, tick_rate, max_objects, _ = struct.unpack("<HHHH", payload)
            out.update(protocol=version, tick_rate=tick_rate, max_objects=max_objects)
            if reset:
                sock.sendall(frame(T_RESET, struct.pack("<ii", 0, -1)))
                t, payload = read_message(sock)
                if t == T_ERROR:
                    out.update(ok=False, stage="reset", error=parse_error(payload))
                    sock.sendall(frame(T_BYE))
                    return out
                out["state"] = parse_state(payload)
            sock.sendall(frame(T_BYE))
            out["ok"] = True
            return out
    except (OSError, ConnectionError, struct.error) as exc:
        out.update(ok=False, stage="connect", error=str(exc))
        return out
