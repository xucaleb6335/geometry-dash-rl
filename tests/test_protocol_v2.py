"""Protocol v2: version negotiation, snapshots and Start Pos, against the mock server."""

import socket

import numpy as np
import pytest

from gdrl.env.backends import MockBackend, SocketBackend
from gdrl.env.client import GDClient, read_frame
from gdrl.env.mock_server import MockServer
from gdrl.env.protocol import (
    HELLO,
    HELLO_ACK,
    STATE_DTYPE_V2,
    ErrorCode,
    GameState,
    MsgType,
    ObjectSubtype,
    PlayerMode,
    ServerError,
    StateFlag,
    encode_frame,
)
from tests.conftest import make_state


def test_v2_state_round_trip_keeps_new_fields():
    state = make_state(flags=StateFlag.MINI | StateFlag.DUAL, player_size=18.0, floor_y=0.0, ceil_y=240.0,
                       p2_x=10.0, p2_y=200.0, p2_vel_y=-50.0, p2_mode=PlayerMode.WAVE,
                       p2_flags=StateFlag.UPSIDE_DOWN)
    state.objects["subtype"][0] = ObjectSubtype.SPIKE
    raw = state.to_bytes(2)
    assert len(raw) == STATE_DTYPE_V2.itemsize == 1344
    back = GameState.from_bytes(raw)
    assert back.mini and back.dual
    assert back.player_size == 18.0 and back.ceil_y == 240.0
    assert back.p2_mode == PlayerMode.WAVE and back.p2_flags == StateFlag.UPSIDE_DOWN
    assert back.objects[0]["subtype"] == ObjectSubtype.SPIKE
    # v1 bytes of the same state are unchanged by v2's existence.
    assert len(state.to_bytes()) == 676


def test_old_client_still_gets_v1():
    with MockServer(level_seed=2) as server, socket.create_connection(("127.0.0.1", server.port), timeout=5) as s:
        s.sendall(encode_frame(MsgType.HELLO, HELLO.pack(1, 0)))
        msg_type, payload = read_frame(s)
        assert msg_type == MsgType.HELLO_ACK
        version, _, max_objects, reserved = HELLO_ACK.unpack(payload)
        assert (version, max_objects, reserved) == (1, 32, 0)


class V1OnlyServer(MockServer):
    """A server that, like the unpatched mod, only speaks version 1."""

    def _serve_client(self, conn):
        msg_type, payload = read_frame(conn)
        version, _ = HELLO.unpack(payload)
        if version != 1:
            self._send_error(conn, ErrorCode.VERSION_MISMATCH, "protocol version mismatch")
            return
        conn.sendall(encode_frame(MsgType.HELLO_ACK, HELLO_ACK.pack(1, 240, 32, 0)))
        while True:
            try:
                msg_type, payload = read_frame(conn)
            except (ConnectionError, OSError):
                return
            if msg_type == MsgType.BYE:
                return
            conn.sendall(encode_frame(MsgType.STATE, self.game.reset().to_bytes(1)))


def test_client_falls_back_to_v1():
    with V1OnlyServer(level_seed=2) as server:
        with GDClient(port=server.port) as client:
            assert client.info.protocol_version == 1 and client.version == 1
            assert not client.supports_snapshots
            assert client.reset().tick == 0
            with pytest.raises(ServerError):
                client.save_state(0)


def test_socket_snapshots_restore_exactly():
    with MockServer(level_seed=4) as server:
        backend = SocketBackend(port=server.port)
        assert backend.max_snapshots > 0
        backend.reset()
        backend.step(False, 40)
        saved = backend.save_state(3)
        actions = np.random.default_rng(0).integers(2, size=60).astype(bool)
        first = [backend.step(bool(a), 2).to_bytes(2) for a in actions]
        restored = backend.load_state(3)
        assert restored.to_bytes(2) == saved.to_bytes(2)
        second = [backend.step(bool(a), 2).to_bytes(2) for a in actions]
        assert first == second
        with pytest.raises(ServerError) as exc:
            backend.load_state(7)
        assert exc.value.code == ErrorCode.NO_SNAPSHOT
        backend.close()


def test_socket_matches_in_process_backend_on_demon_level():
    kwargs = {"kind": "demon", "n_sections": 4}
    actions = np.random.default_rng(3).integers(2, size=400).astype(bool)
    local = MockBackend(level_seed=7, **kwargs)
    with MockServer(level_seed=7, **kwargs) as server:
        remote = SocketBackend(port=server.port)
        assert remote.reset().to_bytes(2) == local.reset().to_bytes(2)
        for a in actions:
            r, l_ = remote.step(bool(a), 2), local.step(bool(a), 2)
            assert r.to_bytes(2) == l_.to_bytes(2)
            if l_.terminal:
                break
        remote.close()


def test_start_pos_reset():
    backend = MockBackend(level_seed=5, kind="demon", n_sections=6)
    starts = sorted(backend.game.level.start_positions, key=lambda s: s.x)
    assert len(starts) >= 2
    state = backend.reset(start_pos=1)
    assert state.tick == 0
    assert state.player_x == pytest.approx(starts[1].x)
    assert state.mode == starts[1].mode
