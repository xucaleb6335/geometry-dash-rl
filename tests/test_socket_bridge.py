"""The Python client against the mock server, over a real TCP socket."""

import socket

import numpy as np
import pytest

from gdrl.env.client import GDClient, read_frame
from gdrl.env.mock_game import MockGame, generate_level
from gdrl.env.mock_server import MockServer
from gdrl.env.protocol import HELLO, ErrorCode, MsgType, ServerError, decode_error, encode_frame


@pytest.fixture
def server():
    with MockServer(level_seed=4) as srv:
        yield srv


def test_handshake(server):
    with GDClient(port=server.port) as client:
        assert client.info.tick_rate == 240
        assert client.info.max_objects == 32


def test_reset_and_step(server):
    with GDClient(port=server.port) as client:
        state = client.reset()
        assert state.tick == 0 and state.on_ground
        state = client.step(button=False, ticks=4)
        assert state.tick == 4
        peek = client.step(button=False, ticks=0)
        assert peek.tick == 4


def test_socket_matches_in_process(server):
    """Same actions through the socket and in-process give identical states."""
    actions = np.random.default_rng(1).integers(2, size=300).astype(bool)
    game = MockGame(generate_level(4))
    game.reset()
    with GDClient(port=server.port) as client:
        client.reset()
        for a in actions:
            remote = client.step(bool(a), 4)
            local = game.step(bool(a), 4)
            assert remote.to_bytes() == local.to_bytes()
            if local.terminal:
                break


def test_reset_switches_level(server):
    with GDClient(port=server.port) as client:
        first = client.reset(level_id=4)
        other = client.reset(level_id=9)
        assert first.objects[0]["x"] != other.objects[0]["x"] or len(first.objects) != len(other.objects)


def test_version_mismatch_is_rejected(server):
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as sock:
        sock.sendall(encode_frame(MsgType.HELLO, HELLO.pack(999, 0)))
        msg_type, payload = read_frame(sock)
    assert msg_type == MsgType.ERROR
    assert decode_error(payload).code == ErrorCode.VERSION_MISMATCH


def test_step_before_hello_is_rejected(server):
    client = GDClient(port=server.port)
    client._sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    with pytest.raises(ServerError):
        client.step(True, 1)
    client._sock.close()
