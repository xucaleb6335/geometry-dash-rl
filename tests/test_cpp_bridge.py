"""The Python client against the C++ Bridge (mod/tools/standalone_server.cpp).

Proves Protocol.hpp and protocol.py agree byte for byte, without the game.
Skipped when no C++ compiler is on PATH.
"""

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from gdrl.env.client import GDClient, read_frame
from gdrl.env.protocol import HELLO, ErrorCode, MsgType, ObjectKind, decode_error, encode_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "mod/tools/standalone_server.cpp", ROOT / "mod/src/Bridge.cpp"]


@pytest.fixture(scope="module")
def cpp_server(tmp_path_factory):
    if sys.platform == "win32":
        pytest.skip("build mod/tools with CMake on Windows; this test compiles with a POSIX toolchain")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler on PATH")
    exe = tmp_path_factory.mktemp("cpp") / "standalone_server"
    subprocess.run(
        [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{ROOT / 'mod/src'}",
         *map(str, SOURCES), "-o", str(exe)],
        check=True,
    )
    proc = subprocess.Popen([str(exe), "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        line = proc.stdout.readline().split()
        assert line and line[0] == "PORT", f"unexpected server output: {line}"
        yield int(line[1])
    finally:
        proc.kill()
        proc.wait()


def test_cpp_handshake_and_state_layout(cpp_server):
    with GDClient(port=cpp_server) as client:
        assert client.info.tick_rate == 240 and client.info.max_objects == 32
        state = client.reset()
        assert state.tick == 0 and state.on_ground and not state.holding
        assert state.player_x == pytest.approx(15.0)
        assert state.level_length == pytest.approx(3000.0)
        assert len(state.objects) == 1
        obj = state.objects[0]
        assert obj["kind"] == ObjectKind.HAZARD
        assert (obj["x"], obj["y"], obj["w"], obj["h"]) == pytest.approx((600.0, 9.0, 9.0, 18.0))

        state = client.step(True, 4)
        assert state.tick == 4 and state.holding and state.player_y == pytest.approx(45.0)

        state = client.step(False, 5000)
        assert state.tick == 1000 and state.dead


def test_cpp_rejects_wrong_version(cpp_server):
    with socket.create_connection(("127.0.0.1", cpp_server), timeout=5) as sock:
        sock.sendall(encode_frame(MsgType.HELLO, HELLO.pack(999, 0)))
        msg_type, payload = read_frame(sock)
    assert msg_type == MsgType.ERROR
    assert decode_error(payload).code == ErrorCode.VERSION_MISMATCH
