"""gdctl pure-Python parts: framing, level specs, level strings, PNG, mod
flags, arg parsing, and a round trip against a fake control server. No game
needed.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import zlib

import pytest

from gdctl import bridge_probe, client, levelstring, mods, png
from gdctl.client import ControlError, LineReader, decode_reply, encode_request

# --------------------------------------------------------------------------
# framing


def test_encode_request_is_one_json_line():
    data = encode_request("click", {"path": "a/b", "force": True, "skip": None}, req_id=7, timeout=3)
    assert data.endswith(b"\n") and data.count(b"\n") == 1
    msg = json.loads(data)
    assert msg == {"cmd": "click", "seq": 7, "timeout": 3, "path": "a/b", "force": True}


def test_encode_request_level_id_is_a_plain_parameter():
    msg = json.loads(encode_request("open_level", {"type": "main", "id": 1}, req_id=3))
    assert msg["id"] == 1 and msg["seq"] == 3


def test_encode_request_rejects_reserved_keys():
    with pytest.raises(ValueError):
        encode_request("ping", {"cmd": "x"})
    with pytest.raises(ValueError):
        encode_request("ping", {"seq": 1})


def test_line_reader_splits_partial_chunks():
    r = LineReader()
    assert r.feed(b'{"a":1}\n{"b"') == [b'{"a":1}']
    assert r.pending == 4
    assert r.feed(b':2}\r\n\n{"c":3}\n') == [b'{"b":2}', b'{"c":3}']
    assert r.pending == 0


def test_decode_reply_errors():
    assert decode_reply(b'{"ok":true}') == {"ok": True}
    with pytest.raises(ControlError):
        decode_reply(b"not json")
    with pytest.raises(ControlError):
        decode_reply(b"[1,2]")


class FakeControlServer:
    """Speaks the control protocol: echoes the request back as the reply."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.requests = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                reader = LineReader()
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    for line in reader.feed(chunk):
                        req = json.loads(line)
                        self.requests.append(req)
                        if req["cmd"] == "silent":
                            continue
                        reply = {"ok": req["cmd"] != "fail", "seq": req.get("seq"), "cmd": req["cmd"], "echo": req}
                        # Send in two pieces to exercise reassembly.
                        data = (json.dumps(reply) + "\n").encode()
                        conn.sendall(data[:5])
                        conn.sendall(data[5:])

    def close(self):
        self.sock.close()


@pytest.fixture
def fake_server():
    server = FakeControlServer()
    yield server
    server.close()


def test_request_round_trip(fake_server):
    reply = client.request("tree", {"depth": 3}, timeout=2, port=fake_server.port)
    assert reply["ok"] is True and reply["cmd"] == "tree"
    assert reply["echo"]["depth"] == 3 and reply["echo"]["timeout"] == 2
    assert client.request("fail", port=fake_server.port, timeout=2)["ok"] is False


def test_request_unreachable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens there now
    with pytest.raises(ControlError) as info:
        client.request("ping", port=port, timeout=1, connect_timeout=0.5)
    assert info.value.code == "unreachable"
    assert client.reachable(port=port, timeout=0.5) is False


def test_request_timeout(fake_server):
    with pytest.raises(ControlError) as info:
        client.request("silent", port=fake_server.port, timeout=-4.5)  # socket waits timeout + 5 = 0.5 s
    assert info.value.code == "timeout"


# --------------------------------------------------------------------------
# level specs (game.py imports win32, which needs Windows)

game = pytest.importorskip("gdctl.game") if sys.platform == "win32" else None


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.game needs Windows")
@pytest.mark.parametrize("spec,expected", [
    ("1", {"type": "main", "id": 1}),
    ("main:22", {"type": "main", "id": 22}),
    ("local:gdrl-test-a", {"type": "local", "name": "gdrl-test-a"}),
    ("local:name:with:colons", {"type": "local", "name": "name:with:colons"}),
    ("local-id:42", {"type": "local", "id": 42}),
    ("local-index:0", {"type": "local", "index": 0}),
    ("saved:128", {"type": "saved", "id": 128}),
    ("online:128", {"type": "saved", "id": 128}),
])
def test_parse_level_spec(spec, expected):
    assert game.parse_level_spec(spec) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.game needs Windows")
@pytest.mark.parametrize("spec", ["", "main:", "main:x", "foo:1", "local"])
def test_parse_level_spec_errors(spec):
    with pytest.raises(ValueError):
        game.parse_level_spec(spec)


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.game needs Windows")
def test_summarize_build_errors():
    text = "\n".join([
        "  Compiling...",
        r"C:\x\src\main.cpp(12,5): error C2065: 'foo': undeclared identifier [C:\x\build\X.vcxproj]",
        r"C:\x\src\main.cpp(12,5): error C2065: 'foo': undeclared identifier [C:\x\build\X.vcxproj]",
        r"C:\x\a.cpp(1,1): warning C4100: unused",
        "CMake Error at CMakeLists.txt:3 (project):",
    ])
    errors = game.summarize_build_errors(text)
    assert errors == [r"C:\x\src\main.cpp(12,5): error C2065: 'foo': undeclared identifier",
                      "CMake Error at CMakeLists.txt:3 (project):"]


# --------------------------------------------------------------------------
# level strings


def test_level_string_round_trip():
    raw = levelstring.test_level(length_blocks=30, spikes_every=5)
    packed = levelstring.compress(raw)
    assert packed.startswith("H4sI") and levelstring.is_compressed(packed)
    assert "+" not in packed and "/" not in packed  # URL-safe alphabet
    assert levelstring.decompress(packed) == raw
    assert levelstring.decompress(raw) == raw


def test_level_string_objects():
    raw = levelstring.build([levelstring.obj(levelstring.SPIKE, 315, 15), levelstring.obj(1, 345.5, 45, k6=90)])
    objs = levelstring.parse_objects(levelstring.compress(raw))
    assert objs == [{"1": "8", "2": "315", "3": "15"}, {"1": "1", "2": "345.5", "3": "45", "6": "90"}]


def test_decompress_unpadded_gd_data():
    packed = levelstring.compress("kA2,0;1,8,2,525,3,15;").rstrip("=")
    assert levelstring.decompress(packed) == "kA2,0;1,8,2,525,3,15;"


# --------------------------------------------------------------------------
# PNG


def test_png_encode_decodes():
    w, h = 3, 2
    rgb = bytes(range(w * h * 3))
    data = png.encode_rgb(w, h, rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR then IDAT; inflate IDAT and strip the per-row filter bytes.
    ihdr_len = struct.unpack(">I", data[8:12])[0]
    assert data[12:16] == b"IHDR" and struct.unpack(">II", data[16:24]) == (w, h)
    pos = 8 + 12 + ihdr_len
    idat_len = struct.unpack(">I", data[pos:pos + 4])[0]
    assert data[pos + 4:pos + 8] == b"IDAT"
    raw = zlib.decompress(data[pos + 8:pos + 8 + idat_len])
    rows = [raw[i * (w * 3 + 1):(i + 1) * (w * 3 + 1)] for i in range(h)]
    assert all(r[0] == 0 for r in rows)
    assert b"".join(r[1:] for r in rows) == rgb


def test_bgra_to_rgb_and_downscale():
    bgra = bytes([1, 2, 3, 255, 4, 5, 6, 255])  # 2x1
    assert png.bgra_to_rgb(2, 1, bgra) == bytes([3, 2, 1, 6, 5, 4])
    w, h, out = png.downscale_rgb(4, 2, bytes(range(24)), 2)
    assert (w, h) == (2, 1) and len(out) == 6


# --------------------------------------------------------------------------
# mod flags (saved.json)


def test_set_flags_round_trip(tmp_path):
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"should-load-a.b": True, "other": 1}), encoding="utf-8")
    res = mods.set_flags({"a.b": False, "c.d": False}, path=saved)
    assert res["before"] == {"a.b": True, "c.d": True}
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data == {"should-load-a.b": False, "other": 1, "should-load-c.d": False}
    assert res["backup"] and json.loads(open(res["backup"], encoding="utf-8").read())["should-load-a.b"] is True
    assert mods.enabled_flags(data, ["a.b", "c.d", "e.f"]) == {"a.b": False, "c.d": False, "e.f": True}


def test_installed_mods_reads_mod_json(tmp_path):
    import zipfile

    with zipfile.ZipFile(tmp_path / "x.y.geode", "w") as z:
        z.writestr("mod.json", json.dumps({"id": "x.y", "name": "XY", "version": "v1.0.0"}))
    (tmp_path / "broken.geode").write_bytes(b"not a zip")
    found = {m["id"]: m for m in mods.installed_mods(tmp_path)}
    assert found["x.y"]["name"] == "XY" and found["broken"]["name"] is None


# --------------------------------------------------------------------------
# RL bridge probe framing


def test_bridge_frame_and_state_parse():
    f = bridge_probe.frame(bridge_probe.T_HELLO, struct.pack("<HH", 1, 0))
    assert f[:12] == struct.pack("<IHHI", 0x4C524447, 1, 1, 4)
    state = struct.pack("<II6f", 0, 4, 15.0, 0.0, 0.0, 0.0, 311.58, 9000.0) + struct.pack("<BBH", 0, 0, 3)
    parsed = bridge_probe.parse_state(state + bytes(676 - len(state)))
    assert parsed["tick"] == 0 and parsed["dead"] is False and parsed["n_objects"] == 3
    err = struct.pack("<HH", 3, 0) + b"open a level first".ljust(64, b"\0")
    assert bridge_probe.parse_error(err)["name"] == "NOT_IN_LEVEL"


def test_bridge_probe_unreachable():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    res = bridge_probe.probe(port, timeout=0.5)
    assert res["ok"] is False and res["stage"] == "connect"


# --------------------------------------------------------------------------
# CLI argument parsing (no game: only commands that fail before any I/O)

cli = pytest.importorskip("gdctl.cli") if sys.platform == "win32" else None


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.cli imports Windows-only modules")
def test_cli_usage_errors_are_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GDCTL_LOCK", str(tmp_path / "game.lock"))
    code, out = cli.run(["no-such-command"])
    assert code == 2 and out["ok"] is False and out["code"] == "usage"
    code, out = cli.run(["click"])
    assert code == 2 and "click needs" in out["error"]
    code, out = cli.run(["open-level", "foo:1"])
    assert code == 2 and out["code"] == "usage"
    code, out = cli.run(["delete-level", "main:1"])
    assert code == 2


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.cli imports Windows-only modules")
def test_cli_parser_shapes():
    p = cli.build_parser()
    a = p.parse_args(["tree", "--clickable", "--match", "play", "--depth", "5"])
    assert a.clickable and a.match == "play" and a.depth == 5 and a.func is cli.c_tree
    a = p.parse_args(["build", "mod", "--no-relaunch"])
    assert a.targets == ["mod"] and a.no_relaunch
    a = p.parse_args(["build"])
    assert a.targets == []
    a = p.parse_args(["prepare-training", "--level", "main:1"])
    assert a.level == "main:1" and not a.practice
    a = p.parse_args(["input", "click", "--fx", "0.5", "--fy", "0.25"])
    assert a.action == "click" and a.fx == 0.5
    a = p.parse_args(["mods", "disable", "syzzi.click_between_frames", "--restart"])
    assert a.action == "disable" and a.mod_ids == ["syzzi.click_between_frames"] and a.restart


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.cli imports Windows-only modules")
def test_cli_proxies_to_control_mod(fake_server, monkeypatch, tmp_path):
    from gdctl import paths

    monkeypatch.setattr(paths, "CONTROL_PORT", fake_server.port)
    monkeypatch.setenv("GDCTL_LOCK", str(tmp_path / "game.lock"))  # no lock held
    code, out = cli.run(["open-level", "local:gdrl-test-x", "--practice"])
    assert code == 0 and out["cmd"] == "open_level"
    assert out["echo"]["type"] == "local" and out["echo"]["name"] == "gdrl-test-x" and out["echo"]["practice"]
    code, out = cli.run(["create-level", "--name", "gdrl-test-y", "--test"])
    assert code == 0 and levelstring.is_compressed(out["echo"]["level_string"])
    code, out = cli.run(["raw", '{"cmd": "fail", "x": 1}'])
    assert code == 1 and out["echo"]["x"] == 1


# --------------------------------------------------------------------------
# level editing (StartPos)

from gdctl import leveledit, lock  # noqa: E402


def _level():
    objs = [levelstring.obj(13, 300, 105),     # ship portal
            levelstring.obj(202, 600, 105),    # 2x speed portal
            levelstring.obj(11, 900, 105),     # yellow gravity portal
            levelstring.obj(levelstring.BLOCK, 1500, 15)]
    return levelstring.build(objs)


def test_infer_state_from_portals():
    raw = _level()
    assert leveledit.infer_state(raw, 100).describe()["mode_name"] == "cube"
    st = leveledit.infer_state(raw, 700)
    assert st.mode == 1 and st.speed == 2 and not st.flipped
    assert leveledit.infer_state(levelstring.compress(raw), 1000).flipped


def test_startpos_add_set_remove():
    raw = _level()
    raw, st = leveledit.add_startpos(raw, 700, 45)
    raw, _ = leveledit.add_startpos(raw, 1200, 15, leveledit.make_state("wave", "3x"))
    sps = leveledit.list_startpos(raw)
    assert [(s["x"], s["mode_name"], s["speed_name"]) for s in sps] == [(700, "ship", "2x"), (1200, "wave", "3x")]
    assert "1,31,2,700,3,45,kA2,1,kA3,0,kA8,0,kA4,2,kA9,1,kA10,0,kA11,0" in raw
    raw2, st2, removed = leveledit.set_only_startpos(raw, 1000)
    assert removed == 2 and [s["x"] for s in leveledit.list_startpos(raw2)] == [1000]
    assert st2.flipped and st2.mode == 1
    raw3, n = leveledit.remove_startpos(raw, [705])
    assert n == 1 and [s["x"] for s in leveledit.list_startpos(raw3)] == [1200]
    info = leveledit.info(raw)
    assert info["objects"] == 6 and info["last_object_x"] == 1500 and info["startpos_count"] == 2
    assert info["portals"] == 3


def test_make_state_overrides():
    base = leveledit.StartState(mode=1, speed=2)
    st = leveledit.make_state(speed="0.5x", flipped=True, base=base)
    assert (st.mode, st.speed, st.flipped) == (1, 1, True) and base.speed == 2


# --------------------------------------------------------------------------
# game lock


def test_lock_acquire_release(tmp_path):
    p = tmp_path / "game.lock"
    assert lock.acquire("a", path=p)["action"] == "acquired"
    assert lock.acquire("a", path=p)["action"] == "refreshed"
    res = lock.acquire("b", path=p)
    assert not res["ok"] and res["code"] == "locked" and res["lock"]["owner"] == "a"
    assert lock.check("b", path=p)["code"] == "locked" and lock.check("a", path=p) is None
    assert lock.release("b", path=p)["code"] == "not_owner"
    assert lock.release("a", path=p)["action"] == "released"
    assert lock.check("b", path=p) is None
    assert lock.release("a", path=p)["action"] == "not_held"


def test_lock_stale_takeover_and_wait(tmp_path):
    p = tmp_path / "game.lock"
    lock.acquire("old", path=p)
    data = json.loads(p.read_text())
    data["refreshed"] -= lock.STALE_SECONDS + 10
    p.write_text(json.dumps(data))
    assert lock.check("new", path=p) is None
    res = lock.acquire("new", path=p)
    assert res["action"] == "took_over_stale" and res["previous"]["owner"] == "old"
    t0 = __import__("time").monotonic()
    assert lock.acquire("third", wait=0.3, path=p)["code"] == "locked"
    assert __import__("time").monotonic() - t0 >= 0.25


@pytest.mark.skipif(sys.platform != "win32", reason="gdctl.cli imports Windows-only modules")
def test_cli_enforces_lock(tmp_path, monkeypatch, fake_server):
    from gdctl import paths

    monkeypatch.setenv("GDCTL_LOCK", str(tmp_path / "game.lock"))
    monkeypatch.setattr(paths, "CONTROL_PORT", fake_server.port)
    assert cli.run(["--owner", "a", "lock", "acquire"])[1]["ok"]
    code, out = cli.run(["--owner", "b", "close-popups"])
    assert code == 1 and out["code"] == "locked"
    assert cli.run(["--owner", "b", "scene"])[0] == 0          # read-only is allowed
    assert cli.run(["--owner", "a", "close-popups"])[0] == 0
    assert cli.run(["--owner", "a", "lock", "release"])[1]["action"] == "released"
