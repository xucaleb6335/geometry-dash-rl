"""gdrl.tools.nn_trace (traces for the network visualizer) and the dashboard routes that serve them."""

from __future__ import annotations

import base64
import gzip
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from gdrl.config import load_config
from gdrl.env.factory import make_env
from gdrl.tools import nn_trace as nt

CFG = {
    "env": {"id": "gd", "backend": "mock", "level_ids": [1], "frame_skip": 4, "max_steps": 150,
            "encoder": {"name": "object_list", "k": 8, "focus": "obstacles", "profile": True, "position": True},
            "reward": {"name": "progress"}},
    "agent": {"name": "ppo", "hidden_sizes": [16, 8], "rollout_steps": 64, "device": "cpu"},
}


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    path = tmp_path_factory.mktemp("cfg") / "cfg.yaml"
    path.write_text(json.dumps(CFG), encoding="utf-8")  # JSON is YAML
    cfg = load_config(path)
    env = make_env(dict(cfg["env"]))
    agent = nt.load_agent(cfg, None, env.observation_space, env.action_space)
    states, actions, obs = nt.run_episode(env, agent, -1, log=lambda *_: None)
    env.close()
    meta = {"name": "t", "title": "Test", "level": "mock:1", "source": "mock", "config": str(path)}
    trace = nt.build_trace(states, actions, cfg, None, meta, obs_recorded=obs, agent=agent)
    return cfg, states, actions, obs, trace


def test_pack_roundtrip():
    a = np.array([[-1.5, -0.25, 0.0, 0.5, 2.0]])
    q = nt.unpack(nt.pack(a, "int16", 32767))
    np.testing.assert_allclose(q, np.clip(a, -1, 1), atol=1 / 32767)  # clipped to the int16 range
    i = np.arange(-3, 4, dtype=np.int32)
    assert nt.unpack(nt.pack(i)).tolist() == i.tolist()
    assert json.dumps(nt.pack(np.zeros(3, np.float32)))  # JSON-able


def test_trace_contents(episode):
    cfg, states, actions, obs, t = episode
    n, m = len(actions), t["meta"]
    assert t["format"] == nt.FORMAT and m["decisions"] == n and n > 5
    assert m["greedy_agreement"] == n and m["obs_check"] < 1e-5
    D = t["encoder"]["size"]
    assert t["network"]["sizes"] == [D, 16, 8, 2]
    # quantized observations and activations round-trip
    np.testing.assert_allclose(nt.unpack(t["frames"]["obs"]).reshape(n, D), obs, atol=1 / 32767 + 1e-6)
    hid = [nt.unpack(h) for h in t["frames"]["hidden"]]
    assert [h.shape for h in hid] == [(n, 16), (n, 8)] and all(np.abs(h).max() <= 1 for h in hid)
    # every feature is named and drawn exactly once
    assert len(t["features"]) == D
    cells = sorted(i for g in t["groups"] for row in g["grid"] for i in row)
    assert cells == list(range(D))
    assert {g["id"] for g in t["groups"]} == {"player", "slots", "timing", "terrain", "position"}
    # the logits match the stored weights (the page recomputes knock-outs from them)
    x = obs[n // 2].astype(np.float64)
    for k, L in enumerate(t["network"]["layers"]):
        W, b = nt.unpack(L["W"]), nt.unpack(L["b"])
        x = W @ x + b
        if k < 2:
            x = np.tanh(x)
    np.testing.assert_allclose(x, nt.unpack(t["frames"]["logits"]).reshape(n, 2)[n // 2], atol=1e-4)


def test_objects_view_and_slots(episode):
    cfg, states, actions, obs, t = episode
    O = {k: nt.unpack(v) for k, v in t["objects"].items() if isinstance(v, dict) and "b64" in v}
    n, k = len(actions), t["encoder"]["k"]
    # replaying the view diffs gives exactly the objects each state reported
    ap, ad = nt.unpack(t["view"]["add_ptr"]), nt.unpack(t["view"]["add"])
    dp, dd = nt.unpack(t["view"]["del_ptr"]), nt.unpack(t["view"]["del"])
    cur: set[int] = set()
    slots = nt.unpack(t["frames"]["slots"]).reshape(n, k)
    for f in range(n):
        cur -= set(dd[dp[f]:dp[f + 1]].tolist())
        cur |= set(ad[ap[f]:ap[f + 1]].tolist())
        got = sorted((float(O["x"][i]), float(O["y"][i]), int(O["kind"][i])) for i in cur)
        want = sorted((float(o["x"]), float(o["y"]), int(o["kind"])) for o in states[f].objects)
        assert sorted(set(got)) == sorted(set(want))
        for i in slots[f][slots[f] >= 0]:
            assert i in cur  # slot objects come from this frame's report
    assert (O["first"] <= O["last"]).all()


def test_write_list_export(episode, tmp_path):
    t = episode[4]
    p = nt.write_trace(t, tmp_path / ("demo" + nt.SUFFIX))
    assert nt.read_trace(p)["meta"]["decisions"] == t["meta"]["decisions"]
    lst = nt.list_traces(tmp_path)
    assert [x["name"] for x in lst] == ["demo"] and lst[0]["title"] == "Test"
    html_path = nt.export_html(p)
    page = html_path.read_text(encoding="utf-8")
    assert html_path.name == "demo.html"
    assert "/*__JS__*/" not in page and "/*__CSS__*/" not in page and "<!--__TRACE__-->" not in page
    start = page.index('id="trace-data"')
    blob = page[page.index(">", start) + 1: page.index("</script>", start)]
    assert json.loads(gzip.decompress(base64.b64decode(blob)))["meta"]["name"] == "t"


def test_dashboard_routes(episode, tmp_path):
    from gdrl.dashboard.aggregate import Store
    from gdrl.dashboard.server import make_handler

    nt.write_trace(episode[4], tmp_path / "results" / "viz" / ("demo" + nt.SUFFIX))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Store(tmp_path, tmp_path / "runs")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        req = urllib.request.Request(base + path, headers={"Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read()
                return r.status, gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
        except urllib.error.HTTPError as e:
            return e.code, b""

    try:
        code, page = get("/viz")
        assert code == 200 and b"GD-RL Network" in page
        code, body = get("/api/traces")
        assert code == 200 and [x["name"] for x in json.loads(body)] == ["demo"]
        code, body = get("/api/trace?name=demo")
        assert code == 200 and json.loads(body)["format"] == nt.FORMAT
        assert get("/api/trace?name=..%2Fdemo")[0] == 404
        assert get("/api/trace?name=missing")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
