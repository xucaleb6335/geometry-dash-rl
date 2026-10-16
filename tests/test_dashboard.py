"""Aggregation logic of the dashboard (gdrl/dashboard/aggregate.py)."""

from __future__ import annotations

import json
import os
import time

from gdrl.dashboard.aggregate import Store, clean, decode_log_bytes, downsample, parse_local_time

import numpy as np


def _mk_run(root, name="20261011-010000_demo-ppo_s1", progress=(0.1, 0.4), live=True):
    d = root / "runs" / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(
        "name: demo-ppo\nenv:\n  id: gd\n  backend: mock\n  level_ids: [1]\nagent:\n  name: ppo\n  rollout_steps: 100\n"
        "train:\n  num_envs: 2\n  total_steps: 100000\n", encoding="utf-8")
    lines = ["step,key,value"]
    for i, p in enumerate(progress):
        s = (i + 1) * 200
        lines += [f"{s},episode/progress_mean,{p}", f"{s},episode/count,5", f"{s},perf/steps_per_second,1000"]
    (d / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not live:
        old = time.time() - 3600
        os.utime(d / "metrics.csv", (old, old))
    return d


def test_helpers():
    assert parse_local_time("2026-10-11 00:28:39 PDT") == parse_local_time("2026-10-11 00:28:39 Pacific Daylight Time")
    assert parse_local_time("garbage") is None
    assert clean({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(2)]}) == {"a": None, "b": 1.5, "c": [2]}
    x, y = downsample(np.arange(5000.0), np.arange(5000.0), 100)
    assert len(x) <= 102 and x[-1] == 4999.0


def test_decode_mixed_utf16():
    mixed = b"plain line\n" + "wide line\r\n".encode("utf-16-le") + b"more\n"
    text = decode_log_bytes(b"\xff\xfe" + mixed)
    assert "plain line" in text and "wide line" in text and "more" in text and "\x00" not in text


def test_incremental_metrics_and_status(tmp_path):
    d = _mk_run(tmp_path)
    store = Store(tmp_path, tmp_path / "runs")
    r = store.runs()[0]
    assert r["status"] == "running" and r["steps"] == 400 and r["levels"] == ["mock:1"]
    assert abs(r["best_progress"] - 0.4) < 1e-9 and r["updates"] == 2  # 400 steps / (100*2 envs)
    with (d / "metrics.csv").open("a") as f:  # live append, including a half written line
        f.write("600,episode/progress_mean,0.7\n600,perf/steps_per_second,1000\n600,episode/pro")
    r = store.runs()[0]
    assert r["steps"] == 600 and abs(r["best_progress"] - 0.7) < 1e-9
    mf = store.metrics[d / "metrics.csv"]
    assert mf.series["episode/progress_mean"][1] == [0.1, 0.4, 0.7]


def test_stopped_and_finished(tmp_path):
    _mk_run(tmp_path, live=False)
    store = Store(tmp_path, tmp_path / "runs")
    assert store.runs()[0]["status"] == "stopped"
    (tmp_path / "runs" / "20261011-010000_demo-ppo_s1" / "checkpoints").mkdir()
    (tmp_path / "runs" / "20261011-010000_demo-ppo_s1" / "checkpoints" / "final.pt").write_bytes(b"")
    store = Store(tmp_path, tmp_path / "runs")
    assert store.runs()[0]["status"] == "finished"


def test_ledger_and_board(tmp_path):
    _mk_run(tmp_path, live=False)
    hp = tmp_path / "holistic_progress"
    hp.mkdir()
    (hp / "level_beats.csv").write_text(
        "beaten_at_pacific,level,level_id,game,difficulty,agent,attempts,generations,env_steps,wall_hours,run,notes\n"
        "2026-10-11 00:28:39 PDT,mock level 1,1,mock,mock,genetic,192,6,,0.003,x,note\n"
        "2026-10-11 04:46:48 Pacific Daylight Time,mock-demon-2,2,mock,mock demon,checkpoint-search,43000,317,5000000,0.01,y,robust replay\n"
        "2026-10-11 05:00:00 PDT,\"mock training levels 1-5 (first completion, any level)\",1-5,mock,mock,ppo,9,,,,z,\n",
        encoding="utf-8")
    store = Store(tmp_path, tmp_path / "runs")
    st = store.state()
    led = {b["level"]: b for b in st["ledger"]}
    assert led["mock level 1"]["keys"] == ["mock:1"] and led["mock-demon-2"]["keys"] == ["mock-demon:2"]
    ranged = next(b for b in st["ledger"] if b["partial"])
    assert ranged["keys"] == [f"mock:{i}" for i in range(1, 6)]
    lv = {l["key"]: l for l in st["levels"]}
    assert lv["mock:1"]["status"] == "beaten" and lv["mock:1"]["beat"]["agent"] == "genetic"
    assert lv["mock-demon:2"]["status"] == "robust-beaten"
    assert lv["mock:2"]["status"] == "training"  # run exists, only a partial-range beat
    assert lv["gd:10565740"]["status"] == "untrained"
    json.dumps(st, allow_nan=False)  # fully JSON safe


def test_results_compare_parses_leniently(tmp_path):
    res = tmp_path / "results" / "m8"
    res.mkdir(parents=True)
    lv = lambda c: {"episodes": 20, "completion_rate": c, "progress_mean": c}
    for seed, c in ((1, 1.0), (2, 0.5)):
        (res / f"a_{seed}.json").write_text(json.dumps({
            "agent": "alpha", "train_levels": [1], "heldout_levels": [101],
            "results": {"greedy": {"levels": {"1": lv(c), "101": lv(c)}}, "sticky": {"levels": {"1": lv(c / 2)}}}}), encoding="utf-8")
    (res / "broken.json").write_text("{not json", encoding="utf-8")
    (res / "other.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    (res / "gen.json").write_text(json.dumps({"agent": "genetic", "train_levels": [1], "heldout_levels": [],
                                              "per_level": {"1": {"solved": True, "generations": 6, "episodes": 192}}}), encoding="utf-8")
    store = Store(tmp_path, tmp_path / "runs")
    rows = {r["agent"]: r for r in store.compare()["rows"]}
    assert abs(rows["alpha"]["greedy_train"] - 0.75) < 1e-9 and rows["alpha"]["n"] == 2
    assert rows["alpha"]["greedy_train_sd"] > 0 and rows["alpha"]["sticky_heldout"] is None
    assert rows["genetic"]["solved"] == 1 and rows["genetic"]["gens_mean"] == 6


def test_curriculum_detail(tmp_path):
    d = _mk_run(tmp_path)
    (d / "config.yaml").write_text(
        "name: demo-ppo\nenv:\n  id: gd\n  backend: socket\n  curriculum:\n    starts: starts.yaml\n    log_path: eps.csv\n    ema: 0.5\n"
        "    p_first: 0.25\nagent:\n  name: ppo\n  rollout_steps: 100\ntrain:\n  num_envs: 1\n", encoding="utf-8")
    (d / "starts.yaml").write_text("level: main:1\nstarts:\n- name: start\n  x: 0\n- name: s1\n  x: 100\n- name: s2\n  x: 200\n", encoding="utf-8")
    now = time.time()
    rows = ["time,start,start_x,final_x,progress,length,completed,reached_next"]
    for k in range(12):  # start 1 is always reached
        rows.append(f"{now - 10 + k * .1},1,100,150,0.5,10,0,1")
    rows.append(f"{now},0,0,40,0.2,5,0,0")
    (d / "eps.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    store = Store(tmp_path, tmp_path / "runs")
    st = store.state()
    r = st["runs"][0]
    assert r["levels"] == ["main:1"] and r["has"]["curriculum"]
    det = store.run_detail(r["id"])
    tiles = {t["name"]: t for t in det["curriculum"]["tiles"]}
    assert tiles["s1"]["status"] == "mastered" and tiles["s2"]["status"] == "locked"
    assert tiles["start"]["status"] == "training" and tiles["start"]["attempts"] == 1
    assert abs(sum(t["weight"] for t in tiles.values()) - 1.0) < 1e-9
    assert det["deaths"]["n"] == 13 and det["deaths"]["best"]["x"] == 150
    assert abs(r["best_progress"] - 0.2) < 1e-9  # from-start best, not the any-start 0.5
