"""Regenerate the synthetic fixtures under gdrl/dashboard/demo/runs/ (deterministic).

    python gdrl/dashboard/demo/make_demo.py

These are FAKE: they only exist so views that need extreme-demon style data
(many curriculum segments, a long search run) can be previewed with --demo.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
T0 = 1791651600.0  # fixed fake clock (2026-10-10 10:00 PDT)


def write_curriculum_run() -> None:
    d = RUNS / "20261010-100000_demo-bloodbath-segments_s1"
    shutil.rmtree(d, ignore_errors=True)
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "final.pt").write_bytes(b"")
    rng = np.random.default_rng(7)
    length, n = 96000.0, 38
    starts = [{"name": "start", "spec": "bloodbath", "x": 0.0}] + [
        {"name": f"bb-s{i:02d}", "spec": f"local:bb-s{i:02d}", "x": float(round(i * length / n, 0)), "mode": ["cube", "ship", "ball", "ufo", "wave", "robot"][i % 6]}
        for i in range(1, n)
    ]
    (d / "starts.yaml").write_text("level: bloodbath\nstarts:\n" + "".join(
        f"- name: {s['name']}\n  spec: {s['spec']}\n  x: {s['x']}\n" + (f"  mode: {s['mode']}\n" if 'mode' in s else "") for s in starts), encoding="utf-8")
    (d / "config.yaml").write_text(
        "name: demo-bloodbath-segments\nseed: 1\nenv:\n  id: gd\n  backend: socket\n  level_ids: [0]\n  curriculum:\n"
        "    starts: starts.yaml\n    p_first: 0.2\n    min_weight: 0.1\n    frontier_extra: 3\n    log_path: episodes.csv\n"
        "agent:\n  name: ppo\n  rollout_steps: 4096\ntrain:\n  num_envs: 1\n  total_steps: 6000000\n", encoding="utf-8")
    # episodes: segment i is mastered around episode ~ 90*i, harder ones later
    succ = np.zeros(n)
    rows = ["time,start,start_x,final_x,progress,length,completed,reached_next"]
    t = T0
    skill = 0.0
    reach = []
    for ep in range(5200):
        skill += 0.0036 * (1.0 if ep < 3600 else 0.6)
        opened = min(n, 1 + int(skill) + 3)
        w = np.maximum(0.1, 1 - succ)
        w[opened:] = 0
        p = 0.8 * w / w.sum()
        p[0] += 0.2
        i = int(rng.choice(n, p=p / p.sum()))
        diff = 0.55 + 0.42 * math.sin(i * 1.3) ** 2 + i * 0.012
        pr = 1 / (1 + math.exp(-(skill - i * 0.9 + 3 - diff * 3.2) * 1.6))
        nxt = starts[i + 1]["x"] if i + 1 < n else length
        ok = rng.random() < pr
        sx = starts[i]["x"]
        fx = nxt + rng.random() * 400 if ok else sx + (nxt - sx) * min(0.98, rng.beta(1.6, 3.2) + 0.08)
        succ[i] += 0.1 * (float(ok) - succ[i])
        if i == 0:
            reach.append(fx)
        comp = 1 if (i == n - 1 and ok) else 0
        t += rng.uniform(0.8, 3.2)
        rows.append(f"{t:.1f},{i},{sx},{fx:.1f},{min(1, fx / length):.4f},{int((fx - sx) / 12)},{comp},{int(ok)}")
    (d / "episodes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # PPO metrics
    out = ["step,key,value"]
    steps = np.arange(1, 1200) * 4096
    for k, s in enumerate(steps):
        f = k / len(steps)
        prog = 0.02 + 0.2 * (1 - math.exp(-4 * f)) + rng.normal(0, 0.006)
        vals = {
            "episode/progress_mean": prog + 0.25 * f, "episode/return_mean": -0.9 + 1.3 * f + rng.normal(0, .03),
            "episode/completed_mean": max(0.0, (f - .85) * 0.4), "episode/count": 30 + rng.integers(0, 10),
            "episode/length_mean": 200 + 700 * f, "train/entropy": 0.6 - 0.35 * f + rng.normal(0, .01),
            "train/approx_kl": abs(0.004 * math.exp(-2 * f) + rng.normal(0, .0006)) + 1e-4,
            "train/value_loss": 0.08 * math.exp(-3 * f) + 0.005 + abs(rng.normal(0, .003)),
            "train/explained_variance": 0.2 + 0.7 * (1 - math.exp(-4 * f)) + rng.normal(0, .03),
            "train/policy_loss": rng.normal(0, .004), "train/clipfrac": 0.1 * math.exp(-f), "train/learning_rate": 2.5e-4 * (1 - f),
            "perf/steps_per_second": 560 + rng.normal(0, 25),
        }
        for key, v in vals.items():
            out.append(f"{s},{key},{v}")
        if k % 20 == 0:
            out.append(f"{s},eval/progress,{0.02 + 0.2 * (1 - math.exp(-4 * f)) + rng.normal(0, .004)}")
            out.append(f"{s},eval/completed,0.0")
    (d / "metrics.csv").write_text("\n".join(out) + "\n", encoding="utf-8")


def write_search_run() -> None:
    d = RUNS / "20261010-110000_demo-rauchkammer-search_s0"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rng = np.random.default_rng(11)
    length = 140000.0
    (d / "meta.json").write_text(json.dumps({
        "created": T0 + 3600, "level": "Rauchkammer", "level_id": 146399247, "game": "real", "difficulty": "extreme_demon",
        "length": length, "agent": "checkpoint-search", "step_unit": "physics_tick", "search": {"beam_width": 24, "segment_ticks": 60}}), encoding="utf-8")
    gens, eps = [], []
    t = T0 + 3600
    best = 0.0
    steps = 0
    for g in range(1, 521):
        best = min(0.62, best + abs(rng.normal(0.0014, 0.0009)))
        if g in (140, 141, 300):
            best -= 0.012  # a backtrack
        steps += 6000 + int(rng.integers(0, 3000))
        t += rng.uniform(.4, 1.2)
        gens.append({"t": round(t, 2), "step": steps, "generation": g, "segment": g, "best_fitness": best * 2, "mean_fitness": best * 1.4 + rng.normal(0, .02),
                     "best_progress": best, "agent": "checkpoint-search", "rollouts": int(30 + g * 0.2 + rng.integers(0, 20)), "survivors": int(rng.integers(6, 24)), "beam": 24, "backtracks": int(g > 140) + int(g > 300)})
        for a in range(int(rng.integers(3, 8))):
            p = max(0.0, best - abs(rng.normal(0.01, 0.03)))
            eps.append({"t": round(t + a * .01, 2), "step": steps + a * 200, "level": "Rauchkammer", "segment": g, "start_x": 0, "progress": p, "completed": False, "death_x": p * length, "length": 200, "agent": "checkpoint-search"})
    (d / "generations.jsonl").write_text("\n".join(json.dumps(x) for x in gens) + "\n", encoding="utf-8")
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in eps) + "\n", encoding="utf-8")
    (d / "result.json").write_text(json.dumps({"solved": False, "best_progress": best, "generations": 520, "rollouts": 61000, "wall_s": 5400.0, "env_ticks": steps}), encoding="utf-8")
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "final.pt").write_bytes(b"")


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    write_curriculum_run()
    write_search_run()
    print("wrote", RUNS)
