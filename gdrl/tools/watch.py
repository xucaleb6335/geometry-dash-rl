"""Play a trained policy in Geometry Dash at 1x speed, for watching.

    python -m gdrl.tools.watch sm                 # Stereo Madness (beaten)
    python -m gdrl.tools.watch bb --attempts 3    # Bloodbath (30%)
    python -m gdrl.tools.watch --list
    python -m gdrl.tools.watch bb --speed 0       # as fast as the game goes

Policies are listed in configs/watch.yaml. Steps: take the gdctl game lock,
open the level (prepare-training; launches GD if needed, re-downloads saved
levels), turn on the bridge's watch-mode (one STEP per rendered frame, so 60
decisions/s render as 60 smooth frames) and, if the policy needs it,
legacy-view, then play greedy episodes paced at `speed` x real time. Both
settings are turned off and the lock released at the end. GD is restored
whenever it gets minimized (a minimized GD runs no frames).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

OWNER = "watch"
BRIDGE = "xucaleb6335.gd-rl-bridge"
TICKS_PER_SECOND = 240


def gdctl(*args: str, timeout: float = 300.0) -> dict:
    env = {**os.environ, "GDCTL_OWNER": OWNER}
    p = subprocess.run([sys.executable, "-m", "gdctl", *args], capture_output=True, text=True,
                       timeout=timeout, env=env)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {"ok": False, "error": (p.stdout + p.stderr).strip()[-500:]}
    return out


def must(out: dict, what: str) -> dict:
    if not out.get("ok"):
        raise SystemExit(f"{what} failed: {out.get('code', '')} {out.get('error', out)}")
    return out


def keep_restored(stop: threading.Event) -> None:
    """Un-minimize GD every half second while the policy plays."""
    from gdctl import game, paths, win32

    while not stop.wait(0.5):
        try:
            pids = game.game_pids()
            win = win32.main_window(pids[0], paths.WINDOW_TITLE) if pids else None
            if win:
                win32.unminimize(win["hwnd"])
        except Exception:
            pass


def load_presets(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def play(preset: dict, attempts: int, speed: float, log=print) -> list[dict]:
    import torch

    from gdrl.agents.registry import make_agent
    from gdrl.config import load_config
    from gdrl.env.backends import wait_until_running
    from gdrl.env.factory import make_env

    cfg = load_config(preset["config"])
    env_cfg = dict(cfg["env"])
    env_cfg.pop("curriculum", None)
    env = make_env(env_cfg)
    start_pos = int(preset.get("start_pos", -2))
    results = []
    try:
        wait_until_running(env.unwrapped.backend, start_pos)
        agent = make_agent(cfg["agent"], env.observation_space, env.action_space, num_envs=1, total_steps=1)
        agent.load(preset["checkpoint"])
        dt = env_cfg.get("frame_skip", 4) / TICKS_PER_SECOND / speed if speed > 0 else 0.0
        for a in range(attempts):
            obs, _ = env.reset(options={"start_pos": start_pos})
            if dt:
                time.sleep(1.0)
            next_t = time.perf_counter()
            while True:
                with torch.no_grad():
                    act = int(agent.act(obs[None], deterministic=True).action[0])
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    break
                if dt:
                    next_t += dt
                    time.sleep(max(0.0, next_t - time.perf_counter()))
            s = env.unwrapped.state
            r = {"attempt": a + 1, "completed": bool(info.get("completed")),
                 "progress": round(100 * float(info.get("progress", 0.0)), 1),
                 "x": round(float(s.player_x)), "seconds": round(s.tick / TICKS_PER_SECOND, 1)}
            results.append(r)
            log(f"attempt {r['attempt']}: {'COMPLETED' if r['completed'] else 'died'} at "
                f"{r['progress']}% (x={r['x']}, {r['seconds']} s of level)")
            if dt and a + 1 < attempts:
                time.sleep(2.0)
    finally:
        env.close()
    return results


def watch(name: str, attempts: int = 1, speed: float = 1.0, presets_path: Path = Path("configs/watch.yaml"),
          log=print) -> list[dict]:
    presets = load_presets(presets_path)
    if name not in presets:
        raise SystemExit(f"unknown policy {name!r}; known: {', '.join(presets)}")
    preset = presets[name]
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    lock = gdctl("lock", "acquire", "--note", f"watch {name}", "--wait", "5")
    if not lock.get("ok") or not lock.get("held"):
        raise SystemExit(f"game lock is held by someone else (training?): {lock.get('owner', lock)}. "
                         "Stop it first (python -m gdctl stop-training).")
    stop = threading.Event()
    # From the start: a minimized GD runs no frames, so even the control mod
    # (download, open level) would time out.
    threading.Thread(target=keep_restored, args=(stop,), daemon=True).start()
    try:
        level = preset["level"]
        log(f"opening {level} ...")
        if level.startswith("saved:"):
            must(gdctl("download-level", level[6:], "--no-song"), "download-level")
        must(gdctl("prepare-training", "--level", level), "prepare-training")
        must(gdctl("mod-setting", "legacy-view", "on" if preset.get("legacy_view") else "off"), "legacy-view")
        must(gdctl("mod-setting", "watch-mode", "on" if speed > 0 else "off"), "watch-mode")
        gdctl("unminimize")
        gdctl("focus")
        log(f"playing {preset.get('name', name)} ({preset['checkpoint']}) at "
            f"{'max' if speed <= 0 else f'{speed:g}x'} speed")
        return play(preset, attempts, speed, log)
    finally:
        stop.set()
        for key in ("watch-mode", "legacy-view"):
            gdctl("mod-setting", key, "off")
        gdctl("lock", "release")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="policy name from configs/watch.yaml")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time; 0 = as fast as possible")
    ap.add_argument("--presets", type=Path, default=Path("configs/watch.yaml"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list or not a.name:
        for k, v in load_presets(a.presets).items():
            print(f"{k:6s} {v.get('name', '')}  [{v['level']}]  {v['checkpoint']}")
        return
    watch(a.name, a.attempts, a.speed, a.presets)


if __name__ == "__main__":
    main()
