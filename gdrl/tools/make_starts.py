"""Build StartPos copies of a level for the start curriculum (gdrl/curriculum.py).

    python -m gdrl.tools.make_starts --level main:1 --prefix gdrl-sm --every 1500 --out configs/starts/sm.yaml

Needs the game running with the gdctl lock held, and no RL client connected.
For each target x (every ``--every`` units): makes a local copy of the level
(``<prefix>-sNN``) with one StartPos there, then checks it in the game: the
spawn must land at x, and the player must survive ``--check-steps`` decisions
with the button released (so it does not spawn inside a block or one step
from a spike). Spawn heights tried: the ground and just above each object in
that column (cube-like modes), or free heights (ship, wave, ...). If none
works, x moves right by one block and it tries again. Writes the list of
starts, the level start first, as YAML for ``env.curriculum.starts``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from gdctl import levelstring as ls
from gdrl.env.backends import SocketBackend

BLOCK = 30.0
FLYING = {"ship", "wave", "ufo", "swing"}


def gdctl(*args: str) -> dict:
    r = subprocess.run([sys.executable, "-m", "gdctl", *args], capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": r.stdout[-500:] + r.stderr[-500:]}


def candidates(objs: list[tuple[float, float]], x: float, mode: str) -> list[float]:
    column = [oy for ox, oy in objs if abs(ox - x) <= BLOCK / 2 + 1]
    if mode in FLYING:
        busy = [oy for ox, oy in objs if abs(ox - x) <= 2 * BLOCK]
        free = [y for y in np.arange(45.0, 300.0, 30.0) if all(abs(y - oy) >= BLOCK for oy in busy)]
        return sorted(free, key=lambda y: abs(y - 105.0))
    ys = {15.0} | {oy + BLOCK for oy in column}
    return sorted((y for y in ys if y >= 15.0), reverse=True)  # top surface first: the usual route


def wait_running(b: SocketBackend, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s0 = b.reset(0, -1)
        s1 = b.step(False, 4)
        if s1.dead or s1.player_x != s0.player_x or s1.player_y != s0.player_y:
            return True
        time.sleep(0.2)
    return False


def check(b: SocketBackend, x: float, steps: int, mode: str) -> tuple[bool, str]:
    s = b.reset(0, -1)
    if abs(s.player_x - x) > 2:
        return False, f"spawned at x={s.player_x:.1f}"
    grounded = mode in FLYING
    for _ in range(steps):
        s = b.step(False, 4)
        grounded = grounded or s.on_ground
        if s.dead:
            return False, f"died at x={s.player_x:.1f}"
    if not grounded:
        return False, "never landed"
    return True, f"ok, x={s.player_x:.1f} y={s.player_y:.1f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", default="main:1")
    p.add_argument("--prefix", default="gdrl-sm")
    p.add_argument("--every", type=float, default=1500.0)
    p.add_argument("--xs", type=float, nargs="*", help="explicit x positions instead of --every")
    p.add_argument("--first", type=float, default=None, help="first x (default: --every)")
    p.add_argument("--check-steps", type=int, default=15)
    p.add_argument("--max-shifts", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(".level.txt")
    exp = gdctl("level-export", args.level, "--path", str(raw_path))
    if not exp.get("ok"):
        sys.exit(f"level-export failed: {exp}")
    raw = raw_path.read_text(encoding="utf-8")
    objs = [(float(o.get("2", 0)), float(o.get("3", 0))) for o in ls.parse_objects(raw) if o.get("1")]
    last_x = max(x for x, _ in objs)
    xs = args.xs or list(np.arange(args.first or args.every, last_x - 2 * BLOCK, args.every))

    starts = [{"name": "start", "spec": args.level, "x": 0.0}]
    for n, x0 in enumerate(xs, start=1):
        name = f"{args.prefix}-s{n:02d}"
        gdctl("exit-level")
        made = gdctl("create-level", "--name", name, "--file", str(raw_path), "--replace")
        if not made.get("ok"):
            print(f"{name}: create-level failed: {made}")
            continue
        found = None
        for shift in range(args.max_shifts + 1):
            x = float(x0 + shift * BLOCK)
            gdctl("exit-level")  # a level can only be edited while it is not open
            sp = gdctl("startpos", f"local:{name}", "set", "--x", str(x), "--y", "15")
            if not sp.get("ok"):
                print(f"{name}: startpos failed: {sp}")
                break
            mode = sp["added"].get("mode_name", "cube")
            for y in candidates(objs, x, mode)[:6]:
                gdctl("exit-level")
                sp = gdctl("startpos", f"local:{name}", "set", "--x", str(x), "--y", str(y))
                op = gdctl("open-level", f"local:{name}")
                if not (sp.get("ok") and op.get("ok")):
                    print(f"{name}: x={x} y={y}: set/open failed")
                    continue
                b = SocketBackend(timeout=30.0)
                try:
                    if not wait_running(b):
                        ok, why = False, "level did not start"
                    else:
                        ok, why = check(b, x, args.check_steps, mode)
                finally:
                    b.close()
                print(f"{name}: x={x:.0f} y={y:.0f} {mode}: {why}", flush=True)
                if ok:
                    found = {"name": name, "spec": f"local:{name}", "x": x, "y": float(y), "mode": mode}
                    break
            if found:
                break
        if found:
            starts.append(found)
        else:
            print(f"{name}: no valid spawn near x={x0:.0f}; skipped")
        out.write_text(yaml.safe_dump({"level": args.level, "starts": starts}, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(starts)} starts to {out}")


if __name__ == "__main__":
    main()
