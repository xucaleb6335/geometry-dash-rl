"""Curriculum starts from a community StartPos copy of a level.

    python -m gdrl.tools.starts_from_copy --copy local:gdrl-smsp --original main:1 --prefix gdrl-smc --out configs/starts/sm.yaml

The copy is a level someone uploaded with StartPos objects (id 31) already
placed (find one with ``gdctl search-levels "<name> start pos"``, then
``download-level`` and ``copy-to-local``). For each StartPos, by x, this
makes a local level ``<prefix>-kNN`` that is the copy with only that StartPos
left, so the bridge's RESET (start_pos -1) respawns there. It first checks
that the copy is the original plus StartPos objects: every other object of
the original must be present at the same id and position. Each start is then
checked in game: the spawn must land at the StartPos x (the agent may die
soon after; the curriculum handles that). Needs the game running, the gdctl
lock held and no RL client connected. Writes the starts as YAML, the original
level start first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from gdrl.env.backends import SocketBackend

STARTPOS_ID = "31"


def gdctl(*args: str) -> dict:
    r = subprocess.run([sys.executable, "-m", "gdctl", *args], capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": r.stdout[-500:] + r.stderr[-500:]}


def export(spec: str, path: Path) -> str:
    out = gdctl("level-export", spec, "--path", str(path))
    if not out.get("ok"):
        sys.exit(f"level-export {spec} failed: {out}")
    return path.read_text(encoding="utf-8")


def parse(seg: str) -> dict:
    parts = seg.split(",")
    return dict(zip(parts[::2], parts[1::2]))


def key(o: dict) -> tuple:
    return (o.get("1"), round(float(o.get("2", 0)), 1), round(float(o.get("3", 0)), 1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--copy", required=True, help="local copy with StartPos objects, e.g. local:gdrl-smsp")
    p.add_argument("--original", required=True, help="the original level, e.g. main:1")
    p.add_argument("--prefix", required=True, help="name prefix for the per-start levels (<= 15 chars)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-gap", type=float, default=0.0, help="skip a StartPos closer than this to the previous kept one")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / (out.stem + "_levels")
    work.mkdir(exist_ok=True)
    raw = export(args.copy, work / "copy.txt")
    orig = export(args.original, work / "original.txt")

    header, _, rest = raw.partition(";")
    segs = [seg for seg in rest.split(";") if seg.strip()]
    objs = [parse(seg) for seg in segs]
    orig_objs = [parse(seg) for seg in orig.partition(";")[2].split(";") if seg.strip()]
    others = [o for o in objs if o.get("1") != STARTPOS_ID]
    missing = Counter(key(o) for o in orig_objs) - Counter(key(o) for o in others)
    extra = Counter(key(o) for o in others) - Counter(key(o) for o in orig_objs)
    print(f"original {len(orig_objs)} objects; copy {len(objs)} ({len(objs) - len(others)} StartPos); "
          f"missing from copy {sum(missing.values())}, extra in copy {sum(extra.values())}")
    if missing:
        print("  missing examples:", list(missing)[:10])
    if extra:
        print("  extra examples:", list(extra)[:10])
    starts_idx = sorted((i for i, o in enumerate(objs) if o.get("1") == STARTPOS_ID), key=lambda i: float(objs[i]["2"]))

    starts = [{"name": "start", "spec": args.original, "x": 0.0}]
    last_x = -1e9
    n = 0
    for i in starts_idx:
        o = objs[i]
        x, y = float(o["2"]), float(o.get("3", 0))
        if x - last_x < args.min_gap:
            continue
        body = [seg for j, seg in enumerate(segs) if j == i or objs[j].get("1") != STARTPOS_ID]
        n += 1
        name = f"{args.prefix}-k{n:02d}"
        path = work / f"{name}.txt"
        path.write_text(header + ";" + ";".join(body) + ";", encoding="utf-8")
        gdctl("exit-level")
        made = gdctl("create-level", "--name", name, "--file", str(path), "--replace")
        op = gdctl("open-level", f"local:{name}")
        if not (made.get("ok") and op.get("ok")):
            print(f"{name}: create/open failed: {made} {op}")
            continue
        b = SocketBackend(timeout=30.0)
        try:
            deadline, s = time.time() + 20, None
            while time.time() < deadline:
                s0 = b.reset(0, -1)
                s = b.step(False, 4)
                if s.dead or s.player_x != s0.player_x or s.player_y != s0.player_y:
                    break
                time.sleep(0.2)
            s = b.reset(0, -1)
        finally:
            b.close()
        ok = abs(s.player_x - x) < 3
        settings = {k: o[k] for k in o if k.startswith("kA")}
        print(f"{name}: StartPos x={x:.0f} y={y:.0f} -> spawn x={s.player_x:.1f} y={s.player_y:.1f} mode={s.mode} "
              f"{'ok' if ok else 'MISMATCH'} {settings}", flush=True)
        if ok:
            starts.append({"name": name, "spec": f"local:{name}", "x": x, "y": y, "mode": int(s.mode),
                           "source": args.copy})
            last_x = x
        out.write_text(yaml.safe_dump({"level": args.original, "copy": args.copy, "starts": starts},
                                      sort_keys=False), encoding="utf-8")
    print(f"wrote {len(starts)} starts to {out}")


if __name__ == "__main__":
    main()
