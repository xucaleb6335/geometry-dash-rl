"""Gate for protocol v2 snapshots: are SAVE/LOAD exact, and how fast are they?

    python -m gdrl.tools.snapshot_check --port 5555               # real game, level open
    python -m gdrl.tools.snapshot_check --mock 7 --mock-kind demon

For each of ``--points`` save points (random inputs up to there, sometimes
with the button held), it checks:

1. LOAD returns the same STATE that SAVE returned;
2. the same inputs after a LOAD give byte-identical STATEs, twice;
3. those STATEs equal the ones from a continuous run from RESET (the real
   test: a restore must be indistinguishable from never having left).

On a mismatch it prints the first differing field. It also reports the
time per SAVE, LOAD and STEP tick, which sizes the search budget.
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gdrl.env.protocol import GameState


def diff(a: GameState, b: GameState) -> str:
    for f in ("tick", "flags", "player_x", "player_y", "vel_y", "rotation", "x_speed", "mode", "player_size",
              "p2_x", "p2_y", "p2_vel_y"):
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb and not (isinstance(va, float) and np.isnan(va) and np.isnan(vb)):
            return f"{f}: {va!r} != {vb!r}"
    if len(a.objects) != len(b.objects):
        return f"n_objects: {len(a.objects)} != {len(b.objects)}"
    for i, (oa, ob) in enumerate(zip(a.objects, b.objects)):
        oa, ob = oa.copy(), ob.copy()
        oa["oflags"] &= np.uint8(0xFB)
        ob["oflags"] &= np.uint8(0xFB)  # HIDDEN: visual only
        if oa.tobytes() != ob.tobytes():
            return f"object {i}: {oa} != {ob}"
    return "identical"


def play(backend, actions, dt):
    out = []
    for a in actions:
        s = backend.step(bool(a), dt)
        out.append(s)
        if s.terminal:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--mock", type=int, default=None, help="use the in-process mock level with this id instead")
    p.add_argument("--mock-kind", default="simple")
    p.add_argument("--points", type=int, default=5)
    p.add_argument("--max-prefix", type=int, default=400, help="max decisions before a save point")
    p.add_argument("--after", type=int, default=150, help="decisions compared after each save point")
    p.add_argument("--dt", type=int, default=2, help="ticks per decision")
    p.add_argument("--start-pos", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solution", default=None,
                   help="solution.npz: use its inputs as the prefix (save points deep in the level)")
    args = p.parse_args(argv)
    solution = None
    if args.solution:
        solution = np.load(args.solution)["ticks"].astype(bool)[:: args.dt]

    if args.mock is not None:
        from gdrl.env.backends import MockBackend

        backend = MockBackend(level_seed=args.mock, kind=args.mock_kind)
    else:
        from gdrl.env.backends import SocketBackend

        backend = SocketBackend(port=args.port, timeout=60.0)
    if int(getattr(backend, "max_snapshots", 0)) <= 0:
        print("[FAIL] the server has no snapshots (protocol v1 mod?)")
        return 1
    rng = np.random.default_rng(args.seed)
    ok = True
    t_save, t_load, t_step, n_step_ticks = [], [], [], 0
    try:
        for k in range(args.points):
            hold_bias = float(rng.uniform(0.05, 0.6))
            if solution is not None:
                prefix = solution[: int(rng.integers(10, max(11, len(solution) - 10)))]
            else:
                prefix = rng.random(int(rng.integers(10, args.max_prefix))) < hold_bias
            after = rng.random(args.after) < hold_bias
            # continuous run from RESET: prefix + after
            backend.reset(0, args.start_pos)
            t0 = time.perf_counter()
            cont_prefix = play(backend, prefix, args.dt)
            t_step.append(time.perf_counter() - t0)
            n_step_ticks += len(cont_prefix) * args.dt
            if not cont_prefix or cont_prefix[-1].terminal:
                print(f"point {k}: the random prefix died at tick {cont_prefix[-1].tick if cont_prefix else 0}; "
                      "skipped")
                continue
            t0 = time.perf_counter()
            saved = backend.save_state(0)
            t_save.append(time.perf_counter() - t0)
            cont_after = play(backend, after, args.dt)
            runs = []
            for rep in range(2):
                t0 = time.perf_counter()
                restored = backend.load_state(0)
                t_load.append(time.perf_counter() - t0)
                if restored.physics_bytes() != saved.physics_bytes():
                    ok = False
                    print(f"[FAIL] point {k} (tick {saved.tick}): LOAD != SAVE: {diff(restored, saved)}")
                runs.append(play(backend, after, args.dt))
            for name, other in (("second LOAD", runs[1]), ("continuous run", cont_after)):
                bad = next((i for i, (x, y) in enumerate(zip(runs[0], other)) if x.physics_bytes() != y.physics_bytes()),
                           None)
                if bad is None and len(runs[0]) != len(other):
                    bad = min(len(runs[0]), len(other))
                if bad is not None:
                    ok = False
                    a = runs[0][min(bad, len(runs[0]) - 1)]
                    b = other[min(bad, len(other) - 1)]
                    print(f"[FAIL] point {k} (saved at tick {saved.tick}, holding={saved.holding}): after LOAD vs "
                          f"{name} differ at decision {bad}: {diff(a, b)}")
            if ok:
                print(f"point {k}: saved at tick {saved.tick} x={saved.player_x:.1f} mode={int(saved.mode)} "
                      f"holding={saved.holding}: exact over {len(runs[0])} decisions")
            backend.drop_state(0)
    finally:
        backend.close()
    if t_save:
        print(f"SAVE {1000 * np.mean(t_save):.2f} ms, LOAD {1000 * np.mean(t_load):.2f} ms, "
              f"STEP {1e6 * sum(t_step) / max(1, n_step_ticks):.1f} us/tick "
              f"({n_step_ticks / max(1e-9, sum(t_step)):.0f} ticks/s in {args.dt}-tick steps)")
    print("[PASS] snapshots are exact" if ok else "[FAIL] snapshots are not exact")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
