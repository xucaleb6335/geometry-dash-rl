"""Play a searched route (per-tick inputs, .npy) in Geometry Dash at ~1x, for watching.

    python -m gdrl.tools.watch_route --level saved:10565740 --route configs/demos/bb_start_end_4tick.npy \n        --config configs/gd_bb_community.yaml

Uses the bridge's watch-mode (one STEP per rendered frame). Runs of equal input are
sent as STEPs of up to 4 ticks (one frame at 60 fps = 4 physics ticks), so the
replay is real time except where the input changes every tick or two (it slows
down there, since each change needs its own frame). Takes the game lock.
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from gdrl.tools.watch import TICKS_PER_SECOND, gdctl, keep_restored, must


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", required=True, help="saved:<id> or main:<n>")
    ap.add_argument("--route", required=True, help="per-tick inputs (.npy, bool) from the level start")
    ap.add_argument("--config", default="configs/gd_bb_community.yaml")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args(argv)

    lock = gdctl("lock", "acquire", "--note", "watch route", "--wait", "5")
    if not lock.get("ok") or not lock.get("held"):
        raise SystemExit(f"game lock is held by someone else: {lock.get('owner', lock)}")
    stop = threading.Event()
    threading.Thread(target=keep_restored, args=(stop,), daemon=True).start()
    try:
        if a.level.startswith("saved:"):
            must(gdctl("download-level", a.level[6:], "--no-song"), "download-level")
        must(gdctl("prepare-training", "--level", a.level), "prepare-training")
        must(gdctl("mod-setting", "legacy-view", "off"), "legacy-view")
        must(gdctl("mod-setting", "watch-mode", "on"), "watch-mode")
        gdctl("unminimize")
        gdctl("focus")

        from gdrl.config import load_config
        from gdrl.env.backends import wait_until_running
        from gdrl.env.factory import make_env

        env_cfg = dict(load_config(a.config)["env"])
        env_cfg.pop("curriculum", None)
        env = make_env(env_cfg)
        b = env.unwrapped.backend
        try:
            wait_until_running(b, -2)
            ticks = np.load(a.route).astype(bool)
            b.reset(0, -2)
            time.sleep(1.0)
            i, st, next_t = 0, None, time.perf_counter()
            while i < len(ticks):
                n = 1
                while n < 4 and i + n < len(ticks) and ticks[i + n] == ticks[i]:
                    n += 1
                st = b.step(bool(ticks[i]), n)
                i += n
                if st.terminal:
                    break
                next_t += n / TICKS_PER_SECOND / a.speed
                time.sleep(max(0.0, next_t - time.perf_counter()))
            print(f"route ends at x={st.player_x:.0f} ({100 * st.progress:.1f}%), "
                  f"{'died' if st.dead else 'completed' if st.completed else 'alive'}", flush=True)
            time.sleep(2.0)
        finally:
            env.close()
    finally:
        stop.set()
        gdctl("mod-setting", "watch-mode", "off")
        gdctl("lock", "release")


if __name__ == "__main__":
    main()
