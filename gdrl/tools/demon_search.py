"""Beat a level with the checkpoint search, then prove it with a clean replay.

    python -m gdrl.tools.demon_search --config configs/demon_search_mock.yaml --level 1
    python -m gdrl.tools.demon_search --config configs/demon_search_gd.yaml          # real game, open level
    python -m gdrl.tools.demon_search --config ... --replay runs/<run>/solution.npz    # replay only

Writes ``runs/<time>_<name>_s<seed>/``: meta.json, episodes.jsonl (one line
per search rollout), generations.jsonl (one line per segment expansion),
solution.npz (per-tick button array) and result.json. The beat counts only
if a replay from RESET with no snapshots completes the level; with
``--record`` it is then written to the ledger (gdrl.tools.record_beat).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from gdrl.config import load_config, save_config
from gdrl.runlog import RunLog
from gdrl.search.segment_search import SearchConfig, SegmentSearch, replay
from gdrl.utils import make_run_dir


def make_backend(env_cfg: dict[str, Any]):
    """The bare backend (no encoder, no reward) for an ``env:`` config section."""
    name = env_cfg.get("backend", "mock")
    if name == "mock":
        from gdrl.env.backends import MockBackend

        level_ids = env_cfg.get("level_ids") or [1]
        return MockBackend(level_seed=int(level_ids[0]), **dict(env_cfg.get("mock", {})))
    if name == "socket":
        from gdrl.env.backends import SocketBackend

        from gdrl.env.backends import wait_until_running

        backend = SocketBackend(host=env_cfg.get("host", "127.0.0.1"), port=int(env_cfg.get("port", 5555)),
                                timeout=env_cfg.get("timeout", 30.0))
        if not wait_until_running(backend):
            raise RuntimeError("the level's physics never started (paused? not in a level?)")
        return backend
    raise ValueError(f"unknown backend {name!r}")


def search_config(cfg: dict[str, Any]) -> SearchConfig:
    raw = dict(cfg.get("search", {}))
    raw.setdefault("seed", int(cfg.get("seed", 0)))
    known = {f.name for f in fields(SearchConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown search options {sorted(unknown)}")
    if "toggle_probs" in raw:
        raw["toggle_probs"] = tuple(raw["toggle_probs"])
    return SearchConfig(**raw)


def level_info(cfg: dict[str, Any], backend) -> dict[str, Any]:
    env = cfg["env"]
    info = {"level": env.get("level_name", ""), "level_id": env.get("level_id_label", ""),
            "difficulty": env.get("difficulty", ""), "game": "mock" if env.get("backend", "mock") == "mock" else "real"}
    game = getattr(backend, "game", None)
    if game is not None:
        info.update(level=info["level"] or game.level.name or f"mock-{game.level.seed}",
                    level_id=info["level_id"] or game.level.seed, length=game.level.length,
                    difficulty=info["difficulty"] or env.get("mock", {}).get("kind", "simple"))
    return info


def record_beat(info: dict[str, Any], result: dict[str, Any], run_dir: Path, notes: str) -> None:
    from gdrl.tools import record_beat as rb

    rb.main(["--level", str(info["level"]), "--level-id", str(info.get("level_id", "")), "--game", info["game"],
             "--difficulty", str(info.get("difficulty", "")), "--agent", "checkpoint-search",
             "--attempts", str(result["rollouts"]), "--generations", str(result["generations"]),
             "--env-steps", str(result["env_ticks"]), "--wall-hours", f"{result['wall_s'] / 3600:.4f}",
             "--run", str(run_dir), "--notes", notes])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--level", type=int, default=None, help="level id (mock: seed); default: the config's")
    parser.add_argument("--start-pos", type=int, default=None, help="search from this Start Pos (default -1)")
    parser.add_argument("--replay", type=Path, default=None, help="only replay a solution.npz and report")
    parser.add_argument("--record", action="store_true", help="write a verified beat to the ledger")
    parser.add_argument("--resume", type=Path, default=None,
                        help="best_prefix.npz (or solution.npz) of an interrupted run: replay it, minus "
                             "--resume-back ticks, then search on")
    parser.add_argument("--resume-back", type=int, default=480)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.level is not None:
        cfg["env"]["level_ids"] = [args.level]
    start_pos = args.start_pos if args.start_pos is not None else int(cfg["env"].get("start_pos", -1))
    level_id = int(cfg["env"].get("reset_level_id", 0))  # what RESET sends; 0 = the open level
    backend = make_backend(cfg["env"])

    if args.replay is not None:
        data = np.load(args.replay)
        state = replay(backend, data["ticks"].astype(bool), level_id, int(data["start_pos"]))
        print(json.dumps({"completed": state.completed, "dead": state.dead, "progress": state.progress,
                          "tick": state.tick}))
        backend.close()
        return 0 if state.completed else 1

    scfg = search_config(cfg)
    run_dir = make_run_dir(cfg.get("train", {}).get("runs_dir", "runs"), cfg["name"], scfg.seed)
    save_config(cfg, run_dir / "config.yaml")
    info = level_info(cfg, backend)
    log = RunLog(run_dir, {**info, "agent": "checkpoint-search", "step_unit": "physics_tick",
                           "tick_rate": backend.tick_rate, "start_pos": start_pos,
                           "search": {f.name: getattr(scfg, f.name) for f in fields(scfg)}})
    print(f"run dir: {run_dir}", flush=True)
    search = SegmentSearch(backend, scfg, level_id=level_id, start_pos=start_pos, log=log,
                           level_name=str(info["level"]), print_fn=lambda s: print(s, flush=True))
    prefix = None
    if args.resume is not None:
        prefix = np.load(args.resume)["ticks"].astype(bool)
        prefix = prefix[: max(0, len(prefix) - args.resume_back)]
        print(f"resuming after a {len(prefix)}-tick prefix from {args.resume}", flush=True)

    def save_progress(ticks):
        np.savez(run_dir / "best_prefix.npz", ticks=ticks, start_pos=start_pos, level_id=level_id,
                 tick_rate=backend.tick_rate)

    try:
        res = search.run(prefix=prefix, progress_fn=save_progress)
        np.savez(run_dir / "solution.npz", ticks=res.ticks, start_pos=start_pos, level_id=level_id,
                 tick_rate=backend.tick_rate)
        out = res.summary()
        out["generations"] = search.generation
        verified = False
        if res.solved:
            t0 = time.time()
            state = replay(backend, res.ticks, level_id, start_pos)
            verified = bool(state.completed)
            out.update(replay_completed=verified, replay_tick=state.tick, replay_progress=state.progress,
                       replay_s=time.time() - t0)
            log.episode(step=res.env_ticks + len(res.ticks), level=info["level"], segment=-1,
                        start_x=None, progress=state.progress, completed=state.completed,
                        death_x=state.player_x if state.dead else None, length=state.tick,
                        agent="checkpoint-search-replay", attempt=res.rollouts + 1)
        out["verified"] = verified
        (run_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2), flush=True)
        if verified and args.record:
            where = "the level start" if start_pos < 0 else f"Start Pos {start_pos}"
            record_beat(info, out, run_dir,
                        f"checkpoint search, then a clean replay of the found inputs from {where} (RESET, no "
                        f"snapshots) completed the level at tick {out['replay_tick']}; attempts = search "
                        f"rollouts ({res.deaths} deaths), generations = segment expansions, env steps = physics "
                        f"ticks; {res.backtracks} backtracks")
    finally:
        log.close()
        backend.close()
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
