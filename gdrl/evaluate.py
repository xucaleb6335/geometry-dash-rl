"""Evaluate an agent: completion rate and progress per level, optionally with
sticky actions (each decision repeats the previous action with probability p).

    python -m gdrl.evaluate --run runs/<run> --episodes 20 --sticky 0.1
    python -m gdrl.evaluate --run runs/<run> --levels 7 8 9      # held-out levels

Plumbing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gdrl.agents.registry import make_agent
from gdrl.config import load_config
from gdrl.env.factory import make_env


def evaluate(
    cfg: dict[str, Any],
    checkpoint: Path | None,
    episodes: int,
    sticky_prob: float = 0.0,
    level_ids: list[int] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    env = make_env(cfg["env"])
    agent = make_agent(cfg["agent"], env.observation_space, env.action_space,
                       num_envs=1, total_steps=1, seed=seed)
    if checkpoint is not None:
        agent.load(checkpoint)

    is_gd = cfg["env"].get("id", "gd") == "gd"
    levels = level_ids or (cfg["env"].get("level_ids") if is_gd else None) or [None]
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"sticky_prob": sticky_prob, "levels": {}}

    try:
        for level in levels:
            rows = []
            for ep in range(episodes):
                options = {"level_id": int(level)} if level is not None else None
                obs, info = env.reset(seed=seed + ep, options=options)
                prev_action, total, length = 0, 0.0, 0
                while True:
                    action = int(agent.act(obs[None], deterministic=True).action[0])
                    if length > 0 and rng.random() < sticky_prob:
                        action = prev_action
                    obs, reward, terminated, truncated, info = env.step(action)
                    prev_action, total, length = action, total + float(reward), length + 1
                    if terminated or truncated:
                        break
                rows.append({"return": total, "length": length,
                             "progress": float(info.get("progress", 0.0)),
                             "completed": bool(info.get("completed", False))})
            results["levels"][str(level)] = {
                "episodes": len(rows),
                "completion_rate": float(np.mean([r["completed"] for r in rows])),
                "progress_mean": float(np.mean([r["progress"] for r in rows])),
                "return_mean": float(np.mean([r["return"] for r in rows])),
            }
    finally:
        env.close()
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate an agent.")
    parser.add_argument("--run", type=Path, help="run folder from train.py (uses its config.yaml and final.pt)")
    parser.add_argument("--config", type=Path, help="config file, if not using --run")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint file, overrides the run's final.pt")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sticky", type=float, default=None, help="sticky-action probability")
    parser.add_argument("--levels", type=int, nargs="*", help="level ids to evaluate on")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    if args.run is None and args.config is None:
        parser.error("pass --run or --config")
    cfg = load_config(args.run / "config.yaml" if args.run else args.config, args.set)
    checkpoint = args.checkpoint or (args.run / "checkpoints" / "final.pt" if args.run else None)
    if checkpoint is not None and not checkpoint.exists():
        checkpoint = None if cfg["agent"]["name"] in ("random", "rule") else checkpoint

    results = evaluate(
        cfg,
        checkpoint,
        episodes=args.episodes or int(cfg["eval"]["episodes"]),
        sticky_prob=args.sticky if args.sticky is not None else float(cfg["eval"]["sticky_prob"]),
        level_ids=args.levels or cfg["eval"].get("level_ids"),
    )
    print(json.dumps(results, indent=2))
    if args.run:
        (args.run / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
