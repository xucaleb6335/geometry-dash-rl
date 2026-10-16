"""Evolve a fixed press sequence for one level: the GeneticTimingAgent baseline (M8).

    python -m gdrl.tools.genetic --config configs/mock_ppo.yaml --level 1 --generations 100
    python -m gdrl.tools.genetic --config configs/gd_ppo.yaml --generations 50   # real game, open level

Writes the best sequence to ``runs/genetic_<level>.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gdrl.agents.baselines import GeneticTimingAgent
from gdrl.config import load_config
from gdrl.env.factory import make_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--level", type=int, default=None, help="level id (default: the config's first)")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.level is not None:
        cfg["env"]["level_ids"] = [args.level]
    level = cfg["env"]["level_ids"][0]
    env = make_env(cfg["env"])
    agent = GeneticTimingAgent(env.observation_space, env.action_space, population=args.population,
                               max_len=int(cfg["env"].get("max_steps", 5000)), seed=args.seed)
    try:
        result = agent.evolve(env, args.generations)
    finally:
        env.close()
    out = Path(cfg["train"]["runs_dir"]) / f"genetic_{level}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(out)
    print(f"best fitness {result['best_fitness']:.3f} (2.0 = completed); saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
