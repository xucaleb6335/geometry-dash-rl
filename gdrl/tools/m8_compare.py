"""M8 comparison on the mock game: evaluate agents, run the genetic baseline,
and aggregate everything into the results table (results/m8_mock.md).

    # one PPO run (its config + final.pt), or a checkpoint-free agent from a config
    python -m gdrl.tools.m8_compare eval --agent ppo-object-list --run runs/sweep/<run>
    python -m gdrl.tools.m8_compare eval --agent rule --config configs/mock_ppo.yaml --set "agent={name: rule}"
    python -m gdrl.tools.m8_compare eval --agent random --config configs/mock_random.yaml

    # genetic: evolve per level, then replay every sequence on every level
    python -m gdrl.tools.m8_compare genetic --config configs/mock_ppo.yaml

    # the table
    python -m gdrl.tools.m8_compare table

Each ``eval`` writes ``results/m8/<agent>_<run or config>.json`` with greedy and
sticky results on the training and held-out levels from levels.yaml. Runs with
the same ``--agent`` label are seeds of one agent: the table reports their mean
and standard deviation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gdrl.config import load_config
from gdrl.evaluate import evaluate

OUT_DIR = Path("results/m8")


def _levels() -> tuple[list[int], list[int]]:
    data = yaml.safe_load(Path("levels.yaml").read_text(encoding="utf-8"))["mock"]
    return [int(x) for x in data["train"]], [int(x) for x in data["heldout"]]


def _run_info(run: Path) -> dict[str, Any]:
    """Env steps and wall-clock training time of a run folder (config.yaml is
    written at the start, final.pt at the end)."""
    steps = 0
    with (run / "metrics.csv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            steps = max(steps, int(line.split(",", 1)[0]))
    started = (run / "config.yaml").stat().st_mtime
    ended = (run / "checkpoints" / "final.pt").stat().st_mtime
    return {"env_steps": steps, "train_seconds": ended - started}


def cmd_eval(args: argparse.Namespace) -> None:
    if (args.run is None) == (args.config is None):
        sys.exit("pass exactly one of --run or --config")
    cfg = load_config(args.run / "config.yaml" if args.run else args.config, args.set)
    checkpoint = args.run / "checkpoints" / "final.pt" if args.run else None
    train, heldout = _levels()
    record: dict[str, Any] = {"agent": args.agent, "source": str(args.run or args.config), "set": args.set,
                              "train_levels": train, "heldout_levels": heldout, "results": {}}
    if args.run:
        record.update(_run_info(args.run))
        record["seed"] = int(cfg["seed"])
    started = time.time()
    for mode, sticky in (("greedy", 0.0), ("sticky", args.sticky)):
        res = evaluate(cfg, checkpoint, episodes=args.episodes, sticky_prob=sticky, level_ids=train + heldout)
        record["results"][mode] = res
    record["eval_seconds"] = time.time() - started
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.run.name if args.run else Path(args.config).stem
    out = OUT_DIR / f"{args.agent}_{tag}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({m: _summary(record, m) for m in ("greedy", "sticky")}, indent=2))


def _summary(record: dict[str, Any], mode: str) -> dict[str, dict[str, float]]:
    levels = record["results"][mode]["levels"]
    out = {}
    for split in ("train_levels", "heldout_levels"):
        rows = [levels[str(lv)] for lv in record[split]]
        out[split] = {
            "completion": float(np.mean([r["completion_rate"] for r in rows])),
            "progress": float(np.mean([r["progress_mean"] for r in rows])),
            "episodes": int(sum(r["episodes"] for r in rows)),
        }
    return out


def cmd_genetic(args: argparse.Namespace) -> None:
    from gdrl.agents.baselines import GeneticTimingAgent
    from gdrl.env.factory import make_env

    train, heldout = _levels()
    levels = args.levels or train + heldout
    base = load_config(args.config, args.set)
    rng = np.random.default_rng(args.seed)
    per_level: dict[str, Any] = {}
    sequences: dict[int, np.ndarray] = {}
    for level in levels:
        cfg = json.loads(json.dumps(base))
        cfg["env"]["level_ids"] = [level]
        env = make_env(cfg["env"])
        agent = GeneticTimingAgent(env.observation_space, env.action_space, population=args.population,
                                   max_len=int(cfg["env"]["max_steps"]), seed=args.seed + level)
        gens: list[str] = []
        started = time.time()
        try:
            agent.evolve(env, args.generations, log=gens.append)
        finally:
            env.close()
        solved = agent.best_fitness >= 2.0
        per_level[str(level)] = {
            "solved": bool(solved),
            "generations": len(gens),
            "episodes": len(gens) * args.population,  # each generation plays the whole population
            "best_fitness": float(agent.best_fitness),
            "seconds": time.time() - started,
        }
        sequences[level] = agent.best.copy()
        print(f"level {level}: solved={solved} after {len(gens)} generations "
              f"({len(gens) * args.population} episodes), best fitness {agent.best_fitness:.3f}", flush=True)

    # Replay: each level's sequence on its own level with sticky actions, and
    # greedily on every other level (transfer).
    replay: dict[str, dict[str, float]] = {}
    sticky: dict[str, dict[str, float]] = {}
    for src, seq in sequences.items():
        replay[str(src)] = {}
        for dst in levels:
            cfg = json.loads(json.dumps(base))
            cfg["env"]["level_ids"] = [dst]
            env = make_env(cfg["env"])
            try:
                fitness, _, completed = GeneticTimingAgent.play(env, seq)
                replay[str(src)][str(dst)] = fitness - float(completed)  # progress
                if dst == src:
                    rows = [_play_sticky(env, seq, args.sticky, rng) for _ in range(args.episodes)]
                    sticky[str(src)] = {
                        "episodes": len(rows),
                        "completion_rate": float(np.mean([r[1] for r in rows])),
                        "progress_mean": float(np.mean([r[0] for r in rows])),
                    }
            finally:
                env.close()

    record = {"agent": "genetic", "source": str(args.config), "population": args.population,
              "max_generations": args.generations, "sticky_prob": args.sticky,
              "train_levels": train, "heldout_levels": heldout,
              "per_level": per_level, "replay_progress": replay, "sticky": sticky}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "genetic.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def _play_sticky(env, seq: np.ndarray, p: float, rng: np.random.Generator) -> tuple[float, bool]:
    env.reset()
    prev, info = 0, {}
    for t in range(len(seq)):
        action = int(seq[t])
        if t > 0 and rng.random() < p:
            action = prev
        _, _, terminated, truncated, info = env.step(action)
        prev = action
        if terminated or truncated:
            break
    return float(info.get("progress", 0.0)), bool(info.get("completed", False))


def _fmt(values: list[float], pct: bool = True) -> str:
    scale = 100.0 if pct else 1.0
    m = float(np.mean(values)) * scale
    if len(values) == 1:
        return f"{m:.0f}%" if pct else f"{m:.2f}"
    s = float(np.std(values)) * scale
    return f"{m:.0f} ± {s:.0f}%" if pct else f"{m:.2f} ± {s:.2f}"


def cmd_table(args: argparse.Namespace) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(OUT_DIR.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["agent"] != "genetic":
            groups.setdefault(rec["agent"], []).append(rec)

    head = ("| Agent | Seeds | Train compl. greedy | Train prog. greedy | Train compl. sticky | Train prog. sticky "
            "| Held-out compl. greedy | Held-out prog. greedy | Held-out compl. sticky | Held-out prog. sticky "
            "| Episodes (per mode) | Env steps | Train time |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    order = args.order or list(groups)
    for agent in order:
        recs = groups.get(agent)
        if not recs:
            continue
        cells = [agent, str(len(recs)) if "seed" in recs[0] else "-"]
        for split in ("train_levels", "heldout_levels"):
            for mode in ("greedy", "sticky"):
                sums = [_summary(r, mode)[split] for r in recs]
                cells.append(_fmt([s["completion"] for s in sums]))
                cells.append(_fmt([s["progress"] for s in sums]))
        eps = {m: _summary(recs[0], m) for m in ("greedy", "sticky")}
        cells.append(f"{eps['greedy']['train_levels']['episodes']} + {eps['greedy']['heldout_levels']['episodes']}")
        if "env_steps" in recs[0]:
            cells.append(f"{np.mean([r['env_steps'] for r in recs]) / 1e6:.2f}M")
            cells.append(f"{np.mean([r['train_seconds'] for r in recs]) / 60:.0f} min")
        else:
            cells += ["0", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines))

    gpath = OUT_DIR / "genetic.json"
    if gpath.exists():
        g = json.loads(gpath.read_text(encoding="utf-8"))
        print()
        print("| Level | Solved | Generations | Episodes | Sticky 0.1 completion | Sticky 0.1 progress "
              "| Mean progress of this sequence on the other levels |")
        print("|---|---|---|---|---|---|---|")
        for lv, row in g["per_level"].items():
            others = [p for dst, p in g["replay_progress"][lv].items() if dst != lv]
            st = g["sticky"][lv]
            print(f"| {lv} | {'yes' if row['solved'] else 'no'} | {row['generations']} | {row['episodes']} "
                  f"| {st['completion_rate']:.0%} ({st['episodes']} ep) | {st['progress_mean']:.0%} "
                  f"| {np.mean(others):.0%} |")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="evaluate one agent greedily and with sticky actions")
    p.add_argument("--agent", required=True, help="label; runs with the same label are seeds of one agent")
    p.add_argument("--run", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--sticky", type=float, default=0.1)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("genetic", help="evolve a press sequence per level, then test transfer")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--levels", type=int, nargs="*")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--sticky", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_genetic)

    p = sub.add_parser("table", help="print the markdown tables from results/m8/*.json")
    p.add_argument("--order", nargs="*", help="agent labels in row order")
    p.set_defaults(func=cmd_table)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
