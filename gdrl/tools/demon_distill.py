"""Distill a search solution into a policy (BC + DAgger with the search as expert).

    python -m gdrl.tools.demon_distill --config configs/demon_distill_mock.yaml \
        --solution runs/<search run>/solution.npz --record

Writes a run folder that ``gdrl.evaluate --run`` understands (config.yaml with
``agent: ppo`` and checkpoints/final.pt), plus meta.json, episodes.jsonl,
generations.jsonl and result.json. The beat counts if the greedy policy
completes the level from RESET with no snapshots.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

from gdrl.config import load_config, save_config
from gdrl.features.demon_encoder import DemonEncoder  # noqa: F401  (registers "demon")
from gdrl.features.encoder import make_encoder
from gdrl.runlog import RunLog
from gdrl.search.distill import DistillConfig, Distiller
from gdrl.tools.demon_search import level_info, make_backend, record_beat, search_config
from gdrl.utils import make_run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--solution", type=Path, action="append", required=True,
                        help="solution.npz from gdrl.tools.demon_search (repeatable)")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.level is not None:
        cfg["env"]["level_ids"] = [args.level]
    raw = dict(cfg.get("distill", {}))
    known = {f.name for f in fields(DistillConfig)} - {"search"}
    if set(raw) - known:
        raise ValueError(f"unknown distill options {sorted(set(raw) - known)}")
    if "hidden_sizes" in raw:
        raw["hidden_sizes"] = tuple(raw["hidden_sizes"])
    dcfg = DistillConfig(**raw, search=search_config(cfg))
    if int(cfg["env"].get("frame_skip", dcfg.decision_ticks)) != dcfg.decision_ticks:
        raise ValueError("env.frame_skip must equal distill.decision_ticks")

    backend = make_backend(cfg["env"])
    enc_cfg = dict(cfg["env"]["encoder"])
    encoder = make_encoder(enc_cfg.pop("name"), **enc_cfg)
    # The run folder doubles as a PPO run for gdrl.evaluate.
    cfg["agent"] = {"name": "ppo", "hidden_sizes": list(dcfg.hidden_sizes), "device": "cpu"}
    run_dir = make_run_dir(cfg.get("train", {}).get("runs_dir", "runs"), cfg["name"], dcfg.seed)
    save_config(cfg, run_dir / "config.yaml")
    info = level_info(cfg, backend)
    log = RunLog(run_dir, {**info, "agent": "search-distilled-policy", "step_unit": "physics_tick",
                           "solutions": [str(p) for p in args.solution]})
    print(f"run dir: {run_dir}", flush=True)
    start_pos = int(cfg["env"].get("start_pos", -1))
    dist = Distiller(backend, encoder, dcfg, start_pos=start_pos, log=log, level_name=str(info["level"]),
                     print_fn=lambda s: print(s, flush=True))
    try:
        sols = [np.load(p)["ticks"].astype(bool) for p in args.solution]
        out = dist.run(sols)
        dist.agent.save(run_dir / "checkpoints" / "final.pt")
        verified, progress = dist.verify()
        out.update(verified=verified, verify_progress=progress)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2), flush=True)
        if verified and args.record:
            from gdrl.tools import record_beat as rb

            rb.main(["--level", str(info["level"]), "--level-id", str(info.get("level_id", "")),
                     "--game", info["game"], "--difficulty", str(info.get("difficulty", "")),
                     "--agent", "search-distilled-policy", "--attempts", str(out["attempts"]),
                     "--generations", str(out["rounds"]), "--env-steps", str(out["env_ticks"]),
                     "--wall-hours", f"{out['wall_s'] / 3600:.4f}", "--run", str(run_dir),
                     "--notes", "greedy MLP policy (BC on the search solution + DAgger with the search as expert) "
                                "completed the level from RESET with no snapshots; attempts = policy episodes + "
                                "expert search rollouts (search attempts for the solution itself not included), "
                                "generations = DAgger rounds, env steps = physics ticks"])
    finally:
        log.close()
        backend.close()
    return 0 if out.get("verified") else 1


if __name__ == "__main__":
    sys.exit(main())
