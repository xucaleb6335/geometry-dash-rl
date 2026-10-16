"""Carry a PPO checkpoint over to an observation with extra features appended.

    python -m gdrl.tools.widen_checkpoint --ckpt runs/<run>/checkpoints/update_000565.pt \
        --config configs/gd_sm_community.yaml --out runs/<new>/checkpoints/widened.pt

The new observation must start with the old one (new features at the end,
e.g. the encoder's ``profile``). The first layer of the policy and value nets
gets zero weights for the new inputs, so the widened policy acts exactly like
the old one until training uses the new features. The optimizer starts fresh
(its moments have the old shapes); the update count is kept so learning-rate
annealing continues. Resume with ``--set train.resume=<out>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gdrl.agents.registry import make_agent
from gdrl.config import load_config
from gdrl.env.factory import make_env


def widen(old: dict, new: dict) -> dict:
    out = {}
    for key, tensor in new.items():
        src = old[key]
        if src.shape == tensor.shape:
            out[key] = src
        elif src.dim() == 2 and src.shape[0] == tensor.shape[0] and src.shape[1] < tensor.shape[1]:
            t = torch.zeros_like(tensor)
            t[:, : src.shape[1]] = src
            out[key] = t
        else:
            raise ValueError(f"{key}: cannot map {tuple(src.shape)} to {tuple(tensor.shape)}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    env_cfg = dict(cfg["env"])
    env_cfg.pop("curriculum", None)
    env_cfg["backend"] = "mock"  # only the spaces are needed; no game connection
    env = make_env(env_cfg)
    agent = make_agent(cfg["agent"], env.observation_space, env.action_space, num_envs=1, total_steps=1)
    old = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    agent.policy.load_state_dict(widen(old["policy"], agent.policy.state_dict()))
    agent.value.load_state_dict(widen(old["value"], agent.value.state_dict()))
    agent.num_updates_done = int(old.get("num_updates_done", 0))
    agent._sync_actor()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.out)
    print(f"widened {args.ckpt} -> {args.out} (obs {env.observation_space.shape}, update {agent.num_updates_done})")


if __name__ == "__main__":
    main()
