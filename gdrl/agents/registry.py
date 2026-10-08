"""Build an agent from the ``agent:`` section of a config file."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from gdrl.agents.base import Agent


def make_agent(
    agent_cfg: dict[str, Any],
    observation_space: spaces.Space,
    action_space: spaces.Space,
    *,
    num_envs: int,
    total_steps: int,
    seed: int | None = None,
) -> Agent:
    cfg = dict(agent_cfg)
    name = cfg.pop("name")

    if name == "random":
        from gdrl.agents.baselines import RandomAgent

        return RandomAgent(observation_space, action_space, seed=seed, **cfg)
    if name == "rule":
        from gdrl.agents.baselines import RuleBasedAgent

        return RuleBasedAgent(observation_space, action_space, **cfg)
    if name == "ppo":
        from gdrl.agents.ppo import PPOAgent, PPOConfig  # torch is only imported when needed

        if "hidden_sizes" in cfg:
            cfg["hidden_sizes"] = tuple(cfg["hidden_sizes"])
        ppo_cfg = PPOConfig(**cfg)
        ppo_cfg.total_updates = max(1, total_steps // (ppo_cfg.rollout_steps * num_envs))
        return PPOAgent(observation_space, action_space, ppo_cfg)
    raise ValueError(f"unknown agent {name!r}; choose random, rule or ppo")
