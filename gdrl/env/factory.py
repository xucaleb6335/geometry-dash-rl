"""Build envs from the ``env:`` section of a config file."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from gdrl.env.backends import MockBackend, SocketBackend
from gdrl.env.gd_env import GDEnv
from gdrl.features.encoder import make_encoder
from gdrl.rewards.reward import make_reward


def make_env(env_cfg: dict[str, Any]) -> gym.Env:
    """``env_cfg["id"] == "gd"`` builds a GDEnv; anything else goes to ``gym.make``
    (so CartPole-v1 runs through the same train loop for the M6 gate)."""
    env_id = env_cfg.get("id", "gd")
    if env_id != "gd":
        return gym.make(env_id, **env_cfg.get("kwargs", {}))

    backend_name = env_cfg.get("backend", "mock")
    if backend_name == "mock":
        mock_cfg = dict(env_cfg.get("mock", {}))
        level_ids = env_cfg.get("level_ids") or [1]
        backend = MockBackend(level_seed=int(level_ids[0]), **mock_cfg)
    elif backend_name == "socket":
        backend = SocketBackend(
            host=env_cfg.get("host", "127.0.0.1"),
            port=int(env_cfg.get("port", 5555)),
            timeout=env_cfg.get("timeout", 10.0),
        )
    else:
        raise ValueError(f"unknown backend {backend_name!r}; use 'mock' or 'socket'")

    encoder_cfg = dict(env_cfg.get("encoder", {"name": "debug"}))
    reward_cfg = dict(env_cfg.get("reward", {"name": "constant"}))
    return GDEnv(
        backend,
        make_encoder(encoder_cfg.pop("name"), **encoder_cfg),
        make_reward(reward_cfg.pop("name"), **reward_cfg),
        frame_skip=int(env_cfg.get("frame_skip", 4)),
        max_steps=int(env_cfg.get("max_steps", 5000)),
        level_ids=env_cfg.get("level_ids"),
        start_pos=int(env_cfg.get("start_pos", -1)),
    )


def make_vector_env(env_cfg: dict[str, Any], num_envs: int) -> SyncVectorEnv:
    """N copies of the env stepped together.

    Same-step autoreset: when an episode ends, ``step`` returns the first
    observation of the next episode, and the real last observation is in
    ``info["final_obs"]`` (masked by ``info["_final_obs"]``). The train loop
    keeps it for bootstrapping truncated episodes.
    """
    if env_cfg.get("id", "gd") == "gd" and env_cfg.get("backend") == "socket" and num_envs != 1:
        raise ValueError("the real game is one instance: use num_envs: 1 with the socket backend")
    return SyncVectorEnv(
        [lambda: make_env(env_cfg) for _ in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
