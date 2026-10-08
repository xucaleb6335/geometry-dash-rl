"""Proximal Policy Optimization, written by you. Milestone M6.

Checklist: core details 1 to 13 of "The 37 Implementation Details of PPO"
(https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
The rollout collection (detail 1, vectorized envs) is already done for you
in ``train.py``; this file is the learning side.

Gate before touching GD: CartPole-v1 average return >= 475 on 3 seeds
(``python -m gdrl.train --config configs/cartpole_ppo.yaml``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from gdrl.agents.base import ActOutput, Agent, Rollout


@dataclass
class PPOConfig:
    """Hyperparameters. Defaults follow CleanRL's ppo.py for classic control."""

    rollout_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    norm_adv: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_sizes: tuple[int, ...] = (64, 64)
    device: str = "auto"  # "auto" picks cuda when available, else cpu
    total_updates: int = 1  # set by train.py; used for learning-rate annealing


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over a ``(T, N)`` rollout.

    Args:
        rewards:     r_t, shape (T, N)
        values:      V(s_t), shape (T, N)
        next_values: V of the state after step t, shape (T, N), already chosen
                     per the table in ``Rollout``'s docstring (V(final_obs) for
                     truncated steps). Ignored where ``terminated``.
        terminated:  episode really ended at t (no bootstrap)
        truncated:   episode cut off at t (bootstrap from next_values, but do
                     not carry the advantage across into the next episode)

    Returns:
        (advantages, returns), both (T, N), with ``returns = advantages + values``.

    delta_t = r_t + gamma * next_values_t * (1 - terminated_t) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1},   done = terminated | truncated
    """
    # LEARN(M6): loop backwards over t. Test: tests/learn/test_gae.py has a
    # hand-worked example for each case (plain, terminated, truncated).
    raise NotImplementedError("LEARN(M6): compute_gae")


class PPOAgent(Agent):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, cfg: PPOConfig):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("this PPO handles Discrete actions; GD uses Discrete(2)")
        self.cfg = cfg
        self.rollout_steps = cfg.rollout_steps
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if cfg.device == "auto" else cfg.device))
        self.obs_shape = observation_space.shape
        self.n_actions = int(action_space.n)
        self.num_updates_done = 0

        # LEARN(M6): build two separate networks (detail 13 says separate nets
        # beat a shared trunk on simple tasks) and one Adam optimizer over both:
        #   self.policy:    obs -> logits (n_actions), output std 0.01
        #   self.value:     obs -> V(s) (1), output std 1.0
        #   self.optimizer: Adam(lr=cfg.learning_rate, eps=cfg.adam_eps)
        # Use gdrl.agents.networks.mlp. Move both networks to self.device.
        raise NotImplementedError("LEARN(M6): PPOAgent.__init__")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        # LEARN(M6): under torch.no_grad(), turn obs into a float tensor on
        # self.device, build a Categorical from the policy logits, sample (or
        # argmax if deterministic) and return
        #   ActOutput(action, extras={"logprob": ..., "value": ...})
        # as numpy arrays of shape (N,). The train loop stores extras for update().
        raise NotImplementedError("LEARN(M6): PPOAgent.act")

    def update(self, rollout: Rollout) -> dict[str, float]:
        # LEARN(M6), in order:
        #  1. anneal the learning rate: frac = 1 - num_updates_done / cfg.total_updates
        #  2. next_values (T, N) per the table in Rollout's docstring: V(obs[t+1]),
        #     V(next_obs) at the last step, V(final_obs[t]) where truncated
        #  3. advantages, returns = compute_gae(...)
        #  4. flatten (T, N) -> (T*N); for each epoch shuffle indices and split
        #     into cfg.num_minibatches minibatches (each sample used once per epoch)
        #  5. per minibatch: new logprob, entropy, value; ratio = exp(new - old);
        #     normalize advantages within the minibatch; clipped surrogate loss;
        #     (clipped) value loss; loss = pg - ent_coef * entropy + vf_coef * v_loss;
        #     backward, clip_grad_norm_(max_grad_norm), step
        #  6. optional early stop when approx_kl > target_kl
        #  7. self.num_updates_done += 1 and return metrics:
        #     policy_loss, value_loss, entropy, approx_kl, clipfrac,
        #     explained_variance, learning_rate
        # Sanity check: on the first minibatch of every update the ratio must be
        # exactly 1. If it is not, your logprobs do not match the rollout's.
        raise NotImplementedError("LEARN(M6): PPOAgent.update")

    # Checkpoint I/O is plumbing; it works once __init__ builds the attributes.
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_updates_done": self.num_updates_done,
                "cfg": asdict(self.cfg),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.num_updates_done = ckpt.get("num_updates_done", 0)
