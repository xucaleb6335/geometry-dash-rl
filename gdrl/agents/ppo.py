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
from torch import nn
from torch.distributions import Categorical

from gdrl.agents.base import ActOutput, Agent, Rollout
from gdrl.agents.networks import GridCNN, layer_init, mlp


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
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    done = terminated | np.asarray(truncated, dtype=bool)

    advantages = np.zeros_like(rewards)
    next_adv = np.zeros(rewards.shape[1:], dtype=np.float64)  # A_{t+1}
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        next_adv = delta + gamma * lam * (1.0 - done[t]) * next_adv
        advantages[t] = next_adv
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


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

        # Separate policy and value networks (detail 13), one Adam over both.
        self.policy = self._build_net(self.n_actions, out_std=0.01).to(self.device)
        self.value = self._build_net(1, out_std=1.0).to(self.device)
        self.optimizer = torch.optim.Adam(self._parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)

    def _parameters(self) -> list[nn.Parameter]:
        return [*self.policy.parameters(), *self.value.parameters()]

    def _build_net(self, out_dim: int, out_std: float) -> nn.Module:
        """An MLP for flat observations, a GridCNN trunk for ``(C, H, W)`` grids."""
        if len(self.obs_shape) == 1:
            return mlp(int(self.obs_shape[0]), self.cfg.hidden_sizes, out_dim, out_std)
        if len(self.obs_shape) == 3:
            c, h, w = (int(d) for d in self.obs_shape)
            features = self.cfg.hidden_sizes[-1] if self.cfg.hidden_sizes else 128
            trunk = GridCNN(c, h, w, out_dim=features)
            return nn.Sequential(trunk, layer_init(nn.Linear(features, out_dim), std=out_std))
        raise ValueError(f"unsupported observation shape {self.obs_shape}")

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        with torch.no_grad():
            x = self._tensor(obs)
            dist = Categorical(logits=self.policy(x))
            action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
            logprob = dist.log_prob(action)
            value = self._values(x)
        return ActOutput(
            action=action.cpu().numpy().astype(np.int64),
            extras={
                "logprob": logprob.cpu().numpy().astype(np.float32),
                "value": value.cpu().numpy().astype(np.float32),
            },
        )

    def update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self.cfg
        T, N = rollout.num_steps, rollout.num_envs

        # 1. Learning-rate annealing (detail 4).
        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - self.num_updates_done / max(1, cfg.total_updates))
            lr = frac * cfg.learning_rate
        else:
            lr = cfg.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # 2. Value of the state after each step, per Rollout's docstring.
        #    Terminated steps are masked out inside compute_gae.
        with torch.no_grad():
            next_values = np.zeros((T, N), dtype=np.float32)
            if T > 1:
                after = self._tensor(rollout.obs[1:]).flatten(0, 1)
                next_values[:-1] = self._values(after).reshape(T - 1, N).cpu().numpy()
            next_values[-1] = self._values(self._tensor(rollout.next_obs)).cpu().numpy()
            truncated = np.asarray(rollout.truncated, dtype=bool)
            if truncated.any():
                next_values[truncated] = self._values(self._tensor(rollout.final_obs[truncated])).cpu().numpy()

        # 3. GAE (detail 5), against the values recorded at act() time.
        old_values = np.asarray(rollout.extras["value"], dtype=np.float32)
        advantages, returns = compute_gae(
            rollout.rewards, old_values, next_values, rollout.terminated, rollout.truncated,
            cfg.gamma, cfg.gae_lambda,
        )

        # 4. Flatten (T, N) -> B and train on shuffled minibatches (detail 6).
        B = T * N
        b_obs = self._tensor(rollout.obs.reshape(B, *self.obs_shape))
        b_actions = torch.as_tensor(rollout.actions.reshape(B), dtype=torch.int64, device=self.device)
        b_logprobs = self._tensor(rollout.extras["logprob"].reshape(B))
        b_advantages = self._tensor(advantages.reshape(B))
        b_returns = self._tensor(returns.reshape(B))
        b_values = self._tensor(old_values.reshape(B))

        minibatch_size = max(1, B // cfg.num_minibatches)
        stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
        for _epoch in range(cfg.update_epochs):
            perm = torch.randperm(B, device=self.device)
            epoch_kl = []
            for start in range(0, B, minibatch_size):
                mb = perm[start : start + minibatch_size]

                # 5. Losses on this minibatch.
                dist = Categorical(logits=self.policy(b_obs[mb]))
                new_logprob = dist.log_prob(b_actions[mb])
                entropy = dist.entropy().mean()
                new_value = self._values(b_obs[mb])

                logratio = new_logprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()  # detail 12: low-variance estimator
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                mb_adv = b_advantages[mb]
                if cfg.norm_adv and mb_adv.numel() > 1:  # detail 7
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate objective (detail 8), negated into a loss.
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                # Value loss, optionally clipped around the old value (detail 9).
                if cfg.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(new_value - b_values[mb], -cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss  # detail 10

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._parameters(), cfg.max_grad_norm)  # detail 11
                self.optimizer.step()

                stats["policy_loss"].append(pg_loss.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl.item())
                stats["clipfrac"].append(clipfrac.item())
                epoch_kl.append(approx_kl.item())

            # 6. Optional early stop once the policy has moved too far.
            if cfg.target_kl is not None and float(np.mean(epoch_kl)) > cfg.target_kl:
                break

        # 7. Metrics. Explained variance: how much of the return the critic predicted.
        y_true, y_pred = returns.reshape(B), old_values.reshape(B)
        var_y = float(np.var(y_true))
        explained_var = 0.0 if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y
        self.num_updates_done += 1
        metrics = {k: float(np.mean(v)) for k, v in stats.items()}
        metrics.update(explained_variance=explained_var, learning_rate=lr, value_mean=float(old_values.mean()))
        return metrics

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
