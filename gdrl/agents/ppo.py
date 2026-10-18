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
import torch.nn.functional as F
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
    device: str = "auto"  # "auto" picks cuda when available, else cpu (training updates)
    # Device for act() (one observation per env step). Copies of the networks
    # live there and are refreshed after each update. "cpu" with device cuda
    # keeps the updates on the GPU but avoids a GPU round trip per game step
    # (measured on the real game: act() took 1.8 ms/step on cuda, the env 0.7).
    act_device: str | None = None  # None = same as device
    total_updates: int = 1  # set by train.py; used for learning-rate annealing
    # Behaviour-cloning anchor: (obs, action, weight) pairs saved by
    # gdrl.tools.demo_bc (X, Y, W in an .npz). Each minibatch adds
    # bc_coef * weighted cross-entropy on bc_batch of them, so PPO keeps a
    # cloned route while it learns elsewhere (on Bloodbath, PPO after
    # cloning drifted off each searched route within an hour).
    bc_data: str | None = None
    bc_coef: float = 0.0
    bc_batch: int = 256
    # Hard pairs: before each epoch, pairs where the policy gives the demo
    # action less than bc_hard_p are added to every minibatch's BC batch (up
    # to bc_batch of them, lowest first). Why: a route breaks on one flipped
    # decision; with 16k pairs and 256 sampled per minibatch that pair was
    # seen in ~1.6% of minibatches and weighed 1/256 there, so PPO flipped it
    # (Bloodbath 2026-10-18: one decision at x ~17,490, mean bc_loss 0.0002,
    # eval 85% -> 41-58% in 1.5 h). 0 disables.
    bc_hard_p: float = 0.0


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
        self.act_device = torch.device(cfg.act_device) if cfg.act_device else self.device
        self._bc = None
        if cfg.bc_data and cfg.bc_coef > 0:
            d = np.load(cfg.bc_data)
            self._bc = (self._tensor(d["X"]), torch.as_tensor(d["Y"], dtype=torch.int64, device=self.device),
                        self._tensor(d["W"]))
        self._sync_actor()

    def add_bc(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> None:
        """Append (obs, action, weight) pairs to the BC anchor (train.eval_ratchet)."""
        if self._bc is None:
            return
        bx, by, bw = self._bc
        self._bc = (torch.cat([bx, self._tensor(X)]),
                    torch.cat([by, torch.as_tensor(Y, dtype=torch.int64, device=self.device)]),
                    torch.cat([bw, self._tensor(W)]))

    def bc_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._bc is None:
            return None
        return tuple(t.detach().cpu().numpy() for t in self._bc)

    def _sync_actor(self) -> None:
        """Point act() at the networks, or at fresh copies on act_device."""
        if self.act_device == self.device:
            self._act_policy, self._act_value = self.policy, self.value
            return
        import copy

        self._act_policy = copy.deepcopy(self.policy).to(self.act_device).eval()
        self._act_value = copy.deepcopy(self.value).to(self.act_device).eval()

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
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.act_device)
            logp_all = torch.log_softmax(self._act_policy(x), dim=-1)
            if deterministic:
                action = torch.argmax(logp_all, dim=-1)
            else:
                action = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
            logprob = logp_all.gather(-1, action[:, None]).squeeze(-1)
            value = self._act_value(x).squeeze(-1)
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
        if self._bc is not None:
            stats["bc_loss"] = []
        for _epoch in range(cfg.update_epochs):
            perm = torch.randperm(B, device=self.device)
            epoch_kl = []
            hard = None
            if self._bc is not None and cfg.bc_hard_p > 0:
                bx, by, _bw = self._bc
                with torch.no_grad():
                    p_demo = torch.softmax(self.policy(bx), dim=-1).gather(1, by[:, None]).squeeze(1)
                low = (p_demo < cfg.bc_hard_p).nonzero().squeeze(1)
                if len(low):
                    hard = low[p_demo[low].argsort()[: cfg.bc_batch]]
                stats.setdefault("bc_hard", []).append(float(len(low)))
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
                if self._bc is not None:
                    bx, by, bw = self._bc
                    i = torch.randint(0, len(by), (min(cfg.bc_batch, len(by)),), device=self.device)
                    if hard is not None:
                        i = torch.cat([i, hard])
                    bc_loss = (F.cross_entropy(self.policy(bx[i]), by[i], reduction="none") * bw[i]).sum() / bw[i].sum()
                    loss = loss + cfg.bc_coef * bc_loss
                    stats["bc_loss"].append(bc_loss.item())

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
        self._sync_actor()
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
        self._sync_actor()
