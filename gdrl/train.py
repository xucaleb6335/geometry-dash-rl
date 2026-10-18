"""Training loop: collect rollouts from vector envs, hand them to the agent.

    python -m gdrl.train --config configs/cartpole_ppo.yaml
    python -m gdrl.train --config configs/mock_ppo.yaml --set seed=2
    python -m gdrl.train --config configs/gd_ppo.yaml --set train.resume=runs/<run>/checkpoints/final.pt

Everything algorithm-specific lives in the agent; this file only collects
data, logs and checkpoints. Plumbing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from gdrl.agents.base import Rollout
from gdrl.agents.registry import make_agent
from gdrl.config import load_config, save_config
from gdrl.env.factory import make_vector_env
from gdrl.utils import RunLogger, make_run_dir, set_seed


def _store_extras(buffers: dict[str, np.ndarray], t: int, extras: dict[str, np.ndarray], T: int) -> None:
    for key, value in extras.items():
        value = np.asarray(value)
        if key not in buffers:
            buffers[key] = np.zeros((T, *value.shape), dtype=value.dtype)
        buffers[key][t] = value


def greedy_episode(agent, env, options: dict[str, Any] | None = None, record: list | None = None) -> dict[str, float]:
    """Run one deterministic episode on a single (unvectorized) env.

    With ``record`` (a list), (obs, action) of every decision is appended to it."""
    obs, info = env.reset(options=dict(options) if options else None)
    total, length = 0.0, 0
    while True:
        action = int(agent.act(obs[None], deterministic=True).action[0])
        if record is not None:
            record.append((np.array(obs, copy=True), action))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        length += 1
        if terminated or truncated:
            break
    return {"return": total, "length": float(length), "progress": float(info.get("progress", 0.0)),
            "completed": float(bool(info.get("completed", False)))}


def _explore(out, envs, rng: np.random.Generator) -> None:
    """Epsilon exploration for envs that ask for it (``explore_eps`` attribute,
    set by the curriculum on starts the agent never passes).

    With probability eps the action is replaced by a uniform random one, and
    the stored log-probability is that of the behaviour policy,
    ``(1 - eps) * pi(a) + eps / n``, so PPO's ratio stays consistent. Two
    actions only (the policy's own probability of the other action is
    ``1 - pi(sampled)``)."""
    if "logprob" not in out.extras:
        return
    for i, env in enumerate(envs.envs):
        eps = float(getattr(env, "explore_eps", 0.0) or 0.0)
        if eps <= 0.0:
            continue
        a = int(out.action[i])
        p_a = float(np.exp(out.extras["logprob"][i]))
        if rng.random() < eps:
            new = int(rng.integers(2))
            p_new = p_a if new == a else 1.0 - p_a
            a, p_a = new, p_new
            out.action[i] = a
        out.extras["logprob"][i] = np.log((1.0 - eps) * p_a + eps / 2.0 + 1e-12)


def train(cfg: dict[str, Any]) -> Path:
    seed = int(cfg["seed"])
    tcfg = cfg["train"]
    num_envs = int(tcfg["num_envs"])
    total_steps = int(tcfg["total_steps"])

    set_seed(seed)
    run_dir = make_run_dir(tcfg["runs_dir"], cfg["name"], seed)
    save_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir, tensorboard=bool(tcfg.get("tensorboard", True)))

    envs = make_vector_env(cfg["env"], num_envs)
    agent = make_agent(
        cfg["agent"],
        envs.single_observation_space,
        envs.single_action_space,
        num_envs=num_envs,
        total_steps=total_steps,
        seed=seed,
    )
    set_seed(seed)  # again, now that the agent may have imported torch
    if tcfg.get("torch_threads") and "torch" in sys.modules:
        # Small MLPs run faster on 1-2 threads than on every core.
        sys.modules["torch"].set_num_threads(int(tcfg["torch_threads"]))

    T = int(agent.rollout_steps)
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    obs, _ = envs.reset(seed=seed)

    ep_return = np.zeros(num_envs, dtype=np.float64)
    ep_length = np.zeros(num_envs, dtype=np.int64)
    global_step = 0
    update = 0
    if tcfg.get("resume"):
        # Continue from a checkpoint: the agent keeps its weights, optimizer and
        # update count, so learning-rate annealing picks up where it stopped.
        # Keep total_steps the same as the original run.
        agent.load(tcfg["resume"])
        update = int(getattr(agent, "num_updates_done", 0))
        global_step = update * T * num_envs
        print(f"resumed from {tcfg['resume']} at update {update}, step {global_step}")
    started = time.time()
    rng_explore = np.random.default_rng(int(cfg.get("seed", 0)) + 12345)
    start_step = global_step  # steps per second counts this session only (not resumed steps)
    print(f"run dir: {run_dir}")

    try:
        best_eval = float((tcfg.get("eval_ratchet") or {}).get("start", -1.0))  # train.eval_ratchet; >= 0 also adds the first eval
        while global_step < total_steps:
            obs_buf = np.zeros((T, num_envs, *obs_shape), dtype=obs_dtype)
            final_buf = np.zeros_like(obs_buf)
            act_buf = np.zeros((T, num_envs), dtype=np.int64)
            rew_buf = np.zeros((T, num_envs), dtype=np.float32)
            term_buf = np.zeros((T, num_envs), dtype=bool)
            trunc_buf = np.zeros((T, num_envs), dtype=bool)
            extras: dict[str, np.ndarray] = {}
            finished: list[dict[str, float]] = []

            for t in range(T):
                out = agent.act(obs)
                _explore(out, envs, rng_explore)
                next_obs, reward, terminated, truncated, info = envs.step(out.action)

                obs_buf[t], act_buf[t], rew_buf[t] = obs, out.action, reward
                term_buf[t], trunc_buf[t] = terminated, truncated
                _store_extras(extras, t, out.extras, T)

                ep_return += reward
                ep_length += 1
                done = terminated | truncated
                for i in np.nonzero(done)[0]:
                    final_buf[t, i] = info["final_obs"][i]
                    episode = {"return": ep_return[i], "length": ep_length[i]}
                    final_info = info.get("final_info", {})
                    for key in ("progress", "completed"):
                        if key in final_info and final_info.get(f"_{key}", np.ones(num_envs, bool))[i]:
                            episode[key] = float(final_info[key][i])
                    finished.append(episode)
                    ep_return[i] = 0.0
                    ep_length[i] = 0

                obs = next_obs
                global_step += num_envs

            rollout = Rollout(
                obs=obs_buf,
                actions=act_buf,
                rewards=rew_buf,
                terminated=term_buf,
                truncated=trunc_buf,
                final_obs=final_buf,
                next_obs=obs.copy(),
                extras=extras,
            )
            metrics = {f"train/{k}": v for k, v in agent.update(rollout).items()}
            update += 1

            sps = (global_step - start_step) / max(time.time() - started, 1e-9)
            metrics["perf/steps_per_second"] = sps
            if finished:
                for key in finished[0]:
                    values = [ep[key] for ep in finished if key in ep]
                    metrics[f"episode/{key}_mean"] = float(np.mean(values))
                metrics["episode/count"] = len(finished)
            logger.log(global_step, metrics)

            line = f"step {global_step:>9d}  update {update:>5d}  sps {sps:7.0f}"
            if finished:
                line += f"  return {metrics['episode/return_mean']:8.2f}"
                if "episode/progress_mean" in metrics:
                    line += f"  progress {metrics['episode/progress_mean']:6.1%}"
            print(line, flush=True)

            eval_every = int(tcfg.get("eval_every", 0) or 0)
            if eval_every and update % eval_every == 0:
                # One greedy episode from eval_options (e.g. {start: 0}, the
                # level start under a curriculum). The episode in progress is
                # dropped; the rollout already bootstrapped from its last state.
                ratchet = tcfg.get("eval_ratchet") or {}
                record = [] if ratchet else None
                result = greedy_episode(agent, envs.envs[0], tcfg.get("eval_options"), record)
                logger.log(global_step, {f"eval/{k}": v for k, v in result.items()})
                if ratchet and hasattr(agent, "add_bc") and result["progress"] > best_eval + float(ratchet.get("margin", 0.002)):
                    # Ratchet: a greedy run that beats the best one so far is
                    # added to the BC anchor (minus its last decisions, which
                    # led into the death), so the next updates cannot lose it.
                    # Why: PPO's greedy run flipped between consecutive updates
                    # (Bloodbath 2026-10-18: 91.5% at one update, 89.1% at the
                    # next, never kept).
                    keep = record[: max(0, len(record) - int(ratchet.get("drop", 50)))]
                    if best_eval >= 0 and keep:
                        agent.add_bc(np.stack([o for o, _ in keep]).astype(np.float32),
                                     np.array([a for _, a in keep], dtype=np.int64),
                                     np.full(len(keep), float(ratchet.get("weight", 3.0)), dtype=np.float32))
                        arrays = agent.bc_arrays()
                        if arrays is not None:
                            np.savez_compressed(run_dir / "checkpoints" / "bc_ratchet.npz", X=arrays[0], Y=arrays[1], W=arrays[2])
                        agent.save(run_dir / "checkpoints" / "best_eval.pt")
                        print(f"ratchet: eval {result['progress']:.1%} > best {best_eval:.1%}; "
                              f"{len(keep)} decisions added to the BC anchor", flush=True)
                    best_eval = result["progress"]
                print(f"eval  step {global_step:>9d}  progress {result['progress']:6.1%}  "
                      f"completed {int(result['completed'])}  length {int(result['length'])}", flush=True)
                obs, _ = envs.reset()
                ep_return[:] = 0.0
                ep_length[:] = 0

            if update % int(tcfg["checkpoint_every"]) == 0:
                agent.save(run_dir / "checkpoints" / f"update_{update:06d}.pt")
    finally:
        agent.save(run_dir / "checkpoints" / "final.pt")
        envs.close()
        logger.close()
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument("--config", required=True, help="YAML file under configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.total_steps=20000")
    args = parser.parse_args(argv)
    return train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
