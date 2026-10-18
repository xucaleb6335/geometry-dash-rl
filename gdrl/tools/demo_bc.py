"""Teach a PPO policy a demo by behaviour cloning (game must be open, lock held).

    python -m gdrl.tools.demo_bc --config configs/gd_bb_community.yaml \
        --ckpt runs/<run>/checkpoints/update_007875.pt \
        --demo configs/demos/bb_ufo_4tick.npy --start-pos 4 \
        --out runs/bc-gd-bb-community/checkpoints/bc.pt

Why: on Bloodbath's UFO tunnel (x 11,715-12,225) PPO kept pressing while
falling into the ceiling spikes for 30M steps; random exploration almost
never finds the ~20-unit-wide path. A checkpoint search found one at the
policy's own decision rate (one action per frame_skip ticks), so the policy
can represent it. This replays the demo in game, records (observation, demo
action) at every decision, and also records the policy's own greedy
(observation, action) on its run from the level start, so cloning the demo
does not change what it already does elsewhere. It then trains the policy
head with cross-entropy until it reproduces both, and checks in game that
the greedy policy now follows the demo. The value net and optimizer are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gdrl.agents.registry import make_agent
from gdrl.config import load_config
from gdrl.env.backends import wait_until_running
from gdrl.env.factory import make_env


def demo_actions(ticks: np.ndarray, frame_skip: int) -> np.ndarray:
    n = len(ticks) // frame_skip
    blocks = ticks[: n * frame_skip].reshape(n, frame_skip)
    mixed = int((blocks.min(1) != blocks.max(1)).sum())
    if mixed:
        raise SystemExit(f"demo has {mixed} decisions with mixed inputs; it needs frame_skip={frame_skip} granularity")
    return blocks[:, 0].astype(int)


def greedy(agent, obs) -> int:
    return int(agent.act(obs[None], deterministic=True).action[0])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--demo", required=True, help="per-tick inputs (.npy, bool) from --start-pos")
    ap.add_argument("--start-pos", type=int, default=4)
    ap.add_argument("--until-x", type=float, default=float("inf"), help="use demo decisions only up to this x")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--demo-weight", type=float, default=3.0)
    ap.add_argument("--repeats", type=int, default=3,
                    help="record demo and retention runs this many times: an attempt's dynamics depend a "
                         "little on how the previous one ended, so the observations differ slightly")
    ap.add_argument("--retain-until-x", type=float, default=None,
                    help="keep the policy's own actions from the level start up to this x (default: where the demo "
                         "starts). Set it to where the demo's own actions start (the search root) when the demo "
                         "opens with the policy's actions: from the level start it arrives in a slightly different state")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    env_cfg = dict(cfg["env"])
    env_cfg.pop("curriculum", None)
    fs = int(env_cfg.get("frame_skip", 4))
    env = make_env(env_cfg)
    wait_until_running(env.unwrapped.backend, a.start_pos)
    agent = make_agent(cfg["agent"], env.observation_space, env.action_space, num_envs=1, total_steps=1)
    agent.load(a.ckpt)
    acts = demo_actions(np.load(a.demo).astype(bool), fs)

    d_obs, d_act, r_obs, r_act = [], [], [], []
    demo_x0 = None
    for rep_i in range(a.repeats):
        # 1. Demo pairs.
        obs, _ = env.reset(options={"start_pos": a.start_pos})
        demo_x0 = env.unwrapped.state.player_x
        n0, agree = len(d_act), 0
        for act in acts:
            if env.unwrapped.state.player_x > a.until_x:
                break
            d_obs.append(obs)
            d_act.append(int(act))
            agree += greedy(agent, obs) == act
            obs, _, term, trunc, info = env.step(int(act))
            if term or trunc:
                break
        end_x = env.unwrapped.state.player_x
        print(f"demo run {rep_i}: {len(d_act) - n0} decisions, policy already agrees on {agree}; ends at x={end_x:.0f} "
              f"dead={env.unwrapped.state.dead}", flush=True)
        if env.unwrapped.state.dead and end_x <= a.until_x:
            raise SystemExit("the demo does not survive when replayed at the policy's decision rate")

        # 2. Retention pairs: the policy's own greedy run from the level start,
        # up to where the demo starts (past it the demo is the teacher; the
        # policy's own actions there include the mistake being fixed).
        obs, _ = env.reset(options={"start_pos": -2})
        n0 = len(r_act)
        while True:
            act = greedy(agent, obs)
            if env.unwrapped.state.player_x < (a.retain_until_x if a.retain_until_x is not None else demo_x0):
                r_obs.append(obs)
                r_act.append(act)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        before_x = env.unwrapped.state.player_x
        print(f"retention run {rep_i}: {len(r_act) - n0} decisions, greedy run from the start ends at x={before_x:.0f}",
              flush=True)

    # 3. Clone.
    dev = agent.device
    X = torch.as_tensor(np.array(d_obs + r_obs), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.array(d_act + r_act), dtype=torch.long, device=dev)
    W = torch.cat([torch.full((len(d_act),), a.demo_weight), torch.ones(len(r_act))]).to(dev)
    nd = len(d_act)
    opt = torch.optim.Adam(agent.policy.parameters(), lr=a.lr)
    for ep in range(a.max_epochs):
        logits = agent.policy(X)
        loss = (F.cross_entropy(logits, Y, reduction="none") * W).sum() / W.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == a.max_epochs - 1:
            agent._sync_actor()  # judge the copy that acts in game (CPU)
            with torch.no_grad():
                pred = agent._act_policy(X.to(agent.act_device)).argmax(-1).to(dev)
                dacc = (pred[:nd] == Y[:nd]).float().mean().item()
                racc = (pred[nd:] == Y[nd:]).float().mean().item()
            print(f"epoch {ep}: loss {loss.item():.4f} demo acc {dacc:.4f} retention acc {racc:.4f}", flush=True)
            if dacc == 1.0 and racc == 1.0:
                break
    agent._sync_actor()

    # 4. Check in game (several attempts: see --repeats).
    for rep_i in range(a.repeats):
        obs, _ = env.reset(options={"start_pos": a.start_pos})
        while True:
            obs, _, term, trunc, info = env.step(greedy(agent, obs))
            if term or trunc or env.unwrapped.state.player_x > a.until_x:
                break
        sp_x = env.unwrapped.state.player_x
        obs, _ = env.reset(options={"start_pos": -2})
        while True:
            obs, _, term, trunc, info = env.step(greedy(agent, obs))
            if term or trunc:
                break
        print(f"check {rep_i}: greedy from start_pos {a.start_pos} reaches x={sp_x:.0f} (demo used to {a.until_x:.0f}); "
              f"from the level start x={env.unwrapped.state.player_x:.0f}", flush=True)
    env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    agent.save(a.out)
    # The pairs, for PPO's behaviour-cloning anchor (agent.bc_data / bc_coef).
    np.savez_compressed(Path(a.out).with_suffix(".bcdata.npz"), X=X.cpu().numpy(), Y=Y.cpu().numpy(), W=W.cpu().numpy())
    print("saved", a.out)


if __name__ == "__main__":
    main()
