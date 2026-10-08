"""M6: PPOAgent behind the Agent interface. The CartPole gate is marked slow:
run it with  pytest -m slow tests/learn/test_ppo.py"""

import numpy as np
import pytest
from gymnasium import spaces

from gdrl.agents.base import Rollout
from gdrl.agents.ppo import PPOAgent, PPOConfig

OBS = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
ACT = spaces.Discrete(2)
METRICS = {"policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance", "learning_rate"}


def fake_rollout(agent: PPOAgent, T: int = 32, N: int = 2, seed: int = 0) -> Rollout:
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, size=(T, N, 4)).astype(np.float32)
    extras: dict[str, list] = {}
    actions = []
    for t in range(T):
        out = agent.act(obs[t])
        actions.append(out.action)
        for k, v in out.extras.items():
            extras.setdefault(k, []).append(v)
    terminated = rng.random((T, N)) < 0.05
    truncated = (rng.random((T, N)) < 0.02) & ~terminated
    return Rollout(
        obs=obs,
        actions=np.stack(actions),
        rewards=rng.normal(size=(T, N)).astype(np.float32),
        terminated=terminated,
        truncated=truncated,
        final_obs=np.where((terminated | truncated)[..., None], obs, 0).astype(np.float32),
        next_obs=rng.uniform(-1, 1, size=(N, 4)).astype(np.float32),
        extras={k: np.stack(v) for k, v in extras.items()},
    )


def make_agent(**overrides) -> PPOAgent:
    cfg = PPOConfig(rollout_steps=32, device="cpu", total_updates=10, **overrides)
    return PPOAgent(OBS, ACT, cfg)


def test_act_shapes():
    out = make_agent().act(np.zeros((3, 4), dtype=np.float32))
    assert out.action.shape == (3,)
    assert set(out.action) <= {0, 1}
    assert out.extras["logprob"].shape == (3,) and out.extras["value"].shape == (3,)


def test_deterministic_act_is_repeatable():
    agent = make_agent()
    obs = np.random.default_rng(0).uniform(-1, 1, size=(16, 4)).astype(np.float32)
    a = agent.act(obs, deterministic=True).action
    b = agent.act(obs, deterministic=True).action
    np.testing.assert_array_equal(a, b)


def test_update_returns_metrics():
    agent = make_agent()
    metrics = agent.update(fake_rollout(agent))
    assert METRICS <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_ratio_is_one_without_learning():
    """With lr = 0 the policy never moves, so every ratio is exactly 1: no KL, no clipping.
    If this fails, the logprobs in update() do not match the ones from act()."""
    agent = make_agent(learning_rate=0.0, anneal_lr=False)
    metrics = agent.update(fake_rollout(agent))
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clipfrac"] == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_round_trip(tmp_path):
    agent = make_agent()
    agent.update(fake_rollout(agent))
    agent.save(tmp_path / "ckpt.pt")
    other = make_agent()
    other.load(tmp_path / "ckpt.pt")
    obs = np.random.default_rng(1).uniform(-1, 1, size=(16, 4)).astype(np.float32)
    np.testing.assert_array_equal(agent.act(obs, True).action, other.act(obs, True).action)


@pytest.mark.slow
def test_cartpole_gate(tmp_path):
    """M6 gate: average return >= 475 on CartPole-v1 (one seed here; run 3 by hand)."""
    from gdrl import evaluate, train

    run_dir = train.main(["--config", "configs/cartpole_ppo.yaml",
                          "--set", f"train.runs_dir={tmp_path}", "--set", "train.tensorboard=false"])
    results = evaluate.main(["--run", str(run_dir), "--episodes", "20"])
    assert results["levels"]["None"]["return_mean"] >= 475
