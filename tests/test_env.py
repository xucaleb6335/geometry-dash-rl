import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gdrl.env.backends import MockBackend, SocketBackend
from gdrl.env.factory import make_env, make_vector_env
from gdrl.env.gd_env import GDEnv
from gdrl.env.mock_server import MockServer
from gdrl.features.encoder import DebugEncoder
from gdrl.rewards.reward import ConstantReward

pytestmark = pytest.mark.filterwarnings("ignore:.*environment not having a spec.*")


def test_check_env_mock():
    check_env(GDEnv(MockBackend(level_seed=2), DebugEncoder(), ConstantReward()))


def test_check_env_socket():
    with MockServer(level_seed=2) as server:
        env = GDEnv(SocketBackend(port=server.port), DebugEncoder(), ConstantReward())
        check_env(env)
        env.close()


def test_death_is_terminated_not_truncated():
    env = GDEnv(MockBackend(level_seed=3), DebugEncoder(), ConstantReward(), max_steps=10_000)
    env.reset(seed=0)
    while True:
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert terminated and not truncated and info["dead"]


def test_step_limit_is_truncated():
    env = GDEnv(MockBackend(level_seed=3), DebugEncoder(), ConstantReward(), max_steps=5)
    env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_level_ids_are_sampled():
    env = GDEnv(MockBackend(level_seed=1), DebugEncoder(), ConstantReward(), level_ids=[1, 2, 3])
    seen = {env.reset(seed=s)[1]["level_id"] for s in range(30)}
    assert seen == {1, 2, 3}


def test_vector_env_keeps_final_obs():
    envs = make_vector_env({"id": "gd", "backend": "mock", "level_ids": [3], "max_steps": 10_000}, num_envs=2)
    envs.reset(seed=0)
    for _ in range(500):
        _, _, terminated, truncated, info = envs.step(np.zeros(2, dtype=np.int64))
        if (terminated | truncated).any():
            assert info["_final_obs"].any()
            assert "progress" in info["final_info"]
            break
    else:
        pytest.fail("no episode ended")


def test_factory_builds_gymnasium_ids():
    env = make_env({"id": "CartPole-v1"})
    assert env.action_space.n == 2


def test_socket_backend_needs_one_env():
    with pytest.raises(ValueError):
        make_vector_env({"id": "gd", "backend": "socket"}, num_envs=2)
