"""M6: compute_gae against hand-worked numbers.

Single env, T = 3, rewards all 1, gamma = 0.9, lambda = 0.8 (so gamma*lambda = 0.72),
values V = [0.5, 0.4, 0.3].

Plain (no episode end):  next_values = [0.4, 0.3, 0.2]
  delta = [1 + .9*.4 - .5, 1 + .9*.3 - .4, 1 + .9*.2 - .3] = [0.86, 0.87, 0.88]
  A2 = 0.88
  A1 = 0.87 + 0.72 * 0.88   = 1.5036
  A0 = 0.86 + 0.72 * 1.5036 = 1.942592

Terminated at t = 1 (death): no bootstrap at t = 1, no carry from t = 2.
  delta1 = 1 - 0.4 = 0.6, A1 = 0.6, A2 = 0.88, A0 = 0.86 + 0.72 * 0.6 = 1.292

Truncated at t = 1 with V(final_obs) = 0.7: bootstrap, but no carry from t = 2.
  delta1 = 1 + 0.9 * 0.7 - 0.4 = 1.23, A1 = 1.23, A2 = 0.88, A0 = 0.86 + 0.72 * 1.23 = 1.7456
"""

import numpy as np
import pytest

from gdrl.agents.ppo import compute_gae

GAMMA, LAM = 0.9, 0.8
REWARDS = np.ones((3, 1), dtype=np.float32)
VALUES = np.array([[0.5], [0.4], [0.3]], dtype=np.float32)
NO = np.zeros((3, 1), dtype=bool)


def at(t: int) -> np.ndarray:
    mask = NO.copy()
    mask[t, 0] = True
    return mask


def test_plain():
    next_values = np.array([[0.4], [0.3], [0.2]], dtype=np.float32)
    adv, ret = compute_gae(REWARDS, VALUES, next_values, NO, NO, GAMMA, LAM)
    np.testing.assert_allclose(adv[:, 0], [1.942592, 1.5036, 0.88], rtol=1e-5)
    np.testing.assert_allclose(ret, adv + VALUES, rtol=1e-6)


def test_terminated():
    next_values = np.array([[0.4], [0.3], [0.2]], dtype=np.float32)  # [1] must be ignored
    adv, _ = compute_gae(REWARDS, VALUES, next_values, at(1), NO, GAMMA, LAM)
    np.testing.assert_allclose(adv[:, 0], [1.292, 0.6, 0.88], rtol=1e-5)


def test_truncated():
    next_values = np.array([[0.4], [0.7], [0.2]], dtype=np.float32)
    adv, _ = compute_gae(REWARDS, VALUES, next_values, NO, at(1), GAMMA, LAM)
    np.testing.assert_allclose(adv[:, 0], [1.7456, 1.23, 0.88], rtol=1e-5)


def test_envs_are_independent():
    next_values = np.array([[0.4], [0.3], [0.2]], dtype=np.float32)
    rewards = np.repeat(REWARDS, 2, axis=1)
    values = np.repeat(VALUES, 2, axis=1)
    nv = np.repeat(next_values, 2, axis=1)
    term = np.zeros((3, 2), dtype=bool)
    term[1, 1] = True  # only env 1 dies
    adv, _ = compute_gae(rewards, values, nv, term, np.zeros_like(term), GAMMA, LAM)
    np.testing.assert_allclose(adv[:, 0], [1.942592, 1.5036, 0.88], rtol=1e-5)
    np.testing.assert_allclose(adv[:, 1], [1.292, 0.6, 0.88], rtol=1e-5)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_lambda_extremes(lam):
    """lambda = 0 is one-step TD; lambda = 1 is the Monte Carlo return minus V."""
    next_values = np.array([[0.4], [0.3], [0.2]], dtype=np.float32)
    adv, _ = compute_gae(REWARDS, VALUES, next_values, NO, NO, GAMMA, lam)
    if lam == 0.0:
        np.testing.assert_allclose(adv[:, 0], [0.86, 0.87, 0.88], rtol=1e-5)
    else:
        mc = [1 + 0.9 + 0.81 + 0.729 * 0.2, 1 + 0.9 + 0.81 * 0.2, 1 + 0.9 * 0.2]
        np.testing.assert_allclose(adv[:, 0], np.array(mc) - VALUES[:, 0], rtol=1e-5)
