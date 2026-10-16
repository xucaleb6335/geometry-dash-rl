"""Baselines: what your PPO has to beat (M8). Every row in the results table
needs one of these next to it, or a high score means nothing."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from gdrl.agents.base import ActOutput, Agent, Rollout
from gdrl.env.protocol import BLOCK, ObjectKind


class RandomAgent(Agent):
    """Uniform random actions. The floor. Also the agent used by smoke tests."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 rollout_steps: int = 128, seed: int | None = None):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("RandomAgent expects a Discrete action space")
        self.n_actions = int(action_space.n)
        self.rollout_steps = rollout_steps
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        n = obs.shape[0]
        return ActOutput(action=self.rng.integers(self.n_actions, size=n, dtype=np.int64))

    def update(self, rollout: Rollout) -> dict[str, float]:
        return {}


class RuleBasedAgent(Agent):
    """Jump when the nearest obstacle ahead is within a distance band. (M8)

    Works on ``ObjectListEncoder`` observations, so it doubles as a check that
    your encoder carries enough information: if a 10-line rule can beat the
    level from the observation, a network can too. Tune the bands per object
    kind: a block needs an earlier jump than a spike.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        rollout_steps: int = 128,
        max_dx_blocks: float = 20.0,
        hazard_band: tuple[float, float] = (28.0, 34.0),
        solid_band: tuple[float, float] = (36.0, 42.0),
    ):
        from gdrl.features.encoder import ObjectListEncoder

        self.rollout_steps = rollout_steps
        enc = ObjectListEncoder
        n = int(np.prod(observation_space.shape))
        # Recover k (and near_slots) from the observation length.
        layouts = [(k, near) for k in range(1, 65) for near in range(k + 1)
                   if enc.PLAYER_FEATURES + enc.SLOT_SIZE * k + near * len(enc.RBF_CENTERS) == n]
        if len(observation_space.shape) != 1 or not layouts:
            raise ValueError(f"RuleBasedAgent needs ObjectListEncoder observations, got shape {observation_space.shape}")
        self.k = layouts[0][0]
        self.player = enc.PLAYER_FEATURES
        self.slot = enc.SLOT_SIZE
        # max_dx_blocks must match the encoder's; the observation stores dx / max_dx.
        self.max_dx = max_dx_blocks * BLOCK
        self.hazard_col = 4 + enc.KINDS.index(ObjectKind.HAZARD)
        self.solid_col = 4 + enc.KINDS.index(ObjectKind.SOLID)
        self.bands = {self.hazard_col: tuple(hazard_band), self.solid_col: tuple(solid_band)}

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        slots = obs[:, self.player : self.player + self.k * self.slot].reshape(len(obs), self.k, self.slot)
        # Distance from the player's centre to each object's left edge, in units.
        left = (slots[..., 0] - slots[..., 2] / 2) * self.max_dx
        press = np.zeros(len(obs), dtype=bool)
        for col, (lo, hi) in self.bands.items():
            press |= ((slots[..., col] == 1.0) & (left >= lo) & (left <= hi)).any(axis=1)
        return ActOutput(action=press.astype(np.int64))

    def update(self, rollout: Rollout) -> dict[str, float]:
        return {}


class GeneticTimingAgent(Agent):
    """Evolve a fixed sequence of button presses for one level, ignoring the
    observation entirely. (M8)

    Pure memorization: it can beat a deterministic level and nothing else,
    which is exactly why it belongs in the comparison. Sketch: a population of
    press/release sequences, fitness = progress reached, keep the best, mutate,
    repeat. Decide whether this fits the act/update loop or wants its own
    script.

    Decision: its own loop (``evolve``, driven by ``gdrl/tools/genetic.py``).
    Fitness needs whole episodes from a known start, which the rollout loop
    does not give. ``act`` replays the best sequence found so far, one entry
    per call; call ``start_episode`` before each episode.

    Mutations concentrate just before where the parent died, since the
    presses that matter are the last few before the death.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        rollout_steps: int = 128,
        population: int = 32,
        elite: int = 4,
        max_len: int = 5000,
        mutation_window: int = 40,
        mutation_rate: float = 0.15,
        seed: int | None = None,
    ):
        self.rollout_steps = rollout_steps
        self.population = population
        self.elite = elite
        self.max_len = max_len
        self.mutation_window = mutation_window
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        self.best = np.zeros(max_len, dtype=bool)
        self.best_fitness = -np.inf
        self._t = 0

    def start_episode(self) -> None:
        self._t = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        press = self.best[min(self._t, self.max_len - 1)]
        self._t += 1
        return ActOutput(action=np.full(len(obs), int(press), dtype=np.int64))

    def update(self, rollout: Rollout) -> dict[str, float]:
        return {}

    @staticmethod
    def play(env, sequence: np.ndarray) -> tuple[float, int, bool]:
        """Run one episode with a fixed sequence. Returns (fitness, steps, completed)."""
        env.reset()
        info: dict = {}
        steps = 0
        for steps in range(1, len(sequence) + 1):
            _, _, terminated, truncated, info = env.step(int(sequence[steps - 1]))
            if terminated or truncated:
                break
        completed = bool(info.get("completed", False))
        return float(info.get("progress", 0.0)) + float(completed), steps, completed

    def _mutate(self, parent: np.ndarray, died_at: int) -> np.ndarray:
        child = parent.copy()
        lo = max(0, died_at - self.mutation_window)
        hi = min(self.max_len, died_at + 1)
        flips = self.rng.random(hi - lo) < self.mutation_rate
        child[lo:hi] ^= flips
        return child

    def evolve(self, env, generations: int, log=print) -> dict[str, float]:
        pop = [self.rng.random(self.max_len) < 0.05 for _ in range(self.population)]
        scored: list[tuple[float, int, np.ndarray]] = []
        for gen in range(generations):
            scored = []
            for seq in pop:
                fitness, steps, completed = self.play(env, seq)
                scored.append((fitness, steps, seq))
            scored.sort(key=lambda r: r[0], reverse=True)
            if scored[0][0] > self.best_fitness:
                self.best_fitness, self.best = scored[0][0], scored[0][2].copy()
            log(f"generation {gen:4d}  best {scored[0][0]:.3f}  mean {np.mean([r[0] for r in scored]):.3f}")
            if self.best_fitness >= 2.0:  # progress 1 + completion 1
                break
            parents = scored[: self.elite]
            pop = [p[2] for p in parents]
            while len(pop) < self.population:
                fitness, steps, seq = parents[int(self.rng.integers(len(parents)))]
                pop.append(self._mutate(seq, steps - 1))
        return {"best_fitness": float(self.best_fitness)}

    def save(self, path) -> None:
        np.savez(path, best=self.best, best_fitness=self.best_fitness)

    def load(self, path) -> None:
        data = np.load(path)
        self.best, self.best_fitness = data["best"], float(data["best_fitness"])
