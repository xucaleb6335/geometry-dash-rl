"""Distill search solutions into a policy: behaviour cloning, then DAgger with
the checkpoint search as the expert.

1. BC: replay a solution (per-tick buttons) from RESET, recording
   (observation, button) at every decision. Train the policy network
   (the PPO agent's policy, so ``gdrl.evaluate`` and PPO fine-tuning load it)
   with cross-entropy until it fits.
2. DAgger: play the greedy policy from the level start, saving a snapshot
   every few decisions. Where it dies, restore a snapshot from shortly before,
   run the segment search from that (policy-visited) state to a point past
   the death, and add the states along that recovery with the search's
   buttons to the data. Retrain. Repeat until the greedy policy completes
   the level from RESET with no snapshots, which is the verification.

The labels come from the states the policy actually reaches, which is what
fixes the compounding errors of plain BC.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from gdrl.agents.ppo import PPOAgent, PPOConfig
from gdrl.features.encoder import Encoder
from gdrl.search.segment_search import SearchConfig, SegmentSearch, center_presses, simplify_inputs


@dataclass
class DistillConfig:
    decision_ticks: int = 2  # must match the search's and the env's frame_skip
    hidden_sizes: tuple[int, ...] = (256, 256)
    lr: float = 1e-3
    batch_size: int = 512
    epochs_per_round: int = 30
    target_accuracy: float = 0.995
    max_dagger_rounds: int = 60
    snapshot_every: int = 15  # decisions between snapshots during policy rollouts
    back_decisions: int = 45  # how far before a death to restart the expert from
    goal_ahead: float = 240.0  # the expert must survive this far past the death (units)
    max_decisions: int = 20_000
    starts: int = 24  # rollouts per round start from this many states spread along the solution
    simplify: bool = True  # drop unneeded presses from expert inputs, then centre presses in their windows
    seed: int = 0
    search: SearchConfig = field(default_factory=lambda: SearchConfig(max_rounds=12, max_backtrack=4,
                                                                      time_limit_s=30, max_rollouts=6000))


class Distiller:
    def __init__(self, backend, encoder: Encoder, cfg: DistillConfig, *, level_id: int = 0, start_pos: int = -1,
                 log=None, level_name: str = "", print_fn: Callable[[str], None] = print):
        if int(getattr(backend, "max_snapshots", 0)) < 64:
            raise ValueError("distillation needs a backend with snapshots")
        self.backend = backend
        self.encoder = encoder
        # The expert must act on the policy's grid, or its labels cannot be imitated.
        cfg.search = SearchConfig(**{**asdict(cfg.search), "decision_ticks": cfg.decision_ticks,
                                     "fine_decision_ticks": cfg.decision_ticks})
        self.cfg = cfg
        self.level_id = level_id
        self.start_pos = start_pos
        self.log = log
        self.level_name = level_name
        self.print = print_fn
        torch.manual_seed(cfg.seed)
        self.agent = PPOAgent(encoder.observation_space, spaces.Discrete(2),
                              PPOConfig(hidden_sizes=tuple(cfg.hidden_sizes), device="cpu"))
        self.opt = torch.optim.Adam(self.agent.policy.parameters(), lr=cfg.lr)
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.rng = np.random.default_rng(cfg.seed)
        n = int(backend.max_snapshots)
        # Slots: start states along the solution, a ring for rollout snapshots, the rest for the expert.
        self.start_slots = list(range(0, cfg.starts))
        self.rollout_slots = list(range(cfg.starts, cfg.starts + 32))
        self.simplify_slots = range(cfg.starts + 32, cfg.starts + 96)
        self.search_slots = (cfg.starts + 96, n - cfg.starts - 96)
        self.presses_removed = 0
        self.presses_moved = 0
        self.starts: list[tuple[int, object]] = []  # (slot, state) along the first solution
        # counters for the ledger
        self.attempts = 0  # policy episodes + expert search rollouts
        self.env_ticks = 0
        self.rounds = 0
        self.started = time.time()

    # -------------------------------------------------------------- data

    def canonical(self, ticks: np.ndarray, root_slot: int | None = None) -> np.ndarray:
        """Expert inputs as labels: unneeded presses dropped, presses centred in their windows."""
        if not self.cfg.simplify:
            return ticks
        ticks, removed = simplify_inputs(self.backend, ticks, slots=self.simplify_slots, root_slot=root_slot,
                                         level_id=self.level_id, start_pos=self.start_pos)
        ticks, moved = center_presses(self.backend, ticks, slots=self.simplify_slots, root_slot=root_slot,
                                      level_id=self.level_id, start_pos=self.start_pos,
                                      step=self.cfg.decision_ticks)
        self.presses_removed += removed
        self.presses_moved += moved
        return ticks

    def add_solution(self, ticks: np.ndarray, root_slot: int | None = None, start_pos: int | None = None) -> int:
        """Record (obs, button) along ``ticks`` from RESET (or from ``root_slot``)."""
        dt = self.cfg.decision_ticks
        if root_slot is None:
            state = self.backend.reset(self.level_id, self.start_pos if start_pos is None else start_pos)
        else:
            state = self.backend.load_state(root_slot)
        self.encoder.reset(state)
        obs, act = [], []
        for i in range(0, len(ticks), dt):
            a = bool(ticks[i])
            obs.append(self.encoder.encode(state))
            act.append(a)
            state = self.backend.step(a, dt)
            self.env_ticks += dt
            if state.terminal:
                break
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.act.append(np.asarray(act, dtype=np.int64))
        return len(obs)

    # ------------------------------------------------------------- train

    def train(self, epochs: int | None = None) -> float:
        X = torch.as_tensor(np.concatenate(self.obs))
        Y = torch.as_tensor(np.concatenate(self.act))
        n = len(X)
        loss_fn = nn.CrossEntropyLoss()
        acc = 0.0
        for _ in range(epochs or self.cfg.epochs_per_round):
            perm = torch.randperm(n)
            for s in range(0, n, self.cfg.batch_size):
                mb = perm[s : s + self.cfg.batch_size]
                loss = loss_fn(self.agent.policy(X[mb]), Y[mb])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            with torch.no_grad():
                acc = float((self.agent.policy(X).argmax(-1) == Y).float().mean())
            if acc >= self.cfg.target_accuracy:
                break
        return acc

    # ------------------------------------------------------------ rollout

    def save_starts(self, ticks: np.ndarray) -> None:
        """Replay a solution from RESET and save ``starts`` states evenly spread along it."""
        state = self.backend.reset(self.level_id, self.start_pos)
        marks = set(np.linspace(0, max(0, len(ticks) - 240), self.cfg.starts).astype(int).tolist())
        self.starts = []
        for t, a in enumerate(ticks):
            if t in marks and len(self.starts) < len(self.start_slots):
                slot = self.start_slots[len(self.starts)]
                self.backend.save_state(slot)
                self.starts.append((slot, state))
            state = self.backend.step(bool(a), 1)
            self.env_ticks += 1
            if state.terminal:
                break

    def rollout(self, snapshots: bool = True, root_slot: int | None = None, stop_x: float | None = None):
        """Greedy policy from RESET (or a saved start), until it dies, completes or passes
        ``stop_x``. Returns (final state, [(slot, decision, state)] snapshots, decisions)."""
        dt = self.cfg.decision_ticks
        if root_slot is None:
            state = self.backend.reset(self.level_id, self.start_pos)
        else:
            state = self.backend.load_state(root_slot)
        self.encoder.reset(state)
        snaps: list[tuple[int, int, object]] = []
        ring = 0
        i = 0
        for i in range(self.cfg.max_decisions):
            if snapshots and i % self.cfg.snapshot_every == 0:
                slot = self.rollout_slots[ring % len(self.rollout_slots)]
                ring += 1
                self.backend.save_state(slot)
                snaps.append((slot, i, state))
                snaps = snaps[-len(self.rollout_slots):]
            obs = self.encoder.encode(state)
            a = int(self.agent.act(obs[None], deterministic=True).action[0])
            state = self.backend.step(bool(a), dt)
            self.env_ticks += dt
            if state.terminal or (stop_x is not None and state.player_x >= stop_x):
                break
        self.attempts += 1
        if self.log is not None:
            self.log.episode(step=self.env_ticks, level=self.level_name, segment=root_slot,
                             start_x=None if root_slot is None else float(self.starts[root_slot][1].player_x),
                             progress=state.progress, completed=state.completed,
                             death_x=state.player_x if state.dead else None, length=state.tick,
                             agent="distill-policy", attempt=self.attempts, round=self.rounds)
        return state, snaps, i + 1

    def expert_fix(self, snaps, death_state) -> int:
        """Search from a snapshot before the death to past it; add those labels. If the
        state there is already doomed, try once more from three times further back."""
        goal = min(death_state.player_x + self.cfg.goal_ahead, death_state.level_length)
        for back in (self.cfg.back_decisions, 3 * self.cfg.back_decisions):
            target = death_state.tick - back * self.cfg.decision_ticks
            candidates = [s for s in snaps if s[2].tick <= target] or snaps[:1]
            slot, _, start_state = candidates[-1]
            scfg = SearchConfig(**{**asdict(self.cfg.search), "seed": int(self.rng.integers(1 << 30))})
            search = SegmentSearch(self.backend, scfg, level_id=self.level_id, start_pos=self.start_pos,
                                   print_fn=None, slot_range=self.search_slots,
                                   agent_name="distill-expert", level_name=self.level_name)
            res = search.run(root_slot=slot, goal_x=goal)
            self.attempts += res.rollouts
            self.env_ticks += res.env_ticks
            if res.solved:
                return self.add_solution(self.canonical(res.ticks, root_slot=slot), root_slot=slot)
            if candidates[-1] is snaps[0]:
                break
        self.print(f"  expert could not get past x={death_state.player_x:.0f}")
        return 0

    # --------------------------------------------------------------- loop

    def run(self, solutions: list[np.ndarray]) -> dict:
        if self.cfg.simplify:
            solutions = [self.canonical(t) for t in solutions]
            self.print(f"canonical labels: {self.presses_removed} unneeded presses removed, "
                       f"{self.presses_moved} presses centred")
        for ticks in solutions:
            self.add_solution(ticks)
        self.save_starts(solutions[0])
        acc = self.train(self.cfg.epochs_per_round * 4)
        self.print(f"BC on {sum(len(a) for a in self.act)} decisions: train accuracy {acc:.4f}")
        best = 0.0
        completed = False
        for r in range(self.cfg.max_dagger_rounds):
            self.rounds = r + 1
            # The real test first: the greedy policy from RESET.
            state, snaps, _ = self.rollout()
            best = max(best, state.progress)
            passed = 0
            if not state.completed:
                self.expert_fix(snaps, state)
                # Then from each start along the solution to the next one: fix every death.
                for j, (slot, sstate) in enumerate(self.starts):
                    stop = (self.starts[j + 1][1].player_x + self.cfg.goal_ahead if j + 1 < len(self.starts)
                            else None)
                    end, snaps_j, _ = self.rollout(root_slot=slot, stop_x=stop)
                    if end.dead:
                        self.expert_fix(snaps_j, end)
                    else:
                        passed += 1
            if self.log is not None:
                self.log.generation(step=self.env_ticks, generation=self.rounds, segment=None,
                                    best_fitness=best, mean_fitness=passed / max(1, len(self.starts)),
                                    best_progress=state.progress, agent="distill-dagger",
                                    data=int(sum(len(a) for a in self.act)), train_accuracy=acc,
                                    attempts=self.attempts, starts_passed=passed)
            self.print(f"round {self.rounds:3d}: from RESET {state.progress:6.1%} "
                       f"({'COMPLETED' if state.completed else f'died at x={state.player_x:.0f}'}); "
                       f"starts passed {passed}/{len(self.starts)}; data {sum(len(a) for a in self.act)}, "
                       f"attempts {self.attempts}")
            if state.completed:
                completed = True
                break
            acc = self.train()
        return {"completed": completed, "best_progress": best, "rounds": self.rounds, "attempts": self.attempts,
                "env_ticks": self.env_ticks, "decisions": int(sum(len(a) for a in self.act)),
                "train_accuracy": acc, "presses_removed": self.presses_removed,
                "presses_moved": self.presses_moved, "wall_s": time.time() - self.started}

    def verify(self) -> tuple[bool, float]:
        """The greedy policy from RESET with no snapshots at all."""
        state, _, _ = self.rollout(snapshots=False)
        return bool(state.completed), state.progress
