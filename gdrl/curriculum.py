"""Start-position curriculum: episodes start from several points along a level.

Why: a level is a long chain of commitments. On Stereo Madness the first
real-game run plateaued at 19.4% with every episode identical: the
near-deterministic policy (entropy 0.1) held the button through a landing at
x ~ 5050, auto-jumped into a slab, and never tried the ~40 releases in a row
that the low route needs. Exploring from the level start alone cannot find
that, and every later part of the level is never seen. Starting episodes from
points spread along the level lets each part be learned on its own, and the
value of the later parts flows back to the earlier decisions (a reverse
curriculum, in effect).

A start is a dict with one of:

- ``spec``: a gdctl level spec (``main:1``, ``local:gdrl-sm-s05``). The real
  bridge supports only ``start_pos -1``, so mid-level starts are local copies
  of the level, each with one StartPos (``gdrl.tools.make_starts`` builds
  them). Switching opens that level with ``gdctl open-level`` (~1.5 s), then
  waits until the new level's physics actually runs (a freshly opened level
  answers STEPs with 0 physics ticks during its intro).
- ``start_pos``: an index passed to the backend's RESET (the mock game).

and optionally ``x`` (where it starts, in game units; used to measure whether
an episode reached the next start) and ``name``.

Which start: start 0 (the real level start, which is what gets evaluated)
with probability ``p_first``; otherwise one of the open starts with weight
``max(min_weight, 1 - success)``, where ``success`` is a running average of
"the episode reached the next start's x (or completed)". Starts the agent
already passes get sampled less.

Frontier gating (``frontier_extra`` not None): only starts up to the first
unsolved start (success < ``solved``) beyond the level-start reach, plus
``frontier_extra`` more, are open. The reach is the furthest x of the last
``reach_window`` episodes from start 0. Without gating, training on every
part of Stereo Madness at once (18 starts, cube and ship) learned the level
start far slower than plain training (mean 3.3% after 210k level-start
steps, against 11%): each update was dominated by one or two unrelated
parts. Gating keeps the extra starts near where the agent is stuck. With ``spec`` starts the wrapper stays on a
start for at least ``switch_steps`` env steps, since switching costs seconds.

``reset(options={"start": i})`` forces start ``i`` (used by greedy evaluation).
Each episode is appended to ``log_path`` as CSV (start, start x, final x,
progress, length, completed), which is also the attempt count.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


class StartCurriculum(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        starts: list[dict[str, Any]],
        *,
        p_first: float = 0.25,
        min_weight: float = 0.1,
        ema: float = 0.1,
        switch_steps: int = 0,
        log_path: str | Path | None = None,
        gdctl: list[str] | None = None,
        warmup_timeout: float = 20.0,
        frontier_extra: int | None = None,
        solved: float = 0.5,
        reach_window: int = 30,
        seed: int = 0,
        spawn_tolerance: float = 5.0,
        horizon_starts: int | None = None,
        explore_eps: float = 0.0,
        explore_below: float = 0.05,
        snapshot_every: float | None = None,
        snapshot_slots: tuple[int, int] = (100, 250),
        snapshot_confirm: int = 60,
        snapshot_retire_after: int | None = None,
        demos: list[dict[str, Any]] | None = None,
    ):
        super().__init__(env)
        if not starts:
            raise ValueError("need at least one start")
        self.starts = [dict(s) for s in starts]
        self.p_first = p_first
        self.min_weight = min_weight
        self.ema = ema
        self.switch_steps = switch_steps
        self.gdctl = gdctl or [sys.executable, "-m", "gdctl"]
        self.warmup_timeout = warmup_timeout
        self.success = np.zeros(len(self.starts))
        self.frontier_extra = frontier_extra
        self.solved = solved
        self.reach_window = reach_window
        self.recent_reach: list[float] = []  # final x of recent start-0 episodes
        self.rng = np.random.default_rng(seed)
        self.spawn_tolerance = spawn_tolerance
        # Episodes from a mid-level start end (truncated, so the value of the
        # rest is bootstrapped) once they pass this many further starts. Why:
        # on Stereo Madness the hard starts died ~26 steps after spawning
        # while easy starts ran ~300 steps to the next wall, so in env steps
        # the hard spots got a tiny share of training. Start 0 is never cut.
        self.horizon_starts = horizon_starts
        # Episodes from a mid-level start the agent (almost) never passes get
        # epsilon-random actions (train.py _explore reads ``explore_eps``).
        # Why: on Stereo Madness six starts stayed at 0% for 700k steps; the
        # policy was near-deterministic there (p(hold) ~0.95-1.0 right into
        # the same spike), while random presses pass them within ~250 tries.
        self.explore_eps_cfg = explore_eps
        self.explore_below = explore_below
        self.explore_eps = 0.0
        # Snapshot starts (needs bridge v2 SAVE/LOAD): while playing, an
        # alive state is saved every ``snapshot_every`` units where no start
        # exists yet, and becomes a start once the run survived
        # ``snapshot_confirm`` more steps (else it is dropped: a doomed
        # mid-air state is no use). Why: Bloodbath's 14 StartPos are ~2,800
        # units apart, so "reached the next start" was 0% on every segment
        # after 1M steps and the curriculum had no signal. Snapshots are exact
        # game states the agent really reached, not placed StartPos objects.
        self.snapshot_every = snapshot_every
        self.free_slots = list(range(snapshot_slots[1] - 1, snapshot_slots[0] - 1, -1))
        self.snapshot_confirm = snapshot_confirm
        self.pending: tuple[int, float, int] | None = None  # (slot, x, ep_len at save)
        self.last_save_x = -1e9
        # A snapshot start never passed in this many attempts is retired (its
        # slot freed, never sampled again) and no new snapshot is made within
        # half a spacing of it. Why: on Bloodbath the save point just before
        # the UFO (x ~ 11,520) was 0/570; the owner judged it a bad save point
        # (a state that cannot be survived), while runs reaching the UFO from
        # earlier starts are fine to keep practising.
        self.snapshot_retire_after = snapshot_retire_after
        self.attempts = np.zeros(len(self.starts), dtype=np.int64)
        self.wins = np.zeros(len(self.starts), dtype=np.int64)
        self.no_snap_x: list[float] = []
        # Demonstrations: per-tick inputs (a .npy of bools) known to survive
        # from a StartPos, e.g. a search solution. Replayed once, saving a
        # snapshot start every ``every`` units from ``from_x``. Why: on
        # Bloodbath every run died 100 units into the UFO tunnel (30.0%) and
        # all practice starts there were states that could not be survived;
        # the checkpoint search found a way through from the ball portal in
        # 65 s, so its states are starts the tunnel can be learned from.
        self.demos = list(demos or [])
        self.demos_loaded = not self.demos
        self.current: int | None = None  # start whose level is loaded (spec starts)
        self.episode_start = 0
        self.steps_on_current = 0
        self.max_x = 0.0
        self.ep_len = 0
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                with self.log_path.open("w", newline="") as f:
                    csv.writer(f).writerow(["time", "start", "start_x", "final_x", "progress", "length",
                                            "completed", "reached_next"])

    # -- choosing ---------------------------------------------------------

    def num_open(self) -> int:
        """How many starts (from the first) may be chosen."""
        n = len(self.starts)
        if self.frontier_extra is None:
            return n
        reach = max(self.recent_reach, default=0.0)
        for k in range(1, n):
            if float(self.starts[k].get("x", 0.0)) > reach and self.success[k] < self.solved:
                return min(n, k + 1 + self.frontier_extra)
        return n

    def weights(self) -> np.ndarray:
        """Probability of each start for the next choice."""
        n = len(self.starts)
        w = np.maximum(self.min_weight, 1.0 - self.success)
        w[self.num_open():] = 0.0
        w[[i for i, st in enumerate(self.starts) if st.get("retired")]] = 0.0
        p = (1.0 - self.p_first) * w / w.sum()
        p[0] += self.p_first
        return p / p.sum() if n > 1 else np.ones(1)

    def _choose(self) -> int:
        has_spec = any("spec" in s for s in self.starts)
        if has_spec and self.current is not None and self.steps_on_current < self.switch_steps:
            return self.current
        return int(self.rng.choice(len(self.starts), p=self.weights()))

    # -- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = dict(options or {})
        forced = options.pop("start", None)
        i = int(forced) if forced is not None else self._choose()
        start = self.starts[i]
        if "spec" in start and i != self.current:
            self._switch_to(i)
        if "start_pos" in start:
            options["start_pos"] = int(start["start_pos"])
        if not self.demos_loaded:
            self._load_demos()
            start = self.starts[i]
        if "slot" in start:
            options["load_slot"] = int(start["slot"])
        self._drop_pending()
        obs, info = self.env.reset(seed=seed, options=options or None)
        self.episode_start = i
        state = self.env.unwrapped.state
        self.max_x = float(state.player_x) if state is not None else 0.0
        if "start_pos" in start and "x" in start and state is not None                 and abs(state.player_x - float(start["x"])) > self.spawn_tolerance:
            # The game spawned somewhere else (e.g. GD reusing a cached Start Pos
            # checkpoint): training on it would silently learn the wrong part.
            raise RuntimeError(f"start {i} ({start.get('name')}): spawned at x={state.player_x:.1f}, "
                               f"expected {float(start['x']):.1f}")
        self.ep_len = 0
        self.last_save_x = self.max_x
        # Not on demo starts: they sit on a searched path through frame-tight
        # parts (Bloodbath's UFO tunnel), where random presses only kill, and
        # the policy was cloned onto that path already. Patience, not noise.
        self.explore_eps = (self.explore_eps_cfg if i > 0 and forced is None and not start.get("demo")
                            and self.success[i] < self.explore_below else 0.0)
        info["start"] = i
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_on_current += 1
        self.ep_len += 1
        state = self.env.unwrapped.state
        if state is not None:
            self.max_x = max(self.max_x, float(state.player_x))
        info["start"] = self.episode_start
        if self.snapshot_every and state is not None:
            self._snapshot_step(state, terminated or truncated)
        if not (terminated or truncated) and self.horizon_starts and self.episode_start > 0:
            far = self._ahead_x(self.episode_start, self.horizon_starts)
            if far is not None and self.max_x >= far:
                truncated = True
        if terminated or truncated:
            if self.pending is not None and not terminated:
                self._commit_pending()  # cut short alive: keep it
            self._drop_pending()
            self._end_episode(info)
        return obs, reward, terminated, truncated, info

    # -- internals --------------------------------------------------------

    def _ahead_x(self, i: int, n: int) -> float | None:
        """x of the n-th start ahead of start i (starts are not stored in x order)."""
        if "x" not in self.starts[i]:
            return None
        x0 = float(self.starts[i]["x"])
        ahead = sorted(float(s["x"]) for s in self.starts
                       if "x" in s and not s.get("retired") and float(s["x"]) > x0)
        return ahead[n - 1] if len(ahead) >= n else None

    def _next_x(self, i: int) -> float | None:
        return self._ahead_x(i, 1)

    def _snapshot_step(self, state, done: bool) -> None:
        if self.pending is not None:
            slot, x, at = self.pending
            if not done and self.ep_len - at >= self.snapshot_confirm:
                self._commit_pending()
            return
        if done or state.dead or state.completed or not self.free_slots:
            return
        x = float(state.player_x)
        if x < self.last_save_x + self.snapshot_every:
            return
        if any(abs(float(s.get("x", -1e9)) - x) < self.snapshot_every / 2 for s in self.starts):
            return
        if any(abs(z - x) < self.snapshot_every / 2 for z in self.no_snap_x):
            return
        slot = self.free_slots.pop()
        try:
            self.env.unwrapped.backend.save_state(slot)
        except Exception:
            self.free_slots.append(slot)
            self.snapshot_every = None  # backend without snapshots: turn the feature off
            return
        self.pending = (slot, x, self.ep_len)
        self.last_save_x = x

    def _load_demos(self) -> None:
        from gdrl.search.segment_search import runs

        self.demos_loaded = True
        backend = self.env.unwrapped.backend
        for d in self.demos:
            ticks = np.load(d["ticks"]).astype(bool)
            every, nxt = float(d.get("every", 100.0)), float(d.get("from_x", 0.0))
            # Save points only on decision boundaries (every `align` ticks from
            # the reset), so a policy acting every frame_skip ticks from a
            # save point is in phase with the demo's inputs.
            align = max(1, int(d.get("align", 1)))
            to_x = float(d.get("to_x", float("inf")))  # no save points past here (e.g. where the demo stops being valid)
            state = backend.reset(0, int(d.get("start_pos", -1)))
            t = 0
            for button, n in runs(ticks):
                for _ in range(n):  # one tick at a time, so no save point is skipped
                    state = backend.step(bool(button), 1)
                    t += 1
                    if state.dead or state.completed:
                        break
                    if nxt <= state.player_x <= to_x and t % align == 0 and self.free_slots:
                        slot = self.free_slots.pop()
                        backend.save_state(slot)
                        self.starts.append({"name": f"demo@{state.player_x:.0f}", "slot": slot,
                                            "x": float(state.player_x), "demo": True})
                        self.success = np.append(self.success, 0.0)
                        self.attempts = np.append(self.attempts, 0)
                        self.wins = np.append(self.wins, 0)
                        nxt = state.player_x + every
                if state.dead or state.completed:
                    break

    def _commit_pending(self) -> None:
        slot, x, _ = self.pending
        self.starts.append({"name": f"snap@{x:.0f}", "slot": slot, "x": x})
        self.success = np.append(self.success, 0.0)
        self.attempts = np.append(self.attempts, 0)
        self.wins = np.append(self.wins, 0)
        self.pending = None

    def _drop_pending(self) -> None:
        if self.pending is not None:
            self.free_slots.append(self.pending[0])
            self.pending = None

    def _end_episode(self, info: dict[str, Any]) -> None:
        i = self.episode_start
        completed = bool(info.get("completed", False))
        nxt = self._next_x(i)
        reached = completed or (nxt is not None and self.max_x >= nxt)
        self.success[i] += self.ema * (float(reached) - self.success[i])
        self.attempts[i] += 1
        self.wins[i] += int(reached)
        st = self.starts[i]
        if (self.snapshot_retire_after and "slot" in st and not st.get("retired") and not st.get("demo")
                and self.wins[i] == 0 and self.attempts[i] >= self.snapshot_retire_after):
            st["retired"] = True
            self.free_slots.append(int(st["slot"]))
            self.no_snap_x.append(float(st["x"]))
        if i == 0:
            self.recent_reach = (self.recent_reach + [self.max_x])[-self.reach_window:]
        if self.log_path:
            with self.log_path.open("a", newline="") as f:
                csv.writer(f).writerow([f"{time.time():.1f}", i, self.starts[i].get("x", ""),
                                        f"{self.max_x:.1f}", f"{float(info.get('progress', 0.0)):.4f}",
                                        self.ep_len, int(completed), int(reached)])

    def _switch_to(self, i: int) -> None:
        spec = self.starts[i]["spec"]
        backend = self.env.unwrapped.backend
        for attempt in range(3):
            r = subprocess.run([*self.gdctl, "open-level", spec], capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and self._wait_running(backend):
                break
            time.sleep(2.0)
        else:
            raise RuntimeError(f"could not switch to {spec}: {r.stdout[-300:]} {r.stderr[-300:]}")
        self.current = i
        self.steps_on_current = 0

    def _wait_running(self, backend) -> bool:
        """True once a STEP moves the player (or it dies): the level's physics runs."""
        deadline = time.time() + self.warmup_timeout
        while time.time() < deadline:
            s0 = backend.reset(0, -1)
            s1 = backend.step(False, 4)
            if s1.dead or s1.player_x != s0.player_x or s1.player_y != s0.player_y:
                return True
            time.sleep(0.2)
        return False


def wrap_curriculum(env: gym.Env, cfg: dict[str, Any] | None, seed: int = 0) -> gym.Env:
    """Wrap ``env`` if the config has a ``curriculum:`` section with ``starts``."""
    if not cfg or not cfg.get("starts"):
        return env
    cfg = dict(cfg)
    starts = cfg.pop("starts")
    if isinstance(starts, str):  # a YAML/JSON file written by gdrl.tools.make_starts
        import yaml

        starts = yaml.safe_load(Path(starts).read_text(encoding="utf-8"))["starts"]
    return StartCurriculum(env, starts, seed=seed, **cfg)
