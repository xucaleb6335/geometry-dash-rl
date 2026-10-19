"""Checkpoint search: beat a deterministic level by searching over inputs, one
segment at a time, from saved game states. (The TAS / bot approach.)

The level is cut into segments of ``segment_ticks`` physics ticks. A *beam*
is a set of game states at a segment boundary, each one saved in a snapshot
slot, together with the inputs that reach it. To get the next beam, every
node is restored and tried with candidate input sequences (its own surviving
plan first, then mutations of it, then random press patterns). A candidate
survives if it lives to the next boundary; it is then saved and played on for
``lookahead_ticks`` more, so that states which are about to die anyway (a ship
too low for the next gate) rank low. The survivors are de-duplicated by
their physical state (mode, gravity, y, vy, ...) and the best ``beam_width``
become the next beam.

When no candidate from a beam survives, the search backtracks: it rebuilds
the previous beam with a bigger budget (a fresh set of states), up to
``max_backtrack`` segments deep.

Inputs are decided every ``decision_ticks`` ticks (1 = frame-perfect at
240 TPS). A solution is a per-tick button array from RESET (or from a Start
Pos) to the level-complete tick. ``replay`` plays it back with no snapshots
at all, which is the final proof: the game's own completion flag.

Restores might not be exact on the real game (practice-mode checkpoints).
With ``verify_every = k`` the best node is replayed from the start every k
segments; if the replay disagrees with the searched state, the beam is
replaced by the replayed (true) state, and if the replay dies, the search
rolls back to the last verified beam.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from gdrl.env.protocol import GameState


@dataclass
class SearchConfig:
    segment_ticks: int = 60
    decision_ticks: int = 2
    fine_decision_ticks: int = 1  # used when a segment is being retried (frame-perfect parts)
    beam_width: int = 12
    max_found: int = 48  # distinct survivors kept per expansion (each holds a snapshot slot)
    candidates_per_node: int = 4  # per beam node per round
    max_rounds: int = 8  # rounds per expansion before giving up (times the retry multiplier)
    min_rounds: int = 1
    lookahead_ticks: int = 150  # longer than a cube jump (~103 ticks), so doomed jumps rank low
    toggle_probs: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.35)
    plan_mutation_window: int = 24  # ticks
    max_backtrack: int = 8  # segments
    retries_per_depth: int = 2  # rebuild attempts of one beam before backing up one more segment
    verify_every: int = 0  # segments; 0 = never replay from the start during search
    quant_y: float = 1.0  # state de-duplication grid
    quant_vy: float = 15.0
    max_segments: int = 100_000
    time_limit_s: float = 1e9
    max_rollouts: int = 10**9
    seed: int = 0


@dataclass
class Node:
    """A saved game state at a segment boundary."""

    slot: int | None
    parent: Node | None
    actions: np.ndarray  # per-tick buttons (bool) from the parent's state to this one
    state: GameState
    segment: int
    plan: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # surviving lookahead, per tick
    score: float = 0.0  # lookahead ticks survived

    def ticks(self) -> np.ndarray:
        """All per-tick buttons from the root to this node."""
        parts = []
        node: Node | None = self
        while node is not None:
            parts.append(node.actions)
            node = node.parent
        return np.concatenate(parts[::-1]) if parts else np.zeros(0, dtype=bool)


@dataclass
class SearchResult:
    solved: bool
    ticks: np.ndarray  # per-tick button (bool) from the start to completion (or the best node)
    best_progress: float
    segments: int
    rollouts: int
    deaths: int
    env_ticks: int
    loads: int
    saves: int
    backtracks: int
    resyncs: int
    wall_s: float
    start_pos: int = -1

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("ticks")
        d["solution_ticks"] = int(len(self.ticks))
        return d


def runs(ticks: np.ndarray) -> list[tuple[bool, int]]:
    """Run-length encode a per-tick button array: [(button, n_ticks), ...]."""
    out: list[tuple[bool, int]] = []
    for b in np.asarray(ticks, dtype=bool):
        if out and out[-1][0] == b:
            out[-1] = (bool(b), out[-1][1] + 1)
        else:
            out.append((bool(b), 1))
    return out


def play(backend, ticks: np.ndarray, max_step: int = 60_000) -> GameState | None:
    """Play a per-tick button array from the current state, stopping at a terminal state."""
    state = None
    for button, n in runs(ticks):
        while n > 0:
            k = min(n, max_step)
            state = backend.step(button, k)
            n -= k
            if state.terminal:
                return state
    return state


def state_diff(a: GameState, b: GameState) -> str:
    """First field where two states differ (for logs)."""
    for f in ("tick", "flags", "player_x", "player_y", "vel_y", "rotation", "x_speed", "mode"):
        if getattr(a, f) != getattr(b, f):
            return f"{f}: {getattr(a, f)!r} vs {getattr(b, f)!r}"
    if len(a.objects) != len(b.objects):
        return f"n_objects: {len(a.objects)} vs {len(b.objects)}"
    for i, (oa, ob) in enumerate(zip(a.objects, b.objects)):
        if oa.tobytes() != ob.tobytes():
            return f"object {i}: {oa} vs {ob}"
    return "only non-physics fields"


def replay(backend, ticks: np.ndarray, level_id: int = 0, start_pos: int = -1) -> GameState:
    """RESET, then play ``ticks`` with no snapshots involved. The final proof."""
    state = backend.reset(level_id, start_pos)
    out = play(backend, ticks)
    return out if out is not None else state


class SlotPool:
    def __init__(self, n: int, base: int = 0):
        self.free = list(range(base + n - 1, base - 1, -1))
        self.n = n

    def get(self) -> int:
        if not self.free:
            raise RuntimeError(f"out of snapshot slots ({self.n}); lower beam_width or max_backtrack")
        return self.free.pop()

    def put(self, slot: int | None) -> None:
        if slot is not None:
            self.free.append(slot)


class SegmentSearch:
    """Run the search over a backend that supports snapshots (``save_state``/``load_state``)."""

    def __init__(self, backend, cfg: SearchConfig, *, level_id: int = 0, start_pos: int = -1,
                 log=None, level_name: str = "", agent_name: str = "checkpoint-search",
                 print_fn: Callable[[str], None] | None = print, proposal: Callable | None = None,
                 slot_range: tuple[int, int] | None = None):
        if int(getattr(backend, "max_snapshots", 0)) <= 0:
            raise ValueError("the backend has no snapshots (protocol v2 SAVE_STATE/LOAD_STATE needed)")
        self.backend = backend
        self.cfg = cfg
        self.level_id = level_id
        self.start_pos = start_pos
        self.log = log
        self.level_name = level_name
        self.agent_name = agent_name
        self.print = print_fn or (lambda s: None)
        self.proposal = proposal  # optional: fn(state, n_decisions, rng) -> bool array (e.g. a policy)
        self.rng = np.random.default_rng(cfg.seed)
        # (first slot, count); default: all of the backend's slots
        base, count = slot_range if slot_range is not None else (0, int(backend.max_snapshots))
        self.slots = SlotPool(count, base)
        self.seg_ticks = cfg.segment_ticks
        self.look_ticks = cfg.lookahead_ticks
        for dt in (cfg.decision_ticks, cfg.fine_decision_ticks):
            if cfg.segment_ticks % dt:
                raise ValueError("segment_ticks must be a multiple of decision_ticks and fine_decision_ticks")
        # counters (exact, for the ledger)
        self.rollouts = self.deaths = self.env_ticks = self.loads = self.saves = 0
        self.backtracks = self.resyncs = self.generation = 0
        self.started = 0.0
        self._loaded: Node | None = None  # node whose state the game is in, untouched since load/save

    # ------------------------------------------------------------ helpers

    def _key(self, s: GameState) -> tuple:
        c = self.cfg
        return (int(s.mode), int(s.upside_down), int(s.on_ground), int(s.holding), round(s.x_speed),
                round(s.player_x / 2.0), round(s.player_y / c.quant_y), round(s.vel_y / c.quant_vy),
                bool(s.dual), round(s.p2_y / c.quant_y) if s.dual else 0)

    def _load(self, node: Node) -> None:
        if self._loaded is node:
            return
        self.backend.load_state(node.slot)
        self.loads += 1
        self._loaded = node

    def _play(self, ticks: np.ndarray) -> tuple[GameState | None, int]:
        """Play per-tick buttons from the current state. Returns (last state, ticks played)."""
        state = None
        done = 0
        start_tick = None
        for button, n in runs(ticks):
            state = self.backend.step(button, n)
            self._loaded = None
            if start_tick is None:
                start_tick = state.tick - n if not state.terminal else None
            if state.terminal:
                ran = n if start_tick is None else max(0, state.tick - start_tick - done)
                self.env_ticks += ran
                return state, done + ran
            self.env_ticks += n
            done += n
        return state, done

    def _random_sequence(self, n: int, start: bool) -> np.ndarray:
        """Random press pattern. Half the time a toggle process with a random
        rate; otherwise alternating runs whose first switch is uniform over the
        sequence (so every press time is equally likely) and whose lengths are
        drawn log-uniformly between 1 and ``n`` decisions."""
        if self.rng.random() < 0.5:
            p = float(self.rng.choice(self.cfg.toggle_probs))
            toggles = self.rng.random(n) < p
            return np.logical_xor(start, np.cumsum(toggles) % 2 == 1)
        out = np.empty(n, dtype=bool)
        value = bool(start)
        i = int(self.rng.integers(0, n + 1))
        out[:i] = value
        max_len = float(np.exp(self.rng.uniform(0.0, np.log(max(n, 2)))))
        while i < n:
            value = not value
            k = int(self.rng.integers(1, max(2, int(max_len) + 1)))
            out[i : i + k] = value
            i += k
        return out

    def _random_ticks(self, n_ticks: int, start: bool, dt: int) -> np.ndarray:
        n = math.ceil(n_ticks / dt)
        return np.repeat(self._random_sequence(n, start), dt)[:n_ticks]

    def _candidates(self, node: Node, count: int, round_idx: int, dt: int) -> list[np.ndarray]:
        """Per-tick candidate inputs for segment + lookahead, changing only every ``dt`` ticks
        (the surviving plan keeps its own granularity)."""
        L = self.seg_ticks + self.look_ticks
        out: list[np.ndarray] = []
        # A plan cut at a death tick can end mid-block; trim it so the random tail
        # stays on dt boundaries (otherwise solutions get mixed dt-tick blocks).
        plan = node.plan[: len(node.plan) // dt * dt]
        if len(plan):
            rest = max(0, L - len(plan))
            base = np.concatenate([plan, self._random_ticks(rest, bool(plan[-1]), dt)])[:L]
            if round_idx == 0:
                out.append(base)
            while len(out) < count // 2:
                child = base.copy()
                w = self.cfg.plan_mutation_window
                lo = int(self.rng.integers(0, L // dt)) * dt
                hi = min(L, lo + int(self.rng.integers(1, max(2, w // dt + 1))) * dt)
                flips = np.repeat(self.rng.random(math.ceil((hi - lo) / dt)) < 0.5, dt)[: hi - lo]
                child[lo:hi] ^= flips
                out.append(child)
        if self.proposal is not None and len(out) < count:
            out.append(np.asarray(self.proposal(node.state, L, self.rng), dtype=bool)[:L])
        while len(out) < count:
            out.append(self._random_ticks(L, node.state.holding, dt))
        return out

    def _log_rollout(self, node: Node, state: GameState | None, length: int) -> None:
        if self.log is None or state is None:
            return
        self.log.episode(step=self.env_ticks, level=self.level_name, segment=node.segment,
                         start_x=node.state.player_x, start_tick=node.state.tick, progress=state.progress,
                         completed=state.completed, death_x=state.player_x if state.dead else None,
                         length=length, agent=self.agent_name, attempt=self.rollouts)

    # -------------------------------------------------------------- core

    def _expand(self, beam: list[Node], multiplier: int) -> tuple[list[Node], Node | None, dict]:
        """One generation: from ``beam`` (at segment s) to the beam at s + 1."""
        cfg = self.cfg
        segment = beam[0].segment + 1
        found: dict[tuple, Node] = {}
        full = 0
        tried = 0
        progress_seen: list[float] = []
        best_reach = 0.0
        solution: Node | None = None
        target = cfg.beam_width
        # A retried segment gets finer inputs: some gaps need frame-perfect presses.
        dt = cfg.decision_ticks if multiplier == 1 else cfg.fine_decision_ticks
        for round_idx in range(cfg.max_rounds * multiplier):
            for node in beam:
                for seq in self._candidates(node, cfg.candidates_per_node, round_idx, dt):
                    self._load(node)
                    self.rollouts += 1
                    tried += 1
                    seg, look = seq[: self.seg_ticks], seq[self.seg_ticks:]
                    state, done = self._play(seg)
                    if state is None:
                        continue
                    if state.dead:
                        self.deaths += 1
                        progress_seen.append(state.progress)
                        best_reach = max(best_reach, state.progress)
                        self._log_rollout(node, state, done)
                        continue
                    if state.completed:
                        solution = Node(None, node, seg[:done].copy(), state, segment)
                        self._log_rollout(node, state, done)
                        return [], solution, {"tried": tried}
                    key = self._key(state)
                    if key in found and found[key].score >= self.look_ticks:
                        continue  # same state already reached with a full lookahead
                    slot = self.slots.get()
                    self.backend.save_state(slot)
                    self.saves += 1
                    child = Node(slot, node, seg.copy(), state, segment)
                    self._loaded = child
                    lstate, ldone = self._play(look)
                    end = lstate if lstate is not None else state
                    if end.completed:
                        # Completed inside the lookahead: that is a solution too.
                        solution = Node(None, child, look[:ldone].copy(), end, segment + 1)
                        self._log_rollout(node, end, self.seg_ticks + ldone)
                        return [], solution, {"tried": tried}
                    if end.dead:
                        self.deaths += 1
                    child.score = float(ldone) if end.dead else float(self.look_ticks)
                    child.plan = look[:ldone].copy() if end.dead else look.copy()
                    progress_seen.append(end.progress)
                    best_reach = max(best_reach, end.progress)
                    self._log_rollout(node, end, self.seg_ticks + ldone)
                    old = found.get(key)
                    if old is None and len(found) >= cfg.max_found:
                        # Full: the new state replaces the worst one, if it is better.
                        worst_key = min(found, key=lambda k_: found[k_].score)
                        if found[worst_key].score < child.score:
                            self.slots.put(found.pop(worst_key).slot)
                        else:
                            self.slots.put(slot)
                            continue
                    if old is None or child.score > old.score:
                        if old is not None:
                            self.slots.put(old.slot)
                        found[key] = child
                    else:
                        self.slots.put(slot)
                    full = sum(1 for n in found.values() if n.score >= self.look_ticks)
                    if self.rollouts >= cfg.max_rollouts:
                        break
            if round_idx + 1 >= cfg.min_rounds and full >= target:
                break
            if self.rollouts >= cfg.max_rollouts or time.time() - self.started > cfg.time_limit_s:
                break
        keep, drop = self._select(list(found.values()), target)
        for n in drop:
            self.slots.put(n.slot)
        stats = {"tried": tried, "survivors": len(found), "full": full,
                 "mean_progress": float(np.mean(progress_seen)) if progress_seen else 0.0,
                 "best_reach": best_reach}
        return keep, solution, stats

    def _select(self, nodes: list[Node], k: int) -> tuple[list[Node], list[Node]]:
        """Keep ``k`` nodes: the best score first, and among nodes with the best
        score, the most different states (farthest-point in y, vy, ground,
        holding), so one kind of state (e.g. all mid-jump) cannot fill the beam."""
        if len(nodes) <= k:
            return sorted(nodes, key=lambda n: -n.score), []
        nodes = sorted(nodes, key=lambda n: (n.score, self.rng.random()), reverse=True)
        top = [n for n in nodes if n.score >= nodes[0].score]
        rest = [n for n in nodes if n.score < nodes[0].score]
        if len(top) > k:
            feats = np.array([[n.state.player_y / 30.0, n.state.vel_y / 300.0, 2.0 * n.state.on_ground,
                               1.0 * n.state.holding, 3.0 * int(n.state.mode), 3.0 * n.state.upside_down]
                              for n in top])
            chosen = [0]
            dist = np.linalg.norm(feats - feats[0], axis=1)
            while len(chosen) < k:
                i = int(np.argmax(dist))
                chosen.append(i)
                dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
            keep = [top[i] for i in chosen]
            drop = [n for i, n in enumerate(top) if i not in set(chosen)] + rest
        else:
            keep = top + rest[: k - len(top)]
            drop = rest[k - len(top):]
        return keep, drop

    def _verify(self, node: Node) -> tuple[str, GameState]:
        """Replay ``node`` from the start. Returns ('ok' | 'resync' | 'dead', true state)."""
        ticks = node.ticks()
        true = replay(self.backend, ticks, self.level_id, self.start_pos)
        self.env_ticks += len(ticks)
        self._loaded = None
        if true.dead:
            return "dead", true
        if true.physics_bytes() == node.state.physics_bytes():
            return "ok", true
        if node.slot is not None:  # diagnose: is the slot itself still what was saved?
            again = self.backend.load_state(node.slot)
            self.loads += 1
            self.print(f"  slot {node.slot} reload vs node: {state_diff(again, node.state)}; "
                       f"chain depth {len(ticks)} ticks")
            true = replay(self.backend, ticks, self.level_id, self.start_pos)  # back to the true state
            self.env_ticks += len(ticks)
            self._loaded = None
        return "resync", true

    def run(self, root_slot: int | None = None, goal_x: float | None = None,
            prefix: np.ndarray | None = None, progress_fn: Callable[[np.ndarray], None] | None = None,
            progress_every: int = 20) -> SearchResult:
        """Search from RESET (or from the snapshot in ``root_slot``, which is left
        untouched) until the level is completed (or a beam node passes ``goal_x``).
        The returned ticks start at the root."""
        cfg = self.cfg
        self.started = time.time()
        prefix = np.zeros(0, dtype=bool) if prefix is None else np.asarray(prefix, dtype=bool)
        if root_slot is None:
            state = self.backend.reset(self.level_id, self.start_pos)
            if len(prefix):
                # Resume: replay a known-good prefix (e.g. a crashed run's best), search on from there.
                state = play(self.backend, prefix) or state
                self.env_ticks += len(prefix)
                if state.terminal:
                    raise ValueError(f"the resume prefix ends dead or completed at tick {state.tick}")
        else:
            state = self.backend.load_state(root_slot)
            self.loads += 1
            cfg = SearchConfig(**{**asdict(cfg), "verify_every": 0})  # no replay from RESET for a mid-level root
            self.cfg = cfg
        root = Node(self.slots.get(), None, np.zeros(0, dtype=bool), state, 0)
        self.backend.save_state(root.slot)
        self.saves += 1
        self._loaded = root
        beams: list[list[Node]] = [[root]]
        retries: dict[int, int] = {}
        verified_depth = 0
        solution: Node | None = None
        while True:
            depth = len(beams) - 1
            mult = 1 + retries.get(depth, 0)
            beam, solution, stats = self._expand(beams[-1], mult)
            self.generation += 1
            best = beam[0] if beam else beams[-1][0]
            if self.log is not None:
                self.log.generation(step=self.env_ticks, generation=self.generation,
                                    segment=beams[-1][0].segment + 1,
                                    best_fitness=stats.get("best_reach", best.state.progress),
                                    mean_fitness=stats.get("mean_progress", 0.0),
                                    best_progress=best.state.progress, agent=self.agent_name,
                                    rollouts=self.rollouts, candidates=stats.get("tried", 0),
                                    survivors=stats.get("survivors", 0), full=stats.get("full", 0),
                                    beam=len(beam), retry=mult - 1, backtracks=self.backtracks,
                                    x=best.state.player_x, tick=best.state.tick)
            if solution is not None:
                break
            if beam:
                beams.append(beam)
                retries.pop(depth + 1, None)
                if len(beams) > cfg.max_backtrack + 2:
                    for n in beams[0]:
                        self.slots.put(n.slot)
                        n.slot = None
                    beams.pop(0)
                    verified_depth = max(0, verified_depth - 1)
                seg_idx = beam[0].segment
                if progress_fn is not None and seg_idx % progress_every == 0:
                    progress_fn(np.concatenate([prefix, beam[0].ticks()]))
                if seg_idx % 10 == 0 or seg_idx < 3:
                    self.print(f"segment {seg_idx:4d}  x {beam[0].state.player_x:9.1f}  "
                               f"progress {beam[0].state.progress:6.1%}  beam {len(beam):3d}  "
                               f"rollouts {self.rollouts:7d}  backtracks {self.backtracks}")
                if cfg.verify_every and seg_idx % cfg.verify_every == 0:
                    verdict, true = self._verify(beam[0])
                    if verdict == "resync":
                        self.resyncs += 1
                        node = beam[0]
                        self.print(f"segment {seg_idx}: replay differs: {state_diff(true, node.state)}")
                        for n in beam[1:]:
                            self.slots.put(n.slot)
                        self.backend.save_state(node.slot)  # the replayed (true) state
                        self.saves += 1
                        node.state, node.plan = true, np.zeros(0, dtype=bool)
                        beams[-1] = [node]
                        self._loaded = node
                        self.print(f"segment {seg_idx}: restore was inexact; beam resynced to the replayed state")
                    elif verdict == "dead":
                        self.resyncs += 1
                        keep = max(1, verified_depth + 1)
                        self.print(f"segment {seg_idx}: the replay died at x={true.player_x:.1f}; "
                                   f"rolling back {len(beams) - keep} segments")
                        while len(beams) > keep:
                            for n in beams.pop():
                                self.slots.put(n.slot)
                        continue
                    verified_depth = len(beams) - 1
                if goal_x is not None and beam[0].state.player_x >= goal_x:
                    solution = beam[0]
                    break
                if beam[0].segment >= cfg.max_segments:
                    break
            else:
                retries[depth] = retries.get(depth, 0) + 1
                if retries[depth] > cfg.retries_per_depth and len(beams) > 1:
                    self.backtracks += 1
                    retries.pop(depth, None)
                    for n in beams.pop():
                        self.slots.put(n.slot)
                    retries[len(beams) - 1] = retries.get(len(beams) - 1, 0) + 1
                    self.print(f"backtrack to segment {beams[-1][0].segment} "
                               f"(stuck at x={beams[-1][0].state.player_x:.1f})")
                elif retries[depth] > cfg.retries_per_depth * 4:
                    self.print("search stuck: giving up")
                    break
            if self.rollouts >= cfg.max_rollouts or time.time() - self.started > cfg.time_limit_s:
                self.print("budget exhausted")
                break

        end = solution if solution is not None else beams[-1][0]
        ticks = np.concatenate([prefix, end.ticks()])
        for beam in beams:
            for n in beam:
                self.slots.put(n.slot)
                n.slot = None
        return SearchResult(
            solved=solution is not None, ticks=ticks.astype(bool), best_progress=end.state.progress,
            segments=end.segment, rollouts=self.rollouts, deaths=self.deaths, env_ticks=self.env_ticks,
            loads=self.loads, saves=self.saves, backtracks=self.backtracks, resyncs=self.resyncs,
            wall_s=time.time() - self.started, start_pos=self.start_pos)


def simplify_inputs(backend, ticks: np.ndarray, *, slots: range | list[int], root_slot: int | None = None,
                    level_id: int = 0, start_pos: int = -1, horizon: int = 480) -> tuple[np.ndarray, int]:
    """Drop every press that is not needed: for each run of held ticks, in
    order, release it if the inputs still survive ``horizon`` ticks past it.
    If a later part then dies anyway, the most recent removal is undone (and
    kept), until the inputs survive again.

    Search solutions press at random on open stretches, which gives a policy
    trained on them contradictory labels (same situation, different buttons).
    The simplified inputs press only where it matters. The result survives to
    the end of ``ticks`` (or completes), else ``ticks`` is returned unchanged.
    ``slots`` are snapshot slots it may use. Returns (inputs, presses removed).
    """
    ticks = np.asarray(ticks, dtype=bool)
    slots = list(slots)

    def start():
        return backend.reset(level_id, start_pos) if root_slot is None else backend.load_state(root_slot)

    out = ticks.copy()
    n = len(out)
    accepted: list[tuple[int, int, int]] = []  # (run start, run end, slot holding the state at run start)
    keep: set[int] = set()
    owner: dict[int, int] = {}  # slot -> run start whose state it holds
    ring = 0
    def undo() -> int | None:
        """Undo the latest removal (and keep that press). Returns where to resume."""
        if not accepted:
            return None
        ri, rj, rslot = accepted.pop()
        out[ri:rj] = True
        keep.add(ri)
        if owner.get(rslot) == ri:
            backend.load_state(rslot)
        else:  # its slot was reused: rebuild the state from the start
            start()
            play(backend, out[:ri])
        return ri

    start()
    t = 0
    guard = 0
    while t < n and guard < 100 * n:
        guard += 1
        rest = np.nonzero(out[t:])[0]
        if not len(rest):
            break
        i = t + int(rest[0])
        s = play(backend, out[t:i]) if i > t else None
        if s is not None and s.dead:
            t = undo()
            if t is None:
                return ticks, 0
            continue
        if s is not None and s.completed:
            break
        j = i + 1
        while j < n and out[j]:
            j += 1
        slot = slots[ring % len(slots)]
        ring += 1
        backend.save_state(slot)
        owner[slot] = i
        if i not in keep:
            trial = out.copy()
            trial[i:j] = False
            end = play(backend, trial[i:min(n, j + horizon)])
            if end is None or not end.dead:
                out = trial
                accepted.append((i, j, slot))
            backend.load_state(slot)
        s = play(backend, out[i:j])
        if s is not None and s.dead:
            t = undo()
            if t is None:
                return ticks, 0
            continue
        t = j
    start()
    end = play(backend, out)
    if end is not None and end.dead:
        return ticks, 0
    return out, len(accepted)


def center_presses(backend, ticks: np.ndarray, *, slots: range | list[int], root_slot: int | None = None,
                   level_id: int = 0, start_pos: int = -1, horizon: int = 240, step: int = 2,
                   max_shift: int = 40) -> tuple[np.ndarray, int]:
    """Move every press run to the middle of the window of start times that
    lead to the *same* game state ``horizon`` ticks after the run as the
    original press does (shifts in ``step`` ticks, at most ``max_shift`` each
    way, never merging with the previous run).

    A jump works anywhere in a window, but a search solution presses at an
    arbitrary point in it, so the press/release boundary a policy should
    learn is arbitrary too. Centred presses follow a consistent rule and
    leave margin on both sides. Because a move must reproduce the original
    state afterwards (a cube that has landed again), it cannot change
    anything later, so the inputs stay a solution. Returns (inputs, runs moved).
    """
    ticks = np.asarray(ticks, dtype=bool)
    slot = list(slots)[0]

    def start():
        return backend.reset(level_id, start_pos) if root_slot is None else backend.load_state(root_slot)

    def key(st):
        return None if st is None else (st.dead, st.completed, round(st.player_y, 4), round(st.vel_y, 3),
                                        st.on_ground, int(st.mode), st.upside_down, round(st.x_speed, 3),
                                        st.holding, st.tick)

    out = ticks.copy()
    n = len(out)
    moved = 0
    start()
    t = 0
    while t < n:
        rest = np.nonzero(out[t:])[0]
        if not len(rest):
            break
        i = t + int(rest[0])
        j = i
        while j < n and out[j]:
            j += 1
        length = j - i
        end_t = min(n, j + max_shift + horizon)
        backend.save_state(slot)  # the state at t
        ref = key(play(backend, out[t:end_t]))

        def same(a: int) -> bool:
            if a < t + step or a + length > n:
                return False
            tr = out.copy()
            tr[i:j] = False
            tr[a : a + length] = True
            if a + length < n and tr[a + length]:
                return False  # would merge with the next run
            backend.load_state(slot)
            return key(play(backend, tr[t:end_t])) == ref

        lo = i
        while i - (lo - step) <= max_shift and same(lo - step):
            lo -= step
        hi = i
        while (hi + step) - i <= max_shift and same(hi + step):
            hi += step
        mid = i + ((lo + hi) // 2 - i) // step * step
        if mid != i:
            out[i:j] = False
            out[mid : mid + length] = True
            moved += 1
            j = mid + length
        backend.load_state(slot)
        st = play(backend, out[t:j])
        if st is not None and st.terminal:
            break
        t = j
    start()
    end = play(backend, out)
    if end is not None and end.dead:
        return ticks, 0
    return out, moved
