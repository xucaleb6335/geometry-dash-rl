"""Checkpoint search, snapshots, distillation and run logs on the mock game."""

import json

import numpy as np
import pytest

from gdrl.env.backends import MockBackend
from gdrl.env.mock_game import MockGame
from gdrl.env.mock_demon import generate_demon_level, max_spikes
from gdrl.runlog import RunLog
from gdrl.search.segment_search import SearchConfig, SegmentSearch, replay, runs


def test_runs_encoding():
    assert runs(np.array([0, 0, 1, 1, 1, 0], dtype=bool)) == [(False, 2), (True, 3), (False, 1)]


def test_snapshot_restore_is_exact_on_demon_level():
    game = MockGame(generate_demon_level(3, n_sections=5))
    game.reset()
    rng = np.random.default_rng(0)
    for a in rng.integers(2, size=50):
        game.run(bool(a), 2)
    snap = game.snapshot()
    actions = rng.integers(2, size=200).astype(bool)
    first = [game.step(bool(a), 2).to_bytes(2) for a in actions]
    game.restore(snap)
    second = [game.step(bool(a), 2).to_bytes(2) for a in actions]
    assert first == second


def test_demon_levels_have_every_mode():
    level = generate_demon_level(2, n_sections=14)
    subtypes = set(level.objects["subtype"].tolist())
    assert {41, 44, 40}.issubset(subtypes)  # ship, wave and cube portals
    assert (level.motion[:, 0] != 0).any()  # something moves
    assert max_spikes(251.16) == 2 and max_spikes(311.58) == 3


def test_search_beats_a_mixed_mode_level_and_the_replay_completes(tmp_path):
    backend = MockBackend(level_seed=11, kind="demon", n_sections=3, section_types=("cube", "ship", "wave"))
    log = RunLog(tmp_path, {"level": "t"})
    result = SegmentSearch(backend, SearchConfig(seed=0, max_rollouts=60_000), log=log, print_fn=None).run()
    log.close()
    assert result.solved
    state = replay(backend, result.ticks)  # RESET + the inputs, no snapshots
    assert state.completed and not state.dead
    gens = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    eps = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert gens and len(eps) == result.rollouts or len(eps) <= result.rollouts
    assert {"t", "step", "generation", "segment", "best_fitness", "mean_fitness", "best_progress"} <= set(gens[0])


def test_search_survives_inexact_restores_with_verification():
    # Every restore lands 0.3 units off; verification replays from RESET and resyncs.
    backend = MockBackend(level_seed=4, kind="demon", n_sections=2, section_types=("cube",), restore_error=0.3)
    cfg = SearchConfig(seed=1, verify_every=5, max_rollouts=60_000)
    search = SegmentSearch(backend, cfg, print_fn=None)
    result = search.run()
    assert result.solved
    assert result.resyncs > 0
    assert replay(backend, result.ticks).completed


def test_search_from_a_snapshot_to_a_goal():
    backend = MockBackend(level_seed=5, kind="demon", n_sections=3)
    backend.reset()
    backend.step(False, 30)
    start = backend.save_state(0)
    search = SegmentSearch(backend, SearchConfig(seed=0, max_rollouts=20_000), print_fn=None, slot_range=(1, 500))
    result = search.run(root_slot=0, goal_x=start.player_x + 600)
    assert result.solved
    backend.load_state(0)
    from gdrl.search.segment_search import play

    end = play(backend, result.ticks)
    assert not end.dead and end.player_x >= start.player_x + 600


@pytest.mark.slow
def test_distilled_policy_completes_short_level():
    from gdrl.features.demon_encoder import DemonEncoder
    from gdrl.search.distill import DistillConfig, Distiller

    backend = MockBackend(level_seed=11, kind="demon", n_sections=2, section_types=("cube",))
    res = SegmentSearch(backend, SearchConfig(seed=0, max_rollouts=40_000), print_fn=None).run()
    assert res.solved
    dist = Distiller(backend, DemonEncoder(), DistillConfig(hidden_sizes=(128, 128), max_dagger_rounds=30),
                     print_fn=lambda s: None)
    out = dist.run([res.ticks])
    assert out["completed"]
    assert dist.verify()[0]


def test_search_resumes_from_a_prefix():
    backend = MockBackend(level_seed=2, kind="demon", n_sections=3, section_types=("cube", "wave"))
    saved = []
    first = SegmentSearch(backend, SearchConfig(seed=0, max_segments=30), print_fn=None)
    first.run(progress_fn=saved.append, progress_every=10)
    assert saved
    prefix = saved[-1][:-240]
    result = SegmentSearch(backend, SearchConfig(seed=1, max_rollouts=60_000), print_fn=None).run(prefix=prefix)
    assert result.solved
    assert np.array_equal(result.ticks[: len(prefix)], prefix)
    assert replay(backend, result.ticks).completed


def test_simplify_drops_needless_presses_and_still_completes():
    from gdrl.search.segment_search import simplify_inputs

    backend = MockBackend(level_seed=11, kind="demon", n_sections=2, section_types=("cube",))
    res = SegmentSearch(backend, SearchConfig(seed=0, max_rollouts=40_000), print_fn=None).run()
    assert res.solved
    # A small ring of slots, so it wraps and undo has to rebuild states.
    simple, removed = simplify_inputs(backend, res.ticks, slots=range(0, 8))
    assert removed > 0 and simple.sum() < res.ticks.sum()
    assert replay(backend, simple).completed
