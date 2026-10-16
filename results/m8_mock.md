# M8 on the mock game

Training levels 1-5, held-out 101-103 (levels.yaml). Every cell is the mean over the 5 training or 3 held-out
levels, 20 episodes per level per mode (100 + 60 episodes). "Greedy" = argmax policy, no noise;
"sticky" = `--sticky 0.1` (each step the previous action is repeated with p = 0.1). Mean ± std over seeds
(population std, 3 seeds). Compl. = completion rate, prog. = mean progress at episode end.

| Agent | Seeds | Train compl. greedy | Train prog. greedy | Train compl. sticky | Train prog. sticky | Held-out compl. greedy | Held-out prog. greedy | Held-out compl. sticky | Held-out prog. sticky | Env steps | Train wall-clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO object-list (MLP, `mock_ppo.yaml`) | 3 | 100 ± 0% | 100 ± 0% | 96 ± 3% | 97 ± 2% | 100 ± 0% | 100 ± 0% | 91 ± 5% | 95 ± 3% | 2.00M | 10 min |
| PPO grid + CNN, binary 1-block grid (`mock_ppo_grid.yaml`) | 3 | 100 ± 0% | 100 ± 0% | 86 ± 3% | 93 ± 2% | 89 ± 16% | 96 ± 5% | 48 ± 3% | 70 ± 3% | 2.00M | 35-57 min* |
| PPO grid + CNN, coverage fill + 3 cols/block (tried, reverted) | 3 | 33 ± 47% | 64 ± 25% | 32 ± 45% | 63 ± 25% | 33 ± 47% | 46 ± 38% | 28 ± 39% | 43 ± 34% | 2.00M | 87-99 min* |
| Rule agent | - | 100% | 100% | 40% | 65% | 100% | 100% | 32% | 59% | 0 | - |
| Random | - | 0% | 10% | 0% | 12% | 0% | 9% | 0% | 9% | 0 | - |
| Genetic (per level, below) | - | 100% (replay of own level) | 100% | 24% | 60% | 100% | 100% | 18% | 52% | see below | 5-54 s per level |

\* Wall-clock is noisy: the grid runs shared the CPU with each other, with a real-game training and with
evals (about 400 steps/s vs ~900 uncontended for the binary grid). The object-list runs ran alone, ~3400 steps/s.
Per-seed binary-grid held-out greedy: seeds 1,2 = 100%, seed 3 = 87% (level 102 failed, 0/20).
Coverage-grid: seed 1 = 100% everywhere (sticky 91%/83% train/held-out), seeds 2 and 3 never completed one episode (progress 20-78%).

## Genetic agent (open-loop press sequence, population 32, <= 100 generations, fitness >= 2 = completed)

| Level | Solved | Generations | Episodes (gen x 32) | Wall | Sticky 0.1 completion | Sticky progress | Progress of this sequence on the other levels |
|---|---|---|---|---|---|---|---|
| 1 | yes | 6 | 192 | 11 s | 25% | 63% | 12% |
| 2 | yes | 7 | 224 | 10 s | 35% | 57% | 9% |
| 3 | yes | 5 | 160 | 6 s | 20% | 59% | 11% |
| 4 | yes | 19 | 608 | 38 s | 25% | 61% | 13% |
| 5 | yes | 26 | 832 | 44 s | 25% | 60% | 13% |
| 101 | yes | 9 | 288 | 10 s | 20% | 42% | 13% |
| 102 | yes | 9 | 288 | 14 s | 10% | 64% | 13% |
| 103 | yes | 31 | 992 | 54 s | 25% | 50% | 13% |

The genetic agent evolves on each level it is scored on (including 101-103; it has no training/held-out split)
and the sequence does not transfer.

## Reproduce

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"; $py = "..\geometry-dash-rl\.venv\Scripts\python.exe"
# PPO object-list: runs/sweep/*rbf_s{1,2,3} (configs/mock_ppo.yaml, identical settings)
& $py -m gdrl.train --config configs/mock_ppo.yaml --set seed=1
& $py -m gdrl.tools.m8_compare eval --agent ppo-object-list --run runs/<run>
# PPO grid + CNN (binary grid), seeds 1-3
& $py -m gdrl.train --config configs/mock_ppo_grid.yaml --set seed=1 --set train.runs_dir=runs/m8
& $py -m gdrl.tools.m8_compare eval --agent ppo-grid-binary --run runs/m8/<run>
# coverage variant: git apply results/gridencoder_coverage.patch, then add
#   fill: coverage, x_cells_per_block: 3 to env.encoder
# rule / random / genetic
& $py -m gdrl.tools.m8_compare eval --agent rule --config configs/mock_ppo.yaml --set "agent={name: rule}"
& $py -m gdrl.tools.m8_compare eval --agent random --config configs/mock_random.yaml
& $py -m gdrl.tools.m8_compare genetic --config configs/mock_ppo.yaml
# table (labels in results/m8/*.json: ppo-grid = coverage variant, ppo-grid-binary = default grid)
& $py -m gdrl.tools.m8_compare table --order ppo-object-list ppo-grid-binary ppo-grid rule random
```
Raw results: `results/m8/*.json`; runs: `runs/m8/*`, `runs/sweep/*rbf_s*`.

## Findings

- Object-list PPO is the best learner: all 3 seeds complete every training and held-out level greedily, 91% held-out
  under sticky actions, in 10 minutes. First completion in training came after ~2.5-3.7k episodes (0.37-0.68M steps).
- Grid + CNN with the plain binary one-block grid also learns (3/3 seeds, 100% train greedy; first completion after
  1.3-2.2k episodes, earlier than object-list) but generalizes less well: held-out greedy 89 ± 16% (one seed fails level 102)
  and held-out sticky only 48% vs 91%. It is 2-5x slower per step. Hypothesis, not tested: the grid hides the exact
  x-offset within a 30-unit cell, so timing is less precise and it is less robust to a flipped action.
- The previous agent's premise (binary grid cannot resolve the ~16-unit jump window) is not supported: the binary grid
  solved all levels on 3/3 seeds. The coverage-fill/3-column variant was worse (1/3 seeds learned at all; the failed seeds
  never completed a single training episode, so it is an exploration/optimization failure, not a resolution one).
  The GridEncoder changes were therefore reverted (`results/gridencoder_coverage.patch` keeps them).
- Where PPO fails: with `ent_coef` 0.003 (sweep) the object-list run failed completely; coverage-grid seeds 2-3 plateau at
  20-78% progress with no completion, so there is no reward signal beyond partial progress. Seeds matter more than
  architecture for the grid.
- Rule agent: perfect greedy, but fragile to sticky actions (40% train, 32% held-out): its press timing window is tight
  and it cannot recover from a delayed press. The PPO agents re-decide from the current state every step and lose much less.
- Genetic: solves any single level in 5-31 generations (<1000 episodes, <1 min) but is an open-loop memorized sequence:
  it fails under sticky actions (~22% over all 8 levels) and reaches only ~12% progress on other levels. Random: 0% completion, ~10% progress.
- Caveats: 20 episodes per level; the mock is deterministic, so greedy episodes within a level are identical (greedy
  completion rates are effectively per level); 3 seeds.
