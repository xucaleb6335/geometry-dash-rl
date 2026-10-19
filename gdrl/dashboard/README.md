# GD-RL dashboard

Local training dashboard: read-only, except the Play buttons, which run `python -m gdrl.tools.watch` (a policy from `configs/watch.yaml`, played in GD at 1x; POST `/api/watch/start`, `/api/watch/stop`, status at `/api/watch`, output in `logs/watch.log`). Real-game levels only; `--mock` adds the mock levels, mock runs and the mock agent comparison. The sidebar's Network entry opens the network visualizer at `/viz` (`gdrl/visualizer`; traces from `results/viz/*.trace.json.gz` via `/api/traces` and `/api/trace?name=`, recorded with `python -m gdrl.tools.nn_trace`). No installs, no network: stdlib server plus a vanilla JS app (no libraries).

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
..\geometry-dash-rl\.venv\Scripts\python.exe -m gdrl.dashboard                 # http://127.0.0.1:8050
..\geometry-dash-rl\.venv\Scripts\python.exe -m gdrl.dashboard --port 8060 --runs runs
..\geometry-dash-rl\.venv\Scripts\python.exe -m gdrl.dashboard --export out.html   # single-file snapshot
..\geometry-dash-rl\.venv\Scripts\python.exe -m gdrl.dashboard --demo          # adds fake extreme-demon runs
```

Run it from the project root. The page polls `/api/state` every 4 s (only re-renders when something changed).

## What it reads

| Source | Used for |
|---|---|
| `runs/**/metrics.csv` | curves (read incrementally by byte offset) |
| `runs/**/{config.yaml,meta.json,result.json,eval.json,checkpoints/}` | level, agent, status |
| `runs/**/{episodes,generations}.jsonl` (`gdrl/runlog.py`) | death map, search charts |
| `env.curriculum.{starts,log_path}` of a run's config | curriculum tiles, death map |
| `holistic_progress/level_beats.csv` | beat ledger, "beaten" status and beat time |
| `results/**/*.json` | agent comparison (files without an `agent` key are skipped) |
| `logs/*.log` | running/finished/crashed hints (UTF-8, UTF-16 or mixed) |
| `results/viz/*.trace.json.gz` | network visualizer traces (`/viz`) |

Status: a run is *running* while its files changed in the last 3 minutes; *finished* with `final.pt`, `result.json`,
`eval.json` or all steps done; *crashed* when a later run resumed from it or its log shows an error; else *stopped*.
Level status: *beaten* needs a ledger row for that level; *robust-beaten* adds a sticky-action completion >= 0.8 in
`results/` or "robust" in the ledger note. Best progress counts greedy/from-start values (curriculum runs use start-0 episodes).

Wall time on the x-axis is estimated from `perf/steps_per_second` (metrics.csv has no timestamps).

## Files

- `levels_catalog.json`: difficulty metadata and aliases (edit to add levels; unknown levels are added automatically). `run_level_overrides` maps run names without level info to a level.
- `aggregate.py` (all logic, tested in `tests/test_dashboard.py`), `server.py`, `__main__.py`, `static/` (app).
- `demo/`: `make_demo.py` regenerates synthetic runs in `demo/runs/` (only loaded with `--demo`).
