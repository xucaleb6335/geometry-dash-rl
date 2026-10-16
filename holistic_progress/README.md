# Holistic progress

The long-running record of what the agents have beaten.

- `level_beats.md` / `level_beats.csv`: every level beat, with attempts, generations, env steps, wall time and the time it was beaten (Pacific).
- `progress_reports/` (repo root): the 30-minute check-in reports.

To record a beat (agents: do this the moment a level is beaten and verified):

```powershell
..\geometry-dash-rl\.venv\Scripts\python.exe -m gdrl.tools.record_beat --level "<name>" --level-id <id> --game real --difficulty "<difficulty>" --agent <agent> --attempts <n> --generations <n> --env-steps <n> --wall-hours <h> --run runs\<run> --notes "<how it was verified>"
```

"Attempts" = episodes (deaths + the winning run) from the start of training to the first beat. For segment/StartPos training, count all attempts across all segments, and say in the notes whether the beat was a full run from the level start.
