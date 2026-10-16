# DEMON_PLAN: beating extreme demons

Goal: the agent beats Bloodbath (online id 10565740), Cataclysm (3979721,
Ggb0y) and Rauchkammer (146399247, Devilmine), then the Pointercrate top 10
(as of 2026-10-11: GRIEF 149992717, Society 127323087, Thinking Space II
119544028, Amethyst 119550490, Flamewall 126242564, Tidal Wave 86407629,
Green Bullet 142896409, ORBIT 133175713, Antarctic Lights 144988390,
Nullscapes 109780665).

Written 2026-10-11 by the demon agent. Status: everything below marked
**built** works and is tested on the mock game; nothing has touched the real
game yet (it was in use by PPO training).

## 1. Strategy in one paragraph

An extreme demon under the lockstep bridge is a **deterministic** function
from a per-tick button sequence to an outcome. So the reliable way to beat it
is not to learn a policy that reacts well, but to **search** for one button
sequence that completes it: save the game state at segment boundaries,
try many input variations from each saved state, keep the best distinct
survivors, and move on (beam search with backtracking, the way TAS bots
work). Once found, the sequence is replayed from RESET with no snapshots,
and the game's own level-complete event is the proof. A policy is then
**distilled** from the search (behaviour cloning, then DAgger with the
search as the expert), which gives a network that plays the level greedily,
and a starting point for PPO with a segment curriculum. Pure PPO from
scratch stays a baseline: on demon-like mock levels the search solves
what PPO cannot reach.

### Level facts (offline read of a copy of CCLocalLevels.dat, 2026-10-11)

| Local copy | Objects | Length (x) | Start Pos objects | Random triggers (1912, 2068) | Gameplay |
|---|---|---|---|---|---|
| `gdrl-bloodbath` | 24,748 | 39,029 | **1** | none | cube, ship, ball, UFO, wave, dual, mini, 36/32 gravity portals, speeds 0.5x to 3x |
| `rauchkammer` | 8,252 | 14,295 | 2 | none | cube, ship, UFO, wave, mini, gravity, speeds 1x to 3x |
| Cataclysm | not downloaded yet (`gdctl download-level 3979721`) | | | | |

Both copies contain Start Pos objects, so with the patched mod a full-run
proof must use `start_pos: -2` (`--start-pos -2`), or the Start Pos objects
must be removed from the copy (`gdctl startpos`). No random triggers, so
both should be deterministic under lockstep (to confirm with bridge_check).

### Segment starts: community Start Pos copies (owner direction, 2026-10-11)

We do **not** place Start Pos objects ourselves (our estimates of mode,
speed and y at a point were unreliable). Segment starts come from community
"start pos" copies of each level, found online:

1. `gdctl search-levels "<name> start pos"`, then variants `"<name> startpos"`,
   `"<name> sp"`; names are capped at 20 characters, so expect truncation
   ("Bloodbath Startpo").
2. Verify a candidate against the original before use: object count close to
   the original's (Start Pos objects aside), same length (max x), and Start
   Pos modes/speeds/gravity consistent with the portals before each one
   (offline read of the level string, as in the table above). Reject copies
   that are buffed, nerfed or layouts.
3. Download and copy it locally (`gdctl download-level`, `copy-to-local`),
   then its Start Pos objects are the segment starts: RESET `start_pos = k`
   (patched mod) for the k-th by x.

How they are used:

- **Segment map**: run the search from each Start Pos to the next one
  (`demon_search --start-pos k` with a goal of the next Start Pos x) to find
  the hard segments early and in parallel with the full run, and to get
  per-segment success numbers ("Bloodbath seg 3/12 mastered" ledger rows).
- **Policy curriculum**: starts for PPO / DAgger rollouts (the PPO agent's
  `gdrl/curriculum.py` already samples Start Pos starts by failure rate).
- **Not for the proof**: a Start Pos does not replay earlier triggers, so a
  segment beaten from a Start Pos is not a level beaten. The final run is
  always one search (snapshots) from the real level start, verified by a
  clean replay from RESET -2 (or -1 on a copy without Start Pos objects).
  Inputs found from a Start Pos are reused as the plan seed for that
  segment in the full run, where they usually still fit.

## 2. Architecture

```
            Geometry Dash 2.2081 + bridge mod (protocol v2)
  RESET(level, start_pos)  STEP(button, ticks)  SAVE/LOAD/DROP_STATE(slot)
                                   |
                gdrl.env.client.GDClient (v2, falls back to v1)
                                   |
          SocketBackend / MockBackend (same API, both with snapshots)
           |                         |                          |
 gdrl.search.segment_search   gdrl.search.distill       GDEnv (+ curriculum)
 (checkpoint beam search)     (BC + DAgger, expert =    PPO (gdrl.train)
   -> solution.npz             segment search)            fine-tuning
   -> replay proof             -> policy checkpoint
           \__________________________|__________________________/
                      gdrl.runlog: meta.json, episodes.jsonl,
                      generations.jsonl (dashboard), ledger (record_beat)
```

## 3. What was built (all offline, tested on the mock)

| Piece | Where | Notes |
|---|---|---|
| Protocol v2, Python side | `gdrl/env/protocol.py`, `client.py`, `backends.py`, `protocol/PROTOCOL.md` | SAVE/LOAD/DROP_STATE, RESET from Start Pos, STATE v2 (1344 B: player 2, hitbox size, floor/ceil, 64 objects, object subtype + flags). Client asks for v2, falls back to v1 on VERSION_MISMATCH (works with the unpatched mod; tested). v1 bytes unchanged. |
| Mock game extended | `gdrl/env/mock_game.py` | ship, wave, gravity portals, speed portals, yellow orbs (press or buffered press), pads, moving saws, a roof, Start Pos, exact `snapshot()/restore()`, optional `restore_error` to simulate inexact restores. Old cube levels are bit-identical (trajectory hash checked). |
| Demon-like mock levels | `gdrl/env/mock_demon.py` (`mock: {kind: demon}`) | 14 sections (cube spikes, orb pits, gravity flips, ship gates with saws, wave zigzags), speeds slow to 3x, ~80 s long (~19,000 ticks). Generator bugs found by the search were fixed (impossible pad and triple-spike jumps at slow speed, a gate unreachable at 3x). |
| Mock server v2 | `gdrl/env/mock_server.py` | speaks v1 and v2, snapshots, Start Pos |
| Checkpoint search | `gdrl/search/segment_search.py`, `gdrl/tools/demon_search.py` | see section 5 |
| Distillation | `gdrl/search/distill.py`, `gdrl/tools/demon_distill.py`, `gdrl/features/demon_encoder.py` | see section 6 |
| Snapshot gate tool | `gdrl/tools/snapshot_check.py` | in-game exactness and speed check for SAVE/LOAD |
| Run logs | `gdrl/runlog.py` | section 8 |
| Mod patch (C++) | `logs/mod-patch/` | compiled cleanly against Geode 5.10.1 / GD 2.2081 (not installed) |
| Tests | `tests/test_protocol_v2.py`, `tests/test_search.py` | 13 new tests; `pytest -q` all green |

### Results on the mock

| Level | Agent | Result | Attempts (rollouts) | Wall | Proof |
|---|---|---|---|---|---|
| mock-demon-1 (14 sections, 18,844 ticks) | checkpoint search | beaten | 48,021 | 53 s | clean replay from RESET completes |
| mock-demon-2 (19,130 ticks) | checkpoint search | beaten | 42,341 | 45 s | same |
| mock-demon-3 (20,426 ticks) | checkpoint search | beaten | 43,829 | 46 s | same |
| 20 single-type sections (cube, orb, gravity, ship, wave; 4 seeds each) | checkpoint search | 20/20 | 5k to 14k each | 4 to 15 s | replay |
| mock-demon-1 | PPO from scratch (demon encoder, 256x256, 3M decisions = ~6M ticks, `configs/demon_ppo_mock.yaml`) | **15.1%** greedy | ~3M decisions | 80 min | `runs/20261011-044703_demon-ppo-mock_s1` |
| mock-demon-1 | distilled policy (BC + DAgger) | see `holistic_progress/level_beats.md` | | | greedy from RESET |

Same level, about the same number of physics ticks: the search finishes it
in under a minute, PPO from scratch gets 15% in 80 minutes.

No backtracking was needed on the three full levels; it fires on hard spots
(e.g. a ship gate reachable only by a jump timed before the portal).

## 4. Audit: what the bridge is missing for demons

Most of these matter only for the *policy* (its observation). The search
never looks at the observation: it needs only exact snapshots, RESET and
the dead/completed flags. That is the main reason it is the primary path.

1. **No snapshots** (v1). Search needs them. Fixed in protocol v2 (patch).
2. **RESET start_pos unsupported** (v1 replies UNSUPPORTED). Also, a level
   copy containing Start Pos objects starts *at* the Start Pos on RESET(-1)
   (the PPO agent's curriculum relies on that). Patch: RESET k >= 0 picks the
   k-th Start Pos by x via `setStartPosObject`; -1 keeps GD's own choice
   (unchanged); new -2 = the real level start, ignoring Start Pos objects
   (use it for full-run proofs on copies that contain Start Pos objects).
   Start Pos runs skip earlier triggers, so they are for practice only;
   exact mid-level starts use snapshots.
3. **Speed portals are invisible**: they are not collision objects
   (`m_objectType` is not a portal type; v1's classify drops them). Patch:
   recognise ids 200/201/202/203/1334 and scan `PlayLayer::m_speedObjects`.
   `x_speed` (current speed) was already correct.
4. **Orbs, pads and portals may be missing from STATE**: v1 asks only
   `staticObjectsInRect` and `damagingObjectsInRect`. To verify in game; if
   missing, scan the level's section objects. (Search unaffected.)
5. **No object subtype**: a yellow orb and a gravity orb, a ship and a wave
   portal, a spike and a saw all look alike. v2 adds `subtype` (in a byte
   that was reserved, so v1 layout is unchanged).
6. **Dual mode**: no player 2 in v1. v2 adds p2 x/y/vel/mode/flags. Deaths of
   either player already end the episode (`destroyPlayer` hook). To check in
   game: `handleButton(down, 1, true)` drives both players in 1-player dual.
7. **Mini and hitbox size** not reported (mini, wave and swing hitboxes are
   smaller). v2: MINI flag and `player_size`.
8. **Ship/ball/wave corridor floor and roof** not reported. v2 has
   `floor_y`/`ceil_y` fields, sent as NaN until their source is confirmed in
   game (`GJGameState::m_portalY`, `getGroundHeightForMode`).
9. **vel_y scale** (0.9 except wave) was measured for some modes only; ball,
   UFO, robot, spider, swing and mini are unverified.
10. **Moving objects**: `getObjectRect` is live, so positions are current each
    tick, if GD keeps its collision sections up to date for moved objects
    (to verify). Rotated hazards are sent as their axis-aligned box.
11. **Invisible / toggled objects**: alpha-0 objects still collide; v2 flags
    them HIDDEN. Group-disabled objects should not appear (to verify).
12. **32 objects** can be too few in dense wave corridors; v2 sends 64.
13. **Determinism threats**: random triggers (2.2 Random / Advanced Random,
    ids 1912 and 2068) or anything time-of-day based. Bloodbath (2.0) and
    Cataclysm (1.9) predate them; Rauchkammer and the top 10 are 2.2 levels
    and must be checked (`gdctl level-export`, grep the object ids). The
    bridge_check soak checks RESET determinism; snapshot_check checks LOAD.
14. **Practice mode** is refused by RESET (correct); LOAD uses it internally
    and restores the flag.

## 5. Checkpoint search (built)

`SegmentSearch` (`gdrl/search/segment_search.py`):

- **Segments** of `segment_ticks` (60 = 0.25 s). A **beam** is up to
  `beam_width` (12) saved states at a segment boundary.
- **Candidates** per node: its own surviving plan first (the inputs that
  survived the previous lookahead), mutations of it, then random press
  patterns. Half are a toggle process with a random rate; half are runs
  whose first switch is uniform over the segment (so every press time is
  equally likely) with log-uniform run lengths. Inputs change every
  `decision_ticks` (2); a **retried** segment uses `fine_decision_ticks` (1,
  frame-perfect). Inputs are stored per tick, so both mix freely.
- A candidate that lives to the boundary is saved, then played on for
  `lookahead_ticks` (150, longer than a cube jump) to score it: doomed states
  (mid-jump into a spike) rank low.
- Survivors are **de-duplicated** by physical state (mode, gravity, ground,
  button, speed, y, vy, x, player 2), and the beam is filled by score and
  then **farthest-point diversity** in (y, vy, ground, button, mode,
  gravity). Without diversity, many mid-jump states pushed the one on-ground
  state out, and the search died at a gap that needed a late jump.
- **Backtracking**: no survivor means the previous beam is rebuilt with a
  bigger budget (and finer inputs); after `retries_per_depth` failures it
  goes back one more segment, up to `max_backtrack`.
- **Verification during search** (`verify_every`): the best node is replayed
  from RESET; if the true state differs the beam is resynced to it, if it
  died the search rolls back to the last verified beam. Tested with a mock
  whose every restore is 0.3 units off: still solved, replay completes.
- **Final proof**: `replay()` = RESET + the per-tick inputs, no snapshots.
- **Mid-level roots**: `run(root_slot=..., goal_x=...)` searches from any
  saved state to a target x (used by DAgger, and for segment work).

- **Slots**: at most `max_found` (48) survivors per expansion hold a slot
  (the worst is evicted), so a search needs about
  `(max_backtrack + 2) * beam_width + max_found + 2` slots (170 with the
  real-game config; the mod has 256).
- **Resume**: every 20 segments the best prefix is written to
  `best_prefix.npz`; `demon_search --resume <file>` replays it (minus 2 s)
  and searches on, so a GD crash costs minutes, not the night.

Cost model for the real game, from the mock runs: about 45k rollouts,
~45k LOADs, ~15k SAVEs and 5.2M to 5.8M physics ticks per 80 s of level. If GD runs
~20k ticks/s under lockstep and a LOAD costs ~30 ms (resetLevel on a big
level), that is ~5 min of ticks plus ~25 min of loads per level, plus
verification replays. At 300 ms per LOAD it becomes ~4 h: still one night.
**Measuring SAVE/LOAD cost on Bloodbath is the first in-game task.**

## 6. Distillation and PPO (built / planned)

- `DemonEncoder` (`encoder: {name: demon}`): player (mode one-hot, gravity,
  mini, button, speed, hitbox, floor/roof distance), 12 nearest objects
  (relative box, gaps to the player's top and bottom, kind one-hot, subtype
  group), Gaussian distance codes in x and y for the 4 nearest, optional
  frame stacking (moving objects).
- **Label simplification** (`simplify_inputs`, built): search solutions
  press at random on open stretches, so BC on them saw contradictory labels
  and plateaued at 97 to 98% train accuracy, and DAgger then oscillated
  (from RESET 12% / 45% / 12%, 22 rounds). Before training, every press run
  is released if the inputs still survive the next 2 s, with undo when a
  later part dies; the result must still survive to the end. On
  mock-demon-1 this removed 356 of 855 press runs (8,680 to 6,014 held
  ticks); the replay still completes.
- `Distiller` (built): BC on the simplified solution, then DAgger rounds. Each round
  plays the greedy policy from RESET and from 24 states spread along the
  solution (to the next one). Wherever it dies, the segment search runs from
  a snapshot taken about 0.4 s before the death (a state the policy really
  reached) to 240 units past it, and those states get the search's inputs as
  labels. Stops when the greedy policy completes from RESET; then a final
  run with no snapshots at all is the verification. The policy is saved as
  a PPO checkpoint, so `gdrl.evaluate --run` and PPO fine-tuning load it.
- PPO segment curriculum (planned, small): starts at snapshots along the
  solution (exact), weighted by failure rate (the PPO agent's
  `gdrl/curriculum.py` already does this for Start Pos starts; a snapshot
  start type is a ~30-line addition), reward = progress delta, death
  penalty, completion bonus; initialised from the distilled policy. The
  point is robustness (sticky actions, small perturbations) and transfer to
  unseen levels, not the beat itself.

## 7. Milestones

| # | Milestone | Gate |
|---|---|---|
| D0 | Offline machinery (this document) | mock demons beaten by search + replay; tests green. **Done** |
| D1 | Apply mod patch, build | `gdctl build mod`; handshake v2; v1 client (gdctl bridge-probe) still works |
| D2 | Snapshot exactness in game | `snapshot_check --port 5555` [PASS] on a simple level, then on Bloodbath, with `--solution` at deep save points; record SAVE/LOAD ms. If not exact: extend `GDRL_PLAYER_FIELDS`, try `snapshot-load=1`, last resort replay-based restore (exact, slower) |
| D3 | Search beats an easy real level (Stereo Madness), replay verified | ledger row |
| D4 | Search beats a hard demon (e.g. a 2.2 cube/ship/wave mix) | ledger row |
| D4b | Bloodbath segment map from a verified community Start Pos copy | per-segment success from each Start Pos to the next; ledger rows "Bloodbath seg k/N" |
| D5 | **Bloodbath** full run from the level start, replay verified | ledger row. Per-segment milestones logged ("seg k/N") |
| D6 | Cataclysm, then Rauchkammer (check determinism first) | ledger rows |
| D7 | Distilled policy beats each level greedily from RESET | ledger rows (agent search-distilled-policy) |
| D8 | Pointercrate top 10, one by one, same pipeline | ledger rows |

## 8. Measuring progress, and the run-log format (for the dashboard)

Per run folder `runs/<time>_<name>_s<seed>/`:

- `meta.json`: `level`, `level_id`, `game` (mock/real), `difficulty`,
  `length` (level x length if known), `agent`, `step_unit`
  (`physics_tick` for search/distill), `tick_rate`, `start_pos`, and the
  search config.
- `episodes.jsonl`, one JSON object per line, one per episode or search
  rollout: `t` (unix s), `step` (cumulative env steps), `level`, `segment`
  (search: segment index of the start node; distill: start index or null),
  `start_x`, `start_tick`, `progress` (0 to 1, at the end), `completed`,
  `death_x` (null if it did not die), `length` (ticks), `agent`
  (`checkpoint-search`, `checkpoint-search-replay`, `distill-policy`,
  `distill-expert`), `attempt` (cumulative attempt number).
- `generations.jsonl`, one per search generation (segment expansion) or
  DAgger round: `t`, `step`, `generation`, `segment`, `best_fitness`
  (furthest progress reached in that generation), `mean_fitness`,
  `best_progress` (progress of the best beam node), `agent`, plus
  `rollouts`, `candidates`, `survivors`, `full` (survivors with a full
  lookahead), `beam`, `retry`, `backtracks`, `x`, `tick`.
- `result.json`: totals (rollouts, deaths, env ticks, loads, saves,
  backtracks, resyncs, wall time, `verified`).
- `solution.npz`: `ticks` (per-tick button), `start_pos`, `level_id`, `tick_rate`.

Progress measures:

- **best %**: the furthest progress any beam node has reached
  (`generations.jsonl: best_progress`).
- **per-segment success**: survivors / candidates per generation, and the
  retries and backtracks it took; the hard segments are where these spike.
  Deaths per x (`episodes.jsonl: death_x`) give the heat map.
- **full-run success**: a clean replay from RESET completes
  (`result.json: verified`), and for the policy, a greedy run from RESET.
- Every beat goes to the ledger (`python -m gdrl.tools.record_beat`, done
  automatically by `--record`), with attempts = rollouts, generations =
  segment expansions (or DAgger rounds), env steps = physics ticks.

## 9. The mod/protocol patch

Files: `logs/mod-patch/src/*` (replaces `mod/src/*`),
`logs/mod-patch/tools/standalone_server.cpp` (replaces
`mod/tools/standalone_server.cpp`), `logs/mod-patch/mod.json` (version
v0.2.0, new setting `snapshot-load`). Originals are in `logs/mod-patch/orig/`.
Apply and build (game lock held):

```powershell
Copy-Item logs\mod-patch\src\* mod\src\ -Force
Copy-Item logs\mod-patch\tools\standalone_server.cpp mod\tools\ -Force
Copy-Item logs\mod-patch\mod.json mod\ -Force
& $py -m gdctl build mod
```

What it changes:

- `Protocol.hpp`: version 2 (accepts 1), `StateV2`, `Snapshot`, subtypes,
  flags, `toV1()` for v1 clients, `kMaxSnapshots = 256`.
- `Bridge`: agreed version per client (header + STATE layout), SAVE/LOAD/DROP
  parsing (v2 headers only), `sendHelloAck(tickRate, maxSnapshots)`.
- `Controller`: HELLO 1..2; RESET with Start Pos (`setStartPosObject`, -1
  clears); SAVE = `createCheckpoint()` (retained) + `PlayerFix` (player
  fields the checkpoint misses: exact `m_yVelocity`, ground/slope flags,
  jump buffers, ring/pad touch state, `m_holdingButtons`, rotation, ...) +
  tick + button; LOAD = practice-mode respawn at that checkpoint
  (`m_isPracticeMode` set temporarily, checkpoint pushed on
  `m_checkpointArray`, `resetLevel()`), then `PlayerFix`, tick and button
  restored without a press/release event; DROP; slots cleared on level exit.
- `StateReader`: fills StateV2 (subtypes, speed portals, player size, mini,
  dual/player 2, HIDDEN flag).

Risks in it: `createCheckpoint` outside practice mode, `resetLevel` honouring
the temporary practice flag, and the PlayerFix field list. `snapshot_check`
finds out; the setting `snapshot-load = 1` switches LOAD to
`resetLevel(); loadFromCheckpoint(cp)` without a rebuild.

## 10. Risks

| Risk | Mitigation |
|---|---|
| LOAD is not exact (practice checkpoints are known to drift: rotation, velocity rounding, trigger state) | PlayerFix; verify_every resyncs; final proof is always a clean replay. Last resort: restore = RESET + replay of the prefix (exact; cost grows with x, so use with sparse beams and long segments) |
| LOAD is slow on huge levels (resetLevel re-initialises tens of thousands of objects) | measure in D2; fewer loads (bigger segments, smaller beams, early stop already in place); keep the game window small; overnight budgets |
| Non-determinism (random triggers, frame-time effects) | check level strings; bridge_check soak; replay twice |
| Frame-perfect inputs | fine_decision_ticks = 1 on retries; can be forced globally |
| Dual mode, swing, spider, teleports behave in ways the mock does not model | the search does not need a model; only the policy does |
| The policy memorizes one level | expected; the beat comes from the search; generalization is a separate study |
| GD crashes during a long search | the search logs every generation; add resume-from-solution-prefix (replay prefix, continue) if needed; gdctl can relaunch GD |

## 11. Needed from the game (handover)

1. Game lock, GD running, Click Between Frames and QOLMod off.
2. Apply the patch and build (section 9); restart GD.
3. `gdctl bridge-probe` (v1 client) and a v2 handshake.
4. `snapshot_check` on Stereo Madness, then on `local:gdrl-bloodbath`
   (normal mode), with timings.
5. Then D3 to D5: `demon_search` on Stereo Madness, then Bloodbath, with
   `--record`.
