"""Trace the policy network decision by decision, for the network visualizer.

A trace holds one greedy run of a policy: for every decision the observation,
each hidden layer's activations, the logits (release / jump), the value
estimate and the action, plus what the visualizer's scene needs (the player,
every object the bridge reported and the frames it was in view, which objects
filled the encoder's slots), the feature names and the network's weights.
The visualizer (``gdrl/visualizer``) replays it; the dashboard serves it at
``/viz``.

    python -m gdrl.tools.nn_trace record rk_beat        # watch preset (configs/watch.yaml): opens the level in GD
    python -m gdrl.tools.nn_trace record bb_beat --export
    python -m gdrl.tools.nn_trace build logs/viz/rk_beat.rec.npz     # rebuild the trace from a saved recording
    python -m gdrl.tools.nn_trace export results/viz/rk_beat.trace.json.gz   # -> results/viz/rk_beat.html
    python -m gdrl.tools.nn_trace mock --config C --checkpoint P --level 1 --name mock-l1   # no game needed
    python -m gdrl.tools.nn_trace list

``record`` takes the gdctl game lock (owner ``viz``), opens the level the way
``gdrl.tools.watch`` does (download + prepare-training), sets the bridge's
legacy-view as the preset needs and watch-mode off (full speed), keeps GD
un-minimized, plays one greedy episode from ``start_pos`` and releases the
lock. The raw states go to ``logs/viz/<name>.rec.npz`` (so the trace can be
rebuilt without the game), the trace to ``results/viz/<name>.trace.json.gz``.

Trace format (JSON, gzipped): arrays are packed as ``{"dtype", "shape",
"b64"}`` (little-endian; ``scale`` means "divide by it", for quantized
values: observations as int16, tanh activations as int8). Top-level keys:
``meta`` (level, checkpoint, result, training), ``encoder`` (its parameters),
``features`` and ``groups`` (names and layout of the observation),
``network`` (layer sizes and the policy's weights), ``frames`` (per
decision: obs, hidden, logits, value, action, player state, slot object
ids), ``objects`` (every object the bridge reported, with the frames it was
in view), ``view`` (per-frame report as add/remove lists), ``events``
(mode, gravity, size, dual and speed changes) and ``end`` (the final state).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gdrl.env.protocol import BLOCK, GameState, ObjectFlag, ObjectKind, ObjectSubtype, PlayerMode

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "results" / "viz"
REC_DIR = ROOT / "logs" / "viz"
SUFFIX = ".trace.json.gz"
FORMAT = "gdrl-nn-trace"
VERSION = 1
OWNER = "viz"
TICKS_PER_SECOND = 240

MODE_NAMES = [m.name.lower() for m in PlayerMode]
KIND_NAMES = {int(k): k.name.lower() for k in ObjectKind}
SUBTYPE_NAMES = {int(s): s.name.lower() for s in ObjectSubtype}

# --------------------------------------------------------------------------- packing

_CODES = {"float32": "f4", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4"}
_DTYPES = {v: np.dtype(k).newbyteorder("<") for k, v in _CODES.items()}


def pack(a: Any, dtype: str | None = None, scale: float | None = None) -> dict[str, Any]:
    """A numpy array as ``{"dtype", "shape", "b64"[, "scale"]}`` (little-endian).

    With ``scale``, values are multiplied by it, rounded and clipped to the
    integer ``dtype`` (readers divide by ``scale``)."""
    a = np.asarray(a)
    if scale is not None:
        info = np.iinfo(np.dtype(dtype))
        a = np.clip(np.rint(a.astype(np.float64) * scale), info.min, info.max)
    if dtype is not None:
        a = a.astype(dtype)
    name = a.dtype.name
    if name not in _CODES:
        raise TypeError(f"cannot pack dtype {a.dtype}")
    a = np.ascontiguousarray(a, dtype=_DTYPES[_CODES[name]])
    out: dict[str, Any] = {"dtype": _CODES[name], "shape": list(a.shape),
                           "b64": base64.b64encode(a.tobytes()).decode("ascii")}
    if scale is not None:
        out["scale"] = float(scale)
    return out


def unpack(p: dict[str, Any]) -> np.ndarray:
    """Inverse of ``pack`` (float64 when the array was quantized)."""
    a = np.frombuffer(base64.b64decode(p["b64"]), dtype=_DTYPES[p["dtype"]]).reshape(p["shape"])
    if "scale" in p:
        return a.astype(np.float64) / p["scale"]
    return a


# --------------------------------------------------------------------------- encoder mirror

def check_encoder(enc) -> None:
    from gdrl.features.encoder import ObjectListEncoder

    if not isinstance(enc, ObjectListEncoder):
        raise SystemExit(f"the visualizer needs the object_list encoder, not {type(enc).__name__}")


def slot_indices(enc, state: GameState) -> np.ndarray:
    """Indices into ``state.objects`` of the objects in the encoder's slots, in
    slot order. Mirrors ``ObjectListEncoder.encode`` (build_trace checks the
    result against the observation, so a drift fails loudly)."""
    objs = state.objects
    x, y = objs["x"].astype(np.float64), objs["y"].astype(np.float64)
    w, h = objs["w"].astype(np.float64), objs["h"].astype(np.float64)
    dx, dy = x - state.player_x, y - state.player_y
    visible = ((dx + w / 2 > -BLOCK / 2) & (dx - w / 2 <= enc.max_dx) & (np.abs(dy) <= enc.max_dy)
               & (objs["kind"] != ObjectKind.NONE))
    idx = np.nonzero(visible)[0]
    kinds = objs["kind"][idx].astype(np.int64)
    idx = idx[np.lexsort((h[idx], w[idx], kinds, dy[idx], dx[idx]))]
    if enc.focus == "obstacles":
        feet = state.player_y - BLOCK / 2
        ym = 2 * state.player_y - y if state.upside_down else y
        obstacle = (objs["kind"][idx] == ObjectKind.HAZARD) | (
            (objs["kind"][idx] == ObjectKind.SOLID) & (ym[idx] + h[idx] / 2 > feet + 2.0))
        idx = idx[np.argsort(~obstacle, kind="stable")]
    return idx[: enc.k]


def encoder_params(enc) -> dict[str, Any]:
    from gdrl.env.mock_game import JUMP_VELOCITY

    return {
        "name": "object_list", "k": enc.k, "focus": enc.focus, "near_slots": enc.near_slots,
        "max_dx": enc.max_dx, "max_dy": enc.max_dy, "block": BLOCK,
        "vel_scale": 2 * JUMP_VELOCITY, "max_x_speed": enc.MAX_X_SPEED,
        "player_features": enc.PLAYER_FEATURES, "slot_size": enc.SLOT_SIZE,
        "kinds": [KIND_NAMES[int(k)] for k in enc.KINDS],
        "rbf_start": enc.rbf_start, "rbf_centers": [float(c) for c in enc.RBF_CENTERS], "rbf_width": enc.RBF_WIDTH,
        "profile": bool(enc.profile), "profile_start": enc.profile_start,
        "profile_dx": [float(c) for c in enc.PROFILE_DX],
        "position": bool(enc.position), "position_start": enc.position_start,
        "position_periods": [float(p) for p in enc.POSITION_PERIODS],
        "size": int(enc.observation_space.shape[0]),
    }


def feature_spec(enc) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Names and groups of the observation's features, in observation order.

    Each feature: ``id`` (short), ``label`` (for tooltips), ``group``, and how
    to show its value: ``unit`` + ``scale`` (value * scale in game units),
    or ``kind`` ``bool`` / ``onehot`` / ``bump`` / ``wave``. Each group: a
    ``grid`` of feature indices (rows of cells, as the visualizer draws them),
    with optional row and column labels."""
    p = encoder_params(enc)
    feats: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []

    def add(fid: str, label: str, group: str, **kw) -> int:
        feats.append({"id": fid, "label": label, "group": group, **kw})
        return len(feats) - 1

    # Player.
    row1 = [add("y", "Height above the ground", "player", unit="u", scale=p["max_dy"]),
            add("vy", "Vertical speed (up is +)", "player", unit="u/s", scale=p["vel_scale"]),
            add("ground", "On the ground", "player", kind="bool"),
            add("flip", "Gravity flipped", "player", kind="bool"),
            add("speed", "Horizontal speed", "player", unit="u/s", scale=p["max_x_speed"])]
    row2 = [add(f"mode_{m}", f"Game mode is {m}", "player", kind="onehot", mode=i) for i, m in enumerate(MODE_NAMES)]
    groups.append({"id": "player", "label": "Player", "sub": "own state and game mode",
                   "grid": [row1 + row2],
                   "col_labels": [["y", "vy", "gd", "fl", "sp", "C", "S", "B", "U", "W", "R", "Sp", "Sw"]]})
    # Object slots.
    kinds = p["kinds"]
    slot_rows = []
    for s in range(enc.k):
        n = s + 1
        r = [add(f"o{n}.dx", f"Object {n}: distance ahead (centre)", f"slot{n}", unit="u", scale=p["max_dx"], slot=s),
             add(f"o{n}.dy", f"Object {n}: height above the player (centre)", f"slot{n}", unit="u", scale=p["max_dy"], slot=s),
             add(f"o{n}.w", f"Object {n}: width", f"slot{n}", unit="u", scale=p["max_dx"], slot=s),
             add(f"o{n}.h", f"Object {n}: height", f"slot{n}", unit="u", scale=p["max_dy"], slot=s)]
        r += [add(f"o{n}.{k}", f"Object {n} is a {k}", f"slot{n}", kind="onehot", slot=s, objkind=k) for k in kinds]
        slot_rows.append(r)
    groups.append({"id": "slots", "label": "Objects", "sub": f"{enc.k} nearest {'obstacles' if enc.focus == 'obstacles' else 'objects'}",
                   "grid": slot_rows, "row_labels": [str(i + 1) for i in range(enc.k)],
                   "col_labels": [["dx", "dy", "w", "h", *[k[:3] for k in kinds]]], "slot_rows": True})
    # Fine distance code (Gaussian bumps of the left-edge distance) for the nearest slots.
    rbf_rows = []
    for s in range(enc.near_slots):
        rbf_rows.append([add(f"t{s + 1}@{c:+.0f}", f"Object {s + 1}: left edge about {c:+.0f} u ahead (bump)",
                             f"timing{s + 1}", kind="bump", slot=s, center=float(c)) for c in enc.RBF_CENTERS])
    if rbf_rows:
        groups.append({"id": "timing", "label": "Timing code", "sub": "left-edge distance, objects 1-%d" % enc.near_slots,
                       "grid": rbf_rows, "row_labels": [str(i + 1) for i in range(enc.near_slots)],
                       "axis": [float(enc.RBF_CENTERS[0]), float(enc.RBF_CENTERS[-1])], "axis_unit": "u"})
    # Terrain profile.
    if enc.profile:
        cols = [float(c) for c in enc.PROFILE_DX]
        sup = [add(f"floor@{c:.0f}", f"Floor {c:.0f} u ahead (height vs the feet)", "terrain", unit="u", scale=p["max_dy"], column=j, row="floor")
               for j, c in enumerate(cols)]
        haz = [add(f"spike@{c:.0f}", f"Hazard on the floor {c:.0f} u ahead", "terrain", kind="bool", column=j, row="hazard")
               for j, c in enumerate(cols)]
        cei = [add(f"roof@{c:.0f}", f"Clearance overhead {c:.0f} u ahead (vs the feet)", "terrain", unit="u", scale=p["max_dy"], column=j, row="roof")
               for j, c in enumerate(cols)]
        groups.append({"id": "terrain", "label": "Terrain ahead", "sub": f"{len(cols)} columns, 0-{cols[-1]:.0f} u",
                       "grid": [sup, haz, cei], "row_labels": ["floor", "hazard", "roof"],
                       "axis": [cols[0], cols[-1]], "axis_unit": "u"})
    # Position in the level.
    if enc.position:
        per = [float(q) for q in enc.POSITION_PERIODS]
        sin = [add(f"sin{q:.0f}", f"sin(2 pi x / {q:.0f})", "position", kind="wave", period=q) for q in per]
        cos = [add(f"cos{q:.0f}", f"cos(2 pi x / {q:.0f})", "position", kind="wave", period=q) for q in per]
        groups.append({"id": "position", "label": "Position", "sub": "x as sin/cos, periods 30-61k u",
                       "grid": [sin, cos], "row_labels": ["sin", "cos"],
                       "axis": [per[0], per[-1]], "axis_unit": "u", "axis_log": True})
    if len(feats) != p["size"]:
        raise RuntimeError(f"feature_spec covers {len(feats)} features, the encoder has {p['size']}")
    return feats, groups


# --------------------------------------------------------------------------- policy

def load_agent(cfg: dict[str, Any], checkpoint: str | Path | None, observation_space, action_space):
    """The config's PPO agent on the CPU (no BC anchor), with the checkpoint's weights."""
    from gdrl.agents.registry import make_agent

    acfg = dict(cfg["agent"])
    if acfg.get("name") != "ppo":
        raise SystemExit(f"the visualizer traces PPO policies, not {acfg.get('name')!r}")
    acfg.update(device="cpu", act_device="cpu", bc_data=None, bc_coef=0.0)
    agent = make_agent(acfg, observation_space, action_space, num_envs=1, total_steps=1)
    if checkpoint:
        agent.load(checkpoint)
    return agent


def policy_layers(agent) -> list[tuple[np.ndarray, np.ndarray]]:
    """(W, b) of each Linear layer of the policy MLP, input to output."""
    from torch import nn

    mods = list(agent.policy)
    lins = [m for m in mods if isinstance(m, nn.Linear)]
    acts = [m for m in mods if not isinstance(m, nn.Linear)]
    if not lins or not all(isinstance(m, nn.Tanh) for m in acts) or len(acts) != len(lins) - 1:
        raise SystemExit("the visualizer supports tanh MLP policies (Linear -> Tanh -> ... -> Linear)")
    return [(m.weight.detach().cpu().numpy().astype(np.float32), m.bias.detach().cpu().numpy().astype(np.float32))
            for m in lins]


def forward(agent, obs: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Hidden activations (one array per layer), logits and values for ``obs`` (N, D)."""
    import torch
    from torch import nn

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        hidden, h = [], x
        for m in agent.policy:
            h = m(h)
            if isinstance(m, nn.Tanh):
                hidden.append(h.numpy().copy())
        logits = h.numpy().copy()
        value = agent.value(x).squeeze(-1).numpy().copy()
    return hidden, logits, value


def run_episode(env, agent, start_pos: int = -2, log=print) -> tuple[list[GameState], np.ndarray, np.ndarray]:
    """One greedy episode: the state before each decision (plus the final
    state), the actions and the observations the policy saw."""
    obs, _ = env.reset(options={"start_pos": start_pos})
    states = [env.unwrapped.state]
    seen, actions = [], []
    t0 = time.time()
    while True:
        a = int(agent.act(obs[None], deterministic=True).action[0])
        seen.append(obs)
        actions.append(a)
        obs, _r, term, trunc, _info = env.step(a)
        states.append(env.unwrapped.state)
        if len(actions) % 1000 == 0:
            s = states[-1]
            log(f"  {len(actions)} decisions, x = {s.player_x:.0f} ({100 * s.progress:.1f}%), {time.time() - t0:.0f} s")
        if term or trunc:
            break
    return states, np.asarray(actions, dtype=np.uint8), np.asarray(seen, dtype=np.float32)


# --------------------------------------------------------------------------- recordings

def save_recording(path: Path, states: list[GameState], actions: np.ndarray, obs: np.ndarray,
                   meta: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.stack([np.frombuffer(s.to_bytes(2), dtype=np.uint8) for s in states])
    np.savez_compressed(path, states=raw, actions=actions, obs=obs, meta=np.array(json.dumps(meta)))
    return path


def load_recording(path: Path) -> tuple[list[GameState], np.ndarray, np.ndarray, dict[str, Any]]:
    d = np.load(path, allow_pickle=False)
    states = [GameState.from_bytes(row.tobytes()) for row in d["states"]]
    return states, d["actions"], d["obs"], json.loads(str(d["meta"]))


# --------------------------------------------------------------------------- the trace

def _orient_hazards(tab: dict[str, np.ndarray], path_x: np.ndarray | None = None,
                    path_y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Which way each hazard points, and the gap to the surface it sits on.

    The bridge sends hitboxes, not sprites: a spike is a small box. To draw it
    as a triangle, find the solid (or the ground) its base touches: the
    nearest surface below (point up), above (down), left (point right) or
    right (point left) within ~16 units. 0 up, 1 down, 2 left, 3 right.
    A spike with no reported surface (it hangs from decoration) points at
    the player's path: down when the player passed below it, else up."""
    x, y, w, h, kind = tab["x"], tab["y"], tab["w"], tab["h"], tab["kind"]
    orient = np.zeros(len(x), dtype=np.uint8)
    gap = np.zeros(len(x), dtype=np.float32)
    sol = np.nonzero(kind == ObjectKind.SOLID)[0]
    sl, sr = x[sol] - w[sol] / 2, x[sol] + w[sol] / 2
    sb, st = y[sol] - h[sol] / 2, y[sol] + h[sol] / 2
    reach = 16.0
    for i in np.nonzero(kind == ObjectKind.HAZARD)[0]:
        if tab["subtype"][i] == ObjectSubtype.SAW:
            continue
        l, r, b, t = x[i] - w[i] / 2, x[i] + w[i] / 2, y[i] - h[i] / 2, y[i] + h[i] / 2
        xo = (sl < r) & (sr > l)
        yo = (sb < t) & (st > b)
        cand = []
        below = b - st[xo]
        below = below[(below >= -1.0) & (below <= reach)]
        g_floor = b - 0.0
        if -1.0 <= g_floor <= reach:
            below = np.append(below, g_floor)
        if len(below):
            cand.append((float(below.min()), 0))
        above = sb[xo] - t
        above = above[(above >= -1.0) & (above <= reach)]
        if len(above):
            cand.append((float(above.min()), 1))
        if w[i] > h[i] * 1.2:  # lying on its side: hitbox wider than tall
            right = sl[yo] - r
            right = right[(right >= -1.0) & (right <= reach)]
            if len(right):
                cand.append((float(right.min()) - 0.5, 2))
            left = l - sr[yo]
            left = left[(left >= -1.0) & (left <= reach)]
            if len(left):
                cand.append((float(left.min()) - 0.5, 3))
        if cand:
            g, o = min(cand)
            orient[i], gap[i] = o, max(0.0, g)
        else:
            gap[i] = -1.0  # floating: draw centred on the hitbox
            if path_x is not None and len(path_x) > 1 and h[i] >= w[i] * 0.8:
                orient[i] = 1 if y[i] > np.interp(x[i], path_x, path_y) else 0
    return orient, gap


def build_trace(states: list[GameState], actions: np.ndarray, cfg: dict[str, Any],
                checkpoint: str | Path | None, meta: dict[str, Any], obs_recorded: np.ndarray | None = None,
                agent=None) -> dict[str, Any]:
    """The trace dict of one recorded episode (``states`` has one more entry
    than ``actions``: the final state)."""
    from gymnasium import spaces

    from gdrl.features.encoder import make_encoder

    n = len(actions)
    if len(states) != n + 1:
        raise ValueError(f"need len(states) == len(actions) + 1, got {len(states)} and {n}")
    ecfg = dict(cfg["env"]["encoder"])
    enc = make_encoder(ecfg.pop("name"), **ecfg)
    check_encoder(enc)
    obs = np.stack([enc.encode(s) for s in states[:-1]]).astype(np.float32)
    obs_err = float(np.abs(obs - obs_recorded).max()) if obs_recorded is not None and len(obs_recorded) == n else None
    if agent is None:
        agent = load_agent(cfg, checkpoint, enc.observation_space, spaces.Discrete(2))
    layers = policy_layers(agent)
    hidden, logits, value = forward(agent, obs)
    greedy = logits.argmax(axis=1)
    agree = int((greedy == actions).sum())

    # Objects: one table entry per distinct (x, y, w, h, kind, subtype); per
    # frame the entries in the bridge's view and the ones in the slots.
    table: dict[tuple, int] = {}
    rows: list[tuple] = []
    first: list[int] = []
    last: list[int] = []
    oflags: list[int] = []
    act: list[int] = []
    view_prev: set[int] = set()
    add_ptr, add_idx, del_ptr, del_idx = [0], [], [0], []
    slots = np.full((n, enc.k), -1, dtype=np.int32)
    slot_err = 0.0
    for f in range(n):
        s = states[f]
        objs = s.objects
        ids = np.empty(len(objs), dtype=np.int64)
        for j, o in enumerate(objs):
            key = (float(o["x"]), float(o["y"]), float(o["w"]), float(o["h"]), int(o["kind"]), int(o["subtype"]))
            t = table.get(key)
            if t is None:
                t = table[key] = len(rows)
                rows.append(key)
                first.append(f)
                last.append(f)
                oflags.append(0)
                act.append(-1)
            last[t] = f
            fl = int(o["oflags"])
            oflags[t] |= fl
            if fl & ObjectFlag.ACTIVATED and act[t] < 0:
                act[t] = f
            ids[j] = t
        cur = set(ids.tolist())
        add_idx += sorted(cur - view_prev)
        del_idx += sorted(view_prev - cur)
        add_ptr.append(len(add_idx))
        del_ptr.append(len(del_idx))
        view_prev = cur
        si = slot_indices(enc, s)
        slots[f, : len(si)] = ids[si]
        if len(si):  # the slots we found must be the ones the observation describes
            o = objs[si]
            want = np.stack([(o["x"] - s.player_x) / enc.max_dx, (o["y"] - s.player_y) / enc.max_dy,
                             o["w"] / enc.max_dx, o["h"] / enc.max_dy], axis=1)
            got = obs[f, enc.PLAYER_FEATURES: enc.PLAYER_FEATURES + enc.SLOT_SIZE * len(si)].reshape(len(si), -1)[:, :4]
            slot_err = max(slot_err, float(np.abs(np.clip(want, -1, 1) - got).max()))
    if slot_err > 1e-3:
        raise RuntimeError(f"slot objects do not match the observation (max error {slot_err:.4f}): "
                           "nn_trace.slot_indices no longer mirrors the encoder")
    rows_a = np.array(rows, dtype=np.float64).reshape(-1, 6)
    tab = {"x": rows_a[:, 0].astype(np.float32), "y": rows_a[:, 1].astype(np.float32),
           "w": rows_a[:, 2].astype(np.float32), "h": rows_a[:, 3].astype(np.float32),
           "kind": rows_a[:, 4].astype(np.uint8), "subtype": rows_a[:, 5].astype(np.uint8)}
    px = np.array([s.player_x for s in states], dtype=np.float64)
    keep = np.r_[True, px[1:] > np.maximum.accumulate(px)[:-1]]  # np.interp wants increasing x
    orient, gap = _orient_hazards(tab, px[keep], np.array([s.player_y for s in states], dtype=np.float64)[keep])

    # Events along the run (mode, gravity, size, dual, speed), from the states.
    events: list[dict[str, Any]] = []
    for f in range(1, n + 1):
        a, b = states[f - 1], states[f]
        if int(b.mode) != int(a.mode):
            events.append({"f": f, "type": "mode", "to": MODE_NAMES[int(b.mode)], "from": MODE_NAMES[int(a.mode)]})
        if b.upside_down != a.upside_down:
            events.append({"f": f, "type": "gravity", "flipped": b.upside_down})
        if b.mini != a.mini:
            events.append({"f": f, "type": "size", "mini": b.mini})
        if b.dual != a.dual:
            events.append({"f": f, "type": "dual", "on": b.dual})
        if a.x_speed > 0 and abs(b.x_speed - a.x_speed) > 0.02 * a.x_speed and not b.terminal:
            events.append({"f": f, "type": "speed", "speed": round(float(b.x_speed), 2)})
    end = states[-1]
    fs = int(cfg["env"].get("frame_skip", 4))
    p_jump = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0]).astype(np.float64)))
    meta = {
        **meta,
        "decisions": n,
        "frame_skip": fs,
        "tick_rate": TICKS_PER_SECOND,
        "level_length": float(states[0].level_length),
        "start_x": float(states[0].player_x),
        "completed": bool(end.completed),
        "dead": bool(end.dead),
        "progress": float(end.progress),
        "seconds": round((end.tick - states[0].tick) / TICKS_PER_SECOND, 2),
        "greedy_agreement": agree,
        "obs_check": obs_err,
        "presses": int(((actions[1:] == 1) & (actions[:-1] == 0)).sum() + (actions[:1] == 1).sum()),
        "held_fraction": round(float(actions.mean()), 4),
        "p_jump_mean": round(float(p_jump.mean()), 4),
    }
    feats, groups = feature_spec(enc)
    return {
        "format": FORMAT,
        "version": VERSION,
        "meta": meta,
        "encoder": encoder_params(enc),
        "features": feats,
        "groups": groups,
        "network": {
            "sizes": [int(layers[0][0].shape[1])] + [int(W.shape[0]) for W, _ in layers],
            "activation": "tanh",
            "outputs": ["Release", "Jump"],
            "value": "separate critic MLP of the same shape (only its output is traced)",
            "layers": [{"W": pack(W), "b": pack(b)} for W, b in layers],
        },
        "frames": {
            "obs": pack(obs, "int16", 32767),
            "hidden": [pack(hd, "int8", 127) for hd in hidden],
            "logits": pack(logits.astype(np.float32)),
            "value": pack(value.astype(np.float32)),
            "action": pack(actions.astype(np.uint8)),
            "x": pack(np.array([s.player_x for s in states[:-1]], dtype=np.float32)),
            "y": pack(np.array([s.player_y for s in states[:-1]], dtype=np.float32)),
            "vy": pack(np.array([s.vel_y for s in states[:-1]], dtype=np.float32)),
            "rot": pack(np.array([s.rotation for s in states[:-1]], dtype=np.float32)),
            "xs": pack(np.array([s.x_speed for s in states[:-1]], dtype=np.float32)),
            "mode": pack(np.array([int(s.mode) for s in states[:-1]], dtype=np.uint8)),
            "flags": pack(np.array([int(s.flags) & 0xFF for s in states[:-1]], dtype=np.uint8)),
            "size": pack(np.array([s.player_size for s in states[:-1]], dtype=np.float32)),
            "floor": pack(np.array([s.floor_y for s in states[:-1]], dtype=np.float32)),
            "ceil": pack(np.array([s.ceil_y for s in states[:-1]], dtype=np.float32)),
            "p2x": pack(np.array([s.p2_x for s in states[:-1]], dtype=np.float32)),
            "p2y": pack(np.array([s.p2_y for s in states[:-1]], dtype=np.float32)),
            "p2mode": pack(np.array([int(s.p2_mode) for s in states[:-1]], dtype=np.uint8)),
            "p2flags": pack(np.array([int(s.p2_flags) & 0xFF for s in states[:-1]], dtype=np.uint8)),
            "tick": pack(np.array([s.tick for s in states[:-1]], dtype=np.uint32)),
            "slots": pack(slots),
        },
        "end": {"x": float(end.player_x), "y": float(end.player_y), "mode": int(end.mode),
                "flags": int(end.flags) & 0xFF, "completed": bool(end.completed), "dead": bool(end.dead)},
        "objects": {
            "count": len(rows),
            "x": pack(tab["x"]), "y": pack(tab["y"]), "w": pack(tab["w"]), "h": pack(tab["h"]),
            "kind": pack(tab["kind"]), "subtype": pack(tab["subtype"]),
            "oflags": pack(np.array(oflags, dtype=np.uint8)),
            "first": pack(np.array(first, dtype=np.int32)), "last": pack(np.array(last, dtype=np.int32)),
            "activated": pack(np.array(act, dtype=np.int32)),
            "orient": pack(orient), "gap": pack(gap),
            "kind_names": KIND_NAMES, "subtype_names": SUBTYPE_NAMES,
        },
        "view": {"add_ptr": pack(np.array(add_ptr, dtype=np.int32)), "add": pack(np.array(add_idx, dtype=np.int32)),
                 "del_ptr": pack(np.array(del_ptr, dtype=np.int32)), "del": pack(np.array(del_idx, dtype=np.int32))},
        "events": events,
        "modes": MODE_NAMES,
    }


def _finite(o: Any) -> Any:
    """JSON has no NaN/inf: map them to null (packed arrays keep theirs in binary)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def write_trace(trace: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(_finite(trace), separators=(",", ":"), allow_nan=False)
    path.write_bytes(gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0))
    return path


def read_trace(path: Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)


def trace_name(path: Path) -> str:
    name = Path(path).name
    for suf in (SUFFIX, ".trace.json", ".json.gz", ".json"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(path).stem


def list_traces(base: Path = TRACE_DIR) -> list[dict[str, Any]]:
    """Summaries of the traces in ``base`` (reads only each file's meta)."""
    out = []
    for p in sorted(Path(base).glob("*" + SUFFIX)):
        try:
            t = read_trace(p)
        except (OSError, ValueError):
            continue
        m = t.get("meta", {})
        out.append({"name": trace_name(p), "file": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime,
                    "title": m.get("title") or trace_name(p), "level": m.get("level"),
                    "difficulty": m.get("difficulty"), "decisions": m.get("decisions"),
                    "completed": m.get("completed"), "progress": m.get("progress"), "seconds": m.get("seconds"),
                    "checkpoint": m.get("checkpoint"), "recorded_at": m.get("recorded_at")})
    return out


def export_html(trace_path: Path, out: Path | None = None) -> Path:
    """A single offline HTML file: the visualizer with the trace embedded."""
    from gdrl.visualizer import render_page

    trace_path = Path(trace_path)
    data = trace_path.read_bytes()
    if data[:2] != b"\x1f\x8b":
        data = gzip.compress(data, compresslevel=9, mtime=0)
    name = trace_name(trace_path)
    out = Path(out) if out else trace_path.with_name(name + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(embed=base64.b64encode(data).decode("ascii"), name=name), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- the game

def _stamp() -> str:
    """Now, Pacific time (the project's convention), e.g. "2026-10-19 09:48 PDT"."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:  # no tz database on Windows: local time is Pacific here
        now = datetime.now().astimezone()
        off = now.utcoffset().total_seconds() / 3600 if now.utcoffset() else 0.0
        name = {-7.0: "PDT", -8.0: "PST"}.get(off, f"UTC{off:+.0f}")
        return now.strftime("%Y-%m-%d %H:%M ") + name


def _training(cfg: dict[str, Any]) -> dict[str, Any]:
    """How the policy was trained, as far as its config says (for the page's caveat)."""
    a, t = cfg.get("agent", {}), cfg.get("train", {})
    cloned = bool(a.get("bc_data")) or (float(a.get("bc_coef") or 0) > 0 and bool(t.get("eval_ratchet")))
    return {"algo": a.get("name"), "behaviour_cloning": cloned,
            "curriculum": bool(cfg.get("env", {}).get("curriculum")), "hidden_sizes": a.get("hidden_sizes")}


def gdctl(*args: str, timeout: float = 300.0) -> dict[str, Any]:
    env = {**os.environ, "GDCTL_OWNER": OWNER}
    p = subprocess.run([sys.executable, "-m", "gdctl", *args], capture_output=True, text=True,
                       timeout=timeout, env=env, cwd=ROOT)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (p.stdout + p.stderr).strip()[-500:]}


def _must(out: dict[str, Any], what: str) -> dict[str, Any]:
    if not out.get("ok"):
        raise SystemExit(f"{what} failed: {out.get('code', '')} {out.get('error', out)}")
    return out


def _catalog_level(key: str | None) -> dict[str, Any]:
    if not key:
        return {}
    try:
        cat = json.loads((ROOT / "gdrl" / "dashboard" / "levels_catalog.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return next((lv for lv in cat.get("levels", []) if lv.get("key") == key), {})


def record_game(name: str, level: str, config: str, checkpoint: str, start_pos: int = -2,
                legacy_view: bool = False, title: str | None = None, difficulty: str | None = None,
                log=print) -> Path:
    """Open ``level`` in GD (takes the game lock), play one greedy episode and
    save the recording to ``logs/viz/<name>.rec.npz``."""
    from gdrl.config import load_config
    from gdrl.env.backends import wait_until_running
    from gdrl.env.factory import make_env
    from gdrl.tools.watch import keep_restored

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    cfg = load_config(config)
    lock = gdctl("lock", "acquire", "--note", f"nn_trace {name}", "--wait", "5")
    if not lock.get("ok") or not lock.get("held"):
        raise SystemExit(f"the game lock is held by someone else: {lock.get('owner', lock)}")
    stop = threading.Event()
    threading.Thread(target=keep_restored, args=(stop,), daemon=True).start()
    try:
        log(f"opening {level} ...")
        scene = gdctl("scene")
        if scene.get("ok") and not scene.get("in_level") and scene.get("main_layer") != "MenuLayer":
            # prepare-training waits for the main menu (or a level); another
            # menu (e.g. the Create page someone left open) would time it out.
            _must(gdctl("menu"), "menu")
        if level.startswith("saved:"):
            _must(gdctl("download-level", level[6:], "--no-song"), "download-level")
        _must(gdctl("prepare-training", "--level", level), "prepare-training")
        _must(gdctl("mod-setting", "legacy-view", "on" if legacy_view else "off"), "legacy-view")
        _must(gdctl("mod-setting", "watch-mode", "off"), "watch-mode")
        gdctl("unminimize")
        env_cfg = dict(cfg["env"])
        env_cfg.pop("curriculum", None)
        env = make_env(env_cfg)
        try:
            if not wait_until_running(env.unwrapped.backend, start_pos):
                raise SystemExit("the level never started running (paused? a popup?)")
            agent = load_agent(cfg, checkpoint, env.observation_space, env.action_space)
            log(f"playing {checkpoint} greedily from start_pos {start_pos} ...")
            states, actions, obs = run_episode(env, agent, start_pos, log)
        finally:
            env.close()
    finally:
        stop.set()
        for key in ("watch-mode", "legacy-view", "show-hitboxes"):
            gdctl("mod-setting", key, "off")
        gdctl("lock", "release")
    end = states[-1]
    log(f"{'COMPLETED' if end.completed else 'died'} at x = {end.player_x:.0f} ({100 * end.progress:.1f}%) "
        f"after {len(actions)} decisions")
    meta = {"name": name, "title": title or name, "difficulty": difficulty, "level": level,
            "config": str(config), "checkpoint": str(checkpoint), "start_pos": start_pos,
            "legacy_view": bool(legacy_view), "source": "game",
            "recorded_at": _stamp()}
    return save_recording(REC_DIR / f"{name}.rec.npz", states, actions, obs, meta)


def record_mock(name: str, config: str, checkpoint: str | None, level: int = 1, kind: str | None = None,
                max_steps: int | None = None, log=print) -> Path:
    """Like ``record_game`` on the in-process mock game (no GD, no lock)."""
    from gdrl.config import load_config
    from gdrl.env.factory import make_env

    cfg = load_config(config)
    env_cfg = dict(cfg["env"])
    env_cfg.pop("curriculum", None)
    env_cfg.update(backend="mock", level_ids=[int(level)])
    if kind:
        env_cfg["mock"] = {**env_cfg.get("mock", {}), "kind": kind}
    if max_steps:
        env_cfg["max_steps"] = int(max_steps)
    env = make_env(env_cfg)
    try:
        agent = load_agent(cfg, checkpoint, env.observation_space, env.action_space)
        states, actions, obs = run_episode(env, agent, -1, log)
    finally:
        env.close()
    meta = {"name": name, "title": f"Mock level {level}" + (f" ({kind})" if kind else ""), "difficulty": "mock",
            "level": f"mock:{level}", "config": str(config), "checkpoint": str(checkpoint) if checkpoint else None,
            "start_pos": -1, "source": "mock",
            "recorded_at": _stamp()}
    return save_recording(REC_DIR / f"{name}.rec.npz", states, actions, obs, meta)


def build_from_recording(rec: Path, out: Path | None = None, log=print) -> Path:
    from gdrl.config import load_config

    states, actions, obs, meta = load_recording(rec)
    cfg = load_config(meta["config"])
    meta = {**meta, "training": _training(cfg)}
    trace = build_trace(states, actions, cfg, meta.get("checkpoint"), meta, obs_recorded=obs)
    out = Path(out) if out else TRACE_DIR / f"{meta['name']}{SUFFIX}"
    write_trace(trace, out)
    m = trace["meta"]
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {m['decisions']} decisions, "
        f"{trace['objects']['count']} objects, greedy agreement {m['greedy_agreement']}/{m['decisions']}, "
        f"obs check {m['obs_check']}")
    return out


# --------------------------------------------------------------------------- CLI

def _presets(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m gdrl.tools.nn_trace", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="play a policy in GD and trace it")
    r.add_argument("preset", nargs="?", help="policy name from configs/watch.yaml (e.g. rk_beat)")
    r.add_argument("--presets", type=Path, default=ROOT / "configs" / "watch.yaml")
    r.add_argument("--name", help="trace name (default: the preset)")
    r.add_argument("--level", help="gdctl level spec, e.g. saved:146399247 (instead of a preset)")
    r.add_argument("--config")
    r.add_argument("--checkpoint")
    r.add_argument("--title")
    r.add_argument("--start-pos", type=int)
    r.add_argument("--legacy-view", action="store_true")
    r.add_argument("--export", action="store_true", help="also write the standalone HTML")
    b = sub.add_parser("build", help="(re)build a trace from a recording (.rec.npz)")
    b.add_argument("recording", type=Path)
    b.add_argument("--out", type=Path)
    b.add_argument("--export", action="store_true")
    e = sub.add_parser("export", help="write a standalone HTML with the trace embedded")
    e.add_argument("trace", type=Path)
    e.add_argument("--out", type=Path)
    m = sub.add_parser("mock", help="trace a policy on a mock level (no game needed)")
    m.add_argument("--config", required=True)
    m.add_argument("--checkpoint")
    m.add_argument("--level", type=int, default=1)
    m.add_argument("--kind", choices=["simple", "demon"])
    m.add_argument("--max-steps", type=int)
    m.add_argument("--name", default="mock")
    m.add_argument("--export", action="store_true")
    sub.add_parser("list", help="list the traces in results/viz")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for t in list_traces():
            result = "completed" if t["completed"] else f"died at {100 * (t['progress'] or 0):.1f}%"
            print(f"{t['name']:14s} {t['title']:24s} {t['decisions']:>6} decisions  {result:16s} "
                  f"{t['bytes'] / 1e6:5.2f} MB  {t['file']}")
        return
    if a.cmd == "export":
        out = export_html(a.trace, a.out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
        return
    if a.cmd == "build":
        out = build_from_recording(a.recording, a.out)
        if a.export:
            h = export_html(out)
            print(f"wrote {h} ({h.stat().st_size / 1e6:.2f} MB)")
        return
    if a.cmd == "mock":
        rec = record_mock(a.name, a.config, a.checkpoint, a.level, a.kind, a.max_steps)
        out = build_from_recording(rec)
        if a.export:
            h = export_html(out)
            print(f"wrote {h} ({h.stat().st_size / 1e6:.2f} MB)")
        return
    # record
    if a.preset:
        presets = _presets(a.presets)
        if a.preset not in presets:
            raise SystemExit(f"unknown preset {a.preset!r}; known: {', '.join(presets)}")
        p = presets[a.preset]
        cat = _catalog_level(p.get("dashboard_level"))
        name = a.name or a.preset
        rec = record_game(name, a.level or p["level"], a.config or p["config"], a.checkpoint or p["checkpoint"],
                          a.start_pos if a.start_pos is not None else int(p.get("start_pos", -2)),
                          a.legacy_view or bool(p.get("legacy_view")), a.title or cat.get("name") or p.get("name"),
                          cat.get("difficulty"))
    else:
        if not (a.level and a.config and a.checkpoint and a.name):
            raise SystemExit("record needs a preset, or --level, --config, --checkpoint and --name")
        rec = record_game(a.name, a.level, a.config, a.checkpoint,
                          a.start_pos if a.start_pos is not None else -2, a.legacy_view, a.title)
    out = build_from_recording(rec)
    if a.export:
        h = export_html(out)
        print(f"wrote {h} ({h.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
