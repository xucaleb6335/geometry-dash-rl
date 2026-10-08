"""Experiment configs: one YAML file per experiment, plus command-line overrides.

    python -m gdrl.train --config configs/mock_ppo.yaml --set train.total_steps=50000 --set seed=2
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "run",
    "seed": 1,
    "env": {
        "id": "gd",
        "backend": "mock",
        "level_ids": [1],
        "frame_skip": 4,
        "max_steps": 5000,
        "encoder": {"name": "debug"},
        "reward": {"name": "constant"},
    },
    "agent": {"name": "random"},
    "train": {
        "num_envs": 1,
        "total_steps": 100_000,
        "checkpoint_every": 50,  # updates
        "runs_dir": "runs",
        "tensorboard": True,
        "torch_threads": 1,  # null = let torch use every core (better for a big CNN on CPU)
    },
    "eval": {
        "episodes": 20,
        "sticky_prob": 0.0,
        "level_ids": None,  # None = the env's level_ids
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict[str, Any], assignment: str) -> None:
    """Apply ``a.b.c=value`` in place. The value is parsed as YAML (numbers, lists, bools)."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key.path=value, got {assignment!r}")
    path, raw = assignment.split("=", 1)
    keys = path.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None, overrides: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    user: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    # An env that is not GD (e.g. CartPole) ignores the GD-only defaults.
    if cfg["env"].get("id", "gd") != "gd":
        cfg["env"] = {k: v for k, v in cfg["env"].items() if k in ("id", "kwargs")}
    for assignment in overrides:
        apply_override(cfg, assignment)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
