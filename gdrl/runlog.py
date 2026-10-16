"""Per-run JSONL logs that a dashboard can read (format: DEMON_PLAN.md, "Run logs").

    runs/<run>/meta.json          one object: level, level_id, game, difficulty, length, agent, ...
    runs/<run>/episodes.jsonl     one line per episode (or search rollout)
    runs/<run>/generations.jsonl  one line per search generation / segment expansion

Every line has ``t`` (unix seconds) and ``step`` (cumulative env steps:
physics ticks for search, decisions for PPO; ``step_unit`` in meta.json says
which). Lines are flushed as they are written, so a live reader sees them.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

EPISODE_FIELDS = ("t", "step", "level", "segment", "start_x", "progress", "completed", "death_x", "length",
                  "return", "agent")
GENERATION_FIELDS = ("t", "step", "generation", "segment", "best_fitness", "mean_fitness", "best_progress", "agent")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):  # numpy scalars
        return _clean(value.item())
    return value


class RunLog:
    """Writes meta.json, episodes.jsonl and generations.jsonl into ``run_dir``."""

    def __init__(self, run_dir: str | Path, meta: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._episodes = open(self.run_dir / "episodes.jsonl", "a", encoding="utf-8")
        self._generations = open(self.run_dir / "generations.jsonl", "a", encoding="utf-8")
        self.meta: dict[str, Any] = {}
        if meta:
            self.write_meta(meta)

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta = {"created": time.time(), **self.meta, **meta}
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")

    def episode(self, **fields: Any) -> None:
        self._write(self._episodes, fields)

    def generation(self, **fields: Any) -> None:
        self._write(self._generations, fields)

    def _write(self, f, fields: dict[str, Any]) -> None:
        line = {"t": round(time.time(), 3), **{k: _clean(v) for k, v in fields.items()}}
        f.write(json.dumps(line) + "\n")
        f.flush()

    def close(self) -> None:
        self._episodes.close()
        self._generations.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
