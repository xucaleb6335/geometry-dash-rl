"""Incremental aggregation of everything the dashboard shows.

Sources (all read-only, all optional, parsed defensively):

    runs/**/<run>/metrics.csv         long format ``step,key,value`` (PPO training)
    runs/**/<run>/{config.yaml,meta.json,result.json,eval.json,checkpoints/}
    runs/**/<run>/episodes.jsonl      gdrl/runlog.py (search, distill)
    runs/**/<run>/generations.jsonl   gdrl/runlog.py (search)
    <env.curriculum.log_path>         curriculum episode csv (shared between resumed runs)
    <env.curriculum.starts>           start list yaml
    holistic_progress/level_beats.csv the beat ledger
    results/**/*.json                 M8 comparison files
    logs/*.log                        run status hints (UTF-8 / UTF-16 / mixed)

Files are re-read only when their (size, mtime) changed; append-only files
(metrics.csv, *.jsonl, the curriculum csv) are read from the last offset.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

PKG = Path(__file__).resolve().parent
RUN_NAME = re.compile(r"^(\d{8}-\d{6})_(.+?)(?:_s(\d+))?$")
LIVE_SECONDS = 180.0
MAX_POINTS = 500


# --------------------------------------------------------------------------- helpers

def clean(o: Any) -> Any:
    """Make an object JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def read_yaml(path: Path) -> Any:
    if yaml is None:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def decode_log_bytes(b: bytes) -> str:
    """Decode a log that may be UTF-8, UTF-16 (BOM) or a mix of both."""
    if not b:
        return ""
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        b = b[2:]
    # Mixed UTF-16LE / UTF-8: UTF-16LE ASCII text has NUL after every char.
    if b"\x00" in b:
        b = b.replace(b"\xff\xfe", b"").replace(b"\x00", b"")
    return b.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def tail_bytes(path: Path, n: int) -> bytes:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read()
    except OSError:
        return b""


def head_bytes(path: Path, n: int) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(n)
    except OSError:
        return b""


def parse_local_time(s: str) -> float | None:
    """'2026-10-11 00:28:39 PDT' / '... Pacific Daylight Time' / ISO -> epoch (local clock)."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?", s or "")
    if not m:
        return None
    try:
        y, mo, d, h, mi, se = (int(x) if x else 0 for x in m.groups())
        return datetime(y, mo, d, h, mi, se).timestamp()
    except ValueError:
        return None


def fnum(x: Any) -> float | None:
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def downsample(xs: np.ndarray, ys: np.ndarray, n: int = MAX_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Bucket-average down to ~n points, keeping the final point exact."""
    m = len(xs)
    if m <= n:
        return xs, ys
    edges = np.linspace(0, m, n, endpoint=False).astype(int)
    edges = np.unique(edges)
    xo, yo = [], []
    for a, b in zip(edges, list(edges[1:]) + [m - 1]):
        if b <= a:
            b = a + 1
        xo.append(xs[a:b].mean())
        yo.append(ys[a:b].mean())
    xo.append(xs[-1])
    yo.append(ys[-1])
    return np.array(xo), np.array(yo)


# --------------------------------------------------------------------------- incremental files

class Appendable:
    """Reads an append-only text file from the last complete line."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.sig: tuple[int, float] | None = None
        self.header_done = False

    def _new_lines(self) -> list[str]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        if st.st_size < self.offset:  # truncated / replaced
            self.reset()
        if st.st_size == self.offset:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return []
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return []
        self.offset += cut + 1
        self.sig = (st.st_size, st.st_mtime)
        return chunk[: cut + 1].decode("utf-8", errors="replace").splitlines()

    def reset(self) -> None:
        self.offset = 0
        self.header_done = False


class MetricsFile(Appendable):
    def __init__(self, path: Path):
        super().__init__(path)
        self.series: dict[str, tuple[list[float], list[float]]] = {}
        self.version = 0

    def reset(self) -> None:
        super().reset()
        self.series = {}

    def update(self) -> None:
        lines = self._new_lines()
        if not lines:
            return
        for ln in lines:
            parts = ln.split(",")
            if len(parts) != 3 or parts[0] == "step":
                continue
            try:
                step, val = float(parts[0]), float(parts[2])
            except ValueError:
                continue
            xs, ys = self.series.setdefault(parts[1], ([], []))
            xs.append(step)
            ys.append(val)
        self.version += 1


class JsonlFile(Appendable):
    def __init__(self, path: Path):
        super().__init__(path)
        self.rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.rows = []

    def update(self) -> None:
        for ln in self._new_lines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except ValueError:
                continue
            if isinstance(o, dict):
                self.rows.append(o)


class CurriculumCsv(Appendable):
    """time,start,start_x,final_x,progress,length,completed,reached_next"""

    def __init__(self, path: Path):
        super().__init__(path)
        self.rows: list[tuple[float, int, float | None, float, float, float, int, int]] = []

    def reset(self) -> None:
        super().reset()
        self.rows = []

    def update(self) -> None:
        for ln in self._new_lines():
            p = ln.split(",")
            if len(p) < 8 or p[0] == "time":
                continue
            try:
                self.rows.append((float(p[0]), int(float(p[1])), fnum(p[2]), float(p[3]), float(p[4]),
                                  float(p[5]), int(float(p[6])), int(float(p[7]))))
            except ValueError:
                continue


class Cached:
    """Whole-file cache keyed by (size, mtime_ns)."""

    def __init__(self):
        self.items: dict[str, tuple[tuple[int, int], Any]] = {}

    def get(self, path: Path, loader) -> Any:
        try:
            st = path.stat()
        except OSError:
            self.items.pop(str(path), None)
            return None
        sig = (st.st_size, st.st_mtime_ns)
        hit = self.items.get(str(path))
        if hit and hit[0] == sig:
            return hit[1]
        try:
            val = loader(path)
        except Exception:
            val = None
        self.items[str(path)] = (sig, val)
        return val


# --------------------------------------------------------------------------- catalog

MOCK_GROUPS = ("Mock", "Mock demons")


def is_mock_key(key: str) -> bool:
    return key.startswith(("mock:", "mock-demon:", "other:"))


class Catalog:
    def __init__(self, path: Path | None = None):
        data = read_json(path or PKG / "levels_catalog.json") or {}
        self.levels: list[dict[str, Any]] = data.get("levels", [])
        self.groups: list[str] = data.get("groups", [])
        self.overrides: dict[str, str] = data.get("run_level_overrides", {})
        self.by_key = {lv["key"]: lv for lv in self.levels}
        self.alias: dict[str, str] = {}
        for lv in self.levels:
            self.alias[lv["key"].lower()] = lv["key"]
            for a in lv.get("aliases", []):
                self.alias[str(a).lower()] = lv["key"]

    def resolve(self, text: Any) -> str | None:
        if text is None:
            return None
        t = str(text).strip().lower()
        if t in self.alias:
            return self.alias[t]
        m = re.fullmatch(r"mock[ _-]?demon[ _:-]*(\d+)", t)
        if m:
            return f"mock-demon:{m.group(1)}"
        m = re.fullmatch(r"mock[ _:-]*(?:level)?[ _:-]*(\d+)", t)
        if m:
            return f"mock:{m.group(1)}"
        m = re.fullmatch(r"main[ _:-]*(\d+)", t)
        if m:
            return f"main:{m.group(1)}"
        return None

    def ensure(self, key: str, name: str | None = None, game: str = "real") -> dict[str, Any]:
        if key not in self.by_key:
            lv = {"key": key, "name": name or key, "group": "Other", "difficulty": "unknown", "game": game,
                  "aliases": []}
            self.by_key[key] = lv
            self.levels.append(lv)
            if "Other" not in self.groups:
                self.groups.append("Other")
        return self.by_key[key]


# --------------------------------------------------------------------------- the store

class Store:
    def __init__(self, root: Path, runs_dir: Path | None = None, demo: bool = False,
                 extra_runs: list[Path] | None = None, include_mock: bool = True):
        self.root = Path(root)
        self.runs_dir = Path(runs_dir) if runs_dir else self.root / "runs"
        self.demo = demo
        self.demo_dir = PKG / "demo"
        # include_mock=False: only real-game levels (mock levels, mock runs,
        # CartPole and the mock agent comparison are left out).
        self.include_mock = include_mock
        self.catalog = Catalog()
        self.lock = threading.RLock()
        self.metrics: dict[Path, MetricsFile] = {}
        self.jsonl: dict[Path, JsonlFile] = {}
        self.curcsv: dict[Path, CurriculumCsv] = {}
        self.cache = Cached()
        self.extra_runs = extra_runs or []
        self._scan_at = 0.0
        self._run_dirs: list[tuple[Path, bool]] = []
        self._state: dict[str, Any] | None = None
        self._state_sig: Any = None
        self._detail: dict[str, tuple[Any, Any]] = {}
        self._runs_ctx: list[dict[str, Any]] = []

    # -- discovery ---------------------------------------------------------

    def _scan_runs(self) -> list[tuple[Path, bool]]:
        now = time.time()
        if self._run_dirs and now - self._scan_at < 2.0:
            return self._run_dirs
        found: list[tuple[Path, bool]] = []
        roots = [(self.runs_dir, False)] + [(p, False) for p in self.extra_runs]
        if self.demo:
            roots.append((self.demo_dir / "runs", True))
        for base, is_demo in roots:
            if not base.is_dir():
                continue
            stack = [(base, 0)]
            while stack:
                d, depth = stack.pop()
                try:
                    entries = list(d.iterdir())
                except OSError:
                    continue
                names = {e.name for e in entries}
                if depth > 0 and names & {"metrics.csv", "meta.json", "config.yaml", "episodes.jsonl"}:
                    found.append((d, is_demo))
                    continue
                if depth < 3:
                    for e in entries:
                        if e.is_dir() and e.name not in ("checkpoints", "__pycache__"):
                            stack.append((e, depth + 1))
        found.sort(key=lambda t: t[0].name)
        self._run_dirs = found
        self._scan_at = now
        return found

    def run_id(self, d: Path, is_demo: bool) -> str:
        if is_demo:
            return "demo/" + d.name
        try:
            return d.relative_to(self.runs_dir).as_posix()
        except ValueError:
            return d.name

    # -- file accessors ----------------------------------------------------

    def _metrics(self, d: Path) -> MetricsFile | None:
        p = d / "metrics.csv"
        if not p.exists():
            return None
        mf = self.metrics.get(p)
        if mf is None:
            mf = self.metrics[p] = MetricsFile(p)
        mf.update()
        return mf

    def _jsonl(self, d: Path, name: str) -> JsonlFile | None:
        p = d / name
        if not p.exists():
            return None
        jf = self.jsonl.get(p)
        if jf is None:
            jf = self.jsonl[p] = JsonlFile(p)
        jf.update()
        return jf

    def _curcsv(self, p: Path) -> CurriculumCsv | None:
        if not p.exists():
            return None
        cf = self.curcsv.get(p)
        if cf is None:
            cf = self.curcsv[p] = CurriculumCsv(p)
        cf.update()
        return cf

    # -- logs --------------------------------------------------------------

    def _log_index(self) -> dict[str, dict[str, Any]]:
        """run folder name -> {log, tail_lines, bad(after mention)} using logs/*.log."""
        out: dict[str, dict[str, Any]] = {}
        dirs = [self.root / "logs", self.root / "results"]
        if self.demo:
            dirs.append(self.demo_dir / "logs")
        for ld in dirs:
            if not ld.is_dir():
                continue
            for p in sorted(ld.glob("*.log")):
                info = self.cache.get(p, self._load_log)
                if not info:
                    continue
                for name in info["runs"]:
                    out[name] = {**info, "log": p.name}
        return out

    @staticmethod
    def _load_log(p: Path) -> dict[str, Any]:
        head = decode_log_bytes(head_bytes(p, 4096))
        tail = decode_log_bytes(tail_bytes(p, 120_000))
        runs = set(re.findall(r"runs[\\/](?:[\w\-.]+[\\/])?(\d{8}-\d{6}_[\w\-.]+?)(?=[\\/\s\"']|$)", head + "\n" + tail))
        lines = [ln for ln in tail.split("\n") if ln.strip()]
        bad = [ln for ln in lines[-80:] if re.search(r"Traceback|Error|failures so far [1-9]|exited with|crash", ln, re.I)]
        last_run = None
        for m in re.finditer(r"runs[\\/](?:[\w\-.]+[\\/])?(\d{8}-\d{6}_[\w\-.]+?)(?=[\\/\s\"']|$)", tail):
            last_run = m.group(1)
        return {"runs": sorted(runs), "tail": [ln[:300] for ln in lines[-4:]], "bad": [b[:200] for b in bad[-3:]],
                "last_run": last_run, "mtime": p.stat().st_mtime}

    # -- run summaries -----------------------------------------------------

    def _config(self, d: Path) -> dict[str, Any]:
        cfg = self.cache.get(d / "config.yaml", read_yaml)
        return cfg if isinstance(cfg, dict) else {}

    def _meta(self, d: Path) -> dict[str, Any]:
        m = self.cache.get(d / "meta.json", read_json)
        return m if isinstance(m, dict) else {}

    def _resolve_levels(self, name: str, cfg: dict[str, Any], meta: dict[str, Any], run_dir: Path | None = None) -> tuple[list[str], str]:
        """-> (level keys, human label)."""
        cat = self.catalog
        for k, v in cat.overrides.items():
            if k in name and not meta.get("level"):
                return [v], cat.by_key.get(v, {}).get("name", v)
        if meta.get("level"):
            key = cat.resolve(meta["level"])
            if key is None and meta.get("level_id") is not None and meta.get("game") == "mock":
                key = f"mock:{meta['level_id']}"
            if key is None:
                key = f"other:{meta['level']}"
                cat.ensure(key, str(meta["level"]), str(meta.get("game", "real")))
            return [key], cat.by_key[key]["name"]
        env = cfg.get("env") or {}
        if not env:
            return [], "-"
        eid = str(env.get("id", ""))
        if "cartpole" in eid.lower() or "cartpole" in name.lower():
            key = "other:cartpole"
            cat.ensure(key, "CartPole (sanity)", "toy")
            cat.by_key[key]["group"] = "Sanity"
            cat.by_key[key]["difficulty"] = "toy"
            if "Sanity" not in cat.groups:
                cat.groups.append("Sanity")
            return [key], "CartPole (sanity)"
        backend = str(env.get("backend", ""))
        ids = env.get("level_ids") or []
        mockcfg = env.get("mock") or {}
        cur = env.get("curriculum") or {}
        if cur.get("starts"):
            sy = self.cache.get(self._path(cur["starts"], run_dir), read_yaml)
            if isinstance(sy, dict) and sy.get("level"):
                key = cat.resolve(sy["level"])
                if key:
                    return [key], cat.by_key[key]["name"]
        if backend == "mock":
            kind = "mock-demon" if str(mockcfg.get("kind", "")) == "demon" else "mock"
            keys = [f"{kind}:{i}" for i in ids] or [f"{kind}:1"]
            for k in keys:
                cat.ensure(k, k.replace(":", " "), "mock")
            return keys, ", ".join(str(i) for i in ids) and f"{kind} {', '.join(str(i) for i in ids)}"
        key = "gd:open"
        cat.ensure(key, "Real game (open level)", "real")
        return [key], "Real game (open level)"

    def _path(self, p: str, run_dir: Path | None) -> Path:
        pp = Path(str(p))
        if pp.is_absolute():
            return pp
        for base in ([run_dir] if run_dir else []) + [self.root]:
            if (base / pp).exists():
                return base / pp
        return self.root / pp

    def _agent_name(self, cfg: dict[str, Any], meta: dict[str, Any], name: str) -> str:
        if meta.get("agent"):
            return str(meta["agent"])
        a = (cfg.get("agent") or {}).get("name")
        enc = ((cfg.get("env") or {}).get("encoder") or {}).get("name")
        if a:
            a = str(a)
            if a == "ppo" and enc:
                return f"ppo-{str(enc).replace('_', '-')}"
            return a
        if "distill" in name:
            return "distill"
        if "search" in name:
            return "checkpoint-search"
        return "unknown"

    def runs(self) -> list[dict[str, Any]]:
        logs = self._log_index()
        out: list[dict[str, Any]] = []
        now = time.time()
        for d, is_demo in self._scan_runs():
            rid = self.run_id(d, is_demo)
            m = RUN_NAME.match(d.name)
            created_name = None
            if m:
                try:
                    created_name = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp()
                except ValueError:
                    pass
            base = m.group(2) if m else d.name
            seed = int(m.group(3)) if m and m.group(3) else None
            cfg = self._config(d)
            meta = self._meta(d)
            mf = self._metrics(d)
            eps = self._jsonl(d, "episodes.jsonl")
            gens = self._jsonl(d, "generations.jsonl")
            result = self.cache.get(d / "result.json", read_json)
            evalj = self.cache.get(d / "eval.json", read_json)
            created = meta.get("created") or created_name
            if created is None:
                try:
                    created = d.stat().st_ctime
                except OSError:
                    created = now
            mtimes = []
            for fn in ("metrics.csv", "episodes.jsonl", "generations.jsonl"):
                try:
                    mtimes.append((d / fn).stat().st_mtime)
                except OSError:
                    pass
            last_update = max(mtimes) if mtimes else created
            levels, label = self._resolve_levels(base, cfg, meta, d)
            agent = self._agent_name(cfg, meta, base)
            tr = cfg.get("train") or {}
            env = cfg.get("env") or {}
            ag = cfg.get("agent") or {}
            per_update = (int(ag.get("rollout_steps", 0) or 0) * int(tr.get("num_envs", 1) or 1)) or None
            total_steps = tr.get("total_steps")
            steps = 0.0
            first_step = 0.0
            updates = None
            ser = mf.series if mf else {}
            if ser:
                allx = ser.get("perf/steps_per_second") or next(iter(ser.values()))
                if allx[0]:
                    steps = allx[0][-1]
                    first_step = allx[0][0] if allx[0][0] > 20_000 else 0.0
                if per_update:
                    updates = round(steps / per_update)
            elif eps and eps.rows:
                steps = float(eps.rows[-1].get("step") or 0)
            elif gens and gens.rows:
                steps = float(gens.rows[-1].get("step") or 0)
            if isinstance(result, dict) and result.get("env_ticks"):
                steps = max(steps, float(result["env_ticks"]))
            cdir = d / "checkpoints"
            has_final = (cdir / "final.pt").exists() if cdir.exists() else False
            resume = str(tr.get("resume") or "")
            kind = "search" if (gens and gens.rows) or "search" in agent else ("ppo" if ser else "other")
            if "distill" in base:
                kind = "distill"
            # progress / return summaries
            def last(key):
                s = ser.get(key)
                return s[1][-1] if s and s[1] else None
            def best(key):
                s = ser.get(key)
                return max(s[1]) if s and s[1] else None
            sps = last("perf/steps_per_second")
            wall = 0.0
            if ser.get("perf/steps_per_second"):
                sx, sy = ser["perf/steps_per_second"]
                prev = first_step
                for x, v in zip(sx, sy):
                    if v and v > 0:
                        wall += (x - prev) / v
                    prev = x
            elif isinstance(result, dict) and result.get("wall_s"):
                wall = float(result["wall_s"])
            elif eps and len(eps.rows) > 1:
                wall = float(eps.rows[-1]["t"]) - float(eps.rows[0]["t"])
            ep_count = 0
            if ser.get("episode/count"):
                ep_count = int(sum(ser["episode/count"][1]))
            elif eps:
                ep_count = len(eps.rows)
            best_prog = None
            for cand in (best("eval/progress"), best("episode/progress_mean")):
                if cand is not None:
                    best_prog = max(best_prog or 0.0, cand)
            if eps and eps.rows:
                mp = max((fnum(r.get("progress")) or 0.0) for r in eps.rows)
                best_prog = max(best_prog or 0.0, mp)
            if gens and gens.rows:
                mp = max((fnum(r.get("best_progress")) or 0.0) for r in gens.rows)
                best_prog = max(best_prog or 0.0, mp)
            if isinstance(result, dict) and fnum(result.get("best_progress")) is not None:
                best_prog = max(best_prog or 0.0, float(result["best_progress"]))
            completed_frac = last("episode/completed_mean")
            solved = bool(isinstance(result, dict) and result.get("solved"))
            finished = bool(has_final or solved or evalj is not None or
                            (total_steps and steps >= 0.995 * float(total_steps)) or
                            (isinstance(result, dict) and result.get("replay_completed")))
            spark = []
            if ser.get("episode/progress_mean"):
                _, ys = downsample(np.array(ser["episode/progress_mean"][0]), np.array(ser["episode/progress_mean"][1]), 30)
                spark = [round(float(v), 4) for v in ys]
            log = logs.get(d.name)
            out.append({
                "id": rid, "dir": str(d), "demo": is_demo, "base": base, "name": base, "seed": seed,
                "kind": kind, "agent": agent, "backend": env.get("backend") or ("mock" if "mock" in base else None),
                "levels": levels, "level_label": label, "created": created, "last_update": last_update,
                "steps": steps, "first_step": first_step, "updates": updates, "total_steps": total_steps,
                "per_update": per_update, "wall_s": wall, "episodes": ep_count, "sps": sps,
                "best_progress": best_prog, "last_progress": last("episode/progress_mean"),
                "last_eval_progress": last("eval/progress"), "last_return": last("episode/return_mean"),
                "last_completed": completed_frac, "solved": solved, "finished": finished,
                "resume": resume, "spark": spark, "log": (log or {}).get("log"),
                "log_tail": (log or {}).get("tail", []), "log_bad": (log or {}).get("bad", []),
                "log_mtime": (log or {}).get("mtime"),
                "has": {"metrics": bool(ser), "episodes": bool(eps and eps.rows), "generations": bool(gens and gens.rows),
                        "curriculum": bool(env.get("curriculum"))},
                "set": None,
            })
        # status (needs the family relation)
        by_base: dict[str, list[dict[str, Any]]] = {}
        for r in out:
            by_base.setdefault(r["base"], []).append(r)
        for fam in by_base.values():
            fam.sort(key=lambda r: r["created"] or 0)
            for i, r in enumerate(fam):
                later = fam[i + 1:]
                resumed_later = any(l["resume"] and Path(r["id"]).name in l["resume"].replace("\\", "/") for l in later)
                age = now - (r["last_update"] or 0)
                if age < LIVE_SECONDS and not r["finished"] and not later:
                    st = "running"
                elif r["finished"]:
                    st = "finished"
                elif resumed_later or (r["log_bad"] and r["log_mtime"] and r["log_mtime"] >= (r["last_update"] or 0) - 5):
                    st = "crashed"
                elif later:
                    st = "crashed" if r["resume"] or resumed_later else "stopped"
                else:
                    st = "stopped"
                r["status"] = st
                r["family"] = r["base"]
        return out

    # -- ledger / results --------------------------------------------------

    def ledger(self) -> list[dict[str, Any]]:
        paths = [self.root / "holistic_progress" / "level_beats.csv"]
        if self.demo:
            paths.append(self.demo_dir / "level_beats.csv")
        rows: list[dict[str, Any]] = []
        for p in paths:
            rs = self.cache.get(p, self._load_ledger)
            if rs:
                rows.extend(dict(r, demo=(p.parent == self.demo_dir)) for r in rs)
        rows.sort(key=lambda r: r.get("ts") or 0)
        return rows

    def _load_ledger(self, p: Path) -> list[dict[str, Any]]:
        out = []
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ts = parse_local_time(row.get("beaten_at_pacific", ""))
                lvl = (row.get("level") or "").strip()
                keys: list[str] = []
                partial = False
                key = self.catalog.resolve(lvl)
                if key:
                    keys = [key]
                else:
                    m = re.search(r"levels?\s+(\d+)\s*-\s*(\d+)", lvl)
                    if m and "mock" in lvl.lower():
                        keys = [f"mock:{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1)]
                        partial = True
                    else:
                        lid = (row.get("level_id") or "").strip()
                        game = (row.get("game") or "").strip()
                        k2 = f"{'mock' if game == 'mock' else 'gd'}:{lid}" if lid.isdigit() else None
                        k3 = self.catalog.resolve(lid)
                        keys = [k3 or k2 or f"other:{lvl}"]
                out.append({
                    "ts": ts, "when": row.get("beaten_at_pacific"), "level": lvl, "level_id": row.get("level_id"),
                    "game": row.get("game"), "difficulty": row.get("difficulty"), "agent": row.get("agent"),
                    "attempts": fnum(row.get("attempts")), "generations": fnum(row.get("generations")),
                    "env_steps": fnum(row.get("env_steps")), "wall_hours": fnum(row.get("wall_hours")),
                    "run": row.get("run"), "notes": row.get("notes"), "keys": keys, "partial": partial,
                })
        return out

    def results(self) -> list[dict[str, Any]]:
        files: list[Path] = []
        base = self.root / "results"
        if base.is_dir():
            files += sorted(base.rglob("*.json"))
        if self.demo and (self.demo_dir / "results").is_dir():
            files += sorted((self.demo_dir / "results").rglob("*.json"))
        out = []
        for p in files:
            r = self.cache.get(p, lambda q: self._load_result(q))
            if r:
                out.append(r)
        return out

    def _load_result(self, p: Path) -> dict[str, Any] | None:
        d = read_json(p)
        if not isinstance(d, dict) or "agent" not in d:
            return None
        train = [str(x) for x in d.get("train_levels") or []]
        held = [str(x) for x in d.get("heldout_levels") or []]
        rec: dict[str, Any] = {"file": p.name, "agent": str(d["agent"]), "source": str(d.get("source", "")),
                               "train": train, "heldout": held, "mtime": p.stat().st_mtime,
                               "greedy": {}, "sticky": {}, "solved": {}, "extra": {}}
        res = d.get("results")
        if isinstance(res, dict):
            for mode in ("greedy", "sticky"):
                lv = ((res.get(mode) or {}).get("levels")) or {}
                for k, v in lv.items():
                    if isinstance(v, dict):
                        rec[mode][str(k)] = {"completion": fnum(v.get("completion_rate")),
                                             "progress": fnum(v.get("progress_mean")),
                                             "episodes": v.get("episodes")}
        pl = d.get("per_level")
        if isinstance(pl, dict):
            for k, v in pl.items():
                if isinstance(v, dict):
                    rec["solved"][str(k)] = {"solved": bool(v.get("solved")), "generations": fnum(v.get("generations")),
                                             "episodes": fnum(v.get("episodes")), "seconds": fnum(v.get("seconds")),
                                             "best_fitness": fnum(v.get("best_fitness"))}
            rec["extra"]["population"] = d.get("population")
            sk = d.get("sticky")
            if isinstance(sk, dict):
                for k, v in sk.items():
                    if isinstance(v, dict) and "completion_rate" in v:
                        rec["sticky"][str(k)] = {"completion": fnum(v.get("completion_rate")),
                                                 "progress": fnum(v.get("progress_mean")),
                                                 "episodes": v.get("episodes")}
            rp = d.get("replay_progress")
            if isinstance(rp, dict):
                flat = [float(val) for a, row in rp.items() if isinstance(row, dict)
                        for b, val in row.items() if fnum(val) is not None and str(a) != str(b)]
                if flat:
                    rec["extra"]["greedy_transfer"] = sum(flat) / len(flat)
        if not rec["greedy"] and not rec["sticky"] and not rec["solved"]:
            return None
        return rec

    def compare(self) -> dict[str, Any]:
        recs = self.results()
        agents: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            agents.setdefault(r["agent"], []).append(r)

        def mean_c(rec, mode, ids, field="completion"):
            vals = [rec[mode][i][field] for i in ids if i in rec[mode] and rec[mode][i].get(field) is not None]
            return sum(vals) / len(vals) if vals else None

        rows = []
        for agent, rs in sorted(agents.items()):
            row: dict[str, Any] = {"agent": agent, "n": len(rs), "files": [r["file"] for r in rs]}
            for mode in ("greedy", "sticky"):
                for part in ("train", "heldout"):
                    vals = []
                    for r in rs:
                        v = mean_c(r, mode, r[part])
                        if v is not None:
                            vals.append(v)
                    row[f"{mode}_{part}"] = (sum(vals) / len(vals)) if vals else None
                    row[f"{mode}_{part}_sd"] = statistics.pstdev(vals) if len(vals) > 1 else None
                    pv = [mean_c(r, mode, r[part], "progress") for r in rs]
                    pv = [x for x in pv if x is not None]
                    row[f"{mode}_{part}_progress"] = (sum(pv) / len(pv)) if pv else None
            solved = [s for r in rs for s in r["solved"].values()]
            if solved:
                row["solved"] = sum(1 for s in solved if s["solved"])
                row["solved_of"] = len(solved)
                gens = [s["generations"] for s in solved if s["solved"] and s["generations"] is not None]
                row["gens_mean"] = sum(gens) / len(gens) if gens else None
                row["episodes_mean"] = (sum(s["episodes"] or 0 for s in solved) / len(solved)) if solved else None
            for k in ("greedy_transfer", "sticky_transfer"):
                v = [r["extra"].get(k) for r in rs if r["extra"].get(k) is not None]
                if v:
                    row[k] = sum(v) / len(v)
            per_level: dict[str, Any] = {}
            for r in rs:
                for mode in ("greedy", "sticky"):
                    for lid, v in r[mode].items():
                        per_level.setdefault(lid, {}).setdefault(mode, []).append(v["completion"])
                for lid, v in r["solved"].items():
                    per_level.setdefault(lid, {})["solved"] = v
            for lid, v in per_level.items():
                for mode in ("greedy", "sticky"):
                    xs = [x for x in v.get(mode, []) if x is not None]
                    v[mode] = sum(xs) / len(xs) if xs else None
            row["per_level"] = per_level
            row["is_train"] = {lid: (lid in rs[0]["train"]) for lid in per_level}
            rows.append(row)
        return {"rows": rows, "n_files": len(recs)}

    # -- state -------------------------------------------------------------

    def _signature(self) -> Any:
        sig = []
        for d, _ in self._scan_runs():
            for fn in ("metrics.csv", "episodes.jsonl", "generations.jsonl", "result.json", "meta.json"):
                try:
                    st = (d / fn).stat()
                    sig.append((str(d), fn, st.st_size, st.st_mtime_ns))
                except OSError:
                    pass
        for p in [self.root / "holistic_progress" / "level_beats.csv"]:
            try:
                st = p.stat()
                sig.append((str(p), st.st_size, st.st_mtime_ns))
            except OSError:
                pass
        for sub in ("logs", "results"):
            base = self.root / sub
            if base.is_dir():
                for p in list(base.glob("*.log")) + list(base.rglob("*.json")):
                    try:
                        st = p.stat()
                        sig.append((p.name, st.st_size, st.st_mtime_ns))
                    except OSError:
                        pass
        for p in self.curcsv:
            try:
                st = p.stat()
                sig.append((str(p), st.st_size, st.st_mtime_ns))
            except OSError:
                pass
        # live status flips with time even if no file changes
        sig.append(int(time.time() // 30))
        return tuple(sig)

    def state(self) -> dict[str, Any]:
        with self.lock:
            sig = self._signature()
            if self._state is not None and sig == self._state_sig:
                return self._state
            runs = self.runs()
            if not self.include_mock:
                runs = [r for r in runs if not all(is_mock_key(k) for k in r["levels"]) and r.get("backend") != "mock"]
            self._runs_ctx = runs
            for r in runs:
                if r["has"].get("curriculum"):
                    rows = self._curriculum_rows(r)
                    p0 = [x[4] for x in rows if x[1] == 0]
                    ev = (self._metrics(Path(r["dir"])).series.get("eval/progress") or ([], []))[1]
                    cand = max(p0 + list(ev), default=None)
                    r["best_progress"] = cand
                    r["best_any_start"] = max((x[4] for x in rows), default=None)
                    r["episodes"] = len(rows) or r["episodes"]
            ledger = self.ledger()
            compare = self.compare()
            if not self.include_mock:
                ledger = [b for b in ledger if (b.get("game") or "") != "mock"]
                compare = {**compare, "rows": []}
            board = self._board(runs, ledger, compare)
            for r in runs:  # attach curriculum availability
                r["has"]["curriculum_rows"] = False
            version = hash(sig) & 0xFFFFFFFF
            st = {
                "version": version, "generated": time.time(), "demo": self.demo,
                "runs": runs, "ledger": ledger, "compare": compare,
                "levels": board["levels"],
                "groups": [g for g in self.catalog.groups if self.include_mock or g not in MOCK_GROUPS],
                "feed": self._feed(runs), "totals": board["totals"],
            }
            self._state = clean(st)
            self._state_sig = sig
            return self._state

    def _board(self, runs, ledger, compare) -> dict[str, Any]:
        cat = self.catalog
        per: dict[str, dict[str, Any]] = {}

        def slot(key):
            return per.setdefault(key, {"runs": [], "agents": set(), "attempts": 0, "steps": 0.0, "wall_s": 0.0,
                                        "best": {}, "beats": [], "robust": False, "live": False,
                                        "last_update": None})
        for r in runs:
            n = max(1, len(r["levels"]))
            for key in r["levels"]:
                s = slot(key)
                s["runs"].append(r["id"])
                s["agents"].add(r["agent"])
                s["attempts"] += r["episodes"] / n
                s["steps"] += max(0.0, r["steps"] - r["first_step"]) / n
                s["wall_s"] += r["wall_s"] / n
                if r["best_progress"] is not None:
                    src = f"run:{r['kind']}"
                    s["best"][src] = max(s["best"].get(src, 0.0), r["best_progress"])
                if r["status"] == "running":
                    s["live"] = True
                s["last_update"] = max(s["last_update"] or 0, r["last_update"] or 0)
            # curriculum / episode csv best progress from start 0
        for r in runs:
            if r["has"].get("curriculum"):
                rows = self._curriculum_rows(r)
                if rows:
                    p0 = [x[4] for x in rows if x[1] == 0]
                    for key in r["levels"]:
                        s = slot(key)
                        s["attempts"] = max(s["attempts"], 0)
                        if p0:
                            s["best"]["from-start"] = max(s["best"].get("from-start", 0.0), max(p0))
        for row in compare["rows"]:
            for lid, v in row["per_level"].items():
                key = f"mock:{lid}"
                if key not in cat.by_key and cat.resolve(key) is None:
                    continue
                s = slot(key)
                s["agents"].add(row["agent"])
                prog = None
                if v.get("greedy") is not None:
                    prog = v["greedy"]
                if v.get("greedy") is not None and v["greedy"] >= 0.999:
                    s["best"]["greedy-eval"] = 1.0
                if v.get("sticky") is not None and v["sticky"] >= 0.8:
                    s["robust"] = True
                if v.get("solved", {}).get("solved"):
                    s["agents"].add(row["agent"])
        for b in ledger:
            if b.get("demo") and not self.demo:
                continue
            for key in b["keys"]:
                if key not in cat.by_key:
                    cat.ensure(key, b["level"], b.get("game") or "real")
                s = slot(key)
                if b.get("agent"):
                    s["agents"].add(b["agent"])
                if b["partial"]:
                    s["beats"].append({**{k: b[k] for k in ("ts", "when", "agent", "attempts", "generations", "env_steps",
                                                            "wall_hours", "run")}, "partial": True})
                    continue
                s["beats"].append({k: b[k] for k in ("ts", "when", "agent", "attempts", "generations", "env_steps",
                                                      "wall_hours", "run")} | {"partial": False})
                s["best"]["ledger"] = 1.0
                notes = (b.get("notes") or "").lower()
                # Only an explicit "robust" (sticky completion >= 0.8, per the README);
                # "5/10 with sticky actions" is a plain beat.
                if "robust" in notes and "not robust" not in notes:
                    s["robust"] = True
        levels = []
        for lv in cat.levels:
            if not self.include_mock and (lv.get("game") == "mock" or lv.get("group") in MOCK_GROUPS or is_mock_key(lv["key"])):
                continue
            s = per.get(lv["key"])
            item = {**{k: lv.get(k) for k in ("key", "name", "group", "difficulty", "game", "gd_id", "length")}}
            if not s:
                item.update({"status": "untrained", "best_progress": 0.0, "attempts": 0, "steps": 0, "wall_s": 0,
                             "agents": [], "runs": [], "beat": None, "beats": [], "live": False})
            else:
                full = [b for b in s["beats"] if not b["partial"]]
                best = max(s["best"].values(), default=0.0)
                if full:
                    best = 1.0
                status = "untrained"
                if s["runs"] or s["best"] or s["beats"]:
                    status = "training"
                if full:
                    status = "robust-beaten" if s["robust"] else "beaten"
                item.update({
                    "status": status, "best_progress": min(1.0, best), "best_by": s["best"],
                    "attempts": round(s["attempts"]), "steps": s["steps"], "wall_s": s["wall_s"],
                    "agents": sorted(s["agents"]), "runs": s["runs"], "beat": full[0] if full else None,
                    "beats": s["beats"], "live": s["live"], "last_update": s["last_update"],
                })
            levels.append(item)
        totals = {
            "levels": len(levels), "beaten": sum(1 for l in levels if l["status"] in ("beaten", "robust-beaten")),
            "training": sum(1 for l in levels if l["status"] == "training"),
            "running": sum(1 for r in runs if r["status"] == "running"),
            "steps": sum(max(0.0, r["steps"] - r["first_step"]) for r in runs),
            "runs": len(runs), "wall_s": sum(r["wall_s"] for r in runs),
        }
        return {"levels": levels, "totals": totals}

    def _feed(self, runs) -> list[dict[str, Any]]:
        ev: list[dict[str, Any]] = []
        for r in runs:
            d = Path(r["dir"])
            if r["has"]["episodes"]:
                jf = self._jsonl(d, "episodes.jsonl")
                for row in (jf.rows[-12:] if jf else []):
                    ev.append({"t": row.get("t"), "run": r["id"], "agent": r["agent"], "kind": "episode",
                               "progress": fnum(row.get("progress")), "completed": bool(row.get("completed")),
                               "x": fnum(row.get("death_x")), "start": row.get("segment")})
            if r["has"]["curriculum"]:
                for row in self._curriculum_rows(r)[-12:]:
                    ev.append({"t": row[0], "run": r["id"], "agent": r["agent"], "kind": "episode",
                               "progress": row[4], "completed": bool(row[6]), "x": row[3], "start": row[1],
                               "reached": bool(row[7])})
            if r["status"] == "running" and r["has"]["metrics"]:
                ev.append({"t": r["last_update"], "run": r["id"], "agent": r["agent"], "kind": "update",
                           "progress": r["last_progress"], "ret": r["last_return"], "steps": r["steps"],
                           "sps": r["sps"]})
        ev = [e for e in ev if e.get("t")]
        ev.sort(key=lambda e: e["t"], reverse=True)
        return ev[:80]

    # -- curriculum --------------------------------------------------------

    def _curriculum_cfg(self, r) -> tuple[dict[str, Any], Path | None, list[dict[str, Any]]]:
        d = Path(r["dir"])
        cfg = self._config(d)
        cur = ((cfg.get("env") or {}).get("curriculum")) or {}
        starts: list[dict[str, Any]] = []
        if cur.get("starts"):
            sy = self.cache.get(self._path(cur["starts"], d), read_yaml)
            if isinstance(sy, dict):
                starts = [s for s in (sy.get("starts") or []) if isinstance(s, dict)]
        lp = self._path(cur["log_path"], d) if cur.get("log_path") else None
        return cur, lp, starts

    def _curriculum_rows(self, r) -> list[tuple]:
        cur, lp, _ = self._curriculum_cfg(r)
        if not lp:
            return []
        cf = self._curcsv(lp)
        if not cf:
            return []
        lo = (r["created"] or 0) - 2
        hi = float("inf") if r["status"] == "running" else (r["last_update"] or 0) + 90
        # a later run of the same family bounds this one
        if r["status"] != "running":
            for o in self._state_runs_cache(r):
                if o["family"] == r["family"] and (o["created"] or 0) > (r["created"] or 0):
                    hi = min(hi, o["created"] or hi)
        return [x for x in cf.rows if lo <= x[0] <= hi]

    def _state_runs_cache(self, r):
        return self._runs_ctx

    # -- per-run detail ----------------------------------------------------

    def run_detail(self, rid: str) -> dict[str, Any] | None:
        with self.lock:
            st = self.state()
            run = next((r for r in st["runs"] if r["id"] == rid), None)
            if run is None:
                return None
            sig = (self._state_sig, rid)
            hit = self._detail.get(rid)
            if hit and hit[0] == sig:
                return hit[1]
            d = Path(run["dir"])
            detail = {"run": run, "curves": self._curves(run, d), "deaths": self._deaths(run, d),
                      "curriculum": self._curriculum(run), "search": self._search(run, d)}
            detail = clean(detail)
            self._detail[rid] = (sig, detail)
            return detail

    def _curves(self, run, d: Path) -> dict[str, Any]:
        mf = self._metrics(d)
        if not mf:
            return {}
        sps = mf.series.get("perf/steps_per_second")
        wall_x = wall_y = None
        if sps and len(sps[0]) > 1:
            sx = np.array(sps[0])
            sv = np.array(sps[1])
            dt = np.diff(np.concatenate([[run["first_step"]], sx])) / np.maximum(sv, 1e-6)
            wall_x, wall_y = sx, np.cumsum(dt)
        out = {}
        pu = run.get("per_update")
        for key, (xs, ys) in mf.series.items():
            x = np.array(xs)
            y = np.array(ys)
            x, y = downsample(x, y)
            ent = {"x": np.round(x, 1).tolist(), "y": [round(float(v), 6) for v in y]}
            if pu:
                ent["u"] = np.round(x / pu, 2).tolist()
            if wall_x is not None:
                ent["w"] = np.round(np.interp(x, wall_x, wall_y), 1).tolist()
            out[key] = ent
        return out

    def _level_length(self, run, rows_progress) -> float | None:
        for key in run["levels"]:
            ln = self.catalog.by_key.get(key, {}).get("length")
            if ln:
                return float(ln)
        meta = self._meta(Path(run["dir"]))
        if meta.get("length"):
            return float(meta["length"])
        est = [x / p for x, p in rows_progress if p and p > 0.05]
        return statistics.median(est) if est else None

    def _deaths(self, run, d: Path) -> dict[str, Any] | None:
        pts: list[tuple[float, float, int]] = []  # (t, x, completed)
        best = None
        starts_x: list[float] = []
        length = None
        if run["has"]["curriculum"]:
            rows = self._curriculum_rows(run)
            if rows:
                length = self._level_length(run, [(x[3], x[4]) for x in rows])
                for i, x in enumerate(rows):
                    pts.append((x[0], x[3], x[6]))
                _, _, starts = self._curriculum_cfg(run)
                starts_x = [float(s["x"]) for s in starts if s.get("x") is not None]
                bi = max(range(len(rows)), key=lambda i: rows[i][3])
                best = {"x": rows[bi][3], "t": rows[bi][0], "i": bi, "start": rows[bi][1]}
        elif run["has"]["episodes"]:
            jf = self._jsonl(d, "episodes.jsonl")
            meta = self._meta(d)
            length = meta.get("length")
            length = float(length) if length else None
            if length is None:
                length = self._level_length(run, [])
            for r in jf.rows if jf else []:
                prog = fnum(r.get("progress"))
                dx = fnum(r.get("death_x"))
                if dx is None and prog is not None and length:
                    dx = prog * length
                if dx is None:
                    continue
                pts.append((float(r.get("t") or 0), dx, 1 if r.get("completed") else 0))
            if pts:
                bi = max(range(len(pts)), key=lambda i: pts[i][1])
                best = {"x": pts[bi][1], "t": pts[bi][0], "i": bi, "start": None}
        if not pts:
            return None
        if not length:
            length = max(p[1] for p in pts) * 1.05
        bx = 120
        n = len(pts)
        bt = max(1, min(60, n // 15))
        grid = np.zeros((bt, bx), dtype=np.int32)
        edges_t = []
        for j in range(bt):
            a, b = j * n // bt, (j + 1) * n // bt
            edges_t.append([pts[a][0], pts[max(a, b - 1)][0], a, b])
            for t, x, comp in pts[a:b]:
                xi = min(bx - 1, max(0, int(x / length * bx)))
                grid[j, xi] += 1
        return {"length": length, "bx": bx, "bt": bt, "grid": grid.tolist(), "t_bins": edges_t,
                "best": best, "starts_x": starts_x, "n": n,
                "completed": int(sum(p[2] for p in pts))}

    def _curriculum(self, run) -> dict[str, Any] | None:
        if not run["has"]["curriculum"]:
            return None
        cur, lp, starts = self._curriculum_cfg(run)
        rows = self._curriculum_rows(run)
        if not starts:
            return None
        ema = float(cur.get("ema", 0.1))
        solved = float(cur.get("solved", 0.5))
        p_first = float(cur.get("p_first", 0.25))
        min_w = float(cur.get("min_weight", 0.1))
        window = int(cur.get("reach_window", 30))
        frontier = cur.get("frontier_extra")
        n = len(starts)
        succ = np.zeros(n)
        attempts = [0] * n
        reached = [0] * n
        best_x = [0.0] * n
        hist: list[list[float]] = [[] for _ in range(n)]
        recent: list[float] = []
        last_t = [None] * n
        for t, i, sx, fx, prog, ln, comp, rn in rows:
            if not (0 <= i < n):
                continue
            succ[i] += ema * (float(rn) - succ[i])
            attempts[i] += 1
            reached[i] += rn
            best_x[i] = max(best_x[i], fx)
            last_t[i] = t
            hist[i].append(float(succ[i]))
            if i == 0:
                recent = (recent + [fx])[-window:]
        # weights as StartCurriculum.weights()
        num_open = n
        if frontier is not None:
            reach = max(recent, default=0.0)
            for k in range(1, n):
                if float(starts[k].get("x", 0.0)) > reach and succ[k] < solved:
                    num_open = min(n, k + 1 + int(frontier))
                    break
        w = np.maximum(min_w, 1.0 - succ)
        w[num_open:] = 0.0
        p = (1.0 - p_first) * w / w.sum() if w.sum() > 0 else w
        if n > 1:
            p[0] += p_first
            p = p / p.sum()
        else:
            p = np.ones(1)
        tiles = []
        for i, s in enumerate(starts):
            if attempts[i] == 0 and (i >= num_open or not rows):
                status = "locked"
            elif attempts[i] == 0:
                status = "locked"
            elif succ[i] >= solved and attempts[i] >= 10:
                status = "mastered"
            else:
                status = "training"
            h = hist[i]
            if len(h) > 24:
                idx = np.linspace(0, len(h) - 1, 24).astype(int)
                h = [h[k] for k in idx]
            tiles.append({"i": i, "name": s.get("name", f"start-{i}"), "x": s.get("x"), "mode": s.get("mode"),
                          "attempts": attempts[i], "reached": reached[i], "success": float(succ[i]),
                          "raw_rate": (reached[i] / attempts[i]) if attempts[i] else None,
                          "weight": float(p[i]), "status": status, "best_x": best_x[i], "spark": h,
                          "last_t": last_t[i]})
        return {"tiles": tiles, "num_open": num_open, "solved": solved, "episodes": len(rows),
                "p_first": p_first, "ema": ema, "level_length": self._level_length(run, [(x[3], x[4]) for x in rows])}

    def _search(self, run, d: Path) -> dict[str, Any] | None:
        if not run["has"]["generations"]:
            return None
        jf = self._jsonl(d, "generations.jsonl")
        rows = jf.rows if jf else []
        keys = ["generation", "segment", "step", "t", "best_fitness", "mean_fitness", "best_progress", "rollouts",
                "survivors", "beam", "backtracks", "x"]
        n = len(rows)
        idx = np.unique(np.linspace(0, n - 1, min(n, 600)).astype(int)) if n else []
        series = {k: [fnum(rows[i].get(k)) for i in idx] for k in keys}
        result = self.cache.get(d / "result.json", read_json)
        meta = self._meta(d)
        return {"n": n, "series": series, "result": result if isinstance(result, dict) else None,
                "params": meta.get("search"), "last": rows[-1] if rows else None}
