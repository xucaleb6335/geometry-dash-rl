"""stdlib HTTP server: static app + JSON API, and the single-file exporter."""

from __future__ import annotations

import gzip
import json
import mimetypes
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .aggregate import Store, clean

STATIC = Path(__file__).resolve().parent / "static"
TRACE_SUFFIX = ".trace.json.gz"


class Traces:
    """Network-visualizer traces in results/viz (gdrl.tools.nn_trace): the list
    (meta only, cached per file size and mtime) and the raw gzipped files."""

    def __init__(self, root: Path):
        self.dir = Path(root) / "results" / "viz"
        self.cache: dict[str, tuple[tuple[int, int], dict]] = {}

    def path(self, name: str) -> Path | None:
        if not name or not all(c.isalnum() or c in "-_." for c in name) or name.startswith("."):
            return None
        p = (self.dir / (name + TRACE_SUFFIX)).resolve()
        return p if p.parent == self.dir.resolve() and p.is_file() else None

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*" + TRACE_SUFFIX)) if self.dir.is_dir() else []:
            try:
                st = p.stat()
            except OSError:
                continue
            sig = (st.st_size, st.st_mtime_ns)
            hit = self.cache.get(p.name)
            if not hit or hit[0] != sig:
                try:
                    m = json.loads(gzip.decompress(p.read_bytes())).get("meta", {})
                except (OSError, ValueError, EOFError):
                    continue
                name = p.name[: -len(TRACE_SUFFIX)]
                hit = (sig, {"name": name, "title": m.get("title") or name, "level": m.get("level"),
                             "difficulty": m.get("difficulty"), "decisions": m.get("decisions"),
                             "completed": m.get("completed"), "progress": m.get("progress"),
                             "seconds": m.get("seconds"), "recorded_at": m.get("recorded_at"),
                             "source": m.get("source"), "bytes": st.st_size})
                self.cache[p.name] = hit
            out.append(hit[1])
        # real-game runs first, then by title
        return sorted(out, key=lambda t: (t.get("source") != "game", str(t.get("title"))))


class Watcher:
    """Runs `python -m gdrl.tools.watch <name>` (play a policy in GD at 1x) as
    one background job at a time; its output goes to logs/watch.log."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.presets_path = self.root / "configs" / "watch.yaml"
        self.log_path = self.root / "logs" / "watch.log"
        self.proc: subprocess.Popen | None = None
        self.job: dict | None = None
        self.lock = threading.Lock()

    def presets(self) -> list[dict]:
        try:
            import yaml
            data = yaml.safe_load(self.presets_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return []
        return [{"id": k, "name": v.get("name", k), "level": v.get("dashboard_level")} for k, v in data.items()]

    def status(self) -> dict:
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            if self.job is not None and not running and self.job.get("exit") is None and self.proc is not None:
                self.job["exit"] = self.proc.returncode
                self.job["ended"] = time.time()
            try:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            except OSError:
                tail = []
            return {"presets": self.presets(), "running": running, "job": self.job, "log": tail}

    def start(self, name: str, attempts: int) -> dict:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": False, "error": "a policy is already playing"}
            if name not in {p["id"] for p in self.presets()}:
                return {"ok": False, "error": f"unknown policy {name}"}
            attempts = max(1, min(int(attempts), 50))
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "gdrl.tools.watch", name, "--attempts", str(attempts)],
                cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            self.job = {"name": name, "attempts": attempts, "started": time.time(), "exit": None}
            return {"ok": True}

    def stop(self) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "error": "nothing is playing"}
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # The killed job could not clean up: bridge settings back to training
        # defaults and release its game lock.
        env_py = [sys.executable, "-m", "gdctl"]
        for key in ("watch-mode", "legacy-view", "show-hitboxes"):
            subprocess.run(env_py + ["--owner", "watch", "mod-setting", key, "off"], cwd=self.root,
                           capture_output=True, timeout=60)
        subprocess.run(env_py + ["--owner", "watch", "lock", "release"], cwd=self.root, capture_output=True, timeout=60)
        return {"ok": True}


def make_handler(store: Store, watcher: Watcher | None = None, traces: Traces | None = None):
    watcher = watcher or Watcher(store.root)
    traces = traces or Traces(store.root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "gdrl-dashboard"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None,
                  gzipped: bool = False) -> None:
            accept = self.headers.get("Accept-Encoding", "")
            if gzipped:  # already gzip: the browser inflates it (every browser sends gzip)
                extra = {**(extra or {}), "Content-Encoding": "gzip"}
            elif len(body) > 2048 and "gzip" in accept:
                body = gzip.compress(body, 3)
                extra = {**(extra or {}), "Content-Encoding": "gzip"}
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(clean(obj), separators=(",", ":"), allow_nan=False).encode(),
                       "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    self._send(200, render_html(embed=None).encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    st = store.state()
                    if q.get("v", [""])[0] == str(st["version"]):
                        self._json({"unchanged": True, "version": st["version"], "generated": time.time()})
                    else:
                        self._json(st)
                elif u.path == "/api/run":
                    d = store.run_detail(q.get("id", [""])[0])
                    if d is None:
                        self._json({"error": "unknown run"}, 404)
                    else:
                        self._json(d)
                elif u.path == "/api/watch":
                    self._json(watcher.status())
                elif u.path in ("/viz", "/viz/"):
                    from gdrl.visualizer import render_page

                    self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/traces":
                    self._json(traces.list())
                elif u.path == "/api/trace":
                    p = traces.path(q.get("name", [""])[0])
                    if p is None:
                        self._json({"error": "unknown trace"}, 404)
                    else:
                        self._send(200, p.read_bytes(), "application/json; charset=utf-8", gzipped=True)
                elif u.path.startswith("/static/"):
                    p = (STATIC / u.path[len("/static/"):]).resolve()
                    if STATIC in p.parents and p.is_file():
                        self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                    else:
                        self._send(404, b"not found", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep the server alive
                import traceback
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            # Only same-origin page requests (the app sends this header).
            if self.headers.get("X-GDRL") != "1":
                return self._json({"ok": False, "error": "forbidden"}, 403)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if u.path == "/api/watch/start":
                    self._json(watcher.start(str(body.get("name", "")), int(body.get("attempts", 1))))
                elif u.path == "/api/watch/stop":
                    self._json(watcher.stop())
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json({"ok": False, "error": str(e)}, 500)

    return Handler


def render_html(embed: dict | None) -> str:
    """index.html with css/js inlined (always) and an optional data snapshot."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    html = html.replace("/*__CSS__*/", css).replace("/*__JS__*/", js.replace("</script", "<\\/script"))
    snap = "null"
    if embed is not None:
        snap = json.dumps(clean(embed), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return html.replace("/*__SNAPSHOT__*/null", snap)


def export_snapshot(store: Store, out: Path) -> Path:
    st = store.state()
    details = {}
    for r in st["runs"]:
        if r["has"]["metrics"] or r["has"]["episodes"] or r["has"]["generations"]:
            d = store.run_detail(r["id"])
            if d:
                details[r["id"]] = d
    out = Path(out)
    out.write_text(render_html({"state": st, "details": details, "exported": time.time()}), encoding="utf-8")
    return out


def serve(store: Store, port: int, host: str = "127.0.0.1") -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(store))
    print(f"GD-RL dashboard on http://{host}:{port}/   (Ctrl+C to stop)")
    th = threading.Thread(target=store.state, daemon=True)  # warm the cache
    th.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
