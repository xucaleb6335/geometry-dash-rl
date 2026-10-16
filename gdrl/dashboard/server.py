"""stdlib HTTP server: static app + JSON API, and the single-file exporter."""

from __future__ import annotations

import gzip
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .aggregate import Store, clean

STATIC = Path(__file__).resolve().parent / "static"


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gdrl-dashboard"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            accept = self.headers.get("Accept-Encoding", "")
            if len(body) > 2048 and "gzip" in accept:
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
