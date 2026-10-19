"""Network visualizer: a single page (index.html + viz.css + viz.js, no build step,
no external requests) that replays a ``gdrl.tools.nn_trace`` trace.

The dashboard serves it at ``/viz`` (traces from ``/api/traces`` and
``/api/trace?name=``); ``python -m gdrl.tools.nn_trace export`` writes it as one
offline HTML file with the trace embedded (gzipped, base64, unpacked in the
browser with DecompressionStream).
"""

from __future__ import annotations

import html
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent


def render_page(embed: str | None = None, name: str = "") -> str:
    """index.html with the CSS and JS inlined, plus an optional embedded trace
    (``embed`` = base64 of the gzipped trace JSON)."""
    page = (VIZ_DIR / "index.html").read_text(encoding="utf-8")
    css = (VIZ_DIR / "viz.css").read_text(encoding="utf-8")
    js = (VIZ_DIR / "viz.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    data = ""
    if embed:
        data = (f'<script id="trace-data" type="application/octet-stream" data-encoding="gzip+base64" '
                f'data-name="{html.escape(name, quote=True)}">{embed}</script>')
    # Plain replace on the placeholders; the inserted texts are not scanned again.
    head, rest = page.split("/*__CSS__*/", 1)
    page = head + css + rest
    head, rest = page.split("<!--__TRACE__-->", 1)
    page = head + data + rest
    head, rest = page.split("/*__JS__*/", 1)
    return head + js + rest
