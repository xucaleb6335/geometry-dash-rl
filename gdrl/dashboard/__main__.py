"""python -m gdrl.dashboard [--port 8050] [--runs runs] [--export out.html] [--demo] [--mock]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import Store
from .server import export_snapshot, serve


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m gdrl.dashboard", description=__doc__)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs", default="runs", help="runs directory (default: runs)")
    ap.add_argument("--root", default=".", help="project root holding logs/, results/, holistic_progress/")
    ap.add_argument("--export", metavar="OUT.html", help="write a self-contained snapshot and exit")
    ap.add_argument("--demo", action="store_true", help="add synthetic fixtures from gdrl/dashboard/demo/")
    ap.add_argument("--mock", action="store_true", help="also show mock levels, mock runs and the mock agent comparison")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    runs = Path(args.runs)
    store = Store(root, runs if runs.is_absolute() else root / runs, demo=args.demo, include_mock=args.mock)
    if args.export:
        out = export_snapshot(store, Path(args.export))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
        return
    serve(store, args.port, args.host)


if __name__ == "__main__":
    main()
