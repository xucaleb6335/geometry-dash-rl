"""Is the agent learning, or repeating the same mistakes? Reads a curriculum episode log.

    python -m gdrl.tools.curriculum_report logs/sm_community_episodes.csv [--windows 4] [--last 8000]

Splits the last ``--last`` episodes into ``--windows`` equal windows (oldest
first) and prints, per window: the success rate per start (reached the next
start, or completed), and the most common death spots (final x, 50-unit
bins) with their counts. A death spot whose count does not fall from window
to window is a mistake the agent keeps repeating.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--last", type=int, default=8000)
    p.add_argument("--bin", type=float, default=50.0)
    p.add_argument("--top", type=int, default=6)
    args = p.parse_args(argv)

    rows = list(csv.DictReader(args.log.open(newline="")))[-args.last:]
    if not rows:
        raise SystemExit("no episodes yet")
    n = len(rows)
    size = max(1, n // args.windows)
    windows = [rows[i:i + size] for i in range(0, n, size)][: args.windows]
    starts = sorted({int(r["start"]) for r in rows})

    print(f"{n} episodes, {len(windows)} windows of ~{size} (oldest first)\n")
    print("success per start (reached next start or completed), % per window:")
    for s in starts:
        cells = []
        for w in windows:
            ws = [r for r in w if int(r["start"]) == s]
            cells.append(f"{100 * sum(int(r['reached_next']) for r in ws) / len(ws):5.0f}" if ws else "    -")
        print(f"  start {s:2d}: " + " ".join(cells))

    print("\nmost common death spots per window (x bin: deaths):")
    for k, w in enumerate(windows):
        deaths = Counter(int(float(r["final_x"]) // args.bin * args.bin) for r in w if r["completed"] == "0"
                         and r["reached_next"] == "0")
        top = ", ".join(f"{x}: {c}" for x, c in deaths.most_common(args.top))
        print(f"  window {k}: {top}")
    first = [r for r in windows[0] if r["start"] == "0"]
    last = [r for r in windows[-1] if r["start"] == "0"]
    if first and last:
        mean = lambda rs: sum(float(r["final_x"]) for r in rs) / len(rs)
        print(f"\nfrom the level start: mean final x {mean(first):.0f} -> {mean(last):.0f}, "
              f"best {max(float(r['final_x']) for r in first):.0f} -> {max(float(r['final_x']) for r in last):.0f}")


if __name__ == "__main__":
    main()
