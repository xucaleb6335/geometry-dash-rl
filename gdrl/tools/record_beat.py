"""Record a level beat in holistic_progress/level_beats.csv (and regenerate the .md table).

Every agent that beats a level (mock or real game) records it here, once per
(level, agent) the first time it is beaten, and again for later milestones
(e.g. first full greedy run, first run from the level start after segment work).

    python -m gdrl.tools.record_beat --level "Bloodbath" --level-id 10565740 \
        --game real --difficulty "extreme demon" --agent checkpoint-search \
        --attempts 51234 --generations 412 --env-steps 9000000 \
        --wall-hours 6.5 --run runs/2026-10-11_bloodbath --notes "greedy replay verified"

The time is recorded automatically in Pacific time (PST/PDT).
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2] / "holistic_progress"
CSV_PATH = ROOT / "level_beats.csv"
MD_PATH = ROOT / "level_beats.md"
FIELDS = [
    "beaten_at_pacific", "level", "level_id", "game", "difficulty", "agent",
    "attempts", "generations", "env_steps", "wall_hours", "run", "notes",
]


def pacific_now() -> str:
    try:
        tz = ZoneInfo("America/Los_Angeles")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:  # no tz database (Windows without tzdata): local time is Pacific here
        now = datetime.now().astimezone()
        return now.strftime("%Y-%m-%d %H:%M:%S ") + (now.tzname() or "local")


def render_md(rows: list[dict]) -> str:
    lines = [
        "# Level beats",
        "",
        "Every level an agent has beaten, with the attempts/generations it took.",
        "Times are Pacific. Source of truth: `level_beats.csv` "
        "(append with `python -m gdrl.tools.record_beat`).",
        "",
        "| Beaten at (Pacific) | Level | ID | Game | Difficulty | Agent | Attempts "
        "| Generations | Env steps | Wall h | Run | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", required=True)
    p.add_argument("--level-id", default="")
    p.add_argument("--game", choices=["mock", "real"], required=True)
    p.add_argument("--difficulty", default="")
    p.add_argument("--agent", required=True, help="e.g. ppo-objlist, ppo-grid, rule, genetic, checkpoint-search")
    p.add_argument("--attempts", default="", help="episodes/attempts until the first beat")
    p.add_argument("--generations", default="", help="generations/search iterations, if any")
    p.add_argument("--env-steps", default="")
    p.add_argument("--wall-hours", default="")
    p.add_argument("--run", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--at", default="", help="override the timestamp (Pacific) for back-filling")
    a = p.parse_args(argv)

    ROOT.mkdir(exist_ok=True)
    rows = []
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append({
        "beaten_at_pacific": a.at or pacific_now(), "level": a.level, "level_id": a.level_id,
        "game": a.game, "difficulty": a.difficulty, "agent": a.agent, "attempts": a.attempts,
        "generations": a.generations, "env_steps": a.env_steps, "wall_hours": a.wall_hours,
        "run": a.run, "notes": a.notes,
    })
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    MD_PATH.write_text(render_md(rows), encoding="utf-8")
    print(f"recorded: {rows[-1]}")


if __name__ == "__main__":
    main()
