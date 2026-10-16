"""Where things live on this machine. Override with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STEAM_APP_ID = 322170
EXE_NAME = "GeometryDash.exe"
WINDOW_TITLE = "Geometry Dash"

CONTROL_PORT = int(os.environ.get("GDCTL_PORT", "5556"))
BRIDGE_PORT = int(os.environ.get("GDCTL_BRIDGE_PORT", "5555"))

CONTROL_MOD_ID = "xucaleb6335.gd-agent-control"
BRIDGE_MOD_ID = "xucaleb6335.gd-rl-bridge"
# Mods that break the lockstep tick loop; must be off while training.
TRAINING_BLOCKLIST = ("syzzi.click_between_frames", "thesillydoggo.qolmod")
TRAINING_REQUIRED = (BRIDGE_MOD_ID, CONTROL_MOD_ID)

# Buildable mod folders in this repo, by short name.
MOD_DIRS = {"mod": REPO / "mod", "bridge": REPO / "mod", "control-mod": REPO / "control-mod",
            "control": REPO / "control-mod"}


def gd_dir() -> Path:
    env = os.environ.get("GDCTL_GD_DIR")
    if env:
        return Path(env)
    return Path(r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash")


def gd_exe() -> Path:
    return gd_dir() / EXE_NAME


def geode_logs_dir() -> Path:
    return gd_dir() / "geode" / "logs"


def geode_crashlogs_dir() -> Path:
    return gd_dir() / "geode" / "crashlogs"


def geode_mods_dir() -> Path:
    return gd_dir() / "geode" / "mods"


def save_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "GeometryDash"


def geode_saved_json() -> Path:
    """Geode's loader save: holds ``should-load-<mod id>`` (the enabled flags)."""
    return save_dir() / "geode" / "mods" / "geode.loader" / "saved.json"


def screens_dir() -> Path:
    return REPO / "logs" / "screens"


def gdctl_logs_dir() -> Path:
    return REPO / "logs" / "gdctl"
