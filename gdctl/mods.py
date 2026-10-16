"""Installed Geode mods and their enabled flags, offline.

Geode keeps each mod's enabled state as ``"should-load-<mod id>": bool`` in
``%LOCALAPPDATA%\\GeometryDash\\geode\\mods\\geode.loader\\saved.json``. A
missing key means enabled. Geode reads the file at startup and rewrites it on
exit, so edit it only while the game is closed (while it runs, use the
control mod's ``set_mod_enabled``; it takes effect after a restart).
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

from gdctl import paths


def installed_mods(mods_dir: Path | None = None) -> list[dict]:
    """Every .geode in the mods folder, with id/name/version from its mod.json."""
    mods_dir = mods_dir or paths.geode_mods_dir()
    out = []
    for f in sorted(mods_dir.glob("*.geode")):
        info = {"id": f.stem, "file": f.name, "name": None, "version": None}
        try:
            with zipfile.ZipFile(f) as z:
                meta = json.loads(z.read("mod.json").decode("utf-8"))
            info.update(id=meta.get("id", f.stem), name=meta.get("name"), version=meta.get("version"))
        except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            pass
        out.append(info)
    return out


def read_saved(path: Path | None = None) -> dict:
    path = path or paths.geode_saved_json()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def enabled_flags(saved: dict, mod_ids: list[str]) -> dict[str, bool]:
    return {m: bool(saved.get(f"should-load-{m}", True)) for m in mod_ids}


def set_flags(changes: dict[str, bool], path: Path | None = None, backup: bool = True) -> dict:
    """Write should-load flags. Only call while the game is closed."""
    path = path or paths.geode_saved_json()
    saved = read_saved(path)
    before = {m: bool(saved.get(f"should-load-{m}", True)) for m in changes}
    for mod_id, enabled in changes.items():
        saved[f"should-load-{mod_id}"] = bool(enabled)
    backup_path = None
    if backup and path.exists():
        backup_path = path.with_name(f"saved.json.gdctl-{time.strftime('%Y%m%d-%H%M%S')}.bak")
        shutil.copy2(path, backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, indent=4), encoding="utf-8")
    tmp.replace(path)
    return {"before": before, "after": {m: bool(v) for m, v in changes.items()},
            "file": str(path), "backup": str(backup_path) if backup_path else None}


def offline_listing() -> list[dict]:
    saved = read_saved()
    mods = installed_mods()
    flags = enabled_flags(saved, [m["id"] for m in mods])
    for m in mods:
        m["enabled"] = flags[m["id"]]
    return mods
