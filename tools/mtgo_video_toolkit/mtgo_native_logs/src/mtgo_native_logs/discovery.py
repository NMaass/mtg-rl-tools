"""Find MTGO GameLog/DraftLog candidates without modifying them."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os


def default_roots() -> List[Path]:
    home = Path.home()
    roots = [home / "Documents"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Apps" / "2.0")
    else:
        roots.append(home / "AppData" / "Local" / "Apps" / "2.0")
    return roots


def discover_logs(roots: Optional[Iterable[str]] = None,
                  maximum_files: int = 10000) -> List[Dict[str, object]]:
    result = []
    candidates = [Path(value).expanduser() for value in roots] \
        if roots is not None else default_roots()
    for root in candidates:
        if not root.exists():
            continue
        try:
            iterator = root.rglob("*")
            for path in iterator:
                if len(result) >= maximum_files:
                    return result
                if not path.is_file() or not _looks_like_log(path.name):
                    continue
                stat = path.stat()
                result.append({
                    "path": str(path.resolve()),
                    "kind": "draft" if "draftlog" in path.name.casefold()
                            else "game",
                    "bytes": stat.st_size,
                    "mtimeNs": stat.st_mtime_ns,
                })
        except (OSError, PermissionError):
            continue
    result.sort(key=lambda row: (str(row["kind"]), str(row["path"])))
    return result


def _looks_like_log(name: str) -> bool:
    normalized = name.casefold()
    return ("gamelog" in normalized or "draftlog" in normalized) and \
        normalized.endswith((".txt", ".log", ".dat", ".xml"))
