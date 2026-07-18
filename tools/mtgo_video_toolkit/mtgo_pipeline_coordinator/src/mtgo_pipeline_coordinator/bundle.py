"""Pipeline bundle manifest helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import hashlib
import json
import os


class PipelineBundle:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, value: Dict[str, Any]) -> str:
        path = self.path / "pipeline_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        return str(path)

    def inventory(self) -> Dict[str, Any]:
        files = []
        for path in sorted(self.path.rglob("*")):
            if path.is_file():
                files.append({
                    "path": str(path.relative_to(self.path)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
        return {"root": str(self.path.resolve()), "files": files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
