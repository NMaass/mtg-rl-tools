"""Serializable source/provenance manifest."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import time


@dataclass
class SourceEntry:
    source_url: Optional[str] = None
    extractor: Optional[str] = None
    source_id: Optional[str] = None
    title: Optional[str] = None
    uploader: Optional[str] = None
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    duration_seconds: Optional[float] = None
    webpage_url: Optional[str] = None
    local_path: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    permission_note: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SourceEntry":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in value.items() if key in fields})

    def to_dict(self):
        return asdict(self)

    def attach_local_file(self, path: str) -> None:
        target = Path(path).expanduser().resolve()
        self.local_path = str(target)
        self.bytes = target.stat().st_size
        self.sha256 = file_sha256(target)


@dataclass
class SourceManifest:
    entries: List[SourceEntry] = field(default_factory=list)
    schema_version: int = 1
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "SourceManifest":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            entries=[SourceEntry.from_dict(row)
                     for row in payload.get("entries") or []],
            schema_version=int(payload.get("schemaVersion", 1)),
            created_at=float(payload.get("createdAt", time.time())),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self):
        return {
            "schemaVersion": self.schema_version,
            "createdAt": self.created_at,
            "metadata": dict(self.metadata),
            "entries": [row.to_dict() for row in self.entries],
        }

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)

    def deduplicate(self) -> None:
        result = []
        seen = set()
        for row in self.entries:
            key = row.webpage_url or row.source_url or row.sha256 or row.local_path
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            result.append(row)
        self.entries = result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
