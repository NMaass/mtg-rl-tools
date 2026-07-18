"""Sequential corpus runner for source manifests.

GPU OCR/VLM jobs are intentionally sequential by default so one 8 GB card is
not oversubscribed.  Interrupted runs are resumable: completed extraction
manifests are skipped unless ``replace`` is requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List
import json
import re

from mtgo_video_acquisition.manifest import SourceManifest


class CorpusRunner:
    def __init__(self, extractor: Callable[[str, str], Dict[str, Any]]):
        self.extractor = extractor

    def run(self, source_manifest: str, output_root: str,
            replace: bool = False) -> Dict[str, Any]:
        manifest = SourceManifest.load(source_manifest)
        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        completed, skipped, failed = [], [], []
        for index, entry in enumerate(manifest.entries):
            if not entry.local_path:
                failed.append({"index": index, "reason": "no-local-path"})
                continue
            video = Path(entry.local_path)
            if not video.is_file():
                failed.append({"index": index, "reason": "missing-local-file",
                               "path": str(video)})
                continue
            run_id = _run_id(entry, index)
            destination = root / run_id
            existing = destination / "manifest.json"
            if existing.exists() and not replace:
                skipped.append({"index": index, "runId": run_id,
                                "manifest": str(existing)})
                continue
            try:
                result = self.extractor(str(video), str(destination))
                completed.append({"index": index, "runId": run_id,
                                  "manifest": result})
            except Exception as error:
                failed.append({"index": index, "runId": run_id,
                               "error": f"{type(error).__name__}: {error}"})
        report = {
            "schemaVersion": 1,
            "sourceManifest": str(Path(source_manifest).resolve()),
            "outputRoot": str(root),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
        }
        with (root / "corpus_run_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return report


def _run_id(entry, index: int) -> str:
    identity = entry.source_id or (entry.sha256[:12] if entry.sha256 else None)
    title = entry.title or Path(entry.local_path or "video").stem
    slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:60] or "video"
    return f"{index:06d}-{identity or slug}-{slug}"[:120]
