"""Narrow subprocess wrapper around yt-dlp.

No authentication, cookie extraction, geo-bypass, or DRM workarounds are used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
import json
import shutil
import subprocess

from .manifest import SourceEntry, SourceManifest


class YtDlpClient:
    def __init__(self, executable: Optional[str] = None,
                 runner: Optional[Callable[..., subprocess.CompletedProcess]] = None):
        self.executable = executable or shutil.which("yt-dlp") or "yt-dlp"
        self.runner = runner or subprocess.run

    def available(self) -> bool:
        return bool(shutil.which(self.executable) or Path(self.executable).is_file())

    def discover(self, url: str) -> SourceManifest:
        command = [
            self.executable, "--flat-playlist", "--dump-single-json",
            "--no-warnings", str(url),
        ]
        result = self.runner(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            entries = [payload]
        rows = [self._entry(row, url) for row in entries if isinstance(row, dict)]
        manifest = SourceManifest(rows, metadata={
            "discoveredFrom": url,
            "extractor": payload.get("extractor") if isinstance(payload, dict) else None,
        })
        manifest.deduplicate()
        return manifest

    def download(self, entry: SourceEntry, output_dir: str,
                 *, rights_acknowledged: bool,
                 maximum_height: int = 1080) -> List[str]:
        if not rights_acknowledged:
            raise PermissionError(
                "download requires explicit acknowledgement of rights and platform terms")
        url = entry.webpage_url or entry.source_url
        if not url:
            raise ValueError("source entry has no URL")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        output_template = str(target /
            "%(uploader_id,channel_id,extractor)s/%(upload_date)s-%(id)s-%(title).180B.%(ext)s")
        command = [
            self.executable,
            "--no-warnings", "--no-progress",
            "--write-info-json", "--write-description", "--write-thumbnail",
            "--write-subs", "--sub-langs", "all,-live_chat",
            "--merge-output-format", "mp4",
            "--format", f"bv*[height<={int(maximum_height)}]+ba/b[height<={int(maximum_height)}]",
            "--output", output_template,
            "--print", "after_move:filepath",
            str(url),
        ]
        result = self.runner(command, check=True, capture_output=True, text=True)
        paths = [line.strip() for line in result.stdout.splitlines()
                 if line.strip() and Path(line.strip()).is_file()]
        if paths:
            entry.attach_local_file(paths[-1])
        return paths

    @staticmethod
    def _entry(row: Dict, source_url: str) -> SourceEntry:
        webpage = row.get("webpage_url") or row.get("url")
        if webpage and not str(webpage).startswith(("http://", "https://")):
            webpage = None
        return SourceEntry(
            source_url=source_url,
            extractor=row.get("extractor") or row.get("extractor_key"),
            source_id=str(row.get("id")) if row.get("id") is not None else None,
            title=row.get("title"), uploader=row.get("uploader"),
            channel=row.get("channel"), upload_date=row.get("upload_date"),
            duration_seconds=float(row["duration"]) if row.get("duration") else None,
            webpage_url=webpage,
            metadata={key: row.get(key) for key in (
                "playlist_id", "playlist_title", "availability", "live_status")
                if row.get(key) is not None},
        )
