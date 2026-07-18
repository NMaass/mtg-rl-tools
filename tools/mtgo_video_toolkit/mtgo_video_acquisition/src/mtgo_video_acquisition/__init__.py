"""MTGO video source acquisition and provenance manifests."""

from .manifest import SourceEntry, SourceManifest
from .ytdlp import YtDlpClient

__all__ = ["SourceEntry", "SourceManifest", "YtDlpClient"]
