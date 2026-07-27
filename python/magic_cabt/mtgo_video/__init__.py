"""MTGO gameplay-video ingestion.

Turns MTGO client footage into a structured game-event log by OCRing the
"Chat & Game Log" pane, then reconstructs board snapshots in the same
mirror-state format the Arena mirror produces, so the extracted game can be
replayed and verified in XMage.

Pipeline stages:

    video -> frames (ffmpeg crop of the log pane)
          -> OCR (tesseract)
          -> log reconstruction (dedupe scrolling windows)
          -> event parsing (MTGO log grammar)
          -> board simulation -> mirror_states.jsonl (XMage mirror format)
"""

from .regions import Region, LOG_PANE_1080
from .extract import extract_log_frames
from .ocr import ocr_image, lines_to_entries
from .reconstruct import LogReconstructor
from .parse import parse_entry, parse_log
from .catalog import CardCatalog, build_catalog
from .games import split_games, discover_players
from .state import GameSimulator

__all__ = [
    "Region",
    "LOG_PANE_1080",
    "extract_log_frames",
    "ocr_image",
    "lines_to_entries",
    "LogReconstructor",
    "parse_entry",
    "parse_log",
    "CardCatalog",
    "build_catalog",
    "split_games",
    "discover_players",
    "GameSimulator",
]
