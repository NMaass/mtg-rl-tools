"""Reconstruct the full game log from overlapping per-frame OCR windows.

The log pane shows a scrolling window over an append-only log. Consecutive
frames therefore share a (fuzzy) overlap. Each frame is merged into the tail
of the accumulated log: exact matches (over normalized entries) anchor the
alignment, gaps between anchors are paired fuzzily so OCR variants collapse
onto one entry (keeping the longest reading), and genuinely new entries are
appended.

Each entry also keeps the timestamp of the earliest frame it appeared in,
which is what lets an extracted log be replayed in sync with the footage.
"""

import difflib
import re
from typing import List, NamedTuple, Optional

from .ocr import strip_timestamp

_NORM_RE = re.compile(r"[^a-z0-9 ]+")
# Frames OCR the same scrollbar glyphs differently; normalize() also drops
# short trailing junk tokens so variants collapse to the same key.
_TAIL_JUNK_RE = re.compile(r"(?: [a-z]{1,2})+$")

# How far back into the accumulated log a frame window may reach. The pane
# holds ~18 entries, so anything beyond a couple of pane-heights is a false
# anchor -- game 2's opening lines look almost exactly like game 1's.
_TAIL_WINDOW = 36


class Entry(NamedTuple):
    text: str
    norm: str
    time: Optional[float]  # video seconds of the first frame showing it


def normalize(entry: str) -> str:
    text = strip_timestamp(entry).lower()
    text = _NORM_RE.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > 12:
        text = _TAIL_JUNK_RE.sub("", text)
    return text


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    if a == b:
        return True
    if len(a) >= 12 and len(b) >= 12 and (a.startswith(b) or b.startswith(a)):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _earliest(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _pick(a: Entry, b: Entry) -> Entry:
    """Keep the longest (least clipped) reading, but the earliest sighting."""
    best = b if len(b.norm) > len(a.norm) else a
    return Entry(best.text, best.norm, _earliest(a.time, b.time))


def _merge_gap(tail_gap: List[Entry], frame_gap: List[Entry],
               at_end: bool) -> List[Entry]:
    """Merge unanchored stretches: pair OCR variants, keep true entries.

    Frame entries that pair with nothing are only kept when the gap runs to
    the end of the accumulated log (a genuine new tail). Mid-log, an unpaired
    frame entry is a misalignment, not a discovery: the log is append-only,
    so nothing new can appear between two entries already recorded.
    """
    out: List[Entry] = []
    i = j = 0
    while i < len(tail_gap) and j < len(frame_gap):
        if _similar(tail_gap[i].norm, frame_gap[j].norm):
            out.append(_pick(tail_gap[i], frame_gap[j]))
            i += 1
            j += 1
        else:
            out.append(tail_gap[i])
            i += 1
    out.extend(tail_gap[i:])
    if at_end:
        out.extend(frame_gap[j:])
    return out


def merge_windows(tail: List[Entry], frame: List[Entry]) -> List[Entry]:
    sm = difflib.SequenceMatcher(
        None, [e.norm for e in tail], [e.norm for e in frame], autojunk=False
    )
    out: List[Entry] = []
    pa = pb = 0
    for block in sm.get_matching_blocks():  # ends with a terminal size-0 block
        out.extend(_merge_gap(tail[pa:block.a], frame[pb:block.b],
                              at_end=block.a >= len(tail)))
        for k in range(block.size):
            out.append(_pick(tail[block.a + k], frame[block.b + k]))
        pa, pb = block.a + block.size, block.b + block.size
    return out


class LogReconstructor:
    def __init__(self):
        self._entries: List[Entry] = []

    @property
    def entries(self) -> List[str]:
        return [e.text for e in self._entries]

    @property
    def times(self) -> List[Optional[float]]:
        """Video-seconds timestamp of the frame each entry first appeared in."""
        return [e.time for e in self._entries]

    def feed(self, frame_entries: List[str],
             timestamp: Optional[float] = None) -> int:
        """Merge one frame's entries; returns the change in entry count."""
        frame = [Entry(text, normalize(text), timestamp) for text in frame_entries]
        frame = [e for e in frame if e.norm]
        if not frame:
            return 0

        base = max(0, len(self._entries) - _TAIL_WINDOW)
        before = len(self._entries)
        merged = merge_windows(self._entries[base:], frame)
        self._entries = self._entries[:base] + merged
        return len(self._entries) - before
