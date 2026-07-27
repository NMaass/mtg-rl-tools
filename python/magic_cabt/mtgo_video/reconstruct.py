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
from typing import List, NamedTuple, Optional, Tuple

from .ocr import strip_timestamp

_NORM_RE = re.compile(r"[^a-z0-9 ]+")
# Frames OCR the same scrollbar glyphs differently; normalize() also drops
# short trailing junk tokens so variants collapse to the same key.
_TAIL_JUNK_RE = re.compile(r"(?: [a-z]{1,2})+$")

# How far back into the accumulated log a frame window may reach. The pane
# holds ~18 entries, so anything beyond a couple of pane-heights is a false
# anchor -- game 2's opening lines look almost exactly like game 1's.
_TAIL_WINDOW = 36
# How far behind the previous frame's anchor a new frame may still align,
# to absorb a line or two of jitter without letting it slip a whole pane.
_ANCHOR_SLACK = 6


class Entry(NamedTuple):
    text: str
    norm: str
    time: Optional[float]  # video seconds of the first frame showing it
    readings: Tuple[str, ...] = ()  # every OCR reading of this line
    frames: Tuple[int, ...] = ()    # frame indices this line was seen in

    def sightings(self) -> Tuple[str, ...]:
        return self.readings or (self.text,)


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
    """Merge two sightings of one log line.

    Keeps the longest (least clipped) reading as the working representative
    so alignment stays stable, the earliest timestamp, and every reading
    seen so far -- the readings are what `consensus` later votes over.
    """
    best = b if len(b.norm) > len(a.norm) else a
    return Entry(best.text, best.norm, _earliest(a.time, b.time),
                 a.sightings() + b.sightings(), a.frames + b.frames)


def consensus(readings: Tuple[str, ...]) -> str:
    """The most likely true text of a line, from repeated OCR readings.

    A log line stays on screen for a dozen frames or more, and subpixel
    differences between them mean OCR does not fail the same way every time
    -- so the majority reading is usually right even where any single one is
    not. Clipped sightings (the line half-scrolled past the pane edge) are
    excluded first: they are shorter but not wrong, and would otherwise
    outvote the full readings.
    """
    if not readings:
        return ""
    longest = max(len(normalize(text)) for text in readings)
    if longest == 0:
        return readings[0]
    full = [text for text in readings
            if len(normalize(text)) >= longest * 0.9] or list(readings)
    counts = {}
    for text in full:
        counts[text] = counts.get(text, 0) + 1
    # Most frequent; ties break toward the longest reading, then toward the
    # lexicographically first so the result never depends on frame order.
    return max(counts, key=lambda text: (counts[text], len(text),
                                         [-ord(c) for c in text]))


def collapse_duplicates(entries: List[Entry], window: int = 4,
                        threshold: float = 0.86) -> List[Entry]:
    """Fold entries that are the same log line read two different ways.

    When OCR is noisy enough that two readings of one line fail to align,
    the line is recorded twice. Collapsing similar neighbours blindly would
    be wrong -- MTGO really does log the same sentence twice in a row (two
    copies of a creature cast back to back) -- so the test is whether the
    two were ever on screen *at the same time*. Two genuinely distinct lines
    coexist in the pane for many frames; a line and its misread duplicate
    never do.
    """
    out: List[Entry] = []
    for entry in entries:
        merged = False
        for index in range(len(out) - 1, max(-1, len(out) - 1 - window), -1):
            candidate = out[index]
            if not _similar(candidate.norm, entry.norm, threshold):
                continue
            if set(candidate.frames) & set(entry.frames):
                continue  # seen together, so they are two real lines
            out[index] = _pick(candidate, entry)
            merged = True
            break
        if not merged:
            out.append(entry)
    return out


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


def merge_windows(tail: List[Entry], frame: List[Entry]):
    """Merge a frame's window into the tail; also report where it anchored.

    The anchor is where this frame's first line landed in the tail. It
    matters because a game log is full of near-identical stretches ("Turn N:
    X" / "X draws a card." / "X plays Island." repeats every turn), and a
    plain sequence match will happily lock onto the wrong occurrence of one
    -- duplicating or swallowing a whole pane's worth of lines. The pane only
    ever scrolls forward, so the caller uses the anchor to stop the next
    frame from aligning behind this one.
    """
    sm = difflib.SequenceMatcher(
        None, [e.norm for e in tail], [e.norm for e in frame], autojunk=False
    )
    blocks = sm.get_matching_blocks()  # ends with a terminal size-0 block
    real = [b for b in blocks if b.size > 0]
    anchor = max(0, real[0].a - real[0].b) if real else len(tail)

    out: List[Entry] = []
    pa = pb = 0
    for block in blocks:
        out.extend(_merge_gap(tail[pa:block.a], frame[pb:block.b],
                              at_end=block.a >= len(tail)))
        for k in range(block.size):
            out.append(_pick(tail[block.a + k], frame[block.b + k]))
        pa, pb = block.a + block.size, block.b + block.size
    return out, anchor


class LogReconstructor:
    def __init__(self):
        self._entries: List[Entry] = []
        self._frame_index = 0
        # Where the previous frame's window aligned. The pane scrolls one
        # way, so the next frame cannot align meaningfully behind it.
        self._anchor = 0

    def finalize(self) -> List[Entry]:
        return collapse_duplicates(self._entries)

    @property
    def entries(self) -> List[str]:
        return [consensus(e.sightings()) for e in self.finalize()]

    @property
    def sighting_counts(self) -> List[int]:
        return [len(e.sightings()) for e in self.finalize()]

    @property
    def times(self) -> List[Optional[float]]:
        """Video-seconds timestamp of the frame each entry first appeared in."""
        return [e.time for e in self.finalize()]

    def feed(self, frame_entries: List[str],
             timestamp: Optional[float] = None) -> int:
        """Merge one frame's entries; returns the change in entry count."""
        index = self._frame_index
        self._frame_index += 1
        frame = [Entry(text, normalize(text), timestamp, (text,), (index,))
                 for text in frame_entries]
        frame = [e for e in frame if e.norm]
        if not frame:
            return 0

        # Look back from wherever the last frame anchored, minus a little
        # slack for jitter, and never further than one window.
        base = max(0, len(self._entries) - _TAIL_WINDOW,
                   self._anchor - _ANCHOR_SLACK)
        before = len(self._entries)
        merged, anchor = merge_windows(self._entries[base:], frame)
        self._entries = self._entries[:base] + merged
        self._anchor = base + anchor
        return len(self._entries) - before
