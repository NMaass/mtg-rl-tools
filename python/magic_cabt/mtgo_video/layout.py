"""Locate MTGO's UI in a frame, whatever resolution and arrangement it has.

Hardcoded pixel regions only work for the one capture they were measured on.
Everything downstream of the crop -- OCR, log reconstruction, the card
catalog, the simulator -- is already layout-agnostic, so the whole dependency
on how the client was arranged lives here.

MTGO's duel scene is not a fixed picture. Panes are dockable and resizable,
the client is often not full-screen, streamers overlay a facecam and a
scoreboard on top of it, the whole UI can be scaled up, extra zone panes
(graveyard, exile) can be docked beside the seats, and the chat/game-log pane
can be moved, resized or closed. Anything measured as "x percent across the
frame" is therefore a guess about one recording rather than a fact about the
client.

So this module detects the UI from *anchors it can recognise*, all of which
are text MTGO itself draws:

- **The phase bar.** A horizontal row of step labels -- Untap, Upkeep, Draw,
  Main, ... Cleanup -- is present in every duel and nowhere else. Finding
  that row identifies the frame as a duel, and its extent separates the play
  area from the margins that hold the seat panels.
- **The seat panels.** Each seat's life is a large numeral, several times the
  height of any other text near it. The two tallest digit tokens in a margin,
  one per half of the frame, are the two life totals -- and the player name
  is the text directly under each.
- **The log pane.** A bright panel whose text *reads as timestamped log
  lines*. Brightness alone finds candidates (a facecam and a sponsor graphic
  are bright too); reading them is what picks the right one, and reading the
  text inside it is also what measures where the text column ends, rather
  than assuming a scrollbar width.

Everything located is then validated by reading it back, so a layout this
module gets wrong is reported rather than silently turned into a plausible
looking wrong log.
"""

import os
import re
import tempfile
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

from .ocr import Word, ocr_image, ocr_words
from .regions import Region

# Reference geometry, measured on a full-screen 1920x1080 capture with the
# default duel scene. Only used when detection finds nothing to anchor on.
REFERENCE_WIDTH = 1920.0

_LIFE_REF = {"x": 112, "w": 80, "h": 48, "y_top": 120, "y_bottom": 704}
_NAME_REF = {"x": 45, "w": 170, "h": 18, "y_top": 168, "y_bottom": 748}

# A log-pane candidate must be at least this bright, and this much of the
# frame, to be believed.
_BRIGHT = 170
# A log pane can be narrow: a windowed client on a 1080p canvas puts it at
# about a sixteenth of the frame. Candidates this loose are safe because each
# one is read before it is believed.
_MIN_PANE_WIDTH_FRAC = 0.05
_MIN_PANE_HEIGHT_FRAC = 0.15

_TIMESTAMP_HINT = re.compile(r"[\dOl|/It]{1,2}\s*[:.]\s*[\dO]{2}\s*[AP4/][A-Za-z]{0,2}")

# Words only a game log contains. A chat pane carries timestamps too, so
# timestamps alone cannot tell the two apart when they are separate panes.
_LOG_VERBS = re.compile(
    r"\b(casts?|plays?|draws?|mills?|discards?|attack|attacked|blocks?|"
    r"sacrifices|destroyed|exiled|graveyard|library|triggered|Turn)\b",
    re.IGNORECASE)

# The phase bar's labels. MTGO draws all twelve; OCR reliably returns some.
_STEP_WORDS = {"untap", "upkeep", "draw", "main", "begin", "combat", "attack",
               "attackers", "block", "blockers", "damage", "end", "cleanup"}
_MIN_STEP_WORDS = 4

# Below roughly this glyph height tesseract stops being dependable, whatever
# it is upscaled to afterwards: the information is not in the pixels.
MIN_RELIABLE_TEXT_HEIGHT = 9.0


@dataclass
class Layout:
    """Where MTGO's UI sits in a particular capture."""

    frame_width: int
    frame_height: int
    content: Region        # client area, letterboxing removed
    log_pane: Region       # game-log text column, scrollbar excluded
    life_top: Region
    life_bottom: Region
    name_top: Region
    name_bottom: Region
    scale: float           # content width relative to the 1920px reference
    detected: bool         # was the log pane found, or is it proportional?
    samples: int = 1       # frames the detection agreed across
    phase_bar: Optional[Region] = None   # the anchor the seats were found from
    seats_detected: bool = False         # or scaled from the reference?
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        # asdict recurses, so every Region field comes back as a plain dict.
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Layout":
        """Rebuild a layout recorded in a bundle's match.json.

        Tolerant of bundles written by earlier versions, which have no phase
        bar or seat-detection fields.
        """
        fields = {k: v for k, v in data.items()
                  if k in cls.__dataclass_fields__}
        for key in ("content", "log_pane", "life_top", "life_bottom",
                    "name_top", "name_bottom", "phase_bar"):
            if isinstance(fields.get(key), dict):
                fields[key] = Region(**fields[key])
        return cls(**fields)

    def describe(self) -> str:
        return ("%dx%d content=%dx%d+%d+%d log=%dx%d+%d+%d life=%s scale=%.3f "
                "%s/%s"
                % (self.frame_width, self.frame_height,
                   self.content.width, self.content.height,
                   self.content.x, self.content.y,
                   self.log_pane.width, self.log_pane.height,
                   self.log_pane.x, self.log_pane.y,
                   "%d+%d/%d+%d" % (self.life_top.x, self.life_top.y,
                                    self.life_bottom.x, self.life_bottom.y),
                   self.scale,
                   "log:detected" if self.detected else "log:reference",
                   "seats:detected" if self.seats_detected else "seats:reference"))


# --------------------------------------------------------------------------
# Reading a frame
# --------------------------------------------------------------------------

_WORD_CACHE: Dict[Tuple[str, int, bool], List[Word]] = {}


def _load_gray(path: str):
    from PIL import Image

    return Image.open(path).convert("L")


def frame_words(path: str, psm: int = 11, invert: bool = False) -> List[Word]:
    """Every word tesseract sees in a whole frame, memoized per pass.

    Layout detection reads each sampled frame several times -- once to decide
    whether it is a duel, again to find the seats -- so the passes are cached
    rather than repeated.
    """
    key = (os.path.realpath(path), psm, invert)
    if key in _WORD_CACHE:
        return _WORD_CACHE[key]
    source = path
    work = None
    if invert:
        from PIL import ImageOps

        work = tempfile.mkdtemp(prefix="mtgo_words_")
        source = os.path.join(work, "inverted.png")
        ImageOps.invert(_load_gray(path)).save(source)
    try:
        words = ocr_words(source, psm=psm)
    finally:
        if work:
            import shutil

            shutil.rmtree(work, ignore_errors=True)
    _WORD_CACHE[key] = words
    return words


def _anchor_words(path: str) -> List[Word]:
    """The two passes that between them see MTGO's UI labels.

    psm 11 finds scattered small labels; psm 12 segments differently and
    catches the large numerals psm 11 merges into the art behind them. Most
    words are seen by both, so the two passes are merged rather than
    concatenated: a duplicate would otherwise be paired with itself and read
    as two seats, or as two turn indicators.
    """
    return _merge_words(frame_words(path, psm=11) + frame_words(path, psm=12))


def _inverted_words(path: str) -> List[Word]:
    """The same passes over an inverted frame.

    Tesseract expects dark text on light. Most of MTGO's chrome is, but a
    seat panel is not: the player's name is white on a near-black plate, and
    in a tournament broadcast's darker skin it can be invisible to the
    ordinary passes while reading cleanly inverted. Only used when the seats
    were not found otherwise, since it costs two more passes over the frame.
    """
    return _merge_words(frame_words(path, psm=11, invert=True)
                        + frame_words(path, psm=12, invert=True))


def _merge_words(words: Sequence[Word]) -> List[Word]:
    """One entry per thing on screen, keeping the most confident reading."""
    kept: List[Word] = []
    for word in sorted(words, key=lambda w: -w.conf):
        duplicate = False
        for other in kept:
            overlap_x = min(word.right, other.right) - max(word.x, other.x)
            overlap_y = min(word.bottom, other.bottom) - max(word.y, other.y)
            if (overlap_x > 0.5 * min(word.width, other.width)
                    and overlap_y > 0.5 * min(word.height, other.height)):
                duplicate = True
                break
        if not duplicate:
            kept.append(word)
    return kept


# --------------------------------------------------------------------------
# Anchor: the phase bar
# --------------------------------------------------------------------------

def detect_phase_bar(words: Sequence[Word], frame_height: int) -> Optional[Region]:
    """The row of step labels MTGO draws across the play area, if present.

    This is the one element that says "this is a duel". A deck editor, a
    sideboarding screen or a menu has none, and neither does a facecam or an
    overlay, so a row of four or more distinct step names is both a scene
    test and a positional anchor.
    """
    tolerance = max(4, int(frame_height * 0.012))
    rows: Dict[int, List[Word]] = {}
    for word in words:
        token = word.text.strip(":.,()[]").lower()
        if token in _STEP_WORDS:
            rows.setdefault(word.y // tolerance, []).append(word)

    best = None
    for members in rows.values():
        distinct = {m.text.strip(":.,()[]").lower() for m in members}
        if len(distinct) < _MIN_STEP_WORDS:
            continue
        left = min(m.x for m in members)
        right = max(m.right for m in members)
        score = (len(distinct), right - left)
        if best is None or score > best[0]:
            best = (score, members)
    if best is None:
        return None
    members = best[1]
    top = min(m.y for m in members)
    return Region(x=min(m.x for m in members), y=top,
                  width=max(m.right for m in members) - min(m.x for m in members),
                  height=max(1, max(m.bottom for m in members) - top))


def find_phase_bar(frame_path: str, frame_height: int
                   ) -> Tuple[Optional[Region], List[Word]]:
    """The phase bar, and the words it was found among.

    The bar's labels are dim grey on a dark strip, and a compressed stream
    can smear them past what the ordinary passes read while leaving them
    perfectly legible inverted -- so the frame is read that way too rather
    than concluding a duel frame is not one.
    """
    words = _anchor_words(frame_path)
    bar = detect_phase_bar(words, frame_height)
    if bar is not None:
        return bar, words
    words = _merge_words(words + _inverted_words(frame_path))
    return detect_phase_bar(words, frame_height), words


def looks_like_duel(frame_path: str, ffmpeg: str = "ffmpeg") -> bool:
    """Whether a frame shows the duel scene rather than a menu screen.

    A VOD spends a lot of its length on deck-building and sideboarding
    screens. Those have a game log too -- a differently shaped one, in a
    different place -- so including them in the layout consensus produces a
    pane that matches neither.

    The test is the phase bar, which is drawn only during a duel. That is
    both more reliable and cheaper than the previous test (reading both life
    totals), which needed the seat positions this module is trying to find.
    """
    image = _load_gray(frame_path)
    return find_phase_bar(frame_path, image.size[1])[0] is not None


def select_duel_frames(frame_paths: List[str]) -> List[str]:
    """The sampled frames that show a duel; all of them if none do."""
    duels = [path for path in frame_paths if looks_like_duel(path)]
    return duels or list(frame_paths)


# --------------------------------------------------------------------------
# Anchor: the seat panels (life totals and player names)
# --------------------------------------------------------------------------

# A life numeral is far taller than the timer, the mana pips and the zone
# counts drawn around it, which is what makes "the tallest digits in this
# margin" a safe way to find it. These bounds are of the frame's height and
# are deliberately loose -- the pairing rules below do the real work.
_MIN_LIFE_DIGIT = 0.022
_MAX_LIFE_DIGIT = 0.085


def _digit_words(image, strip: Region, frame_height: int) -> List[Word]:
    """Digit tokens inside a vertical strip of the frame, life-numeral sized."""
    if strip.width < 8 or strip.height < 8:
        return []
    work = tempfile.mkdtemp(prefix="mtgo_digits_")
    try:
        crop = os.path.join(work, "strip.png")
        image.crop((strip.x, strip.y, strip.x + strip.width,
                    strip.y + strip.height)).save(crop)
        found: List[Word] = []
        for psm in (11, 12):
            for word in ocr_words(crop, psm=psm, whitelist="0123456789"):
                found.append(word)
        out = []
        for word in found:
            text = word.text.strip()
            if not text.isdigit() or len(text) > 2:
                continue
            if not (frame_height * _MIN_LIFE_DIGIT <= word.height
                    <= frame_height * _MAX_LIFE_DIGIT):
                continue
            # A tall, thin sliver is a piece of card art or a panel edge, not
            # a numeral: digits are roughly half as wide as they are tall.
            ratio = word.width / float(word.height * len(text))
            if not 0.30 <= ratio <= 1.20:
                continue
            out.append(Word(text=text, x=word.x + strip.x, y=word.y + strip.y,
                            width=word.width, height=word.height,
                            conf=word.conf))
        return out
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def _dedupe(words: Sequence[Word]) -> List[Word]:
    """Collapse the same numeral seen by more than one pass."""
    kept: List[Word] = []
    for word in sorted(words, key=lambda w: (-w.conf, -w.height)):
        overlap = False
        for other in kept:
            if (abs(word.y - other.y) < max(4, other.height * 0.5)
                    and abs(word.x - other.x) < max(6, other.height)):
                overlap = True
                break
        if not overlap:
            kept.append(word)
    return kept


def _pair_seats(candidates: Sequence[Word], frame_height: int
                ) -> Optional[Tuple[Word, Word]]:
    """The two candidates that look like one seat panel above and one below.

    Requiring one in each half is what rejects a stream overlay: a scoreboard
    reading "0 - 0" puts two big numerals side by side in the same half, and
    a duel never does.
    """
    upper = [w for w in candidates if w.bottom < frame_height * 0.48]
    lower = [w for w in candidates if w.y > frame_height * 0.52]
    best = None
    for top in upper:
        for bottom in lower:
            # The two seats are the same widget drawn twice, so their
            # numerals are the same size and sit in the same column.
            height_gap = abs(top.height - bottom.height) / float(
                max(top.height, bottom.height))
            column_gap = abs((top.x + top.width / 2.0)
                             - (bottom.x + bottom.width / 2.0))
            if height_gap > 0.35:
                continue
            if column_gap > max(top.height, bottom.height) * 2.5:
                continue
            score = (height_gap * 2.0
                     + column_gap / max(1.0, float(top.height))
                     - (top.conf + bottom.conf) / 400.0)
            if best is None or score < best[0]:
                best = (score, (top, bottom))
    return best[1] if best else None


def _life_regions_from(anchors: Tuple[Word, Word]) -> Tuple[Region, Region]:
    """Boxes that hold each life total, from the digits that were found.

    OCR may have caught only part of a two-digit total (the "2" of a "20"
    whose zero sits over bright art), so the crop is widened to the column
    the two seats agree on rather than to whatever one reading happened to
    cover.
    """
    top, bottom = anchors
    height = max(top.height, bottom.height)
    left = min(top.x, bottom.x) - int(round(height * 0.55))
    right = max(top.right, bottom.right) + int(round(height * 0.55))
    pad = int(round(height * 0.30))
    return (Region(x=max(0, left), y=max(0, top.y - pad),
                   width=max(8, right - left), height=top.height + 2 * pad),
            Region(x=max(0, left), y=max(0, bottom.y - pad),
                   width=max(8, right - left), height=bottom.height + 2 * pad))


def _name_region_under(anchor: Word, life: Region, words: Sequence[Word]
                       ) -> Region:
    """The player name MTGO prints under a seat's avatar.

    Preferring a word actually seen there over a measured-once offset is what
    carries this across client scales: the same panel drawn at 1.5x has its
    name 1.5x further down.
    """
    height = anchor.height
    band_top = life.y + life.height
    band_bottom = band_top + int(round(height * 1.8))
    column_left = life.x - int(round(height * 1.6))
    column_right = life.x + life.width + int(round(height * 0.5))
    near = [w for w in words
            if band_top <= w.y <= band_bottom
            and w.right > column_left and w.x < column_right
            and len(w.text.strip()) >= 3
            and not w.text.strip().isdigit()]
    if near:
        top = min(w.y for w in near)
        return Region(x=max(0, min(w.x for w in near) - 4),
                      y=max(0, top - 3),
                      width=max(20, max(w.right for w in near)
                                - min(w.x for w in near) + 8),
                      height=max(8, max(w.bottom for w in near) - top + 6))
    # Nothing readable there in this frame; fall back to where the panel puts
    # it, in units of the life numeral rather than of the frame.
    return Region(x=max(0, column_left), y=band_top + int(round(height * 0.15)),
                  width=max(20, column_right - column_left),
                  height=max(8, int(round(height * 0.6))))


# A player name is ordinary small text; these bounds are of the frame height.
_MIN_NAME_TEXT = 0.006
_MAX_NAME_TEXT = 0.030

# Where a seat's life numeral sits relative to the name under it, in units of
# the name's own text height. The seat panel is one widget MTGO draws at
# whatever scale the client is set to, so measuring it in its own units --
# rather than as a fraction of the frame -- is what carries across clients
# that have been scaled up, and across a client that is not full-screen.
# Top and bottom of the band the life numeral sits in, above the name, in
# units of the name's text height. Generous at both ends: the old broadcast
# client draws the numeral lower in the plate than the modern one, and the
# band is trimmed onto the glyphs it actually contains before it is read.
# The lower edge stops just short of the name itself, so the crop can never
# contain the letters.
_LIFE_ABOVE_NAME = (6.5, 0.3)
# Half the seat panel's width, in units of the name's text height. The name is
# centred in the panel and the life numeral is right-aligned inside it, so the
# panel -- not the name's own extent -- is what the crop has to span: a short
# name like "bob" sits in exactly as wide a panel as a long one.
_LIFE_PANEL_HALF_WIDTH = 7.0

# How tall a life numeral is relative to the name under it. Measured at
# roughly 3.7x across four clients; the floor below is deliberately well
# under that, since it is a rejection test rather than a measurement.
_MIN_LIFE_TO_NAME = 2.0

# How many name pairs are worth reading before giving up on this margin.
_NAME_PAIRS_TRIED = 12


def _name_words(words: Sequence[Word], strip: Region, frame_height: int
                ) -> List[Word]:
    """Text tokens in a margin that could be a player name."""
    out = []
    for word in words:
        text = word.text.strip()
        # Three characters is MTGO's own minimum for a screen name.
        if len(text) < 3 or text.isdigit():
            continue
        if sum(ch.isalnum() for ch in text) < len(text) * 0.6:
            continue
        # No confidence gate: a player name is an arbitrary string, which
        # tesseract often reports at zero confidence even when it has the
        # glyphs right. Candidates are cheap and every one is verified by
        # reading the life box it implies, so let them through.
        if not (frame_height * _MIN_NAME_TEXT <= word.height
                <= frame_height * _MAX_NAME_TEXT):
            continue
        if not (strip.x <= word.x and word.right <= strip.x + strip.width):
            continue
        out.append(word)
    return out


def _pair_names(candidates: Sequence[Word], frame_height: int
                ) -> List[Tuple[Word, Word]]:
    """Name candidates that could be the two seats, best first."""
    # The client's own title bar runs across the very top of the frame and is
    # full of ordinary words ("Pauper League: Stage 1 - Match 2 vs. ..."),
    # none of which is a seat.
    candidates = [w for w in candidates if w.y > frame_height * 0.035]
    upper = [w for w in candidates if w.bottom < frame_height * 0.48]
    lower = [w for w in candidates if w.y > frame_height * 0.52]
    middle = frame_height / 2.0
    pairs = []
    for top in upper:
        for bottom in lower:
            height_gap = abs(top.height - bottom.height) / float(
                max(top.height, bottom.height))
            if height_gap > 0.4:
                continue
            # MTGO centres each name in its seat panel, and the two panels are
            # the same column, so the names agree on their centre -- not on
            # their left edge, which moves with how long the name is.
            column_gap = abs((top.x + top.width / 2.0)
                             - (bottom.x + bottom.width / 2.0))
            if column_gap > max(top.height, bottom.height) * 4:
                continue
            # MTGO draws the two seat panels symmetrically about the board, so
            # the pair's midpoint is the frame's. Without this, any two words
            # that happen to share an x beat the real names.
            skew = abs((top.y + bottom.y) / 2.0 - middle) / frame_height
            pairs.append((height_gap + column_gap / max(1.0, float(top.height))
                          + skew * 6.0 - (top.conf + bottom.conf) / 400.0,
                          (top, bottom)))
    pairs.sort(key=lambda item: item[0])
    return [pair for _, pair in pairs]


def _life_region_above(name: Word) -> Region:
    """The band a seat's life numeral occupies, given the name under it."""
    unit = float(name.height)
    centre = name.x + name.width / 2.0
    top = int(round(name.y - _LIFE_ABOVE_NAME[0] * unit))
    bottom = int(round(name.y - _LIFE_ABOVE_NAME[1] * unit))
    left = int(round(centre - _LIFE_PANEL_HALF_WIDTH * unit))
    right = int(round(centre + _LIFE_PANEL_HALF_WIDTH * unit))
    return Region(x=max(0, left), y=max(0, top),
                  width=max(8, right - left), height=max(8, bottom - top))


def _name_box(word: Word) -> Region:
    pad = max(2, int(round(word.height * 0.4)))
    return Region(x=max(0, word.x - pad), y=max(0, word.y - pad),
                  width=word.width + 2 * pad, height=word.height + 2 * pad)


def _clamp(image, region: Region) -> Optional[Tuple[int, int, int, int]]:
    box = (max(0, region.x), max(0, region.y),
           min(image.size[0], region.x + region.width),
           min(image.size[1], region.y + region.height))
    if box[2] - box[0] < 6 or box[3] - box[1] < 6:
        return None
    return box


# What a box holding a life numeral looks like in pixels: white glyphs on a
# darker panel, spanning most of the box's height. Both bounds matter -- an
# empty stretch of board is dark and fails the lower one, and a card face or
# the log pane itself is uniformly bright and fails the upper one. This runs
# before any OCR because it is nearly free and rejects most candidates.
_GLYPH_WHITE = 190
_MIN_INK = 0.04
_MAX_INK = 0.30
_MIN_GLYPH_SPAN = 0.45


def _looks_like_numeral(image, region: Region,
                        min_glyph: Optional[float] = None) -> bool:
    """Whether a box plausibly holds a large numeral, by pixels alone.

    `min_glyph` is how tall the numeral should be, in source pixels, when the
    caller knows -- from the name under the seat, which is drawn at a fixed
    fraction of it. Judging the ink run against that rather than against the
    box's own height keeps the test honest when the box is deliberately
    larger than the numeral it is looking for.
    """
    box = _clamp(image, region)
    if box is None:
        return False
    crop = image.crop(box)
    pixels = crop.load()
    width, height = crop.size
    ink = 0
    rows = []
    for y in range(height):
        lit = sum(1 for x in range(width) if pixels[x, y] > _GLYPH_WHITE)
        ink += lit
        rows.append(lit > 0)
    fraction = ink / float(width * height)
    # A uniformly bright region -- a card face, the log pane -- is not a
    # numeral on a panel, whatever else is true of it.
    if fraction > _MAX_INK:
        return False
    longest = current = 0
    for row in rows:
        current = current + 1 if row else 0
        longest = max(longest, current)
    if min_glyph is not None:
        # The lower ink bound is for when nothing is known about the glyph's
        # size; here the run length says it directly, and a deliberately
        # generous box would fail a fraction test for being generous.
        return longest >= min_glyph
    return fraction >= _MIN_INK and longest >= height * _MIN_GLYPH_SPAN


def _tighten_to_glyphs(image, region: Region) -> Region:
    """Shrink a life box onto the numeral actually drawn inside it.

    The box a seat anchor implies is deliberately generous, since the anchor
    only says roughly where the panel is. Trimming it to the lit pixels
    afterwards leaves the crop the crosscheck will use holding the numeral
    and little else, which is worth a digit or two of accuracy on the seat
    whose avatar art is bright.
    """
    box = _clamp(image, region)
    if box is None:
        return region
    crop = image.crop(box)
    pixels = crop.load()
    width, height = crop.size
    columns = [x for x in range(width)
               if any(pixels[x, y] > _GLYPH_WHITE for y in range(height))]
    rows = [y for y in range(height)
            if any(pixels[x, y] > _GLYPH_WHITE for x in range(width))]
    if not columns or not rows:
        return region
    pad = max(2, int(round(height * 0.15)))
    left = max(0, columns[0] - pad)
    top = max(0, rows[0] - pad)
    return Region(x=box[0] + left, y=box[1] + top,
                  width=max(8, min(width, columns[-1] + pad) - left),
                  height=max(8, min(height, rows[-1] + pad) - top))


def _trims(region: Region) -> List[Region]:
    """Narrower crops of a life box, for when bright art shares it.

    A seat's numeral is drawn over the avatar, and a crop wide enough to be
    sure of catching it also catches whatever bright paint is beside it,
    which thresholds into shapes OCR reads as extra digits. Which end the
    numeral sits at is a property of the client's layout, so it is chosen by
    which trim reads more consistently rather than assumed.
    """
    span = int(round(region.height * 1.8))
    if span >= region.width:
        return [region]
    return [
        region,
        Region(x=region.x + region.width - span, y=region.y,
               width=span, height=region.height),
        Region(x=region.x, y=region.y, width=span, height=region.height),
    ]


def _reads_as_life(image, region: Region) -> Tuple[Optional[int], int]:
    """Read a candidate life box back, the way the crosscheck will read it.

    Returns the value and how many of the readings agreed on it, so a
    candidate that only just scraped a number can be ranked below one that
    every reading agreed about.
    """
    from PIL import Image

    from .hud import read_int_votes

    box = _clamp(image, region)
    if box is None:
        return None, 0
    crop = image.crop(box)
    factor = max(6.0, 240.0 / max(1, crop.width))
    crop = crop.resize((int(round(crop.width * factor)),
                        int(round(crop.height * factor))), Image.LANCZOS)
    work = tempfile.mkdtemp(prefix="mtgo_life_")
    try:
        path = os.path.join(work, "life.png")
        crop.save(path)
        votes = read_int_votes(path)
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    if not votes:
        return None, 0
    value, count = max(votes.items(), key=lambda item: (item[1], -item[0]))
    if count < 2 and len(votes) > 1:
        return None, 0
    return value, count


def turn_player(words: Sequence[Word], bar: Region) -> Optional[str]:
    """The player MTGO names in "Turn N: <player>" beside the phase bar.

    Whoever that is has a seat panel, so it is independent confirmation that
    a name found in a margin is a player's and not a card's.
    """
    band = [w for w in words
            if abs(w.y - bar.y) <= max(6, bar.height) and w.x < bar.x]
    if len(band) < 2:
        return None
    band.sort(key=lambda w: w.x)
    # The label reads "Turn 4: <player>". Matching the word "Turn" itself is
    # not dependable -- OCR renders it "Tum" as often as not -- but its shape
    # is: a couple of tokens, then the name last.
    last = band[-1].text.strip(":.")
    if len(last) >= 3 and any(ch.isalpha() for ch in last) and not last.isdigit():
        return last
    return None


def detect_seats(image, bar: Region, words: Sequence[Word]
                 ) -> Optional[Dict[str, object]]:
    """Locate both seat panels: life totals and player names.

    The seats sit in a margin beside the play area, and which margin varies:
    the modern client puts them on the left, older broadcast layouts on the
    right. Both are searched, under two independent anchors, and a candidate
    is only accepted once both of its life boxes actually read back as a
    number -- which is what stops a mana pip or a stream overlay's scoreboard
    from being adopted as a seat.
    """
    width, height = image.size
    margins = [
        ("left", Region(x=0, y=0, width=max(0, bar.x), height=height)),
        ("right", Region(x=min(width, bar.x + bar.width), y=0,
                         width=max(0, width - (bar.x + bar.width)),
                         height=height)),
    ]
    # Widest margin first: the seats need room, and the play area's other
    # side is usually a sliver.
    margins.sort(key=lambda item: -item[1].width)
    for side, strip in margins:
        if strip.width < height * 0.04:
            continue

        candidates: List[Dict[str, object]] = []
        # Anchor 1: the numerals themselves, when OCR sees both of them.
        digits = _dedupe(_digit_words(image, strip, height))
        pair = _pair_seats(digits, height)
        if pair is not None:
            top_box, bottom_box = _life_regions_from(pair)
            candidates.append({
                "anchor": "life-digits",
                "life_top": top_box, "life_bottom": bottom_box,
                "name_top": _name_region_under(pair[0], top_box, words),
                "name_bottom": _name_region_under(pair[1], bottom_box, words),
                "names": None,
                "minGlyph": 0.6 * min(pair[0].height, pair[1].height),
            })
        # Anchor 2: the player names. A name is small ordinary text, which
        # OCR finds far more dependably than a numeral drawn over card art,
        # and the panel puts the life total a fixed distance above it.
        # Only the best-ranked pairs are tried: every candidate costs an OCR
        # read, and a margin full of card text can offer hundreds.
        for top_name, bottom_name in _pair_names(
                _name_words(words, strip, height), height)[:_NAME_PAIRS_TRIED]:
            candidates.append({
                "anchor": "player-names",
                "life_top": _life_region_above(top_name),
                "life_bottom": _life_region_above(bottom_name),
                "name_top": _name_box(top_name),
                "name_bottom": _name_box(bottom_name),
                "names": (top_name.text.strip(), bottom_name.text.strip()),
                # A life numeral is several times the height of the name
                # under it; anything much shorter in that band is not one.
                "minGlyph": _MIN_LIFE_TO_NAME * min(top_name.height,
                                                    bottom_name.height),
            })

        active = turn_player(words, bar)
        best = None
        for candidate in candidates:
            # Cheap pixel test first: it rejects nearly everything without
            # paying for OCR, and what survives is worth reading.
            floor = candidate.get("minGlyph")
            if not (_looks_like_numeral(image, candidate["life_top"], floor)
                    and _looks_like_numeral(image, candidate["life_bottom"], floor)):
                continue
            candidate["life_top"] = _tighten_to_glyphs(
                image, candidate["life_top"])
            candidate["life_bottom"] = _tighten_to_glyphs(
                image, candidate["life_bottom"])
            top_life, top_votes = _reads_as_life(image, candidate["life_top"])
            bottom_life, bottom_votes = _reads_as_life(
                image, candidate["life_bottom"])
            if top_life is None or bottom_life is None:
                continue
            score = top_votes + bottom_votes
            names = candidate.get("names")
            if names:
                if names[0].lower() == names[1].lower():
                    # One string found twice is a repeated label, not two
                    # players.
                    continue
                if active and max(_similar(active, names[0]),
                                  _similar(active, names[1])) >= 0.6:
                    # MTGO says whose turn it is; a margin holding that name
                    # is holding a seat panel.
                    score += 12
            candidate["side"] = side
            candidate["life"] = {"top": top_life, "bottom": bottom_life}
            candidate["turnPlayer"] = active
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            return _refine_seat_crops(image, best[1])
    return None


def _refine_seat_crops(image, candidate: Dict[str, object]) -> Dict[str, object]:
    """Settle each seat's crop on the trim both seats read best under.

    The two seats are the same widget, so they are trimmed the same way or
    not at all -- if only one box is wide enough to trim, there is no shared
    trim to choose and the boxes stay as they are.
    """
    top_trims = _trims(candidate["life_top"])
    bottom_trims = _trims(candidate["life_bottom"])
    if len(top_trims) != len(bottom_trims) or len(top_trims) < 2:
        return candidate
    variants = list(zip(top_trims, bottom_trims))
    best = None
    for top_box, bottom_box in variants:
        top_value, top_votes = _reads_as_life(image, top_box)
        bottom_value, bottom_votes = _reads_as_life(image, bottom_box)
        if top_value is None or bottom_value is None:
            continue
        score = top_votes + bottom_votes
        if best is None or score > best[0]:
            best = (score, top_box, bottom_box,
                    {"top": top_value, "bottom": bottom_value})
    if best is not None:
        candidate["life_top"] = best[1]
        candidate["life_bottom"] = best[2]
        candidate["life"] = best[3]
    return candidate


def _similar(a: str, b: str) -> float:
    import difflib

    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# --------------------------------------------------------------------------
# The log pane
# --------------------------------------------------------------------------

def detect_content_area(image) -> Region:
    """The client area within the frame, with black letterboxing removed."""
    width, height = image.size
    pixels = image.load()
    step = max(1, height // 200)

    def column_is_black(x):
        return all(pixels[x, y] < 16 for y in range(0, height, step))

    def row_is_black(y):
        return all(pixels[x, y] < 16 for x in range(0, width, step))

    left = 0
    while left < width - 1 and column_is_black(left):
        left += 1
    right = width - 1
    while right > left and column_is_black(right):
        right -= 1
    top = 0
    while top < height - 1 and row_is_black(top):
        top += 1
    bottom = height - 1
    while bottom > top and row_is_black(bottom):
        bottom -= 1
    return Region(x=left, y=top, width=right - left + 1, height=bottom - top + 1)


def _runs(flags: List[bool], min_length: int, max_gap: int) -> List[Tuple[int, int]]:
    """Index ranges where `flags` is True, bridging gaps up to `max_gap`."""
    runs = []
    start = None
    gap = 0
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, index - gap))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, len(flags) - 1 - gap))
    return [(a, b) for a, b in runs if b - a + 1 >= min_length]


def _bright_panels(image, content: Region) -> List[Region]:
    """Every tall, bright panel in the frame, as candidate log panes."""
    pixels = image.load()
    x0, x1 = content.x, content.x + content.width - 1
    y0, y1 = content.y, content.y + content.height - 1
    xstep = max(1, content.width // 640)
    ystep = max(1, content.height // 240)

    # How much of each column is bright, over the whole client rather than a
    # band partway down it: the log pane can be docked anywhere and sized to
    # anything, and a band assumes it covers one particular stripe.
    rows = list(range(y0, y1 + 1, ystep))
    column_bright = []
    for x in range(x0, x1 + 1):
        lit = sum(1 for y in rows if pixels[x, y] > _BRIGHT)
        column_bright.append(lit > len(rows) * _MIN_PANE_HEIGHT_FRAC)

    panels = []
    for left, right in _runs(column_bright,
                             int(content.width * _MIN_PANE_WIDTH_FRAC),
                             max_gap=max(2, content.width // 200)):
        left += x0
        right += x0
        row_bright = []
        for y in range(y0, y1 + 1):
            values = [pixels[x, y] for x in range(left, right + 1, xstep)]
            row_bright.append(sum(values) / len(values) > _BRIGHT)
        vertical = _runs(row_bright, int(content.height * _MIN_PANE_HEIGHT_FRAC),
                         max_gap=max(4, content.height // 100))
        if not vertical:
            continue
        top = min(a for a, _ in vertical) + y0
        bottom = max(b for _, b in vertical) + y0
        right = _trim_scrollbar(image, left, right, top, bottom, ystep)
        if right - left < 20:
            continue
        panels.append(Region(x=left, y=top, width=right - left + 1,
                             height=bottom - top + 1))
    return panels


def _trim_scrollbar(image, left: int, right: int, top: int, bottom: int,
                    ystep: int) -> int:
    """Cut a panel's right edge in front of its scrollbar.

    The scrollbar's arrows and thumb OCR as junk glued onto the end of every
    wrapped line, and the text-column measurement cannot exclude them on its
    own: a stray glyph read inside the band is a word like any other. It
    cannot be trimmed from the outside in either -- the panel draws a bright
    border column *outside* the scrollbar, so scanning inwards stops
    immediately. Find the darker band and cut in front of it.
    """
    pixels = image.load()
    means = []
    for x in range(left, right + 1):
        values = [pixels[x, y] for y in range(top, bottom + 1, ystep)]
        means.append(sum(values) / len(values))
    if not means:
        return right
    dark = max(means) - 40
    width = right - left + 1
    search_from = max(0, width - max(4, int(width * 0.16)))
    band_start = None
    for index in range(width - 1, search_from - 1, -1):
        if means[index] < dark:
            band_start = index
        elif band_start is not None and index < band_start - 1:
            break
    return left + band_start - 1 if band_start else right


def _pane_crop(image, pane: Region, out_path: str, target_width: int = 1035):
    """The pane, upscaled the way the ingest pipeline upscales it for OCR."""
    factor = max(1.0, target_width / max(1, pane.width))
    crop = image.crop((pane.x, pane.y, pane.x + pane.width,
                       pane.y + pane.height))
    if factor > 1.0:
        from PIL import Image

        crop = crop.resize((int(round(crop.width * factor)),
                            int(round(crop.height * factor))), Image.LANCZOS)
    crop.save(out_path)
    return factor


def _read_pane(image, pane: Region) -> Tuple[int, int, List[Word], float]:
    """Read a panel: (timestamps, log verbs, word boxes, upscale factor)."""
    work = tempfile.mkdtemp(prefix="mtgo_pane_")
    try:
        path = os.path.join(work, "pane.png")
        factor = _pane_crop(image, pane, path)
        text = " ".join(ocr_image(path, strict=False))
        return (len(_TIMESTAMP_HINT.findall(text)),
                len(_LOG_VERBS.findall(text)),
                ocr_words(path, psm=6), factor)
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def _text_bounds(words: Sequence[Word], factor: float, pane: Region,
                 ) -> Optional[Region]:
    """Where the text actually sits inside a pane, in source pixels.

    This is what keeps the scrollbar out of the crop. The previous version
    guessed the scrollbar's width from the resolution and the toolbar's
    height from the reference capture; both are properties of how the client
    was scaled, not of the frame size. Measuring the text column instead
    works at any client scale, and needs no assumption about which chrome the
    pane happens to have.
    """
    solid = [w for w in words if len(w.text.strip()) >= 3 and w.conf > 30]
    if len(solid) < 4:
        return None
    heights = sorted(w.height for w in solid)
    line_height = heights[len(heights) // 2]
    pad = max(2, int(round(line_height * 0.35)))
    left = min(w.x for w in solid) - pad
    right = max(w.right for w in solid) + pad
    top = min(w.y for w in solid) - pad
    bottom = max(w.bottom for w in solid) + pad
    # The pad must not push the crop back out into the scrollbar the panel
    # bounds already excluded.
    x0 = pane.x + max(0, int(round(left / factor)))
    y0 = pane.y + max(0, int(round(top / factor)))
    x1 = min(pane.x + pane.width, pane.x + int(round(right / factor)))
    y1 = min(pane.y + pane.height, pane.y + int(round(bottom / factor)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return Region(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def detect_log_pane(image, content: Region) -> Optional[Region]:
    """Find the game-log text column among the frame's bright panels.

    Brightness alone is not enough on a streamed capture: a facecam, a
    sponsor graphic and a scoreboard are all bright panels, and "the
    rightmost one" is whichever the streamer happened to put on the right.
    So every candidate is read, and the one whose text looks like a game log
    wins.
    """
    best = None
    for pane in _bright_panels(image, content):
        # Measure the text column *before* judging the panel. A panel whose
        # bright bounds include the pane's border starts a few pixels inside
        # the timestamps, which reads as "9:41 AM" -> "1] iM" and scores as
        # having no timestamps at all -- rejecting the real log pane for the
        # very reason the refinement exists to fix.
        _, _, words, factor = _read_pane(image, pane)
        refined = _text_bounds(words, factor, pane) or pane
        stamps, verbs, _, _ = _read_pane(image, refined)
        if stamps == 0:
            continue
        score = (stamps * 2 + verbs, refined.width * refined.height)
        if best is None or score > best[0]:
            best = (score, refined)
    return best[1] if best else None


def estimate_text_height(image, pane: Region) -> Optional[float]:
    """Measure the log's glyph height, in source pixels.

    Upscaling before OCR cannot add detail that the capture never had, so
    what matters is how tall the text is in the *original* frame. Measure it
    rather than infer it from the resolution: a windowed client at 1080p has
    smaller text than a full-screen one.
    """
    pixels = image.load()
    x0, x1 = pane.x, pane.x + pane.width - 1
    rows = []
    for y in range(pane.y, pane.y + pane.height):
        dark = sum(1 for x in range(x0, x1 + 1) if pixels[x, y] < 128)
        rows.append(dark > 0)
    runs = []
    start = None
    for index, has_text in enumerate(rows):
        if has_text and start is None:
            start = index
        elif not has_text and start is not None:
            runs.append(index - start)
            start = None
    if start is not None:
        runs.append(len(rows) - start)
    runs = [r for r in runs if 2 <= r <= pane.height * 0.2]
    if not runs:
        return None
    runs.sort()
    return float(runs[len(runs) // 2])


# --------------------------------------------------------------------------
# Putting a layout together
# --------------------------------------------------------------------------

def _too_small_note(image, content: Region,
                    pane: Optional[Region] = None) -> List[str]:
    """Say so when nothing was found because the client is drawn too small.

    "No panel read as a game log" is the same message whether the log is
    somewhere unexpected or whether the capture simply has no readable text
    in it, and the two call for opposite responses from whoever ran it. The
    brightest panel's glyph height tells them apart.
    """
    if pane is None:
        panels = _bright_panels(image, content)
        if not panels:
            return []
        pane = max(panels, key=lambda p: p.width * p.height)
    height = estimate_text_height(image, pane)
    if height is not None and height < MIN_RELIABLE_TEXT_HEIGHT:
        return ["the client's text is only ~%.0fpx tall in this capture "
                "(want >=%.0f): it is recorded too small to read, whatever "
                "the frame size is" % (height, MIN_RELIABLE_TEXT_HEIGHT)]
    return []


def _scaled(content: Region, scale: float, x: float, y: float,
            width: float, height: float) -> Region:
    return Region(
        x=content.x + int(round(x * scale)),
        y=content.y + int(round(y * scale)),
        width=max(1, int(round(width * scale))),
        height=max(1, int(round(height * scale))),
    )


def build_layout(frame_path: str, detect: bool = True) -> Layout:
    """Locate the UI in one frame."""
    image = _load_gray(frame_path)
    content = detect_content_area(image)
    scale = content.width / REFERENCE_WIDTH
    notes: List[str] = []

    pane = detect_log_pane(image, content) if detect else None
    detected = pane is not None
    if pane is None:
        if detect:
            notes.append("no panel read as a game log; using reference offsets")
            notes.extend(_too_small_note(image, content))
        pane = _scaled(content, scale, 1544, 88, 345, 445)

    seats = None
    bar = None
    if detect:
        bar, words = find_phase_bar(frame_path, image.size[1])
        if bar is None:
            notes.append("no phase bar: this frame is not a duel scene")
            notes.extend(_too_small_note(image, content,
                                         pane if detected else None))
        else:
            seats = detect_seats(image, bar, words)
            if seats is None:
                # A broadcast skin draws the seat plates dark enough that
                # their white text only reads inverted.
                seats = detect_seats(image, bar,
                                     _merge_words(words + _inverted_words(frame_path)))
            if seats is None:
                notes.append("phase bar found but no seat panel pair beside it")

    if seats is None:
        life, name = _LIFE_REF, _NAME_REF
        seats = {
            "life_top": _scaled(content, scale, life["x"], life["y_top"],
                                life["w"], life["h"]),
            "life_bottom": _scaled(content, scale, life["x"], life["y_bottom"],
                                   life["w"], life["h"]),
            "name_top": _scaled(content, scale, name["x"], name["y_top"],
                                name["w"], name["h"]),
            "name_bottom": _scaled(content, scale, name["x"], name["y_bottom"],
                                   name["w"], name["h"]),
        }
        seats_detected = False
    else:
        seats_detected = True

    return Layout(
        frame_width=image.size[0],
        frame_height=image.size[1],
        content=content,
        log_pane=pane,
        life_top=seats["life_top"],
        life_bottom=seats["life_bottom"],
        name_top=seats["name_top"],
        name_bottom=seats["name_bottom"],
        scale=scale,
        detected=detected,
        phase_bar=bar,
        seats_detected=seats_detected,
        notes=notes,
    )


def detect_layout(frame_path: str, detect: bool = True) -> Layout:
    return build_layout(frame_path, detect=detect)


def _median(values: List[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _median_region(regions: Sequence[Region]) -> Region:
    left = _median([r.x for r in regions])
    top = _median([r.y for r in regions])
    return Region(x=left, y=top,
                  width=max(8, _median([r.x + r.width for r in regions]) - left),
                  height=max(8, _median([r.y + r.height for r in regions]) - top))


def _union_region(regions: Sequence[Region]) -> Region:
    left = min(r.x for r in regions)
    top = min(r.y for r in regions)
    return Region(x=left, y=top,
                  width=max(8, max(r.x + r.width for r in regions) - left),
                  height=max(8, max(r.y + r.height for r in regions) - top))


def detect_layout_from_frames(frame_paths: List[str],
                              detect: bool = True) -> Layout:
    """Agree a layout across several frames.

    A single frame is a poor witness: early in a game the log has not
    overflowed, so its text column is measured shorter than it will be, and a
    seat's life numeral can be hidden behind a card being dragged over it.
    The UI itself does not move, so take each edge's median across whichever
    samples saw the thing at all.
    """
    layouts = [build_layout(path, detect=detect) for path in frame_paths]
    detected = [l for l in layouts if l.detected]
    base = detected[0] if detected else layouts[0]

    if len(detected) >= 2:
        # The union of the text seen, not the median of it. Each frame's
        # bounds are measured from the words in it, so they only ever
        # *under*-state the column: a frame whose lines all happen to be
        # short reads a narrower pane than the pane is, and cropping to that
        # clips the last character off every full-width line ("fror" for
        # "from"). Text cannot appear outside the column, so the widest
        # sighting is the column.
        left = min(l.log_pane.x for l in detected)
        right = max(l.log_pane.x + l.log_pane.width for l in detected)
        top = min(l.log_pane.y for l in detected)
        bottom = max(l.log_pane.y + l.log_pane.height for l in detected)
        base.log_pane = Region(x=left, y=top, width=max(20, right - left),
                               height=max(20, bottom - top))
        base.samples = len(detected)

    seated = [l for l in layouts if l.seats_detected]
    if seated:
        # The union again, and for a sharper version of the same reason. A
        # life box is trimmed to the numeral currently in it, and that
        # numeral changes: a player on 9 has a one-digit box and the same
        # player on 16 a two-digit one. Taking the median of those crops the
        # box to something that fits neither, and a crop that clips a digit
        # reads 16 as 1 -- a wrong answer, where a slightly generous crop is
        # only a slightly noisier one.
        base.life_top = _union_region([l.life_top for l in seated])
        base.life_bottom = _union_region([l.life_bottom for l in seated])
        # A name, unlike a life total, does not change while the game runs,
        # so its box has no legitimate reason to differ between frames --
        # which means an outlier is an outlier, and the median is what
        # discards it. Unioning names instead grows the crop around whatever
        # one frame mistook for a name, and it stops reading.
        base.name_top = _median_region([l.name_top for l in seated])
        base.name_bottom = _median_region([l.name_bottom for l in seated])
        base.seats_detected = True
        base.phase_bar = base.phase_bar or seated[0].phase_bar
        base.notes = [n for n in base.notes if "seat" not in n]
    return base


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_layout(layout: Layout, frame_path: str,
                    ffmpeg: str = "ffmpeg") -> Dict:
    """Read the located regions back and report whether they make sense.

    A detector that silently returns the wrong rectangle is worse than one
    that fails: the pipeline would produce a confident, wrong log. So the log
    pane has to actually contain timestamped lines.

    The seats are held to the same standard -- *both* must show a life total,
    since one alone used to be enough and let a layout pass while pointing the
    other seat at a mana pip -- but failing it is not fatal. The log is what
    the decode is made of; the seats are what two of the five verifications
    are made of. A capture whose text is large enough to read but whose life
    numerals are not still decodes into a game, and refusing it outright threw
    away work over a check it was never going to be able to run. So an
    unreadable seat is reported, recorded, and left to the checks that need
    it to refuse.
    """
    import subprocess

    from .extract import OCR_TARGET_WIDTH
    from .hud import grab, read_int
    from .ocr import ocr_image

    problems = []
    warnings = []
    work = os.path.realpath(tempfile.mkdtemp(prefix="mtgo_layout_"))
    try:
        # The log pane is greyscale text on white and must not be binarized;
        # the HUD numerals are near-white on art and must be.
        log_crop = os.path.join(work, "log.png")
        factor = max(1.0, OCR_TARGET_WIDTH / max(1, layout.log_pane.width))
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", frame_path,
             "-vf", "%s,scale=%d:%d:flags=lanczos,format=gray"
             % (layout.log_pane.ffmpeg_crop(),
                int(round(layout.log_pane.width * factor)),
                int(round(layout.log_pane.height * factor))),
             log_crop], check=True)
        log_text = " ".join(ocr_image(log_crop, strict=False))
        stamps = len(_TIMESTAMP_HINT.findall(log_text))
        if stamps == 0:
            problems.append("log pane has no timestamped lines")

        text_height = estimate_text_height(_load_gray(frame_path), layout.log_pane)
        if text_height is not None and text_height < MIN_RELIABLE_TEXT_HEIGHT:
            warnings.append(
                "log text is only ~%.0fpx tall (want >=%.0f); this capture is "
                "below the resolution where OCR is dependable"
                % (text_height, MIN_RELIABLE_TEXT_HEIGHT))

        lives = {}
        for label, region in (("top", layout.life_top),
                              ("bottom", layout.life_bottom)):
            path = grab(frame_path, None, region,
                        os.path.join(work, "life_%s.png" % label), ffmpeg=ffmpeg)
            # Read it exactly the way the crosscheck will: same crop, same
            # vote across thresholds. A validation that reads more leniently
            # than the pipeline would pass layouts the pipeline cannot use.
            lives[label] = read_int(path)
        unreadable = [label for label, value in lives.items() if value is None]
        if unreadable:
            warnings.append(
                "no life total readable at %s seat%s: this capture can be "
                "decoded but not cross-checked against the HUD"
                % (" and ".join(unreadable), "s" if len(unreadable) > 1 else ""))

        names = {}
        for label, region in (("top", layout.name_top),
                              ("bottom", layout.name_bottom)):
            path = grab(frame_path, None, region,
                        os.path.join(work, "name_%s.png" % label), ffmpeg=ffmpeg)
            lines = ocr_image(path, psm=7, strict=False)
            names[label] = lines[0].strip() if lines else ""
        if not any(names.values()):
            # Not fatal: names are used to map panels to seats, and the
            # crosscheck reports it if that mapping fails.
            warnings.append("no player name readable under either seat")

        return {"logTimestamps": stamps, "life": lives, "names": names,
                "textHeight": text_height,
                "seatsDetected": layout.seats_detected,
                "seatsReadable": not unreadable,
                "logDetected": layout.detected,
                "notes": layout.notes,
                "problems": problems, "warnings": warnings,
                "ok": not problems}
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
