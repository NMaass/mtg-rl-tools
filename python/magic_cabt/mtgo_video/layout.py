"""Locate MTGO's UI in a frame, at whatever resolution it was captured.

Hardcoded pixel regions only work for the one resolution they were measured
at. Everything downstream of the crop -- OCR, log reconstruction, the card
catalog, the simulator -- is already resolution-agnostic, so the whole
dependency on capture size lives here.

The game-log pane is the one region worth *detecting* rather than computing:
it is a near-white panel against a dark board (mean brightness ~250 vs ~80),
which makes it separable by a plain brightness profile at any size, and it
is the region whose exact bounds matter most. Its right edge also carries a
scrollbar whose arrows and thumb OCR as junk glued to the end of every
wrapped line, so the text column has to stop short of it.

Everything else is derived proportionally from the detected client area and
then *validated by reading it*, so a layout this module gets wrong is
reported rather than silently turned into a plausible-looking wrong log.
"""

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .regions import Region

# Reference geometry, measured on a full-screen 1920x1080 capture with the
# default duel scene. Stored as fractions of the client area so they carry
# to other resolutions; the log pane is detected instead of using these.
REFERENCE_WIDTH = 1920.0
REFERENCE_HEIGHT = 1080.0

_LIFE_REF = {"x": 112, "w": 80, "h": 48, "y_top": 120, "y_bottom": 704}
_NAME_REF = {"x": 45, "w": 170, "h": 18, "y_top": 168, "y_bottom": 748}

# A log-pane candidate must be at least this bright, and this much of the
# frame, to be believed.
_BRIGHT = 170
_MIN_PANE_WIDTH_FRAC = 0.10
_MIN_PANE_HEIGHT_FRAC = 0.15

_TIMESTAMP_HINT = re.compile(r"[\dOl|/It]{1,2}\s*[:.]\s*[\dO]{2}\s*[AP4/][A-Za-z]{0,2}")


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

    def to_dict(self) -> Dict:
        out = {k: (asdict(v) if isinstance(v, Region) else v)
               for k, v in asdict(self).items()}
        return out

    def describe(self) -> str:
        return ("%dx%d content=%dx%d+%d+%d log=%dx%d+%d+%d scale=%.3f %s"
                % (self.frame_width, self.frame_height,
                   self.content.width, self.content.height,
                   self.content.x, self.content.y,
                   self.log_pane.width, self.log_pane.height,
                   self.log_pane.x, self.log_pane.y,
                   self.scale, "detected" if self.detected else "proportional"))


def _load_gray(path: str):
    from PIL import Image

    return Image.open(path).convert("L")


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


def detect_log_pane(image, content: Region) -> Optional[Region]:
    """Find the game-log text column: the bright panel, minus its scrollbar."""
    pixels = image.load()
    x0, x1 = content.x, content.x + content.width - 1
    y0, y1 = content.y, content.y + content.height - 1
    xstep = max(1, content.width // 640)
    ystep = max(1, content.height // 240)

    # The pane spans a tall band on the right; sample the middle of the
    # frame vertically so board art and the hand row do not dominate.
    band_top = y0 + int(content.height * 0.10)
    band_bottom = y0 + int(content.height * 0.45)
    column_bright = []
    for x in range(x0, x1 + 1):
        values = [pixels[x, y] for y in range(band_top, band_bottom, ystep)]
        column_bright.append(sum(values) / len(values) > _BRIGHT)

    candidates = _runs(column_bright, int(content.width * _MIN_PANE_WIDTH_FRAC),
                       max_gap=max(2, content.width // 200))
    if not candidates:
        return None
    # The log pane is the rightmost such panel.
    left, right = candidates[-1]
    left += x0
    right += x0

    # Trim the scrollbar: a band of columns clearly darker than the white
    # text column, sitting near the pane's right edge. It cannot simply be
    # trimmed from the outside in -- the pane draws a bright border column
    # *outside* the scrollbar, so scanning inwards stops immediately. Find
    # the darker band itself and cut in front of it.
    text_values = []
    for x in range(left, right + 1):
        values = [pixels[x, y] for y in range(band_top, band_bottom, ystep)]
        text_values.append(sum(values) / len(values))
    peak = max(text_values)
    dark = peak - 40
    width = right - left + 1
    search_from = max(0, width - max(4, int(width * 0.16)))
    band_start = None
    for index in range(width - 1, search_from - 1, -1):
        if text_values[index] < dark:
            band_start = index
        elif band_start is not None and index < band_start - 1:
            break
    if band_start is not None:
        right = left + band_start - 1

    # Vertical extent, measured inside the text column.
    row_bright = []
    for y in range(y0, y1 + 1):
        values = [pixels[x, y] for x in range(left, right + 1, xstep)]
        row_bright.append(sum(values) / len(values) > _BRIGHT)
    vertical = _runs(row_bright, int(content.height * _MIN_PANE_HEIGHT_FRAC),
                     max_gap=max(4, content.height // 100))
    if not vertical:
        return None
    top = min(a for a, _ in vertical) + y0
    bottom = max(b for _, b in vertical) + y0

    # Inset by a pixel or two so the panel border never enters the crop.
    inset = max(1, int(round(content.width / REFERENCE_WIDTH * 3)))
    left += inset
    right -= inset
    bottom -= inset
    # The pane's top holds a toolbar strip (a filter dropdown at its right)
    # above the first log line. It is inside the white panel, so brightness
    # cannot separate it; skip it by the height it occupies at the reference
    # resolution.
    top += max(inset, int(round(content.width / REFERENCE_WIDTH * 26)))
    if right - left < 20 or bottom - top < 20:
        return None
    return Region(x=left, y=top, width=right - left + 1, height=bottom - top + 1)


# Below roughly this glyph height tesseract stops being dependable, whatever
# it is upscaled to afterwards: the information is not in the pixels.
MIN_RELIABLE_TEXT_HEIGHT = 9.0


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


def _scaled(content: Region, scale: float, x: float, y: float,
            width: float, height: float) -> Region:
    return Region(
        x=content.x + int(round(x * scale)),
        y=content.y + int(round(y * scale)),
        width=max(1, int(round(width * scale))),
        height=max(1, int(round(height * scale))),
    )


def build_layout(image, detect: bool = True) -> Layout:
    """Locate the UI in one frame."""
    content = detect_content_area(image)
    scale = content.width / REFERENCE_WIDTH

    pane = detect_log_pane(image, content) if detect else None
    detected = pane is not None
    if pane is None:
        pane = _scaled(content, scale, 1544 - 0, 88, 345, 445)

    life = _LIFE_REF
    name = _NAME_REF
    return Layout(
        frame_width=image.size[0],
        frame_height=image.size[1],
        content=content,
        log_pane=pane,
        life_top=_scaled(content, scale, life["x"], life["y_top"],
                         life["w"], life["h"]),
        life_bottom=_scaled(content, scale, life["x"], life["y_bottom"],
                            life["w"], life["h"]),
        name_top=_scaled(content, scale, name["x"], name["y_top"],
                         name["w"], name["h"]),
        name_bottom=_scaled(content, scale, name["x"], name["y_bottom"],
                            name["w"], name["h"]),
        scale=scale,
        detected=detected,
    )


def detect_layout(frame_path: str, detect: bool = True) -> Layout:
    return build_layout(_load_gray(frame_path), detect=detect)


def _median(values: List[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def looks_like_duel(frame_path: str, ffmpeg: str = "ffmpeg") -> bool:
    """Whether a frame shows the duel scene rather than a menu screen.

    A VOD spends a lot of its length on deck-building and sideboarding
    screens. Those have a game log too -- but a differently shaped one, in a
    different place -- so including them in the layout consensus produces a
    pane that matches neither. The duel scene is the one with player life
    totals on it, so read for those.
    """
    import os
    import re
    import tempfile

    from .hud import grab
    from .ocr import ocr_image

    layout = build_layout(_load_gray(frame_path))
    work = os.path.realpath(tempfile.mkdtemp(prefix="mtgo_duel_"))
    try:
        # Both seats, not either: a menu screen can happen to show a numeral
        # where one life total would be (a deck-editor column count, say),
        # but a duel always shows two.
        for region in (layout.life_top, layout.life_bottom):
            path = grab(frame_path, None, region,
                        os.path.join(work, "life.png"), ffmpeg=ffmpeg)
            text = " ".join(ocr_image(path, psm=7, strict=False,
                                      whitelist="0123456789"))
            match = re.search(r"\d+", text)
            if not (match and 0 <= int(match.group(0)) <= 99):
                return False
        return True
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def select_duel_frames(frame_paths: List[str]) -> List[str]:
    """The sampled frames that show a duel; all of them if none do."""
    duels = [path for path in frame_paths if looks_like_duel(path)]
    return duels or list(frame_paths)


def detect_layout_from_frames(frame_paths: List[str],
                              detect: bool = True) -> Layout:
    """Agree a layout across several frames.

    A single frame is a poor witness: the scrollbar only appears once the
    log has overflowed, and a short log leaves the lower pane blank, so the
    detected rectangle drifts with the game's progress. The UI itself does
    not move, so take each edge's median across samples.
    """
    layouts = [build_layout(_load_gray(path), detect=detect)
               for path in frame_paths]
    detected = [l for l in layouts if l.detected]
    base = detected[0] if detected else layouts[0]
    if len(detected) < 2:
        return base

    lefts = [l.log_pane.x for l in detected]
    tops = [l.log_pane.y for l in detected]
    rights = [l.log_pane.x + l.log_pane.width for l in detected]
    bottoms = [l.log_pane.y + l.log_pane.height for l in detected]
    # The narrowest credible right edge wins: a frame whose log has not yet
    # overflowed shows no scrollbar, and including that column band would
    # glue its glyphs onto the end of every wrapped line.
    right = _median(rights)
    left = _median(lefts)
    base.log_pane = Region(x=left, y=_median(tops),
                           width=max(20, min(right, min(rights)) - left),
                           height=max(20, _median(bottoms) - _median(tops)))
    base.samples = len(detected)
    return base


def validate_layout(layout: Layout, frame_path: str,
                    ffmpeg: str = "ffmpeg") -> Dict:
    """Read the located regions back and report whether they make sense.

    A detector that silently returns the wrong rectangle is worse than one
    that fails: the pipeline would produce a confident, wrong log. So the
    log pane has to actually contain timestamped lines, and the life
    positions have to actually contain small integers.
    """
    import os
    import subprocess
    import tempfile

    from .extract import OCR_TARGET_WIDTH
    from .hud import grab
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
            text = " ".join(ocr_image(path, psm=7, strict=False,
                                      whitelist="0123456789"))
            digits = re.search(r"\d+", text)
            lives[label] = int(digits.group(0)) if digits else None
        if all(v is None for v in lives.values()):
            problems.append("no life total readable at either seat")

        return {"logTimestamps": stamps, "life": lives,
                "textHeight": text_height,
                "problems": problems, "warnings": warnings,
                "ok": not problems}
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
