"""Read MTGO's on-screen player HUD straight from the footage.

The decoded log is a *derivation*: life totals come from applying damage and
life-change lines in order. MTGO also prints each player's life on screen,
which is an independent ground truth from the same frames. Comparing the two
catches a dropped or misread log line, which log-vs-XMage agreement cannot:
XMage faithfully renders whatever the log claimed, right or wrong.

Only life is read. The library/hand badges are ~10px tall inside a shaped
icon and do not OCR reliably at 1080p, so they are deliberately not used
rather than reported at low confidence.
"""

import re
import subprocess
import tempfile
import os
from typing import Dict, List, Optional

from .ocr import ocr_image
from .regions import Region

# Player panels in a 1920x1080 MTGO capture: opponent top, local player
# bottom. Offsets below are relative to each panel's origin.
PANEL_TOP_Y = 20
PANEL_BOTTOM_Y = 600

# The life total is right-aligned over the avatar art. Keep the crop tight:
# the local player's art is bright and busy, and anything to the left of the
# digits gets read as a stray numeral.
LIFE_OFFSET = Region(x=112, y=100, width=80, height=48)
NAME_OFFSET = Region(x=45, y=148, width=170, height=18)


def _at(offset: Region, panel_y: int) -> Region:
    return Region(x=offset.x, y=panel_y + offset.y,
                  width=offset.width, height=offset.height)


def life_regions() -> Dict[str, Region]:
    return {"top": _at(LIFE_OFFSET, PANEL_TOP_Y),
            "bottom": _at(LIFE_OFFSET, PANEL_BOTTOM_Y)}


def name_regions() -> Dict[str, Region]:
    return {"top": _at(NAME_OFFSET, PANEL_TOP_Y),
            "bottom": _at(NAME_OFFSET, PANEL_BOTTOM_Y)}


def grab(video: str, timestamp: float, region: Region, out_path: str,
         scale: int = 6, invert: bool = True, ffmpeg: str = "ffmpeg") -> str:
    """Extract one upscaled, grayscale crop at a video timestamp."""
    filters = [
        region.ffmpeg_crop(),
        "scale=%d:%d:flags=lanczos" % (region.width * scale, region.height * scale),
        "format=gray",
    ]
    if invert:
        # MTGO's HUD is light text on a dark panel; tesseract is trained for
        # the opposite.
        filters.append("negate")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-ss", str(timestamp),
         "-i", video, "-frames:v", "1", "-vf", ",".join(filters), out_path],
        check=True,
    )
    return out_path


def _ocr_int(path: str) -> Optional[int]:
    lines = ocr_image(path, psm=7, strict=False, whitelist="0123456789")
    match = re.search(r"\d+", " ".join(lines))
    return int(match.group(0)) if match else None


def read_life(video: str, timestamp: float, work_dir: Optional[str] = None,
              ffmpeg: str = "ffmpeg") -> Dict[str, Optional[int]]:
    """Life totals shown on screen at a timestamp, as {"top":n,"bottom":n}."""
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        out = {}
        for slot, region in life_regions().items():
            path = os.path.join(os.path.realpath(work_dir), "life_%s.png" % slot)
            grab(video, timestamp, region, path, ffmpeg=ffmpeg)
            out[slot] = _ocr_int(path)
        return out
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def read_names(video: str, timestamp: float, work_dir: Optional[str] = None,
               ffmpeg: str = "ffmpeg") -> Dict[str, str]:
    """Player names under each panel, used to map panels to seats."""
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        out = {}
        for slot, region in name_regions().items():
            path = os.path.join(os.path.realpath(work_dir), "name_%s.png" % slot)
            grab(video, timestamp, region, path, ffmpeg=ffmpeg)
            lines = ocr_image(path, psm=7, strict=False)
            out[slot] = (lines[0].strip() if lines else "")
        return out
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def align_derived_states(video: str, states: List[dict], kinds=("COMBAT_DAMAGE",),
                         max_lookahead: float = 40.0, step: float = 1.0,
                         work_dir: Optional[str] = None) -> Dict:
    """Timestamp derived state changes by finding them on screen.

    Some state changes are inferred rather than logged -- combat damage is
    the main one, since MTGO prints no line for it. The log therefore says
    *that* the change happened but not *when*: all we know is that it falls
    somewhere between the attack and the next logged line, which can be tens
    of seconds later.

    Rather than guess an offset (the gap is player-paced -- measured at 4-5s
    here, but it is however long a human takes to click through the damage
    step), scan forward from the attack for the first frame whose on-screen
    life matches the derived value, and use that. Failing to find it inside
    the window is reported: it means the derivation is unconfirmed.
    """
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        slots = None
        aligned, unconfirmed = [], []
        for index, state in enumerate(states):
            if (state.get("sourceEvent") or {}).get("type") not in kinds:
                continue
            start = state.get("videoTime")
            if start is None:
                continue
            if slots is None:
                slots = map_slots_to_seats(video, start, state.get("players", []),
                                           work_dir)
                if len(slots) < 2:
                    slots = None
                    continue
            seat_to_slot = {seat: slot for slot, seat in slots.items()}
            expected = {p["seat"]: p.get("life") for p in state.get("players", [])}

            found = None
            when = start
            limit = start + max_lookahead
            while when <= limit:
                shown = read_life(video, when, work_dir=work_dir)
                if all(shown.get(seat_to_slot.get(seat)) == life
                       for seat, life in expected.items()
                       if seat in seat_to_slot):
                    found = when
                    break
                when += step
            if found is None:
                unconfirmed.append({"stateIndex": index, "videoTime": start,
                                    "expected": expected})
            else:
                aligned.append({"stateIndex": index, "from": start, "to": found})
                state["videoTime"] = found
                state["videoTimeSource"] = "hud"
        return {"aligned": aligned, "unconfirmed": unconfirmed,
                "ok": not unconfirmed}
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def crosscheck_life(video: str, states: List[dict], sample: int = 1,
                    settle: float = 0.5, work_dir: Optional[str] = None) -> Dict:
    """Compare each state's derived life totals against MTGO's own display.

    `settle` shifts the sample slightly later than the log line's first
    sighting, since the HUD and the log pane do not repaint on the same
    frame. Frames whose life could not be read are counted separately from
    genuine disagreements: an unreadable frame is missing evidence, not
    contradicting evidence.
    """
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        checked = timed = agreed = unreadable = 0
        disagreements = []
        slots = None
        for index, state in enumerate(states):
            if index % sample:
                continue
            when = state.get("videoTime")
            if when is None:
                continue
            timed += 1
            if slots is None:
                slots = map_slots_to_seats(video, when + settle,
                                           state.get("players", []), work_dir)
                if len(slots) < 2:
                    slots = None
                    continue
            shown = read_life(video, when + settle, work_dir=work_dir)
            by_seat = {p["seat"]: p for p in state.get("players", [])}
            for slot, seat in slots.items():
                player = by_seat.get(seat)
                if player is None:
                    continue
                checked += 1
                if shown.get(slot) is None:
                    unreadable += 1
                elif shown[slot] == player.get("life"):
                    agreed += 1
                else:
                    disagreements.append({
                        "stateIndex": index,
                        "videoTime": when,
                        "player": player.get("name"),
                        "log": player.get("life"),
                        "onScreen": shown[slot],
                        "event": (state.get("sourceEvent") or {}).get("text"),
                    })
        readable = checked - unreadable
        return {
            "statesSampled": timed,
            "lifeReadingsChecked": checked,
            "lifeReadingsUnreadable": unreadable,
            "lifeReadingsAgreed": agreed,
            "agreementRate": round(agreed / readable, 4) if readable else None,
            "disagreements": disagreements,
            "ok": not disagreements,
        }
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def map_slots_to_seats(video: str, timestamp: float, players: List[dict],
                       work_dir: Optional[str] = None) -> Dict[str, int]:
    """Match each HUD panel to a seat number by reading the player names."""
    import difflib

    names = read_names(video, timestamp, work_dir=work_dir)
    mapping = {}
    for slot, shown in names.items():
        best, score = None, 0.0
        for player in players:
            ratio = difflib.SequenceMatcher(
                None, shown.lower(), (player.get("name") or "").lower()).ratio()
            if ratio > score:
                best, score = player["seat"], ratio
        if best is not None and score >= 0.5:
            mapping[slot] = best
    return mapping
