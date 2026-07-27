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


def life_regions(layout=None) -> Dict[str, Region]:
    if layout is not None:
        return {"top": layout.life_top, "bottom": layout.life_bottom}
    return {"top": _at(LIFE_OFFSET, PANEL_TOP_Y),
            "bottom": _at(LIFE_OFFSET, PANEL_BOTTOM_Y)}


def name_regions(layout=None) -> Dict[str, Region]:
    if layout is not None:
        return {"top": layout.name_top, "bottom": layout.name_bottom}
    return {"top": _at(NAME_OFFSET, PANEL_TOP_Y),
            "bottom": _at(NAME_OFFSET, PANEL_BOTTOM_Y)}


# Life totals and player names are drawn in near-white over the avatar art.
# Thresholding on that separates the glyphs from a busy, brightly coloured
# background far better than inverting the whole crop, which at lower
# capture resolutions leaves the digits swimming in mid-grey.
_TEXT_WHITE = 190
_MIN_CROP_WIDTH = 240


def grab(video: str, timestamp: Optional[float], region: Region, out_path: str,
         scale: Optional[int] = None, threshold: Optional[int] = None,
         ffmpeg: str = "ffmpeg") -> str:
    """Extract one upscaled crop, binarized if a threshold is given.

    `timestamp` is video seconds; pass None when the source is a still.

    Numerals are read from an *unbinarized* crop and thresholded afterwards
    (see `read_int`), because which threshold separates white glyphs from the
    art behind them depends on the art: a life total over a dark avatar and
    one over a bright one do not survive the same cut. Keeping the grey crop
    lets several be tried without going back to the video.
    """
    factor = scale if scale else max(6.0, _MIN_CROP_WIDTH / max(1, region.width))
    filters = [
        region.ffmpeg_crop(),
        "scale=%d:%d:flags=lanczos" % (int(round(region.width * factor)),
                                       int(round(region.height * factor))),
        "format=gray",
    ]
    if threshold:
        filters.append("lut=y='if(gt(val,%d),255,0)'" % threshold)
        filters.append("negate")  # tesseract expects dark text on light
    command = [ffmpeg, "-y", "-loglevel", "error"]
    if timestamp is not None:
        command += ["-ss", str(timestamp)]
    command += ["-i", video, "-frames:v", "1", "-vf", ",".join(filters), out_path]
    subprocess.run(command, check=True)
    return out_path


# Thresholds and segmentation modes to read a numeral under. No single
# setting reads every seat: a life total over dark art wants a low cut, one
# over bright art a high one, and tesseract's line vs. word vs. raw-line
# segmentation each fail on different glyph spacings. Reading under all of
# them and taking the majority is the same defence the log reconstruction
# uses -- redundancy, not a tuned constant.
_THRESHOLDS = (140, 190, 230)
_NUMBER_PSMS = (7, 8, 13)


def read_int(path: str, maximum: int = 99) -> Optional[int]:
    """The number in an unbinarized crop, by majority vote across readings.

    Returns None when nothing legible is there, or when the readings do not
    agree on one value -- a split vote is missing evidence, and the callers
    treat it as such rather than as a disagreement with the log.
    """
    votes = read_int_votes(path, maximum=maximum)
    if not votes:
        return None
    best, count = max(votes.items(), key=lambda item: (item[1], -item[0]))
    if count < 2 and len(votes) > 1:
        return None
    return best


def read_int_votes(path: str, maximum: int = 99, quorum: int = 2) -> Dict[int, int]:
    """How many readings of this crop produced each candidate number.

    Stops as soon as `quorum` readings agree and nothing disagrees: an
    unambiguous numeral is the common case, and reading it nine ways to
    confirm what the first two already agreed on makes a HUD crosscheck of a
    whole game several times slower for no extra evidence.
    """
    from PIL import Image, ImageOps

    base = Image.open(path).convert("L")
    votes: Dict[int, int] = {}
    work = tempfile.mkdtemp(prefix="mtgo_int_")
    try:
        # Threshold varies fastest, so the first readings to be compared are
        # of *differently binarized* images. Two segmentation modes reading
        # the same binarization agree or disagree together, so they are not
        # independent evidence and quorum must not be reached on them.
        for psm in _NUMBER_PSMS:
            for threshold in _THRESHOLDS:
                shot = os.path.join(work, "t%d.png" % threshold)
                if not os.path.exists(shot):
                    ImageOps.invert(
                        base.point(lambda v, t=threshold: 255 if v > t else 0)
                    ).save(shot)
                lines = ocr_image(shot, psm=psm, strict=False,
                                  whitelist="0123456789")
                match = re.search(r"\d+", " ".join(lines))
                if not match:
                    continue
                value = int(match.group(0))
                if 0 <= value <= maximum:
                    votes[value] = votes.get(value, 0) + 1
            if len(votes) == 1 and max(votes.values()) >= quorum:
                return votes
        return votes
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def read_life(video: str, timestamp: float, work_dir: Optional[str] = None,
              ffmpeg: str = "ffmpeg", layout=None) -> Dict[str, Optional[int]]:
    """Life totals shown on screen at a timestamp, as {"top":n,"bottom":n}."""
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        out = {}
        for slot, region in life_regions(layout).items():
            path = os.path.join(os.path.realpath(work_dir), "life_%s.png" % slot)
            grab(video, timestamp, region, path, ffmpeg=ffmpeg)
            out[slot] = read_int(path)
        return out
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def read_names(video: str, timestamp: float, work_dir: Optional[str] = None,
               ffmpeg: str = "ffmpeg", layout=None) -> Dict[str, str]:
    """Player names under each panel, used to map panels to seats."""
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        out = {}
        for slot, region in name_regions(layout).items():
            path = os.path.join(os.path.realpath(work_dir), "name_%s.png" % slot)
            grab(video, timestamp, region, path, threshold=_TEXT_WHITE,
                 ffmpeg=ffmpeg)
            lines = ocr_image(path, psm=7, strict=False)
            out[slot] = (lines[0].strip() if lines else "")
        return out
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


# The events that move a life total. Everything else -- a land, a draw, a
# cantrip -- leaves it alone, which is what makes the span up to the next one
# of these a window in which the screen can only be showing *this* combat's
# result.
_LIFE_CHANGING = ("COMBAT_DAMAGE", "DAMAGE", "LOSE_LIFE", "GAIN_LIFE",
                  "LIFE_TOTAL")


def _next_life_change(states: List[dict], index: int) -> Optional[float]:
    """When something else next changes a life total, if anything does."""
    for state in states[index + 1:]:
        if (state.get("sourceEvent") or {}).get("type") in _LIFE_CHANGING:
            return state.get("videoTime")
    return None


def _settled_life(video: str, states: List[dict], index: int, start: float,
                  work_dir: str, layout, samples: int = 3,
                  max_window: float = 240.0):
    """The life the screen has settled on before anything else changes it.

    Combat damage can land long after the attack -- the defender may take a
    whole turn first -- so the value has to be read at the *end* of the
    window, not shortly after the attack. Several readings must agree, so a
    frame caught mid-animation cannot define it.
    """
    end = _next_life_change(states, index)
    end = min(end, start + max_window) if end else start + max_window
    if end <= start + 2:
        return None
    readings = []
    for offset in range(samples):
        when = end - 2.0 - offset * 3.0
        if when <= start:
            return None
        shown = read_life(video, when, work_dir=work_dir, layout=layout)
        if any(value is None for value in shown.values()):
            return None
        readings.append(shown)
    if any(reading != readings[0] for reading in readings[1:]):
        return None  # still moving; no settled value to adopt
    return end - 2.0, readings[0]


def _correct_from_hud(states: List[dict], index: int, seat_to_slot: Dict[int, str],
                      settled) -> Optional[Dict]:
    """Adopt the life the footage shows, and carry it into the later states.

    Only accepted when the reading is coherent with a combat that dealt
    *less* than was derived, which is the failure MTGO's incomplete logging
    produces: an attacker removed mid-combat deals nothing, and no line says
    so. A reading that shows more damage than the attack could deal is not a
    quieter version of that story, so it is left unconfirmed instead.
    """
    if settled is None:
        return None
    when, shown = settled
    state = states[index]
    deltas = {}
    for player in state.get("players", []):
        seat = player["seat"]
        observed = shown.get(seat_to_slot.get(seat))
        if observed is None:
            return None
        derived = player.get("life")
        if derived is None:
            return None
        delta = observed - derived
        if delta < 0:
            return None  # more damage than the attack can explain
        deltas[seat] = delta
    if not any(deltas.values()):
        return None

    for later in states[index:]:
        for player in later.get("players", []):
            delta = deltas.get(player["seat"])
            if delta and player.get("life") is not None:
                player["life"] += delta
    state["lifeSource"] = "hud"
    state["videoTime"] = when
    state["videoTimeSource"] = "hud"
    return {"stateIndex": index, "videoTime": when,
            "correctedBy": {str(seat): delta for seat, delta in deltas.items()
                            if delta},
            "life": {str(p["seat"]): p.get("life")
                     for p in state.get("players", [])}}


def align_derived_states(video: str, states: List[dict], kinds=("COMBAT_DAMAGE",),
                         max_lookahead: float = 40.0, step: float = 1.0,
                         work_dir: Optional[str] = None, layout=None) -> Dict:
    """Timestamp derived state changes by finding them on screen.

    Some state changes are inferred rather than logged -- combat damage is
    the main one, since MTGO prints no line for it. The log therefore says
    *that* the change happened but not *when*: all we know is that it falls
    somewhere between the attack and the next logged line, which can be tens
    of seconds later.

    Rather than guess an offset (the gap is player-paced -- measured at 4-5s
    here, but it is however long a human takes to click through the damage
    step), scan forward from the attack for the first frame whose on-screen
    life matches the derived value, and use that.

    When no frame in the window shows the derived value, the derivation is
    wrong rather than merely mistimed -- and the footage says what actually
    happened. MTGO's log is incomplete about combat: it prints no damage
    line, and it prints nothing at all when an attacker is killed by damage
    mid-combat, so an attack the log describes fully can still deal less than
    the attackers' total power. The life total on screen is direct evidence
    where the derivation is inference, so the observed value is adopted, the
    difference is carried into every later state for that player, and the
    correction is recorded (`lifeSource: "hud"` on the state, and an entry in
    the report). A correction is not silent, and it is not a verification:
    `crosscheck` counts states corrected this way apart from states that
    agreed on their own.
    """
    own_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="mtgo_hud_")
    try:
        slots = None
        aligned, unconfirmed, corrected = [], [], []
        for index, state in enumerate(states):
            if (state.get("sourceEvent") or {}).get("type") not in kinds:
                continue
            start = state.get("videoTime")
            if start is None:
                continue
            if slots is None:
                slots = map_slots_to_seats(video, start, state.get("players", []),
                                           work_dir, layout=layout)
                if len(slots) < 2:
                    slots = None
                    continue
            seat_to_slot = {seat: slot for slot, seat in slots.items()}
            expected = {p["seat"]: p.get("life") for p in state.get("players", [])}

            found = None
            when = start
            limit = start + max_lookahead
            while when <= limit:
                shown = read_life(video, when, work_dir=work_dir, layout=layout)
                if all(shown.get(seat_to_slot.get(seat)) == life
                       for seat, life in expected.items()
                       if seat in seat_to_slot):
                    found = when
                    break
                when += step
            if found is not None:
                aligned.append({"stateIndex": index, "from": start, "to": found})
                state["videoTime"] = found
                state["videoTimeSource"] = "hud"
                continue
            settled = _settled_life(video, states, index, start, work_dir, layout)
            correction = _correct_from_hud(states, index, seat_to_slot, settled)
            if correction is None:
                unconfirmed.append({"stateIndex": index, "videoTime": start,
                                    "expected": expected})
            else:
                corrected.append(correction)
        return {"aligned": aligned, "corrected": corrected,
                "unconfirmed": unconfirmed,
                "ok": not unconfirmed}
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def crosscheck_life(video: str, states: List[dict], sample: int = 1,
                    settle: float = 0.5, work_dir: Optional[str] = None,
                    layout=None) -> Dict:
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
        checked = timed = agreed = unreadable = from_hud = 0
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
                                           state.get("players", []), work_dir,
                                           layout=layout)
                if len(slots) < 2:
                    slots = None
                    continue
            shown = read_life(video, when + settle, work_dir=work_dir,
                              layout=layout)
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
                    # A state whose life was *taken* from the HUD agrees with
                    # the HUD by construction. Counting it as independent
                    # agreement would make the check confirm its own input.
                    if state.get("lifeSource") == "hud":
                        from_hud += 1
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
        independent = readable - from_hud
        return {
            "statesSampled": timed,
            "lifeReadingsChecked": checked,
            "lifeReadingsUnreadable": unreadable,
            "lifeReadingsAgreed": agreed,
            "lifeReadingsTakenFromHud": from_hud,
            "agreementRate": round(agreed / readable, 4) if readable else None,
            "independentAgreementRate": (
                round((agreed - from_hud) / independent, 4) if independent else None),
            "disagreements": disagreements,
            "ok": not disagreements,
        }
    finally:
        if own_dir:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def map_slots_to_seats(video: str, timestamp: float, players: List[dict],
                       work_dir: Optional[str] = None, layout=None) -> Dict[str, int]:
    """Match each HUD panel to a seat number by reading the player names."""
    import difflib

    names = read_names(video, timestamp, work_dir=work_dir, layout=layout)
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
