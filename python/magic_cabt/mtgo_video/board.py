"""Read the battlefield off the screen, instead of deriving it from the log.

Everything else in this pipeline is a *derivation*: the board is what folding
the log's events produces, and the log is what OCR read. That makes one whole
class of error invisible. A dropped "casts X" line leaves no trace anywhere
-- XMage renders the board it was given, nobody's life changes, and the event
stream stays internally consistent. The permanent is simply missing, and the
only witness that it ever existed is the picture.

So this module looks at the picture. It locates the two battlefield panels
from the phase bar the layout detector already found, segments the cards on
them, and identifies each by **its art** -- a perceptual signature of the art
box, compared against the same cards' art from Scryfall, restricted to the
few dozen cards this match has actually named. A card's *title* is drawn on
it too, and is read as a second opinion where the art says nothing, but only
there: at battlefield scale a name is about seven pixels tall and reads
correctly around half the time, while its art is a hundred pixels wide.

Nothing here is claimed on one frame. A permanent does not flicker -- it sits
in the same place for as long as it is in play -- so a card the log never
mentioned has to be seen in several sampled states, at the same place, before
it is reported as a finding. That is the same defence the log reconstruction
uses, for the same reason: one reading is a guess, and agreement across
readings is evidence.

Two things this deliberately does not do. It does not claim a card it cannot
identify: piles of overlapping duplicates and basic lands (whose printing is
unknown) come back anonymous rather than as the nearest guess. And it does
not treat "derived but not seen" as a defect, because plenty of real
permanents cannot be identified from a frame -- which makes the asymmetry
here honest rather than lazy: seeing something the log never mentioned is
evidence, not seeing something it did is not.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .regions import Region

# A card on the battlefield, as seen.
@dataclass
class SeenCard:
    box: Region
    title: Optional[str] = None       # what its title bar read as
    art: Optional[str] = None         # what its art hashed to
    art_distance: Optional[int] = None
    stacked: int = 1                  # how many cards this pile holds

    @property
    def name(self) -> Optional[str]:
        """The identity both readings support, or the one that spoke."""
        if self.title and self.art:
            return self.title if self.title == self.art else None
        return self.title or self.art

    @property
    def disputed(self) -> bool:
        return bool(self.title and self.art and self.title != self.art)

    def to_dict(self) -> Dict:
        return {"box": {"x": self.box.x, "y": self.box.y,
                        "width": self.box.width, "height": self.box.height},
                "title": self.title, "art": self.art,
                "artDistance": self.art_distance, "stacked": self.stacked,
                "name": self.name, "disputed": self.disputed}


# --------------------------------------------------------------------------
# Where the battlefields are
# --------------------------------------------------------------------------

def _mode(values: Sequence[int]) -> int:
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def play_area(image, layout, words=()) -> Optional[Region]:
    """The board: above the phase bar, between the seats and the log pane.

    Bounded by things already located rather than by a fresh pixel
    heuristic. The phase bar is drawn along the board's bottom edge and its
    row begins with "Turn 4: <player>" at the board's left edge, so the
    leftmost word on that row is the left edge. The right edge is whatever
    comes next -- the log pane if it is docked there, otherwise the client's
    own edge. Segmenting cards out of a *slightly* generous board is fine;
    guessing its sides from panel colour is not, because a card standing on a
    column looks exactly like not-the-board.
    """
    bar = getattr(layout, "phase_bar", None)
    if bar is None:
        return None
    top = layout.content.y
    bottom = bar.y - 2
    if bottom - top < 80:
        return None

    band = [w for w in words if abs(w.y - bar.y) <= max(8, bar.height)]
    left = min([w.x for w in band] + [bar.x]) - max(2, bar.height // 4)
    right = layout.content.x + layout.content.width - 1
    pane = layout.log_pane
    if layout.detected and pane.x > bar.x + bar.width // 2:
        right = min(right, pane.x - max(2, bar.height // 2))
    left = max(layout.content.x, left)
    # The seat panels are beside the board, not on it, and the bar's row runs
    # past them -- MTGO draws its settings icons down there. Whichever side
    # the seats are on, the board starts after them.
    if layout.seats_detected:
        seat_left = min(layout.life_top.x, layout.name_top.x)
        seat_right = max(layout.life_top.x + layout.life_top.width,
                         layout.name_top.x + layout.name_top.width)
        if seat_right < bar.x:
            left = max(left, seat_right + max(4, bar.height // 2))
        elif seat_left > bar.x:
            right = min(right, seat_left - max(4, bar.height // 2))
    if right - left < image.size[0] * 0.2:
        return None
    return Region(x=left, y=top, width=right - left + 1, height=bottom - top + 1)


# How far a pixel may sit from the panel's fill colour and still be panel.
_PANEL_TOLERANCE = 26
# What share of a column has to be panel for it to seed the board's extent,
# and to be swept into it once seeded.
_PANEL_CORE = 0.45
_PANEL_EDGE = 0.25
# How far it must sit from it to count as belonging to a card.
_CARD_INK = 34


def battlefield_zones(image, area: Region) -> Optional[Dict[str, Region]]:
    """Split the board into the two players' halves.

    MTGO fills each half with its own flat colour, so the split is a step in
    the row profile rather than a fraction of the height -- which matters,
    because the halves are not equal and the divider moves when a player has
    more rows of permanents than the other.
    """
    pixels = image.load()
    x0 = area.x + max(4, area.width // 100)
    x1 = area.x + area.width - max(4, area.width // 100)
    step = max(1, (x1 - x0) // 120)
    rows = []
    for y in range(area.y, area.y + area.height):
        # The median across the row ignores the cards sitting on it.
        rows.append(_median([pixels[x, y] for x in range(x0, x1, step)]))

    # The largest brightness step in the middle half of the board.
    lo = int(len(rows) * 0.25)
    hi = int(len(rows) * 0.75)
    best, split = 0.0, None
    for index in range(lo, hi):
        change = abs(rows[index] - rows[index - 1])
        if change > best:
            best, split = change, index
    if split is None or best < 10:
        return None
    divider = area.y + split
    inset = max(2, area.height // 120)
    top = Region(x=x0, y=area.y + inset, width=x1 - x0,
                 height=max(10, divider - area.y - 2 * inset))
    bottom = Region(x=x0, y=divider + inset, width=x1 - x0,
                    height=max(10, area.y + area.height - divider - 2 * inset))
    return {"top": _trim_to_panel(pixels, top),
            "bottom": _trim_to_panel(pixels, bottom)}


def _trim_to_panel(pixels, zone: Region) -> Region:
    """Shrink a zone to the panel it sits on.

    The board's sides are bounded by things beside it, which leaves a little
    of the client's own background inside the zone. That background is not
    the panel's colour, so it reads as a card -- a permanent that is really a
    strip of window frame. Trimming to where the panel actually is removes
    it, and costs nothing where the zone was already right.
    """
    rows = list(range(zone.y, zone.y + zone.height,
                      max(1, zone.height // 60)))
    if not rows:
        return zone
    fill = _mode([pixels[x, y]
                  for x in range(zone.x, zone.x + zone.width,
                                 max(1, zone.width // 120))
                  for y in rows])
    columns = []
    for x in range(zone.x, zone.x + zone.width):
        matched = sum(1 for y in rows
                      if abs(pixels[x, y] - fill) <= _PANEL_TOLERANCE)
        # A column covered by a card still shows panel above or below it;
        # a column outside the panel shows none at all.
        columns.append(matched > len(rows) * 0.12)
    spans = _runs(columns, int(zone.width * 0.1), max_gap=max(2, zone.width // 40))
    if not spans:
        return zone
    left, right = max(spans, key=lambda span: span[1] - span[0])
    return Region(x=zone.x + left, y=zone.y,
                  width=max(10, right - left + 1), height=zone.height)


# --------------------------------------------------------------------------
# Segmenting the cards on a panel
# --------------------------------------------------------------------------

def segment_cards(image, zone: Region, min_width_frac: float = 0.012
                  ) -> List[Region]:
    """The card-shaped things sitting on one panel.

    A panel is a flat fill and a card is not, so "everything that is not the
    fill colour" finds the cards without knowing anything about how they are
    laid out. MTGO stacks duplicate permanents into overlapping piles, whose
    columns are contiguous; a pile therefore comes back as one wide box, and
    how many cards it holds is read from its width afterwards.
    """
    pixels = image.load()
    x0, x1 = zone.x, zone.x + zone.width - 1
    y0, y1 = zone.y, zone.y + zone.height - 1
    ystep = max(1, zone.height // 160)
    sample = [pixels[x, y]
              for x in range(x0, x1 + 1, max(1, zone.width // 200))
              for y in range(y0, y1 + 1, ystep)]
    if not sample:
        return []
    fill = _mode(sample)

    rows = list(range(y0, y1 + 1, ystep))
    ink_columns = []
    for x in range(x0, x1 + 1):
        ink = sum(1 for y in rows if abs(pixels[x, y] - fill) > _CARD_INK)
        ink_columns.append(ink > len(rows) * 0.06)

    cards = []
    for left, right in _runs(ink_columns, int(zone.width * min_width_frac),
                             max_gap=max(2, zone.width // 400)):
        left += x0
        right += x0
        column = [x for x in range(left, right + 1,
                                   max(1, (right - left) // 60 or 1))]
        ink_rows = []
        for y in range(y0, y1 + 1):
            ink = sum(1 for x in column if abs(pixels[x, y] - fill) > _CARD_INK)
            ink_rows.append(ink > len(column) * 0.06)
        vertical = _runs(ink_rows, int(zone.height * 0.08),
                         max_gap=max(2, zone.height // 60))
        for top, bottom in vertical:
            cards.append(Region(x=left, y=y0 + top, width=right - left + 1,
                                height=bottom - top + 1))
    return cards


def _runs(flags: List[bool], min_length: int, max_gap: int
          ) -> List[Tuple[int, int]]:
    out, start, gap = [], None, 0
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                out.append((start, index - gap))
                start, gap = None, 0
    if start is not None:
        out.append((start, len(flags) - 1 - gap))
    return [(a, b) for a, b in out if b - a + 1 >= min_length]


# An MTGO battlefield card is drawn about this much taller than it is wide,
# measured from the title bar (where segmentation starts) to its bottom edge.
CARD_ASPECT = 1.6


def split_pile(card: Region, unit_width: Optional[int]) -> int:
    """How many cards a box holds, from how much wider than one card it is."""
    if not unit_width or unit_width <= 0:
        return 1
    return max(1, int(round(card.width / float(unit_width))))


# The share of a card's height taken by its title bar. MTGO's battlefield
# rendering puts the name in a strip across the top; reading only that strip
# is what keeps a card's *rules text* from being mistaken for its identity --
# plenty of cards name other cards in their text.
_TITLE_BAND = 0.16
# What the title crop is upscaled to before OCR, per the pipeline's habit of
# normalising glyph size rather than letting it drift with the capture.
_TITLE_TARGET_WIDTH = 520


def read_titles(image, card: Region, vocabulary: Sequence[str],
                min_ratio: float = 0.72, margin: float = 0.08) -> List[str]:
    """The card names written across the top of a card box.

    A box may hold several cards side by side -- MTGO packs a row of
    permanents tight, and overlaps duplicates into a pile -- so this returns
    every name it can see in the strip rather than one.

    Matching is against the *run vocabulary*: the cards this match has
    already named in its log. A few dozen candidates make a misread title
    recoverable where an open-set match would be a guess, and a title that
    matches nothing in the vocabulary is returned as None-worthy evidence by
    the caller rather than forced onto the nearest name.
    """
    import difflib

    from PIL import Image

    from .ocr import ocr_image

    band = max(6, int(round(card.height * _TITLE_BAND)))
    crop = image.crop((card.x, card.y, card.x + card.width, card.y + band))
    factor = max(1.0, _TITLE_TARGET_WIDTH / max(1, crop.width))
    crop = crop.resize((int(round(crop.width * factor)),
                        int(round(crop.height * factor))), Image.LANCZOS)
    import tempfile

    work = tempfile.mkdtemp(prefix="mtgo_title_")
    try:
        path = os.path.join(work, "title.png")
        crop.save(path)
        text = " ".join(ocr_image(path, psm=7, strict=False))
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return match_names(text, vocabulary, min_ratio=min_ratio, margin=margin)


def match_names(text: str, vocabulary: Sequence[str], min_ratio: float = 0.72,
                margin: float = 0.08) -> List[str]:
    """Which vocabulary names a strip of OCR'd title text contains.

    The strip may hold several titles in a row with no reliable separator, so
    every window of consecutive words is tried and the best non-overlapping
    matches are kept. A match must beat the runner-up by a margin, for the
    same reason the card catalog demands one: among names that all look
    alike, the margin -- not the score -- is what separates "that card,
    misread" from "not that card".
    """
    import difflib

    words = [w for w in text.replace("|", " ").split() if w]
    if not words or not vocabulary:
        return []
    folded = {name: name.lower() for name in vocabulary}
    found: List[Tuple[int, int, str, float]] = []
    for start in range(len(words)):
        for length in range(1, min(6, len(words) - start) + 1):
            phrase = " ".join(words[start:start + length]).lower()
            scored = sorted(
                ((difflib.SequenceMatcher(None, phrase, key).ratio(), name)
                 for name, key in folded.items()), reverse=True)
            best_score, best_name = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= min_ratio and best_score - runner_up >= margin:
                found.append((start, start + length, best_name, best_score))
    # Prefer the strongest matches, and never let two of them claim the same
    # words: "Snow-Covered Island" must not also count as "Island".
    chosen: List[Tuple[int, int, str]] = []
    for start, end, name, _ in sorted(found, key=lambda f: (-f[3], f[0] - f[1])):
        if any(start < other_end and end > other_start
               for other_start, other_end, _ in chosen):
            continue
        chosen.append((start, end, name))
    return [name for _, _, name in sorted(chosen)]


# Where a card's art window sits inside the box segmentation returns, as
# fractions of that box. Measured off MTGO's battlefield rendering: the box
# starts at the title bar, the art runs from just under it to the type line.
ART_TOP, ART_BOTTOM = 0.13, 0.55
ART_INSET = 0.05


def art_window(card: Region, tapped: bool = False) -> Region:
    """Where the art sits inside a card box, upright or turned sideways.

    A tapped permanent is drawn rotated, so its art is a *column* of the box
    rather than a band across it. Reading a rotated card as though it were
    upright grabs its rules text and half the panel, which then matches
    whichever card in the vocabulary has the flattest art -- the single
    largest source of false identifications before this was handled.
    """
    if tapped:
        left = card.x + int(round(card.width * ART_TOP))
        right = card.x + int(round(card.width * ART_BOTTOM))
        inset = int(round(card.height * ART_INSET))
        return Region(x=left, y=card.y + inset,
                      width=max(4, right - left),
                      height=max(4, card.height - 2 * inset))
    top = card.y + int(round(card.height * ART_TOP))
    bottom = card.y + int(round(card.height * ART_BOTTOM))
    inset = int(round(card.width * ART_INSET))
    return Region(x=card.x + inset, y=top,
                  width=max(4, card.width - 2 * inset),
                  height=max(4, bottom - top))


def identify(colour_image, zone_cards: Sequence[Region], index,
             unit: Optional[int], gray_image=None,
             vocabulary: Sequence[str] = ()) -> List[SeenCard]:
    """Name every card in a panel's boxes by sliding a card-width window.

    MTGO packs a row of permanents edge to edge and overlaps duplicates into
    a pile, so the boxes segmentation returns are groups, not cards. Rather
    than guess where inside a group each card starts, a card-sized window is
    slid across it and every position identified; positions that agree are
    then collapsed. This finds a card wherever it sits in the group, and a
    pile of copies collapses to one entry with the pile's width recorded.
    """
    seen: List[SeenCard] = []
    for box in zone_cards:
        width = unit or box.width
        # A row of tapped permanents is only as tall as a card is wide,
        # because each one is turned on its side. Its "cards" are therefore
        # a card's *height* across, and their art has to be turned back
        # upright before it can be compared with anything.
        tapped = bool(unit) and box.height < unit * CARD_ASPECT * 0.8
        slot = int(round(width * CARD_ASPECT)) if tapped else width
        if box.width <= slot * 1.25:
            candidates = [box]
        else:
            # Fine enough that a window lands on each card: the art
            # signature is sensitive to being half a card out.
            stride = max(4, slot // 6)
            candidates = [
                Region(x=x, y=box.y, width=slot, height=box.height)
                for x in range(box.x, box.x + box.width - slot + 1, stride)]
        hits = []
        for candidate in candidates:
            name, score = _best_match(colour_image, candidate, index, tapped)
            if name:
                hits.append((score, name, candidate))
        found = _collapse(hits, box, slot)
        # Only where the art said nothing is the title worth the OCR call.
        # At battlefield scale a card's name is drawn about seven pixels
        # tall, which reads correctly around half the time -- useful as a
        # second opinion on a box that would otherwise be anonymous, not as
        # the primary identification.
        if gray_image is not None and vocabulary:
            for card in found:
                if card.art is None:
                    titles = read_titles(gray_image, card.box, vocabulary)
                    if len(titles) == 1:
                        card.title = titles[0]
        seen.extend(found)
    return seen


def _best_match(colour_image, candidate: Region, index, tapped: bool):
    """Identify a window, trying both ways a tapped card can be turned.

    MTGO turns a tapped permanent consistently, but which way that is is
    not worth asserting when both can simply be tried: the wrong rotation
    scores no better than an unrelated card, so the right one wins on its
    own evidence.
    """
    window = art_window(candidate, tapped=tapped)
    crop = colour_image.crop((window.x, window.y, window.x + window.width,
                              window.y + window.height))
    if not tapped:
        return index.match(crop)
    best_name, best_score = None, None
    for angle in (90, -90):
        name, score = index.match(crop.rotate(angle, expand=True))
        if name and (best_score is None or score < best_score):
            best_name, best_score = name, score
    return best_name, best_score


def _collapse(hits, box: Region, unit: int) -> List[SeenCard]:
    """One entry per card, from many overlapping window matches."""
    chosen: List[Tuple[float, str, Region]] = []
    for score, name, region in sorted(hits, key=lambda hit: hit[0]):
        if any(other.x < region.x + region.width
               and region.x < other.x + other.width
               for _, _, other in chosen):
            continue
        chosen.append((score, name, region))
    out = []
    for score, name, region in sorted(chosen, key=lambda hit: hit[2].x):
        out.append(SeenCard(box=region, art=name, art_distance=round(score, 1)))
    if not out:
        # Nothing in the vocabulary matched this box. That is itself worth
        # returning: a permanent the log never named is what a dropped line
        # looks like from the screen.
        out.append(SeenCard(box=box, stacked=split_pile(box, unit)))
    elif len(out) == 1 and box.width > unit * 1.25:
        # A pile of copies: every window matched the same card, so the count
        # comes from how wide the pile is rather than from how many matched.
        out[0].stacked = split_pile(box, unit)
    return out


def card_unit_width(cards: Sequence[Region], tolerance: float = 0.18
                    ) -> Optional[int]:
    """The width of one card, from a box that is shaped like exactly one.

    Neither "the narrowest box" nor "the tallest box's height over the
    aspect" is safe on its own: a row of cards packed edge to edge is one
    wide box, a pile shows slivers narrower than a card, and a row of
    *tapped* cards is only as tall as a card is wide. What is safe is a box
    whose proportions are a card's -- that is one card, upright, and its
    width is the unit.
    """
    if not cards:
        return None
    upright = [card.width for card in cards
               if abs(card.height / float(max(1, card.width)) - CARD_ASPECT)
               <= CARD_ASPECT * tolerance]
    if upright:
        return int(round(sum(upright) / float(len(upright))))
    sideways = [card.height for card in cards
                if abs(card.width / float(max(1, card.height)) - CARD_ASPECT)
                <= CARD_ASPECT * tolerance]
    if sideways:
        # A single tapped card: its height is the card's width.
        return int(round(sum(sideways) / float(len(sideways))))
    return None


def check_states(video: str, states: Sequence[dict], layout, index,
                 sample: int = 1, settle: float = 0.5,
                 work_dir: Optional[str] = None) -> Dict:
    """Compare the board on screen with the board the log produced.

    Two directions, and they are not equally strong evidence.

    A card *seen and not derived* is the finding this module was built for: a
    permanent the pipeline never put on the board, which is what a dropped
    "casts X" line looks like from the outside. It is only claimed when the
    art matched one specific card in the vocabulary by a clear margin.

    A card *derived and not seen* is weaker. Plenty of permanents cannot be
    identified from a frame however good the decode is: MTGO overlaps
    duplicates into piles that show slivers of art, basic lands have dozens
    of printings and match none of them, and a card can be off-screen or
    covered by a dialog. So it is reported, not counted against the decode.
    """
    import tempfile

    from .extract import grab_frame
    from .hud import map_slots_to_seats
    from .layout import find_phase_bar, _load_gray

    own_dir = work_dir is None
    work_dir = os.path.realpath(work_dir or tempfile.mkdtemp(prefix="mtgo_board_"))
    try:
        slots = None
        checked = 0
        unlogged: List[Dict] = []
        unseen: List[Dict] = []
        identified = 0
        # Card size is a property of the client, not of a frame, and a frame
        # whose permanents are all tapped or all piled cannot measure it. So
        # it is agreed across frames first and then held fixed.
        unit = _agree_card_size(video, states, layout, work_dir, sample)
        for index_of, state in enumerate(states):
            if index_of % sample:
                continue
            when = state.get("videoTime")
            if when is None:
                continue
            frame = grab_frame(video, when + settle,
                               os.path.join(work_dir, "board.png"))
            if slots is None:
                slots = map_slots_to_seats(video, when + settle,
                                           state.get("players", []), work_dir,
                                           layout=layout)
                if len(slots) < 2:
                    slots = None
                    continue
            bar, words = find_phase_bar(frame, _load_gray(frame).size[1])
            if bar is not None:
                layout.phase_bar = bar
            seen = observe(frame, layout, index, words, unit=unit)
            if seen is None:
                continue
            checked += 1
            for slot, zone in seen["zones"].items():
                seat = slots.get(slot)
                if seat is None:
                    continue
                found = [card for card in zone["cards"] if card.name]
                on_screen = [card.name for card in found]
                identified += len(on_screen)
                derived = [_fold(o.get("name"))
                           for o in (state.get("zones") or {}).get("battlefield", [])
                           if str(o.get("controllerSeat", o.get("ownerSeat")))
                           == str(seat)]
                for card in found:
                    if _fold(card.name) not in derived:
                        unlogged.append({
                            "stateIndex": index_of, "videoTime": when,
                            "seat": seat, "card": card.name,
                            "distance": card.art_distance, "x": card.box.x,
                            "event": (state.get("sourceEvent") or {}).get("text"),
                        })
                shown = [_fold(name) for name in on_screen]
                for name in derived:
                    if name not in shown:
                        unseen.append({"stateIndex": index_of, "videoTime": when,
                                       "seat": seat, "card": name})
        confirmed, fleeting = _confirm(unlogged, sample=sample)
        return {
            "statesChecked": checked,
            "cardsIdentified": identified,
            "vocabulary": len(index.signatures),
            "unloggedPermanents": confirmed,
            "unloggedNotPersistent": fleeting,
            "derivedButNotSeen": unseen,
            "ok": not confirmed,
        }
    finally:
        if own_dir:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


def _agree_card_size(video: str, states: Sequence[dict], layout,
                     work_dir: str, sample: int, limit: int = 6
                     ) -> Optional[int]:
    """The card width this client draws, from whichever frames can say."""
    from .extract import grab_frame
    from .layout import _load_gray

    widths = []
    for index_of, state in enumerate(states):
        if index_of % sample or state.get("videoTime") is None:
            continue
        frame = grab_frame(video, state["videoTime"],
                           os.path.join(work_dir, "size.png"))
        gray = _load_gray(frame)
        area = play_area(gray, layout)
        zones = battlefield_zones(gray, area) if area else None
        for zone in (zones or {}).values():
            width = card_unit_width(segment_cards(gray, zone))
            if width:
                widths.append(width)
        if len(widths) >= limit:
            break
    return int(_median(widths)) if widths else None


def _confirm(findings: List[Dict], tolerance: int = 60, sample: int = 1
             ) -> Tuple[List[Dict], List[Dict]]:
    """Keep the sightings that persist; set the rest aside.

    A permanent does not flicker. It sits in the same place on the same
    panel for as long as it is in play, so a card the log never mentioned
    should be seen at about the same x in states that are *next to each
    other* -- not merely twice at some point in the game. Two sightings
    minutes apart with nothing in between is what a matcher finding a
    plausible neighbour twice looks like; a permanent that was really there
    was there in the states between as well.

    Both weaker cases are still reported, just not as defects.
    """
    groups: Dict[Tuple[int, str], List[Dict]] = {}
    for finding in findings:
        groups.setdefault((finding["seat"], finding["card"]), []).append(finding)
    confirmed, fleeting = [], []
    for (seat, card), items in groups.items():
        items.sort(key=lambda item: item["x"])
        cluster: List[Dict] = []
        for item in items + [None]:
            if cluster and (item is None
                            or item["x"] - cluster[-1]["x"] > tolerance):
                states = sorted({entry["stateIndex"] for entry in cluster})
                adjacent = any(b - a <= 2 * sample
                               for a, b in zip(states, states[1:]))
                target = confirmed if adjacent else fleeting
                best = min(cluster, key=lambda entry: entry["distance"])
                target.append(dict(best, sightings=len(states),
                                   consecutive=adjacent,
                                   videoTimes=sorted({entry["videoTime"]
                                                      for entry in cluster})))
                cluster = []
            if item is not None:
                cluster.append(item)
    confirmed.sort(key=lambda item: item["videoTime"])
    fleeting.sort(key=lambda item: item["videoTime"])
    return confirmed, fleeting


def _fold(name: Optional[str]) -> str:
    from .catalog import fold

    return fold(name or "")


def observe(frame_path: str, layout, index, words=(),
            unit: Optional[int] = None) -> Optional[Dict]:
    """What is on the two battlefields in this frame.

    Returns each panel's cards with whatever identity could be established,
    or None when the board could not be located at all -- which is not a
    failure of this module so much as a frame that is not showing a duel.
    """
    from PIL import Image

    from .layout import _load_gray

    gray = _load_gray(frame_path)
    colour = Image.open(frame_path).convert("RGB")
    area = play_area(gray, layout, words)
    if area is None:
        return None
    zones = battlefield_zones(gray, area)
    if zones is None:
        return None

    out: Dict[str, object] = {"playArea": area, "zones": {}}
    measured = unit
    for slot, zone in zones.items():
        boxes = segment_cards(gray, zone)
        if measured is None:
            # Whatever this frame can say about card size; a caller that
            # watches many frames passes in the size they agreed on instead.
            measured = card_unit_width(boxes)
        cards = identify(colour, boxes, index, measured, gray,
                         list(index.signatures))
        out["zones"][slot] = {
            "zone": zone,
            "unitWidth": measured,
            "cards": cards,
        }
    return out
