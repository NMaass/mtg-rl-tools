"""Parse MTGO game-log entries into structured events.

Each parser returns a dict with at least {"type": ..., "text": original}.
Unrecognized entries come back as {"type": "UNPARSED"} so coverage gaps are
visible instead of silent.
"""

import difflib
import re
from typing import Dict, List, Optional

from .ocr import strip_timestamp, TIMESTAMP_RE


def _split_cards(blob: str) -> List[str]:
    """Split "A, B, and C" / "A and B" / "A" into card names."""
    blob = blob.strip().rstrip(".")
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", blob)
    return [p.strip() for p in parts if p.strip()]


_PATTERNS = [
    ("ROLL", re.compile(r"^(?P<player>\S+) rolled a (?P<value>\d+)\.?$")),
    ("JOIN", re.compile(r"^(?P<player>\S+) joined the game\.?$")),
    ("PLAY_FIRST", re.compile(r"^(?P<player>\S+) chooses to play first\.?$")),
    (
        "OPENING_HAND",
        re.compile(
            r"^(?P<player>\S+) begins the game with (?P<count>\w+) cards? in hand\.?$"
        ),
    ),
    (
        "MULLIGAN",
        re.compile(r"^(?P<player>\S+) mulligans to (?P<count>\w+) cards?\.?$"),
    ),
    (
        "BOTTOM_BEGIN",
        re.compile(
            r"^(?P<player>\S+) puts (?P<bottomed>\w+) cards? on the bottom of "
            r"their \w* ?library and begins the game with (?P<count>\w+) "
            r"cards? in hand\.?$"
        ),
    ),
    ("TURN", re.compile(r"^Turn (?P<turn>[\dSlIOo]+): (?P<player>\S+)\.?$")),
    ("SKIP_DRAW", re.compile(r"^(?P<player>\S+) skips their draw step\.?$")),
    (
        "COUNTERS_WITH",
        re.compile(
            r"^(?P<player>\S+) counters (?P<card>.+?) with (?P<counter>.+?)\.?$"
        ),
    ),
    (
        "LEADS_MATCH",
        re.compile(r"^(?P<player>\S+) leads the match (?P<score>[\d-]+)\.?$"),
    ),
    (
        "PUTS_TOP_N",
        re.compile(
            r"^(?P<player>\S+) puts (?P<count>\w+) cards? on top of their library\.?$"
        ),
    ),
    ("PLAY_LAND", re.compile(r"^(?P<player>\S+) plays (?P<card>.+?)\.?$")),
    (
        "CAST",
        re.compile(
            r"^(?P<player>\S+) casts (?P<card>.+?)"
            r"(?: with (?:a )?(?P<alternate>Prototype|Flashback|Escape|Foretell|"
            r"Disturb|Madness|Evoke|Overload|Kicker|Bargain|Adventure))?"
            r"(?: targeting (?P<targets>.+?))?\.?$"
        ),
    ),
    # The article in "draws a card" is a single glyph and is misread often
    # ("draws 3 card"), so accept any token there and take the count from
    # the noun's plurality instead -- "card" vs "cards" is a longer, far
    # more reliable signal than one character.
    (
        "DRAW_WITH",
        re.compile(r"^(?P<player>\S+) draws (?P<count>\S+) (?P<plural>cards?) "
                   r"with (?P<card>.+?)\.?$"),
    ),
    (
        "DRAW",
        re.compile(r"^(?P<player>\S+) draws (?P<count>\S+) (?P<plural>cards?)\.?$"),
    ),
    ("MILL", re.compile(r"^(?P<player>\S+) mills (?P<cards>.+?)\.?$")),
    (
        "REVEAL",
        re.compile(
            r"^(?P<player>\S+) reveals (?P<cards>.+?)(?: with (?P<source>.+?))?\.?$"
        ),
    ),
    ("DISCARD", re.compile(r"^(?P<player>\S+) discards (?P<card>.+?)\.?$")),
    ("CYCLE", re.compile(r"^(?P<player>\S+) cycles (?P<card>.+?)\.?$")),
    (
        "TRANSFORM",
        re.compile(r"^(?P<card>.+?) transforms into (?P<into>.+?)\.?$"),
    ),
    (
        "TRIGGER",
        re.compile(
            r"^(?P<player>\S+) puts \w triggered ability from (?P<card>.+?) "
            r"onto the stack\b.*$"
        ),
    ),
    (
        "ACTIVATE",
        re.compile(
            r"^(?P<player>\S+) \wctivates an ability of (?P<card>.+?)"
            r"(?: targeting (?P<targets>.+?))?(?: \(.*)?\.?$"
        ),
    ),
    (
        "ATTACKED_BY",
        re.compile(r"^(?P<player>\S+) is being attacked by (?P<cards>.+?)\.?$"),
    ),
    (
        "BLOCK",
        re.compile(r"^(?P<player>\S+) blocks (?P<attacker>.+?) with (?P<blocker>.+?)\.?$"),
    ),
    (
        "DAMAGE",
        re.compile(
            r"^(?P<source>.+?) deals (?P<amount>\d+) damage to (?P<target>.+?)\.?$"
        ),
    ),
    ("LOSE_LIFE", re.compile(r"^(?P<player>\S+) loses (?P<amount>\d+) life\.?.*$")),
    ("GAIN_LIFE", re.compile(r"^(?P<player>\S+) gains (?P<amount>\d+) life\.?.*$")),
    (
        "LIFE_TOTAL",
        re.compile(r"^(?P<player>\S+) is now at (?P<total>-?\d+) life\.?$"),
    ),
    ("DIES", re.compile(r"^(?P<card>.+?) is put into (?P<owner>.+?) graveyard\.?$")),
    ("DESTROYED", re.compile(r"^(?P<card>.+?) is destroyed\.?$")),
    ("SACRIFICE", re.compile(r"^(?P<player>\S+) sacrifices (?P<card>.+?)\.?$")),
    ("EXILE_CARD", re.compile(r"^(?P<card>.+?) is exiled\.?$")),
    ("COUNTERED", re.compile(r"^(?P<card>.+?) is countered\.?$")),
    (
        "RETURN_HAND",
        re.compile(r"^(?P<card>.+?) is returned to (?P<owner>.+?) hand\.?$"),
    ),
    ("SHUFFLE", re.compile(r"^(?P<player>\S+) shuffles (?P<what>.+?)\.?$")),
    (
        "SCRY_BOTTOM",
        re.compile(
            r"^(?P<player>\S+) puts (?P<count>\w+) cards? on the bottom of "
            r"(?P<whose>.+?) library\.?$"
        ),
    ),
    (
        "PUT_TOP",
        re.compile(
            r"^(?P<player>\S+) puts (?P<card>.+?) on top of (?P<whose>.+?) library\.?$"
        ),
    ),
    ("CONCEDE", re.compile(r"^(?P<player>\S+) has conceded\.?.*$")),
    ("WIN", re.compile(r"^(?P<player>\S+) wins the game\.?$")),
    ("LOSE_GAME", re.compile(r"^(?P<player>\S+) has lost the game\.?.*$")),
    ("CHAT", re.compile(r"^(?P<player>\S+) ?: (?P<message>.+)$")),
]


def parse_entry(entry: str) -> Dict:
    """Parse one full log entry (timestamp optional) into an event dict."""
    match = TIMESTAMP_RE.match(entry)
    clock = match.group(0).strip().rstrip(":;.,").strip() if match else None
    # A line's closing period is often read as a comma; the patterns below
    # allow it to be absent, so drop whatever it turned into.
    text = strip_timestamp(entry).rstrip().rstrip(",;")

    for kind, pattern in _PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        event: Dict = {"type": kind, "text": text}
        if clock:
            event["clock"] = clock
        event.update({k: v for k, v in m.groupdict().items() if v is not None})
        if kind in ("MILL", "REVEAL", "ATTACKED_BY"):
            event["cards"] = _split_cards(event["cards"])
        if "targets" in event:
            event["targets"] = _split_cards(event["targets"])
        if "turn" in event:
            event["turn"] = _fix_number(event["turn"])
        for key in ("amount", "value", "total"):
            if key in event:
                event[key] = int(event[key])
        if "plural" in event:
            event["count"] = _draw_count(event.pop("count"),
                                         event.pop("plural").endswith("s"))
        elif "count" in event:
            event["count"] = _WORD_NUMBERS.get(event["count"].lower(),
                                               event["count"])
        return event

    repaired = _repair_verb(text)
    if repaired is not None:
        repaired["text"] = text
        repaired["repaired"] = True
        if clock:
            repaired["clock"] = clock
        return repaired

    event = {"type": "UNPARSED", "text": text}
    if clock:
        event["clock"] = clock
    return event


# The verbs the grammar recognizes. This is a small closed vocabulary, which
# is what makes repairing a misread one safe: there is nothing else a word in
# this position could legitimately be.
_VERBS = (
    "plays", "casts", "draws", "mills", "reveals", "discards", "cycles",
    "transforms", "counters", "activates", "blocks", "sacrifices",
    "shuffles", "mulligans", "skips", "puts", "begins", "rolled", "joined",
    "chooses", "gains", "loses", "deals", "leads", "wins", "destroyed",
    "exiled", "conceded", "attacked", "returned",
)


def _repair_verb(text: str):
    """Re-parse a line whose verb OCR mangled ("eyeles" for "cycles").

    Only a repair that then *parses* is accepted, so a wrong guess produces
    an unparsed line as before rather than a plausible wrong event.
    """
    words = text.split()
    if any(word.lower().strip(".,") in _VERBS for word in words):
        return None  # the verb is intact; something else is wrong
    for index, word in enumerate(words):
        token = word.lower().strip(".,")
        if len(token) < 4:
            continue
        match = difflib.get_close_matches(token, _VERBS, n=1, cutoff=0.62)
        if not match:
            continue
        candidate = list(words)
        candidate[index] = match[0]
        event = parse_entry(" ".join(candidate))
        if event["type"] != "UNPARSED":
            return event
    return None


def _draw_count(token: str, plural: bool) -> int:
    """How many cards "draws <token> card(s)" means.

    A singular noun means exactly one, whatever the article OCR'd as -- so a
    misread "draws 3 card" is still one card. Only a plural noun licenses
    reading the token as a number.
    """
    text = token.lower().strip()
    if not plural:
        return 1
    if text in _WORD_NUMBERS:
        return _WORD_NUMBERS[text]
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 2


def _fix_number(raw) -> int:
    """Turn an OCR'd number like "5", "S", "5S", "l2" into an int."""
    text = str(raw)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits)
    mapped = text.translate(str.maketrans({"S": "5", "s": "5", "l": "1",
                                           "I": "1", "O": "0", "o": "0"}))
    digits = "".join(ch for ch in mapped if ch.isdigit())
    return int(digits) if digits else 0


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def parse_log(entries: List[str], times: Optional[List] = None) -> List[Dict]:
    """Parse entries; `times` attaches each entry's video-seconds timestamp."""
    events = [parse_entry(entry) for entry in entries]
    if times:
        for event, when in zip(events, times):
            if when is not None:
                event["videoTime"] = round(when, 3)
    return events
