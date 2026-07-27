"""Parse MTGO game-log entries into structured events.

Each parser returns a dict with at least {"type": ..., "text": original}.
Unrecognized entries come back as {"type": "UNPARSED"} so coverage gaps are
visible instead of silent.
"""

import re
from typing import Dict, List, Optional, Optional

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
        "DRAW_N_WITH",
        re.compile(
            r"^(?P<player>\S+) draws (?P<count>\w+) cards with (?P<card>.+?)\.?$"
        ),
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
    (
        "DRAW_WITH",
        re.compile(r"^(?P<player>\S+) draws a card with (?P<card>.+?)\.?$"),
    ),
    ("DRAW", re.compile(r"^(?P<player>\S+) draws a card\.?$")),
    (
        "DRAW_N",
        re.compile(r"^(?P<player>\S+) draws (?P<count>\w+) cards\.?$"),
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
    text = strip_timestamp(entry)

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
        if "count" in event:
            event["count"] = _WORD_NUMBERS.get(event["count"].lower(), event["count"])
        return event

    event = {"type": "UNPARSED", "text": text}
    if clock:
        event["clock"] = clock
    return event


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
