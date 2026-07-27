"""Split a match-long event stream into individual games.

A match VOD contains several games back to back. MTGO marks each new game
with a fresh set of "joined the game" lines, so a game boundary is the first
join/roll event that follows any earlier game content.
"""

from typing import Dict, List

_GAME_START = ("JOIN", "ROLL")
_GAME_END = ("WIN", "CONCEDE", "LOSE_GAME")
# Events that only ever appear once per game, before any turn is taken.
_PREGAME = _GAME_START + ("PLAY_FIRST", "OPENING_HAND", "MULLIGAN",
                          "BOTTOM_BEGIN", "LEADS_MATCH", "CHAT", "UNPARSED")


def split_games(events: List[Dict]) -> List[List[Dict]]:
    """Partition events into per-game lists (pregame chatter included)."""
    games: List[List[Dict]] = []
    current: List[Dict] = []
    seen_play = False
    for event in events:
        if event["type"] in _GAME_START and seen_play:
            games.append(current)
            current = []
            seen_play = False
        current.append(event)
        if event["type"] not in _PREGAME:
            seen_play = True
    if current:
        games.append(current)
    return [g for g in games if any(e["type"] not in _PREGAME for e in g)]


def game_is_complete(events: List[Dict]) -> bool:
    return any(e["type"] in _GAME_END for e in events)


# Pregame lines name both players before any play happens, and are the most
# reliably OCR'd (short, no card names), so they set the seat order.
_SEAT_ORDER_EVENTS = ("JOIN", "ROLL", "PLAY_FIRST", "OPENING_HAND",
                      "MULLIGAN", "BOTTOM_BEGIN")


def discover_players(events: List[Dict], limit: int = 2,
                     similarity: float = 0.7) -> List[str]:
    """Player names in first-appearance order, merging OCR variants.

    Falls back to scanning the whole game when the pregame lines are missing
    (a clip that starts mid-game). Returns at most `limit` names.
    """
    import difflib

    def collect(candidates):
        names: List[str] = []
        for name in candidates:
            name = name.strip().rstrip(".,:;")
            if not name:
                continue
            match = None
            for known in names:
                if difflib.SequenceMatcher(
                        None, name.lower(), known.lower()).ratio() >= similarity:
                    match = known
                    break
            if match is None and len(names) < limit:
                names.append(name)
        return names

    pregame = [e["player"] for e in events
               if e["type"] in _SEAT_ORDER_EVENTS and e.get("player")]
    names = collect(pregame)
    if len(names) < limit:
        names = collect(pregame + [e["player"] for e in events if e.get("player")])
    return names
