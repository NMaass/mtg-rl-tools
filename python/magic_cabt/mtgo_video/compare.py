"""Compare games decoded from different captures of the same match.

The claim "this pipeline is resolution-independent" is only worth anything
if it is checked: the same game, recorded at 720p and at 1440p, has to
decode to the same game. This compares bundles on their *content* -- the
event sequence and the board states -- and deliberately ignores video
timestamps, which legitimately differ by a frame or two between captures.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

# Fields that carry decoded meaning. Everything else in an event (the raw
# text, the clock reading, the video time) is presentation or provenance.
_EVENT_KEYS = ("type", "player", "card", "cards", "into", "counter", "targets",
               "turn", "amount", "count", "total", "attacker", "blocker")


def event_signature(event: Dict) -> Tuple:
    return tuple(
        (key, tuple(event[key]) if isinstance(event.get(key), list)
         else event.get(key))
        for key in _EVENT_KEYS if key in event
    )


def state_signature(state: Dict) -> Tuple:
    zones = state.get("zones") or {}

    def objects(items):
        return tuple(sorted(
            (o.get("name"), bool(o.get("tapped")),
             o.get("controllerSeat", o.get("ownerSeat")))
            for o in items or []))

    return (
        state.get("turnNumber"),
        state.get("phase"),
        state.get("step"),
        state.get("activeSeat"),
        tuple((p.get("seat"), p.get("name"), p.get("life"),
               p.get("libraryCount"), p.get("handCount"))
              for p in sorted(state.get("players", []),
                              key=lambda q: q.get("seat", 0))),
        objects(zones.get("battlefield")),
        tuple(sorted((seat, objects(cards))
                     for seat, cards in (zones.get("graveyards") or {}).items())),
    )


def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_game(bundle: str) -> Dict:
    return {
        "bundle": bundle,
        "events": _read_jsonl(os.path.join(bundle, "mtgo_events.jsonl")),
        "states": _read_jsonl(os.path.join(bundle, "mirror_states.jsonl")),
    }


def _first_difference(a: List, b: List) -> Optional[int]:
    for index in range(min(len(a), len(b))):
        if a[index] != b[index]:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_games(reference: Dict, other: Dict) -> Dict:
    """Diff two decoded games; empty `differences` means they are the same."""
    differences = []
    for label, key, signature in (("events", "events", event_signature),
                                  ("states", "states", state_signature)):
        left = [signature(x) for x in reference[key]]
        right = [signature(x) for x in other[key]]
        if len(left) != len(right):
            differences.append({
                "kind": label, "reason": "count",
                "reference": len(left), "other": len(right),
            })
        index = _first_difference(left, right)
        if index is not None and index < min(len(left), len(right)):
            differences.append({
                "kind": label, "reason": "content", "index": index,
                "reference": _describe(reference[key][index], label),
                "other": _describe(other[key][index], label),
            })
    return {
        "reference": reference["bundle"],
        "other": other["bundle"],
        "events": len(other["events"]),
        "states": len(other["states"]),
        "identical": not differences,
        "differences": differences,
    }


def _describe(item: Dict, kind: str) -> str:
    if kind == "events":
        return "%s %s" % (item.get("type"), item.get("text", ""))[:160]
    source = (item.get("sourceEvent") or {}).get("text", "")
    return "seq %s turn %s: %s" % (item.get("seq"), item.get("turnNumber"),
                                   source)[:160]


def compare_bundles(bundles: List[str]) -> Dict:
    """Compare every bundle against the first."""
    games = [load_game(b) for b in bundles]
    reference = games[0]
    results = [compare_games(reference, other) for other in games[1:]]
    return {
        "reference": reference["bundle"],
        "referenceEvents": len(reference["events"]),
        "referenceStates": len(reference["states"]),
        "compared": results,
        "allIdentical": all(r["identical"] for r in results),
    }
