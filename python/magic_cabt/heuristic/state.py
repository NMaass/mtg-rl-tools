"""Allowlisted acting-player projection, shared by the heuristic and Jev.

Do not pass the original observation to a remote evaluator. Opponent hands,
libraries, results, raw transport payloads and future frames are never copied.
Static card metadata may enrich only an already-visible, named object/action.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Dict, List, Optional, Tuple


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value, default=0.0):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def mana_symbols(cost):
    return re.findall(r"\{([^}]+)\}", cost or "")


def mana_value(cost):
    return sum(int(s) if s.isdigit() else (0 if s in ("X", "Y", "Z") else 1)
               for s in mana_symbols(cost))


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    controller: str = ""
    types: Tuple[str, ...] = ()
    power: Optional[float] = None
    toughness: Optional[float] = None
    tapped: bool = False
    mana_cost: str = ""
    text: str = ""
    face_down: bool = False

    @property
    def creature(self):
        return "creature" in self.types

    @property
    def land(self):
        return "land" in self.types


@dataclass(frozen=True)
class Action:
    index: int
    kind: str
    label: str
    source_id: str = ""
    source_name: str = ""
    source_types: Tuple[str, ...] = ()
    mana_cost: str = ""
    rule: str = ""
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    prompt: str
    minimum: int
    maximum: int
    seat: int
    hero: str
    opponent: str
    life: float
    opponent_life: float
    hand_count: int
    opponent_hand_count: int
    active: str
    turn: int
    step: str
    hand: Tuple[Card, ...]
    battlefield: Tuple[Card, ...]
    stack: Tuple[Card, ...]
    actions: Tuple[Action, ...]
    warnings: Tuple[str, ...]

    @property
    def ours(self):
        return tuple(c for c in self.battlefield if c.controller == self.hero)

    @property
    def theirs(self):
        return tuple(c for c in self.battlefield if c.controller == self.opponent)

    def card(self, identity):
        return next((c for c in self.battlefield + self.hand + self.stack
                     if c.id == identity), None)

    def public_dict(self):
        return asdict(self)


# This is deliberately a small list, not a recursive payload copy.
ACTION_FIELDS = ("attackerId", "attackerName", "defenderId", "defenderType",
                 "blockerId", "blockerName", "targetId", "targetName", "zone",
                 "alreadyChosen", "objectId", "objectName", "abilityRule",
                 "manaType", "available", "unpaid", "mulliganDownTo",
                 "outcome", "targeted", "sourceRule", "value", "number")


def _card(raw, catalog):
    merged = dict(raw.get("ref") or {})
    merged.update({k: v for k, v in raw.items() if k != "ref"})
    hidden = bool(merged.get("faceDown") or merged.get("faceDownFlag"))
    name = "Face-down object" if hidden else str(merged.get("name") or merged.get("cardName") or "Unknown")
    static = {} if hidden else catalog.get(name, {})
    types = merged.get("cardTypes", merged.get("types", static.get("types", [])))
    return Card(
        id=str(merged.get("objectId", merged.get("instanceId", ""))),
        name=name,
        controller=str(merged.get("controllerId", merged.get("controllerSeat", ""))),
        types=tuple(str(t).lower() for t in types),
        power=number(merged.get("power", static.get("power")), None),
        toughness=number(merged.get("toughness", static.get("toughness")), None),
        tapped=bool(merged.get("tapped")),
        mana_cost="" if hidden else str(merged.get("manaCost", static.get("manaCost")) or ""),
        text="" if hidden else str(merged.get("rulesText", merged.get("oracleText", static.get("rulesText"))) or ""),
        face_down=hidden,
    )


def project(observation, catalog=None):
    """Project a live XMage observation. Arena observations are rejected explicitly.

    Matching the selecting seat rather than priorityPlayerId is essential for
    combat and target prompts. A missing acting identity is not guessed.
    """
    if not isinstance(observation, dict):
        raise ValueError("Expected an observation object")
    current, select = observation.get("current", {}), observation.get("select", {})
    if "zones" in current:
        raise ValueError("Arena requires a dedicated normalized strategy adapter; use the replay reviewer")
    seat = select.get("playerIndex")
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("Strategy experiment supports an explicit two-player acting seat")
    players = current.get("players") or []
    hero = next((p for p in players if p.get("playerIndex") == seat), None)
    opponent = next((p for p in players if p.get("playerIndex") == 1 - seat), None)
    if (len(players) != 2 or not hero or not opponent or not hero.get("playerId")
            or not opponent.get("playerId") or hero["playerId"] == opponent["playerId"]):
        raise ValueError("Acting-player identity is missing or ambiguous")
    if select.get("playerId") is not None and select["playerId"] != hero["playerId"]:
        raise ValueError("Selecting player identity does not match its snapshot index")
    catalog = catalog or {}
    hand = tuple(_card(c, catalog) for c in hero.get("hand") or [])
    board = tuple(_card(c, catalog) for c in current.get("battlefield") or [])
    stack = tuple(_card(c, catalog) for c in current.get("stack") or [])
    visible = {c.id: c for c in hand + board + stack if c.id}
    actions = []
    for index, option in enumerate(select.get("option") or []):
        if option.get("index", index) != index:
            raise ValueError("Option indices must match the engine transport order")
        payload = option.get("payload") or {}
        source_id = str(payload.get("sourceId") or "")
        visible_source = visible.get(source_id)
        name = str(payload.get("sourceName") or (visible_source.name if visible_source else ""))
        if visible_source and visible_source.face_down:
            name = "Face-down object"
        static = catalog.get(name, {})
        hidden_source = bool(visible_source and visible_source.face_down)
        data = {k: payload[k] for k in ACTION_FIELDS if k in payload and
                (payload[k] is None or type(payload[k]) in (str, int, float, bool))}
        hidden_references = []
        for field_name in ("target", "attacker", "blocker", "object"):
            referenced = visible.get(str(data.get(field_name + "Id") or ""))
            if referenced and referenced.face_down:
                data[field_name + "Name"] = "Face-down object"
                hidden_references.append(field_name)
        if hidden_source:
            data.pop("sourceRule", None)
            data.pop("abilityRule", None)
        label = str(option.get("label") or option.get("type") or "Unknown")
        if hidden_source or hidden_references:
            label = str(option.get("type", "UNKNOWN")) + " (face-down object)"
        actions.append(Action(
            index=index, kind=str(option.get("type", "UNKNOWN")),
            label=label,
            source_id=source_id, source_name=name,
            source_types=tuple(str(t).lower() for t in (static.get("types") or
                              (visible_source.types if visible_source else ()))),
            mana_cost="" if hidden_source else str(payload.get("manaCost", static.get("manaCost")) or ""),
            rule="" if hidden_source else str(payload.get("sourceRule") or (static.get("rulesText") if option.get("type") in ("CAST_SPELL", "PLAY_LAND") else payload.get("rule")) or payload.get("rule") or ""),
            data=data,
        ))
    low, high = select.get("minCount", 0), select.get("maxCount", 0)
    if type(low) is not int or type(high) is not int or low < 0 or high < 0:
        raise ValueError("Invalid selection bounds")
    # The live Java protocol has an explicit maxCount, including zero. Do not
    # inherit the legacy Python helper's 'zero means unbounded' interpretation.
    if low > high or low > len(actions):
        raise ValueError("Impossible live-engine selection bounds")
    warnings = ["No dynamic keyword, damage-marking or mana-pool snapshot; tactical estimates are approximate."]
    if any(c.name == "Unknown" for c in hand + board + stack):
        warnings.append("Some visible identities are unresolved.")
    if any(c.creature and (c.power is None or c.toughness is None) for c in hand + board):
        warnings.append("Some creature statistics are unavailable.")
    return Context(
        prompt=str(select.get("type", "UNKNOWN")), minimum=low, maximum=min(high, len(actions)), seat=seat,
        hero=str(hero["playerId"]), opponent=str(opponent["playerId"]),
        life=number(hero.get("life"), 20), opponent_life=number(opponent.get("life"), 20),
        hand_count=int(number(hero.get("handCount"))),
        opponent_hand_count=int(number(opponent.get("handCount"))),
        active=str(current.get("activePlayerId") or ""),
        turn=int(number(current.get("turnNumber"))), step=str(current.get("step") or ""),
        hand=hand, battlefield=board, stack=stack, actions=tuple(actions), warnings=tuple(warnings),
    )
