"""Canonical hidden-information-safe state helpers."""
from __future__ import annotations

import math
import re

_ID_KEYS = frozenset(("id", "objectId", "instanceId", "targetId",
                      "targetInstanceId", "sourceId", "gameInstance",
                      "eventId", "requestId", "uuid"))
_VOLATILE = frozenset(("timestamp", "rawTime", "sequenceNumber", "seq"))
_PRIVATE = frozenset(("raw", "payloadRaw", "rawPayload", "raw_audit"))
ZONE_NAMES = ("global", "player", "battlefield", "stack", "hand",
              "graveyard", "exile", "library", "command", "sideboard",
              "history", "other")
ZONE_INDEX = {name: index for index, name in enumerate(ZONE_NAMES)}


def current_state(value):
    if not isinstance(value, dict):
        return {}
    observation = value.get("observation")
    if isinstance(observation, dict) and isinstance(observation.get("current"), dict):
        return observation["current"]
    return value.get("current") if isinstance(value.get("current"), dict) else value


def select_block(record):
    direct = record.get("select")
    return direct if isinstance(direct, dict) else \
        (record.get("observation") or {}).get("select") or {}


def perspective(state, explicit=None):
    if explicit is not None:
        return explicit
    for key in ("localSeat", "perspectiveSeat", "playerIndex", "seat"):
        if state.get(key) is not None:
            return state[key]
    return None


def seat_of(value):
    if not isinstance(value, dict):
        return None
    for key in ("seat", "playerIndex", "controllerSeat", "ownerSeat",
                "controllerId", "ownerId"):
        if value.get(key) is not None:
            return value[key]
    return None


def iter_zone_objects(state):
    zones = state.get("zones")
    if isinstance(zones, dict):
        for raw_name, contents in zones.items():
            zone = normalize_zone(raw_name)
            if isinstance(contents, dict):
                for seat, objects in contents.items():
                    for obj in objects or []:
                        yield zone, obj, seat
            else:
                for obj in contents or []:
                    yield zone, obj, seat_of(obj)
        return
    for key in ("battlefield", "stack", "exile", "command", "library",
                "sideboard", "graveyard", "hand"):
        for obj in state.get(key) or []:
            yield key, obj, seat_of(obj)
    for key, zone in (("hands", "hand"), ("graveyards", "graveyard"),
                      ("libraries", "library")):
        for seat, objects in (state.get(key) or {}).items():
            for obj in objects or []:
                yield zone, obj, seat


def normalize_zone(name):
    text = str(name or "other").lower()
    if text.endswith("s") and text[:-1] in ZONE_INDEX:
        text = text[:-1]
    return text if text in ZONE_INDEX else "other"


def flatten_text(value, prefix="", depth=0):
    if depth > 4:
        return ""
    parts = []
    if isinstance(value, dict):
        for key in sorted(value):
            if key in _ID_KEYS or key in _VOLATILE or key in _PRIVATE:
                continue
            label = "%s.%s" % (prefix, key) if prefix else str(key)
            text = flatten_text(value[key], label, depth + 1)
            if text:
                parts.append(text)
    elif isinstance(value, (list, tuple)):
        for item in value[:24]:
            text = flatten_text(item, prefix, depth + 1)
            if text:
                parts.append(text)
    elif value is not None:
        text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "", str(value),
                      flags=re.I).strip()
        if text:
            parts.append("%s=%s" % (prefix, text) if prefix else text)
    return " | ".join(parts)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def norm(value, scale):
    return math.tanh(number(value) / float(scale))


def delta(before, after, scale):
    return math.tanh((number(after) - number(before)) / float(scale))


def zone_items(state, zone):
    zones = state.get("zones") or {}
    raw = zones.get(zone) or zones.get(zone + "s") or \
        state.get(zone) or state.get(zone + "s")
    if isinstance(raw, dict):
        result = []
        for values in raw.values():
            result.extend(values or [])
        return result
    return raw or []
