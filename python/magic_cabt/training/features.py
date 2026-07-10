"""Stable text features for single-choice imitation-learning examples.

The first behavior-cloning path intentionally uses plain text fields rather
than tensors. Records from Arena mirrors and XMage engine runs do not expose
identical state detail, so every extractor here treats optional fields as
best-effort context.

Feature text is canonical with respect to per-game object identity: instance
ids (Arena ``instance=<n>``, XMage UUIDs) are stripped so two strategically
identical objects produce identical feature text, while board presence is
kept as explicit multiplicity (``Token x2``) in zone summaries. Card identity
(names, ``grpId``) is preserved -- it is stable across games.
"""

import re

__all__ = [
    "canonical_text",
    "state_text",
    "option_text",
    "option_type",
    "prompt_type",
]

# Per-game identifiers that must not become model features: key=value forms
# for instance ids, and bare UUID literals (XMage object ids).
_INSTANCE_NOISE_RE = re.compile(
    r"\b(?:instance|instanceId|targetInstanceId|objectId|sourceId|id)=\S+"
    r"|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def canonical_text(text):
    """Strip per-game object identifiers from feature text.

    Two options that differ only by which concrete instance they point at
    come out identical; the option index still disambiguates for execution.
    """
    if text is None:
        return ""
    cleaned = _INSTANCE_NOISE_RE.sub("", str(text))
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def prompt_type(record):
    select = _select(record)
    return _string(select.get("type") or record.get("decisionMethod") or "UNKNOWN")


def option_type(option):
    return _string((option or {}).get("type") or "UNKNOWN")


def state_text(record):
    """Return a deterministic, compact text summary for a decision record."""
    observation = record.get("observation") or {}
    current = observation.get("current") or record.get("current") or {}
    select = _select(record)
    parts = []

    _append(parts, "turn", current.get("turnNumber"))
    _append(parts, "active", _player_label(current, current.get("activePlayerId")))
    _append(parts, "priority", _player_label(current, current.get("priorityPlayerId")))
    _append(parts, "phase", current.get("phase"))
    _append(parts, "step", current.get("step"))

    players = current.get("players") or []
    for player in players:
        if not isinstance(player, dict):
            continue
        prefix = "p%s" % player.get("playerIndex", "?")
        player_bits = []
        _append(player_bits, "life", player.get("life"))
        _append(player_bits, "hand", player.get("handCount"))
        _append(player_bits, "library", player.get("libraryCount"))
        if player_bits:
            parts.append("%s[%s]" % (prefix, " ".join(player_bits)))

    stack = current.get("stack")
    if stack is not None:
        parts.append("stack[%s]" % _zone_summary(stack))
    elif current.get("stackSize") is not None:
        parts.append("stackSize=%s" % current.get("stackSize"))

    battlefield = current.get("battlefield")
    if battlefield is not None:
        parts.append("battlefield[%s]" % _zone_summary(battlefield))
    elif current.get("battlefieldSize") is not None:
        parts.append("battlefieldSize=%s" % current.get("battlefieldSize"))

    _append(parts, "prompt", select.get("type"))
    return " | ".join(parts)


def option_text(option):
    """Return text for one legal option, including type, label, and source.

    The assembled text is canonicalized: instance ids never reach the model,
    so identical game objects yield identical option features.
    """
    option = option or {}
    parts = []
    _append(parts, "type", option.get("type"))
    _append(parts, "label", option.get("label"))
    payload = option.get("payload") or {}
    if isinstance(payload, dict):
        _append(parts, "source", _name_from(payload.get("source")))
        _append(parts, "card", _name_from(payload.get("card")))
        _append(parts, "name", payload.get("name"))
    text = canonical_text(" | ".join(parts))
    return text if text else "option"


def _select(record):
    return record.get("select") or (record.get("observation") or {}).get("select") or {}


def _append(parts, label, value):
    if value is None:
        return
    text = _name_from(value)
    if text != "":
        parts.append("%s=%s" % (label, text))


def _string(value):
    if value is None:
        return ""
    return str(value)


def _name_from(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        ref = value.get("ref")
        if isinstance(ref, dict):
            for key in ("name", "objectId", "sourceId"):
                if ref.get(key) is not None:
                    return str(ref.get(key))
        for key in ("name", "label", "type", "objectId", "id"):
            if value.get(key) is not None:
                return str(value.get(key))
        return ""
    return str(value)


def _player_label(current, player_id):
    if player_id is None:
        return None
    for player in current.get("players") or []:
        if isinstance(player, dict) and player.get("playerId") == player_id:
            if player.get("name"):
                return player.get("name")
            if player.get("playerIndex") is not None:
                return player.get("playerIndex")
    return player_id


def _zone_summary(objects):
    """Summarize a zone as canonical object descriptions with multiplicity.

    Identical objects collapse to one entry with an explicit count
    (``2/2 Token x2``) so board presence survives even though instance
    identity does not.
    """
    if not objects:
        return "empty"
    counts = {}
    order = []
    unnamed = 0
    for item in objects:
        desc = _object_desc(item)
        if not desc:
            unnamed += 1
            continue
        if desc not in counts:
            counts[desc] = 0
            order.append(desc)
        counts[desc] += 1
    if not order:
        return "count=%d" % len(objects)
    parts = []
    for desc in order[:8]:
        count = counts[desc]
        parts.append("%s x%d" % (desc, count) if count > 1 else desc)
    if len(order) > 8 or unnamed:
        parts.append("count=%d" % len(objects))
    return ", ".join(parts)


def _object_desc(item):
    """Canonical description of one zone object: name, P/T, tapped state."""
    name = canonical_text(_name_from(item))
    if not name:
        return ""
    if not isinstance(item, dict):
        return name
    bits = [name]
    power = item.get("power")
    toughness = item.get("toughness")
    if power is not None and toughness is not None:
        bits.append("%s/%s" % (power, toughness))
    if item.get("tapped"):
        bits.append("tapped")
    return " ".join(bits)

