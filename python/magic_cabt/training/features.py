"""Stable text features for single-choice imitation-learning examples.

The first behavior-cloning path intentionally uses plain text fields rather
than tensors. Records from Arena mirrors and XMage engine runs do not expose
identical state detail, so every extractor here treats optional fields as
best-effort context.
"""

__all__ = [
    "state_text",
    "option_text",
    "option_type",
    "prompt_type",
]


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
    """Return text for one legal option, including type, label, and source."""
    option = option or {}
    parts = []
    _append(parts, "type", option.get("type"))
    _append(parts, "label", option.get("label"))
    payload = option.get("payload") or {}
    if isinstance(payload, dict):
        _append(parts, "source", _name_from(payload.get("source")))
        _append(parts, "card", _name_from(payload.get("card")))
        _append(parts, "name", payload.get("name"))
    return " | ".join(parts) if parts else "option"


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
    if not objects:
        return "empty"
    names = []
    for item in objects[:8]:
        name = _name_from(item)
        if name:
            names.append(name)
    if not names:
        return "count=%d" % len(objects)
    if len(objects) > len(names):
        names.append("count=%d" % len(objects))
    return ", ".join(names)

