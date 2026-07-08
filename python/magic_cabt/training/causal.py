"""Causal / strategic variable extraction for DecisionRecord streams.

MTG-Causal-RL-style experiments need more than scalar win rate: they need
stable, auditable factors that describe strategic state and let evaluation
track where a policy gains or loses value. This module derives a small,
dependency-free factor set from the public CABT observation shape.

The variables here are intentionally conservative. They use only public fields
already present in ``observation.current`` / ``nextObservation.current`` and do
not infer hidden cards. Missing fields become ``None`` rather than guessed
values, keeping the feature contract safe for Arena-mirror and engine records.
"""

__all__ = [
    "FACTOR_NAMES",
    "causal_variables",
    "factor_delta",
    "factor_credit_trace",
]

FACTOR_NAMES = (
    "life_total",
    "life_diff",
    "hand_count",
    "hand_diff",
    "library_count",
    "library_diff",
    "graveyard_count",
    "graveyard_diff",
    "battlefield_count",
    "battlefield_diff",
    "creature_count",
    "creature_diff",
    "land_count",
    "land_diff",
    "stack_count",
    "turn_number",
    "is_active_player",
    "has_priority",
)


def causal_variables(record, use_next=False):
    """Return strategic variables from a canonical/source decision record.

    ``use_next=True`` reads ``nextObservation`` instead of ``observation`` when
    present, enabling before/after factor deltas. Returned keys are stable and
    aligned to ``FACTOR_NAMES``; unavailable numeric values are ``None``.
    """
    observation = record.get("nextObservation") if use_next else record.get("observation")
    if not isinstance(observation, dict):
        observation = {}
    current = observation.get("current") if isinstance(observation, dict) else None
    if not isinstance(current, dict):
        current = record.get("current") if isinstance(record.get("current"), dict) else {}

    player_index = record.get("playerIndex")
    players = [p for p in current.get("players") or [] if isinstance(p, dict)]
    acting = _player_by_index(players, player_index)
    opponent = _first_other_player(players, player_index)

    battlefield = current.get("battlefield") or []
    if not isinstance(battlefield, list):
        battlefield = []
    stack = current.get("stack") or []
    if not isinstance(stack, list):
        stack = []

    active_player_id = current.get("activePlayerId")
    priority_player_id = current.get("priorityPlayerId")
    acting_player_id = acting.get("playerId") if isinstance(acting, dict) else None

    values = {
        "life_total": _num(acting, "life"),
        "life_diff": _diff(_num(acting, "life"), _num(opponent, "life")),
        "hand_count": _num(acting, "handCount"),
        "hand_diff": _diff(_num(acting, "handCount"), _num(opponent, "handCount")),
        "library_count": _num(acting, "libraryCount"),
        "library_diff": _diff(_num(acting, "libraryCount"), _num(opponent, "libraryCount")),
        "graveyard_count": _num(acting, "graveyardCount"),
        "graveyard_diff": _diff(_num(acting, "graveyardCount"), _num(opponent, "graveyardCount")),
        "battlefield_count": _controlled_count(battlefield, acting_player_id),
        "battlefield_diff": _diff(
            _controlled_count(battlefield, acting_player_id),
            _controlled_count(battlefield, opponent.get("playerId") if isinstance(opponent, dict) else None),
        ),
        "creature_count": _controlled_type_count(battlefield, acting_player_id, "CREATURE"),
        "creature_diff": _diff(
            _controlled_type_count(battlefield, acting_player_id, "CREATURE"),
            _controlled_type_count(battlefield, opponent.get("playerId") if isinstance(opponent, dict) else None, "CREATURE"),
        ),
        "land_count": _controlled_type_count(battlefield, acting_player_id, "LAND"),
        "land_diff": _diff(
            _controlled_type_count(battlefield, acting_player_id, "LAND"),
            _controlled_type_count(battlefield, opponent.get("playerId") if isinstance(opponent, dict) else None, "LAND"),
        ),
        "stack_count": len(stack) if stack is not None else None,
        "turn_number": _coerce_number(current.get("turnNumber")),
        "is_active_player": _bool_as_int(acting_player_id is not None and acting_player_id == active_player_id),
        "has_priority": _bool_as_int(acting_player_id is not None and acting_player_id == priority_player_id),
    }
    return values


def factor_delta(before, after):
    """Return ``after - before`` for every numeric factor available in both."""
    delta = {}
    for name in FACTOR_NAMES:
        left = before.get(name)
        right = after.get(name)
        delta[name] = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
    return delta


def factor_credit_trace(record):
    """Return a compact factor trace for one decision record.

    The trace contains before factors, after factors when ``nextObservation`` is
    present, the delta, selected indices, reward/result metadata, and the prompt
    type. It is intentionally model-agnostic so baseline and future learned
    agents can write comparable annotations.
    """
    before = causal_variables(record, use_next=False)
    has_after = isinstance(record.get("nextObservation"), dict)
    after = causal_variables(record, use_next=True) if has_after else None
    select = record.get("select") or (record.get("observation") or {}).get("select") or {}
    return {
        "gameId": record.get("gameId"),
        "sequenceNumber": record.get("sequenceNumber"),
        "playerIndex": record.get("playerIndex"),
        "promptType": select.get("type") if isinstance(select, dict) else None,
        "selectedIndices": list(record.get("selectedIndices") or []),
        "before": before,
        "after": after,
        "delta": factor_delta(before, after) if after is not None else None,
        "terminal": record.get("terminal"),
        "reward": record.get("reward"),
        "result": record.get("result"),
    }


def _player_by_index(players, index):
    for player in players:
        if player.get("playerIndex") == index:
            return player
    return {}


def _first_other_player(players, index):
    for player in players:
        if player.get("playerIndex") != index:
            return player
    return {}


def _num(player, key):
    if not isinstance(player, dict):
        return None
    return _coerce_number(player.get(key))


def _coerce_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _diff(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left - right
    return None


def _bool_as_int(value):
    return 1 if value else 0


def _controlled_count(objects, controller_id):
    if controller_id is None:
        return None
    count = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if _controller_id(obj) == controller_id:
            count += 1
    return count


def _controlled_type_count(objects, controller_id, type_name):
    if controller_id is None:
        return None
    count = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if _controller_id(obj) != controller_id:
            continue
        types = obj.get("cardTypes") or obj.get("types") or []
        if isinstance(types, list) and type_name in [str(t).upper() for t in types]:
            count += 1
    return count


def _controller_id(obj):
    if obj.get("controllerId") is not None:
        return obj.get("controllerId")
    ref = obj.get("ref")
    if isinstance(ref, dict):
        return ref.get("controllerId")
    return None
