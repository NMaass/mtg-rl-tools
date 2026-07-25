"""Group low-level CABT prompts into strategic macro-action transitions.

A Magic play such as casting a spell commonly spans several engine callbacks:
priority choice, modes, targets, X, and mana payment. Training each callback as
an independent strategic action overweights interface choreography and leaves
``nextObservation`` attached to the wrong semantic unit.

This module derives a conservative grouping view over canonical DecisionRecord
streams. It never mutates source records and never claims engine certainty:
prompt-role classification is recorded with a confidence and reason, orphaned
parameter prompts remain visible, and incomplete end-of-stream groups are
marked explicitly rather than silently discarded.
"""

from __future__ import annotations

import copy
import json

__all__ = [
    "SCHEMA_VERSION",
    "classify_decision",
    "iter_macro_actions",
    "macro_transition",
    "selected_options",
]

SCHEMA_VERSION = 1

# These are prompt-name hints, not a replacement for the XMage decision audit.
# Unknown prompt families are treated as new roots so they cannot be hidden
# inside an unrelated action group.
_ROOT_HINTS = (
    "PRIORITY",
    "MULLIGAN",
    "ATTACK",
    "BLOCK",
    "STARTING_PLAYER",
    "SPECIAL_ACTION",
    "CONCEDE",
)
_PAYMENT_HINTS = ("MANA", "PAY", "COST")
_ORDERING_HINTS = ("ORDER", "TRIGGER", "REPLACEMENT")
_PARAMETER_HINTS = (
    "TARGET",
    "MODE",
    "CHOICE",
    "PILE",
    "AMOUNT",
    "ANNOUNCE",
    "CHOOSE_USE",
    "CHOOSEUSE",
    "USE_ABILITY",
    "X_VALUE",
    "XMANA",
)

_PASS_WORDS = ("pass", "decline", "no action", "done")


def classify_decision(record):
    """Return a transparent role classification for one DecisionRecord.

    The result contains:

    ``role``
        ``root``, ``parameter``, ``payment``, or ``ordering``.
    ``attachable``
        Whether this prompt may be attached to an already-open root action.
    ``deterministic``
        Whether the legal envelope forces a single concrete outcome.
    ``passLike``
        Whether the recorded selection appears to pass/decline.
    ``confidence`` / ``reason``
        Audit metadata explaining how the classification was obtained.
    """
    select = _select(record)
    prompt = str(select.get("type") or "UNKNOWN").upper()
    options = _options(select)
    selected = record.get("selectedIndices") if isinstance(record, dict) else []
    selected = selected if isinstance(selected, list) else []

    deterministic = _forced_selection(select, len(options))
    pass_like = _pass_like(options, selected)

    if _contains(prompt, _PAYMENT_HINTS):
        role, attachable, reason = "payment", True, "payment prompt family"
    elif _contains(prompt, _ORDERING_HINTS):
        role, attachable, reason = "ordering", True, "ordering prompt family"
    elif _contains(prompt, _PARAMETER_HINTS):
        role, attachable, reason = "parameter", True, "parameter prompt family"
    elif _contains(prompt, _ROOT_HINTS) or pass_like:
        role, attachable, reason = "root", False, "strategic root prompt family"
    else:
        # Fail open as a *new root*, not as an attachment. This prevents an
        # unfamiliar callback from being silently laundered into a prior play.
        role, attachable, reason = "root", False, "unknown prompt starts a new root"

    return {
        "promptType": prompt,
        "role": role,
        "attachable": attachable,
        "deterministic": deterministic,
        "passLike": pass_like,
        "strategic": role == "root" and not deterministic,
        "confidence": "known-family" if reason != "unknown prompt starts a new root"
        else "heuristic",
        "reason": reason,
    }


def selected_options(record):
    """Return deep-copied selected option payloads from one record."""
    select = _select(record)
    options = _options(select)
    selected = record.get("selectedIndices") if isinstance(record, dict) else []
    if not isinstance(selected, list):
        return []
    return [
        copy.deepcopy(options[index])
        for index in selected
        if isinstance(index, int) and not isinstance(index, bool)
        and 0 <= index < len(options)
    ]


def iter_macro_actions(records):
    """Yield ``MacroActionRecord-v1`` rows from ordered DecisionRecords.

    Group boundaries are conservative:

    * a new game, acting player, or non-attachable prompt closes the group;
    * target/mode/payment/ordering prompts attach only to an existing same-seat
      group;
    * terminal records close immediately;
    * the next record's observation becomes the macro action's boundary state;
    * EOF without a boundary leaves ``complete == False``.
    """
    active = None
    for raw in records:
        if not isinstance(raw, dict):
            continue
        record = raw
        classification = classify_decision(record)
        identity = _game_identity(record)
        player = _player_index(record)

        if active is not None:
            same_game = identity == active["_identity"]
            same_player = player == active["playerIndex"]
            can_attach = same_game and same_player and classification["attachable"]
            if not can_attach:
                if same_game:
                    reason = "player-boundary" if not same_player else "new-root"
                    boundary = record.get("observation")
                else:
                    reason = "game-boundary"
                    boundary = None
                yield _finish_group(active, boundary, reason)
                active = None

        if active is None:
            active = _start_group(record, classification, identity, player)
        else:
            _append_step(active, record, classification)

        if record.get("terminal") is True:
            yield _finish_group(
                active,
                record.get("nextObservation"),
                "terminal",
            )
            active = None

    if active is not None:
        yield _finish_group(active, None, "end-of-stream")


def macro_transition(group):
    """Return a JEPA-compatible transition row for one complete macro action.

    Incomplete groups return ``None``. ``prev`` and ``next`` use the structured
    ``current`` snapshots when available, matching existing transition files.
    """
    if not isinstance(group, dict) or not group.get("complete"):
        return None
    previous = _current(group.get("rootObservation"))
    following = _current(group.get("nextObservation"))
    if not isinstance(previous, dict) or not isinstance(following, dict):
        return None
    return {
        "source": "macro_actions",
        "matchId": group.get("matchId") or group.get("gameId"),
        "gameNumber": group.get("gameNumber"),
        "gameInstance": previous.get("gameInstance"),
        "horizon": max(1, int(group.get("decisionCount") or 1)),
        "prev": previous,
        "next": following,
        "action": copy.deepcopy(group.get("macroAction") or {}),
        "deltas": _transition_deltas(previous, following),
        "outcome": {
            "terminal": bool(group.get("terminal")),
            "reward": group.get("reward"),
            "result": copy.deepcopy(group.get("result")),
        },
        "macroActionId": group.get("groupId"),
    }


def _start_group(record, classification, identity, player):
    step = _step(record, classification)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    orphan = bool(classification["attachable"])
    group_id = _group_id(identity, player, record.get("sequenceNumber"))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": "MacroActionRecord",
        "source": record.get("source"),
        "groupId": group_id,
        "gameId": record.get("gameId"),
        "matchId": metadata.get("matchId") or record.get("matchId"),
        "gameNumber": metadata.get("gameNumber") or record.get("gameNumber"),
        "playerIndex": player,
        "rootSequenceNumber": record.get("sequenceNumber"),
        "lastSequenceNumber": record.get("sequenceNumber"),
        "rootObservation": copy.deepcopy(record.get("observation") or {}),
        "rootSelect": copy.deepcopy(_select(record)),
        "rootSelectedIndices": copy.deepcopy(record.get("selectedIndices") or []),
        "rootClassification": copy.deepcopy(classification),
        "steps": [step],
        "decisionCount": 1,
        "orphanedParameterGroup": orphan,
        "terminal": bool(record.get("terminal")),
        "reward": record.get("reward"),
        "result": copy.deepcopy(record.get("result")),
        "_identity": identity,
    }


def _append_step(group, record, classification):
    group["steps"].append(_step(record, classification))
    group["decisionCount"] += 1
    group["lastSequenceNumber"] = record.get("sequenceNumber")
    group["terminal"] = group["terminal"] or bool(record.get("terminal"))
    if record.get("reward") is not None:
        group["reward"] = record.get("reward")
    if record.get("result") is not None:
        group["result"] = copy.deepcopy(record.get("result"))


def _finish_group(group, boundary_observation, reason):
    value = dict(group)
    value.pop("_identity", None)
    explicit_next = None
    if group.get("steps"):
        explicit_next = group["steps"][-1].get("nextObservation")
    next_observation = explicit_next if explicit_next is not None else boundary_observation
    value["nextObservation"] = copy.deepcopy(next_observation)
    value["complete"] = bool(group.get("terminal")) or next_observation is not None
    value["completionReason"] = reason
    value["macroAction"] = _macro_action(value)
    confidences = [
        step["classification"].get("confidence")
        for step in value.get("steps") or []
    ]
    value["classificationConfidence"] = (
        "heuristic" if "heuristic" in confidences else "known-family"
    )
    return value


def _step(record, classification):
    return {
        "sequenceNumber": record.get("sequenceNumber"),
        "promptType": classification["promptType"],
        "classification": copy.deepcopy(classification),
        "selectedIndices": copy.deepcopy(record.get("selectedIndices") or []),
        "selectedOptions": selected_options(record),
        "nextObservation": copy.deepcopy(record.get("nextObservation")),
    }


def _macro_action(group):
    steps = group.get("steps") or []
    root = steps[0] if steps else {}
    selected = copy.deepcopy(root.get("selectedOptions") or [])
    labels = [
        str(option.get("label"))
        for option in selected
        if isinstance(option, dict) and option.get("label") is not None
    ]
    option_types = [
        option.get("type")
        for option in selected
        if isinstance(option, dict) and option.get("type") is not None
    ]
    canonical_keys = []
    for option in selected:
        payload = option.get("payload") if isinstance(option, dict) else None
        key = payload.get("canonicalKey") if isinstance(payload, dict) else None
        if isinstance(key, str) and key:
            canonical_keys.append(key)
    parameters = []
    for step in steps[1:]:
        parameters.append({
            "sequenceNumber": step.get("sequenceNumber"),
            "role": (step.get("classification") or {}).get("role"),
            "promptType": step.get("promptType"),
            "selectedIndices": copy.deepcopy(step.get("selectedIndices") or []),
            "selectedOptions": copy.deepcopy(step.get("selectedOptions") or []),
        })
    return {
        "type": option_types[0] if len(option_types) == 1
        else group.get("rootClassification", {}).get("promptType"),
        "label": " + ".join(labels) if labels else None,
        "promptType": group.get("rootClassification", {}).get("promptType"),
        "selectedIndices": copy.deepcopy(group.get("rootSelectedIndices") or []),
        "selectedOptions": selected,
        "canonicalKeys": canonical_keys,
        "parameters": parameters,
        "passLike": bool(group.get("rootClassification", {}).get("passLike")),
        "strategic": bool(group.get("rootClassification", {}).get("strategic")),
        "deterministic": bool(group.get("rootClassification", {}).get("deterministic")),
    }


def _forced_selection(select, option_count):
    if option_count == 1:
        return True
    minimum = select.get("minCount")
    maximum = select.get("maxCount")
    return (
        isinstance(minimum, int) and not isinstance(minimum, bool)
        and isinstance(maximum, int) and not isinstance(maximum, bool)
        and minimum == maximum == option_count
        and option_count > 0
    )


def _pass_like(options, selected):
    for index in selected:
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if not 0 <= index < len(options):
            continue
        option = options[index] if isinstance(options[index], dict) else {}
        text = "%s %s" % (option.get("type") or "", option.get("label") or "")
        lowered = text.lower()
        if any(word in lowered for word in _PASS_WORDS):
            return True
    return False


def _contains(value, hints):
    return any(hint in value for hint in hints)


def _select(record):
    if not isinstance(record, dict):
        return {}
    select = record.get("select")
    if isinstance(select, dict):
        return select
    observation = record.get("observation")
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}


def _options(select):
    options = select.get("option") if isinstance(select, dict) else None
    return options if isinstance(options, list) else []


def _player_index(record):
    value = record.get("playerIndex") if isinstance(record, dict) else None
    if value is None:
        value = _select(record).get("playerIndex")
    return value


def _game_identity(record):
    observation = record.get("observation") if isinstance(record, dict) else {}
    current = observation.get("current") if isinstance(observation, dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return (
        record.get("gameId") or metadata.get("matchId") or record.get("matchId"),
        metadata.get("gameNumber") if metadata.get("gameNumber") is not None
        else record.get("gameNumber"),
        current.get("gameInstance") if isinstance(current, dict) else None,
    )


def _group_id(identity, player, sequence):
    return "macro:%s:%s:%s" % (
        json.dumps(identity, separators=(",", ":"), sort_keys=False),
        player if player is not None else "?",
        sequence if sequence is not None else "?",
    )


def _current(observation):
    if not isinstance(observation, dict):
        return None
    current = observation.get("current")
    return current if isinstance(current, dict) else observation


def _transition_deltas(previous, following):
    previous_life = _life_by_player(previous)
    following_life = _life_by_player(following)
    return {
        "lifeDelta": {
            key: following_life[key] - previous_life[key]
            for key in following_life if key in previous_life
        },
        "gameOver": bool(following.get("gameOver")),
    }


def _life_by_player(state):
    result = {}
    for player in (state or {}).get("players") or []:
        if not isinstance(player, dict):
            continue
        key = player.get("seat", player.get("playerIndex"))
        life = player.get("life")
        if key is not None and isinstance(life, int) and not isinstance(life, bool):
            result[str(key)] = life
    return result
