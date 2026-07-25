"""Exact counterfactual branching by deterministic game replay.

Native cloned-game search is preferable, but it is not required to begin
producing exact search labels. Given the original decks, seed, and action
prefix, this module starts a fresh bridge for each candidate, replays the prefix
fail-closed, verifies that the reconstructed public root matches the recorded
root, and then applies one alternative legal selection.

The method is intentionally slow: O(candidate_count * prefix_length). Its role
is correctness-first dataset generation and regression testing. A future
native ``search_begin/search_step`` implementation can keep this result schema
while replacing the replay backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

__all__ = [
    "ReplayDivergenceError",
    "branch_replay",
    "branch_to_transition",
    "candidate_selections",
    "observation_signature",
    "replay_to_root",
]

SCHEMA_VERSION = 1
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_INSTANCE_RE = re.compile(
    r"\b(?:instance|instanceId|targetInstanceId|objectId|sourceId)=\S+")
_VOLATILE_KEYS = {
    "timestamp",
    "promptTimestamp",
    "responseTimestamp",
    "sequence",
    "sequenceNumber",
    "seq",
    "gameInstance",
}
_STABLE_ID_KEYS = {"grpId", "playerIndex", "seat", "gameNumber", "turnNumber"}
_UNORDERED_LIST_KEYS = {
    "battlefield",
    "graveyard",
    "exile",
    "players",
    "command",
    "emblems",
}


class ReplayDivergenceError(RuntimeError):
    """The replayed action prefix did not reconstruct the expected root."""

    def __init__(self, message, expected=None, actual=None):
        super(ReplayDivergenceError, self).__init__(message)
        self.expected = expected
        self.actual = actual


def candidate_selections(select):
    """Enumerate all at-most-one legal selections for a select envelope.

    Multi-select combinatorics are deliberately not guessed. Callers must
    provide explicit candidate lists for attacker/blocker or pile choices.
    """
    select = select if isinstance(select, dict) else {}
    options = _options(select)
    minimum, maximum = _count_bounds(select, len(options))
    if maximum > 1 or minimum > 1:
        raise ValueError(
            "automatic candidate enumeration only supports at-most-one "
            "selection; provide explicit candidates for %s" %
            (select.get("type") or "UNKNOWN"))
    candidates = []
    if minimum == 0:
        candidates.append([])
    candidates.extend([[index] for index in range(len(options))])
    return candidates


def observation_signature(observation, select=None):
    """Return a stable semantic signature for replay-root verification."""
    if isinstance(observation, dict) and "observation" in observation:
        response = observation
        observation = response.get("observation") or {}
    observation = observation if isinstance(observation, dict) else {}
    nested_select = observation.get("select")
    select = select if isinstance(select, dict) else (
        nested_select if isinstance(nested_select, dict) else {})
    current = observation.get("current")
    if not isinstance(current, dict):
        current = {key: value for key, value in observation.items()
                   if key != "select"}
    payload = {
        "current": _normalize_current(current),
        "select": {
            "type": select.get("type"),
            "minCount": select.get("minCount"),
            "maxCount": select.get("maxCount"),
            "playerIndex": select.get("playerIndex"),
            "options": [_option_signature(option) for option in _options(select)],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def replay_to_root(bridge, deck0, deck1, prefix, seed=None, max_turns=None,
                   player_names=("P0", "P1"), expected_observation=None,
                   expected_select=None):
    """Replay a legal action prefix and return the reconstructed root response."""
    response = bridge.game_start(
        deck0, deck1, player_names=list(player_names), seed=seed,
        max_turns=max_turns)
    for offset, selection in enumerate(prefix):
        if bridge.finished:
            raise ReplayDivergenceError(
                "replay finished before prefix action %d" % offset)
        select = _response_select(response)
        if not _is_legal_selection(selection, select):
            raise ReplayDivergenceError(
                "recorded prefix action %d is illegal at reconstructed prompt" %
                offset,
                expected=selection,
                actual=select,
            )
        response = bridge.game_select(list(selection))

    if expected_observation is not None or expected_select is not None:
        actual_observation = response.get("observation") or {}
        actual_select = _response_select(response)
        expected = observation_signature(
            expected_observation or {}, expected_select or {})
        actual = observation_signature(actual_observation, actual_select)
        if expected["sha256"] != actual["sha256"]:
            raise ReplayDivergenceError(
                "replayed prefix does not match the expected public root",
                expected=expected,
                actual=actual,
            )
    return response


def branch_replay(bridge_factory, deck0, deck1, prefix,
                  expected_observation, expected_select, seed=None,
                  max_turns=None, player_names=("P0", "P1"),
                  candidates=None, root_player_index=None,
                  rollout_agents=None, rollout_agent_factory=None,
                  max_rollout_decisions=0, fail_fast=True):
    """Evaluate candidate selections from one reconstructed decision root.

    Every candidate receives a fresh bridge. By default the function records
    only the exact immediate successor. Set ``max_rollout_decisions`` and pass
    two rollout agents (or a selector callback) to continue toward a terminal
    outcome.
    """
    expected_select = expected_select if isinstance(expected_select, dict) else {}
    prefix = [list(selection) for selection in prefix]
    candidates = list(candidates) if candidates is not None \
        else candidate_selections(expected_select)
    if root_player_index is None:
        root_player_index = expected_select.get("playerIndex")
    root_signature = observation_signature(expected_observation, expected_select)
    branches = []

    for candidate in candidates:
        try:
            branch = _run_branch(
                bridge_factory, deck0, deck1, prefix,
                expected_observation, expected_select, candidate,
                seed=seed, max_turns=max_turns, player_names=player_names,
                root_player_index=root_player_index,
                rollout_agents=(rollout_agent_factory(candidate)
                                if rollout_agent_factory is not None
                                else rollout_agents),
                max_rollout_decisions=max_rollout_decisions)
            branches.append(branch)
        except Exception as error:  # fail-closed result preservation
            if fail_fast:
                raise
            branches.append({
                "selectedIndices": copy.deepcopy(candidate),
                "error": "%s: %s" % (type(error).__name__, error),
                "complete": False,
            })

    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": "ReplaySearchResult",
        "backend": "deterministic-replay",
        "seed": seed,
        "maxTurns": max_turns,
        "prefixLength": len(prefix),
        "rootPlayerIndex": root_player_index,
        "rootObservation": copy.deepcopy(expected_observation),
        "rootSelect": copy.deepcopy(expected_select),
        "rootSignature": root_signature,
        "branches": branches,
    }


def branch_to_transition(search_result, branch):
    """Convert one successful replay branch into a search transition row."""
    if not isinstance(branch, dict) or branch.get("error"):
        return None
    root_observation = search_result.get("rootObservation") or {}
    next_observation = branch.get("nextObservation") or {}
    previous = root_observation.get("current") \
        if isinstance(root_observation, dict) else None
    following = next_observation.get("current") \
        if isinstance(next_observation, dict) else None
    if not isinstance(previous, dict) or not isinstance(following, dict):
        return None
    return {
        "source": "search",
        "horizon": 1,
        "prev": copy.deepcopy(previous),
        "next": copy.deepcopy(following),
        "action": {
            "promptType": (search_result.get("rootSelect") or {}).get("type"),
            "selectedIndices": copy.deepcopy(branch.get("selectedIndices") or []),
            "selectedOptions": copy.deepcopy(branch.get("selectedOptions") or []),
            "optionCount": len(
                _options(search_result.get("rootSelect") or {})),
        },
        "outcome": {
            "reward": branch.get("reward"),
            "result": copy.deepcopy(branch.get("result")),
            "terminal": bool(branch.get("finished")),
            "rolloutTruncated": bool(branch.get("rolloutTruncated")),
        },
        "search": {
            "backend": search_result.get("backend"),
            "rootSignature": (search_result.get("rootSignature") or {}).get(
                "sha256"),
            "prefixLength": search_result.get("prefixLength"),
            "rolloutDecisions": branch.get("rolloutDecisions"),
        },
    }


def _run_branch(bridge_factory, deck0, deck1, prefix,
                expected_observation, expected_select, candidate,
                seed, max_turns, player_names, root_player_index,
                rollout_agents, max_rollout_decisions):
    bridge = bridge_factory()
    try:
        response = replay_to_root(
            bridge, deck0, deck1, prefix, seed=seed, max_turns=max_turns,
            player_names=player_names,
            expected_observation=expected_observation,
            expected_select=expected_select)
        actual_select = _response_select(response)
        if not _is_legal_selection(candidate, actual_select):
            raise ValueError("candidate selection is illegal at replayed root: %r" %
                             (candidate,))
        selected = _selected_options(actual_select, candidate)
        response = bridge.game_select(list(candidate))
        next_observation = copy.deepcopy(response.get("observation")) \
            if not bridge.finished else None

        rollout_decisions = 0
        while not bridge.finished and rollout_decisions < max_rollout_decisions:
            observation = response.get("observation") or {}
            select = observation.get("select") or {}
            seat = select.get("playerIndex")
            selection = _rollout_selection(
                rollout_agents, observation, select, seat)
            if not _is_legal_selection(selection, select):
                raise ValueError(
                    "rollout produced illegal selection %r for seat %r" %
                    (selection, seat))
            response = bridge.game_select(selection)
            rollout_decisions += 1

        result = copy.deepcopy(getattr(bridge, "result", None))
        reward = _reward_for_player(result, root_player_index, player_names)
        return {
            "selectedIndices": copy.deepcopy(candidate),
            "selectedOptions": selected,
            "nextObservation": next_observation,
            "finished": bool(bridge.finished),
            "result": result,
            "reward": reward,
            "rolloutDecisions": rollout_decisions,
            "rolloutTruncated": bool(
                not bridge.finished and
                rollout_decisions >= max_rollout_decisions and
                max_rollout_decisions > 0),
            "complete": True,
        }
    finally:
        close = getattr(bridge, "close", None)
        if callable(close):
            close()


def _rollout_selection(rollout_agents, observation, select, seat):
    if rollout_agents is None:
        minimum, _maximum = _count_bounds(select, len(_options(select)))
        return list(range(minimum))
    if callable(rollout_agents):
        return list(rollout_agents(observation, seat))
    if not isinstance(seat, int) or seat < 0 or seat >= len(rollout_agents):
        raise ValueError("cannot route rollout prompt to seat %r" % (seat,))
    return list(rollout_agents[seat].select(observation))


def _reward_for_player(result, player_index, player_names):
    if player_index not in (0, 1) or not isinstance(result, dict):
        return None
    winner = result.get("winner")
    if isinstance(winner, int) and winner in (0, 1):
        return 1.0 if winner == player_index else 0.0
    if isinstance(winner, str):
        for seat, name in enumerate(player_names):
            if name and name in winner:
                return 1.0 if seat == player_index else 0.0
        lowered = winner.lower()
        if "draw" in lowered:
            return 0.5
    if result.get("draw") is True:
        return 0.5
    return None


def _response_select(response):
    observation = response.get("observation") if isinstance(response, dict) else None
    select = observation.get("select") if isinstance(observation, dict) else None
    return select if isinstance(select, dict) else {}


def _selected_options(select, selected):
    options = _options(select)
    return [
        copy.deepcopy(options[index])
        for index in selected
        if isinstance(index, int) and not isinstance(index, bool)
        and 0 <= index < len(options)
    ]


def _is_legal_selection(indices, select):
    if not isinstance(indices, list):
        return False
    option_count = len(_options(select))
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        or value < 0 or value >= option_count
        for value in indices
    ):
        return False
    if len(indices) != len(set(indices)):
        return False
    minimum, maximum = _count_bounds(select, option_count)
    return minimum <= len(indices) <= maximum


def _count_bounds(select, option_count):
    minimum = select.get("minCount") if isinstance(select, dict) else None
    maximum = select.get("maxCount") if isinstance(select, dict) else None
    minimum = minimum if isinstance(minimum, int) and not isinstance(minimum, bool) else 0
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        maximum = option_count
    return minimum, maximum


def _options(select):
    options = select.get("option") if isinstance(select, dict) else None
    return options if isinstance(options, list) else []


def _normalize_current(current):
    """Normalize a public state while preserving player-role semantics.

    XMage player ids are process-local UUIDs, so direct comparison would make
    deterministic replay appear divergent. Before generic id scrubbing, map
    active/priority player ids to their stable playerIndex/seat labels.
    """
    current = current if isinstance(current, dict) else {}
    id_to_seat = {}
    for player in current.get("players") or []:
        if not isinstance(player, dict):
            continue
        player_id = player.get("playerId")
        seat = player.get("playerIndex", player.get("seat"))
        if player_id is not None and seat is not None:
            id_to_seat[str(player_id)] = seat
    value = dict(current)
    for key in ("activePlayerId", "priorityPlayerId", "startingPlayerId"):
        player_id = value.get(key)
        if player_id is not None:
            value[key[:-2] + "Index"] = id_to_seat.get(str(player_id), "unknown")
    return _normalize_value(value)


def _option_signature(option):
    option = option if isinstance(option, dict) else {}
    payload = option.get("payload") if isinstance(option.get("payload"), dict) else {}
    return {
        "type": option.get("type"),
        "label": _scrub_text(option.get("label")),
        "canonicalKey": payload.get("canonicalKey"),
        "source": _semantic_name(payload.get("source")),
        "card": _semantic_name(payload.get("card")),
        "name": payload.get("name"),
    }


def _semantic_name(value):
    if isinstance(value, dict):
        for key in ("name", "label", "type", "canonicalKey", "grpId"):
            if value.get(key) is not None:
                return value.get(key)
        ref = value.get("ref")
        return _semantic_name(ref) if isinstance(ref, dict) else None
    if value is None:
        return None
    return _scrub_text(value)


def _normalize_value(value, parent_key=None):
    if isinstance(value, dict):
        normalized = {}
        for key in sorted(value):
            if key in _VOLATILE_KEYS:
                continue
            if key.endswith("Id") and key not in _STABLE_ID_KEYS:
                continue
            normalized[key] = _normalize_value(value[key], key)
        return normalized
    if isinstance(value, list):
        items = [_normalize_value(item, parent_key) for item in value]
        if parent_key in _UNORDERED_LIST_KEYS:
            return sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")),
            )
        return items
    if isinstance(value, str):
        return _scrub_text(value)
    return value


def _scrub_text(value):
    if value is None:
        return None
    return _INSTANCE_RE.sub("<instance>", _UUID_RE.sub("<uuid>", str(value)))
