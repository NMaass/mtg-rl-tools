"""Build a state-transition dataset for JEPA-style world-model pretraining.

    magic-cabt-build-transitions --input <run-dir-or-jsonl> --out transitions.jsonl

Consumes either Arena mirror run directories (pairs consecutive
``mirror_states.jsonl`` snapshots within one game) or DecisionRecord JSONL
(pairs consecutive decision observations within one game, keeping the action
taken between them). Emits one JSON object per line:

    {"source", "matchId", "gameNumber", "prev", "next",
     "action" | null, "deltas"}

``action`` (DecisionRecord inputs only) carries the SEMANTICS of the choice,
not just its position: ``selectedOptions`` holds the full selected option
dicts (type, label, payload with ``canonicalKey`` and target references),
because a bare index has no stable meaning across states -- an
action-conditioned predictor must condition on what was chosen, not on where
it sat in the list. ``selectedIndices``/``optionCount`` are retained for
provenance only.

``deltas`` holds exact engine-derived transition labels that are cheap to
compute here (per-player life change, terminal flag) for auxiliary
supervision heads.

World-model pretraining needs trajectories, not human choices, so self-play
replays and mirror states both count; every recorded game contributes
hundreds of transitions even though it has far fewer decisions.
"""

import argparse
import json
import os
import sys

from magic_cabt.training.io import iter_decision_records

__all__ = [
    "transitions_from_states",
    "transitions_from_decisions",
    "build_parser",
    "main",
]

_STATE_FILENAME = "mirror_states.jsonl"


def _game_key(entry):
    return (entry.get("matchId"), entry.get("gameNumber"),
            entry.get("gameInstance"))


def _life_by_player(state):
    result = {}
    for player in (state or {}).get("players") or []:
        if not isinstance(player, dict):
            continue
        key = player.get("seat", player.get("playerIndex"))
        if key is not None and isinstance(player.get("life"), int):
            result[str(key)] = player["life"]
    return result


def _transition_deltas(prev_state, next_state):
    """Exact engine-derived labels for auxiliary supervision heads."""
    prev_life = _life_by_player(prev_state)
    next_life = _life_by_player(next_state)
    return {
        "lifeDelta": {key: next_life[key] - prev_life[key]
                      for key in next_life if key in prev_life},
        "gameOver": bool((next_state or {}).get("gameOver")),
    }


def transitions_from_states(states, source="mirror_states"):
    """Pair consecutive snapshots within one game (no action attribution)."""
    previous = None
    for state in states:
        if not isinstance(state, dict):
            continue
        if previous is not None and _game_key(previous) == _game_key(state):
            yield {
                "source": source,
                "matchId": state.get("matchId"),
                "gameNumber": state.get("gameNumber"),
                "prev": previous,
                "next": state,
                "action": None,
                "deltas": _transition_deltas(previous, state),
            }
        previous = state


def transitions_from_decisions(records, source="decisions"):
    """Pair consecutive decision observations, keeping the acting choice."""
    previous = None
    for record in records:
        observation = record.get("observation") or {}
        current = observation.get("current")
        if not isinstance(current, dict):
            continue
        select = record.get("select") or {}
        options = select.get("option") or []
        selected = record.get("selectedIndices") or []
        entry = {
            "matchId": record.get("matchId") or record.get("gameId"),
            "gameNumber": record.get("gameNumber"),
            "gameInstance": current.get("gameInstance"),
            "state": current,
            "action": {
                "promptType": select.get("type"),
                # the semantics of the choice: full option dicts, whose
                # payloads carry canonicalKey and target references
                "selectedOptions": [
                    options[index] for index in selected
                    if isinstance(index, int) and 0 <= index < len(options)
                ],
                # provenance only -- indices are not stable across states
                "selectedIndices": selected,
                "optionCount": len(options),
            },
        }
        if previous is not None and _game_key(previous) == _game_key(entry):
            yield {
                "source": source,
                "matchId": entry["matchId"],
                "gameNumber": entry["gameNumber"],
                "prev": previous["state"],
                "next": entry["state"],
                "action": previous["action"],
                "deltas": _transition_deltas(previous["state"],
                                             entry["state"]),
            }
        previous = entry


def _iter_json_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _transitions_from_path(path):
    """Yield transitions from a run dir, mirror-state JSONL, or records JSONL."""
    if os.path.isdir(path):
        states_path = os.path.join(path, _STATE_FILENAME)
        if os.path.exists(states_path):
            for transition in transitions_from_states(
                    _iter_json_lines(states_path)):
                yield transition
        decisions_path = os.path.join(path, "decisions.jsonl")
        if os.path.exists(decisions_path):
            for transition in transitions_from_decisions(
                    iter_decision_records(decisions_path)):
                yield transition
        return
    if os.path.basename(path) == _STATE_FILENAME:
        for transition in transitions_from_states(_iter_json_lines(path)):
            yield transition
        return
    for transition in transitions_from_decisions(iter_decision_records(path)):
        yield transition


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-build-transitions",
        description="Build a JEPA pretraining transition dataset.")
    parser.add_argument("--input", required=True, action="append",
                        help="run directory, mirror_states.jsonl, or "
                             "DecisionRecord JSONL (repeatable)")
    parser.add_argument("--out", required=True, help="output JSONL path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    counts = {}
    total = 0
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for path in args.input:
            for transition in _transitions_from_path(path):
                handle.write(json.dumps(transition, sort_keys=True) + "\n")
                counts[transition["source"]] = (
                    counts.get(transition["source"], 0) + 1)
                total += 1
    sys.stderr.write("wrote %d transitions to %s (%s)\n" % (
        total, args.out, json.dumps(counts, sort_keys=True)))
    return 0 if total else 2


if __name__ == "__main__":
    raise SystemExit(main())
