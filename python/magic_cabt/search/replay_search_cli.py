"""CLI for exact one-step counterfactual labels by deterministic replay."""

from __future__ import annotations

import argparse
import json
import os
import sys

from magic_cabt.agents import make_agent
from magic_cabt.protocol import CabtBridge, load_decklist
from magic_cabt.training.io import iter_decision_records

from .replay_search import branch_replay, branch_to_transition

__all__ = ["build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-replay-search",
        description=(
            "Reconstruct a recorded decision from its game seed and action "
            "prefix, then branch each at-most-one legal option in a fresh "
            "XMage process."
        ),
    )
    parser.add_argument("--input", required=True,
                        help="DecisionRecord-compatible replay JSONL")
    parser.add_argument("--decision-index", required=True, type=int,
                        help="zero-based decision index within the selected game")
    parser.add_argument("--game-id", default=None,
                        help="optional canonical gameId filter")
    parser.add_argument("--deck0", required=True)
    parser.add_argument("--deck1", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--classpath", default=None)
    parser.add_argument("--out", required=True,
                        help="ReplaySearchResult JSON output")
    parser.add_argument("--transitions-out", default=None,
                        help="optional successful search transitions JSONL")
    parser.add_argument("--rollout-agent0", default="first")
    parser.add_argument("--rollout-agent1", default="first")
    parser.add_argument("--max-rollout-decisions", type=int, default=0)
    parser.add_argument("--keep-errors", action="store_true",
                        help="record failed branches instead of failing fast")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = list(iter_decision_records(args.input))
    if args.game_id is not None:
        records = [record for record in records
                   if record.get("gameId") == args.game_id]
    if not records:
        sys.stderr.write("no matching decision records\n")
        return 2
    if args.decision_index < 0 or args.decision_index >= len(records):
        sys.stderr.write("decision index is outside the matching game\n")
        return 2

    target = records[args.decision_index]
    target_game = _game_key(target)
    target_position = sum(
        1 for record in records[:args.decision_index]
        if _game_key(record) == target_game
    )
    game_records = [record for record in records if _game_key(record) == target_game]

    prefix = [
        record.get("selectedIndices") or []
        for record in game_records[:target_position]
    ]
    observation = dict(target.get("observation") or {})
    select = target.get("select") or {}

    def rollout_agent_factory(candidate):
        salt = sum((index + 1) * value for index, value in enumerate(candidate))
        return (
            make_agent(args.rollout_agent0, seed=args.seed + 100000 + salt),
            make_agent(args.rollout_agent1, seed=args.seed + 100001 + salt),
        )

    result = branch_replay(
        bridge_factory=lambda: CabtBridge(classpath=args.classpath),
        deck0=load_decklist(args.deck0),
        deck1=load_decklist(args.deck1),
        prefix=prefix,
        expected_observation=observation,
        expected_select=select,
        seed=args.seed,
        max_turns=args.max_turns,
        root_player_index=target.get("playerIndex"),
        rollout_agent_factory=rollout_agent_factory,
        max_rollout_decisions=max(0, args.max_rollout_decisions),
        fail_fast=not args.keep_errors,
    )
    result["gameId"] = target.get("gameId")
    result["sequenceNumber"] = target.get("sequenceNumber")
    result["sourceRecordLine"] = target.get("__source_line")

    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    transition_count = 0
    if args.transitions_out:
        _ensure_parent(args.transitions_out)
        with open(args.transitions_out, "w", encoding="utf-8") as handle:
            for branch in result.get("branches") or []:
                transition = branch_to_transition(result, branch)
                if transition is None:
                    continue
                transition["gameId"] = target.get("gameId")
                transition["sequenceNumber"] = target.get("sequenceNumber")
                handle.write(json.dumps(transition, sort_keys=True) + "\n")
                transition_count += 1

    errors = sum(1 for branch in result.get("branches") or []
                 if branch.get("error"))
    sys.stderr.write(
        "branches=%d errors=%d transitions=%d\n" %
        (len(result.get("branches") or []), errors, transition_count))
    return 0 if not errors else 2


def _game_key(record):
    observation = record.get("observation") or {}
    current = observation.get("current") or {}
    metadata = record.get("metadata") or {}
    return (
        record.get("gameId") or metadata.get("matchId"),
        metadata.get("gameNumber", record.get("gameNumber")),
        current.get("gameInstance"),
    )


def _ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
