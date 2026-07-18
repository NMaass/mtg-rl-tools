"""CLI for replaying observed actions through XMage."""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

from mtg_state_contract.jsonl import read_states

from .follower import FollowConfig, SubprocessReplayBackend, XmageFollower
from .protocol import ReplayManifest, load_decklist


def build_parser():
    parser = argparse.ArgumentParser(prog="xmage-follow")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--deck0", default=None)
    parser.add_argument("--deck1", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--classpath", default=None)
    parser.add_argument("--command", default=None,
                        help="quoted bridge command; overrides classpath")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--minimum-action-score", type=float, default=45.0)
    parser.add_argument("--max-hard-mismatches", type=int, default=0)
    parser.add_argument("--maximum-implicit-steps", type=int, default=8)
    parser.add_argument("--implicit-candidates", type=int, default=2)
    parser.add_argument("--state-time-tolerance-ms", type=int, default=8000)
    parser.add_argument("--ambiguity-is-error", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.manifest:
        manifest = ReplayManifest.load(args.manifest)
    else:
        if not args.deck0 or not args.deck1:
            raise SystemExit("provide --manifest or both --deck0 and --deck1")
        manifest = ReplayManifest(
            load_decklist(args.deck0), load_decklist(args.deck1), seed=args.seed)
    actions = []
    with open(args.actions, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                actions.append(json.loads(line))
    states = list(read_states(args.states))
    command = shlex.split(args.command) if args.command else None
    backend = SubprocessReplayBackend(
        manifest, command=command,
        classpath=args.classpath or os.environ.get("MAGIC_CABT_CLASSPATH"))
    config = FollowConfig(
        beam_width=max(1, args.beam_width),
        candidates_per_step=max(1, args.candidates),
        minimum_action_score=args.minimum_action_score,
        max_hard_mismatches=max(0, args.max_hard_mismatches),
        maximum_implicit_steps=max(0, args.maximum_implicit_steps),
        implicit_candidates_per_prompt=max(0, args.implicit_candidates),
        state_time_tolerance_ms=max(0, args.state_time_tolerance_ms),
        ambiguity_is_error=args.ambiguity_is_error)
    report = XmageFollower(backend, manifest, config).follow(actions, states)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"passed": report.passed, "verified": report.verified,
                      "steps": len(report.steps),
                      "hypotheses": len(report.final_hypotheses),
                      "out": str(target)}, indent=2))
    return 0 if report.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
