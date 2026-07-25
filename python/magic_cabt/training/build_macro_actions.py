"""Compile DecisionRecord streams into strategic macro-action records.

Example:

    magic-cabt-build-macro-actions \
      --input arena-mirror-runs/run-001/decisions.jsonl \
      --out target/macro-actions.jsonl \
      --transitions-out target/transitions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .io import iter_decision_records
from .macro_actions import iter_macro_actions, macro_transition

__all__ = ["build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-build-macro-actions",
        description=(
            "Group low-level CABT target/mode/payment prompts under their "
            "strategic root decisions."
        ),
    )
    parser.add_argument("--input", required=True, action="append",
                        help="DecisionRecord-compatible JSONL (repeatable)")
    parser.add_argument("--out", required=True,
                        help="MacroActionRecord JSONL output")
    parser.add_argument("--transitions-out", default=None,
                        help="optional JEPA-compatible complete-transition JSONL")
    parser.add_argument("--source", default=None,
                        help="canonical source override passed to the record reader")
    parser.add_argument("--strategic-only", action="store_true",
                        help="omit deterministic/non-strategic root groups")
    parser.add_argument("--complete-only", action="store_true",
                        help="omit groups without an observed boundary state")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _ensure_parent(args.out)
    if args.transitions_out:
        _ensure_parent(args.transitions_out)

    stats = {
        "groups": 0,
        "written": 0,
        "complete": 0,
        "strategic": 0,
        "orphanedParameterGroups": 0,
        "transitions": 0,
        "completionReasons": {},
        "rootPromptTypes": {},
    }

    transition_handle = None
    try:
        if args.transitions_out:
            transition_handle = open(args.transitions_out, "w", encoding="utf-8")
        with open(args.out, "w", encoding="utf-8") as output:
            for path in args.input:
                groups = iter_macro_actions(
                    iter_decision_records(path, source_hint=args.source))
                for group in groups:
                    stats["groups"] += 1
                    _inc(stats["completionReasons"], group.get("completionReason"))
                    root_prompt = (group.get("rootClassification") or {}).get(
                        "promptType")
                    _inc(stats["rootPromptTypes"], root_prompt)
                    if group.get("complete"):
                        stats["complete"] += 1
                    if (group.get("macroAction") or {}).get("strategic"):
                        stats["strategic"] += 1
                    if group.get("orphanedParameterGroup"):
                        stats["orphanedParameterGroups"] += 1
                    if args.complete_only and not group.get("complete"):
                        continue
                    if args.strategic_only and not (
                            group.get("macroAction") or {}).get("strategic"):
                        continue
                    output.write(json.dumps(group, sort_keys=True) + "\n")
                    stats["written"] += 1
                    transition = macro_transition(group)
                    if transition_handle is not None and transition is not None:
                        transition_handle.write(
                            json.dumps(transition, sort_keys=True) + "\n")
                        stats["transitions"] += 1
    except (OSError, ValueError) as error:
        sys.stderr.write("could not build macro actions: %s\n" % error)
        return 2
    finally:
        if transition_handle is not None:
            transition_handle.close()

    sys.stderr.write(json.dumps(stats, sort_keys=True) + "\n")
    return 0 if stats["written"] else 2


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _inc(bucket, key):
    key = str(key or "UNKNOWN")
    bucket[key] = bucket.get(key, 0) + 1


if __name__ == "__main__":
    raise SystemExit(main())
