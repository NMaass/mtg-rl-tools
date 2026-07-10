"""CLI: analyse action-space compressibility for DecisionRecord JSONL.

Example:

    python -m magic_cabt.training.analyze_actions \
        --input decisions.jsonl \
        --profile small \
        --out action_entropy.small.json
"""

import argparse
import json
import sys

from .action_entropy import analyze_action_entropy
from .io import iter_decision_records

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.training.analyze_actions",
        description="Measure action-bucket entropy and legal-action-count "
                    "distributions for source/canonical DecisionRecord JSONL.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--profile", default="small", choices=("small", "full"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--out", default=None,
                        help="write JSON report to this path instead of stdout")
    args = parser.parse_args(argv)

    try:
        report = analyze_action_entropy(
            iter_decision_records(args.input, source_hint=args.source),
            profile=args.profile,
        )
    except ValueError as exc:
        sys.stderr.write("could not read dataset: %s\n" % exc)
        return 2
    except FileNotFoundError:
        sys.stderr.write("dataset not found: %s\n" % args.input)
        return 2

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
