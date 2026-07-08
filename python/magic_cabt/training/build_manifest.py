"""CLI: build a manifest for a DecisionRecord JSONL dataset.

Example:

    python -m magic_cabt.training.build_manifest \
        --input decisions.jsonl \
        --out manifest.json \
        --name arena-standard-session
"""

import argparse
import json
import sys

from .io import iter_decision_records
from .manifest import build_manifest, write_manifest

__all__ = ["main"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.training.build_manifest",
        description="Build a reproducibility/statistics manifest for a "
                    "DecisionRecord JSONL dataset.",
    )
    parser.add_argument("--input", required=True,
                        help="source/canonical DecisionRecord JSONL path")
    parser.add_argument("--out", default=None,
                        help="manifest JSON path (default: stdout)")
    parser.add_argument("--name", default=None,
                        help="optional dataset/run label")
    parser.add_argument("--source", default=None,
                        help="canonical source label override")
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(
            iter_decision_records(args.input, source_hint=args.source),
            name=args.name,
        )
    except ValueError as exc:
        sys.stderr.write("could not read dataset: %s\n" % exc)
        return 2
    except FileNotFoundError:
        sys.stderr.write("dataset not found: %s\n" % args.input)
        return 2

    if args.out:
        write_manifest(args.out, manifest)
    else:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
