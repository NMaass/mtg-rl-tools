"""CLI: build a manifest for one or more DecisionRecord datasets.

Example:

    python -m magic_cabt.training.build_manifest \
        --input decisions-a.jsonl \
        --input run-b/ \
        --out manifest.json \
        --name arena-standard-corpus
"""
import argparse
import hashlib
import json
import os
import sys

from .io import iter_decision_records
from .manifest import build_manifest, write_manifest

__all__ = ["main"]


def _resolve_input(path):
    resolved = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(resolved):
        resolved = os.path.join(resolved, "decisions.jsonl")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_records(paths, source_hint=None):
    for path in paths:
        for record in iter_decision_records(path, source_hint=source_hint):
            yield record


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.training.build_manifest",
        description="Build a reproducibility and corpus-coverage manifest for "
                    "one or more DecisionRecord datasets.",
    )
    parser.add_argument("--input", required=True, action="append",
                        help="DecisionRecord JSONL or bundle directory; repeatable")
    parser.add_argument("--out", default=None,
                        help="manifest JSON path (default: stdout)")
    parser.add_argument("--name", default=None,
                        help="optional dataset/corpus label")
    parser.add_argument("--source", default=None,
                        help="canonical source label override")
    args = parser.parse_args(argv)

    try:
        paths = [_resolve_input(path) for path in args.input]
        manifest = build_manifest(
            _iter_records(paths, source_hint=args.source),
            name=args.name,
        )
        manifest["inputs"] = [
            {"path": path, "bytes": os.path.getsize(path), "sha256": _sha256(path)}
            for path in paths
        ]
    except ValueError as exc:
        sys.stderr.write("could not read dataset: %s\n" % exc)
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write("dataset not found: %s\n" % exc)
        return 2

    if args.out:
        write_manifest(args.out, manifest)
    else:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
