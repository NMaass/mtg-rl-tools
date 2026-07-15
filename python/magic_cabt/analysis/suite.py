"""Validated public entry point for mixed checkpoint/baseline comparisons."""
from __future__ import annotations

import json
import os

from .baselines import (
    _parse_entry,
    build_parser,
    compare_suite as _compare_suite,
)


def validate_entries(entries):
    seen = set()
    validated = []
    for name, spec in entries:
        name = str(name).strip()
        spec = str(spec).strip()
        if not name or not spec:
            raise ValueError("model names and specifications must be non-empty")
        if name in seen:
            raise ValueError("duplicate model name: %s" % name)
        seen.add(name)
        validated.append((name, spec))
    if not validated:
        raise ValueError("at least one model is required")
    return validated


def compare_suite(bundle_dir, entries, *args, **kwargs):
    return _compare_suite(bundle_dir, validate_entries(entries), *args, **kwargs)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        entries = validate_entries([_parse_entry(value) for value in args.model])
        result = _compare_suite(
            args.bundle, entries, args.out, out_json=args.json,
            device=args.device, top_k=args.top_k, seed=args.seed,
            title=args.title,
            progress=lambda name, done, total: print(
                "[%s] %d/%d" % (name, done, total), file=os.sys.stderr))
    except (IOError, OSError, ValueError) as error:
        raise SystemExit(str(error))
    print(json.dumps({"html": result["html"], "json": result["json"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
