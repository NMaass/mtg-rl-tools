"""Public trust-audit CLI over raw decision records.

The training compiler intentionally drops non-single-choice or invalid rows.
An audit must inspect those rows before compilation, so this wrapper temporarily
substitutes a raw decision iterator while reusing the core audit implementation.
"""
from __future__ import annotations

import json
import os

from . import trust_audit as core_audit


def iter_raw_decisions(inputs):
    for path in inputs:
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(path):
            path = os.path.join(path, "decisions.jsonl")
            if not os.path.isfile(path):
                continue
        elif os.path.basename(path) in (
                "mirror_states.jsonl", "transitions.jsonl"):
            continue
        yield from core_audit.core._decision_reader(path)


def audit(inputs, checkpoints=None, strict=False, max_records=0,
          visibility_samples=32):
    original = core_audit.core._iter_all_decisions
    core_audit.core._iter_all_decisions = iter_raw_decisions
    try:
        return core_audit.audit(
            inputs, checkpoints=checkpoints, strict=strict,
            max_records=max_records,
            visibility_samples=visibility_samples)
    finally:
        core_audit.core._iter_all_decisions = original


def main(argv=None):
    args = core_audit.build_parser().parse_args(argv)
    result = audit(
        args.input, checkpoints=args.checkpoint, strict=args.strict,
        max_records=max(0, args.max_records),
        visibility_samples=max(0, args.visibility_samples))
    output = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["trusted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
