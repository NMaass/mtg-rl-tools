"""Fail-fast trust audit for Magic datasets and checkpoints.

Dataset gates run without PyTorch. Checkpoint tensor inspection is optional
unless ``--require-checkpoint-audit`` is supplied.
"""
from __future__ import annotations

import argparse
import json
import os

from .trust_audit_common import (
    AuditReport,
    classify_file,
    input_files,
    sha256_file,
)
from .trust_checkpoint_audit import audit_checkpoint
from .trust_dataset_audit import (
    audit_decisions,
    audit_macro_actions,
    audit_transitions,
)


def audit(inputs, checkpoints=None, strict=False, max_records=0,
          require_checkpoint_audit=False):
    checkpoints = checkpoints or []
    report = AuditReport(strict=strict)
    files = []
    for path in input_files(inputs):
        kind, error = classify_file(path)
        row = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "kind": kind,
        }
        if error:
            row["classificationError"] = error
        files.append(row)

    report.add(
        "dataset.inputs.present", "pass" if files else "fail",
        "Input files were resolved" if files else "No readable input files found",
        {"files": len(files)}, scope="dataset")
    invalid = [row for row in files
               if row["kind"] in ("invalid", "unknown", "empty")]
    report.add(
        "dataset.inputs.classification",
        "fail" if invalid else ("pass" if files else "not-applicable"),
        "All input files have recognized non-empty record shapes"
        if files and not invalid else
        ("Some input files are malformed or have unknown record shapes"
         if invalid else "No input files were supplied"),
        {"invalid": invalid[:20]}, scope="dataset")

    decisions = audit_decisions(report, inputs, max_records=max_records)
    transitions = audit_transitions(report, inputs, max_records=max_records)
    audit_macro_actions(report, inputs, max_records=max_records)
    games = decisions["games"].union(transitions["games"])
    for checkpoint in checkpoints:
        audit_checkpoint(
            report, checkpoint, games,
            require_checkpoint_audit=require_checkpoint_audit)
    if not checkpoints:
        report.add("checkpoint.present", "not-applicable",
                   "No checkpoints were supplied", scope="checkpoint")
    return report.finish(inputs, checkpoints, files)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-training-audit",
        description="Fail-fast trust audit for training data and checkpoints.")
    parser.add_argument("--input", action="append", required=True,
                        help="bundle directory or JSONL file; repeatable")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="checkpoint to audit; repeatable")
    parser.add_argument("--out", required=True,
                        help="machine-readable JSON report")
    parser.add_argument("--strict", action="store_true",
                        help="promote warnings to failures")
    parser.add_argument("--max-records", type=int, default=0,
                        help="0 audits all records")
    parser.add_argument(
        "--require-checkpoint-audit", action="store_true",
        help="fail instead of warn when PyTorch is unavailable")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = audit(
        args.input, checkpoints=args.checkpoint, strict=args.strict,
        max_records=max(0, args.max_records),
        require_checkpoint_audit=args.require_checkpoint_audit)
    output = os.path.abspath(os.path.expanduser(args.out))
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["trusted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
