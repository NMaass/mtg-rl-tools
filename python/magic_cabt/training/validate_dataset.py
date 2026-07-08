"""CLI: ``python -m magic_cabt.training.validate_dataset <path>``.

Reads a JSONL file of source records via ``iter_decision_records`` (so any
supported source — Java transition dataset, self-play replay frames, Arena
decisions.jsonl, future search rollouts — is normalized first), then prints
a summary and exits nonzero when invalid records were found.
"""

import argparse
import sys

from .io import iter_decision_records
from .records import validate_records

__all__ = ["main"]

# Hard cap on individually printed errors so a giant broken dataset does not
# flood a CI log. The aggregate counts are still reported in full.
_MAX_PRINTED_ERRORS = 5


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m magic_cabt.training.validate_dataset",
        description="Validate a JSONL DecisionRecord dataset. Exits nonzero "
                    "when any record fails normalizer + validator checks.",
    )
    parser.add_argument("path",
                        help="path to a normalized or source-format JSONL file")
    parser.add_argument("--source", default=None,
                        help="canonical source label override "
                             "(arena_human, engine_selfplay, engine_human, "
                             "search)")
    parser.add_argument("--max-errors", type=int, default=_MAX_PRINTED_ERRORS,
                        help="how many per-record error blocks to print in "
                             "full before truncating (default: %d)"
                             % _MAX_PRINTED_ERRORS)
    args = parser.parse_args(argv)

    try:
        record_iter = list(iter_decision_records(args.path,
                                                   source_hint=args.source))
    except ValueError as exc:
        sys.stderr.write("could not read dataset: %s\n" % exc)
        return 2
    except FileNotFoundError:
        sys.stderr.write("dataset not found: %s\n" % args.path)
        return 2

    summary = validate_records(record_iter)

    _print_summary(summary, args.max_errors)

    if summary["invalid"] > 0:
        return 1
    return 0


def _print_summary(summary, max_errors):
    sys.stdout.write("total records:    %d\n" % summary["total"])
    sys.stdout.write("valid records:    %d\n" % summary["valid"])
    sys.stdout.write("invalid records:  %d\n" % summary["invalid"])
    sys.stdout.write("\n")

    if summary["selectTypes"]:
        sys.stdout.write("select.type distribution:\n")
        for label in sorted(summary["selectTypes"]):
            sys.stdout.write("  %-32s %d\n"
                             % (label, summary["selectTypes"][label]))
        sys.stdout.write("\n")

    if summary["optionTypes"]:
        sys.stdout.write("option.type distribution:\n")
        for label in sorted(summary["optionTypes"]):
            sys.stdout.write("  %-32s %d\n"
                             % (label, summary["optionTypes"][label]))
        sys.stdout.write("\n")

    if summary["selectedCount"]:
        sys.stdout.write("selected-count distribution:\n")
        for bucket in sorted(summary["selectedCount"], key=_count_sort_key):
            sys.stdout.write("  %-8s %d\n"
                             % (bucket, summary["selectedCount"][bucket]))
        sys.stdout.write("\n")

    if summary["errors"]:
        sys.stdout.write("validation errors (first %d):\n" % max_errors)
        for entry in summary["errors"][:max_errors]:
            sys.stdout.write("  record %d (line %s):\n"
                             % (entry["record"], entry["line"]))
            for message in entry["messages"]:
                sys.stdout.write("    - %s\n" % message)
        truncated = len(summary["errors"]) - max_errors
        if truncated > 0:
            sys.stdout.write("  ... and %d more error block(s) truncated\n"
                             % truncated)


def _count_sort_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
