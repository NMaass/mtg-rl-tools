"""Compile DecisionRecord JSONL into single-choice IL examples.

Command:

    python -m magic_cabt.training.compile_il \
      --input decisions.jsonl \
      --out target/il/single_choice.jsonl \
      --single-choice-only

Input may be any source format understood by
``magic_cabt.training.iter_decision_records``: canonical DecisionRecord JSONL,
Java transition JSONL, self-play replay frames, or Arena mirror
``decisions.jsonl``. The compiler normalizes first so downstream code always
sees canonical ``select`` and ``selectedIndices`` fields.

Output is one JSON object per supported decision with the stable contract:
``schemaVersion``, ``gameId``, ``sequenceNumber``, ``playerIndex``,
``promptType``, ``optionTypes``, ``stateText``, ``optionTexts``,
``chosenIndex``, and ``metadata``.
"""

import argparse
import json
import os
import sys

from magic_cabt.training import features
from magic_cabt.training.io import iter_decision_records

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "compile_record",
    "compile_records",
    "is_single_choice",
    "main",
]


def is_single_choice(record):
    select = _select(record)
    selected = record.get("selectedIndices") if isinstance(record, dict) else None
    return (
        select.get("minCount") == 1
        and select.get("maxCount") == 1
        and isinstance(selected, list)
        and len(selected) == 1
    )


def compile_record(record):
    """Compile one single-choice DecisionRecord into one IL example."""
    select = _select(record)
    options = select.get("option") or []
    selected = record.get("selectedIndices") or []
    if not options:
        raise ValueError("record has no legal options")
    if not selected:
        raise ValueError("record has no selected index")
    chosen_index = selected[0]
    if not isinstance(chosen_index, int) or isinstance(chosen_index, bool):
        raise ValueError("selected index is not an int")
    if chosen_index < 0 or chosen_index >= len(options):
        raise ValueError("selected index %s is outside %d options" %
                         (chosen_index, len(options)))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "gameId": record.get("gameId"),
        "sequenceNumber": record.get("sequenceNumber"),
        "playerIndex": select.get("playerIndex", record.get("playerIndex")),
        "promptType": features.prompt_type(record),
        "optionTypes": [features.option_type(option) for option in options],
        "stateText": features.state_text(record),
        "optionTexts": [features.option_text(option) for option in options],
        "chosenIndex": chosen_index,
        "metadata": dict(record.get("metadata") or {}),
    }


def compile_records(records, single_choice_only=True):
    """Return ``(examples, stats)`` for an iterable of DecisionRecords."""
    examples = []
    stats = {
        "records": 0,
        "compiled": 0,
        "discarded_multi_select": 0,
        "discarded_no_options": 0,
        "discarded_invalid": 0,
    }
    for record in records:
        stats["records"] += 1
        if single_choice_only and not is_single_choice(record):
            stats["discarded_multi_select"] += 1
            continue
        if not (_select(record).get("option") or []):
            stats["discarded_no_options"] += 1
            continue
        try:
            examples.append(compile_record(record))
            stats["compiled"] += 1
        except ValueError:
            stats["discarded_invalid"] += 1
    return examples, stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default=None,
                        help="canonical source label override "
                             "(arena_human, engine_selfplay, engine_human, search)")
    parser.add_argument("--single-choice-only", dest="single_choice_only",
                        action="store_true", default=True,
                        help="keep only single-choice decisions (default)")
    parser.add_argument("--allow-multi", dest="single_choice_only",
                        action="store_false",
                        help="also compile multi-select decisions, taking the "
                             "first selected index as the label")
    args = parser.parse_args(argv)

    examples, stats = compile_records(
        iter_decision_records(args.input, source_hint=args.source),
        single_choice_only=args.single_choice_only,
    )
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")
    sys.stderr.write(
        "records=%d compiled=%d discarded_multi_select=%d "
        "discarded_no_options=%d discarded_invalid=%d\n" %
        (
            stats["records"],
            stats["compiled"],
            stats["discarded_multi_select"],
            stats["discarded_no_options"],
            stats["discarded_invalid"],
        )
    )
    return 0


def _select(record):
    if not isinstance(record, dict):
        return {}
    top = record.get("select")
    if isinstance(top, dict):
        return top
    observation = record.get("observation") or {}
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
