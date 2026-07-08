"""Dataset manifest builder for training/evaluation runs.

A manifest is a compact, reproducible summary of a DecisionRecord stream:
schema/source counts, prompt/action distributions, option-count statistics,
reward coverage, terminal coverage, and observed causal-factor ranges. It is
small enough to commit alongside generated datasets and rich enough to catch
schema drift before training starts.
"""

import json

from .causal import FACTOR_NAMES, causal_variables
from .io import iter_decision_records
from .records import validate_record

__all__ = [
    "build_manifest",
    "write_manifest",
]


def build_manifest(records, name=None):
    """Return a JSON-serializable manifest for an iterable of records."""
    summary = _empty_manifest(name)
    for record in records:
        _add_record(summary, record)
    _finalize(summary)
    return summary


def write_manifest(path, manifest):
    """Write ``manifest`` to ``path`` as deterministic pretty JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _empty_manifest(name):
    return {
        "name": name,
        "schema": "DecisionRecord-v1",
        "records": 0,
        "validRecords": 0,
        "invalidRecords": 0,
        "sources": {},
        "promptTypes": {},
        "optionTypes": {},
        "selectedCount": {},
        "optionCount": {
            "min": None,
            "max": None,
            "sum": 0,
            "mean": None,
        },
        "terminalRecords": 0,
        "rewardRecords": 0,
        "resultRecords": 0,
        "captureConfidence": {},
        "causalFactors": {
            name: {"observed": 0, "min": None, "max": None}
            for name in FACTOR_NAMES
        },
        "firstErrors": [],
    }


def _add_record(summary, record):
    summary["records"] += 1
    errors = validate_record(record)
    if errors:
        summary["invalidRecords"] += 1
        if len(summary["firstErrors"]) < 5:
            summary["firstErrors"].append({
                "record": summary["records"] - 1,
                "line": record.get("__source_line") if isinstance(record, dict) else None,
                "messages": errors,
            })
    else:
        summary["validRecords"] += 1

    if not isinstance(record, dict):
        return

    _inc(summary["sources"], record.get("source") or "UNKNOWN")
    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict):
        _inc(summary["captureConfidence"], metadata.get("captureConfidence") or "UNKNOWN")

    select = record.get("select") or (record.get("observation") or {}).get("select") or {}
    if isinstance(select, dict):
        _inc(summary["promptTypes"], select.get("type") or "UNKNOWN")
        options = select.get("option") or []
        if isinstance(options, list):
            _add_option_count(summary["optionCount"], len(options))
            for option in options:
                if isinstance(option, dict):
                    _inc(summary["optionTypes"], option.get("type") or "UNKNOWN")

    selected = record.get("selectedIndices")
    if isinstance(selected, list):
        _inc(summary["selectedCount"], str(len(selected)))

    if record.get("terminal") is True:
        summary["terminalRecords"] += 1
    if record.get("reward") is not None:
        summary["rewardRecords"] += 1
    if record.get("result") is not None:
        summary["resultRecords"] += 1

    factors = causal_variables(record)
    for name, value in factors.items():
        if isinstance(value, (int, float)):
            _add_factor(summary["causalFactors"][name], value)


def _finalize(summary):
    option_count = summary["optionCount"]
    if summary["records"]:
        option_count["mean"] = option_count["sum"] / float(summary["records"])


def _inc(bucket, key):
    key = str(key)
    bucket[key] = bucket.get(key, 0) + 1


def _add_option_count(stats, count):
    stats["min"] = count if stats["min"] is None else min(stats["min"], count)
    stats["max"] = count if stats["max"] is None else max(stats["max"], count)
    stats["sum"] += count


def _add_factor(stats, value):
    stats["observed"] += 1
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)


def _records_from_path(path, source_hint=None):
    return iter_decision_records(path, source_hint=source_hint)
