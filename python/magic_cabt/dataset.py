"""Reader for the CABT bridge's JSONL transition dataset (Task 22).

Each line is one generic transition written by the Java CabtDatasetWriter:
``schemaVersion``, ``gameId``, ``sequenceNumber``, ``decisionMethod``,
``observation``, ``select``, ``selectedIndices``, ``nextObservation``,
``terminal``, ``reward``, ``metadata``. Records store decisions and resulting
states, never outcome labels, so the same file serves imitation learning, RL,
and search analysis.
"""

import json

__all__ = ["read_dataset"]

SCHEMA_VERSION = 1


def read_dataset(source):
    """Yield record dicts from a JSONL dataset.

    ``source`` is a file path or an open text-file object. Blank lines are
    skipped; a line that is not valid JSON raises ``ValueError`` naming the
    line number rather than being silently dropped.
    """
    if hasattr(source, "read"):
        for record in _read_lines(source):
            yield record
    else:
        with open(source, "r", encoding="utf-8") as handle:
            for record in _read_lines(handle):
                yield record


def _read_lines(handle):
    for line_number, line in enumerate(handle, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError:
            raise ValueError("dataset line %d is not valid JSON" % line_number)
        if not isinstance(record, dict):
            raise ValueError("dataset line %d is not a JSON object" % line_number)
        yield record
