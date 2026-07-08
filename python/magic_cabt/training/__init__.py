"""Canonical DecisionRecord normalization + dataset validation (issue #10).

This package turns heterogeneous gameplay captures into one training format:

- arena-mirror recorder output (``arena_human``),
- ``examples/run_selfplay.py`` replay frames (``engine_selfplay``),
- the Java ``CabtDatasetWriter`` JSONL transitions (``engine_selfplay``),
- future search-rollout logs (``search``).

See ``docs/decision-record-schema.md`` for the field-level invariant. The
public surface here is intentionally small so consumers (CLI, downstream
dataset loaders, tests) reach for these helpers instead of reimplementing
shape detection.

The IL compiler (issue #11) and baseline policies live in sibling modules
(``compile_il``, ``features``, ``eval_bc``); they are imported by path
(``from magic_cabt.training.compile_il import ...``), never re-exported
here, so this package's surface stays narrow.

Normalizers are pure functions over plain ``dict`` records — no schema
library is required, mirroring the rest of the ``magic_cabt`` package.
"""

from .records import (
    SCHEMA_VERSION,
    CANONICAL_SOURCES,
    DEFAULT_SOURCE,
    normalize_record,
    validate_record,
    validate_records,
)
from .io import iter_decision_records

__all__ = [
    "SCHEMA_VERSION",
    "CANONICAL_SOURCES",
    "DEFAULT_SOURCE",
    "normalize_record",
    "iter_decision_records",
    "validate_record",
    "validate_records",
]
