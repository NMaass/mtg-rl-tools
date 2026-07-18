"""Canonical symbolic state contract shared by video, logs, XMage, and models."""

from .schema import (
    CanonicalEvent,
    CanonicalObject,
    CanonicalPlayer,
    CanonicalState,
)
from .formatter import CanonicalStateFormatter, canonical_to_model_observation
from .compare import ComparisonPolicy, ComparisonReport, DiffItem, compare_states
from .adapter import MagicCabtTensorizerAdapter

__all__ = [
    "CanonicalEvent",
    "CanonicalObject",
    "CanonicalPlayer",
    "CanonicalState",
    "CanonicalStateFormatter",
    "canonical_to_model_observation",
    "ComparisonPolicy",
    "ComparisonReport",
    "DiffItem",
    "compare_states",
    "MagicCabtTensorizerAdapter",
]
