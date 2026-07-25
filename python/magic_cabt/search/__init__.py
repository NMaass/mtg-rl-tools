"""Counterfactual search backends for CABT games."""

from .replay_search import (
    ReplayDivergenceError,
    branch_replay,
    branch_to_transition,
    candidate_selections,
    observation_signature,
    replay_to_root,
)

__all__ = [
    "ReplayDivergenceError",
    "branch_replay",
    "branch_to_transition",
    "candidate_selections",
    "observation_signature",
    "replay_to_root",
]
