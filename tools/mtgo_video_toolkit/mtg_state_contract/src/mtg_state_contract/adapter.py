"""Adapter from the canonical state to the existing mtg-rl-tools tensorizer."""
from __future__ import annotations

from typing import Any, Optional

from .formatter import canonical_to_model_observation
from .schema import CanonicalState


class MagicCabtTensorizerAdapter:
    """Feed canonical states to any ``magic_cabt`` structured tensorizer.

    The adapter intentionally imports no repository code.  Pass an already
    constructed ``StructuredTensorizer`` or ``VisibilitySafeTensorizer``.
    """

    def __init__(self, tensorizer: Any):
        self.tensorizer = tensorizer

    def observation(self, state: CanonicalState,
                    perspective_seat: Optional[Any] = None) -> dict:
        return canonical_to_model_observation(state, perspective_seat)

    def state_rows(self, state: CanonicalState,
                   perspective_seat: Optional[Any] = None):
        return self.tensorizer.state_rows(
            self.observation(state, perspective_seat),
            perspective_seat=perspective_seat)

    def batch_states(self, states, device,
                     perspective_seat: Optional[Any] = None):
        values = [self.observation(state, perspective_seat) for state in states]
        return self.tensorizer.batch_states(values, device)
