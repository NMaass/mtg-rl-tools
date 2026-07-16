"""Belief-augmented recurrent information-state model."""
from __future__ import annotations

import os
from dataclasses import asdict

from magic_cabt.models.information_state import RecurrentInformationStateModel
from magic_cabt.models.structured_jepa import StructuredJEPAConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

TORCH_AVAILABLE = torch is not None


if nn is not None:
    class BeliefInformationStateModel(RecurrentInformationStateModel):
        """Recurrent policy with calibrated binary hidden-holding probes."""

        def __init__(self, labels, config=None, memory_layers=1):
            labels = tuple(str(label).strip() for label in labels)
            if not labels or any(not label for label in labels):
                raise ValueError("belief labels must be non-empty")
            if len(set(labels)) != len(labels):
                raise ValueError("belief labels must be unique")
            super().__init__(config=config, memory_layers=memory_layers)
            self.belief_labels = labels
            self.belief = nn.Sequential(
                nn.LayerNorm(self.config.d_model),
                nn.Linear(self.config.d_model, len(labels)))

        def belief_logits(self, memory):
            return self.belief(memory)

        def save_checkpoint(self, path, extra=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({
                "kind": "magic-belief-information-state-v1",
                "config": asdict(self.config),
                "memoryLayers": self.memory_layers,
                "beliefLabels": list(self.belief_labels),
                "stateDict": self.state_dict(),
                "extra": extra or {},
            }, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=False)
            if payload.get("kind") != "magic-belief-information-state-v1":
                raise ValueError("not a belief information-state checkpoint")
            model = cls(
                payload["beliefLabels"],
                StructuredJEPAConfig(**payload["config"]),
                memory_layers=payload.get("memoryLayers", 1))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class BeliefInformationStateModel(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("belief model requires magic-cabt[jepa]")
