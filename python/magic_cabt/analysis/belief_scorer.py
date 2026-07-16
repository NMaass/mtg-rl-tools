"""Stateful analysis scorer for belief-augmented information-state models."""
from __future__ import annotations

import os

from magic_cabt.models.belief import BeliefInformationStateModel
from magic_cabt.models.structured_jepa import CardTextResolver
from magic_cabt.models.visibility import VisibilitySafeTensorizer
from .scorer import RecurrentInformationStateScorer, _sha256_file


class BeliefInformationStateScorer(RecurrentInformationStateScorer):
    """Replay a belief checkpoint using only perspective-safe observations."""

    def __init__(self, checkpoint, device=None, card_cache=None,
                 arena_card_db=None):
        import torch
        self.checkpoint = os.path.abspath(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.card_cache = card_cache
        self.arena_card_db = arena_card_db
        resolver = CardTextResolver.from_path(
            card_cache, arena_db_path=arena_card_db)
        self.model, self.extra = BeliefInformationStateModel.load_checkpoint(
            checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.tensorizer = VisibilitySafeTensorizer(
            self.model.config, card_resolver=resolver)
        self.name = "belief-information-state"
        self._digest = _sha256_file(self.checkpoint)
        self._mtime = os.path.getmtime(self.checkpoint)
        self.reset()

    @property
    def model_info(self):
        metrics = self.extra.get("metrics") \
            if isinstance(self.extra, dict) else {}
        return {
            "modelId": self.name,
            "checkpointId": "sha256:" + self._digest,
            "checkpointPath": self.checkpoint,
            "embeddingBackend": self.model.config.embedding_backend,
            "trainingState": "trained" if metrics.get("decisionExamples")
                else "untrained",
            "decisionExamples": metrics.get("decisionExamples"),
            "sequenceLength": metrics.get("sequenceLength"),
            "beliefLabels": list(self.model.belief_labels),
            "visibilityPolicy": metrics.get(
                "visibilityPolicy",
                "public-history-and-perspective-state-v1"),
        }

    def belief_probabilities(self, record):
        """Return post-observation probabilities after ``score(record)``."""
        from .scorer import _record_position
        if self._last_position != _record_position(record):
            raise RuntimeError(
                "belief_probabilities must follow score for the same record")
        import torch
        with torch.no_grad():
            values = torch.sigmoid(
                self.model.belief_logits(self._last_memory))[0].cpu()
        return dict(zip(self.model.belief_labels,
                        [float(value) for value in values]))
