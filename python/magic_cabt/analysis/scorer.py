"""Checkpoint-backed legal-option scorers for live and replay analysis."""
from __future__ import annotations

import hashlib
import json
import os

from magic_cabt.models.structured_jepa import (
    CardTextResolver, MagicJEPA, StructuredTensorizer, TORCH_AVAILABLE)
from magic_cabt.models.visibility import VisibilitySafeTensorizer


class StructuredJEPAScorer:
    def __init__(self, checkpoint, device=None, card_cache=None,
                 arena_card_db=None):
        if not TORCH_AVAILABLE:
            raise ImportError("JEPA scoring requires: pip install -e 'python[jepa]'")
        import torch
        self.checkpoint = os.path.abspath(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.card_cache = card_cache
        self.arena_card_db = arena_card_db
        self._load()

    def _load(self):
        resolver = CardTextResolver.from_path(
            self.card_cache, arena_db_path=self.arena_card_db)
        self.model, self.extra = MagicJEPA.load_checkpoint(
            self.checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.tensorizer = StructuredTensorizer(
            self.model.config, card_resolver=resolver)
        self._mtime = os.path.getmtime(self.checkpoint)
        self._card_mtime = os.path.getmtime(self.card_cache) \
            if self.card_cache and os.path.exists(self.card_cache) else None
        self._digest = _sha256_file(self.checkpoint)
        self.name = "structured-jepa"

    @property
    def model_info(self):
        metrics = self.extra.get("metrics") \
            if isinstance(self.extra, dict) else None
        metrics = metrics or {}
        trained = metrics.get("transitionExamples") or \
            metrics.get("decisionExamples")
        return {
            "modelId": self.name,
            "checkpointId": "sha256:" + self._digest,
            "checkpointPath": self.checkpoint,
            "embeddingBackend": self.model.config.embedding_backend,
            "trainingState": "trained" if trained else "untrained",
            "transitionExamples": metrics.get("transitionExamples"),
            "decisionExamples": metrics.get("decisionExamples"),
        }

    def score(self, record):
        import torch
        select = _select(record)
        options = select.get("option") or []
        if not options:
            return []
        normalized = dict(record)
        normalized["select"] = select
        with torch.no_grad():
            rows, row_mask = self.tensorizer.batch_states(
                [record], self.device)
            option_vectors, option_mask = self.tensorizer.batch_options(
                [normalized], self.device)
            logits = self.model.score_options(
                rows, row_mask, option_vectors, option_mask)[0]
        return [float(value) for value in logits[:len(options)].cpu()]

    def state_value(self, record):
        import torch
        with torch.no_grad():
            rows, mask = self.tensorizer.batch_states([record], self.device)
            return float(self.model.state_value(rows, mask)[0].cpu())

    def reload_if_changed(self):
        mtime = os.path.getmtime(self.checkpoint)
        card_mtime = os.path.getmtime(self.card_cache) \
            if self.card_cache and os.path.exists(self.card_cache) else None
        if mtime <= self._mtime and card_mtime == self._card_mtime:
            return False
        self._load()
        return True


class RecurrentInformationStateScorer:
    """Stateful scorer that replays public decision history in game order."""

    def __init__(self, checkpoint, device=None, card_cache=None,
                 arena_card_db=None):
        if not TORCH_AVAILABLE:
            raise ImportError("information-state scoring requires PyTorch")
        import torch
        from magic_cabt.models.information_state import \
            RecurrentInformationStateModel
        self.checkpoint = os.path.abspath(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.card_cache = card_cache
        self.arena_card_db = arena_card_db
        resolver = CardTextResolver.from_path(
            card_cache, arena_db_path=arena_card_db)
        self.model, self.extra = RecurrentInformationStateModel.load_checkpoint(
            checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.tensorizer = VisibilitySafeTensorizer(
            self.model.config, card_resolver=resolver)
        self.name = "recurrent-information-state"
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
            "trainingState": "trained" if metrics.get("decisionExamples") else "untrained",
            "decisionExamples": metrics.get("decisionExamples"),
            "sequenceLength": metrics.get("sequenceLength"),
            "visibilityPolicy": metrics.get(
                "visibilityPolicy",
                "public-history-and-perspective-state-v1"),
        }

    def reset(self):
        self.hidden = None
        self.previous_action = None
        self.current_game = None
        self._last_memory = None
        self._last_position = None

    def _advance(self, record, return_scores):
        import torch
        game = _game_key(record)
        if game != self.current_game:
            self.reset()
            self.current_game = game
        select = _select(record)
        normalized = dict(record)
        normalized["select"] = select
        with torch.no_grad():
            rows, row_mask = self.tensorizer.batch_states([record], self.device)
            previous = self.tensorizer.batch_actions(
                [self.previous_action], self.device)
            memory, self.hidden = self.model.step(
                rows, row_mask, previous, self.hidden)
            self._last_memory = memory
            self._last_position = _record_position(record)
            scores = None
            if return_scores:
                option_vectors, option_mask = self.tensorizer.batch_options(
                    [normalized], self.device)
                logits = self.model.score_from_memory(
                    memory, option_vectors, option_mask)[0]
                scores = [float(value) for value in
                          logits[:len(select.get("option") or [])].cpu()]
        self.previous_action = _selected_action(record)
        return scores

    def score(self, record):
        return self._advance(record, True) or []

    def observe(self, record):
        """Advance memory for a cached decision without recomputing output."""
        self._advance(record, False)

    def state_value(self, record):
        if self._last_position != _record_position(record):
            raise RuntimeError("state_value must follow score for the same record")
        import torch
        with torch.no_grad():
            return float(self.model.state_value_from_memory(
                self._last_memory)[0].cpu())

    def reload_if_changed(self):
        if os.path.getmtime(self.checkpoint) <= self._mtime:
            return False
        self.__init__(self.checkpoint, self.device, self.card_cache,
                      self.arena_card_db)
        return True


class RankerScorer:
    """Adapter for the smaller hashed-feature OptionRanker in PR #28."""

    def __init__(self, checkpoint, device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("ranker scoring requires PyTorch")
        import torch
        from magic_cabt.models.torch_ranker import OptionRanker, hash_features
        self._hash_features = hash_features
        self.checkpoint = os.path.abspath(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OptionRanker.load_checkpoint(
            checkpoint, map_location=self.device)
        self.model.to(self.device).eval()
        self.name = "torch-option-ranker"
        self._digest = _sha256_file(self.checkpoint)
        self._mtime = os.path.getmtime(self.checkpoint)

    @property
    def model_info(self):
        return {
            "modelId": self.name,
            "checkpointId": "sha256:" + self._digest,
            "checkpointPath": self.checkpoint,
        }

    def score(self, record):
        import torch
        from magic_cabt.training.features import option_text, prompt_type, state_text
        select = _select(record)
        options = select.get("option") or []
        if not options:
            return []
        normalized = dict(record)
        normalized["select"] = select
        config = self.model.config
        state = self._hash_features(
            "%s | %s" % (prompt_type(normalized), state_text(normalized)),
            int(config["stateFeatureDim"]))
        option_vectors = [self._hash_features(
            option_text(option), int(config["optionFeatureDim"]))
            for option in options]
        with torch.no_grad():
            logits = self.model(
                torch.tensor([state], dtype=torch.float32, device=self.device),
                torch.tensor([option_vectors], dtype=torch.float32,
                             device=self.device),
                torch.ones((1, len(options)), dtype=torch.bool,
                           device=self.device))[0]
        return [float(value) for value in logits.cpu()]

    def reload_if_changed(self):
        if os.path.getmtime(self.checkpoint) <= self._mtime:
            return False
        self.__init__(self.checkpoint, self.device)
        return True


def load_checkpoint_scorer(checkpoint, device=None, card_cache=None,
                           arena_card_db=None):
    """Auto-detect all supported learned checkpoint families."""
    if not TORCH_AVAILABLE:
        raise ImportError("checkpoint scoring requires PyTorch")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    kind = payload.get("kind")
    if kind == "magic-structured-jepa-v1":
        return StructuredJEPAScorer(
            checkpoint, device=device, card_cache=card_cache,
            arena_card_db=arena_card_db)
    if kind == "magic-recurrent-information-state-v1":
        return RecurrentInformationStateScorer(
            checkpoint, device=device, card_cache=card_cache,
            arena_card_db=arena_card_db)
    return RankerScorer(checkpoint, device=device)


def _select(record):
    direct = record.get("select")
    if isinstance(direct, dict):
        return direct
    return (record.get("observation") or {}).get("select") or {}


def _selected_action(record):
    select = _select(record)
    options = select.get("option") or []
    selected = record.get("selectedIndices") or record.get("selected") or []
    if len(selected) != 1 or not isinstance(selected[0], int):
        return None
    index = selected[0]
    if not 0 <= index < len(options):
        return None
    return {"promptType": select.get("type"), "selectedOption": options[index]}


def _game_key(record):
    current = (record.get("observation") or {}).get("current") or \
        record.get("current") or {}
    return json.dumps([
        record.get("matchId") or record.get("gameId") or current.get("matchId"),
        record.get("gameNumber", current.get("gameNumber")),
        record.get("gameInstance") or current.get("gameInstance"),
    ], separators=(",", ":"))


def _record_position(record):
    current = (record.get("observation") or {}).get("current") or \
        record.get("current") or {}
    return (_game_key(record), record.get("sequenceNumber",
            record.get("sequence", current.get("seq"))))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
