"""Card-selection model for limited play (draft picks, deck builds, sideboards).

All three tasks share one shape — score candidate cards against a context set
of cards — so one model serves them, distinguished by a mode feature:

- ``draftPick``: context = pool drafted so far, candidates = current pack.
- ``deckBuild``: context = deck built so far, candidates = remaining pool.
- ``sideboard``: context = the deck as last submitted, candidates = every
  card available (deck + sideboard), positives = the deck actually kept.

Cards are represented by their text (via the same embedding providers the
structured JEPA uses) plus a small numeric block, so unseen cards transfer
through their rules text rather than a learned id table. The context set is
encoded with the JEPA's permutation-symmetric ``StateEncoder``.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from .embeddings import make_embedding_provider
from .jepa_model import StateEncoder
from .state_utils import norm
from .structured_config import CardTextResolver

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

TORCH_AVAILABLE = torch is not None

MODES = ("draftPick", "deckBuild", "sideboard")

CHECKPOINT_KIND = "magic-cabt-card-selection-v1"

_COLOR_LETTERS = "WUBRG"
_MANA_SYMBOL_RE = re.compile(r"(\d+)|([WUBRGCXS])", re.IGNORECASE)


@dataclass
class DraftModelConfig:
    text_dim: int = 384
    numeric_dim: int = 24
    d_model: int = 192
    nhead: int = 6
    encoder_layers: int = 4
    ff_dim: int = 768
    dropout: float = 0.1
    max_context: int = 80
    embedding_backend: str = "hash"

    @classmethod
    def preset(cls, name):
        name = (name or "local").lower()
        if name == "tiny":
            return cls(d_model=128, nhead=4, encoder_layers=3, ff_dim=512)
        if name == "local":
            return cls()
        raise ValueError("unknown draft model preset: %s" % name)


class DraftCardResolver:
    """grpId -> display/text metadata for tensorization.

    Wraps the mirror's ``CardDatabase`` (name, types, colors, P/T) and the
    JEPA's ``CardTextResolver`` (mana cost, oracle text). Either source may be
    absent; a card that resolves nowhere still gets a distinct text token from
    its grpId so identities never collapse.
    """

    def __init__(self, card_database=None, text_resolver: Optional[CardTextResolver] = None):
        self.database = card_database
        self.text_resolver = text_resolver
        self._cache = {}

    def describe(self, grp_id):
        if grp_id in self._cache:
            return self._cache[grp_id]
        merged = {"grpId": grp_id}
        info = None
        if self.database is not None:
            try:
                info = self.database.lookup(grp_id)
            except KeyError:
                info = None
        if info is not None:
            merged.update({
                "name": info.name,
                "types": list(info.types),
                "subtypes": list(info.subtypes),
                "power": info.power,
                "toughness": info.toughness,
                "colors": info.colors or info.color_identity,
            })
        if self.text_resolver is not None:
            extra = self.text_resolver.resolve({"grpId": grp_id}) or {}
            for key in ("manaCost", "oracleText", "name", "types"):
                if extra.get(key) is not None and merged.get(key) is None:
                    merged[key] = extra[key]
        self._cache[grp_id] = merged
        return merged

    def card_text(self, grp_id):
        card = self.describe(grp_id)
        parts = ["card | grpId=%s" % grp_id]
        for key in ("name", "manaCost", "types", "subtypes", "power",
                    "toughness", "oracleText"):
            value = card.get(key)
            if value not in (None, "", []):
                if isinstance(value, list):
                    value = " ".join(str(item) for item in value)
                parts.append("%s=%s" % (key, value))
        return " | ".join(parts)


def mana_value(mana_cost):
    """Total mana value of an Arena mana string such as ``3UU`` or ``1G``."""
    if not mana_cost:
        return None
    total = 0
    for digits, symbol in _MANA_SYMBOL_RE.findall(str(mana_cost)):
        if digits:
            total += int(digits)
        elif symbol.upper() not in ("X", "S"):
            total += 1
    return total


class DraftTensorizer:
    """Turn selection examples into padded row tensors."""

    def __init__(self, config: DraftModelConfig, resolver: DraftCardResolver,
                 embedding_provider=None):
        self.config = config
        self.resolver = resolver
        self.embedding = embedding_provider or make_embedding_provider(
            config.embedding_backend, config.text_dim)
        if self.embedding.dimension != config.text_dim:
            raise ValueError("embedding dimension does not match model config")

    @property
    def row_dim(self):
        return self.config.text_dim + self.config.numeric_dim

    def example_rows(self, example: Mapping):
        """Return ``(context_rows, candidate_rows)`` for one example."""
        mode = example.get("mode")
        if mode not in MODES:
            raise ValueError("unknown selection mode: %r" % (mode,))
        context_ids = list(example.get("contextIds") or [])
        candidate_ids = list(example.get("candidateIds") or [])
        if not candidate_ids:
            raise ValueError("selection example has no candidates")
        shared = {
            "mode": mode,
            "packNumber": example.get("packNumber"),
            "pickNumber": example.get("pickNumber"),
            "contextSize": len(context_ids),
            "candidateCount": len(candidate_ids),
        }
        status_text = "limited status | " + " ".join(
            "%s=%s" % (key, value) for key, value in sorted(shared.items())
            if value is not None)
        copies = {}
        for grp_id in context_ids:
            copies[grp_id] = copies.get(grp_id, 0) + 1
        texts = [status_text]
        texts.extend(self.resolver.card_text(grp_id)
                     for grp_id in context_ids[:self.config.max_context])
        texts.extend(self.resolver.card_text(grp_id)
                     for grp_id in candidate_ids)
        vectors = self.embedding.encode_many(texts)
        context_rows = [
            list(vectors[0]) + self._numeric(None, "status", shared, copies)]
        offset = 1
        for index, grp_id in enumerate(
                context_ids[:self.config.max_context]):
            context_rows.append(list(vectors[offset + index]) +
                                self._numeric(grp_id, "context", shared, copies))
        offset += len(context_ids[:self.config.max_context])
        candidate_rows = [
            list(vectors[offset + index]) +
            self._numeric(grp_id, "candidate", shared, copies)
            for index, grp_id in enumerate(candidate_ids)]
        return context_rows, candidate_rows

    def _numeric(self, grp_id, role, shared, copies):
        values = [0.0] * self.config.numeric_dim
        values[0] = 1.0
        values[1] = 1.0 if role == "status" else 0.0
        values[2] = 1.0 if role == "context" else 0.0
        values[3] = 1.0 if role == "candidate" else 0.0
        values[4 + MODES.index(shared["mode"])] = 1.0
        values[7] = norm(shared.get("packNumber"), 3.0)
        values[8] = norm(shared.get("pickNumber"), 15.0)
        values[9] = norm(shared.get("contextSize"), 45.0)
        values[10] = norm(shared.get("candidateCount"), 15.0)
        if grp_id is not None:
            values[11] = norm(copies.get(grp_id), 4.0)
            card = self.resolver.describe(grp_id)
            colors = str(card.get("colors") or "").upper()
            for index, letter in enumerate(_COLOR_LETTERS):
                values[12 + index] = 1.0 if letter in colors else 0.0
            types = [str(item).lower() for item in card.get("types") or []]
            values[17] = 1.0 if "creature" in types else 0.0
            values[18] = 1.0 if "land" in types else 0.0
            values[19] = norm(mana_value(card.get("manaCost")), 10.0)
            values[20] = norm(_number(card.get("power")), 10.0)
            values[21] = norm(_number(card.get("toughness")), 10.0)
            values[22] = 1.0 if len([c for c in _COLOR_LETTERS
                                     if c in colors]) > 1 else 0.0
        return values

    def batch_examples(self, examples: Sequence[Mapping], device=None):
        """Pad a batch to tensors: context rows/mask, candidate rows/mask."""
        _torch_required()
        pairs = [self.example_rows(example) for example in examples]
        max_context = max(len(context) for context, _ in pairs)
        max_candidates = max(len(candidates) for _, candidates in pairs)
        contexts, context_masks, candidates, candidate_masks = [], [], [], []
        for context, options in pairs:
            context_masks.append([True] * len(context) +
                                 [False] * (max_context - len(context)))
            contexts.append(context + [[0.0] * self.row_dim] *
                            (max_context - len(context)))
            candidate_masks.append([True] * len(options) +
                                   [False] * (max_candidates - len(options)))
            candidates.append(options + [[0.0] * self.row_dim] *
                              (max_candidates - len(options)))
        return (torch.tensor(contexts, dtype=torch.float32, device=device),
                torch.tensor(context_masks, dtype=torch.bool, device=device),
                torch.tensor(candidates, dtype=torch.float32, device=device),
                torch.tensor(candidate_masks, dtype=torch.bool, device=device))


if nn is not None:
    class CardSelectionModel(nn.Module):
        """Score candidate cards against an encoded context set."""

        def __init__(self, config=None):
            super().__init__()
            self.config = config or DraftModelConfig()
            c = self.config
            self.context_encoder = StateEncoder(c)
            row_dim = c.text_dim + c.numeric_dim
            self.candidate_encoder = nn.Sequential(
                nn.Linear(row_dim, c.d_model), nn.LayerNorm(c.d_model),
                nn.GELU(), nn.Linear(c.d_model, c.d_model))
            self.scorer = nn.Sequential(
                nn.LayerNorm(c.d_model * 3),
                nn.Linear(c.d_model * 3, c.d_model), nn.GELU(),
                nn.Linear(c.d_model, 1))

        def score_candidates(self, context_rows, context_mask,
                             candidate_rows, candidate_mask):
            context = self.context_encoder(context_rows, context_mask)
            candidates = self.candidate_encoder(candidate_rows)
            expanded = context[:, None, :].expand(
                -1, candidate_rows.shape[1], -1)
            logits = self.scorer(torch.cat(
                [expanded, candidates, expanded * candidates],
                dim=-1)).squeeze(-1)
            return logits.masked_fill(~candidate_mask, float("-inf"))

        def save_checkpoint(self, path, extra=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({"kind": CHECKPOINT_KIND,
                        "config": asdict(self.config),
                        "stateDict": self.state_dict(),
                        "extra": extra or {}}, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=False)
            if payload.get("kind") != CHECKPOINT_KIND:
                raise ValueError("not a card-selection checkpoint")
            model = cls(DraftModelConfig(**payload["config"]))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class CardSelectionModel(object):
        def __init__(self, *args, **kwargs):
            _torch_required()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _torch_required():
    if torch is None:
        raise ImportError("the draft model requires magic-cabt[torch]")
