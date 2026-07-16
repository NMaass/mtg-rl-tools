"""Recurrent information-state policy over structured Magic decisions.

The model encodes each perspective-safe decision state, combines it with the
previous observed action, and updates a recurrent memory. It never consumes
opponent-private cards directly; hidden information can only affect the memory
through observations and legally visible history present in the record.
"""
from __future__ import annotations

import os
from dataclasses import asdict

from magic_cabt.models.structured_jepa import StateEncoder, StructuredJEPAConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

TORCH_AVAILABLE = torch is not None


if nn is not None:
    class RecurrentInformationStateModel(nn.Module):
        """Structured state encoder plus GRU information-state memory."""

        def __init__(self, config=None, memory_layers=1):
            super().__init__()
            self.config = config or StructuredJEPAConfig()
            self.memory_layers = int(memory_layers)
            if self.memory_layers < 1:
                raise ValueError("memory_layers must be positive")
            c = self.config
            row_dim = c.text_dim + c.numeric_dim
            self.state_encoder = StateEncoder(c)
            self.action_encoder = nn.Sequential(
                nn.Linear(row_dim, c.d_model), nn.LayerNorm(c.d_model),
                nn.GELU(), nn.Linear(c.d_model, c.d_model))
            self.memory = nn.GRU(
                c.d_model * 2, c.d_model, num_layers=self.memory_layers,
                dropout=c.dropout if self.memory_layers > 1 else 0.0,
                batch_first=True)
            self.policy = nn.Sequential(
                nn.LayerNorm(c.d_model * 3),
                nn.Linear(c.d_model * 3, c.d_model), nn.GELU(),
                nn.Linear(c.d_model, 1))
            self.value = nn.Sequential(
                nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, 1), nn.Tanh())

        def encode_states(self, rows, row_mask):
            """Encode ``[batch, time, objects, features]`` state tensors."""
            batch, time, objects, features = rows.shape
            flat_rows = rows.reshape(batch * time, objects, features)
            flat_mask = row_mask.reshape(batch * time, objects)
            encoded = self.state_encoder(flat_rows, flat_mask)
            return encoded.reshape(batch, time, -1)

        def information_states(self, rows, row_mask, previous_actions,
                               hidden=None, sequence_mask=None):
            states = self.encode_states(rows, row_mask)
            actions = self.action_encoder(previous_actions)
            values, hidden = self.memory(torch.cat([states, actions], dim=-1),
                                         hidden)
            if sequence_mask is not None:
                values = values * sequence_mask.unsqueeze(-1).to(values.dtype)
            return values, hidden

        def step(self, rows, row_mask, previous_action, hidden=None):
            state = self.state_encoder(rows, row_mask)
            action = self.action_encoder(previous_action)
            value, hidden = self.memory(
                torch.cat([state, action], dim=-1).unsqueeze(1), hidden)
            return value[:, 0], hidden

        def score_from_memory(self, memory, option_vectors, option_mask):
            action = self.action_encoder(option_vectors)
            expanded = memory.unsqueeze(-2).expand(
                *option_vectors.shape[:-1], memory.shape[-1])
            logits = self.policy(torch.cat(
                [expanded, action, action - expanded], dim=-1)).squeeze(-1)
            return logits.masked_fill(~option_mask, float("-inf"))

        def state_value_from_memory(self, memory):
            return self.value(memory).squeeze(-1)

        def save_checkpoint(self, path, extra=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({
                "kind": "magic-recurrent-information-state-v1",
                "config": asdict(self.config),
                "memoryLayers": self.memory_layers,
                "stateDict": self.state_dict(),
                "extra": extra or {},
            }, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=False)
            if payload.get("kind") != "magic-recurrent-information-state-v1":
                raise ValueError("not a recurrent information-state checkpoint")
            model = cls(StructuredJEPAConfig(**payload["config"]),
                        memory_layers=payload.get("memoryLayers", 1))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class RecurrentInformationStateModel(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("recurrent information state requires magic-cabt[jepa]")
