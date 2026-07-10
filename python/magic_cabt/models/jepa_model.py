"""Small action-conditioned JEPA and legal-option policy."""
from __future__ import annotations

import copy
import os
from dataclasses import asdict

from .structured_config import StructuredJEPAConfig

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

TORCH_AVAILABLE = torch is not None


if nn is not None:
    class ResidualMLP(nn.Module):
        def __init__(self, width, layers, dropout):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(width * 4, width))
                for _ in range(layers)])

        def forward(self, value):
            for block in self.blocks:
                value = value + block(value)
            return value


    class StateEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(config.text_dim + config.numeric_dim, config.d_model),
                nn.LayerNorm(config.d_model))
            self.token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            layer = nn.TransformerEncoderLayer(
                config.d_model, config.nhead, config.ff_dim, config.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(
                layer, config.encoder_layers, norm=nn.LayerNorm(config.d_model))
            nn.init.normal_(self.token, std=0.02)

        def forward(self, rows, mask):
            batch = rows.shape[0]
            value = torch.cat([self.token.expand(batch, -1, -1),
                               self.input(rows)], dim=1)
            full_mask = torch.cat([torch.ones((batch, 1), dtype=torch.bool,
                                                device=mask.device), mask], dim=1)
            return self.encoder(value, src_key_padding_mask=~full_mask)[:, 0]


    class MagicJEPA(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config or StructuredJEPAConfig()
            c = self.config
            self.online_encoder = StateEncoder(c)
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            row_dim = c.text_dim + c.numeric_dim
            self.action_encoder = nn.Sequential(
                nn.Linear(row_dim, c.d_model), nn.LayerNorm(c.d_model),
                nn.GELU(), nn.Linear(c.d_model, c.d_model))
            self.horizon_embedding = nn.Embedding(c.horizon_buckets, c.d_model)
            self.predictor_input = nn.Linear(c.d_model * 3, c.d_model)
            self.predictor = ResidualMLP(c.d_model, c.predictor_layers, c.dropout)
            self.predictor_norm = nn.LayerNorm(c.d_model)
            self.predictor_log_scale = nn.Sequential(
                nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model), nn.Tanh())
            self.policy = nn.Sequential(
                nn.LayerNorm(c.d_model * 4), nn.Linear(c.d_model * 4, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, 1))
            self.value = nn.Sequential(
                nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, 1), nn.Tanh())
            self.causal = nn.Sequential(
                nn.LayerNorm(c.d_model * 3), nn.Linear(c.d_model * 3, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, c.causal_dim))

        def encode(self, rows, mask):
            return self.online_encoder(rows, mask)

        @torch.no_grad()
        def encode_target(self, rows, mask):
            return self.target_encoder(rows, mask)

        def _horizon(self, state, horizon):
            if horizon is None:
                horizon = torch.ones(state.shape[0], dtype=torch.long,
                                     device=state.device)
            return self.horizon_embedding(
                horizon.long().clamp(0, self.config.horizon_buckets - 1))

        def predict_distribution(self, state, action_vector, horizon=None):
            action = self.action_encoder(action_vector)
            hidden = self.predictor_input(torch.cat(
                [state, action, self._horizon(state, horizon)], dim=-1))
            mean = self.predictor_norm(self.predictor(hidden))
            return mean, self.predictor_log_scale(mean).clamp(-4.0, 1.5)

        def predict(self, state, action_vector, horizon=None, sample=False):
            mean, log_scale = self.predict_distribution(state, action_vector, horizon)
            return mean + torch.randn_like(mean) * log_scale.exp() if sample else mean

        def score_options(self, rows, state_mask, option_vectors, option_mask):
            state = self.encode(rows, state_mask)
            batch, count, _ = option_vectors.shape
            action = self.action_encoder(option_vectors)
            states = state[:, None, :].expand(-1, count, -1)
            horizon = self.horizon_embedding(torch.ones(
                (batch, count), dtype=torch.long, device=rows.device))
            predicted = self.predictor_norm(self.predictor(self.predictor_input(
                torch.cat([states, action, horizon], dim=-1))))
            logits = self.policy(torch.cat(
                [states, action, predicted, predicted - states], dim=-1)).squeeze(-1)
            return logits.masked_fill(~option_mask, float("-inf"))

        def state_value(self, rows, mask):
            return self.value(self.encode(rows, mask)).squeeze(-1)

        def causal_delta(self, state, action_vector, horizon=None):
            return self.causal(torch.cat([
                state, self.action_encoder(action_vector),
                self._horizon(state, horizon)], dim=-1))

        def jepa_loss(self, predicted, target, log_scale=None,
                      variance_weight=0.05, uncertainty_weight=0.05):
            target = target.detach()
            alignment = 2.0 - 2.0 * (
                F.normalize(predicted, dim=-1) * F.normalize(target, dim=-1)
            ).sum(dim=-1).mean()
            variance = predicted.new_tensor(0.0)
            if predicted.shape[0] > 1:
                variance = F.relu(1.0 - torch.sqrt(
                    predicted.var(dim=0, unbiased=False) + 1e-4)).mean()
            uncertainty = predicted.new_tensor(0.0)
            if log_scale is not None:
                uncertainty = (0.5 * (target - predicted).pow(2) *
                               torch.exp(-2.0 * log_scale) + log_scale).mean()
            return (alignment + variance_weight * variance +
                    uncertainty_weight * uncertainty), {
                        "alignment": alignment.detach(),
                        "variance": variance.detach(),
                        "uncertainty": uncertainty.detach()}

        @torch.no_grad()
        def update_target(self, tau=0.996):
            for target, online in zip(self.target_encoder.parameters(),
                                      self.online_encoder.parameters()):
                target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)

        def save_checkpoint(self, path, extra=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({"kind": "magic-structured-jepa-v1",
                        "config": asdict(self.config),
                        "stateDict": self.state_dict(),
                        "extra": extra or {}}, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=False)
            if payload.get("kind") != "magic-structured-jepa-v1":
                raise ValueError("not a structured JEPA checkpoint")
            model = cls(StructuredJEPAConfig(**payload["config"]))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class StateEncoder(object):
        def __init__(self, *args, **kwargs):
            _torch_required()

    class MagicJEPA(object):
        def __init__(self, *args, **kwargs):
            _torch_required()


def model_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _torch_required():
    if torch is None:
        raise ImportError("structured JEPA requires magic-cabt[jepa]")
