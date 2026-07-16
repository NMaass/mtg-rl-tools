"""Stochastic recurrent state-space model for structured Magic trajectories."""
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
    class StructuredRSSM(nn.Module):
        """Dreamer-style deterministic memory plus diagonal Gaussian latent."""

        def __init__(self, config=None, latent_dim=None):
            super().__init__()
            self.config = config or StructuredJEPAConfig()
            self.latent_dim = int(latent_dim or self.config.d_model)
            if self.latent_dim < 2:
                raise ValueError("latent_dim must be at least 2")
            c = self.config
            row_dim = c.text_dim + c.numeric_dim
            self.observation_encoder = StateEncoder(c)
            self.action_encoder = nn.Sequential(
                nn.Linear(row_dim, c.d_model), nn.LayerNorm(c.d_model),
                nn.GELU(), nn.Linear(c.d_model, c.d_model))
            self.transition = nn.GRUCell(
                self.latent_dim + c.d_model, c.d_model)
            self.prior = nn.Sequential(
                nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, self.latent_dim * 2))
            self.posterior = nn.Sequential(
                nn.LayerNorm(c.d_model * 2),
                nn.Linear(c.d_model * 2, c.d_model), nn.GELU(),
                nn.Linear(c.d_model, self.latent_dim * 2))
            feature_dim = c.d_model + self.latent_dim
            self.decoder = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, c.d_model * 2), nn.GELU(),
                nn.Linear(c.d_model * 2, c.d_model))
            self.causal = nn.Sequential(
                nn.LayerNorm(feature_dim + c.d_model),
                nn.Linear(feature_dim + c.d_model, c.d_model), nn.GELU(),
                nn.Linear(c.d_model, c.causal_dim))
            self.value = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, c.d_model),
                nn.GELU(), nn.Linear(c.d_model, 1), nn.Tanh())

        def initial(self, batch_size, device=None):
            deterministic = torch.zeros(
                batch_size, self.config.d_model, device=device)
            stochastic = torch.zeros(batch_size, self.latent_dim, device=device)
            return deterministic, stochastic

        def encode_observation(self, rows, mask):
            return self.observation_encoder(rows, mask)

        @staticmethod
        def _stats(values):
            mean, raw_scale = values.chunk(2, dim=-1)
            return mean, raw_scale.clamp(-5.0, 2.0)

        @staticmethod
        def _sample(mean, log_scale, sample):
            if not sample:
                return mean
            return mean + torch.randn_like(mean) * log_scale.exp()

        def prior_step(self, action_vector, deterministic, stochastic,
                       sample=True):
            action = self.action_encoder(action_vector)
            deterministic = self.transition(
                torch.cat([stochastic, action], dim=-1), deterministic)
            mean, log_scale = self._stats(self.prior(deterministic))
            stochastic = self._sample(mean, log_scale, sample)
            feature = torch.cat([deterministic, stochastic], dim=-1)
            return {
                "deterministic": deterministic,
                "stochastic": stochastic,
                "mean": mean,
                "logScale": log_scale,
                "feature": feature,
            }

        def posterior_step(self, observation, action_vector, deterministic,
                           stochastic, sample=True):
            prior = self.prior_step(
                action_vector, deterministic, stochastic, sample=sample)
            posterior_input = torch.cat(
                [prior["deterministic"], observation], dim=-1)
            mean, log_scale = self._stats(self.posterior(posterior_input))
            stochastic = self._sample(mean, log_scale, sample)
            feature = torch.cat(
                [prior["deterministic"], stochastic], dim=-1)
            return prior, {
                "deterministic": prior["deterministic"],
                "stochastic": stochastic,
                "mean": mean,
                "logScale": log_scale,
                "feature": feature,
            }

        def decode_observation(self, feature):
            return self.decoder(feature)

        def causal_delta(self, feature, action_vector):
            action = self.action_encoder(action_vector)
            return self.causal(torch.cat([feature, action], dim=-1))

        def state_value(self, feature):
            return self.value(feature).squeeze(-1)

        @staticmethod
        def diagonal_kl(left_mean, left_log_scale,
                        right_mean, right_log_scale):
            """Return KL(left || right) per latent dimension."""
            left_var = torch.exp(2.0 * left_log_scale)
            right_var = torch.exp(2.0 * right_log_scale)
            return (right_log_scale - left_log_scale +
                    (left_var + (left_mean - right_mean).pow(2)) /
                    (2.0 * right_var) - 0.5)

        def save_checkpoint(self, path, extra=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = path + ".tmp"
            torch.save({
                "kind": "magic-structured-rssm-v1",
                "config": asdict(self.config),
                "latentDim": self.latent_dim,
                "stateDict": self.state_dict(),
                "extra": extra or {},
            }, temporary)
            os.replace(temporary, path)

        @classmethod
        def load_checkpoint(cls, path, map_location="cpu"):
            payload = torch.load(path, map_location=map_location,
                                 weights_only=False)
            if payload.get("kind") != "magic-structured-rssm-v1":
                raise ValueError("not a structured RSSM checkpoint")
            model = cls(StructuredJEPAConfig(**payload["config"]),
                        latent_dim=payload.get("latentDim"))
            model.load_state_dict(payload["stateDict"])
            return model, payload.get("extra") or {}
else:  # pragma: no cover
    class StructuredRSSM(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("structured RSSM requires magic-cabt[jepa]")
