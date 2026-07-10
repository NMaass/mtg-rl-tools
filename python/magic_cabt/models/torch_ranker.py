"""PyTorch option-ranker over canonical action groups.

Implements the MLP described by ``magic_cabt.models.configs`` (the parameter
count matches ``estimate_parameter_count``): a state projection and an option
projection are summed into a shared hidden vector, passed through ``layers``
residual-free hidden blocks, and reduced to one logit per legal option.

Feature vectors come from deterministic token hashing over the canonical
feature text (``features.canonical_text`` output) -- no learned vocabulary,
so the encoder is stable across runs and never keys on per-game instance ids.

PyTorch is an optional dependency: importing this module works without torch
(the hashing helpers are pure stdlib); constructing the model raises a clear
error if torch is missing.
"""

import zlib

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None

__all__ = [
    "hash_features",
    "OptionRanker",
    "TORCH_AVAILABLE",
]

TORCH_AVAILABLE = torch is not None


def hash_features(text, dim):
    """Deterministic hashed bag-of-words vector (list of floats, L2-normed).

    Uses crc32 rather than ``hash()`` so vectors are stable across processes
    and runs (``hash()`` is salted per interpreter).
    """
    vector = [0.0] * dim
    tokens = str(text or "").lower().split()
    if not tokens:
        return vector
    for token in tokens:
        vector[zlib.crc32(token.encode("utf-8")) % dim] += 1.0
    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        vector = [value / norm for value in vector]
    return vector


def _require_torch():
    if torch is None:
        raise ImportError(
            "magic_cabt.models.torch_ranker requires PyTorch; "
            "install it with: pip install torch")


class OptionRanker(nn.Module if nn is not None else object):
    """MLP scorer: (state features, option features) -> one logit per option."""

    def __init__(self, config):
        _require_torch()
        super(OptionRanker, self).__init__()
        self.config = dict(config)
        hidden = int(config["hiddenDim"])
        self.state_proj = nn.Linear(int(config["stateFeatureDim"]), hidden)
        self.option_proj = nn.Linear(int(config["optionFeatureDim"]), hidden)
        self.blocks = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(int(config["layers"])))
        self.head = nn.Linear(hidden, 1)
        self.activation = nn.ReLU()

    def forward(self, state_vec, option_vecs, option_mask):
        """Score options.

        state_vec: [B, stateFeatureDim]
        option_vecs: [B, N, optionFeatureDim] (N = padded option count)
        option_mask: [B, N] bool, True where the option is real
        Returns logits [B, N] with padding filled by -inf.
        """
        state_h = self.state_proj(state_vec).unsqueeze(1)     # [B, 1, H]
        option_h = self.option_proj(option_vecs)              # [B, N, H]
        hidden = self.activation(state_h + option_h)
        for block in self.blocks:
            hidden = self.activation(block(hidden))
        logits = self.head(hidden).squeeze(-1)                # [B, N]
        return logits.masked_fill(~option_mask, float("-inf"))

    def save_checkpoint(self, path):
        _require_torch()
        torch.save({"config": self.config,
                    "stateDict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path, map_location="cpu"):
        _require_torch()
        payload = torch.load(path, map_location=map_location,
                             weights_only=False)
        model = cls(payload["config"])
        model.load_state_dict(payload["stateDict"])
        return model
