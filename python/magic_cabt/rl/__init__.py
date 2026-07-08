"""RL-facing helpers for MTG CABT games.

The modules in this package intentionally avoid a hard dependency on Gymnasium
or Stable-Baselines. They expose the same core pieces those frameworks need —
``reset`` / ``step`` / action masks / terminal rewards — while staying usable in
this repository's dependency-free Python test gate.
"""

from .env import (
    DiscreteActionSpace,
    InvalidActionError,
    MagicCabtEnv,
    action_mask,
    terminal_reward,
)

__all__ = [
    "DiscreteActionSpace",
    "InvalidActionError",
    "MagicCabtEnv",
    "action_mask",
    "terminal_reward",
]
