"""Dependency-free option-ranking baseline policies."""

import random

__all__ = [
    "rank_options",
]


def rank_options(example, policy="first", rng=None):
    """Return legal option indices ordered from most to least preferred."""
    option_count = len(example.get("optionTexts") or [])
    indices = list(range(option_count))
    if policy == "first":
        return indices
    if policy == "random":
        rng = rng or random.Random()
        rng.shuffle(indices)
        return indices
    raise ValueError("unknown policy: %s" % policy)