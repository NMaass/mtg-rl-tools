"""Random and first-option baseline agents.

``RandomAgent`` is the Magic twin of the CABT sample submission: it samples a
legal count and a distinct set of option indices uniformly at random. Given a
seed it is fully reproducible. ``FirstAgent`` is the deterministic
counterpart: it always takes the lowest-indexed legal selection.

Both are trivial, but they exist as stable, dependency-free evaluation
opponents and as sanity baselines for the tournament runner.
"""

import random

from .base import Agent, clamp_selection, options_of, select_block

__all__ = ["RandomAgent", "FirstAgent"]


class RandomAgent(Agent):
    """Uniform-random legal selection."""

    name = "random"

    def __init__(self, seed=None, rng=None, name=None):
        Agent.__init__(self, name)
        self.rng = rng if rng is not None else random.Random(seed)

    def select(self, observation):
        select = select_block(observation)
        option_count = len(options_of(select))
        if option_count == 0:
            return []
        min_count = select.get("minCount")
        max_count = select.get("maxCount")
        min_count = 0 if not isinstance(min_count, int) else min_count
        if not isinstance(max_count, int) or max_count <= 0:
            max_count = option_count
        low = min(min_count, option_count)
        high = min(max_count, option_count)
        if high < low:
            high = low
        count = self.rng.randint(low, high)
        return sorted(self.rng.sample(range(option_count), count))

    def score(self, observation):
        # Reproducible-but-arbitrary scores so annotation ranks stably per seed.
        return [self.rng.random()
                for _ in range(len(options_of(select_block(observation))))]


class FirstAgent(Agent):
    """Always take the lowest-indexed legal selection (deterministic)."""

    name = "first"

    def __init__(self, seed=None, name=None):
        Agent.__init__(self, name)

    def select(self, observation):
        select = select_block(observation)
        # clamp_selection([]) pads up to minCount with the lowest indices, so
        # this yields exactly the first minCount options (or [] when optional).
        return clamp_selection([], select)

    def score(self, observation):
        # Prefer earlier options: index 0 scores highest, ties broken by index.
        option_count = len(options_of(select_block(observation)))
        if option_count == 0:
            return []
        return [(option_count - index) / float(option_count)
                for index in range(option_count)]
