"""A random legal agent — the Magic twin of the CABT sample submission.

The agent contract mirrors the Pokemon competition exactly: it receives the
observation dict and returns a list of option indices, each ``>= 0`` and
``< len(observation["select"]["option"])``, with a length between
``minCount`` and ``maxCount`` and no duplicates. Legality comes from the
engine — the bridge only ever offers legal options, so uniform random
sampling is already a valid (if terrible) player.
"""

import random


def agent(observation):
    """Return random legal option indices for the pending decision."""
    select = observation["select"]
    count = random.randint(select["minCount"], select["maxCount"])
    return random.sample(range(len(select["option"])), count)
