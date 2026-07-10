"""Behavior-cloning agent wrapper for live CABT observations."""

import os

from magic_cabt.models import BagOfWordsBCPolicy
from magic_cabt.training import features

from .base import Agent, clamp_selection, options_of, select_block

__all__ = ["BCAgent"]

CHECKPOINT_ENV_VAR = "MAGIC_CABT_BC_CHECKPOINT"


class BCAgent(Agent):
    """Agent that scores current legal options with a BC checkpoint."""

    name = "bc"

    def __init__(self, checkpoint=None, policy=None, name=None):
        Agent.__init__(self, name)
        if policy is None:
            checkpoint = checkpoint or os.environ.get(CHECKPOINT_ENV_VAR)
            if not checkpoint:
                raise ValueError(
                    "BCAgent needs checkpoint=... or $%s" % CHECKPOINT_ENV_VAR)
            policy = BagOfWordsBCPolicy.load(checkpoint)
        self.policy = policy

    def select(self, observation):
        select = select_block(observation)
        scores = self.score(observation)
        ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return clamp_selection(ranking, select)

    def score(self, observation):
        select = select_block(observation)
        options = options_of(select)
        example = {
            "promptType": select.get("type") or "UNKNOWN",
            "optionTypes": [features.option_type(option) for option in options],
            "optionTexts": [features.option_text(option) for option in options],
        }
        return self.policy.score_example(example)
