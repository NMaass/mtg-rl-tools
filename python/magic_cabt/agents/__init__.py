"""Baseline agents for CABT experiments.

The ``Agent`` interface plays directly off live bridge *observation* dicts:
``select`` chooses legal option indices, ``score`` ranks options for
annotation. The three built-in agents -- ``random``, ``first``, ``heuristic``
-- are registered here and constructed by name via ``make_agent``.

Agent 2's model policy can subclass ``Agent`` and override ``score`` with real
logits; nothing here assumes a model.
"""

from .base import (
    Agent,
    available_agents,
    clamp_selection,
    is_legal_selection,
    make_agent,
    options_of,
    register_agent,
    select_block,
)
from .heuristic_agent import HeuristicAgent
from .random_agent import FirstAgent, RandomAgent

register_agent("random", lambda seed=None: RandomAgent(seed=seed))
register_agent("first", lambda seed=None: FirstAgent(seed=seed))
register_agent("heuristic", lambda seed=None: HeuristicAgent(seed=seed))

__all__ = [
    "Agent",
    "RandomAgent",
    "FirstAgent",
    "HeuristicAgent",
    "make_agent",
    "available_agents",
    "register_agent",
    "select_block",
    "options_of",
    "is_legal_selection",
    "clamp_selection",
]
