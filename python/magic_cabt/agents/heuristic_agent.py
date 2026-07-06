"""Deterministic heuristic baseline (policy v0).

Not strategically good -- just a stable, non-random target the tournament and
annotation tooling can lean on. The ordering follows the issue's suggested
priority:

1. keep (don't mulligan) when facing a mulligan decision
2. play a land when one is available
3. cast a spell / activate an ability when available
4. attack with everything; never make a (fiddly) block
5. pay mana toward a cost; pass priority as the fallback

Everything is driven off the option ``type`` values the Java bridge emits
(``PLAY_LAND`` / ``CAST_SPELL`` / ``PASS_PRIORITY`` / ``PROMPT_KEEP`` / ...),
so it needs no card knowledge and never returns an illegal selection.
"""

from .base import Agent, clamp_selection, options_of, select_block

__all__ = ["HeuristicAgent"]

# Per select-type preference over option *types*: earlier = more preferred.
# Any option type not listed sorts after the listed ones (by index), so novel
# prompts still get a deterministic, legal answer rather than a crash.
_PREFERENCES = {
    "PRIORITY": ["PLAY_LAND", "CAST_SPELL", "ACTIVATE_ABILITY",
                 "SPECIAL_ACTION", "PASS_PRIORITY"],
    "MULLIGAN": ["PROMPT_KEEP", "PROMPT_MULLIGAN"],
    "PAY_MANA": ["PROMPT_MANA_SOURCE", "PROMPT_SPECIAL_MANA",
                 "PROMPT_MANA_POOL", "PROMPT_CANCEL_PAYMENT"],
    "YES_NO": ["PROMPT_YES", "PROMPT_NO"],
}


class HeuristicAgent(Agent):
    """Deterministic option-type priority policy."""

    name = "heuristic"

    def __init__(self, seed=None, name=None):
        Agent.__init__(self, name)

    def select(self, observation):
        select = select_block(observation)
        desired = self._preference_order(select)
        return clamp_selection(desired, select)

    def score(self, observation):
        select = select_block(observation)
        option_count = len(options_of(select))
        if option_count == 0:
            return []
        # Score by position in the preference order: the option select() would
        # pick first scores highest (1.0), the last scores 1/n. This keeps
        # score() and select() consistent for single-select prompts.
        order = self._preference_order(select)
        position = dict((index, rank) for rank, index in enumerate(order))
        return [(option_count - position.get(option["index"], option_count - 1))
                / float(option_count)
                for option in options_of(select)]

    # --- ordering ---------------------------------------------------------

    def _preference_order(self, select):
        """Return every option index ordered most- to least-preferred."""
        options = options_of(select)
        select_type = select.get("type") if isinstance(select, dict) else None

        if select_type == "DECLARE_ATTACKERS":
            # Attack with everything; clamp_selection trims to maxCount.
            return _indices_by(options, lambda o: o.get("type") == "PROMPT_ATTACKER")
        if select_type == "DECLARE_BLOCKERS":
            # Blocking assignments are fiddly to get "obviously legal"; decline
            # to block. clamp_selection([]) -> [] unless a block is forced.
            return []

        preferences = _PREFERENCES.get(select_type)
        if not preferences:
            # No opinion: fall back to option order (deterministic + legal).
            return [option["index"] for option in options]

        rank = dict((option_type, position)
                    for position, option_type in enumerate(preferences))
        default = len(preferences)
        ordered = sorted(
            options,
            key=lambda option: (rank.get(option.get("type"), default),
                                option["index"]),
        )
        return [option["index"] for option in ordered]


def _indices_by(options, predicate):
    """Indices of options matching ``predicate`` first, the rest after."""
    matched = [o["index"] for o in options if predicate(o)]
    rest = [o["index"] for o in options if not predicate(o)]
    return matched + rest
