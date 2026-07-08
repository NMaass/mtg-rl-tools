"""Local agent interface for observation-driven CABT play.

An ``Agent`` maps a bridge observation dict to a list of legal option
indices -- exactly the contract ``examples/random_agent.py`` follows and the
one ``CabtBridge.game_select`` expects: each index ``>= 0`` and
``< len(select["option"])``, distinct, and with a length inside
``[minCount, maxCount]`` (``maxCount == 0`` means "no upper bound").

The interface is deliberately plain Python -- no framework, no base-class
machinery beyond ``select`` (choose indices) and ``score`` (rank options for
annotation). Agent 2's model can later subclass ``Agent`` and override
``score`` with real policy logits; nothing here assumes a model.

Legality is the engine's job: the bridge only ever offers legal options, so
the helpers here (``is_legal_selection`` / ``clamp_selection``) only need to
enforce the count/range/uniqueness envelope, never Magic rules.
"""

__all__ = [
    "Agent",
    "select_block",
    "options_of",
    "is_legal_selection",
    "clamp_selection",
    "register_agent",
    "make_agent",
    "available_agents",
]


def select_block(observation):
    """Return the pending-decision ``select`` spec.

    Accepts a full observation (``{"select": {...}}``), a canonical
    DecisionRecord (top-level ``select``), or a bare select dict, so agents
    and scorers can be handed any of them.
    """
    if isinstance(observation, dict):
        nested = observation.get("select")
        if isinstance(nested, dict):
            return nested
    return observation if isinstance(observation, dict) else {}


def options_of(select):
    """Return the option list of a select block (empty when absent)."""
    options = select.get("option") if isinstance(select, dict) else None
    return options if isinstance(options, list) else []


def _count_bounds(select, option_count):
    min_count = select.get("minCount")
    max_count = select.get("maxCount")
    min_count = 0 if not isinstance(min_count, int) else min_count
    # maxCount == 0 (or missing) means "no upper bound" -> cap at option_count
    if not isinstance(max_count, int) or max_count <= 0:
        max_count = option_count
    return min_count, max_count


def is_legal_selection(indices, select):
    """True when ``indices`` is a legal answer to ``select``.

    Enforces the count envelope, index range, integer-ness, and uniqueness --
    the same rules the Java bridge validates server-side.
    """
    if not isinstance(indices, list):
        return False
    option_count = len(options_of(select))
    for value in indices:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if value < 0 or value >= option_count:
            return False
    if len(indices) != len(set(indices)):
        return False
    min_count, max_count = _count_bounds(select, option_count)
    return min_count <= len(indices) <= max_count


def clamp_selection(desired, select):
    """Coerce a candidate index list into a legal selection for ``select``.

    Keeps the first legal, distinct, in-range indices from ``desired`` (order
    preserved), trims to ``maxCount``, then pads up to ``minCount`` with the
    lowest unused indices. Always returns a legal selection when the prompt is
    satisfiable, so a misbehaving agent can never wedge or crash a game.
    """
    option_count = len(options_of(select))
    min_count, max_count = _count_bounds(select, option_count)

    kept = []
    for value in (desired or []):
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if 0 <= value < option_count and value not in kept:
            kept.append(value)
    if len(kept) > max_count:
        kept = kept[:max_count]
    if len(kept) < min_count:
        for value in range(option_count):
            if value not in kept:
                kept.append(value)
                if len(kept) >= min_count:
                    break
    return kept


class Agent(object):
    """Base class: choose option indices for a decision, and score options.

    Subclasses override ``select``. ``score`` defaults to no preference
    (uniform), which is enough for annotation to run against any agent.
    """

    name = "agent"

    def __init__(self, name=None):
        if name:
            self.name = name

    def select(self, observation):
        """Return a legal list of option indices for the pending decision."""
        raise NotImplementedError

    def score(self, observation):
        """Return a per-option score aligned to option index (higher = more
        preferred). Default: uniform ``1.0`` -- no preference."""
        return [1.0] * len(options_of(select_block(observation)))

    def __repr__(self):
        return "%s(name=%r)" % (type(self).__name__, self.name)


# --- registry ---------------------------------------------------------------
#
# Registration happens in ``agents/__init__.py`` (importing the classes there
# avoids a circular import, since the agent modules import this one).

_REGISTRY = {}


def register_agent(name, factory):
    """Register ``factory(seed=...) -> Agent`` under ``name``."""
    _REGISTRY[name] = factory


def make_agent(spec, seed=None):
    """Build an agent by name (``"random"`` / ``"first"``).

    ``seed`` seeds any stochastic agent so tournaments and annotations are
    reproducible; deterministic agents ignore it.
    """
    factory = _REGISTRY.get(spec)
    if factory is None:
        raise ValueError(
            "unknown agent %r; known agents: %s"
            % (spec, ", ".join(available_agents()))
        )
    return factory(seed=seed)


def available_agents():
    """Sorted list of registered agent names."""
    return sorted(_REGISTRY)
