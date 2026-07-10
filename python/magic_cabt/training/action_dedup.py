"""Fungible-option grouping: a derived view over a select block.

Two legal options that point at strategically identical objects (an
opponent's two identical untapped tokens, say) are the same *action* even
though the engine lists them separately. This module folds such options into
canonical action groups using the ``canonicalKey`` fingerprint that capture
sources write into each option's payload.

Design constraints (see docs/TARGET_FUNGIBILITY.md):

- The recorded option list is NEVER mutated or collapsed in place --
  ``records.py`` locks ``option[i].index == i`` and historical
  ``selectedIndices`` reference concrete positions, so dedup is only ever a
  view computed on top.
- Options without a ``canonicalKey`` are never merged. Fungibility must be
  proven by the capture source (which has full object state); absence of a
  fingerprint means "assume distinct." No label-text heuristics.
- Execution expands a group back to a concrete option index (the lowest),
  whose payload still carries the real instance id for the engine.
"""

from magic_cabt.agents.base import options_of

__all__ = [
    "canonical_key_of",
    "canonical_groups",
    "group_index_of",
    "representative_index",
]


def canonical_key_of(option):
    """Return the option's canonical fingerprint, or ``None`` when absent."""
    payload = (option or {}).get("payload")
    if not isinstance(payload, dict):
        return None
    key = payload.get("canonicalKey")
    return key if isinstance(key, str) and key else None


def canonical_groups(select):
    """Group option indices by ``(type, canonicalKey)``.

    Returns a list of groups in first-appearance order, each a dict with:

    - ``key``: the shared canonical key, or ``None`` for a singleton whose
      option carries no fingerprint (never merged);
    - ``indices``: the concrete option indices in the group, ascending;
    - ``option``: the group's first option dict (for feature extraction).
    """
    groups = []
    by_key = {}
    for index, option in enumerate(options_of(select)):
        key = canonical_key_of(option)
        if key is None:
            groups.append({"key": None, "indices": [index], "option": option})
            continue
        typed_key = ((option or {}).get("type"), key)
        group = by_key.get(typed_key)
        if group is None:
            group = {"key": key, "indices": [], "option": option}
            by_key[typed_key] = group
            groups.append(group)
        group["indices"].append(index)
    return groups


def group_index_of(groups, option_index):
    """Map a concrete option index to its group's position.

    Used to fold a recorded human/self-play choice (a concrete index) into
    the canonical action it represents for training targets. Returns ``None``
    when the index is not in any group (out of range).
    """
    for position, group in enumerate(groups):
        if option_index in group["indices"]:
            return position
    return None


def representative_index(group):
    """Concrete option index to execute for a chosen group (lowest member).

    The member's payload still carries its real instance id, so the engine
    path (``game_select`` / GRE response) needs no changes.
    """
    return group["indices"][0]
