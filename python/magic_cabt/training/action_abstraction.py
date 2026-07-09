"""Action abstraction profiles for model-size and action-space ablations.

The bridge's ground-truth action space remains the dynamic legal-option list
from XMage. These profiles deliberately sit *above* that list: they map legal
options to coarse or fine buckets so we can measure compressibility and train
small/full policies without deleting rare legal moves from the engine surface.
"""

__all__ = [
    "SMALL_ACTION_BUCKETS",
    "FULL_ACTION_BUCKETS",
    "ACTION_PROFILES",
    "abstract_option",
    "abstract_record",
    "action_bucket_distribution",
]

SMALL_ACTION_BUCKETS = (
    "PASS",
    "PLAY_LAND",
    "CAST_SPELL",
    "ACTIVATE_ABILITY",
    "ATTACK",
    "BLOCK",
    "TARGET_OR_CHOICE",
    "MANA_PAYMENT",
    "MULLIGAN",
    "OTHER",
)

# Fine enough to compare against MTG-Causal-RL-style categorical masking while
# still remaining independent of any one fixed 478-index specification.
FULL_ACTION_BUCKETS = (
    "PASS_PRIORITY",
    "PLAY_LAND",
    "CAST_CREATURE",
    "CAST_INSTANT_OR_SORCERY",
    "CAST_NONCREATURE_PERMANENT",
    "CAST_OTHER_SPELL",
    "ACTIVATE_MANA_ABILITY",
    "ACTIVATE_NONMANA_ABILITY",
    "SPECIAL_ACTION",
    "PROMPT_TARGET_OPPONENT",
    "PROMPT_TARGET_SELF",
    "PROMPT_TARGET_OWN_OBJECT",
    "PROMPT_TARGET_OPPONENT_OBJECT",
    "PROMPT_TARGET_OTHER",
    "PROMPT_YES",
    "PROMPT_NO",
    "PROMPT_CHOICE",
    "PROMPT_MODE",
    "PROMPT_NUMBER",
    "PROMPT_AMOUNT_ASSIGNMENT",
    "PROMPT_TRIGGERED_ABILITY",
    "PROMPT_REPLACEMENT_EFFECT",
    "PROMPT_MANA_SOURCE",
    "PROMPT_MANA_POOL",
    "PROMPT_SPECIAL_MANA",
    "PROMPT_CANCEL_PAYMENT",
    "PROMPT_ATTACKER",
    "PROMPT_BLOCKER",
    "PROMPT_KEEP",
    "PROMPT_MULLIGAN",
    "OTHER",
)

ACTION_PROFILES = {
    "small": SMALL_ACTION_BUCKETS,
    "full": FULL_ACTION_BUCKETS,
}


def abstract_record(record, profile="small"):
    """Return a record-level action-abstraction summary.

    The chosen bucket is based on the first selected index. Multi-select prompts
    keep every chosen bucket in ``selectedBuckets`` so downstream tools can
    separately analyse the loss from flattening multi-select decisions.
    """
    select = _select(record)
    options = select.get("option") or []
    selected = record.get("selectedIndices") or []
    option_buckets = [abstract_option(option, record, profile=profile)
                      for option in options]
    selected_buckets = []
    for index in selected:
        if isinstance(index, int) and not isinstance(index, bool) \
                and 0 <= index < len(option_buckets):
            selected_buckets.append(option_buckets[index])
    return {
        "profile": profile,
        "promptType": select.get("type"),
        "optionBuckets": option_buckets,
        "selectedBuckets": selected_buckets,
        "chosenBucket": selected_buckets[0] if selected_buckets else None,
    }


def abstract_option(option, record=None, profile="small"):
    """Map one legal option to a profile bucket."""
    option_type = str((option or {}).get("type") or "").upper()
    payload = (option or {}).get("payload") or {}
    prompt = _select(record or {}).get("type") if isinstance(record, dict) else None
    prompt = str(prompt or "").upper()

    if profile == "small":
        return _small_bucket(option_type, prompt)
    if profile == "full":
        return _full_bucket(option_type, prompt, payload, record)
    raise ValueError("unknown action abstraction profile: %r" % (profile,))


def action_bucket_distribution(records, profile="small"):
    """Count selected action buckets in a DecisionRecord iterable."""
    counts = {}
    total = 0
    for record in records:
        summary = abstract_record(record, profile=profile)
        for bucket in summary["selectedBuckets"]:
            counts[bucket] = counts.get(bucket, 0) + 1
            total += 1
    return {"profile": profile, "total": total, "buckets": counts}


def _small_bucket(option_type, prompt):
    if option_type == "PASS_PRIORITY" or option_type == "PASS":
        return "PASS"
    if option_type == "PLAY_LAND":
        return "PLAY_LAND"
    if option_type == "CAST_SPELL":
        return "CAST_SPELL"
    if option_type == "ACTIVATE_ABILITY":
        return "ACTIVATE_ABILITY"
    if option_type == "PROMPT_ATTACKER" or prompt == "DECLARE_ATTACKERS":
        return "ATTACK"
    if option_type == "PROMPT_BLOCKER" or prompt == "DECLARE_BLOCKERS":
        return "BLOCK"
    if option_type.startswith("PROMPT_MANA") or option_type == "PROMPT_CANCEL_PAYMENT":
        return "MANA_PAYMENT"
    if option_type in ("PROMPT_KEEP", "PROMPT_MULLIGAN") or prompt == "MULLIGAN":
        return "MULLIGAN"
    if option_type.startswith("PROMPT_"):
        return "TARGET_OR_CHOICE"
    return "OTHER"


def _full_bucket(option_type, prompt, payload, record):
    if option_type == "CAST_SPELL":
        return _spell_bucket(payload)
    if option_type == "ACTIVATE_ABILITY":
        ability_type = str(payload.get("abilityType") or "").upper()
        if "MANA" in ability_type:
            return "ACTIVATE_MANA_ABILITY"
        return "ACTIVATE_NONMANA_ABILITY"
    if option_type in FULL_ACTION_BUCKETS:
        if option_type in ("PROMPT_OBJECT", "PROMPT_PLAYER", "PROMPT_CARD"):
            return _target_bucket(payload, record)
        return option_type
    if prompt == "TARGET":
        return _target_bucket(payload, record)
    if option_type == "SPECIAL_ACTION":
        return "SPECIAL_ACTION"
    return "OTHER"


def _spell_bucket(payload):
    text = " ".join(str(payload.get(key) or "") for key in (
        "cardType", "type", "rule", "sourceName", "label"))
    upper = text.upper()
    if "CREATURE" in upper:
        return "CAST_CREATURE"
    if "INSTANT" in upper or "SORCERY" in upper:
        return "CAST_INSTANT_OR_SORCERY"
    if any(word in upper for word in ("ARTIFACT", "ENCHANTMENT", "PLANESWALKER", "BATTLE")):
        return "CAST_NONCREATURE_PERMANENT"
    return "CAST_OTHER_SPELL"


def _target_bucket(payload, record):
    acting_player = record.get("playerIndex") if isinstance(record, dict) else None
    target_player = payload.get("playerIndex")
    owner = payload.get("ownerIndex")
    controller = payload.get("controllerIndex")
    if target_player is not None:
        return "PROMPT_TARGET_SELF" if target_player == acting_player \
            else "PROMPT_TARGET_OPPONENT"
    if owner is not None or controller is not None:
        side = controller if controller is not None else owner
        return "PROMPT_TARGET_OWN_OBJECT" if side == acting_player \
            else "PROMPT_TARGET_OPPONENT_OBJECT"
    label = str(payload.get("label") or payload.get("name") or "").lower()
    if "opponent" in label:
        return "PROMPT_TARGET_OPPONENT"
    if "you" in label or "your" in label:
        return "PROMPT_TARGET_SELF"
    return "PROMPT_TARGET_OTHER"


def _select(record):
    if not isinstance(record, dict):
        return {}
    top = record.get("select")
    if isinstance(top, dict):
        return top
    observation = record.get("observation") or {}
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}
