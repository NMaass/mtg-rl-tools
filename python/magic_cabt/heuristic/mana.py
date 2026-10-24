"""Mana sequencing proxies over legal engine choices, never inferred legality."""
import re
from .state import mana_symbols, mana_value

BASICS = {"Plains": "W", "Island": "U", "Swamp": "B", "Mountain": "R", "Forest": "G"}


def produces(name, text=""):
    if name in BASICS:
        return {BASICS[name]}
    clause = text.split("Add", 1)[-1] if "Add" in text else ""
    return set(re.findall(r"\{([WUBRGC])\}", clause))


def evaluate(context, action, role):
    if action.kind == "PLAY_LAND":
        sources = [c for c in context.ours if c.land and not c.tapped]
        held = [c for c in context.hand if not c.land]
        available = set().union(*(produces(c.name, c.text) for c in sources)) if sources else set()
        offered = produces(action.source_name, action.rule)
        needed = set(s for c in held for s in mana_symbols(c.mana_cost) if s in "WUBRG")
        fixes = len((needed - available) & offered)
        enabled = sum(mana_value(c.mana_cost) == len(sources) + 1 for c in held)
        enters_tapped = "enters" in action.rule.lower() and "tapped" in action.rule.lower()
        return 6.0 + 2.0 * fixes + 0.25 * enabled - (0.8 if enters_tapped else 0), \
            "Make a land drop; prefer missing colors and next-step casting capacity"
    if action.kind == "PROMPT_MANA_POOL":
        return 8.0, "Spend floating mana before committing another permanent"
    if action.kind == "PROMPT_MANA_SOURCE":
        colors = produces(str(action.data.get("objectName") or ""),
                          str(action.data.get("abilityRule") or ""))
        unpaid = set(mana_symbols(str(action.data.get("unpaid") or ""))) & set("WUBRGC")
        matches = bool(colors & unpaid)
        return 4.0 + (2.0 if matches else 0) - 0.2 * max(0, len(colors) - 1), \
            "Pay requested color and preserve flexible sources when equivalent"
    if action.kind == "PROMPT_CANCEL_PAYMENT":
        return -30.0, "Avoid cancelling an already selected legal spell"
    if action.kind == "ACTIVATE_ABILITY" and "add" in action.rule.lower():
        return -5.0, "Do not float speculative mana at priority; payment prompts handle sources"
    if action.kind == "CAST_SPELL":
        return 0.25 * min(7, mana_value(action.mana_cost)), \
            "Small mana-efficiency preference; tactical and plan scores may override"
    return 0.0, "No mana-sequencing preference"
