"""Plan-conditioned evaluation keeps cheap action ranking inspectable."""
from .threats import value


def plan(context, role):
    if role.name == "stabilize":
        return {"objective": "Survive the next opposing attack", "risk": "Clock excludes blocks and tricks"}
    if role.name == "beatdown":
        return {"objective": "Convert our faster visible clock into damage", "risk": "Do not assume opponent lacks interaction"}
    if role.name == "control":
        return {"objective": "Trade with important threats and rebuild a stable board", "risk": "Long-game inevitability is not known"}
    return {"objective": "Develop mana and deploy efficient pressure", "risk": "Reassess role after each public change"}


def evaluate(context, action, role):
    if action.kind == "CAST_SPELL" and "creature" in action.source_types:
        card = context.card(action.source_id)
        if card is None:
            card = next((c for c in context.hand if c.name == action.source_name), None)
        return 2.0 + 0.4 * value(card), "Deploy a body supporting development, pressure, or stabilization"
    if action.kind == "CAST_SPELL":
        return 0.2, "No specialized plan for this spell; tactics must justify spending it"
    if action.kind == "PASS_PRIORITY":
        return 0.0, "Passing preserves resources and lets the stack or turn advance"
    if action.kind == "ACTIVATE_ABILITY":
        return -0.1, "Unmodeled activated abilities require search or review"
    return 0.0, "Plan adds no preference for this subdecision"
