"""Visible threat values; target direction requires explicit source semantics."""


def value(card):
    if card is None:
        return 0.0
    score = max(0, card.power or 0) + 0.4 * max(0, card.toughness or 0)
    text = card.text.lower()
    for keyword, weight in (("flying", 1.5), ("lifelink", 1.5), ("deathtouch", 1.0),
                            ("draw a card", 1.0), ("planeswalker", 2.0)):
        if keyword in text or keyword in card.types:
            score += weight
    return score


def assess(context):
    return [{"id": c.id, "name": c.name, "threat": round(value(c), 3),
             "basis": "Visible stats plus printed keyword/engine proxies, not a win probability"}
            for c in sorted(context.theirs, key=lambda c: (-value(c), c.name))]


def harmful(action):
    outcome = str(action.data.get("outcome") or "").lower()
    text = str(action.data.get("sourceRule") or action.rule).lower()
    return any(x in outcome for x in ("damage", "destroy", "exile", "detriment", "returntohand")) or \
        any(x in text for x in ("damage to", "destroy target", "exile target", "counter target spell"))


def beneficial(action):
    text = str(action.data.get("sourceRule") or action.rule).lower()
    return any(x in text for x in ("target creature gets +", "target player gains", "target creature gains"))


def evaluate(context, action, role):
    target = context.card(str(action.data.get("targetId") or ""))
    if not target:
        return 0.0, "No visible permanent target"
    is_enemy = target.controller == context.opponent
    if harmful(action):
        return (value(target) if is_enemy else -value(target) - 10,
                "Remove the most consequential opposing threat; avoid harming our board")
    if beneficial(action):
        return (value(target) if not is_enemy else -10,
                "Direct known beneficial effect toward our board")
    return 0.0, "Target effect unknown; do not assume every target prompt is removal"
