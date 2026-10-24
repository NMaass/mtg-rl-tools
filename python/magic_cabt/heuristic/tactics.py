"""Bounded tactical estimates, not a replacement for legal engine rollouts."""
import re
from .threats import beneficial, harmful, value


def damage(action):
    text = str(action.data.get("sourceRule") or action.rule)
    match = re.search(r"deals? (\d+) damage to (?:any target|target (?:creature|player|opponent))", text, re.I)
    return int(match.group(1)) if match else 0


def evaluate(context, action, role):
    kind, p = action.kind, action.data
    if kind in ("CONCEDE", "PROMPT_CONCEDE"):
        return -1000.0, "Never concede as a heuristic shortcut"
    if kind == "PROMPT_KEEP" or kind == "PROMPT_MULLIGAN":
        lands = sum(c.land for c in context.hand)
        keep = 2 <= lands <= 5 or p.get("mulliganDownTo", 7) <= 4
        return (10.0 if (kind == "PROMPT_KEEP") == keep else -10.0), \
            "Bounded opening-hand land heuristic; stop mulliganing at four"
    target_id = str(p.get("targetId") or "")
    burn = damage(action)
    if target_id in (context.hero, context.opponent) and (harmful(action) or beneficial(action)):
        if harmful(action):
            if target_id == context.hero:
                return -100.0, "Avoid harmful targeting of ourselves"
            if burn >= context.opponent_life > 0:
                return 100.0, "Apparent damage lethal from visible life; responses/prevention remain unknown"
            return 1.0, "Damage opponent when no stronger permanent-removal preference applies"
        return (5.0 if target_id == context.hero else -10.0), "Known beneficial player effect"
    if burn and kind == "CAST_SPELL":
        if burn >= context.opponent_life > 0:
            return 50.0, "Potential direct-damage lethal; downstream target choice remains separate"
        vulnerable = [c for c in context.theirs if c.creature and
                      c.toughness is not None and 0 < c.toughness <= burn]
        return (min(6.0, max(map(value, vulnerable))) if vulnerable else -3.0), \
            "Prefer burn as removal; conserve it when no vulnerable visible threat exists"
    if burn and target_id:
        target = context.card(target_id)
        if target and target.controller == context.opponent and target.toughness is not None:
            return (5.0 if target.toughness <= burn else -15.0), \
                "Prefer killing a creature over nonlethal damage; marked damage is unavailable"
    if kind == "PROMPT_ATTACKER":
        attacker = context.card(str(p.get("attackerId") or ""))
        if not attacker or attacker.power is None:
            return -0.5, "Unknown attacker stats; conservative default"
        blockers = [c for c in context.theirs if c.creature and not c.tapped]
        flying = "flying" in attacker.text.lower()
        if flying:
            blockers = [c for c in blockers if any(k in c.text.lower() for k in ("flying", "reach"))]
        if not blockers:
            return 2.0 + attacker.power, "Visible unblocked attack; unknown interaction may change result"
        bad = any((b.power or 0) >= (attacker.toughness or 0) and
                  (b.toughness or 0) > attacker.power for b in blockers)
        return (-4.0 if bad else 1.0), "Avoid an apparent free loss into an untapped blocker"
    if kind == "PROMPT_BLOCKER":
        blocker = context.card(str(p.get("blockerId") or ""))
        attacker = context.card(str(p.get("attackerId") or ""))
        if not blocker or not attacker:
            return -0.5, "Unknown combat objects; no invented favorable trade"
        kill = (blocker.power or 0) >= (attacker.toughness or 999)
        survive = (blocker.toughness or 0) > (attacker.power or 0)
        lethal_pressure = sum(max(0, c.power or 0) for c in context.theirs if c.creature) >= context.life
        score = (value(attacker) if kill else 0) - (0 if survive else value(blocker))
        score += 12 if lethal_pressure else 0.2 * (attacker.power or 0)
        return score, "Compare visible trade and life pressure; no multi-block or keyword proof"
    return 0.0, "No supported immediate tactic"
