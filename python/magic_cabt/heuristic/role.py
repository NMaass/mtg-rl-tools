"""Role assignment is a visible-board hypothesis, not a hidden-deck oracle."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Role:
    name: str
    our_clock: float
    their_clock: float
    explanation: str

    def to_dict(self):
        return asdict(self)


def assign(context):
    ours = sum(max(0, c.power or 0) for c in context.ours if c.creature)
    theirs = sum(max(0, c.power or 0) for c in context.theirs if c.creature)
    # 999 is an explicit no-current-pressure sentinel, not JSON Infinity.
    our_clock = math.ceil(context.opponent_life / ours) if ours else 999
    their_clock = math.ceil(context.life / theirs) if theirs else 999
    if theirs >= context.life and theirs > 0:
        name, reason = "stabilize", "Visible opposing power threatens life total; preserve survival options."
    elif our_clock < their_clock or (our_clock == their_clock and ours > 0 and
                                    context.hand_count <= context.opponent_hand_count):
        name, reason = "beatdown", "Our unblocked clock is faster; convert board pressure into damage."
    elif theirs > ours:
        name, reason = "control", "Opponent has more visible pressure; prioritize efficient trades and development."
    else:
        name, reason = "develop", "Neither board establishes a decisive race; develop resources."
    return Role(name, our_clock, their_clock,
                reason + " Clocks ignore blockers and future draws; inevitability is unknown.")


def evaluate(context, action, role):
    if action.kind == "PROMPT_ATTACKER":
        return (0.8 if role.name == "beatdown" else -0.5 if role.name == "stabilize" else 0.0,
                "Adjust aggression for the current role")
    if action.kind == "CAST_SPELL" and "creature" in action.source_types:
        return (0.5 if role.name in ("control", "stabilize") else 0.3,
                "Develop a blocker or pressure according to role")
    return 0.0, "Role adds no independent preference"
