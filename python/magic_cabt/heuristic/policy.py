"""Explainable legal-option scorer with explicit compound-selection constraints."""
from dataclasses import asdict, dataclass, field
import math
from . import role, threats, mana, tactics, plans
from .state import digest, project

MODULES = {"role": role, "threats": threats, "mana": mana, "tactics": tactics, "plans": plans}


@dataclass(frozen=True)
class Config:
    weights: dict = field(default_factory=lambda: {name: 1.0 for name in MODULES})
    uncertainty_margin: float = 1.0
    version: str = "explainable-v1"

    def __post_init__(self):
        if set(self.weights) != set(MODULES):
            raise ValueError("Exactly the five named module weights are required")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 4
               for v in self.weights.values()):
            raise ValueError("Module weights must be finite in [0, 4]")
        if type(self.uncertainty_margin) not in (int, float) or not math.isfinite(self.uncertainty_margin) or self.uncertainty_margin < 0:
            raise ValueError("Uncertainty margin must be finite and nonnegative")
        object.__setattr__(self, "weights", dict(self.weights))

    @property
    def fingerprint(self):
        return digest(asdict(self))


def selection_valid(context, selected):
    if not isinstance(selected, list) or any(type(i) is not int for i in selected):
        return False
    if not context.minimum <= len(selected) <= context.maximum or len(set(selected)) != len(selected):
        return False
    if any(i < 0 or i >= len(context.actions) for i in selected):
        return False
    # Per-option legality alone does not make arbitrary combat combinations legal.
    key = "attackerId" if context.prompt == "DECLARE_ATTACKERS" else \
          "blockerId" if context.prompt == "DECLARE_BLOCKERS" else None
    if key:
        ids = [context.actions[i].data.get(key) for i in selected]
        if any(x is None for x in ids) or len(ids) != len(set(ids)):
            return False
    return True


def choose(context, scores):
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], context.actions[i].label, i))
    selected = []
    for index in ranking:
        if len(selected) >= context.maximum:
            break
        if len(selected) >= context.minimum and scores[index] <= 0:
            continue
        proposed = selected + [index]
        # Check compound constraints before final minimum-count validation.
        key = "attackerId" if context.prompt == "DECLARE_ATTACKERS" else \
              "blockerId" if context.prompt == "DECLARE_BLOCKERS" else None
        if key and any(context.actions[j].data.get(key) == context.actions[index].data.get(key)
                       for j in selected):
            continue
        selected.append(index)
    if not selection_valid(context, selected):
        raise ValueError("No valid selection under supported compound constraints")
    return selected


class HeuristicAgent:
    name = "explainable"

    def __init__(self, config=None, catalog=None):
        self.config = config or Config()
        self.catalog = dict(catalog or {})
        self.last_decision = None

    def set_catalog(self, cards):
        self.catalog = {c["name"]: c for c in cards}

    def explain(self, observation):
        context = project(observation, self.catalog)
        assigned = role.assign(context)
        options = []
        for action in context.actions:
            components = []
            for name, module in MODULES.items():
                raw, reason = module.evaluate(context, action, assigned)
                components.append({"module": name, "raw": raw, "weight": self.config.weights[name],
                                   "score": raw * self.config.weights[name], "reason": reason})
            options.append({"index": action.index, "label": action.label,
                            "score": sum(c["score"] for c in components), "components": components})
        scores = [o["score"] for o in options]
        selection = choose(context, scores)
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else None
        return context, {"policy": self.name, "configHash": self.config.fingerprint,
                         "stateHash": digest(context.public_dict()), "role": assigned.to_dict(),
                         "plan": plans.plan(context, assigned), "threats": threats.assess(context),
                         "options": options, "selection": selection, "margin": margin,
                         "uncertain": len(scores) > 1 and (margin is not None and margin <= self.config.uncertainty_margin),
                         "warnings": list(context.warnings),
                         "fallback": context.prompt not in ("PRIORITY", "MULLIGAN", "PAY_MANA", "TARGET", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS"),
                         "coverage": "implemented-proxies" if context.prompt in ("PRIORITY", "MULLIGAN", "PAY_MANA", "TARGET", "DECLARE_ATTACKERS", "DECLARE_BLOCKERS") else "unmodeled-prompt"}

    def score(self, observation):
        _, report = self.explain(observation)
        return [o["score"] for o in report["options"]]

    def select(self, observation):
        _, self.last_decision = self.explain(observation)
        return self.last_decision["selection"]
