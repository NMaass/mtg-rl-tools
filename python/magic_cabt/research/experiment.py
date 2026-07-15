"""Declarative validation for MTG learning experiments.

The validator encodes repository-level scientific guardrails rather than model
hyperparameters.  It catches common sources of false confidence: splitting
individual decisions from the same game, leaking hidden zones or instance IDs,
reporting single-seed benchmark claims, and allowing an expert cost model to
silently replace the actual game objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

SCHEMA_VERSION = 1
_ALLOWED_SPLIT_UNITS = frozenset((
    "game", "match", "draft", "event", "session", "player", "time_block",
    "card_set", "deck_archetype", "matchup"))
_FORBIDDEN_SPLIT_UNITS = frozenset((
    "decision", "record", "row", "transition", "snapshot"))
_ALLOWED_PRIVATE_ZONE_POLICIES = frozenset((
    "redacted", "count_only", "belief_only"))
_ALLOWED_COST_ROLES = frozenset((
    "auxiliary_head", "offline_reranker", "potential_shaping", "evaluation_only"))
_WORLD_MODEL_FAMILIES = frozenset((
    "jepa", "rssm", "dreamer", "muzero", "latent_dynamics"))
_IMITATION_FAMILIES = frozenset((
    "behavior_cloning", "option_ranker", "recurrent_behavior_cloning",
    "dagger", "search_distillation"))


@dataclass(frozen=True)
class Problem:
    level: str
    path: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "path": self.path, "message": self.message}


def validate_experiment_plan(payload: Mapping[str, Any]) -> List[Problem]:
    problems: List[Problem] = []
    if not isinstance(payload, Mapping):
        return [Problem("error", "$", "plan must be a JSON object")]
    version = payload.get("schemaVersion")
    if version != SCHEMA_VERSION:
        problems.append(Problem(
            "error", "$.schemaVersion",
            "schemaVersion must be %d" % SCHEMA_VERSION))
    if not str(payload.get("name") or "").strip():
        problems.append(Problem("error", "$.name", "plan needs a stable name"))

    claim_level = str(payload.get("claimLevel") or "exploratory")
    if claim_level not in ("smoke", "exploratory", "benchmark"):
        problems.append(Problem(
            "error", "$.claimLevel",
            "claimLevel must be smoke, exploratory, or benchmark"))

    data = payload.get("data")
    if not isinstance(data, Mapping):
        problems.append(Problem("error", "$.data", "data must be an object"))
        data = {}
    _validate_data(data, problems)

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        problems.append(Problem(
            "error", "$.evaluation", "evaluation must be an object"))
        evaluation = {}
    _validate_evaluation(evaluation, claim_level, problems)

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        problems.append(Problem(
            "error", "$.experiments", "experiments must be a non-empty list"))
        experiments = []
    names = set()
    for index, experiment in enumerate(experiments):
        path = "$.experiments[%d]" % index
        if not isinstance(experiment, Mapping):
            problems.append(Problem("error", path, "experiment must be an object"))
            continue
        name = str(experiment.get("name") or "")
        if not name:
            problems.append(Problem("error", path + ".name", "experiment needs a name"))
        elif name in names:
            problems.append(Problem(
                "error", path + ".name", "experiment names must be unique"))
        names.add(name)
        _validate_experiment(experiment, path, claim_level, data, problems)

    return sorted(problems, key=lambda problem: (
        0 if problem.level == "error" else 1, problem.path, problem.message))


def plan_report(payload: Mapping[str, Any]) -> Dict[str, Any]:
    problems = validate_experiment_plan(payload)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "experiment-plan-validation-v1",
        "valid": not any(problem.level == "error" for problem in problems),
        "errors": sum(problem.level == "error" for problem in problems),
        "warnings": sum(problem.level == "warning" for problem in problems),
        "problems": [problem.to_dict() for problem in problems],
    }


def _validate_data(data: Mapping[str, Any], problems: List[Problem]) -> None:
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        problems.append(Problem(
            "error", "$.data.sources", "declare at least one data source"))
    split = data.get("split")
    if not isinstance(split, Mapping):
        problems.append(Problem(
            "error", "$.data.split", "split must be an object"))
        split = {}
    unit = str(split.get("unit") or "")
    if unit in _FORBIDDEN_SPLIT_UNITS:
        problems.append(Problem(
            "error", "$.data.split.unit",
            "split whole games/matches/drafts, never individual %s rows" % unit))
    elif unit not in _ALLOWED_SPLIT_UNITS:
        problems.append(Problem(
            "error", "$.data.split.unit",
            "split unit must be one of %s" % sorted(_ALLOWED_SPLIT_UNITS)))
    strategy = str(split.get("strategy") or "")
    if not strategy:
        problems.append(Problem(
            "error", "$.data.split.strategy", "declare a deterministic split strategy"))
    holdout = split.get("holdoutFraction")
    if holdout is not None:
        if isinstance(holdout, bool) or not isinstance(holdout, (int, float)):
            problems.append(Problem(
                "error", "$.data.split.holdoutFraction",
                "holdoutFraction must be numeric"))
        elif not 0.0 < float(holdout) < 1.0:
            problems.append(Problem(
                "error", "$.data.split.holdoutFraction",
                "holdoutFraction must be in (0, 1)"))

    hidden = data.get("hiddenInformation")
    if not isinstance(hidden, Mapping):
        problems.append(Problem(
            "error", "$.data.hiddenInformation",
            "hiddenInformation policy must be explicit"))
        hidden = {}
    private_policy = str(hidden.get("opponentPrivateZones") or "")
    if private_policy not in _ALLOWED_PRIVATE_ZONE_POLICIES:
        problems.append(Problem(
            "error", "$.data.hiddenInformation.opponentPrivateZones",
            "must be redacted, count_only, or belief_only"))
    if hidden.get("allowTrueOpponentHand") is not False:
        problems.append(Problem(
            "error", "$.data.hiddenInformation.allowTrueOpponentHand",
            "training/evaluation observations must explicitly forbid true opponent hands"))

    features = data.get("features")
    if not isinstance(features, Mapping):
        problems.append(Problem(
            "error", "$.data.features", "feature guardrails must be explicit"))
        features = {}
    if features.get("includePerGameInstanceIds") is not False:
        problems.append(Problem(
            "error", "$.data.features.includePerGameInstanceIds",
            "per-game instance IDs are leakage/memorization keys and must be disabled"))
    if features.get("semanticActions") is not True:
        problems.append(Problem(
            "error", "$.data.features.semanticActions",
            "actions must be represented semantically, not only by option index"))


def _validate_evaluation(
        evaluation: Mapping[str, Any],
        claim_level: str,
        problems: List[Problem]) -> None:
    metrics = evaluation.get("primaryMetrics")
    if not isinstance(metrics, list) or not metrics:
        problems.append(Problem(
            "error", "$.evaluation.primaryMetrics",
            "declare at least one primary metric before running"))
    baselines = evaluation.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        problems.append(Problem(
            "error", "$.evaluation.baselines",
            "declare at least one reference baseline"))
    paired = evaluation.get("pairedEvaluation")
    if claim_level == "benchmark" and paired is not True:
        problems.append(Problem(
            "error", "$.evaluation.pairedEvaluation",
            "benchmark claims require paired seeds/scenarios"))
    if claim_level == "benchmark" and evaluation.get("confidenceIntervals") is not True:
        problems.append(Problem(
            "error", "$.evaluation.confidenceIntervals",
            "benchmark claims require confidence intervals"))
    if claim_level == "benchmark" and evaluation.get("multipleComparisonCorrection") not in (
            "holm", "holm-bonferroni"):
        problems.append(Problem(
            "error", "$.evaluation.multipleComparisonCorrection",
            "benchmark comparison families require Holm correction"))
    suites = evaluation.get("suites")
    if not isinstance(suites, list) or not suites:
        problems.append(Problem(
            "warning", "$.evaluation.suites",
            "fixed tactical/matchup suites make regressions easier to interpret"))


def _validate_experiment(
        experiment: Mapping[str, Any],
        path: str,
        claim_level: str,
        data: Mapping[str, Any],
        problems: List[Problem]) -> None:
    family = str(experiment.get("modelFamily") or "")
    if not family:
        problems.append(Problem(
            "error", path + ".modelFamily", "declare modelFamily"))
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        problems.append(Problem(
            "error", path + ".seeds", "declare deterministic training seeds"))
        seeds = []
    required = {"smoke": 1, "exploratory": 3, "benchmark": 5}.get(claim_level, 3)
    if len(set(str(seed) for seed in seeds)) < required:
        problems.append(Problem(
            "error" if claim_level == "benchmark" else "warning",
            path + ".seeds",
            "%s plans should use at least %d distinct seeds" %
            (claim_level, required)))

    if family in _WORLD_MODEL_FAMILIES:
        transition_sources = data.get("transitionSources")
        if not isinstance(transition_sources, list) or not transition_sources:
            problems.append(Problem(
                "error", "$.data.transitionSources",
                "%s needs explicit transition sources" % family))
        objectives = experiment.get("objectives")
        if not isinstance(objectives, list) or not objectives:
            problems.append(Problem(
                "error", path + ".objectives",
                "world-model experiments must list prediction/auxiliary objectives"))
        if family == "jepa" and experiment.get("collapseDiagnostics") is not True:
            problems.append(Problem(
                "error", path + ".collapseDiagnostics",
                "JEPA runs must log latent variance/collapse diagnostics"))

    if family in _IMITATION_FAMILIES:
        decision_sources = data.get("decisionSources")
        if not isinstance(decision_sources, list) or not decision_sources:
            problems.append(Problem(
                "error", "$.data.decisionSources",
                "%s needs expert decision sources" % family))

    if experiment.get("usesOpponentHiddenCards") is not False:
        problems.append(Problem(
            "error", path + ".usesOpponentHiddenCards",
            "models may consume beliefs/counts, not true opponent hidden cards"))

    cost = experiment.get("expertCost")
    if cost is not None:
        if not isinstance(cost, Mapping):
            problems.append(Problem(
                "error", path + ".expertCost", "expertCost must be an object"))
        else:
            role = str(cost.get("role") or "")
            if role not in _ALLOWED_COST_ROLES:
                problems.append(Problem(
                    "error", path + ".expertCost.role",
                    "role must be one of %s" % sorted(_ALLOWED_COST_ROLES)))
            if role == "potential_shaping" and cost.get("form") != "gamma_phi_next_minus_phi":
                problems.append(Problem(
                    "error", path + ".expertCost.form",
                    "policy-preserving shaping must use gamma*Phi(next)-Phi(current)"))
            if role == "terminal_reward_replacement":
                problems.append(Problem(
                    "error", path + ".expertCost.role",
                    "do not silently replace the win/loss objective with an expert score"))
            if cost.get("heldOutExpertEvaluation") is not True:
                problems.append(Problem(
                    "warning", path + ".expertCost.heldOutExpertEvaluation",
                    "evaluate cost models on held-out contexts and experts"))

    ablations = experiment.get("ablations")
    if claim_level == "benchmark" and family in _WORLD_MODEL_FAMILIES:
        if not isinstance(ablations, list) or not ablations:
            problems.append(Problem(
                "warning", path + ".ablations",
                "benchmark world-model claims should isolate representation, dynamics, and policy losses"))
