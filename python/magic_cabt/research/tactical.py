"""Versioned replay-root tactical scenario validation and evaluation."""
from __future__ import annotations

import json
import math
from collections import defaultdict

from magic_cabt.training.action_dedup import canonical_groups

from .trust_audit_common import forbidden_paths

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "action_key",
    "evaluate_scenarios",
    "load_scenarios",
    "validate_scenarios",
]


def load_scenarios(path):
    scenarios = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError("scenario line %d is not an object" % line_number)
            value.setdefault("__sourceLine", line_number)
            scenarios.append(value)
    errors, warnings = validate_scenarios(scenarios)
    if errors:
        raise ValueError("invalid tactical suite: " + "; ".join(errors[:10]))
    return scenarios, warnings


def validate_scenarios(scenarios):
    errors = []
    warnings = []
    seen = set()
    suite_names = set()
    for position, scenario in enumerate(scenarios):
        label = "line %s" % scenario.get("__sourceLine", position + 1)
        if scenario.get("schemaVersion") != SCHEMA_VERSION:
            errors.append("%s: schemaVersion must be %d" %
                          (label, SCHEMA_VERSION))
        suite = scenario.get("suite")
        if not isinstance(suite, str) or not suite.strip():
            errors.append("%s: suite is required" % label)
        else:
            suite_names.add(suite)
        identifier = scenario.get("scenarioId")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append("%s: scenarioId is required" % label)
        elif identifier in seen:
            errors.append("%s: duplicate scenarioId %s" % (label, identifier))
        else:
            seen.add(identifier)
        decision = scenario.get("decision")
        if not isinstance(decision, dict):
            errors.append("%s: decision must be an object" % label)
            continue
        history = scenario.get("history") or []
        if not isinstance(history, list) or any(
                not isinstance(item, dict) for item in history):
            errors.append("%s: history must be a list of records" % label)
            history = []
        leaked = forbidden_paths(decision.get("observation") or {})
        for index, item in enumerate(history):
            leaked.extend(forbidden_paths(
                item.get("observation") or {},
                "history[%d].observation" % index))
        if leaked:
            errors.append("%s: oracle/private observation keys: %s" %
                          (label, ", ".join(sorted(set(leaked))[:10])))
        select = _select(decision)
        groups = canonical_groups(select)
        if len(groups) < 2:
            errors.append(
                "%s: tactical decision needs at least two semantic choices" %
                label)
        available = [action_key(group) for group in groups]
        acceptable_raw = scenario.get("acceptableActionKeys")
        prohibited_raw = scenario.get("prohibitedActionKeys")
        if not _is_string_list(acceptable_raw):
            errors.append(
                "%s: acceptableActionKeys must be a list of strings" % label)
        if prohibited_raw is not None and not _is_string_list(prohibited_raw):
            errors.append(
                "%s: prohibitedActionKeys must be a list of strings" % label)
        acceptable = _string_list(acceptable_raw)
        prohibited = _string_list(prohibited_raw)
        if not acceptable:
            errors.append("%s: acceptableActionKeys must be non-empty" % label)
        missing = sorted(set(acceptable + prohibited).difference(available))
        if missing:
            errors.append("%s: expectations do not resolve: %s" %
                          (label, ", ".join(missing)))
        overlap = sorted(set(acceptable).intersection(prohibited))
        if overlap:
            errors.append("%s: acceptable/prohibited overlap: %s" %
                          (label, ", ".join(overlap)))
        unstable = [key for key in acceptable + prohibited
                    if key.startswith("INDEX|")]
        if unstable:
            warnings.append("%s: positional expectations are unstable: %s" %
                            (label, ", ".join(sorted(set(unstable)))))
        tags = scenario.get("tags") or []
        if not isinstance(tags, list) or any(
                not isinstance(tag, str) for tag in tags):
            errors.append("%s: tags must be a list of strings" % label)
    if len(suite_names) > 1:
        errors.append("one JSONL file must contain one suite name")
    if not scenarios:
        errors.append("tactical suite is empty")
    return errors, warnings


def action_key(group):
    option = group.get("option") or {}
    kind = str(option.get("type") or "UNKNOWN")
    canonical = group.get("key")
    if canonical not in (None, ""):
        return "%s|%s" % (kind, canonical)
    indices = group.get("indices") or []
    index = indices[0] if indices else -1
    return "INDEX|%d" % index


def evaluate_scenarios(scenarios, scorers, top_k=3):
    errors, warnings = validate_scenarios(scenarios)
    if errors:
        raise ValueError("invalid tactical suite: " + "; ".join(errors[:10]))
    top_k = max(1, int(top_k))
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "magic-tactical-benchmark-v1",
        "suite": scenarios[0].get("suite") if scenarios else None,
        "scenarioCount": len(scenarios),
        "warnings": warnings,
        "models": {},
    }
    for display_name, scorer in scorers:
        rows = []
        aggregate = _empty_metrics()
        tags = defaultdict(_empty_metrics)
        prompts = defaultdict(_empty_metrics)
        for scenario in scenarios:
            row = _evaluate_one(scenario, scorer, top_k)
            rows.append(row)
            _add_metrics(aggregate, row)
            for tag in scenario.get("tags") or ["untagged"]:
                _add_metrics(tags[tag], row)
            prompt = _select(scenario["decision"]).get("type") or "UNKNOWN"
            _add_metrics(prompts[prompt], row)
        report["models"][display_name] = {
            "model": dict(getattr(scorer, "model_info", {}) or {}),
            "metrics": _finalize_metrics(aggregate),
            "byTag": {key: _finalize_metrics(value)
                      for key, value in sorted(tags.items())},
            "byPromptType": {key: _finalize_metrics(value)
                             for key, value in sorted(prompts.items())},
            "scenarios": rows,
        }
    return report


def _evaluate_one(scenario, scorer, top_k):
    identifier = scenario["scenarioId"]
    try:
        reset = getattr(scorer, "reset", None)
        if callable(reset):
            reset()
        history = scenario.get("history") or []
        observe = getattr(scorer, "observe", None)
        for record in history:
            if callable(observe):
                observe(record)
            elif callable(reset):
                scorer.score(record)
        decision = scenario["decision"]
        scores = scorer.score(decision)
        options = _select(decision).get("option") or []
        if len(scores) != len(options):
            raise ValueError("scorer returned %d scores for %d options" %
                             (len(scores), len(options)))
        numeric = [float(score) for score in scores]
        if any(not math.isfinite(score) for score in numeric):
            raise ValueError("scorer returned non-finite scores")
        ranked = _rank_groups(_select(decision), numeric)
        acceptable = set(scenario.get("acceptableActionKeys") or [])
        prohibited = set(scenario.get("prohibitedActionKeys") or [])
        acceptable_ranks = [
            position + 1 for position, item in enumerate(ranked)
            if item["actionKey"] in acceptable
        ]
        best_rank = min(acceptable_ranks) if acceptable_ranks else None
        top = ranked[0]["actionKey"] if ranked else None
        return {
            "scenarioId": identifier,
            "status": "scored",
            "topActionKey": top,
            "topActions": ranked[:top_k],
            "acceptableRank": best_rank,
            "top1": best_rank == 1,
            "topK": best_rank is not None and best_rank <= top_k,
            "reciprocalRank": 1.0 / best_rank if best_rank else 0.0,
            "prohibitedTop1": top in prohibited,
            "error": None,
        }
    except Exception as error:  # keep the suite report complete
        return {
            "scenarioId": identifier,
            "status": "error",
            "topActionKey": None,
            "topActions": [],
            "acceptableRank": None,
            "top1": False,
            "topK": False,
            "reciprocalRank": 0.0,
            "prohibitedTop1": False,
            "error": str(error),
        }


def _rank_groups(select, scores):
    ranked = []
    for group in canonical_groups(select):
        indices = group.get("indices") or []
        group_score = max(scores[index] for index in indices)
        ranked.append({
            "actionKey": action_key(group),
            "score": group_score,
            "representativeIndex": min(indices),
            "concreteIndices": list(indices),
            "label": (group.get("option") or {}).get("label"),
        })
    ranked.sort(key=lambda item: (
        -item["score"], item["representativeIndex"]))
    return ranked


def _empty_metrics():
    return {"scenarios": 0, "scored": 0, "errors": 0, "top1": 0,
            "topK": 0, "mrr": 0.0, "prohibitedTop1": 0}


def _add_metrics(bucket, row):
    bucket["scenarios"] += 1
    if row["status"] == "scored":
        bucket["scored"] += 1
        bucket["top1"] += int(row["top1"])
        bucket["topK"] += int(row["topK"])
        bucket["mrr"] += row["reciprocalRank"]
        bucket["prohibitedTop1"] += int(row["prohibitedTop1"])
    else:
        bucket["errors"] += 1


def _finalize_metrics(bucket):
    result = dict(bucket)
    scored = bucket["scored"]
    total = bucket["scenarios"]
    result["coverage"] = scored / float(total) if total else 0.0
    result["top1Rate"] = bucket["top1"] / float(scored) if scored else None
    result["topKRate"] = bucket["topK"] / float(scored) if scored else None
    result["mrr"] = bucket["mrr"] / float(scored) if scored else None
    result["prohibitedTop1Rate"] = (
        bucket["prohibitedTop1"] / float(scored) if scored else None)
    return result


def _select(record):
    direct = record.get("select") if isinstance(record, dict) else None
    if isinstance(direct, dict):
        return direct
    observation = record.get("observation") if isinstance(record, dict) else None
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}


def _is_string_list(value):
    return isinstance(value, list) and all(
        isinstance(item, str) for item in value)


def _string_list(value):
    return list(value) if _is_string_list(value) else []
