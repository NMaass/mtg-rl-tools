"""Extract reviewable tactical scenario drafts from retained DecisionRecords."""
from __future__ import annotations

import copy
import json
import os

from magic_cabt.analysis.schema import decision_fingerprint
from magic_cabt.training.action_dedup import canonical_groups
from magic_cabt.training.io import iter_decision_records

from .tactical import SCHEMA_VERSION, action_key, validate_scenarios
from .trust_audit_common import forbidden_paths

__all__ = [
    "extract_scenario",
    "load_decision_records",
    "resolve_decisions_path",
    "write_scenario_row",
]


def resolve_decisions_path(path):
    resolved = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(resolved):
        resolved = os.path.join(resolved, "decisions.jsonl")
    if not os.path.isfile(resolved):
        raise ValueError("decision dataset not found: %s" % resolved)
    return resolved


def load_decision_records(path):
    resolved = resolve_decisions_path(path)
    return resolved, list(iter_decision_records(resolved))


def extract_scenario(records, suite, scenario_id, decision_index=None,
                     fingerprint=None, history_limit=0, tags=None,
                     acceptable=None, prohibited=None, rationale=None,
                     reviewers=None, source_path=None, source_sha256=None):
    if (decision_index is None) == (fingerprint is None):
        raise ValueError("select exactly one of decision_index or fingerprint")
    if not isinstance(suite, str) or not suite.strip():
        raise ValueError("suite is required")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id is required")
    rows = list(records)
    target_index = _resolve_index(rows, decision_index, fingerprint)
    decision = rows[target_index]
    groups = canonical_groups(_select(decision))
    if len(groups) < 2:
        raise ValueError("selected decision has fewer than two semantic choices")
    history_limit = max(0, int(history_limit))
    game = _game_key(decision)
    history = []
    if history_limit:
        if game == (None, None, None):
            raise ValueError("cannot attach history without a game identity")
        prior = [record for record in rows[:target_index]
                 if _game_key(record) == game]
        history = prior[-history_limit:]
    leaked = forbidden_paths(decision.get("observation") or {})
    for index, record in enumerate(history):
        leaked.extend(forbidden_paths(
            record.get("observation") or {},
            "history[%d].observation" % index))
    if leaked:
        raise ValueError("oracle/private observation keys: %s" %
                         ", ".join(sorted(set(leaked))[:10]))

    available = [_group_summary(group) for group in groups]
    recorded = _recorded_action_keys(decision, groups)
    scenario = {
        "schemaVersion": SCHEMA_VERSION,
        "suite": suite.strip(),
        "scenarioId": scenario_id.strip(),
        "tags": list(tags or []),
        "history": [_clean(record) for record in history],
        "decision": _clean(decision),
        "acceptableActionKeys": list(acceptable or []),
        "prohibitedActionKeys": list(prohibited or []),
        "candidateActions": available,
        "annotation": {
            "reviewStatus": "approved" if acceptable else "needs-expert-review",
            "recordedActionKeys": recorded,
            "reviewers": list(reviewers or []),
            "rationale": rationale or "",
        },
        "provenance": {
            "sourceFile": os.path.basename(source_path)
                if source_path else None,
            "sourceSha256": source_sha256,
            "decisionIndex": target_index,
            "sourceLine": decision.get("__source_line"),
            "decisionFingerprint": decision_fingerprint(decision),
        },
    }
    if acceptable:
        errors, _warnings = validate_scenarios([scenario])
        if errors:
            raise ValueError("invalid approved scenario: " + "; ".join(errors))
    else:
        requested = set(scenario["prohibitedActionKeys"])
        available_keys = set(item["actionKey"] for item in available)
        missing = sorted(requested.difference(available_keys))
        if missing:
            raise ValueError("prohibited actions do not resolve: %s" %
                             ", ".join(missing))
    return scenario


def write_scenario_row(path, scenario, append=False):
    line = json.dumps(scenario, sort_keys=True, ensure_ascii=False) + "\n"
    if path in (None, "-"):
        return line
    resolved = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    if append:
        with open(resolved, "a", encoding="utf-8") as handle:
            handle.write(line)
    else:
        temporary = resolved + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(line)
        os.replace(temporary, resolved)
    return resolved


def _resolve_index(rows, decision_index, fingerprint):
    if decision_index is not None:
        index = int(decision_index)
        if index < 0 or index >= len(rows):
            raise ValueError("decision_index %d outside %d records" %
                             (index, len(rows)))
        return index
    matches = [index for index, record in enumerate(rows)
               if decision_fingerprint(record) == fingerprint]
    if not matches:
        raise ValueError("decision fingerprint not found")
    if len(matches) > 1:
        raise ValueError("decision fingerprint is ambiguous across %d rows" %
                         len(matches))
    return matches[0]


def _group_summary(group):
    option = group.get("option") or {}
    return {
        "actionKey": action_key(group),
        "label": option.get("label"),
        "type": option.get("type"),
        "canonicalKey": group.get("key"),
        "concreteIndices": list(group.get("indices") or []),
    }


def _recorded_action_keys(decision, groups):
    selected = decision.get("selectedIndices") or decision.get("selected") or []
    keys = []
    for group in groups:
        if any(index in selected for index in group.get("indices") or []):
            keys.append(action_key(group))
    return keys


def _game_key(record):
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    observation = record.get("observation") if isinstance(record.get("observation"), dict) else {}
    current = observation.get("current") if isinstance(observation.get("current"), dict) else {}
    game_number = record.get("gameNumber")
    if game_number is None:
        game_number = metadata.get("gameNumber")
    if game_number is None:
        game_number = current.get("gameNumber")
    return (
        record.get("gameId") or record.get("matchId") or
        metadata.get("matchId") or current.get("matchId"),
        game_number,
        current.get("gameInstance"),
    )


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(child) for key, child in value.items()
                if not str(key).startswith("__")}
    if isinstance(value, list):
        return [_clean(child) for child in value]
    return copy.deepcopy(value)


def _select(record):
    direct = record.get("select") if isinstance(record, dict) else None
    if isinstance(direct, dict):
        return direct
    observation = record.get("observation") if isinstance(record, dict) else None
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}
