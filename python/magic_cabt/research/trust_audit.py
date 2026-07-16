"""Audit datasets and checkpoints before trusting a Magic model result."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from magic_cabt.analysis.schema import decision_fingerprint
from magic_cabt.models.structured_jepa import StructuredJEPAConfig
from magic_cabt.models.visibility import VisibilitySafeTensorizer
from magic_cabt.training import train_jepa as core
from magic_cabt.training.train_information_state import (
    game_key as decision_game_key, sequence_number as decision_sequence)
from magic_cabt.training.train_rssm import (
    transition_game_key, transition_sequence)

_SCHEMA_VERSION = 1
_FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "traininglabels", "oraclelabels", "belieflabels", "trueopponenthand",
    "opponenthandtruth", "hiddenstatetruth", "oraclehiddenstate",
    "privatehand", "enginehiddenstate",
})
_SUPPORTED_CHECKPOINT_KINDS = frozenset({
    "magic-structured-jepa-v1",
    "magic-recurrent-information-state-v1",
    "magic-belief-information-state-v1",
    "magic-structured-rssm-v1",
})


class AuditReport:
    def __init__(self, strict=False):
        self.strict = bool(strict)
        self.checks = []

    def add(self, identifier, status, summary, details=None, scope=None):
        if status not in ("pass", "warn", "fail", "not-applicable"):
            raise ValueError("invalid audit status: %s" % status)
        effective = "fail" if self.strict and status == "warn" else status
        self.checks.append({
            "id": identifier,
            "scope": scope,
            "status": effective,
            "originalStatus": status if effective != status else None,
            "summary": summary,
            "details": details or {},
        })
        return effective

    def finish(self, inputs, checkpoints):
        counts = Counter(check["status"] for check in self.checks)
        return {
            "schemaVersion": _SCHEMA_VERSION,
            "kind": "magic-training-trust-audit-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "strict": self.strict,
            "inputs": [os.path.abspath(os.path.expanduser(path))
                       for path in inputs],
            "checkpoints": [os.path.abspath(os.path.expanduser(path))
                            for path in checkpoints],
            "summary": {
                "pass": counts["pass"],
                "warn": counts["warn"],
                "fail": counts["fail"],
                "notApplicable": counts["not-applicable"],
                "trusted": counts["fail"] == 0,
            },
            "checks": self.checks,
        }


def _normalize_key(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def _find_forbidden_paths(value, prefix="observation"):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = "%s.%s" % (prefix, key)
            if _normalize_key(key) in _FORBIDDEN_OBSERVATION_KEYS:
                paths.append(path)
            paths.extend(_find_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_find_forbidden_paths(
                child, "%s[%d]" % (prefix, index)))
    return paths


def _select(record):
    direct = record.get("select")
    if isinstance(direct, dict):
        return direct
    return (record.get("observation") or {}).get("select") or {}


def _selected(record):
    selected = record.get("selectedIndices")
    if selected is None and isinstance(record.get("select"), list):
        selected = record.get("select")
    if selected is None:
        selected = record.get("selected")
    return selected or []


def _sequence_segment_issues(keys):
    closed = set()
    current = None
    reopened = []
    for key in keys:
        if key is None:
            continue
        if current is None:
            current = key
        elif key != current:
            closed.add(current)
            if key in closed:
                reopened.append(key)
            current = key
    return sorted(set(reopened))


def audit_decisions(report, inputs, max_records=0, visibility_samples=32):
    total = 0
    unknown_game = 0
    invalid_selected = []
    empty_options = []
    duplicate_option_indices = []
    forbidden_paths = []
    generic_history = 0
    public_history = 0
    fingerprints = Counter()
    canonical_type_conflicts = []
    sequence_issues = []
    last_sequence = {}
    ordered_games = []
    records_for_visibility = []
    truncated = False

    for source_index, record in enumerate(core._iter_all_decisions(inputs)):
        if max_records and total >= max_records:
            truncated = True
            break
        total += 1
        game = decision_game_key(record)
        ordered_games.append(game)
        if game is None:
            unknown_game += 1
        else:
            sequence = decision_sequence(record, source_index)
            previous = last_sequence.get(game)
            if previous is not None and sequence < previous:
                sequence_issues.append({
                    "game": game, "previous": previous, "current": sequence,
                    "record": source_index})
            last_sequence[game] = sequence

        select = _select(record)
        options = select.get("option") or []
        if not options:
            empty_options.append(source_index)
            continue
        option_indices = [option.get("index", index)
                          for index, option in enumerate(options)]
        if len(set(map(str, option_indices))) != len(option_indices):
            duplicate_option_indices.append(source_index)
        selected = _selected(record)
        if (not selected or any(not isinstance(index, int) or
                                not 0 <= index < len(options)
                                for index in selected)):
            invalid_selected.append({
                "record": source_index,
                "selected": selected,
                "optionCount": len(options),
            })

        canonical_types = defaultdict(set)
        for option in options:
            payload = option.get("payload") or {}
            key = payload.get("canonicalKey") \
                if isinstance(payload, dict) else None
            if key is not None:
                canonical_types[str(key)].add(str(option.get("type") or ""))
        for key, option_types in canonical_types.items():
            if len(option_types) > 1:
                canonical_type_conflicts.append({
                    "record": source_index, "canonicalKey": key,
                    "types": sorted(option_types)})

        observation = record.get("observation") or {}
        forbidden_paths.extend(_find_forbidden_paths(observation))
        if observation.get("history") is not None:
            generic_history += 1
        if observation.get("publicHistory") is not None:
            public_history += 1
        fingerprints[decision_fingerprint(record)] += 1
        if len(records_for_visibility) < max(0, int(visibility_samples)):
            records_for_visibility.append(record)

    reopened = _sequence_segment_issues(ordered_games)
    duplicate_fingerprints = sum(count - 1 for count in fingerprints.values()
                                 if count > 1)
    report.add(
        "dataset.decisions.present",
        "pass" if total else "fail",
        "Decision records are available" if total else
        "No trainable decision records were found",
        {"records": total, "truncated": truncated}, scope="dataset")
    report.add(
        "dataset.decisions.selected-index",
        "fail" if invalid_selected else "pass",
        "Selected option indices are legal" if not invalid_selected else
        "Some decisions select missing or out-of-range options",
        {"count": len(invalid_selected),
         "examples": invalid_selected[:20]}, scope="dataset")
    report.add(
        "dataset.decisions.options",
        "fail" if empty_options or duplicate_option_indices else "pass",
        "Trainable decisions have non-empty, uniquely indexed options"
        if not empty_options and not duplicate_option_indices else
        "Some decisions have empty or ambiguously indexed option sets",
        {"emptyOptionRows": empty_options[:20],
         "duplicateIndexRows": duplicate_option_indices[:20]},
        scope="dataset")
    report.add(
        "dataset.hidden-information.observation",
        "fail" if forbidden_paths else "pass",
        "No oracle/private label keys occur in model observations"
        if not forbidden_paths else
        "Oracle/private fields occur inside model observations",
        {"count": len(forbidden_paths),
         "paths": sorted(set(forbidden_paths))[:50]}, scope="dataset")
    report.add(
        "dataset.sequence.game-identity",
        "warn" if unknown_game else "pass",
        "All decision rows have a recoverable game identity"
        if not unknown_game else
        "Some decision rows cannot participate in leakage-safe game splits",
        {"unknownRows": unknown_game, "records": total}, scope="dataset")
    report.add(
        "dataset.sequence.order",
        "fail" if sequence_issues or reopened else "pass",
        "Decision streams are monotone and game-contiguous"
        if not sequence_issues and not reopened else
        "Decision streams contain out-of-order or reopened games",
        {"nonMonotone": sequence_issues[:20],
         "reopenedGames": reopened[:20]}, scope="dataset")
    report.add(
        "dataset.history.visibility",
        "warn" if generic_history else "pass",
        "Temporal context uses explicitly public history"
        if not generic_history else
        "Generic history fields are present and must remain excluded",
        {"genericHistoryRows": generic_history,
         "publicHistoryRows": public_history}, scope="dataset")
    report.add(
        "dataset.options.canonical-types",
        "warn" if canonical_type_conflicts else "pass",
        "Canonical action keys are type-consistent within prompts"
        if not canonical_type_conflicts else
        "Some canonical keys merge different action types",
        {"count": len(canonical_type_conflicts),
         "examples": canonical_type_conflicts[:20]}, scope="dataset")
    report.add(
        "dataset.decisions.duplicates",
        "warn" if duplicate_fingerprints else "pass",
        "No duplicate public decision fingerprints were detected"
        if not duplicate_fingerprints else
        "Duplicate public decision fingerprints may overweight repeated rows",
        {"duplicateRows": duplicate_fingerprints,
         "uniqueFingerprints": len(fingerprints)}, scope="dataset")
    audit_visibility_invariance(report, records_for_visibility)
    return {
        "records": total,
        "games": sorted(key for key in last_sequence if key is not None),
        "truncated": truncated,
    }


def _perspective(record, state):
    return (record.get("perspectiveSeat") or state.get("perspectiveSeat") or
            state.get("localSeat"))


def _mutate_hidden_hand(record):
    observation = record.get("observation") or {}
    state = observation.get("current") or record.get("current") or {}
    zones = state.get("zones") or {}
    hand = zones.get("hand")
    view = _perspective(record, state)
    if isinstance(hand, dict):
        for seat, cards in hand.items():
            if view is not None and str(seat) == str(view):
                continue
            if isinstance(cards, list) and cards and isinstance(cards[0], dict):
                cards[0].update({
                    "name": "AUDIT MUTATION", "oracleId": "private",
                    "manaValue": 99, "power": 99, "toughness": 99,
                    "rulesText": "private test payload", "tapped": True,
                })
                return True
    if isinstance(hand, list):
        for card in hand:
            if not isinstance(card, dict):
                continue
            seat = card.get("seat", card.get("ownerSeat",
                                             card.get("controllerSeat")))
            if view is not None and seat is not None and str(seat) == str(view):
                continue
            card.update({
                "name": "AUDIT MUTATION", "oracleId": "private",
                "manaValue": 99, "power": 99, "toughness": 99,
                "rulesText": "private test payload", "tapped": True,
            })
            return True
    return False


def audit_visibility_invariance(report, records):
    config = StructuredJEPAConfig(
        text_dim=32, numeric_dim=40, d_model=16, nhead=4,
        encoder_layers=1, predictor_layers=1, ff_dim=32,
        dropout=0.0, max_objects=64, causal_dim=18,
        horizon_buckets=8, embedding_backend="hash")
    tensorizer = VisibilitySafeTensorizer(config)
    tested = 0
    failures = []
    for index, original in enumerate(records):
        mutated = copy.deepcopy(original)
        if not _mutate_hidden_hand(mutated):
            continue
        tested += 1
        if tensorizer.state_rows(original) != tensorizer.state_rows(mutated):
            failures.append(index)
    status = "fail" if failures else ("pass" if tested else "not-applicable")
    report.add(
        "dataset.hidden-information.perturbation",
        status,
        "Hidden opponent-card mutations do not change model inputs"
        if tested and not failures else
        ("Hidden opponent-card mutations changed model inputs"
         if failures else "No opponent-hidden hand objects were available"),
        {"tested": tested, "failedSamples": failures}, scope="dataset")


def audit_transitions(report, inputs, max_records=0):
    total = 0
    invalid = []
    unknown_game = 0
    sequence_issues = []
    last_sequence = {}
    ordered_games = []
    horizons = Counter()
    truncated = False
    for source_index, item in enumerate(core._iter_all_transitions(inputs)):
        if max_records and total >= max_records:
            truncated = True
            break
        total += 1
        horizon = int(item.get("horizon") or 1)
        horizons[horizon] += 1
        if horizon < 1 or not isinstance(item.get("prev"), dict) or \
                not isinstance(item.get("next"), dict):
            invalid.append(source_index)
        game = transition_game_key(item)
        ordered_games.append(game)
        if game is None:
            unknown_game += 1
        else:
            sequence = transition_sequence(item, source_index)
            previous = last_sequence.get(game)
            if previous is not None and sequence < previous:
                sequence_issues.append({
                    "game": game, "previous": previous, "current": sequence,
                    "record": source_index})
            last_sequence[game] = sequence
    reopened = _sequence_segment_issues(ordered_games)
    report.add(
        "dataset.transitions.structure",
        "fail" if invalid else ("pass" if total else "not-applicable"),
        "Transitions contain valid before/after states and horizons"
        if total and not invalid else
        ("Some transitions are malformed" if invalid else
         "No transition rows were found"),
        {"records": total, "invalidRows": invalid[:20],
         "horizons": dict(sorted(horizons.items())),
         "truncated": truncated}, scope="dataset")
    if total:
        report.add(
            "dataset.transitions.game-identity",
            "warn" if unknown_game else "pass",
            "All transitions have game identities" if not unknown_game else
            "Some transitions cannot participate in game-level splits",
            {"unknownRows": unknown_game}, scope="dataset")
        report.add(
            "dataset.transitions.order",
            "fail" if sequence_issues or reopened else "pass",
            "Transition streams are monotone and game-contiguous"
            if not sequence_issues and not reopened else
            "Transition streams contain out-of-order or reopened games",
            {"nonMonotone": sequence_issues[:20],
             "reopenedGames": reopened[:20]}, scope="dataset")
    return {
        "records": total,
        "games": sorted(key for key in last_sequence if key is not None),
        "truncated": truncated,
    }


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _latest_eval(metrics):
    history = metrics.get("history") or []
    for row in reversed(history):
        evaluation = row.get("eval")
        if isinstance(evaluation, dict) and (
                evaluation.get("examples") or
                evaluation.get("transitionExamples") or
                evaluation.get("stateExamples")):
            return evaluation
    return None


def _tensor_finiteness(value, prefix="stateDict"):
    failures = []
    try:
        import torch
    except ImportError:
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(_tensor_finiteness(child,
                                               "%s.%s" % (prefix, key)) or [])
    elif torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            failures.append(prefix)
    return failures


def _split_details(metrics):
    split = metrics.get("split") or {}
    train_ids = split.get("trainGameIds") or []
    eval_ids = split.get("evalGameIds") or split.get("evalGroupIds") or []
    return split, set(map(str, train_ids)), set(map(str, eval_ids))


def _manifest_files(metrics):
    result = {}
    for item in metrics.get("inputs") or []:
        path = item.get("path")
        if item.get("kind") == "file" and path and item.get("sha256"):
            result[os.path.abspath(path)] = item["sha256"]
        for child in item.get("files") or []:
            if path and child.get("name") and child.get("sha256"):
                result[os.path.abspath(os.path.join(
                    path, child["name"]))] = child["sha256"]
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(report, checkpoint, dataset_games):
    scope = "checkpoint:%s" % os.path.basename(checkpoint)
    path = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(path):
        report.add("checkpoint.exists", "fail", "Checkpoint does not exist",
                   {"path": path}, scope=scope)
        return
    try:
        import torch
    except ImportError:
        report.add(
            "checkpoint.torch-available", "fail",
            "PyTorch is required to audit checkpoint tensors",
            {"path": path}, scope=scope)
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - audit must report bad artifacts
        report.add("checkpoint.load", "fail", "Checkpoint cannot be loaded",
                   {"path": path, "error": str(error)}, scope=scope)
        return
    kind = payload.get("kind") or "torch-option-ranker"
    report.add(
        "checkpoint.load", "pass", "Checkpoint is readable",
        {"path": path, "kind": kind, "bytes": os.path.getsize(path),
         "sha256": _sha256(path)}, scope=scope)
    report.add(
        "checkpoint.kind",
        "pass" if kind in _SUPPORTED_CHECKPOINT_KINDS or
        kind == "torch-option-ranker" else "warn",
        "Checkpoint family is recognized" if kind in _SUPPORTED_CHECKPOINT_KINDS
        or kind == "torch-option-ranker" else
        "Checkpoint family is not covered by model-specific trust gates",
        {"kind": kind}, scope=scope)

    state_dict = payload.get("stateDict") or payload.get("modelStateDict")
    if not isinstance(state_dict, dict):
        report.add("checkpoint.parameters", "fail",
                   "Checkpoint does not contain a state dictionary",
                   {"keys": sorted(payload.keys())}, scope=scope)
    else:
        failures = _tensor_finiteness(state_dict) or []
        parameter_count = sum(int(value.numel()) for value in state_dict.values()
                              if torch.is_tensor(value))
        report.add(
            "checkpoint.parameters", "fail" if failures else "pass",
            "All checkpoint tensors are finite" if not failures else
            "Checkpoint contains non-finite tensors",
            {"parameterCount": parameter_count,
             "nonFinite": failures[:50]}, scope=scope)

    extra = payload.get("extra") or {}
    metrics = extra.get("metrics") if isinstance(extra, dict) else None
    if not isinstance(metrics, dict):
        status = "warn" if kind == "torch-option-ranker" else "fail"
        report.add("checkpoint.metrics", status,
                   "Checkpoint lacks embedded training metrics",
                   {"kind": kind}, scope=scope)
        return
    report.add("checkpoint.metrics", "pass",
               "Checkpoint embeds training metrics",
               {"metricKind": metrics.get("kind")}, scope=scope)

    best_metric = metrics.get("bestSelectionMetric")
    report.add(
        "checkpoint.selection-metric",
        "pass" if _is_finite_number(best_metric) else "fail",
        "Best-selection metric is finite" if _is_finite_number(best_metric)
        else "Best-selection metric is absent or non-finite",
        {"bestEpoch": metrics.get("bestEpoch"),
         "bestSelectionMetric": best_metric}, scope=scope)

    split, train_ids, eval_ids = _split_details(metrics)
    overlap = sorted(train_ids.intersection(eval_ids))
    split_status = "pass"
    if split.get("unit") != "game" or overlap:
        split_status = "fail"
    elif not eval_ids:
        split_status = "warn"
    report.add(
        "checkpoint.split",
        split_status,
        "Checkpoint uses a disjoint whole-game evaluation split"
        if split_status == "pass" else
        ("Checkpoint split leaks games or is not game-level"
         if split_status == "fail" else
         "Checkpoint does not record held-out game identities"),
        {"unit": split.get("unit"), "trainGames": len(train_ids),
         "evalGames": len(eval_ids), "overlap": overlap[:20]}, scope=scope)
    if dataset_games and eval_ids:
        missing = sorted(eval_ids.difference(dataset_games))
        report.add(
            "checkpoint.split.dataset-alignment",
            "warn" if missing else "pass",
            "Held-out game IDs occur in the supplied dataset"
            if not missing else
            "Some checkpoint evaluation IDs are absent from supplied data",
            {"missingEvalGames": missing[:50]}, scope=scope)

    manifests = _manifest_files(metrics)
    existing = {path: expected for path, expected in manifests.items()
                if os.path.isfile(path)}
    mismatched = [path for path, expected in existing.items()
                  if _sha256(path) != expected]
    manifest_status = "pass" if existing and not mismatched else \
        ("fail" if mismatched else "warn")
    report.add(
        "checkpoint.input-provenance", manifest_status,
        "Recorded input hashes match available files"
        if manifest_status == "pass" else
        ("Recorded input hashes do not match current files"
         if mismatched else
         "No recorded input files are currently available for hash verification"),
        {"recordedFiles": len(manifests), "availableFiles": len(existing),
         "mismatched": mismatched[:20]}, scope=scope)

    latest = _latest_eval(metrics)
    if kind in ("magic-recurrent-information-state-v1",
                "magic-belief-information-state-v1",
                "magic-structured-rssm-v1"):
        report.add(
            "checkpoint.visibility-policy",
            "pass" if metrics.get("visibilityPolicy") ==
            "public-history-and-perspective-state-v1" else "fail",
            "Checkpoint records the visibility-safe input policy"
            if metrics.get("visibilityPolicy") ==
            "public-history-and-perspective-state-v1" else
            "Checkpoint does not attest the required visibility policy",
            {"visibilityPolicy": metrics.get("visibilityPolicy")}, scope=scope)
        collection = metrics.get("collection") or {}
        report.add(
            "checkpoint.complete-game-collection",
            "pass" if collection.get("unit") == "complete-game" else "warn",
            "Training collection preserved complete games"
            if collection.get("unit") == "complete-game" else
            "Checkpoint does not record complete-game collection",
            collection, scope=scope)

    if kind == "magic-structured-jepa-v1":
        required = ("jepa", "causal", "policy", "collapse")
        missing = [key for key in required
                   if not latest or key not in latest]
        report.add(
            "checkpoint.jepa-diagnostics",
            "fail" if missing else "pass",
            "Held-out JEPA and collapse diagnostics are present"
            if not missing else "Held-out JEPA diagnostics are incomplete",
            {"missing": missing}, scope=scope)
    elif kind == "magic-recurrent-information-state-v1":
        required = ("loss", "policyTop1", "policyTop3", "policyMRR")
        missing = [key for key in required
                   if not latest or latest.get(key) is None]
        report.add(
            "checkpoint.recurrent-diagnostics",
            "fail" if missing else "pass",
            "Held-out recurrent policy diagnostics are present"
            if not missing else "Held-out recurrent diagnostics are incomplete",
            {"missing": missing}, scope=scope)
    elif kind == "magic-belief-information-state-v1":
        calibration = (latest or {}).get("calibration") or {}
        aggregate = calibration.get("aggregate") or {}
        missing = [key for key in
                   ("brier", "logLoss", "expectedCalibrationError")
                   if aggregate.get(key) is None]
        vocabulary = metrics.get("vocabulary") or {}
        if not vocabulary.get("sha256"):
            missing.append("vocabulary.sha256")
        report.add(
            "checkpoint.belief-calibration",
            "fail" if missing else "pass",
            "Held-out calibrated belief diagnostics are present"
            if not missing else "Belief calibration/provenance is incomplete",
            {"missing": missing,
             "labeledCells": (latest or {}).get("beliefCells")}, scope=scope)
    elif kind == "magic-structured-rssm-v1":
        collapse = (latest or {}).get("collapse") or {}
        required_values = {
            "priorNll": (latest or {}).get("priorNll"),
            "standardizedResidualRms":
                (latest or {}).get("standardizedResidualRms"),
            "openLoopMseByHorizon":
                (latest or {}).get("openLoopMseByHorizon"),
            "collapse.effectiveRank": collapse.get("effectiveRank"),
        }
        missing = [key for key, value in required_values.items()
                   if value is None or value == {}]
        report.add(
            "checkpoint.rssm-diagnostics",
            "fail" if missing else "pass",
            "Held-out stochastic rollout diagnostics are present"
            if not missing else "RSSM rollout/collapse diagnostics are incomplete",
            {"missing": missing, "values": required_values}, scope=scope)


def audit(inputs, checkpoints=None, strict=False, max_records=0,
          visibility_samples=32):
    checkpoints = checkpoints or []
    report = AuditReport(strict=strict)
    decisions = audit_decisions(
        report, inputs, max_records=max_records,
        visibility_samples=visibility_samples)
    transitions = audit_transitions(
        report, inputs, max_records=max_records)
    games = set(decisions["games"]).union(transitions["games"])
    for checkpoint in checkpoints:
        audit_checkpoint(report, checkpoint, games)
    if not checkpoints:
        report.add("checkpoint.present", "not-applicable",
                   "No checkpoints were supplied", scope="checkpoint")
    return report.finish(inputs, checkpoints)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magic-cabt-training-audit",
        description="Fail-fast trust audit for training data and checkpoints.")
    parser.add_argument("--input", action="append", required=True,
                        help="bundle directory or JSONL file; repeatable")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="checkpoint to audit; repeatable")
    parser.add_argument("--out", required=True,
                        help="machine-readable JSON report")
    parser.add_argument("--strict", action="store_true",
                        help="promote warnings to failures")
    parser.add_argument("--max-records", type=int, default=0,
                        help="0 audits all records")
    parser.add_argument("--visibility-samples", type=int, default=32)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = audit(
        args.input, checkpoints=args.checkpoint, strict=args.strict,
        max_records=max(0, args.max_records),
        visibility_samples=max(0, args.visibility_samples))
    output = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["trusted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
