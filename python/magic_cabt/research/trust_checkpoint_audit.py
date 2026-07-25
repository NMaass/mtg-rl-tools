"""Optional PyTorch checkpoint gates for the training trust audit."""
from __future__ import annotations

import math
import os

from .trust_audit_common import sha256_file

CHECKPOINT_EVIDENCE = {
    "magic-structured-jepa-v1": {
        "check": "checkpoint.jepa-diagnostics",
        "eval": ("jepa", "causal", "policy", "collapse"),
    },
    "magic-recurrent-information-state-v1": {
        "check": "checkpoint.recurrent-diagnostics",
        "eval": ("loss", "policyTop1", "policyTop3", "policyMRR"),
        "visibility": True,
    },
    "magic-belief-information-state-v1": {
        "check": "checkpoint.belief-calibration",
        "evalNested": (
            ("calibration", "aggregate", "brier"),
            ("calibration", "aggregate", "logLoss"),
            ("calibration", "aggregate", "expectedCalibrationError"),
        ),
        "metricsNested": (("vocabulary", "sha256"),),
        "visibility": True,
    },
    "magic-structured-rssm-v1": {
        "check": "checkpoint.rssm-diagnostics",
        "eval": ("priorNll", "standardizedResidualRms", "openLoopMseByHorizon"),
        "evalNested": (("collapse", "effectiveRank"),),
        "visibility": True,
    },
    "torch-option-ranker": {
        "check": "checkpoint.ranker-diagnostics",
        "eval": (),
    },
}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(float(value))


def _latest_eval(metrics):
    for row in reversed(metrics.get("history") or []):
        evaluation = row.get("eval") if isinstance(row, dict) else None
        if isinstance(evaluation, dict):
            return evaluation
    return None


def _nested(value, path):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _non_finite_tensors(value, torch, prefix="stateDict"):
    failures = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(_non_finite_tensors(
                child, torch, "%s.%s" % (prefix, key)))
    elif torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        failures.append(prefix)
    return failures


def _split_details(metrics):
    split = metrics.get("split") or {}
    train_ids = set(map(str, split.get("trainGameIds") or []))
    eval_ids = set(map(str, split.get("evalGameIds") or
                       split.get("evalGroupIds") or []))
    return split, train_ids, eval_ids


def _manifest_files(metrics):
    result = {}
    for item in metrics.get("inputs") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if item.get("kind") == "file" and path and item.get("sha256"):
            result[os.path.abspath(path)] = item["sha256"]
        for child in item.get("files") or []:
            if path and child.get("name") and child.get("sha256"):
                result[os.path.abspath(os.path.join(
                    path, child["name"]))] = child["sha256"]
    return result


def audit_checkpoint(report, checkpoint, dataset_games,
                     require_checkpoint_audit=False):
    scope = "checkpoint:%s" % os.path.basename(checkpoint)
    path = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(path):
        report.add("checkpoint.exists", "fail", "Checkpoint does not exist",
                   {"path": path}, scope=scope)
        return
    try:
        import torch
    except ImportError:
        status = "fail" if require_checkpoint_audit else "warn"
        report.add(
            "checkpoint.torch-available", status,
            "PyTorch is required to inspect checkpoint tensors",
            {"path": path, "required": bool(require_checkpoint_audit)},
            scope=scope)
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        report.add("checkpoint.load", "fail", "Checkpoint cannot be loaded",
                   {"path": path, "error": str(error)}, scope=scope)
        return
    if not isinstance(payload, dict):
        report.add("checkpoint.load", "fail",
                   "Checkpoint payload is not a mapping",
                   {"path": path, "type": type(payload).__name__}, scope=scope)
        return

    kind = payload.get("kind") or "torch-option-ranker"
    report.add(
        "checkpoint.load", "pass", "Checkpoint is readable",
        {"path": path, "kind": kind, "bytes": os.path.getsize(path),
         "sha256": sha256_file(path)}, scope=scope)
    report.add(
        "checkpoint.kind", "pass" if kind in CHECKPOINT_EVIDENCE else "warn",
        "Checkpoint family is recognized" if kind in CHECKPOINT_EVIDENCE else
        "Checkpoint family has no model-specific evidence registry",
        {"kind": kind}, scope=scope)

    state_dict = payload.get("stateDict") or payload.get("modelStateDict")
    if not isinstance(state_dict, dict):
        report.add("checkpoint.parameters", "fail",
                   "Checkpoint does not contain a state dictionary",
                   {"keys": sorted(payload.keys())}, scope=scope)
    else:
        failures = _non_finite_tensors(state_dict, torch)
        parameter_count = sum(
            int(value.numel()) for value in state_dict.values()
            if torch.is_tensor(value))
        report.add(
            "checkpoint.parameters", "fail" if failures else "pass",
            "All checkpoint tensors are finite" if not failures else
            "Checkpoint contains non-finite tensors",
            {"parameterCount": parameter_count,
             "nonFinite": failures[:50]}, scope=scope)

    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    metrics = extra.get("metrics") or payload.get("metrics")
    if not isinstance(metrics, dict):
        report.add(
            "checkpoint.metrics",
            "warn" if kind == "torch-option-ranker" else "fail",
            "Checkpoint lacks embedded training metrics",
            {"kind": kind}, scope=scope)
        return
    report.add("checkpoint.metrics", "pass",
               "Checkpoint embeds training metrics",
               {"metricKind": metrics.get("kind")}, scope=scope)

    best_metric = metrics.get("bestSelectionMetric")
    report.add(
        "checkpoint.selection-metric",
        "pass" if _finite_number(best_metric) else "warn",
        "Best-selection metric is finite" if _finite_number(best_metric) else
        "Best-selection metric is absent or non-finite",
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
        "checkpoint.split", split_status,
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
            "Held-out game IDs occur in the supplied dataset" if not missing else
            "Some checkpoint evaluation IDs are absent from supplied data",
            {"missingEvalGames": missing[:50]}, scope=scope)

    manifests = _manifest_files(metrics)
    existing = {path: expected for path, expected in manifests.items()
                if os.path.isfile(path)}
    mismatched = [path for path, expected in existing.items()
                  if sha256_file(path) != expected]
    provenance = "pass" if existing and not mismatched else \
        ("fail" if mismatched else "warn")
    report.add(
        "checkpoint.input-provenance", provenance,
        "Recorded input hashes match available files"
        if provenance == "pass" else
        ("Recorded input hashes do not match current files"
         if mismatched else
         "No recorded input files are available for hash verification"),
        {"recordedFiles": len(manifests), "availableFiles": len(existing),
         "mismatched": mismatched[:20]}, scope=scope)

    evidence = CHECKPOINT_EVIDENCE.get(kind)
    if evidence is None:
        return
    if evidence.get("visibility"):
        policy = metrics.get("visibilityPolicy")
        report.add(
            "checkpoint.visibility-policy",
            "pass" if policy == "public-history-and-perspective-state-v1"
            else "fail",
            "Checkpoint records the visibility-safe input policy"
            if policy == "public-history-and-perspective-state-v1" else
            "Checkpoint does not attest the required visibility policy",
            {"visibilityPolicy": policy}, scope=scope)
    latest = _latest_eval(metrics) or {}
    missing = [key for key in evidence.get("eval", ())
               if latest.get(key) is None]
    missing.extend("eval." + ".".join(path)
                   for path in evidence.get("evalNested", ())
                   if _nested(latest, path) is None)
    missing.extend("metrics." + ".".join(path)
                   for path in evidence.get("metricsNested", ())
                   if _nested(metrics, path) is None)
    report.add(
        evidence["check"], "fail" if missing else "pass",
        "Required held-out evidence is present" if not missing else
        "Required held-out evidence is incomplete",
        {"kind": kind, "missing": missing}, scope=scope)
