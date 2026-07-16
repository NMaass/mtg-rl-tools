"""Additional hard gates applied to the base training trust audit."""
from __future__ import annotations

import math
import os
from collections import Counter


_REQUIRED_NUMERIC_PATHS = {
    "magic-recurrent-information-state-v1": (
        "loss", "policyTop1", "policyTop3", "policyMRR"),
    "magic-belief-information-state-v1": (
        "loss", "policyLoss", "beliefLoss",
        "calibration.aggregate.brier",
        "calibration.aggregate.logLoss",
        "calibration.aggregate.expectedCalibrationError"),
    "magic-structured-rssm-v1": (
        "loss", "reconstruction", "oneStepPrediction", "kl",
        "priorNll", "standardizedResidualRms",
        "collapse.meanDimensionStd", "collapse.effectiveRank"),
}


def safe_load_context():
    """Return a context manager forcing PyTorch's restricted loader."""
    return _SafeTorchLoad()


class _SafeTorchLoad:
    def __enter__(self):
        try:
            import torch
        except ImportError:
            self.torch = None
            self.original = None
            return self
        self.torch = torch
        self.original = torch.load

        def restricted_load(*args, **kwargs):
            kwargs["weights_only"] = True
            return self.original(*args, **kwargs)

        torch.load = restricted_load
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.torch is not None and self.original is not None:
            self.torch.load = self.original
        return False


def harden_report(result, checkpoints):
    """Add gates omitted by compatibility-oriented base checks."""
    strict = bool(result.get("strict"))
    for checkpoint in checkpoints:
        _harden_checkpoint(result, checkpoint, strict)
    _recompute_summary(result)
    return result


def _harden_checkpoint(result, checkpoint, strict):
    path = os.path.abspath(os.path.expanduser(checkpoint))
    scope = "checkpoint:%s" % os.path.basename(path)
    if not os.path.isfile(path):
        return
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - report untrusted artifacts
        _add(result, strict, "checkpoint.restricted-load", "fail",
             "Checkpoint cannot be loaded by PyTorch's restricted loader",
             {"error": str(error)}, scope)
        return
    _add(result, strict, "checkpoint.restricted-load", "pass",
         "Checkpoint loads with weights_only=True",
         {"kind": payload.get("kind") or "torch-option-ranker"}, scope)

    metrics = ((payload.get("extra") or {}).get("metrics")
               if isinstance(payload, dict) else None)
    if not isinstance(metrics, dict):
        return
    kind = payload.get("kind") or "torch-option-ranker"
    evaluation = _latest_eval(metrics)

    required = _REQUIRED_NUMERIC_PATHS.get(kind, ())
    failures = {}
    for dotted in required:
        value = _get(evaluation or {}, dotted)
        if not _finite(value):
            failures[dotted] = value
    if kind == "magic-structured-rssm-v1":
        rollout = (evaluation or {}).get("openLoopMseByHorizon") or {}
        if not rollout:
            failures["openLoopMseByHorizon"] = rollout
        else:
            for horizon, value in rollout.items():
                if not _finite(value):
                    failures["openLoopMseByHorizon.%s" % horizon] = value
    if required:
        _add(result, strict, "checkpoint.finite-diagnostics",
             "fail" if failures else "pass",
             "Required held-out diagnostics are finite"
             if not failures else
             "Required held-out diagnostics are missing or non-finite",
             {"failures": failures}, scope)

    split = metrics.get("split") or {}
    train_ids = split.get("trainGameIds") or []
    eval_ids = split.get("evalGameIds") or split.get("evalGroupIds") or []
    sequential = kind in {
        "magic-recurrent-information-state-v1",
        "magic-belief-information-state-v1",
        "magic-structured-rssm-v1",
    }
    missing_split_ids = []
    if not eval_ids:
        missing_split_ids.append("evalGameIds")
    if sequential and not train_ids:
        missing_split_ids.append("trainGameIds")
    _add(result, strict, "checkpoint.split-identities",
         "fail" if missing_split_ids else "pass",
         "Checkpoint records complete train/evaluation game identities"
         if not missing_split_ids else
         "Checkpoint split identities are incomplete",
         {"missing": missing_split_ids,
          "trainGames": len(train_ids), "evalGames": len(eval_ids)}, scope)

    if sequential:
        collection = metrics.get("collection") or {}
        failures = []
        if collection.get("unit") != "complete-game":
            failures.append("unit")
        if collection.get("contiguousGamesRequired") is not True:
            failures.append("contiguousGamesRequired")
        _add(result, strict, "checkpoint.sequence-collection-contract",
             "fail" if failures else "pass",
             "Collection required complete contiguous games"
             if not failures else
             "Collection does not attest complete contiguous games",
             {"missingOrInvalid": failures, "collection": collection}, scope)

    recorded, available, mismatched = _provenance_counts(metrics)
    if mismatched:
        status = "fail"
        summary = "Available source files do not match recorded hashes"
    elif not recorded:
        status = "warn"
        summary = "Checkpoint records no input-file hashes"
    elif available < recorded:
        status = "warn"
        summary = "Only part of the recorded input manifest is available"
    else:
        status = "pass"
        summary = "All recorded input files are available and hash-matched"
    _add(result, strict, "checkpoint.provenance-completeness", status,
         summary, {"recordedFiles": recorded,
                   "availableFiles": available,
                   "mismatchedFiles": mismatched}, scope)


def _latest_eval(metrics):
    for row in reversed(metrics.get("history") or []):
        evaluation = row.get("eval")
        if isinstance(evaluation, dict) and any(
                evaluation.get(key) for key in
                ("examples", "transitionExamples", "stateExamples")):
            return evaluation
    return None


def _get(value, dotted):
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _provenance_counts(metrics):
    recorded = []
    for item in metrics.get("inputs") or []:
        path = item.get("path")
        if item.get("kind") == "file" and path and item.get("sha256"):
            recorded.append((os.path.abspath(path), item["sha256"]))
        for child in item.get("files") or []:
            if path and child.get("name") and child.get("sha256"):
                recorded.append((os.path.abspath(os.path.join(
                    path, child["name"])), child["sha256"]))
    available = 0
    mismatched = []
    for path, expected in recorded:
        if not os.path.isfile(path):
            continue
        available += 1
        if _sha256(path) != expected:
            mismatched.append(path)
    return len(recorded), available, mismatched


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add(result, strict, identifier, status, summary, details, scope):
    effective = "fail" if strict and status == "warn" else status
    result.setdefault("checks", []).append({
        "id": identifier,
        "scope": scope,
        "status": effective,
        "originalStatus": status if effective != status else None,
        "summary": summary,
        "details": details,
    })


def _recompute_summary(result):
    counts = Counter(row.get("status") for row in result.get("checks") or [])
    result["summary"] = {
        "pass": counts["pass"],
        "warn": counts["warn"],
        "fail": counts["fail"],
        "notApplicable": counts["not-applicable"],
        "trusted": counts["fail"] == 0,
    }
