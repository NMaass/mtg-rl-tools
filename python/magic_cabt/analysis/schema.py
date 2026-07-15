"""Versioned, checkpoint-aware replay analysis records."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 1
_ID_KEYS = frozenset(("id", "objectId", "instanceId", "targetId",
                     "targetInstanceId", "sourceId", "eventId", "requestId"))
_VOLATILE_KEYS = frozenset(("timestamp", "rawTime"))
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I)


def fingerprint(value):
    payload = json.dumps(_canonical(value), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decision_fingerprint(record):
    observation = record.get("observation") or {}
    current = observation.get("current") or record.get("current") or {}
    select = record.get("select") if isinstance(record.get("select"), dict) \
        else observation.get("select") or {}
    return fingerprint({"current": current, "select": select})


def analysis_cache_key(record, model_info):
    return "%s|%s|%s" % (
        SCHEMA_VERSION,
        decision_fingerprint(record),
        model_info.get("checkpointId") or model_info.get("modelId") or "unknown")


def make_analysis_record(record, scores, model_info, top_k=5,
                         latency_ms=None, source="live", value=None,
                         causal_factors=None, predicted_consequences=None):
    observation = record.get("observation") or {}
    current = observation.get("current") or record.get("current") or {}
    select = record.get("select") if isinstance(record.get("select"), dict) \
        else observation.get("select") or {}
    options = select.get("option") or []
    if len(scores) != len(options):
        raise ValueError("scorer returned %d scores for %d options" %
                         (len(scores), len(options)))
    numeric = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric):
        raise ValueError("scorer returned non-finite option score")
    probabilities = _softmax(numeric)
    ranking = sorted(range(len(options)), key=lambda index: (-numeric[index], index))
    chosen = record.get("selectedIndices")
    if chosen is None and isinstance(record.get("select"), list):
        chosen = record.get("select")
    chosen = chosen or record.get("selected") or []
    ranks = {index: position + 1 for position, index in enumerate(ranking)}
    first = chosen[0] if chosen and isinstance(chosen[0], int) else None
    top = []
    limit = len(ranking) if not top_k or top_k < 1 else min(top_k, len(ranking))
    for index in ranking[:limit]:
        option = options[index]
        payload = option.get("payload") or {}
        top.append({
            "optionIndex": index,
            "canonicalGroupKey": payload.get("canonicalKey")
                if isinstance(payload, dict) else None,
            "label": option.get("label") or option.get("type") or str(index),
            "score": numeric[index],
            "probability": probabilities[index],
            "value": None,
            "causalFactors": None,
            "predictedConsequences": None,
        })
    model = dict(model_info or {})
    return {
        "schemaVersion": SCHEMA_VERSION,
        "analysisId": fingerprint({
            "decision": decision_fingerprint(record),
            "model": model,
            "created": time.time_ns(),
        }),
        "gameId": record.get("gameId"),
        "matchId": record.get("matchId"),
        "gameNumber": record.get("gameNumber"),
        "gameInstance": current.get("gameInstance"),
        "sequenceNumber": record.get("sequenceNumber", record.get("sequence")),
        "stateSequence": current.get("seq"),
        "decisionFingerprint": decision_fingerprint(record),
        "perspectiveSeat": current.get("localSeat") or current.get("perspectiveSeat"),
        "model": model,
        "analysis": {
            "topK": top,
            "chosenIndices": chosen,
            "chosenRank": ranks.get(first),
            "chosenScore": numeric[first]
                if isinstance(first, int) and 0 <= first < len(numeric) else None,
            "value": value,
            "causalFactors": causal_factors,
            "predictedConsequences": predicted_consequences,
        },
        "latencyMs": latency_ms,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }


def format_analysis(record, max_rows=None):
    model = record.get("model") or {}
    analysis = record.get("analysis") or {}
    title = model.get("modelId") or "model"
    if model.get("trainingState") == "untrained":
        title += " (untrained)"
    lines = ["Model analysis · %s" % title]
    rows = analysis.get("topK") or []
    if max_rows:
        rows = rows[:max_rows]
    for position, row in enumerate(rows, start=1):
        probability = row.get("probability")
        suffix = "%.1f%%" % (100.0 * probability) if probability is not None \
            else "%.4f" % row.get("score", 0.0)
        lines.append("%d. %s  %s" % (position, row.get("label"), suffix))
    # Latency stays in the record but out of the readout — a viewer cares
    # whether the play matched the model, not how long inference took.
    rank = analysis.get("chosenRank")
    if rank == 1:
        lines.append("✓ Played rank: 1 — the model's top pick")
    elif rank is not None:
        top = (analysis.get("topK") or [{}])[0].get("label")
        lines.append("✗ Played rank: %s — model preferred %s" % (rank, top)
                     if top else "✗ Played rank: %s" % rank)
    return "\n".join(lines)


def _softmax(values):
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())
                if key not in _ID_KEYS and key not in _VOLATILE_KEYS
                and key not in ("raw", "rawPayload", "payloadRaw")}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return _UUID_RE.sub("", value)
    return value
