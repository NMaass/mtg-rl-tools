"""Shared primitives for the training trust audit."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

SCHEMA_VERSION = 1
FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "traininglabels", "oraclelabels", "belieflabels", "trueopponenthand",
    "opponenthandtruth", "hiddenstatetruth", "oraclehiddenstate",
    "privatehand", "enginehiddenstate", "fullopponentdeck", "oraclevalue",
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

    def finish(self, inputs, checkpoints, files):
        counts = Counter(check["status"] for check in self.checks)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "magic-training-trust-audit-v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "strict": self.strict,
            "inputs": [os.path.abspath(os.path.expanduser(path)) for path in inputs],
            "checkpoints": [
                os.path.abspath(os.path.expanduser(path)) for path in checkpoints
            ],
            "files": files,
            "summary": {
                "pass": counts["pass"],
                "warn": counts["warn"],
                "fail": counts["fail"],
                "notApplicable": counts["not-applicable"],
                "trusted": counts["fail"] == 0,
            },
            "checks": self.checks,
        }


def normalize_key(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def forbidden_paths(value, prefix="observation"):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = "%s.%s" % (prefix, key)
            if normalize_key(key) in FORBIDDEN_OBSERVATION_KEYS:
                paths.append(path)
            paths.extend(forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(forbidden_paths(child, "%s[%d]" % (prefix, index)))
    return paths


def select_of(record):
    direct = record.get("select") if isinstance(record, dict) else None
    if isinstance(direct, dict):
        return direct
    observation = record.get("observation") if isinstance(record, dict) else None
    nested = observation.get("select") if isinstance(observation, dict) else None
    return nested if isinstance(nested, dict) else {}


def selected_of(record):
    selected = record.get("selectedIndices") if isinstance(record, dict) else None
    if selected is None and isinstance(record.get("select"), list):
        selected = record.get("select")
    if selected is None:
        selected = record.get("selected")
    return selected if isinstance(selected, list) else []


def current_of(record):
    observation = record.get("observation") if isinstance(record, dict) else None
    current = observation.get("current") if isinstance(observation, dict) else None
    return current if isinstance(current, dict) else {}


def game_key(record):
    current = current_of(record)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    match = (record.get("gameId") or record.get("matchId") or
             metadata.get("matchId") or current.get("matchId"))
    game = record.get("gameNumber")
    if game is None:
        game = metadata.get("gameNumber")
    if game is None:
        game = current.get("gameNumber")
    instance = current.get("gameInstance")
    if match is None and game is None and instance is None:
        return None
    return json.dumps([match, game, instance], separators=(",", ":"))


def sequence_of(record, fallback):
    current = current_of(record)
    value = record.get("sequenceNumber", record.get("sequence"))
    if value is None:
        value = current.get("seq")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else fallback


def reopened_segments(keys):
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files(inputs):
    files = []
    seen = set()
    for supplied in inputs:
        path = os.path.abspath(os.path.expanduser(supplied))
        candidates = []
        if os.path.isdir(path):
            for name in ("decisions.jsonl", "transitions.jsonl",
                         "macro_actions.jsonl"):
                candidate = os.path.join(path, name)
                if os.path.isfile(candidate):
                    candidates.append(candidate)
        elif os.path.isfile(path):
            candidates.append(path)
        for candidate in candidates:
            if candidate not in seen:
                files.append(candidate)
                seen.add(candidate)
    return files


def classify_file(path):
    """Return ``(kind, error)`` for one JSONL input."""
    name = os.path.basename(path).lower()
    expected = None
    if name == "decisions.jsonl":
        expected = "decisions"
    elif "transition" in name:
        expected = "transitions"
    elif "macro_action" in name or "macro-action" in name:
        expected = "macro-actions"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    return "invalid", "line %d is not a JSON object" % line_number
                if value.get("recordType") == "MacroActionRecord" or \
                        "macroAction" in value or "orphanedParameterGroup" in value:
                    detected = "macro-actions"
                elif isinstance(value.get("prev"), dict) and \
                        isinstance(value.get("next"), dict):
                    detected = "transitions"
                elif "observation" in value or "select" in value or \
                        "selectedIndices" in value or "selected" in value:
                    detected = "decisions"
                else:
                    return "unknown", "first object has no recognized record shape"
                if expected is not None and detected != expected:
                    return "invalid", "filename implies %s but first object is %s" % (
                        expected, detected)
                return detected, None
    except (OSError, ValueError) as error:
        return "invalid", str(error)
    return "empty", "file has no JSON objects"


def files_of_kind(inputs, expected):
    return [path for path in input_files(inputs)
            if classify_file(path)[0] == expected]


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError("%s:%d: row is not an object" %
                                 (path, line_number))
            value.setdefault("__source_line", line_number)
            yield value
