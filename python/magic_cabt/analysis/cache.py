"""Append-only replay analysis cache."""
from __future__ import annotations

import json
import os
import threading


class AnalysisCache:
    def __init__(self, path=None):
        self.path = path
        self._lock = threading.Lock()
        self._records = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    key = self.key_of(record)
                    if key:
                        self._records[key] = record

    @staticmethod
    def key_of(record):
        model = record.get("model") or {}
        checkpoint = model.get("checkpointId") or model.get("modelId")
        decision = record.get("decisionFingerprint")
        version = record.get("schemaVersion")
        return "%s|%s|%s" % (version, decision, checkpoint) \
            if decision and checkpoint else None

    def get(self, key):
        with self._lock:
            return self._records.get(key)

    def add(self, record, persist=False):
        key = self.key_of(record)
        if not key:
            raise ValueError("analysis record lacks cache identity")
        with self._lock:
            self._records[key] = record
            if persist and self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True,
                                            separators=(",", ":")) + "\n")
        return record
