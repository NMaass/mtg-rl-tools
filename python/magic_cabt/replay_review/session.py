"""A single paid request in flight, a coalesced next request, and session-only caching."""

import copy
import threading

from . import client
from .data import PROMPT_VERSION, ReviewPoint


class ReviewSession:
    def __init__(self, analyzer=client.analyze):
        self._analyzer = analyzer
        self._condition = threading.Condition()
        self._api_key = ""
        self._closed = False
        self._pending = None
        self._active = None
        self._results = {}
        self._calls = []
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def connect(self, key):
        key = key.strip()
        if not key or len(key) > 2048 or any(c.isspace() for c in key):
            raise ValueError("Paste a valid OpenRouter key without whitespace.")
        with self._condition:
            self._api_key = key
            self._pending = None

    @property
    def connected(self):
        with self._condition:
            return bool(self._api_key) and not self._closed

    def disconnect(self):
        with self._condition:
            self._api_key = ""
            self._pending = None

    def cancel_pending(self):
        with self._condition:
            self._pending = None

    def request(self, point: ReviewPoint, retry=False):
        if point.request is None:
            return
        fingerprint = point.fingerprint
        with self._condition:
            if self._closed or not self._api_key or self._active == fingerprint:
                return
            previous = self._results.get(fingerprint)
            if previous is not None and not (retry and previous["status"] == "error"):
                return
            if previous is not None:
                del self._results[fingerprint]
            self._pending = (fingerprint, copy.deepcopy(point.request))
            self._condition.notify()

    def status(self, point: ReviewPoint):
        with self._condition:
            if point.request is None:
                return "unsupported", None
            if point.fingerprint in self._results:
                result = copy.deepcopy(self._results[point.fingerprint])
                return result["status"], result
            if self._active == point.fingerprint:
                return "loading", None
            if self._pending and self._pending[0] == point.fingerprint:
                return "queued", None
            return "idle", None

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                fingerprint, request = self._pending
                self._pending = None
                self._active = fingerprint
                key = self._api_key
            try:
                result = self._analyzer(request, key)
            except Exception:
                result = {"status": "error", "error": "Unexpected analysis failure; no automatic retry.",
                          "costUsd": None, "latencyMs": None}
            finally:
                key = ""
            with self._condition:
                self._calls.append(dict(result, requestHash=fingerprint))
                self._results[fingerprint] = result
                self._active = None
                self._condition.notify_all()

    def summary(self):
        with self._condition:
            calls = copy.deepcopy(self._calls)
            return {"calls": len(calls),
                    "knownCostUsd": sum(c["costUsd"] for c in calls if c.get("costUsd") is not None),
                    "unknownCostCalls": sum(c.get("costUsd") is None for c in calls),
                    "failedCalls": sum(c["status"] == "error" for c in calls)}

    def report(self, points, feedback):
        with self._condition:
            rows = []
            scored = agrees = 0
            for point in points:
                result = self._results.get(point.fingerprint) if point.request else None
                if result and result["status"] == "complete" and point.recorded_choice:
                    scored += 1
                    agrees += result["choice"] == point.recorded_choice
                rows.append({"point": point.caption, "requestHash": point.fingerprint,
                             "input": point.request, "recordedChoice": point.recorded_choice,
                             "issue": point.issue, "warnings": point.warnings,
                             "analysis": copy.deepcopy(result),
                             "humanRating": feedback.get(point.key)})
            return {"schemaVersion": 1, "promptVersion": PROMPT_VERSION,
                    "interpretation": "Post-step recommendations, not live forecasts or verified correct moves.",
                    "agreement": {"matched": agrees, "scored": scored},
                    "usage": self.summary(), "calls": copy.deepcopy(self._calls), "decisions": rows}

    def close(self):
        with self._condition:
            self._closed = True
            self._api_key = ""
            self._pending = None
            self._condition.notify_all()
