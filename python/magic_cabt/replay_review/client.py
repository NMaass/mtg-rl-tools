"""Jev's typed Decisions endpoint; no chat fallback, redirects, or paid retries."""

import http.client
import json
import math
import time
from datetime import datetime, timezone

from .data import canonical

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 1024 * 1024
ERRORS = {
    400: "OpenRouter rejected the Decisions request. Inspect the input and API contract.",
    401: "OpenRouter rejected the key. Replace it in the replay library.",
    402: "OpenRouter credits are exhausted. Add credits or use another key.",
    403: "This key cannot access Jev. Check its model permissions.",
    404: "Jev or the Decisions endpoint is unavailable. No fallback model was called.",
    413: "The provider rejected the state as too large. No choices were truncated.",
    429: "OpenRouter rate-limited the request. Retry explicitly when ready.",
}


class RequestFailure(Exception):
    def __init__(self, status=None):
        self.status = status
        super().__init__("Decisions request failed")


def post_decision(payload, api_key):
    connection = http.client.HTTPSConnection("openrouter.ai", timeout=TIMEOUT_SECONDS)
    try:
        connection.request("POST", "/api/alpha/decisions",
                           body=canonical(payload).encode("utf-8"),
                           headers={"Authorization": "Bearer " + api_key,
                                    "Content-Type": "application/json",
                                    "X-OpenRouter-Title": "MTG Replay Review"})
        response = connection.getresponse()
        if response.status != 200:
            raise RequestFailure(response.status)
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise RequestFailure()
        return json.loads(body)
    finally:
        connection.close()


def _number(value, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def validate_answer(response, option_ids):
    if not isinstance(response, dict):
        raise ValueError("No Decisions response object.")
    answer = (response.get("answers") or {}).get("action")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("No typed action choice in the Decisions response.")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(option_ids):
        raise ValueError("Returned probabilities do not match the captured action space.")
    if any(_number(p, 1) is None for p in probabilities.values()):
        raise ValueError("Returned probabilities must be finite values from zero to one.")
    if abs(sum(probabilities.values()) - 1) > 0.001:
        raise ValueError("Returned probabilities do not sum to one.")
    choice = answer.get("choice")
    if choice not in option_ids:
        raise ValueError("Jev returned an action outside the captured action space.")
    confidence = answer.get("confidence")
    if confidence is not None and _number(confidence, 1) is None:
        raise ValueError("Invalid provider confidence value.")
    return {"choice": choice, "probabilities": probabilities, "confidence": confidence}


def analyze(payload, api_key, transport=post_decision):
    started = time.perf_counter()
    result = {"status": "error", "requestedModel": payload["model"],
              "resolvedModel": None, "provider": None, "generationId": None,
              "costUsd": None, "inputTokens": None, "outputTokens": None,
              "createdAt": datetime.now(timezone.utc).isoformat(),
              "error": None}
    try:
        if not api_key or any(c.isspace() for c in api_key) or len(api_key) > 2048:
            raise ValueError("Paste a valid OpenRouter key without whitespace.")
        response = transport(payload, api_key)
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            if isinstance(usage, dict):
                result["costUsd"] = _number(usage.get("cost"))
                for source, target in (("input_tokens", "inputTokens"),
                                       ("output_tokens", "outputTokens")):
                    value = usage.get(source)
                    result[target] = value if type(value) is int and value >= 0 else None
            for source, target in (("model", "resolvedModel"), ("provider", "provider"),
                                   ("id", "generationId")):
                value = response.get(source)
                if isinstance(value, str):
                    result[target] = value.replace(api_key, "[redacted]")[:256]
        result.update(validate_answer(response, payload["questions"]["action"]["criteria"]))
        result["status"] = "complete"
    except RequestFailure as exc:
        result["error"] = ERRORS.get(exc.status, "Provider request failed. No automatic retry was sent.")
    except (OSError, http.client.HTTPException, TimeoutError):
        result["error"] = "Network error or timeout. Billing may be unknown; retry only explicitly."
    except (ValueError, TypeError, AttributeError, KeyError):
        result["error"] = "Invalid Decisions response or key. No recommendation was accepted."
    result["latencyMs"] = round((time.perf_counter() - started) * 1000, 1)
    return result
