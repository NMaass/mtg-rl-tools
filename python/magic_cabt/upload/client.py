"""HTTP client for opt-in upload envelopes."""

import json
import urllib.error
import urllib.request

__all__ = ["post_envelope"]


def post_envelope(endpoint, envelope, api_key=None, timeout=30):
    """POST a JSON envelope to an ingest endpoint and return its JSON response."""
    body = json.dumps(envelope, sort_keys=True).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "user-agent": "magic-cabt-uploader/0.1",
    }
    if api_key:
        headers["authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {"ok": True}
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8")
        try:
            details = json.loads(payload)
        except ValueError:
            details = {"ok": False, "error": "HTTP_%d" % error.code, "message": payload}
        details.setdefault("ok", False)
        raise RuntimeError("upload failed: %s" % details)
